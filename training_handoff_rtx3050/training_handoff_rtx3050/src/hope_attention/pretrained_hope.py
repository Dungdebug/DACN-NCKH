"""
Pretrained Hope-Attention Language Model (PretrainedHopeLM) for Track 2 Reproduction.
Adapts a pretrained decoder backbone (SmolLM2-135M) with Continuum Memory System (CMS).
Paper reference: arXiv:2512.24695v1 (Section 7.1, 7.3, 8.3 & Section 9.1, Figure 7).
"""

import os
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cms.continuum_memory import ContinuumMemorySystem


class PretrainedHopeLM(nn.Module):
    """
    Hope-Attention wrapper around a frozen pretrained decoder backbone.
    When num_levels == 1:
        Operates as the pure In-Context Learning (ICL) baseline (no CMS adaptation).
    When num_levels > 1:
        Attaches multi-level Continuum Memory System (CMS) with online gradient updates (Eq 71).
    """

    def __init__(
        self,
        model_name_or_path: str = "HuggingFaceTB/SmolLM2-135M",
        num_levels: int = 1,
        lowest_chunk_size: int = 64,
        chunk_sizes: Optional[List[int]] = None,
        learning_rates: Optional[List[float]] = None,
        base_lr: float = 1e-3,
        chain_type: str = "sequential",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        enable_cms: Optional[bool] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_levels = num_levels
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

        # Determine whether to attach CMS (supports B4 single-level adapter when enable_cms=True)
        should_attach_cms = enable_cms if enable_cms is not None else (self.num_levels > 1)

        # 1. Load pretrained tokenizer and causal LM backbone
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        ).to(self.device)

        # Freeze 100% of backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.config = self.backbone.config
        self.d_model = self.config.hidden_size
        self.d_ff = self.config.intermediate_size
        self.vocab_size = self.config.vocab_size

        # 2. Attach Continuum Memory System (CMS) if enabled
        if should_attach_cms:
            self.cms_norm = nn.LayerNorm(self.d_model, elementwise_affine=True).to(
                self.device, dtype=torch_dtype
            )
            self.cms = ContinuumMemorySystem(
                d_model=self.d_model,
                d_ff=self.d_ff,
                num_levels=num_levels,
                lowest_chunk_size=lowest_chunk_size,
                chunk_sizes=chunk_sizes,
                learning_rates=learning_rates,
                base_lr=base_lr,
                chain_type=chain_type,
            ).to(self.device, dtype=torch_dtype)

            # Save initial CMS state as theta_0 for memory resets (Section 7.3)
            self.cms.save_initial_state()
        else:
            self.cms_norm = None
            self.cms = None

    def get_cms_parameters(self) -> List[nn.Parameter]:
        """Returns trainable parameters belonging to the CMS memory module."""
        if self.cms is None:
            return []
        params = list(self.cms.parameters())
        if self.cms_norm is not None:
            params.extend(list(self.cms_norm.parameters()))
        return params

    def reset_memory(self) -> None:
        """Resets CMS parameters to their initial ad-hoc pre-trained state theta_0."""
        if self.cms is not None:
            self.cms.reset_memory()

    def update_cms_online(self, loss: torch.Tensor, num_tokens: int) -> List[int]:
        """
        Executes online CMS gradient update (Equation 71) for scheduled timescales.
        Backbone remains strictly frozen; gradients flow only through CMS.
        """
        if self.cms is None:
            return []
        return self.cms.update_scheduled_levels(loss, num_tokens)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through backbone and optional CMS memory augmentation.
        Returns:
            logits: (B, T, vocab_size)
            loss: Optional scalar cross-entropy loss if targets provided
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Pass through frozen backbone extracting hidden states before lm_head
        # SmolLM2 / Llama architecture: model.model gives transformer outputs
        transformer_outputs = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = transformer_outputs.last_hidden_state  # (B, T, d_model)

        # Apply Continuum Memory System (Equation 70/74) if active
        if self.cms is not None:
            normed_h = self.cms_norm(hidden_states)
            mem_residual = self.cms(normed_h)
            hidden_states = hidden_states + mem_residual

        # Compute output logits via backbone lm_head
        logits = self.backbone.lm_head(hidden_states)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            targets = targets.to(self.device)
            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def ingest_document_chunks(
        self,
        chunk_token_ids: List[torch.Tensor],
        learning_rate: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Ingests a document as sequential chunks, updating CMS memory via Equation 71.
        """
        if self.cms is None:
            return {"updated": False, "num_chunks": len(chunk_token_ids)}

        self.train()
        # Ensure only CMS parameters require gradients
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.get_cms_parameters():
            p.requires_grad = True

        total_loss = 0.0
        updates_per_chunk = []

        for chunk_idx, chunk in enumerate(chunk_token_ids):
            chunk = chunk.to(self.device)
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0)

            num_tokens = chunk.size(1)

            logits, loss = self.forward(chunk, targets=chunk)
            if loss is not None and not torch.isnan(loss):
                # Trigger online CMS update (Equation 71) for scheduled levels
                updated_levels = self.update_cms_online(loss, num_tokens)
                total_loss += loss.item()
                updates_per_chunk.append((chunk_idx, updated_levels, loss.item()))

        self.eval()
        avg_loss = total_loss / max(1, len(chunk_token_ids))
        return {
            "updated": True,
            "num_chunks": len(chunk_token_ids),
            "avg_ingest_loss": avg_loss,
            "updates_per_chunk": updates_per_chunk,
        }

    def save_cms_checkpoint(self, filepath: str) -> None:
        """Saves only the lightweight CMS memory parameters and configuration."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        cms_state = {
            "num_levels": self.num_levels,
            "model_name_or_path": self.model_name_or_path,
            "cms_state_dict": self.cms.state_dict() if self.cms is not None else None,
            "cms_norm_state_dict": self.cms_norm.state_dict() if self.cms_norm is not None else None,
        }
        torch.save(cms_state, filepath)

    def load_cms_checkpoint(self, filepath: str) -> None:
        """Loads CMS memory parameters from checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        if self.cms is not None and checkpoint.get("cms_state_dict") is not None:
            self.cms.load_state_dict(checkpoint["cms_state_dict"])
        if self.cms_norm is not None and checkpoint.get("cms_norm_state_dict") is not None:
            self.cms_norm.load_state_dict(checkpoint["cms_norm_state_dict"])

"""
Full Hope-Attention Causal Language Model (HopeAttentionLM).
"""

from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.backbone.embedding import TransformerEmbedding
from src.hope_attention.block import HopeAttentionBlock


class HopeAttentionLM(nn.Module):
    """
    Hope-Attention Causal Language Model for Long Context and Document QA.
    Paper: arXiv:2512.24695v1 (Hope-Attention variant, Section 8.3 & Section 9.1).
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        num_levels: int = 3,
        lowest_chunk_size: int = 64,
        chunk_sizes: Optional[List[int]] = None,
        learning_rates: Optional[List[float]] = None,
        base_lr: float = 1e-4,
        dropout: float = 0.0,
        chain_type: str = "sequential",
        max_seq_len: int = 1024,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.num_levels = num_levels
        self.max_seq_len = max_seq_len

        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.blocks = nn.ModuleList([
            HopeAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                num_levels=num_levels,
                lowest_chunk_size=lowest_chunk_size,
                chunk_sizes=chunk_sizes,
                learning_rates=learning_rates,
                base_lr=base_lr,
                attn_dropout=dropout,
                proj_dropout=dropout,
                cms_dropout=dropout,
                chain_type=chain_type,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.embedding.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

        # After weight initialization, cache initial CMS state as theta_0 in each block
        for block in self.blocks:
            block.cms.save_initial_state()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            targets: Optional ground-truth token ids for loss calculation (batch_size, seq_len)
            start_pos: Offset for cached decoding
            kv_caches: List of (cached_k, cached_v) tuples per layer
            use_cache: Whether to return updated KV caches

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: Optional scalar Cross-Entropy loss if targets provided
            new_kv_caches: Updated KV caches if use_cache=True
        """
        B, T = input_ids.size()
        x = self.embedding(input_ids, start_pos=start_pos)

        new_kv_caches = [] if use_cache else None

        for layer_idx, block in enumerate(self.blocks):
            layer_cache = kv_caches[layer_idx] if kv_caches is not None else None
            x, updated_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_kv_caches.append(updated_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_kv_caches

    def reset_memory(self) -> None:
        """Reset all CMS memory blocks across all layers to theta_0."""
        for block in self.blocks:
            block.reset_memory()

    def get_cms_parameters(self) -> List[nn.Parameter]:
        """Collect all trainable CMS memory parameters across all layers."""
        params = []
        for block in self.blocks:
            params.extend(list(block.cms.parameters()))
        return params

    def update_cms_online(self, loss: torch.Tensor, num_tokens: int) -> Dict[int, List[int]]:
        """
        Triggers online CMS updates (Equation 71) across all layers for scheduled levels.
        Returns:
            layer_updates: dict mapping layer_idx -> list of updated level indices.
        """
        layer_updates = {}
        for layer_idx, block in enumerate(self.blocks):
            updated_levels = block.cms.update_scheduled_levels(loss, num_tokens)
            if updated_levels:
                layer_updates[layer_idx] = updated_levels
        return layer_updates

    def get_all_memory_states(self) -> List[Dict[str, torch.Tensor]]:
        """Export CMS memory parameter states from all layers."""
        return [block.get_memory_state() for block in self.blocks]

    def set_all_memory_states(self, states: List[Dict[str, torch.Tensor]]) -> None:
        """Restore CMS memory parameter states across all layers."""
        assert len(states) == len(self.blocks)
        for block, s in zip(self.blocks, states):
            block.set_memory_state(s)

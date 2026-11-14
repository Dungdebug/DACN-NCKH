"""
Self-Study Trainer for Structure-Aligned Continuum Memory System (SA-CMS).
Proposal Reference: docs/de_cuong_text.txt Section 6.4 & 6.5.
"Ở cuối mỗi mục và mỗi tài liệu, mô hình tự sinh các cặp hỏi đáp từ phần vừa đọc và dùng chúng để cập nhật mức 2 và mức 3."

Enforces strict parameter isolation:
- 100% of SmolLM2-135M backbone is FROZEN (requires_grad = False).
- Level 1 memory (Paragraph timescale, blocks[0]) is FROZEN (requires_grad = False).
- Level 2 memory (Section timescale, blocks[1]) is TRAINABLE (requires_grad = True).
- Level 3 memory (Document timescale, blocks[2]) is TRAINABLE (requires_grad = True).

Loss formulation:
- Causal LM teacher-forcing cross-entropy loss computed strictly over answer token positions.
- Question prompt prefix tokens are masked with -100 so they never distort the learning signal.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.self_study.types import SelfStudyQAPair, SelfStudyReport
from src.hope_attention.sa_cms import StructureAlignedHopeLM

logger = logging.getLogger(__name__)


class SelfStudyTrainer:
    """
    Self-study adaptation engine.
    Applies gradient updates strictly to SA-CMS Level 2 (Section) and Level 3 (Document)
    using auto-extracted question-answer pairs, preserving Level 1 and backbone parameters.
    """

    def __init__(
        self,
        model: Optional[StructureAlignedHopeLM] = None,
        tokenizer: Optional[Any] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if model is not None:
            self.model = model
            self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        else:
            from src.web.local_model_loader import LocalModelLoader
            loader = LocalModelLoader.get_instance()
            loaded_model, loaded_tok = loader.load_offline_model(prefer_gpu=torch.cuda.is_available())
            if loaded_model is None:
                raise RuntimeError("Failed to load local offline model for SelfStudyTrainer.")
            self.model = loaded_model
            self.tokenizer = loaded_tok

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided or attached to the model.")

        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = ROOT_DIR / "local_runtime" / "checkpoints" / "self_study"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.configure_trainable_parameters()

    def configure_trainable_parameters(self) -> Dict[str, Any]:
        """
        Freezes 100% of backbone parameters and Level 1 memory.
        Enables gradients strictly for Level 2 (Section) and Level 3 (Document) CMS blocks.
        """
        # 1. Freeze backbone
        if hasattr(self.model, "backbone") and self.model.backbone is not None:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        # 2. Freeze cms_norm if present
        if getattr(self.model, "cms_norm", None) is not None:
            for p in self.model.cms_norm.parameters():
                p.requires_grad = False

        # 3. Handle CMS blocks
        has_cms = hasattr(self.model, "cms") and self.model.cms is not None
        num_blocks = 0
        trainable_levels = []

        if has_cms and hasattr(self.model.cms, "chain") and hasattr(self.model.cms.chain, "blocks"):
            blocks = self.model.cms.chain.blocks
            num_blocks = len(blocks)
            for idx, block in enumerate(blocks):
                if idx == 0:
                    # Level 1: Paragraph timescale - FROZEN
                    for p in block.parameters():
                        p.requires_grad = False
                else:
                    # Level 2 & Level 3: Section & Document timescales - TRAINABLE
                    for p in block.parameters():
                        p.requires_grad = True
                    trainable_levels.append(idx + 1)
        elif has_cms:
            for p in self.model.cms.parameters():
                p.requires_grad = True

        trainable_count = sum(p.numel() for p in self.get_trainable_parameters())
        total_count = sum(p.numel() for p in self.model.parameters())

        status = {
            "backbone_frozen": True,
            "level_1_frozen": True,
            "trainable_levels": trainable_levels,
            "num_blocks": num_blocks,
            "trainable_params": trainable_count,
            "total_params": total_count,
            "trainable_ratio": round(trainable_count / max(1, total_count), 4),
        }
        logger.info(f"SelfStudyTrainer parameter configuration: {status}")
        return status

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Returns all parameters currently flagged as requires_grad=True."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def compute_qa_loss(self, pair: SelfStudyQAPair) -> Tuple[torch.Tensor, int]:
        """
        Computes teacher-forced cross-entropy loss ONLY on answer tokens.
        Prompt template:
            Câu hỏi: {question}\nTrả lời: {answer}
        Prefix tokens (question + prompt wrapper) are masked with -100.
        """
        prompt = f"Câu hỏi: {pair.question}\nTrả lời: {pair.answer}"
        prefix = f"Câu hỏi: {pair.question}\nTrả lời: "

        enc_prompt = self.tokenizer(prompt, return_tensors="pt")
        enc_prefix = self.tokenizer(prefix, return_tensors="pt")

        input_ids = enc_prompt.input_ids.to(self.device)
        prefix_len = enc_prefix.input_ids.size(1)

        targets = input_ids.clone()
        # Mask all question and prefix tokens with -100 so no gradients flow through them
        if prefix_len < targets.size(1):
            targets[:, :prefix_len] = -100
        else:
            # Degenerate case: prefix exceeds length, no answer tokens
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        answer_tokens_count = (targets != -100).sum().item()
        if answer_tokens_count <= 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        # Forward pass
        logits, loss = self.model.forward(input_ids, targets=targets)
        return loss, answer_tokens_count

    def train(
        self,
        qa_pairs: List[SelfStudyQAPair],
        epochs: int = 2,
        lr_level2: float = 5e-5,
        lr_level3: float = 1e-4,
        max_steps: int = 20,
        grad_clip: float = 1.0,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs self-study parameter adaptation using extracted QA pairs.

        Args:
            qa_pairs: List of verified SelfStudyQAPair instances.
            epochs: Number of training passes over the pairs.
            lr_level2: Learning rate for Level 2 (Section) memory.
            lr_level3: Learning rate for Level 3 (Document) memory.
            max_steps: Hard cap on total gradient steps to ensure < 20-30s execution.
            grad_clip: Maximum gradient norm for numerical stability.
            document_id: Optional ID to auto-save the adapted checkpoint.

        Returns:
            Dict containing initial/final loss, steps, and duration.
        """
        if not qa_pairs:
            return {
                "status": "empty_pairs",
                "document_id": document_id,
                "num_pairs": 0,
                "steps_executed": 0,
                "initial_loss": 0.0,
                "final_loss": 0.0,
                "loss_history": [],
                "duration_seconds": 0.0,
            }

        start_time = time.time()
        self.configure_trainable_parameters()
        self.model.train()

        # Build parameter groups for multi-timescale learning rates
        param_groups = []
        if hasattr(self.model, "cms") and self.model.cms is not None:
            blocks = getattr(self.model.cms.chain, "blocks", [])
            if len(blocks) > 1:
                p2 = [p for p in blocks[1].parameters() if p.requires_grad]
                if p2:
                    param_groups.append({"params": p2, "lr": lr_level2})
            if len(blocks) > 2:
                p3 = [p for p in blocks[2].parameters() if p.requires_grad]
                if p3:
                    param_groups.append({"params": p3, "lr": lr_level3})

        if not param_groups:
            trainable_params = self.get_trainable_parameters()
            if not trainable_params:
                raise RuntimeError("No trainable parameters found for SelfStudyTrainer.")
            param_groups.append({"params": trainable_params, "lr": lr_level2})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        loss_history: List[float] = []
        step_count = 0
        total_tokens_trained = 0

        for epoch in range(epochs):
            for pair in qa_pairs:
                if step_count >= max_steps:
                    break

                optimizer.zero_grad()
                loss, num_ans_toks = self.compute_qa_loss(pair)

                if loss is None or torch.isnan(loss) or num_ans_toks == 0:
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.get_trainable_parameters(), max_norm=grad_clip)
                optimizer.step()

                loss_val = float(loss.item())
                loss_history.append(loss_val)
                total_tokens_trained += num_ans_toks
                step_count += 1

            if step_count >= max_steps:
                break

        self.model.eval()
        duration = time.time() - start_time

        initial_loss = loss_history[0] if loss_history else 0.0
        final_loss = loss_history[-1] if loss_history else 0.0
        reduction_pct = 0.0
        if initial_loss > 0:
            reduction_pct = round(((initial_loss - final_loss) / initial_loss) * 100.0, 2)

        saved_path = None
        if document_id:
            saved_path = self.save_document_checkpoint(
                document_id=document_id,
                metadata={
                    "num_pairs": len(qa_pairs),
                    "steps": step_count,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "reduction_pct": reduction_pct,
                    "duration_seconds": round(duration, 3),
                },
            )

        report = {
            "status": "success",
            "document_id": document_id,
            "num_pairs": len(qa_pairs),
            "steps_executed": step_count,
            "total_tokens_trained": total_tokens_trained,
            "initial_loss": round(initial_loss, 4),
            "final_loss": round(final_loss, 4),
            "loss_reduction_pct": reduction_pct,
            "loss_history": [round(l, 4) for l in loss_history],
            "saved_checkpoint": str(saved_path) if saved_path else None,
            "duration_seconds": round(duration, 3),
        }
        logger.info(f"Self-study adaptation complete: {report}")
        return report

    def save_document_checkpoint(
        self,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Saves document-adapted Level 2 & Level 3 memory parameters to disk.
        """
        target_path = self.checkpoint_dir / f"{document_id}.pt"
        payload = {
            "document_id": document_id,
            "cms_state_dict": self.model.cms.state_dict() if hasattr(self.model, "cms") and self.model.cms else {},
            "cms_norm_state_dict": (
                self.model.cms_norm.state_dict()
                if getattr(self.model, "cms_norm", None) is not None
                else None
            ),
            "metadata": metadata or {},
        }
        torch.save(payload, target_path)
        logger.info(f"Saved document self-study checkpoint to: {target_path}")
        return target_path

    def load_document_checkpoint(self, document_id_or_path: Union[str, Path]) -> bool:
        """
        Loads document-adapted memory checkpoint from disk into the current model.
        """
        if isinstance(document_id_or_path, str) and not document_id_or_path.endswith(".pt"):
            ckpt_path = self.checkpoint_dir / f"{document_id_or_path}.pt"
        else:
            ckpt_path = Path(document_id_or_path)

        if not ckpt_path.exists():
            logger.warning(f"Checkpoint file not found: {ckpt_path}")
            return False

        state = torch.load(ckpt_path, map_location=self.device)
        if "cms_state_dict" in state and hasattr(self.model, "cms") and self.model.cms is not None:
            self.model.cms.load_state_dict(state["cms_state_dict"], strict=False)
        if (
            "cms_norm_state_dict" in state
            and state["cms_norm_state_dict"] is not None
            and getattr(self.model, "cms_norm", None) is not None
        ):
            self.model.cms_norm.load_state_dict(state["cms_norm_state_dict"], strict=False)

        logger.info(f"Loaded document self-study checkpoint from: {ckpt_path}")
        return True

    def reset_memory(self) -> None:
        """Resets the CMS memory blocks back to their initial unadapted state."""
        if hasattr(self.model, "reset_memory"):
            self.model.reset_memory()
        elif hasattr(self.model, "cms") and hasattr(self.model.cms, "reset_memory"):
            self.model.cms.reset_memory()


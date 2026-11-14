"""
Structure-Aligned Continuum Memory System (SA-CMS).
Paper / Proposal: Replaces fixed token-based update schedules with document-structure-aligned schedules.
Level 1: Paragraph boundaries
Level 2: Section boundaries
Level 3: Document boundaries

Preserves Equation 71 update rule exactly.
Includes comprehensive logging of every update event.
"""

from typing import List, Dict, Any, Optional, Tuple
import copy
import logging
import torch
import torch.nn as nn

from src.document_structure.types import BoundaryType, DocumentStructure
from src.document_structure.parser import DocumentStructureParser
from src.cms.schedule import StructureAlignedSchedule
from src.cms.continuum_memory import ContinuumMemorySystem
from src.hope_attention.pretrained_hope import PretrainedHopeLM


class StructureAlignedHopeLM(PretrainedHopeLM):
    """
    Hope-Attention LM augmented with Structure-Aligned Continuum Memory System (SA-CMS).
    Preserves Equation 71 gradient updates while supporting document structure boundaries.
    """

    def __init__(
        self,
        model_name_or_path: str = "HuggingFaceTB/SmolLM2-135M",
        num_levels: int = 3,
        lowest_chunk_size: int = 64,
        chunk_sizes: Optional[List[int]] = None,
        learning_rates: Optional[List[float]] = None,
        base_lr: float = 1e-3,
        chain_type: str = "sequential",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        enable_cms: Optional[bool] = None,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            num_levels=num_levels,
            lowest_chunk_size=lowest_chunk_size,
            chunk_sizes=chunk_sizes,
            learning_rates=learning_rates,
            base_lr=base_lr,
            chain_type=chain_type,
            device=device,
            torch_dtype=torch_dtype,
            enable_cms=enable_cms,
        )
        self.parser = DocumentStructureParser()
        self.update_event_log: List[Dict[str, Any]] = []

    def get_event_log(self) -> List[Dict[str, Any]]:
        return self.update_event_log

    def clear_event_log(self) -> None:
        self.update_event_log.clear()

    def ingest_structured_document(
        self,
        document_text: str,
        schedule_mode: str = "structure",
        fixed_chunk_sizes: Optional[List[int]] = None,
        seed: int = 42,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """
        Ingests a document using SA-CMS or baseline fixed schedule, logging every update event.

        Args:
            document_text: Raw text of document to ingest.
            schedule_mode: "structure" (SA-CMS), "fixed_token" (Baseline), or "random" (Ablation).
            fixed_chunk_sizes: Token intervals for fixed_token schedule.
            seed: Random seed for random schedule ablation.
            logger: Optional logger for step-by-step trace.
        """
        if self.cms is None:
            return {"updated": False, "num_events": 0, "schedule_mode": schedule_mode}

        # 1. Parse document structure
        doc_struct = self.parser.parse_text(document_text, tokenizer=self.tokenizer)
        enc = self.tokenizer(document_text, return_tensors="pt").to(self.device)
        input_ids = enc.input_ids
        total_tokens = input_ids.size(1)

        # 2. Build multi-timescale schedule
        schedule = StructureAlignedSchedule(
            num_levels=self.num_levels,
            total_tokens=total_tokens,
            schedule_mode=schedule_mode,
            fixed_chunk_sizes=fixed_chunk_sizes,
            doc_structure=doc_struct,
            seed=seed,
        )

        # Combine all boundary positions across all levels
        all_cut_points = set([0])
        for l_idx in range(self.num_levels):
            for b in schedule.boundaries.get(l_idx, []):
                all_cut_points.add(b)
        all_cut_points.add(total_tokens)
        cut_points = sorted(list(all_cut_points))

        self.train()
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.get_cms_parameters():
            p.requires_grad = True

        total_loss = 0.0
        total_updates = 0
        span_records = []

        # 3. Step through spans between consecutive cut points
        for i in range(len(cut_points) - 1):
            start_tok = cut_points[i]
            end_tok = cut_points[i + 1]
            span_len = end_tok - start_tok
            if span_len <= 0:
                continue

            span_ids = input_ids[:, start_tok:end_tok]

            # Forward pass over current span
            logits, loss = self.forward(span_ids, targets=span_ids)
            if loss is None or torch.isnan(loss):
                continue

            total_loss += loss.item()

            # Check which levels trigger an update at end_tok
            for l_idx in range(self.num_levels):
                if schedule.should_update(l_idx, end_tok):
                    block = self.cms.chain.blocks[l_idx]
                    params = [p for p in block.parameters() if p.requires_grad]

                    grad_norm = 0.0
                    update_status = "skipped"

                    if params:
                        grads = torch.autograd.grad(
                            loss,
                            params,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        grad_dict = {}
                        for (name, p), g in zip(block.named_parameters(), grads):
                            if g is not None:
                                grad_dict[name] = g
                                grad_norm += torch.norm(g).item()

                        # Apply Equation 71 gradient update
                        self.cms.apply_gradient_update(l_idx, grad_dict)
                        update_status = "updated"
                        total_updates += 1

                    # Structural boundary label
                    boundary_label = f"level_{l_idx + 1}"
                    if schedule_mode == "structure":
                        labels = ["paragraph", "section", "document"]
                        boundary_label = labels[min(l_idx, len(labels) - 1)]
                    elif schedule_mode == "fixed_token":
                        boundary_label = f"fixed_chunk_{fixed_chunk_sizes[l_idx] if fixed_chunk_sizes else 64}"
                    elif schedule_mode == "random":
                        boundary_label = f"random_boundary_{l_idx + 1}"

                    event = {
                        "memory_level": l_idx + 1,
                        "start_token": start_tok,
                        "end_token": end_tok,
                        "structural_boundary": boundary_label,
                        "num_accumulated_tokens": span_len,
                        "gradient_status": update_status,
                        "grad_norm": round(grad_norm, 6),
                        "loss": round(loss.item(), 4),
                    }
                    self.update_event_log.append(event)
                    span_records.append(event)

                    if logger:
                        logger.info(
                            f"[Update Event] Level {event['memory_level']} ({event['structural_boundary']}) | "
                            f"Tokens: [{start_tok}:{end_tok}] (len={span_len}) | "
                            f"Grad Norm: {event['grad_norm']} | Status: {event['gradient_status']}"
                        )

        self.eval()
        return {
            "updated": True,
            "schedule_mode": schedule_mode,
            "total_tokens": total_tokens,
            "num_update_events": total_updates,
            "avg_loss": total_loss / max(1, len(cut_points) - 1),
            "events": span_records,
        }

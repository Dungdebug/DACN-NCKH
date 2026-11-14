"""
Frequency and chunk-size schedule management for Continuum Memory System (CMS) & SA-CMS.
Supports both baseline fixed token schedules and structure-aligned multi-timescale schedules.
"""

import random
from typing import List, Dict, Any, Optional

from src.document_structure.types import DocumentStructure


class CMSSchedule:
    """
    Manages the multi-timescale hierarchy and chunk sizes for Continuum Memory System.

    Paper Section 7.1 & Definition 2:
    f_l = max_i C^(i) / C^(l)
    Level 1: Lowest frequency (largest chunk size C^(1)), representing the most persistent memory.
    Level k: Highest frequency (smallest chunk size C^(k)), representing the most adaptive memory.
    """

    def __init__(
        self,
        num_levels: int = 3,
        lowest_chunk_size: int = 64,
        chunk_sizes: List[int] = None,
        learning_rates: List[float] = None,
        base_lr: float = 1e-4,
    ):
        """
        Args:
            num_levels: Number of memory levels k (e.g. 1, 2, 3, 4).
            lowest_chunk_size: Chunk size of level 1 (slowest memory).
            chunk_sizes: Optional explicit list of chunk sizes [C^(1), C^(2), ..., C^(k)].
                         If None, generated using power-of-2 divisions:
                         C^(l) = max(1, lowest_chunk_size // (2 ** (l - 1))).
            learning_rates: Optional list of learning rates [eta^(1), ..., eta^(k)].
                            If None, all levels default to base_lr.
            base_lr: Default base learning rate for CMS levels.
        """
        self.num_levels = max(1, num_levels)

        if chunk_sizes is not None:
            assert len(chunk_sizes) == self.num_levels, (
                f"chunk_sizes length ({len(chunk_sizes)}) must match num_levels ({self.num_levels})"
            )
            self.chunk_sizes = list(chunk_sizes)
        else:
            # Power-of-2 hierarchy: Level 1 is largest, Level k is smallest
            self.chunk_sizes = []
            for l in range(1, self.num_levels + 1):
                c_l = max(1, lowest_chunk_size // (2 ** (l - 1)))
                self.chunk_sizes.append(c_l)

        if learning_rates is not None:
            assert len(learning_rates) == self.num_levels, (
                f"learning_rates length ({len(learning_rates)}) must match num_levels ({self.num_levels})"
            )
            self.learning_rates = list(learning_rates)
        else:
            self.learning_rates = [base_lr] * self.num_levels

        max_c = max(self.chunk_sizes)
        self.frequencies = [max_c / c for c in self.chunk_sizes]

    def get_chunk_size(self, level: int) -> int:
        """Get chunk size C^(l) for level index (0-indexed)."""
        return self.chunk_sizes[level]

    def get_lr(self, level: int) -> float:
        """Get learning rate eta^(l) for level index (0-indexed)."""
        return self.learning_rates[level]

    def should_update(self, level: int, current_token_pos: int) -> bool:
        """Fixed chunk schedule update trigger check."""
        c = self.get_chunk_size(level)
        return (current_token_pos > 0) and (current_token_pos % c == 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_levels": self.num_levels,
            "chunk_sizes": self.chunk_sizes,
            "learning_rates": self.learning_rates,
            "frequencies": self.frequencies,
        }


class StructureAlignedSchedule:
    """
    Manages multi-timescale update triggers based on:
    - "structure": Real document structural boundaries (Paragraph, Section, Document)
    - "fixed_token": Fixed token chunk intervals (Baseline CMS)
    - "random": Random token boundaries matching event counts (Ablation A3)
    """

    def __init__(
        self,
        num_levels: int = 3,
        total_tokens: int = 0,
        schedule_mode: str = "structure",
        fixed_chunk_sizes: Optional[List[int]] = None,
        doc_structure: Optional[DocumentStructure] = None,
        learning_rates: Optional[List[float]] = None,
        base_lr: float = 1e-4,
        seed: int = 42,
    ):
        self.num_levels = num_levels
        self.total_tokens = total_tokens
        self.schedule_mode = schedule_mode
        self.seed = seed
        self.boundaries: Dict[int, List[int]] = {}

        if learning_rates is not None:
            assert len(learning_rates) == self.num_levels, (
                f"learning_rates length ({len(learning_rates)}) must match num_levels ({self.num_levels})"
            )
            self.learning_rates = list(learning_rates)
        else:
            self.learning_rates = [base_lr] * self.num_levels

        if schedule_mode == "structure":
            # SA-CMS: Paragraph, Section, Document
            if doc_structure is None:
                raise ValueError("doc_structure must be provided when schedule_mode='structure'")
            for l_idx in range(num_levels):
                level_num = l_idx + 1
                self.boundaries[l_idx] = doc_structure.get_update_boundaries(level_num)

        elif schedule_mode in ("fixed_token", "random"):
            if doc_structure is not None and (schedule_mode == "random" or fixed_chunk_sizes is None):
                target_counts = [len(doc_structure.get_update_boundaries(l + 1)) for l in range(num_levels)]
                k0 = max(1, target_counts[0])
                min_span = 2

                # Base level
                if schedule_mode == "fixed_token":
                    step = total_tokens / float(k0)
                    base = [int(round((i + 1) * step)) for i in range(k0 - 1)]
                    for i in range(len(base)):
                        m = (i + 1) * min_span
                        if base[i] < m:
                            base[i] = m
                        if i > 0 and base[i] <= base[i - 1]:
                            base[i] = base[i - 1] + min_span
                    for i in range(len(base) - 1, -1, -1):
                        m = total_tokens - (len(base) - i) * min_span
                        if base[i] > m:
                            base[i] = m
                        if i < len(base) - 1 and base[i] >= base[i + 1]:
                            base[i] = base[i + 1] - min_span
                    base.append(total_tokens)
                else:  # random
                    reduced = total_tokens - (k0 - 1) * (min_span - 1)
                    rng = random.Random(seed)
                    if reduced > k0:
                        raw = sorted(rng.sample(range(min_span, reduced), k0 - 1))
                        base = [s + i * (min_span - 1) for i, s in enumerate(raw)]
                    else:
                        step = total_tokens / float(k0)
                        base = [int(round((i + 1) * step)) for i in range(k0 - 1)]
                    base.append(total_tokens)

                self.boundaries[0] = sorted(list(set(base)))

                # Higher levels
                for l_idx in range(1, num_levels):
                    target_k = min(target_counts[l_idx], len(self.boundaries[0]))
                    if target_k <= 1:
                        self.boundaries[l_idx] = [total_tokens] if total_tokens > 0 else []
                    elif schedule_mode == "fixed_token":
                        step = len(self.boundaries[0]) / float(target_k)
                        indices = [min(len(self.boundaries[0]) - 1, int(round((i + 1) * step)) - 1) for i in range(target_k - 1)]
                        b_list = sorted(list(set([self.boundaries[0][idx] for idx in indices] + [total_tokens])))
                        self.boundaries[l_idx] = b_list
                    else:  # random
                        rng = random.Random(seed + l_idx * 100)
                        eligible = self.boundaries[0][:-1]
                        k_sub = min(len(eligible), target_k - 1)
                        sampled = sorted(rng.sample(eligible, k_sub))
                        sampled.append(total_tokens)
                        self.boundaries[l_idx] = sampled
            else:
                # Classic fixed chunk size
                chunk_sizes = fixed_chunk_sizes or [64, 32, 16][:num_levels]
                for l_idx in range(num_levels):
                    c_size = chunk_sizes[min(l_idx, len(chunk_sizes) - 1)]
                    b_list = list(range(c_size, total_tokens, c_size))
                    if not b_list or b_list[-1] != total_tokens:
                        b_list.append(total_tokens)
                    self.boundaries[l_idx] = b_list
        else:
            raise ValueError(f"Unknown schedule_mode: {schedule_mode}")

        # Chunk sizes approximation for buffer compatibility
        self.chunk_sizes = [
            max(1, total_tokens // max(1, len(self.boundaries.get(l, [1]))))
            for l in range(num_levels)
        ]

    def get_lr(self, level: int) -> float:
        """Get learning rate for level index."""
        return self.learning_rates[level]

    def should_update(self, level: int, current_token_pos: int) -> bool:
        """Returns True if current_token_pos matches an update boundary for level."""
        return current_token_pos in self.boundaries.get(level, [])

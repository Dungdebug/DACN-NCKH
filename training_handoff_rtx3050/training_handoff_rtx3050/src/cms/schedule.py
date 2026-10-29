"""
Frequency and chunk-size schedule management for Continuum Memory System (CMS).
"""

from typing import List, Dict, Any


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_levels": self.num_levels,
            "chunk_sizes": self.chunk_sizes,
            "learning_rates": self.learning_rates,
            "frequencies": self.frequencies,
        }

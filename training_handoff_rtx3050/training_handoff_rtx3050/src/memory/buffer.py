"""
Token buffer and gradient accumulator for multi-timescale chunked memory updates.
"""

from typing import List, Dict, Optional
import torch


class MemoryChunkBuffer:
    """
    Tracks token counts, buffered activations/losses, and manages chunk boundaries
    for each level in Continuum Memory System (Equation 71).
    """

    def __init__(self, chunk_sizes: List[int]):
        """
        Args:
            chunk_sizes: List of chunk sizes C^(l) for each level l=1..k.
                         Usually C^(1) >= C^(2) >= ... >= C^(k).
        """
        self.chunk_sizes = chunk_sizes
        self.num_levels = len(chunk_sizes)
        self.token_counter = 0
        # Accumulated steps since last update for each level
        self.steps_since_update = [0] * self.num_levels

    def step(self, num_tokens: int = 1) -> List[bool]:
        """
        Advance the token counter by num_tokens and return a boolean mask
        indicating which levels have reached their chunk boundary and must update.

        Returns:
            should_update: list of booleans of length num_levels.
                           True if level l should execute update at this step.
        """
        self.token_counter += num_tokens
        should_update = [False] * self.num_levels

        for l in range(self.num_levels):
            c_l = self.chunk_sizes[l]
            self.steps_since_update[l] += num_tokens
            if self.steps_since_update[l] >= c_l:
                should_update[l] = True
                self.steps_since_update[l] = 0

        return should_update

    def reset(self) -> None:
        """Reset the buffer token counters."""
        self.token_counter = 0
        self.steps_since_update = [0] * self.num_levels

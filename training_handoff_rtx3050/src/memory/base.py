"""
Abstract base class and containers for memory in Nested Learning / CMS.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseMemoryModule(nn.Module, ABC):
    """
    Abstract interface for a memory module in a multi-level Continuum Memory System.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the memory module."""
        pass

    @abstractmethod
    def reset_memory(self) -> None:
        """Reset the memory module to its initial state (theta_0)."""
        pass

    @abstractmethod
    def get_memory_state(self) -> Dict[str, torch.Tensor]:
        """Export current memory parameter states."""
        pass

    @abstractmethod
    def set_memory_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Restore memory parameter states from a dictionary."""
        pass

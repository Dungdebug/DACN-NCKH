"""
Continuum Memory System (CMS) implementation with multi-timescale gradient update (Equation 71).
"""

from typing import Dict, Any, List, Optional
import copy
import torch
import torch.nn as nn
from src.memory.base import BaseMemoryModule
from src.memory.buffer import MemoryChunkBuffer
from src.cms.schedule import CMSSchedule
from src.cms.mlp_chain import SequentialMLPChain, IndependentMLPChain, MLPBlock


class ContinuumMemorySystem(BaseMemoryModule):
    """
    Continuum Memory System (CMS) from arXiv:2512.24695v1 Section 7.

    Integrates:
    - Multi-level MLP chain (Sequential Eq 70 or Independent Eq 74)
    - Multi-timescale frequency schedule (Section 7.1)
    - Gradient accumulation and chunked updates (Equation 71)
    - Ad-hoc pre-trained initialization & memory reset (Section 7.3)
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ff: int = 1024,
        num_levels: int = 3,
        lowest_chunk_size: int = 64,
        chunk_sizes: Optional[List[int]] = None,
        learning_rates: Optional[List[float]] = None,
        base_lr: float = 1e-4,
        dropout: float = 0.0,
        chain_type: str = "sequential",  # "sequential" (Eq 70) or "independent" (Eq 74)
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_levels = num_levels
        self.chain_type = chain_type

        # Multi-timescale schedule
        self.schedule = CMSSchedule(
            num_levels=num_levels,
            lowest_chunk_size=lowest_chunk_size,
            chunk_sizes=chunk_sizes,
            learning_rates=learning_rates,
            base_lr=base_lr,
        )

        # Chunk boundary buffer
        self.buffer = MemoryChunkBuffer(self.schedule.chunk_sizes)

        # Construct MLP chain
        if chain_type == "sequential":
            self.chain = SequentialMLPChain(
                num_levels=num_levels,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
            )
        elif chain_type == "independent":
            self.chain = IndependentMLPChain(
                num_levels=num_levels,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown chain_type: {chain_type}. Expected 'sequential' or 'independent'.")

        # Cache initial state theta_0 for memory reset (Section 7.3)
        self._initial_state_dict: Dict[str, torch.Tensor] = {}
        self.save_initial_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Continuum Memory chain.
        """
        return self.chain(x)

    def save_initial_state(self) -> None:
        """
        Save the current state dict as theta_0 (base state).
        """
        self._initial_state_dict = {
            k: v.detach().clone() for k, v in self.chain.state_dict().items()
        }

    def reset_memory(self) -> None:
        """
        Reset all memory levels back to their initial state theta_0 (Equation 72 / Section 7.3).
        Also resets the token buffer.
        """
        if self._initial_state_dict:
            self.chain.load_state_dict({
                k: v.clone() for k, v in self._initial_state_dict.items()
            })
        self.buffer.reset()

    def get_memory_state(self) -> Dict[str, torch.Tensor]:
        """
        Export current dynamic state of memory blocks.
        """
        return {k: v.detach().clone() for k, v in self.chain.state_dict().items()}

    def set_memory_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Restore memory blocks from state dict.
        """
        self.chain.load_state_dict(state_dict)

    def initialize_from_base(self, base_mlp: MLPBlock) -> None:
        """
        Ad-hoc level stacking (Section 7.3): copy pre-trained base MLP weights to all levels.
        """
        self.chain.initialize_from_base(base_mlp)
        self.save_initial_state()

    def apply_gradient_update(
        self,
        level_idx: int,
        grads: Dict[str, torch.Tensor],
        lr: Optional[float] = None,
    ) -> None:
        """
        Applies a gradient descent update step (Equation 71) to level_idx:
        theta^(f_l) <- theta^(f_l) - eta^(l) * grad
        """
        eta = lr if lr is not None else self.schedule.get_lr(level_idx)
        block = self.chain.blocks[level_idx]

        with torch.no_grad():
            for name, param in block.named_parameters():
                if name in grads and grads[name] is not None:
                    param.add_(grads[name], alpha=-eta)

    def update_scheduled_levels(
        self,
        loss: torch.Tensor,
        num_tokens: int,
        retain_graph: bool = False,
    ) -> List[int]:
        """
        Advance token counter by num_tokens.
        For any level that reaches its chunk boundary C^(l), compute gradient of loss
        w.r.t that level's parameters and execute Equation 71 update.

        Args:
            loss: Scalar loss tensor on the current context/chunk.
            num_tokens: Number of tokens processed in this step.
            retain_graph: Whether to retain graph for autograd if multiple updates occur.

        Returns:
            updated_levels: List of level indices that executed an update step.
        """
        should_update = self.buffer.step(num_tokens)
        updated_levels = []

        for l_idx, must_update in enumerate(should_update):
            if must_update:
                block = self.chain.blocks[l_idx]
                params = [p for p in block.parameters() if p.requires_grad]
                if params and loss is not None:
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
                    self.apply_gradient_update(l_idx, grad_dict)
                    updated_levels.append(l_idx)

        return updated_levels

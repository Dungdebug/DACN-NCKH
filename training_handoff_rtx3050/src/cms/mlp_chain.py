"""
MLP Block and Multi-level Chains for Continuum Memory System (Equations 70 and 74).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """
    Standard Feed-Forward Network (MLP) block with residual connection.
    Used as an associative memory block in a specific frequency level.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ff: int = 1024,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection: x + Dropout(W2(act(W1(x))))
        """
        hidden = self.act(self.w1(x))
        out = self.dropout(self.w2(hidden))
        return x + out


class SequentialMLPChain(nn.Module):
    """
    Sequential Continuum Memory System (Equation 70 of arXiv:2512.24695v1):
    y_t = MLP^(f_k)( MLP^(f_{k-1})( ... MLP^(f_1)(x_t) ) )
    where Level 1 is the slowest (lowest frequency) and Level k is the fastest (highest frequency).
    """

    def __init__(
        self,
        num_levels: int = 3,
        d_model: int = 256,
        d_ff: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.d_model = d_model
        self.d_ff = d_ff

        self.blocks = nn.ModuleList([
            MLPBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
            for _ in range(num_levels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass sequentially through Level 1 -> Level 2 -> ... -> Level k.
        """
        curr = x
        for block in self.blocks:
            curr = block(curr)
        return curr

    def initialize_from_base(self, base_mlp: MLPBlock) -> None:
        """
        Ad-hoc level stacking (Section 7.3):
        Initialize all levels with the pre-trained weights of base_mlp.
        MLP^(f_i)_0 = MLP_pretrained
        """
        base_state = base_mlp.state_dict()
        for block in self.blocks:
            block.load_state_dict({k: v.clone() for k, v in base_state.items()})


class IndependentMLPChain(nn.Module):
    """
    Independent (Head-wise) Continuum Memory System (Equation 74 of arXiv:2512.24695v1):
    y_t = Agg( MLP^(f_k)(x_t), ..., MLP^(f_1)(x_t) )
    Uses learnable gated aggregation over independent memory levels.
    """

    def __init__(
        self,
        num_levels: int = 3,
        d_model: int = 256,
        d_ff: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.d_model = d_model
        self.d_ff = d_ff

        self.blocks = nn.ModuleList([
            MLPBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
            for _ in range(num_levels)
        ])
        # Learnable gate for aggregation
        self.gate = nn.Parameter(torch.ones(num_levels) / num_levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes each level in parallel and combines them using normalized gating.
        """
        weights = F.softmax(self.gate, dim=0)
        outputs = [block(x) for block in self.blocks]
        # Weighted sum aggregation
        out = sum(w * out_l for w, out_l in zip(weights, outputs))
        return out

    def initialize_from_base(self, base_mlp: MLPBlock) -> None:
        base_state = base_mlp.state_dict()
        for block in self.blocks:
            block.load_state_dict({k: v.clone() for k, v in base_state.items()})

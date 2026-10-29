"""
Hope-Attention Transformer Block combining Causal Attention with Continuum Memory System (CMS).
"""

from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from src.attention.causal_attention import CausalSelfAttention
from src.cms.continuum_memory import ContinuumMemorySystem
from src.cms.mlp_chain import MLPBlock


class HopeAttentionBlock(nn.Module):
    """
    Hope-Attention block (arXiv:2512.24695v1 Section 8.3 & Đề cương mục 4).
    Replaces standard MLP with Continuum Memory System (CMS).
    Architecture:
      x' = x + Attention(LayerNorm1(x))
      y  = x' + CMS(LayerNorm2(x'))
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        num_levels: int = 3,
        lowest_chunk_size: int = 64,
        chunk_sizes: Optional[List[int]] = None,
        learning_rates: Optional[List[float]] = None,
        base_lr: float = 1e-4,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        cms_dropout: float = 0.0,
        chain_type: str = "sequential",
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            max_seq_len=max_seq_len,
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.cms = ContinuumMemorySystem(
            d_model=d_model,
            d_ff=d_ff,
            num_levels=num_levels,
            lowest_chunk_size=lowest_chunk_size,
            chunk_sizes=chunk_sizes,
            learning_rates=learning_rates,
            base_lr=base_lr,
            dropout=cms_dropout,
            chain_type=chain_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through HopeAttentionBlock.
        """
        # Attention with residual
        norm_x = self.ln1(x)
        attn_out, new_kv_cache = self.attn(norm_x, kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out

        # CMS with residual
        norm_x = self.ln2(x)
        cms_out = self.cms(norm_x)
        x = x + cms_out

        return x, new_kv_cache

    def reset_memory(self) -> None:
        """Reset CMS memory levels to initial state."""
        self.cms.reset_memory()

    def get_memory_state(self) -> Dict[str, torch.Tensor]:
        return self.cms.get_memory_state()

    def set_memory_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.cms.set_memory_state(state_dict)

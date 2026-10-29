"""
Causal Multi-Head Self-Attention module for Hope-Attention and standard Transformer.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Standard Causal Multi-Head Self-Attention (Vaswani et al. 2017).
    Used as the working memory module in Hope-Attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # Causal mask buffer
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            kv_cache: Optional tuple of (cached_k, cached_v)
            use_cache: Whether to return updated KV cache

        Returns:
            out: Output tensor of shape (batch_size, seq_len, d_model)
            new_kv_cache: Updated KV cache if use_cache is True, else None
        """
        B, T, C = x.size()

        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)

        # Handle KV Cache for incremental decoding
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None
        total_k_len = k.size(2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nh, T, total_k_len)

        # Apply causal mask
        if T == total_k_len:
            mask = self.causal_mask[:, :, :T, :T]
            scores = scores.masked_fill(~mask, float("-inf"))
        elif T > 1:
            # Query sequence length T, key sequence length total_k_len
            q_pos = total_k_len - T
            mask = self.causal_mask[:, :, q_pos:total_k_len, :total_k_len]
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Attention output
        attn_out = torch.matmul(attn_weights, v)  # (B, nh, T, dh)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj_dropout(self.out_proj(attn_out))
        return out, new_kv_cache

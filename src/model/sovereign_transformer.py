"""
SovereignMiniTransformer: Self-Built Pure-PyTorch Local Causal LLM (Pillar 1).

Architectural Features:
- 100% pure PyTorch implementation: zero external APIs, zero third-party model weights
  or network downloads required.
- Modern Causal Decoder-Only Transformer with RMSNorm, Rotary Position Embeddings (RoPE),
  SwiGLU FFN, KV-Cache autoregressive decoding, and optional weight tying.
- Direct integration of Wavelet Low-Rank Adaptation (W = W_0 + B @ Theta @ A) into
  both Causal Self-Attention (Q, K, V, Out) and SwiGLU FFN projections, enabling:
  1. Fast associative forward path W_0 x + B (Theta (A x)) without dense d_out x d_in allocation.
  2. Direct local fine-tuning of tiny r x r memory cores Theta with frozen base weights W_0.
  3. Dynamic injection of WaveTreeMemory / EquilibriumReadout cores Theta(alpha) and
     instant zero-residual unlearning reset back to W_0.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SovereignMiniConfig:
    """Configuration for SovereignMiniTransformer."""

    vocab_size: int = 1024
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 384
    max_seq_len: int = 512
    rank_r: int = 8
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_weights: bool = True
    wavelet_in_attn: bool = True
    wavelet_in_ffn: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SovereignMiniConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (pure PyTorch)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) supporting cached incremental decoding."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 512,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        self.max_seq_len = seq_len

    def forward(
        self,
        seq_len: int,
        start_pos: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        needed = start_pos + seq_len
        if needed > self.max_seq_len:
            self._build_cache(needed)
        cos = self.cos_cached[:, :, start_pos:needed, :]
        sin = self.sin_cached[:, :, start_pos:needed, :]
        if device is not None:
            cos = cos.to(device=device)
            sin = sin.to(device=device)
        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class WaveletLowRankLinear(nn.Module):
    """
    Generalized Wavelet Low-Rank Adapted Linear Layer:
        W_eff(Theta) = W_0 + B @ Theta @ A
    where:
        W_0 in R^{out_features x in_features}  (base weight)
        A   in R^{r x in_features}             (shared input basis)
        B   in R^{out_features x r}            (shared output basis)
        Theta in R^{r x r}                     (low-rank memory core, zero-init)

    Uses the associative fast path:
        y = x W_0^T + ((x A^T) Theta^T) B^T
    avoiding dense out_features x in_features outer-product allocation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.r = int(r)

        self.w_0 = nn.Linear(self.in_features, self.out_features, bias=bias)

        # Shared projection bases A in R^{r x d_in}, B in R^{d_out x r}
        self.a_proj = nn.Parameter(
            torch.randn(self.r, self.in_features) / math.sqrt(max(1, self.in_features))
        )
        self.b_proj = nn.Parameter(
            torch.randn(self.out_features, self.r) / math.sqrt(max(1, self.r))
        )

        # Local memory core Theta in R^{r x r}, zero-initialized so W_eff == W_0 initially
        self.theta = nn.Parameter(torch.zeros(self.r, self.r))
        self._external_theta: Optional[torch.Tensor] = None

    def _resolve_theta(
        self,
        theta: Optional[Union[torch.Tensor, np.ndarray]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        raw = theta if theta is not None else self._external_theta
        if raw is None:
            return self.theta
        if isinstance(raw, np.ndarray):
            th = torch.from_numpy(raw).to(dtype=dtype, device=device)
        else:
            th = raw.to(dtype=dtype, device=device)
        if th.ndim == 1:
            th = th.reshape(self.r, self.r)
        return th

    def forward(
        self,
        x: torch.Tensor,
        theta: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> torch.Tensor:
        base_out = self.w_0(x)
        th = self._resolve_theta(theta, dtype=x.dtype, device=x.device)

        proj_a = F.linear(x, self.a_proj)  # (..., r)
        if th.ndim == 2:
            proj_th = F.linear(proj_a, th)  # (..., r)
        elif th.ndim == 3:
            # Batched per-sample Theta of shape (B, r, r)
            if proj_a.ndim == 3:
                proj_th = torch.bmm(proj_a, th.transpose(-2, -1))
            elif proj_a.ndim == 2:
                proj_th = torch.bmm(proj_a.unsqueeze(1), th.transpose(-2, -1)).squeeze(1)
            else:
                raise ValueError(f"Unsupported tensor rank {proj_a.ndim} for batched theta")
        else:
            raise ValueError(f"Expected 2D or 3D theta, got shape {tuple(th.shape)}")

        adapted_out = F.linear(proj_th, self.b_proj)
        return base_out + adapted_out

    def set_external_theta(self, theta: Optional[Union[torch.Tensor, np.ndarray]]) -> None:
        """Sets an external memory core (e.g., WaveTreeMemory Theta(alpha)) for inference."""
        if theta is None:
            self._external_theta = None
            return
        if isinstance(theta, np.ndarray):
            th = torch.from_numpy(theta).to(dtype=self.w_0.weight.dtype, device=self.w_0.weight.device)
        else:
            th = theta.detach().to(dtype=self.w_0.weight.dtype, device=self.w_0.weight.device)
        self._external_theta = th.reshape(self.r, self.r) if th.ndim == 1 else th

    def reset_theta(self) -> None:
        """Resets local Theta to zeros and clears any injected external Theta."""
        with torch.no_grad():
            self.theta.zero_()
        self._external_theta = None

    def synthesize_effective_weight(
        self,
        theta: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> torch.Tensor:
        """Materializes dense W_eff = W_0 + B @ Theta @ A for auditing and equivalence tests."""
        th = self._resolve_theta(
            theta,
            dtype=self.w_0.weight.dtype,
            device=self.w_0.weight.device,
        )
        if th.ndim != 2:
            raise ValueError("synthesize_effective_weight requires a 2D (r, r) theta matrix")
        return self.w_0.weight + torch.matmul(self.b_proj, torch.matmul(th, self.a_proj))

    def verify_associative_equivalence(
        self,
        x: torch.Tensor,
        theta: Optional[Union[torch.Tensor, np.ndarray]] = None,
        tol: float = 1e-5,
    ) -> Tuple[bool, float]:
        """Checks ||y_associative - y_dense||_inf < tol."""
        with torch.no_grad():
            y_assoc = self.forward(x, theta=theta)
            w_eff = self.synthesize_effective_weight(theta=theta)
            y_dense = F.linear(x, w_eff, self.w_0.bias)
            max_disc = float(torch.max(torch.abs(y_assoc - y_dense)).item())
        return bool(max_disc < tol), max_disc


class WaveletCausalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention with RoPE and Wavelet Low-Rank projections."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        rank_r: int = 8,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        use_wavelet: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_wavelet = use_wavelet

        def _make_proj() -> nn.Module:
            if use_wavelet:
                return WaveletLowRankLinear(d_model, d_model, r=rank_r, bias=False)
            return nn.Linear(d_model, d_model, bias=False)

        self.q_proj = _make_proj()
        self.k_proj = _make_proj()
        self.v_proj = _make_proj()
        self.out_proj = _make_proj()

        self.rope = RotaryEmbedding(self.d_head, max_seq_len=max_seq_len, theta=rope_theta)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def _proj(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        theta: Optional[Union[torch.Tensor, np.ndarray]],
    ) -> torch.Tensor:
        if isinstance(layer, WaveletLowRankLinear):
            return layer(x, theta=theta)
        return layer(x)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        theta: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.size()

        q = self._proj(self.q_proj, x, theta).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self._proj(self.k_proj, x, theta).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self._proj(self.v_proj, x, theta).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        cos, sin = self.rope(seq_len=seq_len, start_pos=start_pos, device=x.device, dtype=x.dtype)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None
        total_k_len = k.size(2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if seq_len > 1:
            q_offset = total_k_len - seq_len
            q_pos = torch.arange(q_offset, q_offset + seq_len, device=x.device).unsqueeze(1)
            k_pos = torch.arange(total_k_len, device=x.device).unsqueeze(0)
            causal_mask = k_pos <= q_pos
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.proj_dropout(self._proj(self.out_proj, attn_out, theta))
        return out, new_kv_cache


class WaveletSwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network with optional Wavelet Low-Rank adaptation."""

    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 384,
        rank_r: int = 8,
        dropout: float = 0.0,
        use_wavelet: bool = True,
    ) -> None:
        super().__init__()
        self.use_wavelet = use_wavelet

        def _make_lin(in_dim: int, out_dim: int) -> nn.Module:
            if use_wavelet:
                return WaveletLowRankLinear(in_dim, out_dim, r=rank_r, bias=False)
            return nn.Linear(in_dim, out_dim, bias=False)

        self.gate_proj = _make_lin(d_model, d_ff)
        self.up_proj = _make_lin(d_model, d_ff)
        self.down_proj = _make_lin(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def _proj(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        theta: Optional[Union[torch.Tensor, np.ndarray]],
    ) -> torch.Tensor:
        if isinstance(layer, WaveletLowRankLinear):
            return layer(x, theta=theta)
        return layer(x)

    def forward(
        self,
        x: torch.Tensor,
        theta: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> torch.Tensor:
        gated = F.silu(self._proj(self.gate_proj, x, theta)) * self._proj(self.up_proj, x, theta)
        return self.dropout(self._proj(self.down_proj, gated, theta))


class SovereignTransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer block with Wavelet Low-Rank Attention & SwiGLU."""

    def __init__(self, config: SovereignMiniConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = WaveletCausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            rank_r=config.rank_r,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
            use_wavelet=config.wavelet_in_attn,
        )
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = WaveletSwiGLU(
            d_model=config.d_model,
            d_ff=config.d_ff,
            rank_r=config.rank_r,
            dropout=config.dropout,
            use_wavelet=config.wavelet_in_ffn,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        theta: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(
            self.norm1(x),
            start_pos=start_pos,
            kv_cache=kv_cache,
            use_cache=use_cache,
            theta=theta,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x), theta=theta)
        return x, new_cache


class SovereignMiniTransformer(nn.Module):
    """
    Self-Built Pure-PyTorch Sovereign Causal Language Model with integrated
    Wavelet Low-Rank Adaptation (W_0 + B Theta A).
    """

    def __init__(self, config: Optional[SovereignMiniConfig] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = SovereignMiniConfig(**kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len
        self.rank_r = config.rank_r

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            SovereignTransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Ensure all Wavelet Theta cores stay exactly zero-initialized after apply()
        self.reset_wavelet_adapters()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, WaveletLowRankLinear):
            nn.init.normal_(module.a_proj, mean=0.0, std=1.0 / math.sqrt(max(1, module.in_features)))
            nn.init.normal_(module.b_proj, mean=0.0, std=1.0 / math.sqrt(max(1, module.r)))

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        theta: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        x = self.emb_dropout(self.tok_emb(input_ids))
        new_kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        for layer_idx, block in enumerate(self.blocks):
            layer_cache = kv_caches[layer_idx] if kv_caches is not None else None
            x, updated_cache = block(
                x,
                start_pos=start_pos,
                kv_cache=layer_cache,
                use_cache=use_cache,
                theta=theta,
            )
            if use_cache and new_kv_caches is not None and updated_cache is not None:
                new_kv_caches.append(updated_cache)

        x = self.norm_f(x)
        return x, new_kv_caches

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        theta: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Matches standard repo LM forward contract: (logits, loss, new_kv_caches).
        """
        hidden, new_kv_caches = self.forward_hidden(
            input_ids=input_ids,
            start_pos=start_pos,
            kv_caches=kv_caches,
            use_cache=use_cache,
            theta=theta,
        )
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, new_kv_caches

    def encode_hidden(
        self,
        input_ids: torch.Tensor,
        theta: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (sequence_hidden [B, T, d_model], mean_pooled [B, d_model])."""
        hidden, _ = self.forward_hidden(input_ids=input_ids, theta=theta)
        pooled = hidden.mean(dim=1)
        return hidden, pooled

    @torch.no_grad()
    def score_sequence(
        self,
        input_ids: torch.Tensor,
        theta: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> float:
        """Computes mean token log-likelihood for candidate re-ranking."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.size(1) < 2:
            return 0.0
        was_training = self.training
        self.eval()
        _, loss, _ = self.forward(
            input_ids[:, :-1],
            targets=input_ids[:, 1:],
            theta=theta,
        )
        if was_training:
            self.train()
        return -float(loss.item()) if loss is not None else 0.0

    # =========================================================================
    # WAVELET LOW-RANK ADAPTER & TREE-MEMORY INTEGRATION
    # =========================================================================

    def iter_wavelet_layers(self) -> List[WaveletLowRankLinear]:
        return [m for m in self.modules() if isinstance(m, WaveletLowRankLinear)]

    def get_wavelet_parameters(self, include_bases: bool = False) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for layer in self.iter_wavelet_layers():
            params.append(layer.theta)
            if include_bases:
                params.append(layer.a_proj)
                params.append(layer.b_proj)
        return params

    def freeze_base_weights(self, freeze_bases_ab: bool = True) -> int:
        """
        Freezes all W_0 base parameters (embeddings, norms, W_0 linear weights),
        leaving Wavelet Low-Rank adapters trainable for fast local adaptation.
        Returns count of trainable parameters.
        """
        for param in self.parameters():
            param.requires_grad = False
        for layer in self.iter_wavelet_layers():
            layer.theta.requires_grad = True
            if not freeze_bases_ab:
                layer.a_proj.requires_grad = True
                layer.b_proj.requires_grad = True
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def unfreeze_all(self) -> int:
        for param in self.parameters():
            param.requires_grad = True
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_wavelet_adapters(self) -> None:
        """Resets all Theta cores to zero, restoring exact base W_0 behavior."""
        for layer in self.iter_wavelet_layers():
            layer.reset_theta()

    def apply_tree_memory_theta(self, tree_or_theta: Any) -> None:
        """
        Injects a WaveTreeMemory root core, DocumentTreeNode theta, or raw tensor/ndarray
        across all WaveletLowRankLinear layers. Passing None clears external theta.
        """
        if tree_or_theta is None:
            for layer in self.iter_wavelet_layers():
                layer.set_external_theta(None)
            return

        if hasattr(tree_or_theta, "root") and hasattr(tree_or_theta.root, "theta"):
            theta_val = tree_or_theta.root.theta
        elif hasattr(tree_or_theta, "theta"):
            theta_val = tree_or_theta.theta
        else:
            theta_val = tree_or_theta

        for layer in self.iter_wavelet_layers():
            layer.set_external_theta(theta_val)

    # =========================================================================
    # LOCAL FINE-TUNING & AUTOREGRESSIVE GENERATION
    # =========================================================================

    @staticmethod
    def _coerce_training_text(item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            if "messages" in item and isinstance(item["messages"], list):
                parts = []
                for msg in item["messages"]:
                    role = str(msg.get("role", "")).lower()
                    content = str(msg.get("content", "")).strip()
                    if not content or role == "system":
                        continue
                    tag = "Hỏi" if role == "user" else "Đáp"
                    parts.append(f"[{tag}]: {content}")
                return "\n".join(parts).strip()
            if "query" in item and "response" in item:
                return f"[Hỏi]: {item['query']}\n[Đáp]: {item['response']}".strip()
            if "text" in item:
                return str(item["text"]).strip()
        return str(item).strip() if item is not None else ""

    def finetune_local(
        self,
        sequences: Union[Sequence[Union[str, Dict[str, Any]]], torch.Tensor],
        tokenizer: Optional[Any] = None,
        steps: int = 20,
        lr: float = 1e-2,
        adapt_only_wavelet: bool = True,
        freeze_bases_ab: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes pure-PyTorch local fine-tuning (CPU/GPU, zero external APIs).
        Accepts token tensors, raw strings, or conversational SFT dicts (e.g. export_sft_training_samples()).
        When adapt_only_wavelet=True, freezes W_0 and adapts only Wavelet Low-Rank cores.
        """
        t0 = time.perf_counter()
        device = next(self.parameters()).device

        if isinstance(sequences, torch.Tensor):
            batch_ids = sequences.to(device=device, dtype=torch.long)
            if batch_ids.dim() == 1:
                batch_ids = batch_ids.unsqueeze(0)
        else:
            if tokenizer is None:
                raise ValueError("tokenizer is required when sequences are strings or SFT samples")
            texts = [self._coerce_training_text(item) for item in sequences]
            encoded = [
                tokenizer.encode(t, add_bos=True, add_eos=True, max_length=self.max_seq_len)
                for t in texts
                if t
            ]
            if not encoded:
                raise ValueError("No non-empty sequences provided for local fine-tuning")
            max_len = max(len(seq) for seq in encoded)
            pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
            padded = [seq + [pad_id] * (max_len - len(seq)) for seq in encoded]
            batch_ids = torch.tensor(padded, dtype=torch.long, device=device)

        if batch_ids.size(1) < 2:
            raise ValueError("Training sequences must contain at least 2 tokens")

        if adapt_only_wavelet:
            trainable_count = self.freeze_base_weights(freeze_bases_ab=freeze_bases_ab)
        else:
            trainable_count = self.unfreeze_all()

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)

        input_ids = batch_ids[:, :-1]
        targets = batch_ids[:, 1:].clone()
        pad_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        if pad_id is not None and not isinstance(sequences, torch.Tensor):
            targets[targets == pad_id] = -100

        self.train()
        loss_history: List[float] = []
        for _ in range(max(1, int(steps))):
            optimizer.zero_grad(set_to_none=True)
            _, loss, _ = self.forward(input_ids, targets=targets)
            if loss is None:
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            loss_history.append(float(loss.item()))

        self.eval()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "steps": len(loss_history),
            "initial_loss": loss_history[0] if loss_history else 0.0,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
            "trainable_params": trainable_count,
            "total_params": sum(p.numel() for p in self.parameters()),
            "adapt_only_wavelet": adapt_only_wavelet,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: Optional[int] = 20,
        eos_token_id: Optional[int] = None,
        theta: Optional[Union[torch.Tensor, np.ndarray]] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive generation with KV-cache and optional Wavelet Theta injection."""
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        generated = prompt_ids.to(device=device, dtype=torch.long)

        if generated.size(1) > self.max_seq_len:
            generated = generated[:, -self.max_seq_len:]

        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        if use_cache:
            logits, _, kv_caches = self.forward(generated, start_pos=0, use_cache=True, theta=theta)
            next_logits = logits[:, -1, :]
        else:
            next_logits = None

        for step in range(max_new_tokens):
            if not use_cache:
                idx_cond = generated[:, -self.max_seq_len:]
                logits, _, _ = self.forward(idx_cond, start_pos=0, use_cache=False, theta=theta)
                next_logits = logits[:, -1, :]
            elif step > 0:
                start_pos = generated.size(1) - 1
                if start_pos >= self.max_seq_len:
                    # Fallback cleanly if context exceeds max_seq_len
                    idx_cond = generated[:, -self.max_seq_len:]
                    logits, _, kv_caches = self.forward(idx_cond, start_pos=0, use_cache=True, theta=theta)
                    next_logits = logits[:, -1, :]
                else:
                    logits, _, kv_caches = self.forward(
                        generated[:, -1:],
                        start_pos=start_pos,
                        kv_caches=kv_caches,
                        use_cache=True,
                        theta=theta,
                    )
                    next_logits = logits[:, -1, :]

            if temperature <= 1e-5:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / float(temperature)
                if top_k is not None and top_k > 0:
                    k_val = min(int(top_k), scaled.size(-1))
                    v, _ = torch.topk(scaled, k_val)
                    scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        if was_training:
            self.train()
        return generated

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        tokenizer: Any,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: Optional[int] = 20,
        theta: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> str:
        """Convenience end-to-end string generation using a local tokenizer."""
        device = next(self.parameters()).device
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        if not prompt_ids:
            prompt_ids = [getattr(tokenizer, "bos_token_id", 1)]
        tensor_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out_ids = self.generate(
            tensor_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            theta=theta,
        )
        new_tokens = out_ids[0, len(prompt_ids):]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def get_architecture_summary(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        wavelet_layers = self.iter_wavelet_layers()
        theta_params = sum(layer.theta.numel() for layer in wavelet_layers)
        basis_params = sum(layer.a_proj.numel() + layer.b_proj.numel() for layer in wavelet_layers)
        base_params = total_params - theta_params - basis_params
        return {
            "architecture": "SovereignMiniTransformer",
            "pure_pytorch_offline": True,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "rank_r": self.rank_r,
            "wavelet_layers_count": len(wavelet_layers),
            "total_params": total_params,
            "base_params": base_params,
            "basis_ab_params": basis_params,
            "wavelet_theta_params": theta_params,
            "theta_compression_ratio": round(base_params / max(1, theta_params), 2),
        }

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
        }
        torch.save(payload, str(target))
        return target

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        map_location: str = "cpu",
    ) -> "SovereignMiniTransformer":
        payload = torch.load(str(path), map_location=map_location, weights_only=True)
        config = SovereignMiniConfig.from_dict(payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        return model

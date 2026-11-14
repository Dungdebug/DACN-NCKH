"""
Self-Built Sovereign Offline LLM & Tokenizer Package (Pillar 1).

Exports:
- SovereignTokenizer: Pure Python byte-level BPE tokenizer (zero external downloads).
- SovereignMiniConfig: Configuration dataclass for SovereignMiniTransformer.
- WaveletLowRankLinear: Generalized W_0 + B Theta A associative linear layer.
- WaveletCausalSelfAttention: Causal RoPE multi-head self-attention with Wavelet Low-Rank projections.
- WaveletSwiGLU: SwiGLU feed-forward network with Wavelet Low-Rank adaptation.
- SovereignTransformerBlock: Pre-RMSNorm decoder block.
- SovereignMiniTransformer: Pure-PyTorch self-built local LLM with Wavelet adaptation.
"""

from src.model.tokenizer import SovereignTokenizer, TokenBatch
from src.model.sovereign_transformer import (
    RMSNorm,
    RotaryEmbedding,
    SovereignMiniConfig,
    SovereignMiniTransformer,
    SovereignTransformerBlock,
    WaveletCausalSelfAttention,
    WaveletLowRankLinear,
    WaveletSwiGLU,
)

__all__ = [
    "SovereignTokenizer",
    "TokenBatch",
    "SovereignMiniConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "WaveletLowRankLinear",
    "WaveletCausalSelfAttention",
    "WaveletSwiGLU",
    "SovereignTransformerBlock",
    "SovereignMiniTransformer",
]

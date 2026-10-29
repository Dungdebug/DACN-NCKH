"""
Standard Transformer Causal Decoder LM baseline (equivalent to Hope with 1-level static memory).
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.backbone.embedding import TransformerEmbedding
from src.attention.causal_attention import CausalSelfAttention
from src.cms.mlp_chain import MLPBlock


class StandardTransformerBlock(nn.Module):
    """Standard Transformer Decoder block: Attention + LayerNorm + MLP + LayerNorm."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            max_seq_len=max_seq_len,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLPBlock(d_model=d_model, d_ff=d_ff, dropout=mlp_dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        norm_x = self.ln1(x)
        attn_out, new_cache = self.attn(norm_x, kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out

        norm_x = self.ln2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        return x, new_cache


class StandardTransformerLM(nn.Module):
    """
    Standard Transformer Causal Language Model.
    Serves as the In-Context Learning (ICL) 1-level baseline from paper Section 9.1.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.blocks = nn.ModuleList([
            StandardTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                attn_dropout=dropout,
                proj_dropout=dropout,
                mlp_dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedding.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        x = self.embedding(input_ids, start_pos=start_pos)
        new_kv_caches = [] if use_cache else None

        for layer_idx, block in enumerate(self.blocks):
            layer_cache = kv_caches[layer_idx] if kv_caches is not None else None
            x, updated_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_kv_caches.append(updated_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_kv_caches

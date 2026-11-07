"""
Token and Positional Embeddings for Transformer and Hope-Attention backbone.
"""

import math
import torch
import torch.nn as nn


class TransformerEmbedding(nn.Module):
    """
    Combines learned token embeddings with learned positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            start_pos: Offset for positional encoding (used in cached autoregressive decoding)

        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        B, T = input_ids.size()
        assert start_pos + T <= self.max_seq_len, (
            f"Sequence length {start_pos + T} exceeds max_seq_len {self.max_seq_len}"
        )

        positions = torch.arange(start_pos, start_pos + T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        return self.dropout(x)

"""
Autoregressive token generator supporting Hope-Attention and standard Transformer.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGenerator:
    """
    Handles autoregressive sequence generation with greedy decoding or top-k / temperature sampling.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            prompt_ids: (batch_size, prompt_len) or (prompt_len,)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = standard, 0.0 = greedy)
            top_k: Top-k filtering threshold
            eos_token_id: Stop generation if this token is produced

        Returns:
            generated_ids: (batch_size, prompt_len + generated_len)
        """
        self.model.eval()
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        generated = prompt_ids.to(self.device)

        for _ in range(max_new_tokens):
            # Crop to context window if needed
            max_len = getattr(self.model, "max_seq_len", 1024)
            idx_cond = generated if generated.size(1) <= max_len else generated[:, -max_len:]

            logits, _, _ = self.model(idx_cond)
            # Take logits at the last position: (batch_size, vocab_size)
            next_token_logits = logits[:, -1, :]

            if temperature <= 0.0 or temperature < 1e-5:
                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

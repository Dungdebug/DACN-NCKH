"""
Autoregressive Text Generator supporting Small Language Models (SLMs) and Self-Attention.

Theoretical Foundations:
1. Autoregressive Factorization:
   P(w_1, w_2, ..., w_T | Context) = \\prod_{t=1}^T P(w_t | w_{<t}, Context)
   Generates token-by-token conditioned on prior generated prefix and context tokens.

2. Causal Self-Attention:
   Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + M ) V
   where M is the causal lower-triangular mask (M_ij = -inf for j > i) enforcing
   strict non-anticipative autoregression.

3. Supported SLM Architectures:
   - SmolLM2-135M (Default ultra-lightweight NCKH backbone, 134.5M params)
   - Qwen-2.5-1.5B / Qwen-2.5-0.5B (Alibaba dense SLM)
   - Phi-3-mini-4k-instruct (Microsoft 3.8B lightweight model)
   - Llama-3.2-1B / Llama-3.2-3B (Meta open SLM)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SLMProfile:
    """Predefined profiles for popular Small Language Models (SLMs)."""
    name: str
    hf_repo_id: str
    parameter_count_str: str
    max_context_length: int
    chat_template_format: str  # e.g. "chatml", "phi3", "llama3", "plain"


SUPPORTED_SLMS: Dict[str, SLMProfile] = {
    "smollm2_135m": SLMProfile(
        name="SmolLM2-135M",
        hf_repo_id="HuggingFaceTB/SmolLM2-135M",
        parameter_count_str="134.5M",
        max_context_length=2048,
        chat_template_format="plain",
    ),
    "qwen2_5_1_5b": SLMProfile(
        name="Qwen-2.5-1.5B",
        hf_repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        parameter_count_str="1.54B",
        max_context_length=32768,
        chat_template_format="chatml",
    ),
    "phi3_mini": SLMProfile(
        name="Phi-3-Mini-4K",
        hf_repo_id="microsoft/Phi-3-mini-4k-instruct",
        parameter_count_str="3.8B",
        max_context_length=4096,
        chat_template_format="phi3",
    ),
    "llama3_2_1b": SLMProfile(
        name="Llama-3.2-1B",
        hf_repo_id="meta-llama/Llama-3.2-1B-Instruct",
        parameter_count_str="1.24B",
        max_context_length=8192,
        chat_template_format="llama3",
    ),
}


class TextGenerator:
    """
    Handles autoregressive sequence generation with causal attention masking,
    greedy decoding, temperature / top-k sampling, and repetition penalty.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        slm_profile: Optional[SLMProfile] = None,
    ):
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self.slm_profile = slm_profile or SUPPORTED_SLMS["smollm2_135m"]

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        stop_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Executes autoregressive token-by-token generation.

        Args:
            prompt_ids: (batch_size, prompt_len) or (prompt_len,)
            max_new_tokens: Maximum number of autoregressive steps
            temperature: Sampling temperature (0.0 = greedy)
            top_k: Top-k filtering threshold
            eos_token_id: Primary stop token ID
            repetition_penalty: Penalty factor (>1.0 reduces token repetition)
            stop_token_ids: Additional stop token IDs

        Returns:
            generated_ids: (batch_size, prompt_len + generated_len)
        """
        self.model.eval()
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        generated = prompt_ids.to(self.device)

        stop_set = set()
        if eos_token_id is not None:
            stop_set.add(eos_token_id)
        if stop_token_ids:
            stop_set.update(stop_token_ids)

        for _ in range(max_new_tokens):
            # Crop to context window if needed
            max_len = getattr(self.model, "max_seq_len", self.slm_profile.max_context_length)
            idx_cond = generated if generated.size(1) <= max_len else generated[:, -max_len:]

            # Forward pass: supports both tuple returns (logits, cache, ...) and CausalLMOutput
            out = self.model(idx_cond)
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            # Take logits at the last position: (batch_size, vocab_size)
            next_token_logits = logits[:, -1, :].clone()

            # Apply repetition penalty if specified
            if repetition_penalty > 1.0:
                for b in range(generated.size(0)):
                    prev_tokens = torch.unique(generated[b])
                    for t in prev_tokens:
                        if next_token_logits[b, t] > 0:
                            next_token_logits[b, t] /= repetition_penalty
                        else:
                            next_token_logits[b, t] *= repetition_penalty

            # Decoding strategy
            if temperature <= 0.0 or temperature < 1e-5:
                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Temperature & Top-k sampling
                scaled_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Check stop conditions
            if stop_set:
                token_val = next_token[0, 0].item()
                if token_val in stop_set:
                    break

        return generated

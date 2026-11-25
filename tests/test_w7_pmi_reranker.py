"""
Tests for Ticket T3 (w7-pmi): PMIReranker in src/slm/pmi_reranker.py.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List

import pytest
import torch
import torch.nn as nn

from src.slm.pmi_reranker import PMIReranker, should_run_pmi


class _StubTokenizer:
    """Deterministic whitespace tokenizer for fast CPU unit tests."""

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def _tok_id(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab)
        return self.vocab[tok]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not text.strip():
            return []
        return [self._tok_id(w) for w in text.strip().split()]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        specials = {0, 1, 2} if skip_special_tokens else set()
        return " ".join(inv.get(i, "<unk>") for i in ids if i not in specials)


class _DeterministicStubLM(nn.Module):
    """
    Deterministic stub LM whose logits at every position depend solely on
    whether a trigger token (representing passage context) appears in the prefix.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        trigger_logits: Dict[int, torch.Tensor] | None = None,
        null_logits: torch.Tensor | None = None,
        sleep_s: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.trigger_logits = trigger_logits or {}
        self.null_logits = (
            null_logits if null_logits is not None else torch.zeros(vocab_size)
        )
        self.sleep_s = sleep_s
        self.call_batches: List[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        bsz, seq_len = input_ids.shape
        self.call_batches.append(bsz)

        logits = torch.zeros((bsz, seq_len, self.vocab_size), dtype=torch.float32)
        for b in range(bsz):
            row = input_ids[b].tolist()
            chosen = self.null_logits
            for trig_id, vec in self.trigger_logits.items():
                if trig_id in row:
                    chosen = vec
                    break
            logits[b, :, :] = chosen.unsqueeze(0).expand(seq_len, -1)
        return (logits, None)


def test_pmi_exact_math_and_single_null_call():
    tok = _StubTokenizer()
    q = "alpha beta"
    q_ids = tok.encode(q)
    trig_good_id = tok.encode("good_ctx")[0]
    trig_bad_id = tok.encode("bad_ctx")[0]

    vocab_size = 64
    null_vec = torch.zeros(vocab_size)
    good_vec = torch.zeros(vocab_size)
    bad_vec = torch.zeros(vocab_size)

    # Boost query token logits in good_ctx, penalize in bad_ctx
    for qid in q_ids:
        null_vec[qid] = 0.0
        good_vec[qid] = 3.0
        bad_vec[qid] = -2.0

    lm = _DeterministicStubLM(
        vocab_size=vocab_size,
        trigger_logits={trig_good_id: good_vec, trig_bad_id: bad_vec},
        null_logits=null_vec,
    )
    reranker = PMIReranker(model=lm, tokenizer=tok, device="cpu", window=16, k=2)

    assert reranker.available is True
    assert reranker.mode_label == "base_smollm2"

    passages = [
        "irrelevant sentence. good_ctx explains alpha beta clearly.",
        "bad_ctx is noisy.",
        "tail passage beyond k=2 should get fallback score.",
    ]
    scores = reranker.score(q, passages, use_budget_guard=False)

    # Single-batch forward: [null_seq] + top-k passages in 1 batch (1 + 2 = 3)
    assert lm.call_batches == [3]

    # Hand-computed expected PMI
    logp_null = sum(torch.log_softmax(null_vec, dim=-1)[qid].item() for qid in q_ids)
    logp_good = sum(torch.log_softmax(good_vec, dim=-1)[qid].item() for qid in q_ids)
    logp_bad = sum(torch.log_softmax(bad_vec, dim=-1)[qid].item() for qid in q_ids)
    expected_good = (logp_good - logp_null) / len(q_ids)
    expected_bad = (logp_bad - logp_null) / len(q_ids)

    assert len(scores) == 3
    assert math.isclose(scores[0], expected_good, abs_tol=1e-6)
    assert math.isclose(scores[1], expected_bad, abs_tol=1e-6)
    assert scores[2] < min(scores[0], scores[1])


def test_window_extraction_prefers_keyword_sentence():
    tok = _StubTokenizer()
    lm = _DeterministicStubLM(vocab_size=256)
    reranker = PMIReranker(model=lm, tokenizer=tok, device="cpu", window=6, k=5)

    passage = (
        "one two three four five six seven eight. "
        "needle target keyword appears here. "
        "nine ten eleven twelve thirteen fourteen."
    )
    win = reranker._extract_best_window("needle keyword", passage)
    assert "needle" in win
    assert "keyword" in win
    assert len(tok.encode(win)) <= 6


def test_budget_guard_reduces_k_then_disables():
    tok = _StubTokenizer()
    # Single-batch forward sleeps 26ms -> > budget_ms=10ms and > 2.5*10=25ms
    lm = _DeterministicStubLM(vocab_size=64, sleep_s=0.026)
    reranker = PMIReranker(
        model=lm, tokenizer=tok, device="cpu", window=8, k=5, budget_ms=10.0
    )

    passages = ["p1", "p2", "p3", "p4", "p5"]

    # Call 1: exceeds 10ms -> k drops from 5 to 2
    reranker.score("alpha", passages, use_budget_guard=True)
    assert reranker.last_latency_ms > 10.0
    assert reranker.k == 2
    assert reranker.available is True

    # Call 2: still exceeds 2.5 * 10ms = 25ms -> disabled
    reranker.score("alpha", passages, use_budget_guard=True)
    assert reranker.last_latency_ms > 25.0
    assert reranker.available is False

    # Call 3: disabled -> returns neutral fallback without calling LM
    prev_calls = len(lm.call_batches)
    out = reranker.score("alpha", passages, use_budget_guard=True)
    assert out == [0.0] * len(passages)
    assert len(lm.call_batches) == prev_calls


def test_from_checkpoint_real_assets_and_cuda_latency():
    # Test W4b (SA-CMS) and W4a (base SmolLM2) checkpoint discovery
    reranker_cms = PMIReranker.from_checkpoint(enable_cms=True, k=5, window=64, budget_ms=60.0)
    assert reranker_cms.available is True
    assert reranker_cms.mode_label == "sa_cms_adapter"
    assert reranker_cms.checkpoint_path is not None

    reranker_base = PMIReranker.from_checkpoint(enable_cms=False, k=5, window=64, budget_ms=60.0)
    assert reranker_base.available is True
    assert reranker_base.mode_label == "base_smollm2"

    if torch.cuda.is_available():
        q = "What is Structure-Aligned Continuum Memory System?"
        passages = [
            "Structure-Aligned Continuum Memory System (SA-CMS) aligns memory updates with document boundaries.",
            "SmolLM2-135M is a compact causal language model used as the frozen backbone.",
            "BM25 provides lexical retrieval before neural PMI reranking.",
            "Unrelated passage about weather and ocean currents.",
            "Another passage describing standard transformer attention layers.",
        ]
        # Warmup CUDA kernels once
        reranker_cms.score(q, passages, use_budget_guard=False)
        # Timed run
        scores = reranker_cms.score(q, passages, use_budget_guard=False)
        assert len(scores) == 5
        assert reranker_cms.last_latency_ms > 0.0
        print(
            f"\n[CUDA] ckpt={reranker_cms.checkpoint_path} "
            f"latency={reranker_cms.last_latency_ms:.2f}ms "
            f"scores={[round(s, 4) for s in scores]}"
        )


def test_margin_gated_selective_pmi():
    tok = _StubTokenizer()
    q = "alpha beta"
    q_ids = tok.encode(q)
    trig_good_id = tok.encode("good_ctx")[0]
    vocab_size = 64
    null_vec = torch.zeros(vocab_size)
    good_vec = torch.zeros(vocab_size)
    for qid in q_ids:
        good_vec[qid] = 3.0

    lm = _DeterministicStubLM(
        vocab_size=vocab_size,
        trigger_logits={trig_good_id: good_vec},
        null_logits=null_vec,
    )
    reranker = PMIReranker(model=lm, tokenizer=tok, device="cpu", window=16, k=2)
    passages = ["good_ctx explains alpha beta.", "other passage."]

    assert should_run_pmi(lse_margin=0.40, cross_agree=0.0) is False
    assert should_run_pmi(lse_margin=0.10, cross_agree=0.70) is False
    assert should_run_pmi(lse_margin=0.10, cross_agree=0.20) is True

    # Gated off by high margin -> (None, False), 0 LM calls
    scores_off, ran_off = reranker.score_passages_gated(q, passages, lse_margin=0.40, cross_agree=0.1)
    assert scores_off is None
    assert ran_off is False
    assert lm.call_batches == []

    # Gated on -> (scores, True), 1 single-batch LM call
    scores_on, ran_on = reranker.score_passages_gated(q, passages, lse_margin=0.15, cross_agree=0.2, k=2)
    assert ran_on is True
    assert scores_on is not None and len(scores_on) == 2
    assert lm.call_batches == [3]

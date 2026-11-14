"""
Tests for Offline Persona SFT & 4-Pillar Closed-Loop Pipeline.

Verifies:
1. Strict offline execution (socket guard blocks non-loopback network access).
2. Prompt-masked conversational batching from AcademicPersonaEngine.export_sft_training_samples().
3. PersonaSFTTrainer converging on pure-PyTorch StandardTransformerLM + SovereignSLMGenerator (< 40 ms CPU).
4. PersonaSFTTrainer converging on SovereignMiniTransformer with W_0 frozen (only Wavelet Theta updated).
5. End-to-end 4-Pillar closed loop (AcademicPersonaEngine -> SLM SFT -> SovereignSLMGenerator -> WaveCMSEngine + Exact Unlearning).
"""

import socket

import pytest
import torch

from src.backbone.transformer import StandardTransformerLM
from src.model import SovereignMiniConfig, SovereignMiniTransformer
from src.self_study.persona import export_sft_training_samples
from src.slm import SovereignSLMGenerator
from src.training.persona_sft import (
    PersonaSFTTrainer,
    build_persona_sft_tokenizer,
    prepare_sft_batch,
    run_four_pillar_closed_loop,
)


@pytest.fixture(autouse=True)
def guard_strict_offline():
    """Blocks any accidental external network calls during SFT and inference."""
    orig_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        allowed = ("127.0.0.1", "localhost", "::1", "testserver")
        if isinstance(host, str) and (host in allowed or host.startswith("127.")):
            return orig_connect(self, address)
        raise ConnectionRefusedError(f"[OFFLINE VIOLATION] Blocked attempt to {address}")

    socket.socket.connect = guarded
    yield
    socket.socket.connect = orig_connect


def test_prepare_sft_batch_masks_prompt_and_preserves_targets():
    samples = export_sft_training_samples()[:6]
    tok = build_persona_sft_tokenizer(samples=samples, vocab_size=480, min_frequency=2)

    input_ids, targets = prepare_sft_batch(
        samples=samples,
        tokenizer=tok,
        max_seq_len=192,
        include_system=False,
        mask_prompt_loss=True,
    )

    assert input_ids.shape == targets.shape
    assert input_ids.size(0) == len(samples)
    assert input_ids.size(1) <= 191

    # Every row must mask at least the prompt prefix and retain supervised assistant tokens
    for row_idx in range(targets.size(0)):
        row = targets[row_idx]
        assert (row == -100).any(), "Prompt/padding tokens should be masked with -100"
        assert (row != -100).any(), "Assistant answer tokens must remain supervised"


def test_standard_transformer_persona_sft_and_slm_generator():
    torch.manual_seed(101)
    samples = export_sft_training_samples()[:8]
    tok = build_persona_sft_tokenizer(samples=samples, vocab_size=480, min_frequency=2)

    model = StandardTransformerLM(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        max_seq_len=192,
        tie_weights=True,
    )
    trainer = PersonaSFTTrainer(tokenizer=tok, max_seq_len=192, mask_prompt_loss=True)
    res = trainer.train(
        model=model,
        samples=samples,
        steps=8,
        lr=5e-3,
        adapt_only_wavelet=False,
    )

    assert res.model_name == "StandardTransformerLM"
    assert res.adapt_only_wavelet is False
    assert res.trainable_params == res.total_params
    assert res.final_loss < res.initial_loss
    assert res.final_ppl < res.initial_ppl

    slm = SovereignSLMGenerator(backbone=model, tokenizer=tok, device="cpu")
    gen = slm.generate(
        query="Mã hóa đối xứng là gì?",
        passages=[
            {
                "passage_id": "P1",
                "citation_label": "[1]",
                "text": "Mã hóa đối xứng sử dụng chung một khóa bí mật cho cả mã hóa và giải mã, tiêu biểu gồm AES và DES.",
            }
        ],
        answer_mode="balanced",
    )
    assert gen.backbone_mode == "SLM_BACKBONE"
    assert gen.is_grounded is True
    assert gen.latency_ms < 40.0


def test_sovereign_mini_transformer_wavelet_sft_and_closed_loop():
    torch.manual_seed(2026)
    samples = export_sft_training_samples()[:8]
    tok = build_persona_sft_tokenizer(samples=samples, vocab_size=480, min_frequency=2)

    cfg = SovereignMiniConfig(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=192,
        rank_r=8,
    )
    model = SovereignMiniTransformer(cfg)
    w0_snapshot = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if "theta" not in name
    }

    report = run_four_pillar_closed_loop(
        model=model,
        tokenizer=tok,
        samples=samples,
        steps=10,
        lr=2e-2,
        adapt_only_wavelet=True,
        max_seq_len=192,
    )

    # W_0 base parameters must stay bit-exact frozen
    for name, param in model.named_parameters():
        if "theta" not in name:
            assert torch.equal(param, w0_snapshot[name]), f"Frozen base weight {name} changed"

    sft = report["sft"]
    assert sft["model_name"] == "SovereignMiniTransformer"
    assert sft["adapt_only_wavelet"] is True
    assert sft["trainable_params"] < sft["total_params"]
    assert sft["final_loss"] < sft["initial_loss"]
    assert sft["final_ppl"] < sft["initial_ppl"]

    assert all(report["intent_checks"].values())
    assert report["slm_generation"]["backbone_mode"] == "SLM_BACKBONE"
    assert report["slm_generation"]["is_grounded"] is True
    assert report["slm_generation"]["latency_ms"] < 40.0

    assert report["wave_query"]["refused"] is False
    assert report["wave_query"]["contraction_factor"] >= 0.0
    assert len(report["wave_query"]["answer"]) > 20
    assert report["unlearning"]["certified_exact"] is True
    assert report["unlearning"]["max_discrepancy"] < 1e-12

"""
Comprehensive Test Suite for Sovereign Offline SLM/LLM Generator (Pillar 1).
Validates:
- 100% Offline execution (strict socket block against non-loopback network calls).
- Sub-40ms CPU execution SLA for both standalone and PyTorch SLM backbone modes.
- Pedagogical 4-tier Vietnamese scientific synthesis via ExtractiveTextStitcher.
- Batched SLM backbone candidate re-ranking without autoregressive drift.
- Integration with WaveCMSEngine & CausalAttributionItem readouts.
"""

import socket
import time
import pytest
import torch

from src.backbone.transformer import StandardTransformerLM
from src.model import SovereignMiniConfig, SovereignMiniTransformer, SovereignTokenizer
from src.slm import SovereignGenerationResult, SovereignSLMGenerator
from src.wave_cms.causal_attribution import CausalAttributionItem
from src.wave_cms.wave_engine import WaveCMSEngine


@pytest.fixture(autouse=True)
def guard_strict_offline():
    """Blocks any external non-loopback network calls."""
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


@pytest.fixture
def sample_crypto_passages():
    return [
        {
            "passage_id": "DOC1::P1",
            "citation_label": "[1]",
            "text": (
                "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:\n"
                "Hệ mã khóa đối xứng sử dụng chung một khóa bí mật K cho cả mã hóa và giải mã.\n"
                "* Mã hóa thay thế (Substitution cipher).\n"
                "* Mã hóa hoán vị (Transposition cipher).\n"
            ),
        },
        {
            "passage_id": "DOC1::P2",
            "citation_label": "[2]",
            "text": (
                "8. HỆ MÃ HOÁ AUTOKEY:\n"
                "Hệ mã Autokey mở rộng khóa bằng chính bản rõ để tránh chu kỳ lặp.\n"
                "Công thức: ci = (pi + ki) mod 26 với k1 = p0.\n"
                "Tuy nhiên, hạn chế của hệ mã cổ điển là dễ bị phá mã thống kê tần suất nếu không gian khóa nhỏ.\n"
            ),
        },
    ]


class _ByteTokenizer:
    """Lightweight offline deterministic tokenizer for PyTorch SLM unit testing."""
    pad_token_id = 0

    def encode(self, text: str, truncation: bool = True, max_length: int = 96):
        ids = [max(1, ord(ch) % 250) for ch in text]
        return ids[:max_length] if truncation else ids


def test_standalone_detailed_4tier_synthesis(sample_crypto_passages):
    gen = SovereignSLMGenerator()
    res: SovereignGenerationResult = gen.generate(
        query="Hệ mã khóa đối xứng là gì và có các phương pháp nào?",
        passages=sample_crypto_passages,
        answer_mode="detailed",
    )

    assert res.is_grounded is True
    assert res.backbone_mode == "STANDALONE_FAST"
    assert res.latency_ms < 40.0
    assert res.telemetry["sub_40ms_sla_passed"] is True
    assert "Khái niệm" in res.answer
    assert "thay thế" in res.answer.lower()
    assert "autokey" in res.answer.lower()


def test_balanced_and_concise_modes(sample_crypto_passages):
    gen = SovereignSLMGenerator()

    res_bal = gen.generate(
        query="Hệ mã Autokey có ưu điểm và hạn chế gì?",
        passages=sample_crypto_passages,
        answer_mode="balanced",
    )
    assert res_bal.is_grounded is True
    assert res_bal.latency_ms < 40.0
    assert len(res_bal.answer) > 20

    res_con = gen.generate(
        query="Hệ mã khóa đối xứng dùng khóa nào?",
        passages=sample_crypto_passages,
        answer_mode="concise",
    )
    assert res_con.is_grounded is True
    assert res_con.latency_ms < 40.0
    assert "[1]" in res_con.answer
    assert len(res_con.answer.split()) < 45


def test_pytorch_slm_backbone_reranking_under_40ms(sample_crypto_passages):
    """Validates batched PyTorch SLM backbone scoring within strict < 40ms SLA."""
    tiny_slm = StandardTransformerLM(
        vocab_size=256,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
    )
    tok = _ByteTokenizer()
    gen = SovereignSLMGenerator(backbone=tiny_slm, tokenizer=tok, device="cpu")
    assert gen.has_neural_backbone is True

    # Warm-up pass (PyTorch JIT/allocator)
    _ = gen.generate(
        query="Hệ mã Autokey hoạt động ra sao?",
        passages=sample_crypto_passages,
        answer_mode="balanced",
    )

    # Measured pass
    res = gen.generate(
        query="Hệ mã Autokey hoạt động ra sao?",
        passages=sample_crypto_passages,
        answer_mode="balanced",
    )
    assert res.backbone_mode == "SLM_BACKBONE"
    assert res.is_grounded is True
    assert res.latency_ms < 40.0, f"Exceeded 40ms SLA: {res.latency_ms} ms"
    assert res.telemetry["sub_40ms_sla_passed"] is True


def test_wavecms_readout_synthesis_adapter():
    gen = SovereignSLMGenerator()
    items = [
        CausalAttributionItem(
            leaf_id="DOC1::L1",
            causal_score=0.4512,
            rank=1,
            original_alpha=0.72,
            full_logprob=-1.2,
            counterfactual_logprob=-1.6512,
            title="9. HỆ MÃ HOÁ DÒNG (STREAM CIPHER)",
            text=(
                "Mã hóa dòng xử lý từng bit hoặc byte dữ liệu theo thời gian thực.\n"
                "- Sử dụng phép toán XOR giữa bản rõ và dòng khóa giả ngẫu nhiên.\n"
                "Ví dụ: ci = pi XOR zi."
            ),
            doc_id="DOC1",
        ),
        CausalAttributionItem(
            leaf_id="DOC1::L2",
            causal_score=0.1840,
            rank=2,
            original_alpha=0.28,
            full_logprob=-1.2,
            counterfactual_logprob=-1.3840,
            title="8. HỆ MÃ HOÁ AUTOKEY",
            text="Hệ mã Autokey dùng công thức ci = (pi + ki) mod 26.",
            doc_id="DOC1",
        ),
    ]

    t0 = time.perf_counter()
    ans_detailed = gen.generate_from_wave_readout(
        query="Mã hóa dòng hoạt động như thế nào?",
        causal_items=items,
        resolution_level="LEAF_PARAGRAPH",
        answer_mode="detailed",
    )
    lat_ms = (time.perf_counter() - t0) * 1000

    assert lat_ms < 40.0
    assert "WaveCMS" in ans_detailed
    assert "LEAF_PARAGRAPH" in ans_detailed
    assert "XOR" in ans_detailed
    assert "Autokey" in ans_detailed

    ans_balanced = gen.generate_from_wave_readout(
        query="Mã hóa dòng hoạt động như thế nào?",
        causal_items=items,
        resolution_level="SECTION",
        answer_mode="balanced",
    )
    assert "Đóng góp nhân quả c(p) = +0.1840" in ans_balanced

    ans_concise = gen.generate_from_wave_readout(
        query="Mã hóa dòng là gì?",
        causal_items=items,
        resolution_level="LEAF_PARAGRAPH",
        answer_mode="concise",
    )
    assert "[Causal Citation: DOC1::L1]" in ans_concise


def test_wave_cms_engine_end_to_end_uses_sovereign_slm():
    engine = WaveCMSEngine(r=8, dim=64)
    engine.ingest_document(
        doc_id="CRYPTO_SLM",
        title="Mật mã học cơ sở",
        sections=[
            {
                "section_title": "1. GIỚI THIỆU HỆ MÃ ĐỐI XỨNG",
                "paragraphs": [
                    "Hệ mã khóa đối xứng sử dụng cùng một khóa bí mật cho cả hai phía gửi và nhận.",
                    "- Gồm hai kỹ thuật chính: Mã hóa thay thế và Mã hóa hoán vị.",
                ],
            }
        ],
    )

    res = engine.query("Hệ mã khóa đối xứng là gì?", answer_mode="detailed")
    assert res.refused is False
    assert "WaveCMS" in res.answer
    assert "đối xứng" in res.answer.lower()
    assert res.telemetry["total_latency_ms"] < 100.0


def test_empty_inputs_safe_fallback():
    gen = SovereignSLMGenerator()
    res = gen.generate(query="", passages=[])
    assert res.is_grounded is False
    assert res.num_citations == 0
    assert "Không tìm thấy đủ thông tin" in res.answer


def test_sovereign_mini_transformer_with_wavelet_theta_under_40ms(sample_crypto_passages):
    """Validates direct integration with self-built SovereignMiniTransformer + SovereignTokenizer + Theta."""
    tok = SovereignTokenizer()
    cfg = SovereignMiniConfig(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        rank_r=8,
        max_seq_len=128,
    )
    model = SovereignMiniTransformer(cfg)
    gen = SovereignSLMGenerator(backbone=model, tokenizer=tok, device="cpu")
    theta = torch.randn(8, 8) * 0.05

    # Warm-up
    _ = gen.generate(
        query="Hệ mã Autokey hoạt động ra sao?",
        passages=sample_crypto_passages,
        answer_mode="balanced",
        theta=theta,
    )

    res = gen.generate(
        query="Hệ mã Autokey hoạt động ra sao?",
        passages=sample_crypto_passages,
        answer_mode="balanced",
        theta=theta,
    )
    assert res.backbone_mode == "SLM_BACKBONE"
    assert res.is_grounded is True
    assert res.latency_ms < 40.0, f"Exceeded 40ms SLA with SovereignMiniTransformer: {res.latency_ms} ms"


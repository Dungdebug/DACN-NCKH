"""
Tests for Self-Built SovereignMiniTransformer & SovereignTokenizer (Pillar 1).
Verifies:
1. 100% offline execution (socket guard blocks non-loopback connections).
2. Lossless Vietnamese/English BPE tokenization and JSON persistence.
3. Exact associative equivalence of WaveletLowRankLinear (W_0 + B Theta A).
4. KV-cache vs non-cached autoregressive equivalence (RoPE + causal masking).
5. Local CPU fine-tuning of only r x r Theta cores with W_0 strictly frozen.
6. Integration with WaveTreeMemory (Theorem 1 & Theorem 2 unlearning transfer).
7. Compatibility with src.inference.generator.TextGenerator and checkpoint I/O.
"""

import socket
import pytest
import numpy as np
import torch

from src.inference.generator import TextGenerator
from src.model import (
    SovereignMiniConfig,
    SovereignMiniTransformer,
    SovereignTokenizer,
    WaveletLowRankLinear,
)
from src.wave_cms.tree_memory import WaveTreeMemory


@pytest.fixture(autouse=True)
def guard_strict_offline():
    """Blocks any accidental external network calls."""
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


def test_sovereign_tokenizer_lossless_vietnamese_and_persistence(tmp_path):
    texts = [
        "Hệ thống WaveCMS sử dụng cây wavelet đa phân giải W_0 + B Theta A.",
        "Mã hóa đối xứng AES và ChaCha20 dùng chung khóa bí mật.",
        "Định lý Parseval và Banach đảm bảo hội tụ điểm bất động.",
    ]
    tok = SovereignTokenizer.build_academic_tokenizer(extra_texts=texts, vocab_size=400, min_frequency=2)
    assert tok.vocab_size > SovereignTokenizer.BASE_VOCAB_SIZE
    assert tok.vocab_size <= 400

    sample = "Kiến trúc WaveCMS (W_0 + BΘA): độ phức tạp O(depth) < 1e-12!"
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id
    assert tok.decode(ids, skip_special_tokens=True) == sample

    # Batch tensor encoding
    batch = tok(["Xin chào WaveCMS", "Tài liệu"], return_tensors="pt", padding=True)
    assert batch.input_ids.shape[0] == 2
    assert batch.attention_mask.shape == batch.input_ids.shape

    # Save & load JSON locally
    save_path = tok.save(tmp_path / "tok.json")
    loaded = SovereignTokenizer.load(save_path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode(sample) == tok.encode(sample)


def test_wavelet_low_rank_linear_equivalence_and_zero_init():
    torch.manual_seed(7)
    layer = WaveletLowRankLinear(in_features=64, out_features=192, r=8, bias=True)
    x = torch.randn(2, 5, 64)

    # Zero-initialized Theta must match W_0 exactly
    y_base = layer.w_0(x)
    y_init = layer(x)
    assert torch.allclose(y_init, y_base, atol=1e-6)

    # Non-zero Theta associative path vs dense W_eff = W_0 + B Theta A
    theta = torch.randn(8, 8)
    is_equiv, max_disc = layer.verify_associative_equivalence(x, theta=theta, tol=1e-5)
    assert is_equiv is True
    assert max_disc < 1e-5

    # Batched 3D theta (B, r, r)
    batch_theta = torch.randn(2, 8, 8)
    y_batched = layer(x, theta=batch_theta)
    y_single_0 = layer(x[0:1], theta=batch_theta[0])
    y_single_1 = layer(x[1:2], theta=batch_theta[1])
    assert torch.allclose(y_batched[0:1], y_single_0, atol=1e-5)
    assert torch.allclose(y_batched[1:2], y_single_1, atol=1e-5)


def test_kv_cache_matches_uncached_greedy_generation():
    torch.manual_seed(42)
    cfg = SovereignMiniConfig(
        vocab_size=320,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=192,
        max_seq_len=64,
        rank_r=8,
    )
    model = SovereignMiniTransformer(cfg)
    model.eval()

    prompt = torch.tensor([[1, 12, 45, 78, 99]], dtype=torch.long)
    theta = torch.randn(8, 8) * 0.2

    out_cached = model.generate(prompt, max_new_tokens=10, temperature=0.0, theta=theta, use_cache=True)
    out_uncached = model.generate(prompt, max_new_tokens=10, temperature=0.0, theta=theta, use_cache=False)
    assert torch.equal(out_cached, out_uncached)

    # Compatibility with existing TextGenerator
    gen = TextGenerator(model)
    out_text_gen = gen.generate(prompt, max_new_tokens=5, temperature=0.0)
    assert out_text_gen.shape == (1, 10)


def test_local_wavelet_finetuning_freezes_w0_and_resets_cleanly():
    torch.manual_seed(123)
    corpus = [
        "WaveCMS dùng cây wavelet đa phân giải cho bộ nhớ liên tục.",
        "WaveCMS đạt sai số unlearning nhỏ hơn 1e-12 trong O(depth).",
        "Lớp thích nghi W_0 + B Theta A cập nhật cục bộ trên CPU.",
    ]
    tok = SovereignTokenizer.build_academic_tokenizer(extra_texts=corpus, vocab_size=360, min_frequency=2)
    cfg = SovereignMiniConfig(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=192,
        max_seq_len=128,
        rank_r=8,
    )
    model = SovereignMiniTransformer(cfg)

    # Snapshot W_0 weights and baseline logits before adaptation
    w0_snapshot = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if "theta" not in name
    }
    probe_ids = tok("WaveCMS dùng cây wavelet", return_tensors="pt").input_ids
    model.eval()
    with torch.no_grad():
        base_logits, _, _ = model(probe_ids)

    # Fine-tune ONLY Theta cores (W_0, A, B frozen)
    stats = model.finetune_local(
        corpus,
        tokenizer=tok,
        steps=18,
        lr=3e-2,
        adapt_only_wavelet=True,
        freeze_bases_ab=True,
    )
    assert stats["final_loss"] < stats["initial_loss"]
    assert stats["trainable_params"] == len(model.iter_wavelet_layers()) * (8 * 8)

    # Verify W_0, A, B are bit-exact unchanged
    for name, p in model.named_parameters():
        if "theta" not in name:
            assert torch.equal(p, w0_snapshot[name]), f"Frozen parameter {name} was modified!"

    # Verify adapted logits differ from base W_0 logits
    model.eval()
    with torch.no_grad():
        adapted_logits, _, _ = model(probe_ids)
    assert not torch.allclose(adapted_logits, base_logits, atol=1e-4)

    # Verify reset_wavelet_adapters() restores base W_0 logits bit-for-bit
    model.reset_wavelet_adapters()
    with torch.no_grad():
        reset_logits, _, _ = model(probe_ids)
    assert torch.allclose(reset_logits, base_logits, atol=1e-6)


def test_wave_tree_memory_integration_and_certified_unlearning_transfer():
    torch.manual_seed(9)
    tok = SovereignTokenizer.build_academic_tokenizer(vocab_size=320)
    cfg = SovereignMiniConfig(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=192,
        max_seq_len=128,
        rank_r=8,
    )
    model = SovereignMiniTransformer(cfg)
    model.eval()

    doc_a = {
        "doc_id": "doc_a",
        "title": "Symmetric Cryptography",
        "sections": [{"section_title": "AES", "paragraphs": ["AES uses 128-bit block size and shared secret key."]}],
    }
    doc_b = {
        "doc_id": "doc_b",
        "title": "WaveCMS Theory",
        "sections": [{"section_title": "Unlearning", "paragraphs": ["Exact unlearning removes doc_b in O(depth)."]}],
    }

    full_tree = WaveTreeMemory(r=8, key_dim=64)
    full_tree.ingest_document(**doc_a)
    full_tree.ingest_document(**doc_b)

    shadow_tree = WaveTreeMemory(r=8, key_dim=64)
    shadow_tree.ingest_document(**doc_a)

    probe_ids = tok("AES block size", return_tensors="pt").input_ids
    hidden, pooled = model.encode_hidden(probe_ids)
    assert hidden.shape == (1, probe_ids.size(1), 64)
    assert pooled.shape == (1, 64)

    # Inject full tree Theta and verify score shifts from base W_0
    base_score = model.score_sequence(probe_ids)
    model.apply_tree_memory_theta(full_tree)
    full_score = model.score_sequence(probe_ids)
    assert base_score != full_score

    # Unlearn doc_b from full_tree and transfer updated root Theta to Transformer
    unlearn_report = full_tree.certify_exact_unlearning("doc_b")
    assert unlearn_report["certified_exact"] is True

    model.apply_tree_memory_theta(full_tree)
    with torch.no_grad():
        logits_unlearned, _, _ = model(probe_ids)

    model.apply_tree_memory_theta(shadow_tree)
    with torch.no_grad():
        logits_shadow, _, _ = model(probe_ids)

    assert torch.allclose(logits_unlearned, logits_shadow, atol=1e-6)


def test_checkpoint_save_load_and_text_generation(tmp_path):
    torch.manual_seed(11)
    tok = SovereignTokenizer.build_academic_tokenizer(vocab_size=320)
    cfg = SovereignMiniConfig(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=192,
        max_seq_len=64,
        rank_r=8,
    )
    model = SovereignMiniTransformer(cfg)
    summary = model.get_architecture_summary()
    assert summary["pure_pytorch_offline"] is True
    assert summary["wavelet_layers_count"] == 2 * (4 + 3)  # 2 layers * (Q,K,V,O + gate,up,down)

    ckpt_path = model.save_checkpoint(tmp_path / "sovereign_mini.pt")
    reloaded = SovereignMiniTransformer.load_checkpoint(ckpt_path)

    probe = tok("WaveCMS", return_tensors="pt").input_ids
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        l1, _, _ = model(probe)
        l2, _, _ = reloaded(probe)
    assert torch.allclose(l1, l2, atol=1e-6)

    gen_text = reloaded.generate_text("WaveCMS", tok, max_new_tokens=6, temperature=0.0)
    assert isinstance(gen_text, str)


def test_closed_loop_persona_sft_to_wavelet_slm():
    """Verifies Pillar 4 SFT samples -> SovereignTokenizer -> SovereignMiniTransformer (W_0 + B Theta A) -> SovereignSLMGenerator."""
    from src.self_study.persona import export_sft_training_samples
    from src.slm import SovereignSLMGenerator

    torch.manual_seed(2026)
    sft_samples = export_sft_training_samples()[:6]
    sft_texts = [SovereignMiniTransformer._coerce_training_text(s) for s in sft_samples]

    tok = SovereignTokenizer.build_academic_tokenizer(extra_texts=sft_texts, vocab_size=512, min_frequency=2)
    cfg = SovereignMiniConfig(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=256,
        rank_r=8,
    )
    model = SovereignMiniTransformer(cfg)

    stats = model.finetune_local(
        sft_samples,
        tokenizer=tok,
        steps=10,
        lr=2e-2,
        adapt_only_wavelet=True,
        freeze_bases_ab=True,
    )
    assert stats["final_loss"] < stats["initial_loss"]

    slm = SovereignSLMGenerator(backbone=model, tokenizer=tok, device="cpu")
    passages = [
        {
            "passage_id": "P1",
            "citation_label": "[1]",
            "text": "Mã hóa đối xứng sử dụng chung một khóa bí mật cho cả mã hóa và giải mã, gồm AES và DES.",
        }
    ]
    res = slm.generate(query="Mã hóa đối xứng là gì?", passages=passages, answer_mode="balanced")
    assert res.backbone_mode == "SLM_BACKBONE"
    assert res.is_grounded is True
    assert res.latency_ms < 40.0


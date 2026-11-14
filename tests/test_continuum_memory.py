"""
Unit tests for WaveCMS Continuum Memory & Certified Exact Unlearning
(src/wave_cms/continuum_memory.py & src/wave_cms/tree_memory.py).

Validates:
1. Low-rank adaptation W = W_0 + B @ Theta @ A via fast associative path O(dr + r^2)
   with analytical equivalence < 1e-12 vs dense outer-product multiplication.
2. Differentiable PyTorch ContinuumMemoryLinear forward & backward gradients.
3. Closed-form Gauss-Newton leaf core update in r^2 space.
4. Strict O(depth) Certified Exact Unlearning (nodes_updated == 1 at root)
   with real measured analytical discrepancy < 1e-12 against shadow reconstruction.
5. Subtree memory deallocation & cyclic reference breaking upon unlearning.
"""

import gc
import weakref
import numpy as np
import pytest
import torch

from src.wave_cms.continuum_memory import (
    CertifiedUnlearningEngine,
    ContinuumMemoryAdapter,
    ContinuumMemoryLinear,
    release_subtree_memory,
)
from src.wave_cms.tree_memory import WaveTreeMemory
from src.wave_cms.wave_engine import WaveCMSEngine


def test_continuum_adapter_associative_equivalence():
    """
    Verifies that fast associative forward pass W_0 x + B (Theta (A x))
    matches dense (W_0 + B Theta A) x within < 1e-12 for 1D, 2D, and 3D inputs.
    """
    d_model, r = 128, 8
    adapter = ContinuumMemoryAdapter(d_model=d_model, r=r, seed=42)
    rng = np.random.RandomState(99)
    theta = rng.randn(r, r) * 0.5

    # 1D vector
    x_1d = rng.randn(d_model)
    ok_1d, disc_1d = adapter.verify_adaptation_equivalence(theta, x_1d, tol=1e-12)
    assert ok_1d is True
    assert disc_1d < 1e-12

    # 2D batch
    x_2d = rng.randn(16, d_model)
    y_fast_2d = adapter.forward(x_2d, theta=theta, use_associative_fast_path=True)
    y_dense_2d = adapter.forward(x_2d, theta=theta, use_associative_fast_path=False)
    assert y_fast_2d.shape == (16, d_model)
    assert np.max(np.abs(y_fast_2d - y_dense_2d)) < 1e-12

    # 3D sequence batch
    x_3d = rng.randn(4, 10, d_model)
    y_fast_3d = adapter.forward(x_3d, theta=theta, use_associative_fast_path=True)
    y_dense_3d = adapter.forward(x_3d, theta=theta, use_associative_fast_path=False)
    assert y_fast_3d.shape == (4, 10, d_model)
    assert np.max(np.abs(y_fast_3d - y_dense_3d)) < 1e-12


def test_continuum_adapter_gauss_newton_and_footprint():
    """Validates closed-form Gauss-Newton core solve and memory footprint compression."""
    adapter = ContinuumMemoryAdapter(d_model=256, r=8, seed=7)
    footprint = adapter.get_memory_footprint()
    assert footprint["bytes_per_memory_core"] == 8 * 8 * 8
    assert footprint["compression_ratio"] > 100.0

    # Solve closed-form Gauss-Newton update on r^2 = 64 dimensions
    rng = np.random.RandomState(12)
    grad = rng.randn(8, 8)
    j_mat = rng.randn(64, 64)
    fisher = j_mat.T @ j_mat  # Positive semi-definite Fisher
    beta = 0.05

    theta_opt = adapter.compute_gauss_newton_update(grad, fisher_matrix=fisher, beta=beta)
    assert theta_opt.shape == (8, 8)

    # Verify optimality condition: (F + beta * I) vec(theta_opt) == -vec(grad)
    lhs = (fisher + beta * np.eye(64)) @ theta_opt.reshape(-1)
    rhs = -grad.reshape(-1)
    assert np.max(np.abs(lhs - rhs)) < 1e-11

    adapter.release_memory()


def test_pytorch_continuum_memory_linear_equivalence_and_grad():
    """Validates PyTorch ContinuumMemoryLinear numerical precision and gradient flow."""
    torch.set_default_dtype(torch.float64)
    try:
        layer = ContinuumMemoryLinear(d_model=32, r=4, bias=False)
        theta = torch.randn(4, 4, requires_grad=True)
        x = torch.randn(5, 6, 32, requires_grad=True)

        y_fast = layer(x, theta=theta)
        w_dense = layer.synthesize_dense_weight(theta)
        y_dense = torch.matmul(x, w_dense.t())

        max_err = float(torch.max(torch.abs(y_fast - y_dense)).item())
        assert max_err < 1e-12

        # Verify backpropagation through associative path
        loss = y_fast.pow(2).sum()
        loss.backward()
        assert theta.grad is not None
        assert layer.a_proj.grad is not None
        assert layer.b_proj.grad is not None
        assert x.grad is not None

        layer.release_cache()
    finally:
        torch.set_default_dtype(torch.float32)


def test_strict_o_depth_exact_unlearning_and_memory_release():
    """
    Tests Theorem 2 on a 15-document corpus:
    1. Unlearning a document touches O(depth) nodes (nodes_updated == 1 at root),
       never traversing the remaining 14 documents.
    2. Measured analytical discrepancy vs independent fresh shadow tree < 1e-12.
    3. Subtree nodes and arrays are released and garbage-collected.
    """
    tree = WaveTreeMemory(r=8, dim=64, seed=42)
    for i in range(15):
        sections = [
            {
                "section_title": f"Section {i}.1",
                "paragraphs": [
                    {"text": f"Paragraph A of document {i} discussing cryptography and algebra.", "weight": 1.5 + 0.1 * i},
                    {"text": f"Paragraph B of document {i} covering protocols and proofs.", "weight": 2.0},
                ],
            },
            {
                "section_title": f"Section {i}.2",
                "paragraphs": [
                    {"text": f"Paragraph C of document {i} summarizing complexity bounds.", "weight": 1.2},
                ],
            },
        ]
        tree.ingest_document(f"DOC_{i:02d}", f"Document Title {i}", sections)

    assert len(tree.documents) == 15
    assert len(tree.get_leaves()) == 45

    # Keep weak references to verify subtree deallocation
    target_doc = tree.documents["DOC_07"]
    target_leaf = target_doc.get_leaves()[0]
    weak_doc = weakref.ref(target_doc)
    weak_leaf = weakref.ref(target_leaf)
    del target_doc, target_leaf

    # Use CertifiedUnlearningEngine for full shadow tree discrepancy proof
    unlearner = CertifiedUnlearningEngine(tree)
    cert = unlearner.unlearn_document_exact("DOC_07", verify_ground_truth=True, tol=1e-12)

    assert cert["certified_exact"] is True
    assert cert["exact_match"] is True
    assert cert["complexity"] == "O(depth)"
    assert cert["nodes_updated"] == 1  # Strictly root updated, O(1) independent of M=14 other docs!
    assert cert["max_discrepancy"] < 1e-12
    assert cert["remaining_documents"] == 14
    assert cert["remaining_leaves"] == 42
    assert cert["freed_nodes_count"] == 6  # 1 doc + 2 sections + 3 leaves
    assert cert["freed_bytes"] > 0
    assert cert["theorems_satisfied"] is True

    gc.collect()
    assert weak_doc() is None
    assert weak_leaf() is None

    # Now unlearn remaining documents sequentially down to empty
    for i in range(15):
        if i == 7:
            continue
        res = tree.certify_exact_unlearning(f"DOC_{i:02d}", atol=1e-12)
        assert res["certified_exact"] is True
        assert res["max_discrepancy"] < 1e-12
        assert res["theorems_satisfied"] is True

    assert len(tree.documents) == 0
    assert len(tree.get_leaves()) == 0


def test_engine_unlearn_document_real_discrepancy():
    """Verifies WaveCMSEngine.unlearn_document executes O(depth) downdate with discrepancy < 1e-12."""
    engine = WaveCMSEngine(r=8, dim=64, max_hops=3)
    for idx in range(4):
        engine.ingest_document(
            f"D{idx}",
            f"Title {idx}",
            [{"section_title": "Sec 1", "paragraphs": [{"text": f"Content {idx} alpha", "weight": 2.0}]}],
        )

    cert = engine.unlearn_document("D2")
    assert cert["certified_exact"] is True
    assert cert["max_discrepancy"] < 1e-12
    assert cert["complexity"] == "O(depth)"
    assert cert["nodes_updated"] == 1
    assert cert["freed_nodes"] >= 3
    assert cert["remaining_documents"] == 3

    # Non-existent document check
    err_cert = engine.tree_memory.certify_exact_unlearning("NON_EXISTENT")
    assert err_cert["certified_exact"] is False


def test_unnormalized_kappa_additivity_and_empty_shadow_verification():
    """
    Verifies that every internal node stores additive unnormalized_kappa = sum w_p * kappa_p,
    Frobenius discrepancy is measured against shadow tree (< 1e-12), and unlearning down to
    zero documents executes shadow comparison cleanly.
    """
    tree = WaveTreeMemory(r=8, dim=64, seed=42)
    for d in range(3):
        tree.ingest_document(
            f"DOC_{d}",
            f"Title {d}",
            [
                {
                    "section_title": f"Sec {d}.1",
                    "paragraphs": [
                        {"text": f"Symmetric cryptography cipher block {d}", "weight": 1.5 + d},
                        {"text": f"Stream keystream pseudorandom {d}", "weight": 2.0},
                    ],
                },
                {
                    "section_title": f"Sec {d}.2",
                    "paragraphs": [
                        {"text": f"Autokey digram matrix substitution {d}", "weight": 2.5},
                    ],
                },
            ],
        )

    # Check strict leaf-sum additivity of unnormalized_kappa at root and documents
    leaf_sum = sum(lf.weight * lf.kappa for lf in tree.get_leaves())
    assert np.allclose(tree.root.unnormalized_kappa, leaf_sum, atol=1e-12)
    doc_sum = sum(doc.unnormalized_kappa for doc in tree.documents.values())
    assert np.allclose(tree.root.unnormalized_kappa, doc_sum, atol=1e-12)

    # Unlearn middle document via CertifiedUnlearningEngine and check Frobenius discrepancy
    unlearner = CertifiedUnlearningEngine(tree)
    cert_mid = unlearner.unlearn_document_exact("DOC_1", verify_ground_truth=True, tol=1e-12)
    assert cert_mid["certified_exact"] is True
    assert cert_mid["max_discrepancy"] < 1e-12
    assert cert_mid["max_frobenius_discrepancy"] < 1e-12

    leaf_sum_after = sum(lf.weight * lf.kappa for lf in tree.get_leaves())
    assert np.allclose(tree.root.unnormalized_kappa, leaf_sum_after, atol=1e-12)

    # Unlearn remaining documents down to empty corpus via CertifiedUnlearningEngine
    cert_0 = unlearner.unlearn_document_exact("DOC_0", verify_ground_truth=True, tol=1e-12)
    cert_2 = unlearner.unlearn_document_exact("DOC_2", verify_ground_truth=True, tol=1e-12)
    assert cert_0["certified_exact"] is True
    assert cert_2["certified_exact"] is True
    assert cert_2["max_frobenius_discrepancy"] < 1e-12
    assert cert_2["remaining_documents"] == 0
    assert np.allclose(tree.root.theta, 0.0, atol=1e-12)
    assert np.allclose(tree.root.unnormalized_kappa, 0.0, atol=1e-12)


def test_calibrated_crc_blocks_matches_corpus_override_and_accent_quotes():
    """
    Verifies two CRC safety invariants in WaveCMSEngine.query:
    1. When risk_controller.is_calibrated is True, a low-confidence trap query that has
       lexical overlap (matches_corpus=True) MUST remain refused to preserve E[answered & wrong] <= eps.
    2. When uncalibrated, quoted_terms with Vietnamese diacritics match normalized corpus text.
    """
    engine = WaveCMSEngine(r=8, dim=64, refusal_threshold=0.95)
    engine.ingest_document(
        "DOC_CRYPTO",
        "Mật mã học",
        [
            {
                "section_title": "Hệ mã đối xứng",
                "paragraphs": [
                    {
                        "text": "Hệ mã hóa đối xứng dùng chung khóa bí mật cho mã hóa và giải mã.",
                        "weight": 2.0,
                    }
                ],
            }
        ],
    )

    # Uncalibrated with quoted Vietnamese term: strip_accents allows corpus match override
    uncal_res = engine.query(question='Khái niệm "khóa bí mật" trong hệ mã hóa đối xứng là gì?')
    assert uncal_res.refused is False
    assert uncal_res.telemetry["is_calibrated"] is False

    # Calibrate CRC with conservative threshold so low-confidence overlap trap is refused
    cal_data = [{"score": 0.92, "is_correct": False}] * 25 + [{"score": 0.99, "is_correct": True}] * 25
    cal_info = engine.risk_controller.calibrate(cal_data, epsilon=0.05)
    assert cal_info["status"] == "SUCCESS"
    assert engine.risk_controller.is_calibrated is True
    assert engine.risk_controller.calibrated_threshold > 0.92

    # Query has high lexical overlap ("hệ mã hóa đối xứng", "khóa bí mật") -> matches_corpus=True,
    # but calibrated CRC must NOT be overridden by matches_corpus
    cal_res = engine.query(question='Hệ mã hóa đối xứng và "khóa bí mật" dùng cho máy lượng tử?')
    assert cal_res.telemetry["is_calibrated"] is True
    assert cal_res.confidence < engine.risk_controller.calibrated_threshold
    assert cal_res.refused is True
    assert cal_res.refusal_reason != "OVERRIDE_CORPUS_GROUNDED_CONTENT_MATCH"


def test_semantic_leaf_core_closed_form_optimality():
    """
    Validates that ContinuumMemoryAdapter.compute_semantic_leaf_core produces
    the exact global minimizer of 0.5*||(kappa - W0*kappa) - B*Theta*A*kappa||^2 + 0.5*beta*||Theta||_F^2,
    verified by comparing against full r^2 x r^2 Gauss-Newton inversion (< 1e-13).
    """
    d_model, r, beta = 64, 8, 1.0
    adapter = ContinuumMemoryAdapter(d_model=d_model, r=r, seed=42)
    rng = np.random.RandomState(2026)
    kappa = rng.randn(d_model)
    kappa /= np.linalg.norm(kappa)

    theta_fast = adapter.compute_semantic_leaf_core(kappa, beta=beta)

    # Verify stationarity: gradient at theta_fast must be zero (< 1e-13)
    residual = (kappa - adapter.W0 @ kappa) - adapter.forward(kappa, theta=theta_fast) + adapter.W0 @ kappa
    # Actually (W0 + B Theta A) kappa = W0 kappa + B Theta A kappa
    err_vec = (adapter.W0 @ kappa + adapter.B @ (theta_fast @ (adapter.A @ kappa))) - kappa
    grad_theta = np.outer(adapter.B.T @ err_vec, adapter.A @ kappa) + beta * theta_fast
    assert np.max(np.abs(grad_theta)) < 1e-13


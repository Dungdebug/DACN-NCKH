"""
Unit tests for WaveCMS Tree Memory Architecture (src/wave_cms/tree_memory.py).
Validates:
- DocumentTreeNode structure & hierarchy
- Bottom-up Telescoping Consolidation
- Theorem 1(a): Exact Telescoping Reconstruction (error < 1e-12)
- Theorem 1(b): Wavelet Expansion vs Direct Leaf Sum (error < 1e-12)
- Theorem 1(c): Haar Vanishing on Tree Siblings (error < 1e-12)
- Theorem 1(d): Parseval Energy Conservation (|E_leaves - E_wavelet| < 1e-10)
- Theorem 2: Certified Exact Unlearning in O(depth) with Zero Residual Forgetting (error < 1e-12)
"""

import numpy as np
import pytest

from src.wave_cms.tree_memory import (
    DocumentTreeNode,
    TreeLevel,
    WaveTreeMemory,
    compute_deterministic_routing_key,
)


@pytest.fixture
def sample_tree_memory() -> WaveTreeMemory:
    tree = WaveTreeMemory(r=8, key_dim=32, corpus_id="TEST_CORPUS")

    doc1_sections = [
        {
            "section_title": "Chương 1: Khái niệm Cơ bản",
            "section_id": "DOC1::SEC1",
            "paragraphs": [
                {
                    "text": "Mã hóa đối xứng là phương pháp dùng chung một khóa cho cả hai quá trình.",
                    "weight": 1.5,
                },
                {
                    "text": "Hệ mã cổ điển bao gồm mã hóa thay thế và mã hóa hoán vị.",
                    "weight": 2.0,
                },
            ],
        },
        {
            "section_title": "Chương 2: Thám mã và Độ an toàn",
            "section_id": "DOC1::SEC2",
            "paragraphs": [
                {
                    "text": "Thám mã là quá trình tìm khóa hoặc bản rõ mà không biết trước khóa bí mật.",
                    "weight": 1.0,
                },
            ],
        },
    ]

    doc2_sections = [
        {
            "section_title": "Phần I: Giới thiệu Hệ phân tán",
            "section_id": "DOC2::SEC1",
            "paragraphs": [
                {
                    "text": "Hệ thống phân tán bao gồm nhiều nút tính toán độc lập kết nối qua mạng.",
                    "weight": 2.5,
                },
                {
                    "text": "Định lý CAP nêu rõ tính đánh đổi giữa Tính nhất quán và Tính sẵn sàng.",
                    "weight": 1.8,
                },
            ],
        },
        {
            "section_title": "Phần II: Cơ chế Đồng thuận Paxos và Raft",
            "section_id": "DOC2::SEC2",
            "paragraphs": [
                {
                    "text": "Raft phân chia bài toán đồng thuận thành bầu chọn lãnh đạo và sao chép nhật ký.",
                    "weight": 3.0,
                },
            ],
        },
    ]

    tree.add_document("DOC_CRYPTO", "Giáo trình An toàn Mật mã", doc1_sections)
    tree.add_document("DOC_DISTRIB", "Nguyên lý Hệ thống Phân tán", doc2_sections)
    return tree


def test_tree_hierarchy_and_dimensions(sample_tree_memory):
    tree = sample_tree_memory
    assert tree.root.level == TreeLevel.CORPUS
    assert len(tree.root.children) == 2

    doc1 = tree.get_node("DOC_CRYPTO")
    assert doc1 is not None
    assert doc1.level == TreeLevel.DOCUMENT
    assert len(doc1.children) == 2

    leaves = tree.get_leaves()
    assert len(leaves) == 6  # 3 paragraphs in doc1 + 3 paragraphs in doc2

    for node in tree.get_all_nodes():
        assert node.theta.shape == (8, 8)
        assert node.detail_dv.shape == (8, 8)
        assert len(node.routing_key) == 32
        assert np.isfinite(node.theta).all()
        assert np.isfinite(node.detail_dv).all()


def test_theorem_1a_exact_telescoping(sample_tree_memory):
    tree = sample_tree_memory
    is_valid, max_err = tree.verify_telescoping_identity(tol=1e-12)
    assert is_valid, f"Theorem 1(a) Telescoping failed with error {max_err}"
    assert max_err < 1e-12


def test_theorem_1b_wavelet_expansion(sample_tree_memory):
    tree = sample_tree_memory
    # Test with random normalized distribution
    is_valid, err = tree.verify_wavelet_expansion(tol=1e-12)
    assert is_valid, f"Theorem 1(b) Wavelet expansion failed with error {err}"
    assert err < 1e-12

    # Test with sparse one-hot routing
    leaves = tree.get_leaves()
    for leaf in leaves:
        alpha = {l.node_id: 1.0 if l.node_id == leaf.node_id else 0.0 for l in leaves}
        t_dir, t_wav = tree.reconstruct_routed_memory(alpha)
        assert np.allclose(t_dir, t_wav, atol=1e-12)
        assert np.allclose(t_dir, leaf.theta, atol=1e-12)


def test_theorem_1c_haar_vanishing(sample_tree_memory):
    tree = sample_tree_memory
    is_valid, max_err = tree.verify_haar_vanishing(tol=1e-12)
    assert is_valid, f"Theorem 1(c) Haar vanishing failed with error {max_err}"
    assert max_err < 1e-12

    # Corollary: When alpha is the uninformative proportional prior, Theta(alpha) = Theta_root
    total_w = tree.root.subtree_weight
    leaves = tree.get_leaves()
    prior_alpha = {l.node_id: l.weight / total_w for l in leaves}
    t_dir, t_wav = tree.reconstruct_routed_memory(prior_alpha)
    assert np.allclose(t_dir, tree.root.theta, atol=1e-12)
    assert np.allclose(t_wav, tree.root.theta, atol=1e-12)


def test_theorem_1d_parseval_energy_conservation(sample_tree_memory):
    tree = sample_tree_memory
    is_conserved, e_leaves, e_wav = tree.verify_parseval_identity(tol=1e-10)
    assert is_conserved, f"Parseval failed: E_leaves={e_leaves}, E_wavelet={e_wav}, delta={abs(e_leaves - e_wav)}"
    assert abs(e_leaves - e_wav) < 1e-10


def test_theorem_2_certified_exact_unlearning(sample_tree_memory):
    tree = sample_tree_memory

    # Unlearn DOC_CRYPTO
    cert = tree.certify_exact_unlearning("DOC_CRYPTO")
    assert cert["certified_exact"] is True
    assert cert["max_discrepancy"] < 1e-12
    assert cert["complexity"] == "O(depth)"
    assert cert["remaining_documents"] == 1
    assert cert["remaining_leaves"] == 3

    # Verify that unlearned tree still satisfies all Theorem 1 identities
    all_thm = tree.verify_all_theorems()
    assert all_thm["all_theorems_verified"] is True


def test_routing_key_deterministic_and_normalized():
    k1 = compute_deterministic_routing_key("Hệ mã hóa đối xứng và mã cổ điển", dim=64)
    k2 = compute_deterministic_routing_key("Hệ mã hóa đối xứng và mã cổ điển", dim=64)
    k3 = compute_deterministic_routing_key("Cơ sở dữ liệu quan hệ và SQL", dim=64)

    assert np.allclose(k1, k2, atol=1e-15)
    assert abs(np.linalg.norm(k1) - 1.0) < 1e-12
    assert abs(np.linalg.norm(k3) - 1.0) < 1e-12

    # Semantic similarity check: identical text has similarity 1.0, distinct text < 0.85
    sim_diff = float(np.dot(k1, k3))
    assert sim_diff < 0.85


def test_aliases_and_properties(sample_tree_memory):
    tree = sample_tree_memory

    # Property aliases on DocumentTreeNode
    leaf = tree.get_leaves()[0]
    assert np.array_equal(leaf.detail, leaf.detail_dv)
    assert np.array_equal(leaf.kappa, leaf.routing_key)

    leaf.detail = leaf.detail + 0.1
    assert np.array_equal(leaf.detail, leaf.detail_dv)

    leaf.kappa = leaf.kappa + 0.1
    assert np.array_equal(leaf.kappa, leaf.routing_key)

    # Method aliases on WaveTreeMemory
    tree.telescoping_consolidation()
    assert tree.verify_telescoping() is True
    assert tree.verify_haar() is True
    e_l, e_w, ok = tree.verify_parseval()
    assert ok is True
    assert abs(e_l - e_w) < 1e-10


def test_four_theorems_across_arbitrary_document_scales():
    """
    Validates all 4 Theorem 1 properties (Telescoping, Wavelet Expansion,
    Haar Vanishing, Parseval Conservation) across empty, minimal, unbalanced,
    extreme-weight, deep, and large multi-document trees.
    """
    # 1. Empty tree
    empty_tree = WaveTreeMemory(r=8, dim=32, seed=1)
    assert empty_tree.verify_all_theorems()["all_theorems_verified"] is True

    # 2. Single-leaf minimal document
    min_tree = WaveTreeMemory(r=8, dim=32, seed=2)
    min_tree.ingest_document("doc_min", "Minimal", [{"title": "S1", "paragraphs": ["Single passage."]}])
    assert min_tree.verify_all_theorems()["all_theorems_verified"] is True

    # 3. Unbalanced document with extreme weight skew (1e-4 to 1e5)
    skew_tree = WaveTreeMemory(r=8, dim=32, seed=3)
    skew_sections = [
        {"title": "Tiny", "paragraphs": [{"text": "Tiny paragraph weight", "weight": 1e-4}]},
        {"title": "Huge", "paragraphs": [{"text": "Huge paragraph weight", "weight": 1e5}]},
        {
            "title": "Uneven",
            "paragraphs": [
                {"text": f"Uneven section leaf {i}", "weight": float(i + 1)}
                for i in range(7)
            ],
        },
    ]
    skew_tree.ingest_document("doc_skew", "Skewed Tree", skew_sections)
    skew_res = skew_tree.verify_all_theorems()
    assert skew_res["all_theorems_verified"] is True
    assert skew_tree.verify_haar() is True

    # 4. Deep hierarchy (depth 5)
    deep_tree = WaveTreeMemory(r=8, dim=32, seed=4)
    topic = DocumentTreeNode("topic_0", TreeLevel.SECTION, title="Topic")
    coll = DocumentTreeNode("coll_0", TreeLevel.SECTION, title="Collection")
    doc = DocumentTreeNode("doc_deep", TreeLevel.DOCUMENT, title="Deep Doc")
    sec = DocumentTreeNode("sec_0", TreeLevel.SECTION, title="SubSection")
    for idx, w in enumerate([1.5, 4.2, 0.8]):
        th, kp = deep_tree._init_leaf_core(f"Deep leaf content {idx}", "SubSection")
        sec.add_child(
            DocumentTreeNode(
                f"leaf_deep_{idx}",
                TreeLevel.LEAF_PARAGRAPH,
                title=f"P{idx}",
                text=f"Deep leaf content {idx}",
                weight=w,
                theta=th,
                kappa=kp,
            )
        )
    doc.add_child(sec)
    coll.add_child(doc)
    topic.add_child(coll)
    deep_tree.root.add_child(topic)
    deep_tree.documents["doc_deep"] = doc
    deep_tree.telescoping_consolidation()
    assert deep_tree.verify_all_theorems()["all_theorems_verified"] is True

    # 5. Large corpus (25 docs, 300 leaves) + rank sweep + post-unlearning verification
    large_tree = WaveTreeMemory(r=16, dim=64, seed=5)
    for d in range(25):
        secs = [
            {"title": f"D{d} S{s}", "paragraphs": [f"Corpus doc {d} sec {s} para {p}" for p in range(4)]}
            for s in range(3)
        ]
        large_tree.ingest_document(f"doc_{d}", f"Document {d}", secs)
    assert len(large_tree.get_leaves()) == 300
    large_res = large_tree.verify_all_theorems()
    assert large_res["all_theorems_verified"] is True
    assert large_res["parseval_conservation"]["relative_error"] < 1e-12

    cert = large_tree.certify_exact_unlearning("doc_12")
    assert cert["certified_exact"] is True
    assert cert["theorems_satisfied"] is True


def test_tree_softmax_unbiased_routing_and_wavelet_expansion_consistency():
    """
    Verifies that TreeSoftmaxRouter maintains exact Theorem 1(b) consolidation
    on unbalanced trees and yields unbiased uniform leaf weights when query h = 0.
    """
    import torch
    from src.wave_cms.equilibrium_readout import TreeSoftmaxRouter

    tree = WaveTreeMemory(r=8, dim=32, seed=7)
    tree.ingest_document("doc_small", "Small", [{"title": "S1", "paragraphs": ["Single small leaf."]}])
    tree.ingest_document(
        "doc_large",
        "Large",
        [
            {"title": "L1", "paragraphs": [f"Large branch leaf {i}" for i in range(4)]},
            {"title": "L2", "paragraphs": [f"Second large branch leaf {i}" for i in range(4)]},
        ],
    )

    router = TreeSoftmaxRouter(dim=32, tau=1.0)
    # Zero query -> unbiased uniform distribution across all 9 leaves via log|L(c)| prior
    res_zero = router.route(torch.zeros(32), tree)
    for w in res_zero.leaf_weights.values():
        assert abs(w - (1.0 / 9.0)) < 1e-6

    # Focused query -> Theorem 1(b) exact consistency between direct sum and wavelet sum
    res_q = router.route(torch.randn(32), tree)
    theta_wav = router.synthesize_memory_core(res_q, tree)
    theta_dir = torch.zeros((8, 8), dtype=torch.float32)
    for lf in tree.get_leaves():
        theta_dir += res_q.leaf_weights[lf.node_id] * torch.tensor(lf.theta, dtype=torch.float32)
    assert torch.norm(theta_wav - theta_dir, p="fro").item() < 1e-5

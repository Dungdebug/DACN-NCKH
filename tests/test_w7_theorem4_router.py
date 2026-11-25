"""
Tests for Ticket T2 (w7-router):
- Theorem 3*: Bit-Exact Homomorphic State (FixedPointVector, TermCounter, corpus_df)
- Theorem 4: Admissible Sparse-LSE Routing (Hierarchy-Exact, Dilution-Free, Banach Contraction)
"""

import math
import random
from typing import Dict, List

import numpy as np
import pytest
import torch

from src.wave_cms.equilibrium_readout import EquilibriumMultiHopReadout, TreeSoftmaxRouter
from src.wave_cms.tree_memory import WaveTreeMemory, extract_sparse_terms
from src.wave_cms.wave_engine import WaveCMSEngine


def _build_multi_doc_tree(n_docs: int = 12, dim: int = 64, r: int = 8) -> WaveTreeMemory:
    tree = WaveTreeMemory(r=r, dim=dim, seed=42)
    vocab_groups = [
        ["quantum", "entanglement", "qubit", "superposition", "decoherence"],
        ["cryptography", "autokey", "vigenere", "cipher", "plaintext"],
        ["wavelet", "haar", "telescoping", "parseval", "multiresolution"],
        ["banach", "contraction", "lipschitz", "equilibrium", "fixedpoint"],
        ["conformal", "calibration", "quantile", "coverage", "refusal"],
        ["homomorphic", "unlearning", "downdate", "bitexact", "integer"],
    ]
    for i in range(n_docs):
        g1 = vocab_groups[i % len(vocab_groups)]
        g2 = vocab_groups[(i + 1) % len(vocab_groups)]
        sections = [
            {
                "title": f"Section A of Doc {i}",
                "passages": [
                    f"Document {i} discusses {g1[0]} and {g1[1]} with {g1[2]} mechanisms.",
                    f"Further analysis of {g1[3]} and {g1[4]} in system {i}.",
                ],
            },
            {
                "title": f"Section B of Doc {i}",
                "passages": [
                    f"Secondary topic {g2[0]} combined with {g2[1]} protocol {i}.",
                ],
            },
        ]
        tree.ingest_document(doc_id=f"doc_{i:02d}", title=f"Doc {i}", sections=sections)
    return tree


def _flat_sparse_lse_reference(
    router: TreeSoftmaxRouter,
    h: torch.Tensor,
    tree: WaveTreeMemory,
    query_text: str,
) -> Dict[str, float]:
    leaves = tree.get_leaves()
    n_leaves = len(leaves)
    query_terms = list(dict.fromkeys(extract_sparse_terms(query_text)))
    idf_map = {
        t: max(0.0, float(tree.corpus_df.idf(t, n_leaves)))
        for t in query_terms
    }
    avgdl = max(1.0, sum(lf.doc_len for lf in leaves) / max(1, n_leaves))
    k1, b = 1.2, 0.75
    tau = max(1e-4, router.temperature)
    tau_s = max(1e-4, router.tau_sparse)
    q_proj = router.w_q(h.squeeze().to(router.w_q.weight.device))

    logits: List[float] = []
    for lf in leaves:
        sparse_sum = 0.0
        denom_norm = k1 * (1.0 - b + b * (lf.doc_len / avgdl))
        for t in query_terms:
            tf = lf.term_tf.get(t, 0)
            if tf > 0:
                sparse_sum += idf_map[t] * (tf / (tf + denom_norm))
        key = router._get_node_key(lf).to(q_proj.device)
        dense_score = float(torch.dot(q_proj, key).item()) / tau
        logits.append((sparse_sum / tau_s) + dense_score)

    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    z_sum = sum(exps)
    return {lf.node_id: e / z_sum for lf, e in zip(leaves, exps)}


def test_theorem4_tree_argmax_matches_flat_200_random_queries():
    """Verifies tree_argmax == flat_argmax on 200 random queries (100% exact match)."""
    dim = 64
    tree = _build_multi_doc_tree(n_docs=12, dim=dim, r=8)
    router = TreeSoftmaxRouter(dim=dim, tau=0.25, routing_mode="sparse_lse", tau_sparse=1.0)

    rng = random.Random(2026)
    torch_gen = torch.Generator().manual_seed(2026)
    word_pool = [
        "quantum", "entanglement", "qubit", "cryptography", "autokey", "vigenere",
        "wavelet", "haar", "telescoping", "parseval", "banach", "contraction",
        "conformal", "calibration", "homomorphic", "unlearning", "unknownword",
    ]

    for _ in range(200):
        q_words = rng.sample(word_pool, k=rng.randint(1, 5))
        q_text = " ".join(q_words)
        h = torch.randn(dim, generator=torch_gen)
        h = h / (torch.norm(h, p=2) + 1e-8)

        res = router.route(h, tree, query_text=q_text, memory_tree=tree)
        flat_alpha = _flat_sparse_lse_reference(router, h, tree, q_text)

        tree_argmax = max(res.leaf_weights.items(), key=lambda kv: kv[1])[0]
        flat_argmax = max(flat_alpha.items(), key=lambda kv: kv[1])[0]
        assert tree_argmax == flat_argmax

        bnb_id, _, _ = router.branch_and_bound_argmax(h, tree, query_text=q_text, memory_tree=tree)
        assert bnb_id == flat_argmax


def test_theorem4_lse_telescoping_and_banach_contraction():
    """
    Verifies LSE-telescoping: sum(alpha) == 1, max|alpha_tree - alpha_flat| <= 1e-12,
    Theorem 1(b) wavelet expansion holds, and Banach rho_hat < 1.
    """
    dim = 64
    tree = _build_multi_doc_tree(n_docs=8, dim=dim, r=8)
    router = TreeSoftmaxRouter(dim=dim, tau=5.0, routing_mode="sparse_lse", tau_sparse=1.0)
    readout = EquilibriumMultiHopReadout(dim=dim, r=8, omega=0.5, max_hops=5, tol=1e-4, router=router)

    q_text = "banach contraction lipschitz equilibrium"
    eq_res = readout.readout(query=q_text, tree=tree, query_text=q_text)

    assert eq_res.is_contractive is True
    assert eq_res.contraction_factor < 1.0

    # Check exact flat equivalence at the final fixed point h*
    res = router.route(eq_res.fixed_point, tree, query_text=q_text, memory_tree=tree)
    flat_alpha = _flat_sparse_lse_reference(router, eq_res.fixed_point, tree, q_text)

    assert abs(sum(res.leaf_weights.values()) - 1.0) <= 1e-12
    max_diff = max(
        abs(res.leaf_weights[lid] - flat_alpha[lid])
        for lid in flat_alpha
    )
    assert max_diff <= 1e-12

    # Check internal node mass p_v == sum_{l in L(v)} alpha_l
    for node in tree.get_all_nodes():
        subtree_mass = sum(res.leaf_weights[lf.node_id] for lf in node.get_leaves())
        assert abs(res.node_probabilities[node.node_id] - subtree_mass) <= 1e-12

    # Theorem 1(b) holds with routed alpha
    t1b_ok, t1b_err = tree.verify_wavelet_expansion(tol=1e-12, alpha=res.leaf_weights)
    assert t1b_ok is True
    assert t1b_err < 1e-12


def test_pillar1_mini_dilution_free_routing_hit_at_1():
    """
    Pillar-1 mini (D0 target + 20 noise documents):
    sparse_lse achieves routing_hit_at_1 >= 0.90 while mean_key suffers dilution.
    """
    noise_docs = [
        {
            "doc_id": f"noise_{i:02d}",
            "title": f"Technical Report {i}: Ma hoa, Wavelet, Banach, Conformal",
            "sections": [
                {
                    "title": f"Overview {i}",
                    "passages": [
                        f"Noise passage {i} discussing ma hoa, thanh ghi, khoa, bien doi, dinh ly, kiem soat rui ro, phan tich, cau truc cay, bo dem, giao thuc.",
                        f"Additional filler content {i} about ma hoa, dinh ly, bien doi, he thong, du lieu, bao mat, thuat toan, mo hinh.",
                        f"Extended background {i} on weather forecast rainfall temperature humidity agriculture market logistics.",
                    ],
                }
            ],
        }
        for i in range(20)
    ]

    d0_facts = [
        ("fact_01", "Ma hoa dong Stream Cipher su dung thanh ghi dich LFSR dong bo."),
        ("fact_02", "He ma Autokey dung ban ro truoc do lam khoa tiep theo."),
        ("fact_03", "Bien doi Haar Wavelet bao toan nang luong theo dinh ly Parseval."),
        ("fact_04", "Dinh ly Banach dam bao hoi tu tuyen tinh ve diem bat dong duy nhat."),
        ("fact_05", "Kiem soat rui ro Conformal Risk Control gioi han ty le sai sot huu han."),
        ("fact_06", "Phan tich nhan qua Leave-One-Leaf-Out do luong dong gop tung doan."),
        ("fact_07", "Cau truc cay Wavelet cho phep xoa tri thuc chinh xac trong O(depth)."),
        ("fact_08", "Bo dem thưa TermCounter luu tan suat xuat hien cua thuat ngu."),
        ("fact_09", "Vector dau phay tinh FixedPointVector dam bao tinh giao hoan bit-exact."),
        ("fact_10", "Giao thuc xac thuc Zero-Knowledge Proof bao ve tinh rieng tu du lieu."),
    ]

    queries = [
        ("Thanh ghi dich LFSR duoc dung trong ma hoa nao?", "leaf:D0:p001"),
        ("He ma Autokey lay khoa tu dau?", "leaf:D0:p002"),
        ("Dinh ly Parseval lien quan den bien doi nao?", "leaf:D0:p003"),
        ("Dinh ly Banach dam bao dieu gi?", "leaf:D0:p004"),
        ("Conformal Risk Control kiem soat dai luong nao?", "leaf:D0:p005"),
        ("Leave-One-Leaf-Out do luong gi?", "leaf:D0:p006"),
        ("Cay Wavelet xoa tri thuc voi do phuc tap nao?", "leaf:D0:p007"),
        ("TermCounter luu tru thong tin gi?", "leaf:D0:p008"),
        ("FixedPointVector dam bao tinh chat nao?", "leaf:D0:p009"),
        ("Zero-Knowledge Proof bao ve dieu gi?", "leaf:D0:p010"),
    ]

    def _run_eval(routing_mode: str) -> float:
        engine = WaveCMSEngine(r=8, dim=128, tau=0.20, routing_mode=routing_mode)
        d0_passages = [{"passage_id": f"leaf:D0:p{idx+1:03d}", "text": txt} for idx, (_, txt) in enumerate(d0_facts)]
        engine.ingest_document(
            doc_id="D0",
            title="Core Technical Specification",
            sections=[{"title": "Core Facts", "passages": d0_passages}],
        )
        for nd in noise_docs:
            engine.ingest_document(doc_id=nd["doc_id"], title=nd["title"], sections=nd["sections"])

        hits = 0
        for q_text, expected_leaf in queries:
            eq_res = engine.readout_engine.readout(
                query=q_text,
                tree=engine.tree_memory,
                max_hops=2,
                query_text=q_text,
            )
            top_leaf = max(eq_res.final_alpha.items(), key=lambda kv: kv[1])[0]
            if top_leaf == expected_leaf:
                hits += 1
        return hits / len(queries)

    sparse_hit_at_1 = _run_eval("sparse_lse")
    mean_key_hit_at_1 = _run_eval("mean_key")

    assert sparse_hit_at_1 >= 0.90, f"Expected sparse_lse hit@1 >= 0.90, got {sparse_hit_at_1}"
    assert sparse_hit_at_1 > mean_key_hit_at_1


def test_theorem3_star_bit_exact_homomorphic_unlearning():
    """
    Verifies Theorem 3*: ingest docs -> unlearn D_target ->
    term_count, corpus_df, kappa_mass_fp match bit-for-bit (==) a clean tree
    that never ingested D_target, and verify_all_theorems() passes.
    """
    docs = [
        ("D1", "Doc 1", [{"title": "S1", "passages": ["Alpha beta gamma delta quantum.", "Epsilon zeta eta theta."]}]),
        ("D2", "Doc 2", [{"title": "S2", "passages": ["Unlearning target with unique keyword xylophone.", "Second leaf in D2."]}]),
        ("D3", "Doc 3", [{"title": "S3", "passages": ["Wavelet haar parseval telescoping.", "Banach contraction equilibrium."]}]),
    ]

    tree_unlearned = WaveTreeMemory(r=8, dim=64, seed=42)
    for doc_id, title, secs in docs:
        tree_unlearned.ingest_document(doc_id=doc_id, title=title, sections=secs)

    unlearn_res = tree_unlearned.unlearn_document("D2")
    assert unlearn_res["certified_exact"] is True

    # Clean shadow tree built without D2
    tree_clean = WaveTreeMemory(r=8, dim=64, seed=42)
    for doc_id, title, secs in docs:
        if doc_id != "D2":
            tree_clean.ingest_document(doc_id=doc_id, title=title, sections=secs)

    # Bit-exact equality (==) on corpus_df, root.term_count, root.kappa_mass_fp
    assert tree_unlearned.corpus_df == tree_clean.corpus_df
    assert tree_unlearned.root.term_count == tree_clean.root.term_count
    assert tree_unlearned.root.kappa_mass_fp == tree_clean.root.kappa_mass_fp

    # Bit-exact equality on every remaining node in the tree
    for d_id in ("D1", "D3"):
        n_unlearned = tree_unlearned.documents[d_id].get_subtree_nodes()
        n_clean = tree_clean.documents[d_id].get_subtree_nodes()
        assert len(n_unlearned) == len(n_clean)
        for nu, nc in zip(n_unlearned, n_clean):
            assert nu.term_count == nc.term_count
            assert nu.kappa_mass_fp == nc.kappa_mass_fp

    theorems = tree_unlearned.verify_all_theorems()
    assert theorems["all_theorems_verified"] is True

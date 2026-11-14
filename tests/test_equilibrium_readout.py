"""
Unit Tests for WaveCMS Equilibrium Multi-Hop Readout & TreeSoftmax Router (Mũi 2)
Verifies:
1. TreeSoftmax leaf probability normalization (\\sum \\alpha_p = 1.0).
2. Theorem 1(b) Wavelet Consolidation: \\sum \\alpha_p \\Theta_p == \\Theta_{root} + \\sum_{v \\ne root} p_v D_v.
3. Theorem 1(c) Haar detail cancellation under uninformative routing.
4. Contraction factor \\hat{\\rho} calculation & contractive flag audit.
5. Banach fixed-point iteration convergence under contraction.
6. Automatic resolution scaling with Shannon entropy.
7. Resilience to string / tensor queries and duck-typed nodes.

100% Offline, Pure Python / PyTorch.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.wave_cms.equilibrium_readout import (
    EquilibriumMultiHopReadout,
    EquilibriumResult,
    ResolutionLevel,
    RoutingResult,
    TreeSoftmaxRouter,
    _deterministic_text_embedding,
)


@dataclass
class MockDocumentTreeNode:
    """Mock document tree node satisfying DocumentTreeNode specification."""
    node_id: str
    level: str  # CORPUS, DOCUMENT, SECTION, LEAF_PARAGRAPH
    theta: torch.Tensor
    detail: torch.Tensor
    kappa: torch.Tensor  # routing key
    weight: float = 1.0
    children: List["MockDocumentTreeNode"] = field(default_factory=list)
    parent: Optional["MockDocumentTreeNode"] = None


def build_synthetic_wave_tree(r: int = 8, dim: int = 64) -> MockDocumentTreeNode:
    """
    Builds a synthetic 3-level tree (Root -> 2 Docs -> 4 Sections / Leaves)
    with exact telescoping and Haar properties:
    - Leaves: L1, L2 under Doc1; L3, L4 under Doc2
    - Details: D_v = \\Theta_v - \\Theta_{par(v)}
    - Haar property: \\sum_{c \\in ch(u)} W_c D_c = 0
    """
    torch.manual_seed(42)

    # 4 Leaves
    th_l1 = torch.randn(r, r) * 0.5
    th_l2 = torch.randn(r, r) * 0.5
    th_l3 = torch.randn(r, r) * 0.5
    th_l4 = torch.randn(r, r) * 0.5

    # Doc1 = (L1 + L2) / 2
    th_d1 = 0.5 * (th_l1 + th_l2)
    # Doc2 = (L3 + L4) / 2
    th_d2 = 0.5 * (th_l3 + th_l4)

    # Root = (Doc1 + Doc2) / 2
    th_root = 0.5 * (th_d1 + th_d2)

    # Wavelet Details D_v = \\Theta_v - \\Theta_{parent}
    d_d1 = th_d1 - th_root
    d_d2 = th_d2 - th_root
    d_l1 = th_l1 - th_d1
    d_l2 = th_l2 - th_d1
    d_l3 = th_l3 - th_d2
    d_l4 = th_l4 - th_d2

    # Keys (normalized)
    k_root = F.normalize(torch.randn(dim), p=2, dim=0)
    k_d1 = F.normalize(torch.randn(dim), p=2, dim=0)
    k_d2 = F.normalize(torch.randn(dim), p=2, dim=0)
    k_l1 = F.normalize(torch.randn(dim), p=2, dim=0)
    k_l2 = F.normalize(torch.randn(dim), p=2, dim=0)
    k_l3 = F.normalize(torch.randn(dim), p=2, dim=0)
    k_l4 = F.normalize(torch.randn(dim), p=2, dim=0)

    l1 = MockDocumentTreeNode("leaf_1", "LEAF_PARAGRAPH", th_l1, d_l1, k_l1, weight=1.0)
    l2 = MockDocumentTreeNode("leaf_2", "LEAF_PARAGRAPH", th_l2, d_l2, k_l2, weight=1.0)
    l3 = MockDocumentTreeNode("leaf_3", "LEAF_PARAGRAPH", th_l3, d_l3, k_l3, weight=1.0)
    l4 = MockDocumentTreeNode("leaf_4", "LEAF_PARAGRAPH", th_l4, d_l4, k_l4, weight=1.0)

    d1 = MockDocumentTreeNode("doc_1", "DOCUMENT", th_d1, d_d1, k_d1, weight=2.0, children=[l1, l2])
    d2 = MockDocumentTreeNode("doc_2", "DOCUMENT", th_d2, d_d2, k_d2, weight=2.0, children=[l3, l4])
    l1.parent = d1
    l2.parent = d1
    l3.parent = d2
    l4.parent = d2

    root = MockDocumentTreeNode("root", "CORPUS", th_root, torch.zeros(r, r), k_root, weight=4.0, children=[d1, d2])
    d1.parent = root
    d2.parent = root

    return root


def test_tree_softmax_leaf_normalization():
    """Verify that hierarchical TreeSoftmax produces a normalized leaf distribution (\\sum \\alpha_p = 1.0)."""
    dim, r = 32, 8
    root = build_synthetic_wave_tree(r=r, dim=dim)
    router = TreeSoftmaxRouter(dim=dim, tau=1.0)

    h_query = torch.randn(dim)
    routing_res = router.route(h_query, root)

    assert isinstance(routing_res, RoutingResult)
    leaf_weights = routing_res.leaf_weights

    # Must contain all 4 leaves
    assert len(leaf_weights) == 4
    assert set(leaf_weights.keys()) == {"leaf_1", "leaf_2", "leaf_3", "leaf_4"}

    # Sum must equal 1.0
    total_alpha = sum(leaf_weights.values())
    assert abs(total_alpha - 1.0) < 1e-6

    # Node probabilities must be hierarchical products
    p_d1 = routing_res.node_probabilities["doc_1"]
    p_d2 = routing_res.node_probabilities["doc_2"]
    assert abs((p_d1 + p_d2) - 1.0) < 1e-6

    # Leaf probabilities sum to their document parent
    assert abs((leaf_weights["leaf_1"] + leaf_weights["leaf_2"]) - p_d1) < 1e-6
    assert abs((leaf_weights["leaf_3"] + leaf_weights["leaf_4"]) - p_d2) < 1e-6

    # Dict access compatibility
    assert routing_res["alpha"] == leaf_weights
    assert routing_res["entropy"] >= 0.0


def test_theorem_1b_wavelet_telescoping_consolidation():
    """
    Mathematical Proof Verification of Theorem 1(b):
    \\sum_{p \\in \\mathcal{L}} \\alpha_p \\Theta_p == \\Theta_{root} + \\sum_{v \\ne root} p_v D_v
    """
    dim, r = 32, 8
    root = build_synthetic_wave_tree(r=r, dim=dim)
    router = TreeSoftmaxRouter(dim=dim, tau=1.5)

    h_query = torch.randn(dim)
    routing_res = router.route(h_query, root)

    # LHS: Direct leaf combination \\sum \\alpha_p \\Theta_p
    leaves = router._collect_leaves(root)
    theta_lhs = torch.zeros(r, r)
    for lf in leaves:
        alpha_p = routing_res.leaf_weights[lf.node_id]
        theta_lhs += alpha_p * lf.theta

    # RHS: Multiresolution wavelet consolidation \\Theta_{root} + \\sum p_v D_v
    theta_rhs = router.synthesize_memory_core(routing_res, root)

    # Discrepancy must be zero up to numerical float precision (< 1e-6)
    diff = torch.norm(theta_lhs - theta_rhs, p="fro").item()
    assert diff < 1e-5, f"Theorem 1(b) violation: ||LHS - RHS||_F = {diff}"


def test_theorem_1c_haar_cancellation_under_uninformative_routing():
    """
    Verifies Theorem 1(c):
    When routing is uninformative (equal weights), the Haar detail terms
    \\sum_{c \\in ch(u)} W_c D_c = 0 cancel out, and \\Theta(\\alpha) == \\Theta_{root}.
    """
    dim, r = 32, 8
    root = build_synthetic_wave_tree(r=r, dim=dim)

    # Under uniform branching: p(d1)=p(d2)=0.5, p(l1)=p(l2)=p(l3)=p(l4)=0.25
    uniform_node_probs = {
        "root": 1.0,
        "doc_1": 0.5,
        "doc_2": 0.5,
        "leaf_1": 0.25,
        "leaf_2": 0.25,
        "leaf_3": 0.25,
        "leaf_4": 0.25,
    }
    uniform_leaf_weights = {
        "leaf_1": 0.25,
        "leaf_2": 0.25,
        "leaf_3": 0.25,
        "leaf_4": 0.25,
    }

    mock_res = RoutingResult(
        leaf_weights=uniform_leaf_weights,
        node_probabilities=uniform_node_probs,
        entropy=math.log(4),
        normalized_entropy=1.0,
        effective_k=4.0,
        resolution_level="CORPUS",
        is_diffuse=True,
    )

    router = TreeSoftmaxRouter(dim=dim)
    theta_synth = router.synthesize_memory_core(mock_res, root)

    # Compare with root theta
    diff_from_root = torch.norm(theta_synth - root.theta, p="fro").item()
    assert diff_from_root < 1e-6, f"Theorem 1(c) Haar cancellation failed: diff = {diff_from_root}"


def test_contraction_factor_computation_and_warning():
    """Verifies that \\hat{\\rho} is correctly estimated and contractive flag functions as expected."""
    dim, r = 32, 8
    root = build_synthetic_wave_tree(r=r, dim=dim)
    readout = EquilibriumMultiHopReadout(dim=dim, r=r, omega=0.5)

    # 1. High temperature -> small rho_hat < 1.0 (Contractive)
    rho_contractive, is_contractive = readout.compute_contraction_factor(root, tau=10.0, l_e=0.1)
    assert rho_contractive < 1.0
    assert is_contractive is True

    # 2. Extremely low temperature -> large rho_hat >= 1.0 (Non-contractive)
    rho_non_contractive, is_not_contractive = readout.compute_contraction_factor(root, tau=0.0001, l_e=10.0)
    assert rho_non_contractive >= 1.0
    assert is_not_contractive is False


def test_banach_fixed_point_monotonic_convergence():
    """
    Verifies Theorem 4:
    Under contraction (\\hat{\\rho} < 1), the Banach fixed-point iteration
    converges monotonically to a unique equilibrium latent state h*.
    """
    dim, r = 32, 8
    root = build_synthetic_wave_tree(r=r, dim=dim)

    # Use large tau to ensure contraction
    router = TreeSoftmaxRouter(dim=dim, tau=10.0)
    readout = EquilibriumMultiHopReadout(dim=dim, r=r, omega=0.6, max_hops=15, tol=1e-3, router=router)

    q = torch.randn(dim)
    res = readout.readout(q, root)

    assert isinstance(res, EquilibriumResult)
    assert res.num_hops >= 2
    assert len(res.trajectory_deltas) == res.num_hops

    # Verify deltas are strictly decreasing (monotonic Banach Cauchy contraction)
    for i in range(len(res.trajectory_deltas) - 1):
        assert res.trajectory_deltas[i + 1] <= res.trajectory_deltas[i] + 1e-6

    # Verify convergence achieved
    assert res.converged is True
    assert res.trajectory_deltas[-1] < 1e-3

    # Dict access compatibility for WaveCMSEngine
    assert res["h_star"].shape == (dim,)
    assert res["theta_star"].shape == (r, r)
    assert isinstance(res["hops_history"], list)
    assert "rho_hat" in res
    assert res["is_contractive"] is True


def test_string_query_deterministic_embedding():
    """Verifies that natural language string queries are deterministically embedded and executed."""
    dim, r = 32, 8
    root = build_synthetic_wave_tree(r=r, dim=dim)
    readout = EquilibriumMultiHopReadout(dim=dim, r=r, max_hops=4)

    # 1. Deterministic embedding check
    emb1 = _deterministic_text_embedding("Mã hóa đối xứng là gì?", dim)
    emb2 = _deterministic_text_embedding("Mã hóa đối xứng là gì?", dim)
    assert torch.allclose(emb1, emb2)
    assert abs(torch.norm(emb1).item() - 1.0) < 1e-5

    # 2. String query execution
    res = readout.readout("Mã hóa dòng và hệ mã Autokey hoạt động ra sao?", root)
    assert res.fixed_point.shape == (dim,)
    assert res.final_theta is not None
    assert res.final_theta.shape == (r, r)
    assert len(res.history_alpha) == res.num_hops


def test_entropy_and_resolution_level_scaling():
    """Verifies that high entropy triggers CORPUS/DOCUMENT resolution while low entropy targets LEAF."""
    dim, r = 32, 8
    root = build_synthetic_wave_tree(r=r, dim=dim)
    router = TreeSoftmaxRouter(dim=dim, tau=0.01)  # Low temp -> very sharp

    # Target specific leaf direction
    leaf1_key = root.children[0].children[0].kappa
    h_focused = leaf1_key.clone()
    focused_res = router.route(h_focused, root)

    # Sharp routing should have low normalized entropy and target LEAF_PARAGRAPH
    assert focused_res.normalized_entropy < 0.6
    assert focused_res.resolution_level in (ResolutionLevel.LEAF_PARAGRAPH.value, ResolutionLevel.SECTION.value)

    # High temp -> diffuse
    router_diffuse = TreeSoftmaxRouter(dim=dim, tau=50.0)
    h_diffuse = torch.zeros(dim)
    diffuse_res = router_diffuse.route(h_diffuse, root)
    assert diffuse_res.normalized_entropy > 0.8
    assert diffuse_res.resolution_level == ResolutionLevel.CORPUS.value
    assert diffuse_res.is_diffuse is True

"""
Unit Tests for WaveCMS Causal Attribution & Conformal Risk Control.
Module: src/wave_cms/causal_attribution.py

Tests:
1. Counterfactual Leaf Attribution (Leave-One-Leaf-Out c(p)):
   - Monotonicity & Sensitivity: Ablating an influential leaf causes drop in log-likelihood (c(p) > 0).
   - Inactive Leaf Invariance: Ablating a leaf with zero routing weight causes c(p) approx 0.
   - Counterfactual Alpha Normalization: Sum_{p' != p} alpha_{-p}(p') == 1.0.
   - Evidence Chain Extraction: Multi-hop chain preserves top causal leaves across hops.
2. Conformal Risk Control Refusal:
   - Calibration Guarantee: Empirically checks that (n / (n+1)) * R_hat(lambda) + 1 / (n+1) <= epsilon.
   - Selective Classification: Refuses low-confidence / out-of-domain inputs while accepting grounded ones.
   - Exchangeability Specification: Ensures decision metadata specifies statistical risk bound.

100% Strict Offline, Zero External APIs, Pure Python & NumPy/PyTorch.
"""

import numpy as np
import pytest
import torch

from src.wave_cms.tree_memory import (
    DocumentTreeNode,
    TreeLevel,
    WaveTreeMemory,
)
from src.wave_cms.causal_attribution import (
    CausalAttributionItem,
    CounterfactualLeafAttributor,
    EvidenceChainHop,
    ConformalRiskController,
    RefusalDecision,
)


@pytest.fixture
def sample_tree_memory() -> WaveTreeMemory:
    """Builds a small two-document WaveCMS tree with known section and leaf structure."""
    mem = WaveTreeMemory(r=8, dim=64)
    mem.ingest_document(
        doc_id="doc_crypto",
        title="Mật mã học đối xứng",
        sections=[
            {
                "title": "Mã hóa dòng",
                "paragraphs": [
                    "Hệ mã hóa dòng sinh ra chuỗi khóa z và thực hiện phép xor ci = pi ^ zi.",
                    "Hệ mã hóa độn một lần OTP là trường hợp an ninh tuyệt đối của mã hóa dòng.",
                ],
            },
            {
                "title": "Mã hóa khối",
                "paragraphs": [
                    "AES là chuẩn mã hóa khối đối xứng với kích thước khối 128 bit.",
                    "DES sử dụng cấu trúc mạng Feistel với 16 vòng lặp.",
                ],
            },
        ],
    )
    mem.ingest_document(
        doc_id="doc_database",
        title="Cơ sở dữ liệu quan hệ",
        sections=[
            {
                "title": "Giao dịch ACID",
                "paragraphs": [
                    "Tính nguyên tử Atomicity đảm bảo giao dịch thực hiện trọn vẹn hoặc rollback.",
                    "Tính cô lập Isolation ngăn chặn các hiện tượng dirty read và phantom read.",
                ],
            }
        ],
    )
    return mem


class TestCounterfactualLeafAttributor:
    def test_counterfactual_alpha_renormalization(self):
        """Checks that alpha_{-p} zeros the ablated leaf and renormalizes remaining leaves to 1.0."""
        attributor = CounterfactualLeafAttributor(d_model=64, r=8)
        alpha = {"leaf_1": 0.5, "leaf_2": 0.3, "leaf_3": 0.2}

        # Ablate leaf_1
        cf_1 = attributor.compute_counterfactual_alpha(alpha, "leaf_1")
        assert cf_1["leaf_1"] == 0.0
        assert pytest.approx(sum(cf_1.values()), abs=1e-9) == 1.0
        assert pytest.approx(cf_1["leaf_2"], abs=1e-9) == 0.3 / 0.5
        assert pytest.approx(cf_1["leaf_3"], abs=1e-9) == 0.2 / 0.5

        # Ablate leaf with 100% mass -> fallback uniform over remaining
        alpha_single = {"leaf_1": 1.0, "leaf_2": 0.0, "leaf_3": 0.0}
        cf_single = attributor.compute_counterfactual_alpha(alpha_single, "leaf_1")
        assert cf_single["leaf_1"] == 0.0
        assert pytest.approx(sum(cf_single.values()), abs=1e-9) == 1.0
        assert pytest.approx(cf_single["leaf_2"], abs=1e-9) == 0.5
        assert pytest.approx(cf_single["leaf_3"], abs=1e-9) == 0.5

    def test_inactive_leaf_invariance(self, sample_tree_memory):
        """A leaf with zero routing weight produces c(p) == 0.0 when ablated."""
        attributor = CounterfactualLeafAttributor(d_model=64, r=8)
        leaves = sample_tree_memory.get_leaves()
        assert len(leaves) >= 4

        # Set alpha focusing purely on leaf 0 and leaf 1; leaf 2 has 0 weight
        alpha = {leaves[0].node_id: 0.6, leaves[1].node_id: 0.4}
        for l in leaves[2:]:
            alpha[l.node_id] = 0.0

        query = "Nguyên lý mã hóa dòng và chuỗi khóa?"
        answer = "Mã hóa dòng dùng chuỗi khóa zi và phép toán xor."

        results = attributor.attribute(
            query=query,
            answer=answer,
            tree_or_leaves=sample_tree_memory,
            alpha=alpha,
            top_m=len(leaves),
        )

        res_map = {r.leaf_id: r for r in results}
        # Inactive leaf 2 must have c(p) == 0.0
        assert leaves[2].node_id in res_map
        assert pytest.approx(res_map[leaves[2].node_id].causal_score, abs=1e-9) == 0.0

    def test_influential_leaf_positive_causal_contribution(self, sample_tree_memory):
        """A leaf that strongly supports the answer yields a positive c(p) > 0."""
        attributor = CounterfactualLeafAttributor(d_model=64, r=8)
        leaves = sample_tree_memory.get_leaves()

        # Find leaf discussing AES
        aes_leaf = [l for l in leaves if "AES" in l.text][0]
        other_leaf = [l for l in leaves if l.node_id != aes_leaf.node_id][0]

        # Put dominant weight on aes_leaf
        alpha = {aes_leaf.node_id: 0.85, other_leaf.node_id: 0.15}

        query = "Chuẩn mã hóa AES sử dụng khối bao nhiêu bit?"
        answer = "AES sử dụng kích thước khối 128 bit."

        results = attributor.attribute(
            query=query,
            answer=answer,
            tree_or_leaves=[aes_leaf, other_leaf],
            alpha=alpha,
            top_m=2,
        )

        assert len(results) == 2
        top_item = results[0]
        assert top_item.leaf_id == aes_leaf.node_id
        assert top_item.rank == 1
        assert top_item.causal_score > 0.0

    def test_multi_hop_evidence_chain_tracing(self, sample_tree_memory):
        """Traces Multi-Hop Evidence Chain across simulated hops."""
        attributor = CounterfactualLeafAttributor(d_model=64, r=8)
        leaves = sample_tree_memory.get_leaves()

        # Hop 1 focuses on OTP, Hop 2 focuses on ACID
        otp_leaf = [l for l in leaves if "OTP" in l.text][0]
        acid_leaf = [l for l in leaves if "Atomicity" in l.text][0]

        hop1_alpha = {l.node_id: 0.05 for l in leaves}
        hop1_alpha[otp_leaf.node_id] = 0.85

        hop2_alpha = {l.node_id: 0.05 for l in leaves}
        hop2_alpha[acid_leaf.node_id] = 0.85

        chain = attributor.trace_evidence_chain(
            tree_or_leaves=sample_tree_memory,
            hops_alpha=[hop1_alpha, hop2_alpha],
            query="So sánh an ninh OTP và tính nguyên tử",
            answer="OTP đạt an ninh tuyệt đối còn Atomicity đảm bảo giao dịch nguyên tử",
        )

        assert len(chain) == 2
        assert chain[0].hop_index == 1
        assert chain[0].top_leaf_id == otp_leaf.node_id
        assert chain[1].hop_index == 2
        assert chain[1].top_leaf_id == acid_leaf.node_id


class TestConformalRiskController:
    def test_composite_confidence_monotonicity(self):
        """Confidence score monotonically increases with both routing mass and memory reliance."""
        controller = ConformalRiskController()

        s_low = controller.compute_composite_confidence(max_alpha=0.2, memory_reliance=-2.0)
        s_mid = controller.compute_composite_confidence(max_alpha=0.6, memory_reliance=0.0)
        s_high = controller.compute_composite_confidence(max_alpha=0.95, memory_reliance=3.5)

        assert 0.0 <= s_low < s_mid < s_high <= 1.0

    def test_conformal_risk_control_calibration_guarantee(self):
        """
        Validates the finite-sample Conformal Risk Control guarantee (Bates et al. 2021):
        (n / (n+1)) * R_hat(lambda) + 1 / (n+1) <= epsilon.
        """
        controller = ConformalRiskController(epsilon=0.15)
        rng = np.random.RandomState(123)

        # Generate synthetic calibration set of n=100 examples
        cal_data = []
        for _ in range(100):
            score = float(rng.uniform(0.1, 0.99))
            # Probability of being correct increases with confidence score
            prob_correct = 0.1 + 0.85 * (score ** 1.5)
            is_correct = bool(rng.rand() < prob_correct)
            cal_data.append({"score": score, "is_correct": is_correct})

        cal_res = controller.calibrate(cal_data, epsilon=0.15)
        assert cal_res["status"] == "SUCCESS"
        assert controller.is_calibrated is True

        n = len(cal_data)
        emp_risk = controller.empirical_risk_at_threshold
        crc_bound = (n / (n + 1.0)) * emp_risk + (1.0 / (n + 1.0))
        assert crc_bound <= 0.15 + 1e-6

    def test_selective_refusal_on_low_confidence(self):
        """Low routing probability or negative reliance triggers certified refusal."""
        controller = ConformalRiskController(epsilon=0.10, default_threshold=0.55)

        # Out-of-domain / diffuse routing
        dec_refuse = controller.evaluate(max_alpha=0.12, memory_reliance=-1.5)
        assert dec_refuse.refused is True
        assert dec_refuse.confidence < dec_refuse.threshold
        assert "diffuse" in dec_refuse.reason.lower() or "below" in dec_refuse.reason.lower()

        # High confidence grounded input
        dec_accept = controller.evaluate(max_alpha=0.88, memory_reliance=2.5)
        assert dec_accept.refused is False
        assert dec_accept.confidence >= dec_accept.threshold

    def test_exchangeability_statement_in_decision(self):
        """Verifies that decision metadata explicitly mentions exchangeability."""
        controller = ConformalRiskController(epsilon=0.10)
        controller.calibrate([{"score": 0.8, "is_correct": True}, {"score": 0.2, "is_correct": False}] * 20)

        decision = controller.evaluate(max_alpha=0.75, memory_reliance=1.2)
        assert "exchangeability" in decision.risk_guarantee.lower()
        d_dict = decision.to_dict()
        assert "refused" in d_dict
        assert "confidence" in d_dict
        assert "threshold" in d_dict

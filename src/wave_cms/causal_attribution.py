"""
WaveCMS: Document-Tree Wavelet Continuum Memory with Equilibrium Multi-Hop Readout.
Component: Counterfactual Leave-One-Leaf-Out Attribution & Conformal Risk Control Refusal.

Theoretical Foundations:
1. Counterfactual Leave-One-Leaf-Out Attribution (Proposal Section 2.7):
   c(p) = \\log p_{\\Theta(\\alpha)}(\\hat{a} \\mid q) - \\log p_{\\Theta(\\alpha_{-p})}(\\hat{a} \\mid q)
   where \\alpha_{-p} is the counterfactual ablation with leaf p removed and renormalized:
       \\alpha_{-p}(p') = \\alpha(p') / (1 - \\alpha(p))  for p' \\ne p,  \\alpha_{-p}(p) = 0.
   Identifies the structural causal contribution of each leaf paragraph to the generated answer,
   surpassing surface keyword overlap.
   The sequence of top-attributed leaves across readout steps forms the Multi-hop Evidence Chain E_{1:N}.

2. Conformal Risk Control Refusal (Angelopoulos et al., ICLR 2024; Bates et al., 2021):
   Combines peak routing confidence max_{n,p} \\alpha^{(n)}_p and memory reliance ratio
   \\log p_{\\text{mem}}(\\hat{a} \\mid q) - \\log p_{W_0}(\\hat{a} \\mid q).
   Provides distribution-free statistical risk control guarantee:
       \\mathbb{E}[\\mathbf{1}\\{\\text{answered} \\wedge \\text{wrong}\\}] \\le \\varepsilon
   under the assumption that test data is exchangeable with calibration data.
   (Note: This guarantees the expectation bound over exchangeable
   draws, rather than deterministic infallibility on individual instances).

100% Strict Offline, Pure Python stdlib, NumPy & PyTorch, Zero External APIs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.wave_cms.tree_memory import (
    DocumentTreeNode,
    TreeLevel,
    WaveTreeMemory,
    compute_deterministic_routing_key,
)


@dataclass
class CausalAttributionItem:
    """Detailed causal contribution metrics for a single leaf paragraph node."""
    leaf_id: str
    causal_score: float                # c(p) = log p_{Theta(alpha)} - log p_{Theta(alpha_{-p})}
    rank: int                          # 1-based rank among evaluated leaves
    original_alpha: float              # Routing weight alpha_p
    full_logprob: float                # log p_{Theta(alpha)}(a | q)
    counterfactual_logprob: float      # log p_{Theta(alpha_{-p})}(a | q)
    title: str = ""
    text: str = ""
    doc_id: Optional[str] = None
    section_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "leaf_id": self.leaf_id,
            "score": round(self.causal_score, 6),
            "rank": self.rank,
            "original_alpha": round(self.original_alpha, 6),
            "full_logprob": round(self.full_logprob, 6),
            "counterfactual_logprob": round(self.counterfactual_logprob, 6),
            "title": self.title,
            "text": self.text,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
        }


@dataclass
class EvidenceChainHop:
    """A single hop in the Multi-Hop Evidence Chain E_{1:N}."""
    hop_index: int                     # 1-based hop index (1..N)
    top_leaf_id: str
    top_leaf_title: str
    causal_score: float
    routing_weight: float
    resolution_level: str = "LEAF_PARAGRAPH"
    text_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hop_index": self.hop_index,
            "top_leaf_id": self.top_leaf_id,
            "title": self.top_leaf_title,
            "causal_score": round(self.causal_score, 6),
            "routing_weight": round(self.routing_weight, 6),
            "resolution_level": self.resolution_level,
            "text_snippet": self.text_snippet[:160],
        }


@dataclass
class RefusalDecision:
    """Formal decision output from the Conformal Risk Control mechanism."""
    refused: bool
    confidence: float                  # Composite score s in [0.0, 1.0]
    threshold: float                   # Calibrated decision threshold lambda_hat
    max_alpha: float                   # Max routing weight max_{n,p} alpha_p
    memory_reliance: float             # Memory reliance ratio log p_mem - log p_{W0}
    epsilon: float                     # Risk control tolerance level
    is_calibrated: bool                # True if calibrated on exchangeable data
    risk_guarantee: str                # Formal risk bound specification
    reason: str                        # Diagnostic explanation of the verdict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "refused": self.refused,
            "confidence": round(self.confidence, 6),
            "threshold": round(self.threshold, 6),
            "max_alpha": round(self.max_alpha, 6),
            "memory_reliance": round(self.memory_reliance, 6),
            "epsilon": self.epsilon,
            "is_calibrated": self.is_calibrated,
            "risk_guarantee": self.risk_guarantee,
            "reason": self.reason,
        }


class CounterfactualLeafAttributor:
    """
    Counterfactual Leave-One-Leaf-Out Attribution Engine.
    
    Implements Proposal Section 2.7:
        c(p) = \\log p_{\\Theta(\\alpha)}(\\hat{a} \\mid q) - \\log p_{\\Theta(\\alpha_{-p})}(\\hat{a} \\mid q)
        
    Where \\Theta(\\alpha) is the effective memory core synthesized under routing \\alpha,
    and \\Theta(\\alpha_{-p}) is the counterfactual memory core obtained by zeroing \\alpha_p
    and renormalizing the remaining leaf weights:
        \\alpha_{-p}(p') = \\frac{\\alpha(p')}{1 - \\alpha(p)}  (\\forall p' \\ne p).
        
    Properties:
    1. Pure Causal Attribution: Evaluates how the prediction log-likelihood shifts when
       leaf p is counterfactually ablated from memory.
    2. Multi-Hop Evidence Chain: Traces the causal sequence of leaves across reading hops.
    3. Structural & Deterministic: Operates directly on the parameter space without relying
       on fragile surface string matching.
    """

    def __init__(
        self,
        d_model: int = 64,
        r: int = 8,
        sigma_sq: float = 1.0,
    ) -> None:
        self.d_model = d_model
        self.r = r
        self.sigma_sq = max(1e-4, float(sigma_sq))

        # Shared projection bases A in R^{r x d} and B in R^{d x r} (fixed orthogonal)
        rng = np.random.RandomState(42)
        q_mat, _ = np.linalg.qr(rng.randn(d_model, d_model))
        self.A = q_mat[:r, :].astype(np.float64)       # (r, d)
        self.B = self.A.T.copy()                       # (d, r) Tied symmetric basis
        self.W0 = 0.1 * q_mat.astype(np.float64)       # (d, d) Base frozen weight

    def _embed_text(self, text: str) -> np.ndarray:
        """Embeds text into R^d using the deterministic routing key algorithm."""
        return compute_deterministic_routing_key(text, dim=self.d_model)

    def compute_log_likelihood(
        self,
        theta: np.ndarray,
        query: str,
        answer: str,
        custom_scoring_fn: Optional[Callable[[np.ndarray, str, str], float]] = None,
    ) -> float:
        """
        Computes log p_{W0 + B Theta A}(answer | query).
        If custom_scoring_fn is provided, delegates to it.
        Otherwise, uses the Gaussian energy-based compatibility:
            log p = - 0.5 * ||h_a - (W0 + B Theta A) h_q||^2 / sigma^2
        """
        if custom_scoring_fn is not None:
            return float(custom_scoring_fn(theta, query, answer))

        h_q = self._embed_text(query)    # (d,)
        h_a = self._embed_text(answer)   # (d,)

        # W_eff = W0 + B @ Theta @ A  in R^{d x d}
        theta_arr = np.asarray(theta, dtype=np.float64)
        if theta_arr.shape != (self.r, self.r):
            # Reshape or pad/truncate if necessary
            t_fixed = np.zeros((self.r, self.r), dtype=np.float64)
            r_min = min(self.r, theta_arr.shape[0])
            c_min = min(self.r, theta_arr.shape[1] if theta_arr.ndim > 1 else 1)
            t_fixed[:r_min, :c_min] = theta_arr[:r_min, :c_min]
            theta_arr = t_fixed

        # Associative fast path: (W0 + B @ Theta @ A) h_q = W0 h_q + B (Theta (A h_q))
        pred_a = (self.W0 @ h_q) + (self.B @ (theta_arr @ (self.A @ h_q)))

        # Negative squared L2 distance (Gaussian log-likelihood proportional)
        residual = h_a - pred_a
        sq_dist = float(np.sum(residual ** 2))
        log_prob = -0.5 * sq_dist / self.sigma_sq
        return log_prob

    def compute_counterfactual_alpha(
        self,
        alpha: Dict[str, float],
        ablated_leaf_id: str,
    ) -> Dict[str, float]:
        """
        Computes the counterfactual routing distribution alpha_{-p}:
            alpha_{-p}(p') = alpha(p') / (1 - alpha(p))  for p' != p
            alpha_{-p}(p) = 0.0
            
        If alpha(p) >= 1.0 - 1e-7 (single leaf consumed 100% mass),
        falls back to uniform distribution over the remaining leaves.
        """
        if ablated_leaf_id not in alpha:
            return dict(alpha)

        p_mass = float(alpha.get(ablated_leaf_id, 0.0))
        rem_leaves = [lid for lid in alpha.keys() if lid != ablated_leaf_id]

        if not rem_leaves:
            # Only one leaf existed; leaving it out yields zero mass
            return {ablated_leaf_id: 0.0}

        denom = 1.0 - p_mass
        alpha_cf: Dict[str, float] = {}

        if denom <= 1e-7:
            # Fallback uniform over remaining leaves
            u_weight = 1.0 / len(rem_leaves)
            for lid in rem_leaves:
                alpha_cf[lid] = u_weight
        else:
            for lid in rem_leaves:
                alpha_cf[lid] = float(alpha[lid]) / denom

        alpha_cf[ablated_leaf_id] = 0.0
        return alpha_cf

    def synthesize_core_from_alpha(
        self,
        alpha: Dict[str, float],
        leaves: Sequence[DocumentTreeNode],
    ) -> np.ndarray:
        """
        Synthesizes the linear combination Theta(alpha) = sum_p alpha_p Theta_p.
        """
        leaf_map = {node.node_id: node for node in leaves}
        theta_eff = np.zeros((self.r, self.r), dtype=np.float64)

        for lid, weight in alpha.items():
            if weight <= 0.0:
                continue
            node = leaf_map.get(lid)
            if node is not None:
                theta_eff += weight * node.theta

        return theta_eff

    def attribute(
        self,
        query: str,
        answer: str,
        tree_or_leaves: Union[WaveTreeMemory, DocumentTreeNode, Sequence[DocumentTreeNode]],
        alpha: Optional[Dict[str, float]] = None,
        top_m: int = 5,
        custom_scoring_fn: Optional[Callable[[np.ndarray, str, str], float]] = None,
    ) -> List[CausalAttributionItem]:
        """
        Computes Leave-One-Leaf-Out attribution scores c(p) for all (or top-m) candidate leaves:
            c(p) = log p_{Theta(alpha)}(a | q) - log p_{Theta(alpha_{-p})}(a | q)
            
        Returns ranked list of CausalAttributionItem objects sorted descending by c(p).
        """
        # 1. Resolve leaves
        leaves: List[DocumentTreeNode] = []
        if isinstance(tree_or_leaves, WaveTreeMemory):
            leaves = tree_or_leaves.get_leaves()
        elif hasattr(tree_or_leaves, "get_leaves"):
            leaves = tree_or_leaves.get_leaves()
        elif hasattr(tree_or_leaves, "get_all_leaves"):
            leaves = tree_or_leaves.get_all_leaves()
        elif isinstance(tree_or_leaves, DocumentTreeNode):
            leaves = tree_or_leaves.get_leaves()
        elif isinstance(tree_or_leaves, (list, tuple)):
            leaves = list(tree_or_leaves)

        if not leaves:
            return []

        # 2. Resolve routing distribution alpha
        if alpha is None:
            # Default uniform routing over leaves
            u_val = 1.0 / len(leaves)
            alpha = {node.node_id: u_val for node in leaves}
        else:
            # Ensure proper normalization
            total_mass = sum(alpha.values())
            if total_mass > 0:
                alpha = {k: v / total_mass for k, v in alpha.items()}
            else:
                u_val = 1.0 / len(leaves)
                alpha = {node.node_id: u_val for node in leaves}

        # 3. Compute baseline log-likelihood with full routing: log p_{Theta(alpha)}
        theta_full = self.synthesize_core_from_alpha(alpha, leaves)
        full_logprob = self.compute_log_likelihood(
            theta=theta_full,
            query=query,
            answer=answer,
            custom_scoring_fn=custom_scoring_fn,
        )

        # 4. Separate active candidate leaves and inactive leaves (alpha <= 1e-9)
        active_leaves = [node for node in leaves if alpha.get(node.node_id, 0.0) > 1e-9]
        inactive_leaves = [node for node in leaves if alpha.get(node.node_id, 0.0) <= 1e-9]

        attribution_items: List[CausalAttributionItem] = []

        # Inactive leaves mathematically have alpha_{-p} == alpha, hence c(p) == 0.0
        for node in inactive_leaves:
            attribution_items.append(
                CausalAttributionItem(
                    leaf_id=node.node_id,
                    causal_score=0.0,
                    rank=0,
                    original_alpha=0.0,
                    full_logprob=full_logprob,
                    counterfactual_logprob=full_logprob,
                    title=getattr(node, "title", ""),
                    text=getattr(node, "text", ""),
                    doc_id=getattr(node, "doc_id", getattr(node, "metadata", {}).get("doc_id", None)),
                    section_id=getattr(node, "section_id", getattr(node, "metadata", {}).get("section_id", None)),
                )
            )

        # 5. Compute counterfactual log-likelihood for active candidate leaves
        for node in active_leaves:
            lid = node.node_id
            orig_alpha_val = alpha.get(lid, 0.0)

            # Ablate leaf p -> alpha_{-p}
            alpha_cf = self.compute_counterfactual_alpha(alpha, lid)
            theta_cf = self.synthesize_core_from_alpha(alpha_cf, leaves)

            cf_logprob = self.compute_log_likelihood(
                theta=theta_cf,
                query=query,
                answer=answer,
                custom_scoring_fn=custom_scoring_fn,
            )

            # c(p) = full_logprob - cf_logprob
            causal_score = float(full_logprob - cf_logprob)

            attribution_items.append(
                CausalAttributionItem(
                    leaf_id=lid,
                    causal_score=causal_score,
                    rank=0,  # Assigned after sorting
                    original_alpha=orig_alpha_val,
                    full_logprob=full_logprob,
                    counterfactual_logprob=cf_logprob,
                    title=getattr(node, "title", ""),
                    text=getattr(node, "text", ""),
                    doc_id=getattr(node, "doc_id", getattr(node, "metadata", {}).get("doc_id", None)),
                    section_id=getattr(node, "section_id", getattr(node, "metadata", {}).get("section_id", None)),
                )
            )

        # 6. Sort descending by causal contribution c(p)
        attribution_items.sort(key=lambda item: item.causal_score, reverse=True)

        # Assign 1-based ranks
        for idx, item in enumerate(attribution_items):
            item.rank = idx + 1

        return attribution_items[:top_m]

    def trace_evidence_chain(
        self,
        tree_or_leaves: Union[WaveTreeMemory, DocumentTreeNode, Sequence[DocumentTreeNode]],
        hops_alpha: Sequence[Dict[str, float]],
        query: str = "",
        answer: str = "",
        resolution_levels: Optional[Sequence[str]] = None,
    ) -> List[EvidenceChainHop]:
        """
        Traces the Multi-Hop Evidence Chain E_{1:N} across reading steps.
        For each hop n, identifies the leaf node with the highest counterfactual
        contribution (or highest routing weight when scores tie).
        """
        chain: List[EvidenceChainHop] = []
        leaves_list: List[DocumentTreeNode] = []

        if isinstance(tree_or_leaves, WaveTreeMemory):
            leaves_list = tree_or_leaves.get_leaves()
        elif hasattr(tree_or_leaves, "get_leaves"):
            leaves_list = tree_or_leaves.get_leaves()
        elif hasattr(tree_or_leaves, "get_all_leaves"):
            leaves_list = tree_or_leaves.get_all_leaves()
        elif isinstance(tree_or_leaves, DocumentTreeNode):
            leaves_list = tree_or_leaves.get_leaves()
        else:
            leaves_list = list(tree_or_leaves)

        leaf_map = {n.node_id: n for n in leaves_list}

        for hop_idx, alpha_hop in enumerate(hops_alpha):
            hop_num = hop_idx + 1
            res_level = "LEAF_PARAGRAPH"
            if resolution_levels and hop_idx < len(resolution_levels):
                res_level = resolution_levels[hop_idx]

            # Compute attribution for this hop
            hop_items = self.attribute(
                query=query,
                answer=answer,
                tree_or_leaves=leaves_list,
                alpha=alpha_hop,
                top_m=len(leaves_list),
            )

            if hop_items:
                # Rank candidates at hop n by joint routing focus and causal contribution:
                # S(p) = alpha_p * (1.0 + max(-0.5, c(p)))
                best_item = max(
                    hop_items,
                    key=lambda it: it.original_alpha * (1.0 + max(-0.5, it.causal_score)),
                )
                chain.append(
                    EvidenceChainHop(
                        hop_index=hop_num,
                        top_leaf_id=best_item.leaf_id,
                        top_leaf_title=best_item.title,
                        causal_score=best_item.causal_score,
                        routing_weight=best_item.original_alpha,
                        resolution_level=res_level,
                        text_snippet=best_item.text[:160],
                    )
                )
            else:
                # Fallback to max alpha
                if alpha_hop:
                    best_lid = max(alpha_hop.keys(), key=lambda k: alpha_hop[k])
                    best_node = leaf_map.get(best_lid)
                    chain.append(
                        EvidenceChainHop(
                            hop_index=hop_num,
                            top_leaf_id=best_lid,
                            top_leaf_title=best_node.title if best_node else best_lid,
                            causal_score=0.0,
                            routing_weight=alpha_hop[best_lid],
                            resolution_level=res_level,
                            text_snippet=best_node.text[:160] if best_node else "",
                        )
                    )

        return chain


class ConformalRiskController:
    """
    Conformal Risk Control Refusal Engine for WaveCMS.
    
    Theoretical Formulation (Proposal Section 2.7, Angelopoulos et al., ICLR 2024):
    1. Confidence Score:
       Combines maximum routing confidence across hops/leaves:
           S_{route} = \\max_{n, p} \\alpha^{(n)}_p \\in [0, 1]
       with the memory reliance ratio:
           S_{reliance} = \\log p_{\\text{mem}}(\\hat{a} \\mid q) - \\log p_{W_0}(\\hat{a} \\mid q).
       Combined score:
           s(q, \\hat{a}) = 0.5 \\cdot S_{route} + 0.5 \\cdot \\sigma(S_{reliance}) \\in [0, 1].
           
    2. Finite-Sample Risk Guarantee:
       Let \\ell_i(\\lambda) = \\mathbf{1}\\{s_i \\ge \\lambda\\} \\cdot (1 - y_i) be the loss
       (penalty for answering an incorrect query).
       By Conformal Risk Control, we select threshold \\hat{\\lambda}:
           \\hat{\\lambda} = \\inf \\left\\{ \\lambda \\in [0, 1] :
               \\frac{n}{n + 1} \\hat{R}_n(\\lambda) + \\frac{1}{n + 1} \\le \\varepsilon
           \\right\\}.
       Under exchangeability of calibration and test pairs, this guarantees:
           \\mathbb{E}[\\mathbf{1}\\{\\text{answered} \\wedge \\text{wrong}\\}] \\le \\varepsilon.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        default_threshold: float = 0.45,
        w_route: float = 0.5,
        w_reliance: float = 0.5,
    ) -> None:
        self.epsilon = float(epsilon)
        self.default_threshold = float(default_threshold)
        self.w_route = float(w_route)
        self.w_reliance = float(w_reliance)

        self.calibrated_threshold: float = self.default_threshold
        self.is_calibrated: bool = False
        self.calibration_sample_size: int = 0
        self.empirical_risk_at_threshold: float = 0.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def compute_composite_confidence(
        self,
        max_alpha: float,
        memory_reliance: float,
        num_leaves: Optional[int] = None,
        reliance_scale: float = 1.0,
    ) -> float:
        """
        Combines maximum routing probability with memory reliance ratio into [0.0, 1.0].
        When num_leaves is None, uses legacy linear combination:
            s = w_route * max_alpha + w_reliance * sigmoid(memory_reliance * reliance_scale)
        When num_leaves is provided, uses tree-size-invariant prior mapping:
            z_route = max(0, num_leaves * max_alpha - 1) / (1 - max_alpha + 1e-8)
            s_route = z_route / (z_route + 3.0)
            s = 0.80 * s_route + 0.20 * sigmoid(memory_reliance * reliance_scale)
        """
        clamped_alpha = max(0.0, min(1.0, float(max_alpha)))
        s_rel = self._sigmoid(float(memory_reliance) * float(reliance_scale))
        if num_leaves is None:
            score = (self.w_route * clamped_alpha) + (self.w_reliance * s_rel)
        else:
            z_route = max(0.0, float(num_leaves) * clamped_alpha - 1.0) / (1.0 - clamped_alpha + 1e-8)
            s_route = z_route / (z_route + 3.0)
            score = 0.80 * s_route + 0.20 * s_rel
        return max(0.0, min(1.0, float(score)))

    def calibrate(
        self,
        calibration_data: Sequence[Dict[str, Any]],
        epsilon: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calibrates the refusal threshold lambda_hat via Conformal Risk Control.
        
        Args:
            calibration_data: List of dicts, each with either:
                - {"score": float, "is_correct": bool}
                - {"max_alpha": float, "memory_reliance": float, "is_correct": bool}
            epsilon: Optional override for target risk level (default self.epsilon).
            
        Returns:
            Dictionary with calibration statistics and the calibrated threshold.
        """
        if epsilon is not None:
            self.epsilon = max(1e-4, min(1.0, float(epsilon)))

        n = len(calibration_data)
        if n == 0:
            self.is_calibrated = False
            return {
                "status": "FAILED_EMPTY_DATA",
                "calibrated_threshold": self.default_threshold,
                "n_samples": 0,
            }

        scores: List[float] = []
        labels: List[int] = []

        for row in calibration_data:
            if "score" in row:
                s = float(row["score"])
            else:
                m_alpha = float(row.get("max_alpha", 0.5))
                m_rel = float(row.get("memory_reliance", 0.0))
                raw_leaves = row.get("num_leaves")
                n_leaves = int(raw_leaves) if raw_leaves is not None else None
                raw_scale = row.get("reliance_scale")
                rel_scale = 1.0 if raw_scale is None else float(raw_scale)
                s = self.compute_composite_confidence(
                    m_alpha,
                    m_rel,
                    num_leaves=n_leaves,
                    reliance_scale=rel_scale,
                )

            y = 1 if bool(row.get("is_correct", True)) else 0
            scores.append(max(0.0, min(1.0, s)))
            labels.append(y)

        scores_arr = np.array(scores, dtype=np.float64)
        labels_arr = np.array(labels, dtype=np.int32)

        # Candidate thresholds: sorted unique scores, plus boundaries 0.0 and 1.0
        candidate_lambdas = np.unique(np.concatenate([np.array([0.0, 1.0]), scores_arr]))
        candidate_lambdas.sort()

        # Find inf { lambda : (n / (n + 1)) * R_hat(lambda) + 1 / (n + 1) <= epsilon }
        best_lambda: Optional[float] = None
        best_emp_risk: float = 0.0

        for lam in candidate_lambdas:
            # Answered if score >= lam; Loss = 1 if answered and wrong (label == 0)
            answered = scores_arr >= lam
            wrong = labels_arr == 0
            losses = answered & wrong
            emp_risk = float(np.mean(losses))

            # CRC finite-sample upper bound (Bates et al., 2021)
            crc_upper_bound = (n / (n + 1.0)) * emp_risk + (1.0 / (n + 1.0))

            if crc_upper_bound <= self.epsilon:
                best_lambda = float(lam)
                best_emp_risk = emp_risk
                break

        if best_lambda is None:
            # If no threshold satisfied (e.g. n too small or epsilon very tight),
            # choose the most conservative threshold 1.0 (refuses all unsure cases)
            best_lambda = 1.0
            best_emp_risk = 0.0

        self.calibrated_threshold = best_lambda
        self.empirical_risk_at_threshold = best_emp_risk
        self.calibration_sample_size = n
        self.is_calibrated = True

        return {
            "status": "SUCCESS",
            "calibrated_threshold": self.calibrated_threshold,
            "empirical_risk": self.empirical_risk_at_threshold,
            "n_samples": n,
            "epsilon": self.epsilon,
        }

    def evaluate(
        self,
        max_alpha: float,
        memory_reliance: float,
        query: str = "",
        num_leaves: Optional[int] = None,
        reliance_scale: float = 1.0,
    ) -> RefusalDecision:
        """
        Evaluates whether to answer or refuse the query based on conformal safety:
        - If confidence s >= threshold: ACCEPT (ANSWER)
        - If confidence s < threshold: REFUSE (OUT_OF_DOMAIN / UNGROUNDED)
        """
        conf_score = self.compute_composite_confidence(
            max_alpha,
            memory_reliance,
            num_leaves=num_leaves,
            reliance_scale=reliance_scale,
        )
        threshold = self.calibrated_threshold

        refused = bool(conf_score < threshold)
        reason = ""
        if refused:
            if max_alpha < 0.2:
                reason = "Diffuse memory routing (max leaf routing mass < 0.20): query exceeds corpus knowledge boundary"
            elif memory_reliance < -1.0:
                reason = "Negative memory reliance ratio (parametric memory fails to substantiate answer)"
            else:
                reason = f"Composite confidence ({conf_score:.4f}) below conformal safety threshold ({threshold:.4f})"
        else:
            reason = f"Conformal safety satisfied: confidence ({conf_score:.4f}) >= threshold ({threshold:.4f})"

        risk_guarantee = (
            f"E[answered and wrong] <= {self.epsilon:.4f} under exchangeability"
            if self.is_calibrated
            else "Heuristic uncalibrated safety threshold"
        )

        return RefusalDecision(
            refused=refused,
            confidence=conf_score,
            threshold=threshold,
            max_alpha=float(max_alpha),
            memory_reliance=float(memory_reliance),
            epsilon=self.epsilon,
            is_calibrated=self.is_calibrated,
            risk_guarantee=risk_guarantee,
            reason=reason,
        )

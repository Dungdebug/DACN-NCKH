"""
WaveCMS: Document-Tree Wavelet Continuum Memory with Equilibrium Multi-Hop Readout
Component: Equilibrium Multi-Hop Readout & TreeSoftmax Router (Phase WaveCMS - Mũi 2)

Theoretical Foundation:
1. Theorem 1(b) (Multiresolution Tree Consolidation):
   \\sum_p \\alpha_p \\Theta_p = \\Theta_{\\text{root}} + \\sum_{v \\ne \\text{root}} A_v(\\alpha) D_v
   where A_v(\\alpha) = \\sum_{p \\in \\mathcal{L}(v)} \\alpha_p is the cumulative mass of subtree v.
2. Theorem 1(c) (Haar Cancellation Property):
   \\sum_{c \\in \\mathrm{ch}(u)} W_c D_c = 0.
   When \\alpha is uninformative (uniform / high entropy), details cancel out and readout collapses to \\Theta_{\\text{root}}.
3. Theorem 4 (Banach Fixed-Point Existence, Uniqueness, and Convergence):
   h^{(n+1)} = T(h^{(n)}) := (1 - \\omega) h^{(n)} + \\omega \\cdot \\mathrm{Enc}_{W_0 + B \\Theta^{(n)} A}(q)
   Contraction condition: \\rho = \\frac{2 L_E R_c \\kappa_{\\max}}{\\tau} < 1.
   Guarantees Lip(T) <= 1 - \\omega(1 - \\rho) < 1 and linear convergence to unique h*.

100% Offline, Pure Python / PyTorch, Zero External APIs.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.wave_cms.tree_memory import compute_deterministic_routing_key, extract_sparse_terms


class ResolutionLevel(str, Enum):
    """Hierarchical resolution levels of the document memory tree."""
    CORPUS = "CORPUS"
    DOCUMENT = "DOCUMENT"
    SECTION = "SECTION"
    LEAF_PARAGRAPH = "LEAF_PARAGRAPH"


@dataclass
class RoutingResult:
    """Encapsulates the hierarchical routing distribution across the document tree."""
    leaf_weights: Dict[str, float]  # \\alpha_p for leaves, \\sum \\alpha_p = 1.0
    node_probabilities: Dict[str, float]  # A_v(\\alpha) = p_v for every node in tree
    entropy: float  # Shannon entropy H(\\alpha)
    normalized_entropy: float  # \\bar{H}(\\alpha) \\in [0.0, 1.0]
    effective_k: float  # k_{eff} = \\exp(H(\\alpha))
    resolution_level: str  # Estimated active resolution level
    is_diffuse: bool  # True if entropy indicates uninformative / diffuse routing

    def __getitem__(self, item: str) -> Any:
        mapping = {
            "alpha": self.leaf_weights,
            "leaf_weights": self.leaf_weights,
            "node_probabilities": self.node_probabilities,
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "effective_k": self.effective_k,
            "effective_resolution": self.resolution_level,
            "resolution_level": self.resolution_level,
            "is_diffuse": self.is_diffuse,
        }
        if item in mapping:
            return mapping[item]
        raise KeyError(item)

    def __contains__(self, item: str) -> bool:
        keys = {
            "h_star", "fixed_point", "theta_star", "final_theta",
            "hops_history", "trajectory_deltas", "rho_hat", "contraction_factor",
            "is_contractive", "converged", "num_hops", "history_alpha",
            "history_theta", "history_entropy", "final_alpha", "resolution_level",
        }
        return item in keys

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default


@dataclass
class EquilibriumResult:
    """Comprehensive output of the Banach Equilibrium Multi-Hop Readout loop."""
    fixed_point: torch.Tensor  # Equilibrium latent state h* \\in \\mathbb{R}^d
    num_hops: int  # Number of Banach iterations performed
    converged: bool  # True if ||h^{(n+1)} - h^{(n)}|| < tol
    contraction_factor: float  # Estimated \\hat{\\rho} = 2 L_E R_c \\kappa_{max} / \\tau
    is_contractive: bool  # True if \\hat{\\rho} < 1.0 (guarantees Banach convergence)
    trajectory_deltas: List[float] = field(default_factory=list)  # Step-wise L2 norm deltas
    history_alpha: List[Dict[str, float]] = field(default_factory=list)  # Step-wise \\alpha distributions
    history_theta: List[torch.Tensor] = field(default_factory=list)  # Step-wise synthesized \\Theta
    history_entropy: List[float] = field(default_factory=list)  # Step-wise entropy values
    final_theta: Optional[torch.Tensor] = None  # \\Theta(h*) synthesized at fixed point
    final_alpha: Optional[Dict[str, float]] = None  # Leaf routing at fixed point
    resolution_level: str = "LEAF_PARAGRAPH"

    def __getitem__(self, item: str) -> Any:
        mapping = {
            "h_star": self.fixed_point,
            "fixed_point": self.fixed_point,
            "theta_star": self.final_theta,
            "final_theta": self.final_theta,
            "hops_history": self.trajectory_deltas,
            "trajectory_deltas": self.trajectory_deltas,
            "rho_hat": self.contraction_factor,
            "contraction_factor": self.contraction_factor,
            "is_contractive": self.is_contractive,
            "converged": self.converged,
            "num_hops": self.num_hops,
            "history_alpha": self.history_alpha,
            "history_theta": self.history_theta,
            "history_entropy": self.history_entropy,
            "final_alpha": self.final_alpha,
            "resolution_level": self.resolution_level,
        }
        if item in mapping:
            return mapping[item]
        raise KeyError(item)

    def __contains__(self, item: str) -> bool:
        keys = {
            "h_star", "fixed_point", "theta_star", "final_theta",
            "hops_history", "trajectory_deltas", "rho_hat", "contraction_factor",
            "is_contractive", "converged", "num_hops", "history_alpha",
            "history_theta", "history_entropy", "final_alpha", "resolution_level",
        }
        return item in keys

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default


def _deterministic_text_embedding(text: str, dim: int) -> torch.Tensor:
    """
    Produces a deterministic, offline unit-L2 semantic representation of text
    using the shared word + char n-gram TF projection so queries align with tree routing keys.
    """
    vec_np = compute_deterministic_routing_key(text, dim=dim)
    vec = torch.from_numpy(vec_np).to(dtype=torch.float32)
    norm = torch.norm(vec, p=2)
    if norm > 1e-8:
        vec = vec / norm
    else:
        vec[0] = 1.0
    return vec


class TreeSoftmaxRouter(nn.Module):
    """
    Hierarchical Softmax Router on Document Trees.
    
    Computes branching probabilities along the tree:
        P(c | u) = Softmax(\\langle W_q h, \\kappa_c \\rangle / \\tau_\\ell)
        p_v = \\prod_{u \\to c \\in \\pi(v)} P(c | u)
        
    Theorem 1(b) allows O(depth) exact synthesis of \\Theta(\\alpha) without leaf enumeration:
        \\Theta(\\alpha) = \\Theta_{\\text{root}} + \\sum_{v \\ne \\text{root}} p_v D_v
        
    Resolution adjusts automatically with Shannon entropy:
    - High entropy (diffuse routing): wavelet details cancel via \\sum W_c D_c = 0.
    - Low entropy (focused routing): activates fine-grained leaf paragraph detail.
    """

    def __init__(
        self,
        dim: int = 64,
        tau: float = 0.20,
        d_model: Optional[int] = None,
        temperature: Optional[float] = None,
        level_temperatures: Optional[Dict[str, float]] = None,
        entropy_threshold: float = 0.85,
        routing_mode: str = "mean_key",
        tau_sparse: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model if d_model is not None else dim
        self.temperature = float(temperature if temperature is not None else tau)
        self.level_temperatures = level_temperatures or {}
        self.entropy_threshold = float(entropy_threshold)
        self.routing_mode = str(routing_mode)
        self.tau_sparse = max(1e-4, float(tau_sparse))

        # Projection matrix W_q \\in \\mathbb{R}^{d \\times d}; initialized as Identity so zero-shot
        # inference preserves exact semantic cosine alignment <h, kappa_v>.
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.eye_(self.w_q.weight)

    def _unwrap_root(self, tree_or_root: Any) -> Any:
        """Unwraps WaveTreeMemory or returns the root node directly."""
        if hasattr(tree_or_root, "root") and tree_or_root.root is not None:
            return tree_or_root.root
        return tree_or_root

    def _get_node_attr(self, node: Any, attr: str, default: Any = None) -> Any:
        """Helper for resilient duck-typing across different node implementations."""
        if hasattr(node, attr):
            return getattr(node, attr)
        if isinstance(node, dict) and attr in node:
            return node[attr]
        return default

    def _get_node_id(self, node: Any) -> str:
        return str(self._get_node_attr(node, "node_id", self._get_node_attr(node, "id", "node")))

    def _get_node_children(self, node: Any) -> List[Any]:
        return list(self._get_node_attr(node, "children", []))

    def _get_node_key(self, node: Any) -> torch.Tensor:
        # Check kappa, routing_key, key
        key = self._get_node_attr(node, "kappa", None)
        if key is None:
            key = self._get_node_attr(node, "routing_key", self._get_node_attr(node, "key", None))

        if key is None:
            key = torch.zeros(self.d_model, dtype=torch.float32)
        elif not isinstance(key, torch.Tensor):
            key = torch.tensor(key, dtype=torch.float32)

        if key.dim() == 1 and key.numel() == self.d_model:
            norm = torch.norm(key, p=2)
            if norm > 1e-8:
                key = key / norm
        return key

    def _get_node_theta(self, node: Any) -> torch.Tensor:
        t = self._get_node_attr(node, "theta", self._get_node_attr(node, "core", None))
        if t is None:
            t = torch.zeros((8, 8), dtype=torch.float32)
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        return t

    def _get_node_detail(self, node: Any) -> torch.Tensor:
        d = self._get_node_attr(node, "detail", self._get_node_attr(node, "wavelet_detail", None))
        if d is None:
            d = torch.zeros_like(self._get_node_theta(node))
        elif not isinstance(d, torch.Tensor):
            d = torch.tensor(d, dtype=torch.float32)
        return d

    def _get_node_level(self, node: Any) -> str:
        lvl = self._get_node_attr(node, "level", "LEAF_PARAGRAPH")
        if isinstance(lvl, Enum):
            return lvl.value
        return str(lvl)

    def _collect_leaves(self, root: Any) -> List[Any]:
        """Gathers all leaf nodes under the given root."""
        root = self._unwrap_root(root)
        children = self._get_node_children(root)
        if not children:
            return [root]
        leaves = []
        for c in children:
            leaves.extend(self._collect_leaves(c))
        return leaves

    def compute_temperature(self, level: str) -> float:
        """Returns level-wise temperature \\tau_\\ell or base temperature."""
        return self.level_temperatures.get(level, self.temperature)

    def _prepare_sparse_context(
        self,
        root: Any,
        query_text: Optional[str],
        memory_tree: Optional[Any] = None,
    ) -> Tuple[List[str], Dict[str, float], float]:
        """Extracts query terms, lazy BM25 IDF weights, and average leaf length."""
        leaves = self._collect_leaves(root)
        n_leaves = len(leaves)
        query_terms = list(dict.fromkeys(extract_sparse_terms(query_text or "")))
        if not query_terms or n_leaves == 0:
            return [], {}, 1.0

        df_counter = None
        if memory_tree is not None and hasattr(memory_tree, "corpus_df"):
            df_counter = memory_tree.corpus_df
        elif hasattr(root, "term_count"):
            df_counter = root.term_count

        idf_map: Dict[str, float] = {}
        for t in query_terms:
            if df_counter is not None and hasattr(df_counter, "idf"):
                idf_map[t] = max(0.0, float(df_counter.idf(t, n_leaves)))
            else:
                idf_map[t] = math.log((n_leaves + 0.5) / 0.5) + 1.0

        total_len = sum(max(0, int(self._get_node_attr(lf, "doc_len", 0))) for lf in leaves)
        avgdl = max(1.0, total_len / max(1, n_leaves))
        return query_terms, idf_map, avgdl

    def _score_leaf_sparse_lse(
        self,
        leaf: Any,
        q_proj: torch.Tensor,
        query_terms: List[str],
        idf_map: Dict[str, float],
        avgdl: float,
        tau: float,
        tau_s: float,
        device: torch.device,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> float:
        """Computes exact leaf logit z_l(q, h) = sparse(q, l)/tau_s + <W_q h, kappa_l>/tau."""
        sparse_sum = 0.0
        if query_terms:
            term_tf = self._get_node_attr(leaf, "term_tf", None)
            if term_tf:
                doc_len = max(0, int(self._get_node_attr(leaf, "doc_len", 0)))
                denom_norm = k1 * (1.0 - b + b * (doc_len / avgdl))
                for t in query_terms:
                    tf = term_tf.get(t, 0)
                    if tf > 0:
                        g = float(tf) / (float(tf) + denom_norm)
                        sparse_sum += idf_map[t] * g
        key = self._get_node_key(leaf).to(device)
        dense_score = float(torch.dot(q_proj, key).item()) / tau
        return (sparse_sum / tau_s) + dense_score

    def compute_upper_bound(
        self,
        node: Any,
        query_terms: List[str],
        idf_map: Dict[str, float],
        q_proj_norm: float,
        tau: float,
        tau_s: float,
    ) -> float:
        """
        Theorem 4 admissible upper bound:
            U_v(q, h) = (1/tau_s) * sum_{t in T(q)} idf_t * 1[c_v(t) > 0] + ||W_q h||_2 / tau
        """
        sparse_ub = 0.0
        if query_terms:
            term_count = self._get_node_attr(node, "term_count", None)
            if term_count is not None and hasattr(term_count, "nonzero"):
                for t in query_terms:
                    if term_count.nonzero(t):
                        sparse_ub += idf_map.get(t, 0.0)
            else:
                for t in query_terms:
                    sparse_ub += idf_map.get(t, 0.0)
        return (sparse_ub / tau_s) + (q_proj_norm / tau)

    def branch_and_bound_argmax(
        self,
        h: torch.Tensor,
        root: Any,
        query_text: Optional[str] = None,
        memory_tree: Optional[Any] = None,
    ) -> Tuple[str, float, int]:
        """
        Exact admissible Branch-and-Bound argmax using U_v(q, h).
        Returns (best_leaf_id, best_score, pruned_subtrees_count).
        """
        mem_tree = memory_tree if memory_tree is not None else (root if hasattr(root, "corpus_df") else None)
        root_node = self._unwrap_root(root)
        h_vec = h.squeeze()
        device = self.w_q.weight.device
        q_proj = self.w_q(h_vec.to(device))
        q_proj_norm = float(torch.norm(q_proj, p=2).item())

        tau = max(1e-4, float(self.temperature))
        tau_s = max(1e-4, float(self.tau_sparse))
        query_terms, idf_map, avgdl = self._prepare_sparse_context(root_node, query_text, mem_tree)

        best_id = ""
        best_score = -math.inf
        pruned = 0

        def _bnb(node: Any) -> None:
            nonlocal best_id, best_score, pruned
            ub = self.compute_upper_bound(node, query_terms, idf_map, q_proj_norm, tau, tau_s)
            if ub < best_score - 1e-12:
                pruned += 1
                return
            children = self._get_node_children(node)
            if not children:
                score = self._score_leaf_sparse_lse(
                    node, q_proj, query_terms, idf_map, avgdl, tau, tau_s, device
                )
                if score > best_score:
                    best_score = score
                    best_id = self._get_node_id(node)
                return
            scored_children = [
                (self.compute_upper_bound(c, query_terms, idf_map, q_proj_norm, tau, tau_s), c)
                for c in children
            ]
            scored_children.sort(key=lambda x: x[0], reverse=True)
            for child_ub, child in scored_children:
                if child_ub < best_score - 1e-12:
                    pruned += 1
                    continue
                _bnb(child)

        _bnb(root_node)
        return best_id, best_score, pruned

    def route(
        self,
        h: torch.Tensor,
        root: Any,
        query_text: Optional[str] = None,
        memory_tree: Optional[Any] = None,
    ) -> RoutingResult:
        """
        Hierarchical TreeSoftmax routing from query latent vector h.

        Args:
            h: Query latent state vector of shape (d_model,) or (1, d_model)
            root: Root node or WaveTreeMemory instance
            query_text: Optional query string for Theorem 4 Sparse-LSE routing
            memory_tree: Optional WaveTreeMemory for corpus_df lookup

        Returns:
            RoutingResult with leaf distribution, node probabilities, and entropy.
        """
        mem_tree = memory_tree if memory_tree is not None else (root if hasattr(root, "corpus_df") else None)
        root = self._unwrap_root(root)
        h_vec = h.squeeze()
        if h_vec.dim() != 1:
            raise ValueError(f"Expected 1D latent vector, got shape {h.shape}")

        # Projected query W_q h
        device = self.w_q.weight.device
        h_vec = h_vec.to(device)
        q_proj = self.w_q(h_vec)

        node_probs: Dict[str, float] = {}
        root_id = self._get_node_id(root)
        node_probs[root_id] = 1.0

        if self.routing_mode == "sparse_lse":
            tau = max(1e-4, float(self.temperature))
            tau_s = max(1e-4, float(self.tau_sparse))
            query_terms, idf_map, avgdl = self._prepare_sparse_context(root, query_text, mem_tree)
            node_z: Dict[int, float] = {}

            def _compute_lse(node: Any, active_terms: List[str]) -> float:
                children = self._get_node_children(node)
                if not children:
                    z_val = self._score_leaf_sparse_lse(
                        node, q_proj, active_terms, idf_map, avgdl, tau, tau_s, device
                    )
                    node_z[id(node)] = z_val
                    return z_val

                # Filter active_terms via c_v(t) > 0 so zero-overlap subtrees skip sparse work
                term_count = self._get_node_attr(node, "term_count", None)
                if active_terms and term_count is not None and hasattr(term_count, "nonzero"):
                    sub_terms = [t for t in active_terms if term_count.nonzero(t)]
                else:
                    sub_terms = active_terms

                child_zs = [_compute_lse(c, sub_terms) for c in children]
                m = max(child_zs)
                lse_val = m + math.log(sum(math.exp(cz - m) for cz in child_zs))
                node_z[id(node)] = lse_val
                return lse_val

            _compute_lse(root, query_terms)

            def _distribute_lse_probs(node: Any, current_p: float) -> None:
                children = self._get_node_children(node)
                if not children:
                    return
                z_u = node_z[id(node)]
                for child in children:
                    z_c = node_z[id(child)]
                    b_p = math.exp(z_c - z_u)
                    child_p = current_p * b_p
                    node_probs[self._get_node_id(child)] = child_p
                    _distribute_lse_probs(child, child_p)

            _distribute_lse_probs(root, 1.0)
        else:
            # Precompute subtree leaf counts for unbiased hierarchical prior log|L(c)|
            # (independent of h, so Lipschitz constant of TreeSoftmax wrt h is unchanged)
            leaf_counts: Dict[int, int] = {}

            def _count_leaves(node: Any) -> int:
                children = self._get_node_children(node)
                if not children:
                    cnt = 1
                else:
                    cnt = sum(_count_leaves(c) for c in children)
                leaf_counts[id(node)] = cnt
                return cnt

            _count_leaves(root)

            # 1. Recursive TreeSoftmax top-down pass
            def _route_node(node: Any, current_p: float) -> None:
                children = self._get_node_children(node)
                if not children:
                    return

                node_level = self._get_node_level(node)
                tau = max(1e-4, self.compute_temperature(node_level))

                # Compute logits log|L(c)| + <W_q h, \kappa_c> / \tau
                logits = []
                for child in children:
                    key = self._get_node_key(child).to(device)
                    prior = math.log(max(1, leaf_counts.get(id(child), 1)))
                    dot = torch.dot(q_proj, key) / tau + prior
                    logits.append(dot)

                logits_tensor = torch.stack(logits)
                branch_probs = F.softmax(logits_tensor, dim=0).detach().cpu().tolist()

                for child, b_p in zip(children, branch_probs):
                    child_id = self._get_node_id(child)
                    child_p = current_p * float(b_p)
                    node_probs[child_id] = child_p
                    _route_node(child, child_p)

            _route_node(root, 1.0)

        # 2. Extract leaf distribution \\alpha_p
        leaves = self._collect_leaves(root)
        leaf_weights: Dict[str, float] = {}
        total_leaf_p = 0.0
        for lf in leaves:
            lid = self._get_node_id(lf)
            lp = node_probs.get(lid, 0.0)
            leaf_weights[lid] = lp
            total_leaf_p += lp

        # Exact numerical renormalization for leaves & internal nodes (Theorem 1(b) consistency)
        if total_leaf_p > 1e-12:
            for lid in leaf_weights:
                leaf_weights[lid] /= total_leaf_p
            for nid in list(node_probs.keys()):
                if nid != root_id:
                    node_probs[nid] /= total_leaf_p
        elif leaf_weights:
            uniform_p = 1.0 / len(leaf_weights)
            for lid in leaf_weights:
                leaf_weights[lid] = uniform_p

        # 3. Calculate Shannon entropy H(\\alpha)
        entropy = 0.0
        for p in leaf_weights.values():
            if p > 1e-12:
                entropy -= p * math.log(p)

        n_leaves = len(leaf_weights)
        max_entropy = math.log(n_leaves) if n_leaves > 1 else 1.0
        norm_entropy = min(1.0, max(0.0, entropy / max_entropy)) if max_entropy > 0 else 0.0
        effective_k = math.exp(entropy)
        is_diffuse = bool(norm_entropy >= self.entropy_threshold)

        # 4. Determine resolution level based on entropy & tree depth
        if is_diffuse or norm_entropy > 0.8:
            res_level = ResolutionLevel.CORPUS.value
        elif norm_entropy > 0.5:
            res_level = ResolutionLevel.DOCUMENT.value
        elif norm_entropy > 0.25:
            res_level = ResolutionLevel.SECTION.value
        else:
            res_level = ResolutionLevel.LEAF_PARAGRAPH.value

        return RoutingResult(
            leaf_weights=leaf_weights,
            node_probabilities=node_probs,
            entropy=round(entropy, 6),
            normalized_entropy=round(norm_entropy, 6),
            effective_k=round(effective_k, 4),
            resolution_level=res_level,
            is_diffuse=is_diffuse,
        )

    def synthesize_memory_core(self, routing_res: RoutingResult, root: Any) -> torch.Tensor:
        """
        Synthesizes effective memory matrix \\Theta(\\alpha) via Theorem 1(b):
            \\Theta(\\alpha) = \\Theta_{\\text{root}} + \\sum_{v \\ne \\text{root}} p_v D_v
        
        Args:
            routing_res: Output from self.route()
            root: Root node or WaveTreeMemory instance
            
        Returns:
            \\Theta(\\alpha) \\in \\mathbb{R}^{r \\times r}
        """
        root = self._unwrap_root(root)
        theta_root = self._get_node_theta(root)
        theta_accum = theta_root.clone()

        def _accum_details(node: Any) -> None:
            for child in self._get_node_children(node):
                cid = self._get_node_id(child)
                p_v = routing_res.node_probabilities.get(cid, 0.0)
                if p_v > 0.0:
                    d_v = self._get_node_detail(child)
                    theta_accum.add_(d_v * p_v)
                _accum_details(child)

        _accum_details(root)
        return theta_accum


class EquilibriumMultiHopReadout(nn.Module):
    """
    Equilibrium Multi-Hop Readout using Banach Fixed-Point Iteration (Theorem 4).
    
    Fixed-point operator:
        T(h) = (1 - \\omega) h + \\omega \\cdot \\mathrm{Enc}_{W_0 + B \\Theta(\\alpha(h)) A}(q)
        
    Monitors contraction factor \\hat{\\rho} = \\frac{2 L_E R_c \\kappa_{\\max}}{\\tau}:
    - When \\hat{\\rho} < 1: Banach fixed-point theorem guarantees unique equilibrium h*
      and monotonic linear convergence without hallucination cycles.
    - When \\hat{\\rho} >= 1: Emits contractive warning flag to prevent divergent latent drift.
    """

    def __init__(
        self,
        dim: int = 64,
        r: int = 8,
        omega: float = 0.5,
        max_hops: int = 3,
        tol: float = 1e-4,
        d_model: Optional[int] = None,
        r_dim: Optional[int] = None,
        default_lipschitz: float = 1.0,
        router: Optional[TreeSoftmaxRouter] = None,
        encoder_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model if d_model is not None else dim
        self.r_dim = r_dim if r_dim is not None else r
        self.omega = float(omega)
        self.max_hops = int(max_hops)
        self.tol = float(tol)
        self.default_lipschitz = float(default_lipschitz)
        self.bridge_weight = 0.008

        self.router = router if router is not None else TreeSoftmaxRouter(d_model=self.d_model)
        self.custom_encoder_fn = encoder_fn

        # Base model parameters: W_0 \\in \\mathbb{R}^{d \\times d}
        self.w_0 = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.eye_(self.w_0.weight)

        # Shared orthogonal low-rank projection bases A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{d \times r}
        # Dimension-scaled gain preserves Banach contraction across d_model while scaling capacity
        gain = 0.20 * min(1.75, max(1.0, math.sqrt(self.d_model / 64.0)))
        a_init = torch.empty(self.r_dim, self.d_model)
        b_init = torch.empty(self.d_model, self.r_dim)
        nn.init.orthogonal_(a_init, gain=gain)
        nn.init.orthogonal_(b_init, gain=gain)
        self.a_proj = nn.Parameter(a_init)
        self.b_proj = nn.Parameter(b_init)

    def compute_encoder_lipschitz(self, q_norm: float = 1.0) -> float:
        """
        Computes the analytical Lipschitz upper bound of
            Theta -> tanh((W_0 + B Theta A) q)
        Since sech^2(z) <= 1 everywhere:
            L_E <= ||B||_2 * ||A||_2 * ||q||_2.
        """
        q_norm_val = max(0.0, float(q_norm))
        if self.custom_encoder_fn is not None:
            return self.default_lipschitz * q_norm_val
        with torch.no_grad():
            norm_b = float(torch.linalg.matrix_norm(self.b_proj, ord=2))
            norm_a = float(torch.linalg.matrix_norm(self.a_proj, ord=2))
        return max(1e-6, norm_b * norm_a * q_norm_val + self.bridge_weight)

    def compute_contraction_factor(
        self,
        root: Any,
        tau: Optional[float] = None,
        l_e: Optional[float] = None,
        q_norm: float = 1.0,
    ) -> Tuple[float, bool]:
        """
        Calculates empirical TreeSoftmax Banach contraction factor:
            \\hat{\\rho}_{\\text{tree}} = 2 L_E R_c \\sum_{\\ell=1}^{D} \\frac{\\kappa_{\\max,\\ell}}{\\tau_\\ell}

        where:
            D: Tree depth (number of hierarchical branching levels)
            R_c = \\max_p ||\\Theta_p - \\Theta_c||_F (radius of parameter cores)
            \\kappa_{\\max,\\ell} = \\max_{v \\text{ at depth } \\ell} ||W_q^\\top \\kappa_v||_2
            L_E \\le ||B||_2 ||A||_2 ||q||_2 (via \\operatorname{sech}^2 \\le 1)
            \\tau_\\ell: Softmax temperature at level \\ell

        Returns:
            Tuple of (\\hat{\\rho}, is_contractive: bool)
        """
        root = self.router._unwrap_root(root)
        tau_override = max(1e-4, float(tau)) if tau is not None else None
        if l_e is not None:
            l_e_val = float(l_e)
        else:
            l_e_val = self.compute_encoder_lipschitz(q_norm=q_norm)

        # 1. Compute R_c across leaf cores
        leaves = self.router._collect_leaves(root)
        if not leaves:
            return 0.0, True

        theta_root = self.router._get_node_theta(root)
        r_c = 0.0
        for lf in leaves:
            th_p = self.router._get_node_theta(lf)
            dist = float(torch.norm(th_p - theta_root, p="fro"))
            if dist > r_c:
                r_c = dist
        r_c = max(r_c, 1e-4)

        # 2. Compute depth-wise sum_{\ell=1}^D (\kappa_{\max,\ell} / \tau_\ell)
        w_q_mat = self.router.w_q.weight  # shape (d, d)
        level_ratio_max: Dict[int, float] = {}

        def _traverse(node: Any, depth: int) -> None:
            children = self.router._get_node_children(node)
            if not children:
                return
            child_depth = depth + 1
            parent_level = self.router._get_node_level(node)
            level_tau = (
                tau_override
                if tau_override is not None
                else max(1e-4, self.router.compute_temperature(parent_level))
            )
            for c in children:
                key = self.router._get_node_key(c)
                proj = torch.matmul(w_q_mat.t(), key)
                norm_proj = max(float(torch.norm(proj, p=2)), 1e-4)
                ratio = norm_proj / level_tau
                if ratio > level_ratio_max.get(child_depth, 0.0):
                    level_ratio_max[child_depth] = ratio
                _traverse(c, child_depth)

        _traverse(root, 0)
        if not level_ratio_max:
            return 0.0, True

        depth_sum_ratio = sum(level_ratio_max.values())

        # 3. Tree contraction factor \hat{\rho}
        rho_hat = 2.0 * l_e_val * r_c * depth_sum_ratio
        is_contractive = bool(rho_hat < 1.0)
        return round(rho_hat, 6), is_contractive

    def encode_step(
        self,
        q_emb: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executes memory-adapted forward pass:
            W_{eff} = W_0 + B \\Theta A
            h_{adapted} = Enc_{W_{eff}}(q)
        """
        if self.custom_encoder_fn is not None:
            return self.custom_encoder_fn(q_emb, theta)

        # Associative Continuum Memory Adapter: (W_0 + B Theta A) q = W_0 q + B (Theta (A q))
        # Avoids materializing dense d x d outer product B @ Theta @ A
        q_vec = q_emb.squeeze()
        base_out = torch.matmul(self.w_0.weight, q_vec)
        proj_a = torch.matmul(self.a_proj, q_vec)
        core_out = torch.matmul(theta, proj_a)
        adapted_out = torch.matmul(self.b_proj, core_out)
        h_adapted = torch.tanh(base_out + adapted_out)
        return h_adapted

    def readout(
        self,
        query: Optional[Union[str, torch.Tensor]] = None,
        tree: Any = None,
        max_hops: Optional[int] = None,
        root: Any = None,
        query_embedding: Optional[Union[str, torch.Tensor]] = None,
        query_text: Optional[str] = None,
    ) -> EquilibriumResult:
        """
        Runs the full Banach Equilibrium Multi-Hop Readout loop.
        
        Args:
            query: Question text string or initial query vector q \\in \\mathbb{R}^d
            tree: Root node or WaveTreeMemory instance
            max_hops: Optional override for max Banach iterations
            root: Optional alias for tree
            query_embedding: Optional alias for query
            query_text: Optional raw query string for Theorem 4 Sparse-LSE routing
            
        Returns:
            EquilibriumResult with fixed point h*, trajectory deltas, and contraction audit.
        """
        if query is None and query_embedding is not None:
            query = query_embedding
        if tree is None and root is not None:
            tree = root
        if query is None:
            query = torch.zeros(self.d_model, dtype=torch.float32)

        q_text_resolved = query_text if query_text is not None else (query if isinstance(query, str) else None)
        memory_tree_ref = tree if hasattr(tree, "corpus_df") else None
        root = self.router._unwrap_root(tree)
        device = self.w_0.weight.device

        # Handle string or tensor query
        if isinstance(query, str):
            q_emb = _deterministic_text_embedding(query, self.d_model).to(device)
        elif isinstance(query, torch.Tensor):
            q_emb = query.squeeze().to(device)
            if q_emb.numel() != self.d_model:
                raise ValueError(f"Query embedding size {q_emb.numel()} != d_model {self.d_model}")
        else:
            q_emb = torch.tensor(query, dtype=torch.float32).to(device)

        hops_limit = max_hops if max_hops is not None else self.max_hops

        # Audit contraction condition (Theorem 4) including ||q||_2 in L_E
        q_norm = float(torch.norm(q_emb, p=2))
        rho_hat, is_contractive = self.compute_contraction_factor(root, q_norm=q_norm)
        if not is_contractive:
            warnings.warn(
                f"[WaveCMS Notice] Contraction factor \\hat{{\\rho}} = {rho_hat:.4f} >= 1.0. "
                f"Banach contractive guarantee not satisfied; fixed-point iteration may exhibit slow convergence.",
                category=UserWarning,
                stacklevel=2,
            )

        # Initial state h^{(0)} = Enc_{W_0}(q)
        h_current = torch.tanh(self.w_0(q_emb))

        trajectory_deltas: List[float] = []
        history_alpha: List[Dict[str, float]] = []
        history_theta: List[torch.Tensor] = []
        history_entropy: List[float] = []

        converged = False
        final_routing: Optional[RoutingResult] = None
        final_theta: Optional[torch.Tensor] = None

        leaf_keys: Dict[str, torch.Tensor] = {}
        if self.custom_encoder_fn is None and root is not None:
            for lf in self.router._collect_leaves(root):
                lid = self.router._get_node_id(lf)
                leaf_keys[lid] = self.router._get_node_key(lf).to(device=device, dtype=q_emb.dtype)
        q_norm_sq = float(torch.dot(q_emb, q_emb))

        for hop in range(hops_limit):
            # Step 1: Hierarchical TreeSoftmax routing (sparse channel is h-invariant)
            routing_res = self.router.route(
                h_current,
                root,
                query_text=q_text_resolved,
                memory_tree=memory_tree_ref,
            )
            history_alpha.append(routing_res.leaf_weights)
            history_entropy.append(routing_res.entropy)
            final_routing = routing_res

            # Step 2: Theorem 1(b) Wavelet core synthesis \Theta^{(n)}
            theta_n = self.router.synthesize_memory_core(routing_res, root)
            history_theta.append(theta_n)
            final_theta = theta_n

            # Step 3: Adapted forward pass + bounded multi-hop bridge key feedback
            h_adapted = self.encode_step(q_emb, theta_n)
            if leaf_keys:
                kappa_mix = torch.zeros_like(q_emb)
                weight_sum = 0.0
                for lid, alpha_p in routing_res.leaf_weights.items():
                    if alpha_p > 0.0 and lid in leaf_keys:
                        kappa_mix = kappa_mix + float(alpha_p) * leaf_keys[lid]
                        weight_sum += float(alpha_p)
                if weight_sum > 1e-12:
                    kappa_mix = kappa_mix / weight_sum
                    if q_norm_sq > 1e-12:
                        kappa_mix = kappa_mix - (torch.dot(kappa_mix, q_emb) / q_norm_sq) * q_emb
                    k_norm = float(torch.norm(kappa_mix, p=2))
                    if k_norm > 1.0:
                        kappa_mix = kappa_mix / k_norm
                    h_adapted = h_adapted + self.bridge_weight * kappa_mix

            # Step 4: Banach equilibrium update h^{(n+1)} = (1 - \\omega) h^{(n)} + \\omega h_{adapted}
            h_next = (1.0 - self.omega) * h_current + self.omega * h_adapted

            # Step 5: Check Banach Cauchy delta ||h^{(n+1)} - h^{(n)}||_2
            delta = float(torch.norm(h_next - h_current, p=2))
            trajectory_deltas.append(round(delta, 8))

            h_current = h_next
            if delta < self.tol:
                converged = True
                break

        res_level = final_routing.resolution_level if final_routing else "LEAF_PARAGRAPH"

        return EquilibriumResult(
            fixed_point=h_current.detach(),
            num_hops=len(trajectory_deltas),
            converged=converged,
            contraction_factor=rho_hat,
            is_contractive=is_contractive,
            trajectory_deltas=trajectory_deltas,
            history_alpha=history_alpha,
            history_theta=history_theta,
            history_entropy=history_entropy,
            final_theta=final_theta,
            final_alpha=final_routing.leaf_weights if final_routing else {},
            resolution_level=res_level,
        )

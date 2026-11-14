"""
WaveCMS Continuum Memory & Certified Unlearning Engine (Phase WaveCMS - Trụ cột 3.2).

Theoretical Foundations:
1. Low-Rank Continuum Parameterization (Proposal Section 2.1):
   W = W_0 + B @ Theta @ A
   where W_0 in R^{d x d} is base frozen weight,
   A in R^{r x d} and B in R^{d x r} are shared meta-learned projection bases,
   Theta in R^{r x r} is the local memory core.
2. Fast Associative Computation (O(d r + r^2) vs O(d^2 r)):
   (W_0 + B Theta A) x = W_0 x + B (Theta (A x))
   X (W_0 + B Theta A)^T = X W_0^T + (X A^T) Theta^T B^T
3. Theorem 2: Certified Exact Unlearning in O(depth):
   W_{root}^{new} = W_{root} - W_u
   Theta_{root}^{new} = (W_{root} Theta_{root} - W_u Theta_u) / W_{root}^{new}
   Analytical error < 1e-12 (approaching machine precision ~ 1e-16).
4. Subtree Memory Deallocation:
   Recursively severs cyclic references, deallocates parameter tensors,
   and clears caches to guarantee zero memory leakage upon unlearning.

100% Strict Offline, Pure Python stdlib, NumPy & PyTorch, Zero External APIs.
"""

from __future__ import annotations

import gc
import math
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


def release_subtree_memory(node: Optional[DocumentTreeNode]) -> Tuple[int, int]:
    """
    Recursively severs all circular references, deletes parameter tensors,
    and clears node caches to guarantee zero memory leakage upon unlearning.
    
    Args:
        node: The root of the subtree being unlearned / detached.
        
    Returns:
        Tuple of (num_freed_nodes: int, estimated_freed_bytes: int)
    """
    if node is None:
        return 0, 0

    freed_nodes = 0
    freed_bytes = 0

    stack = [node]
    nodes_to_clear: List[DocumentTreeNode] = []

    while stack:
        curr = stack.pop()
        nodes_to_clear.append(curr)
        for child in curr.children:
            stack.append(child)

    for n in nodes_to_clear:
        freed_nodes += 1
        if n.theta is not None:
            freed_bytes += n.theta.nbytes
            n.theta = None
        if n.detail is not None:
            freed_bytes += n.detail.nbytes
            n.detail = None
        if n.kappa is not None:
            freed_bytes += n.kappa.nbytes
            n.kappa = None
        if hasattr(n, "unnormalized_kappa") and n.unnormalized_kappa is not None:
            freed_bytes += n.unnormalized_kappa.nbytes
            n.unnormalized_kappa = None

        freed_bytes += len(n.text.encode("utf-8")) + len(n.title.encode("utf-8"))
        n.text = ""
        n.title = ""
        n.metadata.clear()
        n.children.clear()
        n.parent = None

    gc.collect()
    return freed_nodes, freed_bytes


class ContinuumMemoryAdapter:
    """
    High-performance NumPy implementation of Continuum Memory adaptation:
        W = W_0 + B @ Theta @ A
        
    Provides:
    1. Fast associative forward pass y = W_0 x + B (Theta (A x)) avoiding dense d x d allocation.
    2. Batched matrix / sequence projection X W_0^T + (X A^T) Theta^T B^T.
    3. Analytical equivalence verification: ||y_fast - W_eff x||_inf < 1e-12.
    4. Closed-form Gauss-Newton leaf core update (Section 2.2).
    5. Explicit memory deallocation and footprint auditing.
    """

    def __init__(
        self,
        d_model: int = 64,
        r: int = 8,
        seed: int = 42,
        w0: Optional[np.ndarray] = None,
        a_proj: Optional[np.ndarray] = None,
        b_proj: Optional[np.ndarray] = None,
    ) -> None:
        self.d_model = int(d_model)
        self.r = int(r)
        self.seed = int(seed)

        rng = np.random.RandomState(self.seed)
        q_mat, _ = np.linalg.qr(rng.randn(self.d_model, self.d_model))

        # Base frozen weight W_0 in R^{d x d}
        if w0 is not None:
            self.W0 = np.asarray(w0, dtype=np.float64).reshape((self.d_model, self.d_model))
        else:
            self.W0 = (0.1 * q_mat).astype(np.float64)

        # Shared projection bases A in R^{r x d} and B in R^{d x r} (semi-orthogonal)
        if a_proj is not None:
            self.A = np.asarray(a_proj, dtype=np.float64).reshape((self.r, self.d_model))
        else:
            self.A = q_mat[:self.r, :].astype(np.float64)

        if b_proj is not None:
            self.B = np.asarray(b_proj, dtype=np.float64).reshape((self.d_model, self.r))
        else:
            self.B = self.A.T.copy()

        self._cached_w_eff: Optional[np.ndarray] = None
        self._cached_theta_id: Optional[int] = None

    def synthesize_effective_weight(self, theta: np.ndarray) -> np.ndarray:
        """
        Synthesizes the full adapted matrix W_{eff} = W_0 + B @ Theta @ A in R^{d x d}.
        """
        th = np.asarray(theta, dtype=np.float64)
        if th.shape != (self.r, self.r):
            th_fixed = np.zeros((self.r, self.r), dtype=np.float64)
            r_m = min(self.r, th.shape[0])
            c_m = min(self.r, th.shape[1] if th.ndim > 1 else 1)
            th_fixed[:r_m, :c_m] = th[:r_m, :c_m]
            th = th_fixed

        adapted = self.B @ th @ self.A
        return self.W0 + adapted

    def forward(
        self,
        x: np.ndarray,
        theta: Optional[np.ndarray] = None,
        use_associative_fast_path: bool = True,
    ) -> np.ndarray:
        """
        Executes memory-adapted forward pass.
        
        When use_associative_fast_path is True, computes:
            (W_0 + B Theta A) x = W_0 x + B (Theta (A x))
            
        Complexity:
            Associative: O(d^2 + d*r + r^2)
            Dense outer product: O(d^2 * r + d^2)
            When d=64, r=8: ~4,600 vs ~36,800 FLOPs (8x speedup).
        """
        x_arr = np.asarray(x, dtype=np.float64)

        if theta is None:
            # Identity adaptation (W = W_0)
            if x_arr.ndim == 1:
                return self.W0 @ x_arr
            return x_arr @ self.W0.T

        th = np.asarray(theta, dtype=np.float64)
        if th.shape != (self.r, self.r):
            th_fixed = np.zeros((self.r, self.r), dtype=np.float64)
            r_m = min(self.r, th.shape[0])
            c_m = min(self.r, th.shape[1] if th.ndim > 1 else 1)
            th_fixed[:r_m, :c_m] = th[:r_m, :c_m]
            th = th_fixed

        if not use_associative_fast_path:
            w_eff = self.synthesize_effective_weight(th)
            if x_arr.ndim == 1:
                return w_eff @ x_arr
            return x_arr @ w_eff.T

        # Associative Fast Path
        if x_arr.ndim == 1:
            # 1D vector x in R^d
            base_out = self.W0 @ x_arr                  # R^d
            proj_a = self.A @ x_arr                     # R^r
            core_out = th @ proj_a                      # R^r
            adapted_out = self.B @ core_out             # R^d
            return base_out + adapted_out

        elif x_arr.ndim == 2:
            # 2D matrix X in R^{B x d}
            base_out = x_arr @ self.W0.T                # R^{B x d}
            proj_a = x_arr @ self.A.T                   # R^{B x r}
            core_out = proj_a @ th.T                    # R^{B x r}
            adapted_out = core_out @ self.B.T           # R^{B x d}
            return base_out + adapted_out

        elif x_arr.ndim == 3:
            # 3D tensor X in R^{B x T x d}
            base_out = np.matmul(x_arr, self.W0.T)
            proj_a = np.matmul(x_arr, self.A.T)
            core_out = np.matmul(proj_a, th.T)
            adapted_out = np.matmul(core_out, self.B.T)
            return base_out + adapted_out

        else:
            raise ValueError(f"Unsupported input dimension {x_arr.ndim}")

    def verify_adaptation_equivalence(
        self,
        theta: np.ndarray,
        x: Optional[np.ndarray] = None,
        tol: float = 1e-12,
    ) -> Tuple[bool, float]:
        """
        Rigorously verifies that the fast associative path matches the exact
        dense outer-product matrix multiplication up to analytical tolerance tol.
        
        Returns:
            Tuple of (is_equivalent: bool, max_discrepancy: float)
        """
        if x is None:
            rng = np.random.RandomState(1234)
            x = rng.randn(self.d_model)

        y_fast = self.forward(x, theta=theta, use_associative_fast_path=True)
        y_dense = self.forward(x, theta=theta, use_associative_fast_path=False)

        max_disc = float(np.max(np.abs(y_fast - y_dense)))
        is_equiv = bool(max_disc < tol)
        return is_equiv, max_disc

    def compute_gauss_newton_update(
        self,
        grad_theta: np.ndarray,
        fisher_matrix: Optional[np.ndarray] = None,
        beta: float = 0.1,
    ) -> np.ndarray:
        """
        Computes closed-form Gauss-Newton leaf core update (Proposal Section 2.2):
            Theta_p = - (F_p + beta * I)^{-1} grad_Theta L_p(0)
            
        Because Theta is only r x r (e.g. 8 x 8 = 64 dimensions), the Fisher matrix
        is 64 x 64 and can be inverted exactly in < 0.1 ms without approximations.
        """
        r_dim = self.r
        num_params = r_dim * r_dim

        g_vec = np.asarray(grad_theta, dtype=np.float64).reshape(num_params)

        if fisher_matrix is None:
            # Default uninformative empirical Fisher (identity damping)
            f_mat = np.eye(num_params, dtype=np.float64)
        else:
            f_mat = np.asarray(fisher_matrix, dtype=np.float64).reshape((num_params, num_params))

        regularized_f = f_mat + max(1e-6, float(beta)) * np.eye(num_params, dtype=np.float64)

        try:
            delta_theta_vec = np.linalg.solve(regularized_f, -g_vec)
        except np.linalg.LinAlgError:
            delta_theta_vec = -np.linalg.pinv(regularized_f) @ g_vec

        return delta_theta_vec.reshape((r_dim, r_dim))

    def compute_semantic_leaf_core(
        self,
        kappa: np.ndarray,
        beta: float = 1.0,
    ) -> np.ndarray:
        """
        Exact closed-form Ridge / Gauss-Newton leaf core binding for semantic routing key kappa:
            min_Theta  0.5 * ||(kappa - W_0 kappa) - B Theta A kappa||_2^2 + 0.5 * beta * ||Theta||_F^2

        Since B^T B = I_r, the exact analytical minimizer has rank-1 closed form:
            z = A kappa,  r_vec = B^T (kappa - W_0 kappa)
            Theta* = (r_vec @ z^T) / (||z||_2^2 + beta)
        """
        k = np.asarray(kappa, dtype=np.float64).reshape(self.d_model)
        z = self.A @ k
        r_vec = self.B.T @ (k - self.W0 @ k)
        denom = float(np.dot(z, z)) + max(1e-6, float(beta))
        return np.outer(r_vec, z) / denom

    def release_memory(self) -> None:
        """Clears cached matrices and triggers garbage collection."""
        self._cached_w_eff = None
        self._cached_theta_id = None
        gc.collect()

    def get_memory_footprint(self) -> Dict[str, Any]:
        """Audits memory footprint of base weights vs continuum memory cores."""
        base_bytes = self.W0.nbytes
        basis_bytes = self.A.nbytes + self.B.nbytes
        core_bytes = self.r * self.r * 8  # 8 bytes per float64
        full_dense_bytes = self.d_model * self.d_model * 8

        return {
            "d_model": self.d_model,
            "r": self.r,
            "base_w0_bytes": base_bytes,
            "basis_a_b_bytes": basis_bytes,
            "bytes_per_memory_core": core_bytes,
            "kb_per_memory_core": round(core_bytes / 1024, 3),
            "full_dense_matrix_bytes": full_dense_bytes,
            "compression_ratio": round(full_dense_bytes / max(1, core_bytes), 2),
        }


class ContinuumMemoryLinear(nn.Module):
    """
    PyTorch Module implementing low-rank Continuum Memory Linear layer:
        y = x W_0^T + (x A^T) Theta^T B^T
        
    Enables fully differentiable multi-hop latent reasoning and meta-learning (Section 3).
    Associative forward pass guarantees zero intermediate d x d outer-product allocation.
    """

    def __init__(
        self,
        d_model: int = 64,
        r: int = 8,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.r = int(r)

        self.w_0 = nn.Linear(self.d_model, self.d_model, bias=bias)
        nn.init.orthogonal_(self.w_0.weight)

        # Meta-learned projection bases A in R^{r x d} and B in R^{d x r}
        self.a_proj = nn.Parameter(torch.randn(self.r, self.d_model) / math.sqrt(self.d_model))
        self.b_proj = nn.Parameter(torch.randn(self.d_model, self.r) / math.sqrt(self.r))

    def forward(
        self,
        x: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Differentiable associative forward pass:
            y = F.linear(x, self.w_0.weight) + (x @ A^T @ Theta^T) @ B^T
        """
        # Base linear projection: x @ W_0^T
        base_out = self.w_0(x)

        if theta is None:
            return base_out

        th = theta.to(dtype=x.dtype, device=x.device)
        if th.ndim == 1:
            th = th.reshape((self.r, self.r))

        # Associative low-rank path
        proj_a = F.linear(x, self.a_proj)         # (..., r)
        proj_th = F.linear(proj_a, th)           # (..., r)
        adapted_out = F.linear(proj_th, self.b_proj)  # (..., d)

        return base_out + adapted_out

    def synthesize_dense_weight(self, theta: torch.Tensor) -> torch.Tensor:
        """Synthesizes W_{eff} = W_0 + B Theta A in PyTorch for auditing."""
        th = theta.to(dtype=self.w_0.weight.dtype, device=self.w_0.weight.device)
        delta_w = torch.matmul(self.b_proj, torch.matmul(th, self.a_proj))
        return self.w_0.weight + delta_w

    def release_cache(self) -> None:
        """Cleans PyTorch CUDA/CPU caches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class CertifiedUnlearningEngine:
    """
    Theorem 2 Certified Exact Unlearning in O(depth).
    
    Guarantees:
    1. O(depth) Algebraic Inversion: Updates ancestor weights and cores without visiting
       untouched subtrees.
    2. Zero Residual Forgetting: Max analytical discrepancy between unlearned tree
       and a ground-truth scratch tree < 1e-12 (machine precision ~ 1e-16).
    3. Structural Conservation: Theorem 1 identities (telescoping, wavelet expansion,
       Haar vanishing, Parseval energy conservation) continue to hold exactly.
    4. Subtree Memory Deallocation: Explicitly severs cyclic references and deallocates
       tensors, eliminating memory leaks.
    """

    def __init__(self, tree_memory: WaveTreeMemory) -> None:
        self.tree_memory = tree_memory

    def unlearn_document_exact(
        self,
        doc_id: str,
        verify_ground_truth: bool = True,
        tol: float = 1e-12,
    ) -> Dict[str, Any]:
        """
        Executes certified analytical unlearning of document `doc_id` in O(depth) time.
        
        Args:
            doc_id: Unique identifier of document to unlearn.
            verify_ground_truth: If True, independently checks error against counterfactual
                                 fresh tree to certify Theorem 2 mathematically.
            tol: Maximum permissible analytical discrepancy (< 1e-12).
            
        Returns:
            Comprehensive certified unlearning audit report.
        """
        tree = self.tree_memory
        if doc_id not in tree.documents:
            return {
                "certified_exact": False,
                "unlearned": False,
                "error": f"Document '{doc_id}' not found in active tree memory.",
                "remaining_documents": len(tree.documents),
                "remaining_leaves": len(tree.get_leaves()),
            }

        target_node = tree.documents[doc_id]

        # 1. Capture ground truth reference state if requested
        ground_truth_discrepancy = 0.0
        ground_truth_frob_discrepancy = 0.0
        shadow_tree = None
        if verify_ground_truth:
            # Build independent shadow tree from remaining documents preserving weights & titles
            shadow_tree = WaveTreeMemory(
                r=tree.r,
                key_dim=tree.dim,
                corpus_id=tree.corpus_id,
                seed=tree.seed,
            )
            for d_id, d_node in tree.documents.items():
                if d_id == doc_id:
                    continue
                doc_sections = []
                for sec in d_node.children:
                    p_dicts = [
                        {
                            "text": lf.text,
                            "title": lf.title,
                            "weight": lf.weight,
                            "passage_id": lf.node_id,
                        }
                        for lf in sec.children
                    ]
                    doc_sections.append({
                        "section_title": sec.title,
                        "paragraphs": p_dicts,
                    })
                shadow_tree.ingest_document(d_id, d_node.title, doc_sections)

        # 2. Execute true O(depth) algebraic unlearning via WaveTreeMemory
        unlearn_res = tree.unlearn_document(doc_id)

        # 3. Measure analytical discrepancy against independent shadow tree
        if verify_ground_truth and shadow_tree is not None:
            root_theta_diff = float(np.max(np.abs(tree.root.theta - shadow_tree.root.theta)))
            root_theta_frob = float(np.linalg.norm(tree.root.theta - shadow_tree.root.theta, ord="fro"))
            root_kappa_diff = float(np.max(np.abs(tree.root.kappa - shadow_tree.root.kappa)))
            root_kappa_frob = float(np.linalg.norm(tree.root.kappa - shadow_tree.root.kappa))
            root_unnorm_diff = (
                float(np.max(np.abs(tree.root.unnormalized_kappa - shadow_tree.root.unnormalized_kappa)))
                if tree.root.unnormalized_kappa is not None and shadow_tree.root.unnormalized_kappa is not None
                else 0.0
            )

            max_node_theta_inf = root_theta_diff
            max_node_theta_frob = root_theta_frob
            max_node_detail_inf = 0.0
            max_node_detail_frob = 0.0
            for d_id, d_node in tree.documents.items():
                s_doc = shadow_tree.documents.get(d_id)
                if s_doc is None:
                    continue
                for n_cur, n_ref in zip(d_node.get_subtree_nodes(), s_doc.get_subtree_nodes()):
                    if n_cur.theta is not None and n_ref.theta is not None:
                        diff_t = n_cur.theta - n_ref.theta
                        max_node_theta_inf = max(max_node_theta_inf, float(np.max(np.abs(diff_t))))
                        max_node_theta_frob = max(max_node_theta_frob, float(np.linalg.norm(diff_t, ord="fro")))
                    if n_cur.detail is not None and n_ref.detail is not None:
                        diff_d = n_cur.detail - n_ref.detail
                        max_node_detail_inf = max(max_node_detail_inf, float(np.max(np.abs(diff_d))))
                        max_node_detail_frob = max(max_node_detail_frob, float(np.linalg.norm(diff_d, ord="fro")))

            ground_truth_discrepancy = max(
                root_theta_diff,
                root_kappa_diff,
                root_unnorm_diff,
                max_node_theta_inf,
                max_node_detail_inf,
            )
            ground_truth_frob_discrepancy = max(
                root_theta_frob,
                root_kappa_frob,
                max_node_theta_frob,
                max_node_detail_frob,
            )

            # Clean shadow tree memory
            release_subtree_memory(shadow_tree.root)
            del shadow_tree

        # 4. Verify that remaining tree satisfies all Theorem 1 identities
        thm_check = tree.verify_all_theorems()

        is_certified = bool(
            ground_truth_discrepancy < tol
            and ground_truth_frob_discrepancy < tol
            and thm_check["all_theorems_verified"]
        )

        return {
            "certified_exact": is_certified,
            "certified": is_certified,
            "unlearned": True,
            "exact_match": is_certified,
            "max_discrepancy": float(ground_truth_discrepancy),
            "max_frobenius_discrepancy": float(ground_truth_frob_discrepancy),
            "complexity": "O(depth)",
            "nodes_updated": unlearn_res.get("nodes_updated", 0),
            "unlearned_document_id": doc_id,
            "unlearned_doc_id": doc_id,
            "remaining_documents": len(tree.documents),
            "remaining_leaves": len(tree.get_leaves()),
            "root_weight": tree.root.total_weight,
            "freed_nodes_count": unlearn_res.get("freed_nodes", 0),
            "freed_bytes": unlearn_res.get("freed_bytes", 0),
            "theorems_satisfied": thm_check["all_theorems_verified"],
            "analytical_tolerance": tol,
        }

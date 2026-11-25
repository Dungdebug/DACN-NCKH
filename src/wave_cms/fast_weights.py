"""
Additive Ridge Associative Memory & Ephemeral Per-Query Delta-Rule
for WaveCMS-R^2 (Proposition 3 & Theorem 3*).

100% strict offline: pure Python stdlib + NumPy + FixedPointMatrix.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from src.wave_cms.sufficient_stats import FixedPointMatrix


def _as_2d_pair(
    K: np.ndarray,
    V: np.ndarray,
    expected_dim: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and normalize (K, V) key-value batches to 2D float64 (n, d)."""
    k_arr = np.asarray(K, dtype=np.float64)
    v_arr = np.asarray(V, dtype=np.float64)

    if k_arr.ndim == 1:
        k_arr = k_arr.reshape(1, -1)
    if v_arr.ndim == 1:
        v_arr = v_arr.reshape(1, -1)

    if k_arr.ndim != 2 or v_arr.ndim != 2:
        raise ValueError(
            f"K and V must be 1D or 2D arrays, got shapes {k_arr.shape} and {v_arr.shape}"
        )
    if k_arr.shape != v_arr.shape:
        raise ValueError(
            f"K and V must have identical shapes, got {k_arr.shape} and {v_arr.shape}"
        )
    if expected_dim is not None and k_arr.shape[1] != expected_dim:
        raise ValueError(
            f"Expected feature dim {expected_dim}, got {k_arr.shape[1]}"
        )
    if k_arr.shape[1] <= 0:
        raise ValueError(f"Feature dim must be positive, got {k_arr.shape[1]}")
    if not (np.all(np.isfinite(k_arr)) and np.all(np.isfinite(v_arr))):
        raise ValueError("K and V must contain only finite values")

    return k_arr, v_arr


class RidgeAssociativeMemory:
    """
    Additive Ridge Associative Memory with bit-exact fixed-point sufficient
    statistics (G = sum k k^T, B = sum k v^T) and Cholesky closed-form readout.
    """

    def __init__(self, dim: int = 256, lam: float = 1.0, scale_bits: int = 28) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if lam <= 0.0 or not np.isfinite(lam):
            raise ValueError(f"lam must be finite and > 0, got {lam}")

        self.dim = int(dim)
        self.lam = float(lam)
        self.scale_bits = int(scale_bits)

        self.G = FixedPointMatrix(self.dim, self.dim, scale_bits=self.scale_bits)
        self.B = FixedPointMatrix(self.dim, self.dim, scale_bits=self.scale_bits)

        self._version: int = 0
        self._cached_version: int = -1
        self._cached_M: Optional[np.ndarray] = None
        self._doc_contributions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _quantize_pair_stats(
        self, K: np.ndarray, V: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        k_arr, v_arr = _as_2d_pair(K, V, expected_dim=self.dim)
        d_g = k_arr.T @ k_arr
        d_b = k_arr.T @ v_arr
        return self.G.quantize(d_g), self.B.quantize(d_b)

    def _apply_raw_delta(
        self, q_g: np.ndarray, q_b: np.ndarray, add: bool
    ) -> None:
        if add:
            self.G.add_raw(q_g)
            try:
                self.B.add_raw(q_b)
            except Exception:
                self.G.sub_raw(q_g)
                raise
        else:
            self.G.sub_raw(q_g)
            try:
                self.B.sub_raw(q_b)
            except Exception:
                self.G.add_raw(q_g)
                raise
        self._version += 1

    def add_pairs(self, K: np.ndarray, V: np.ndarray) -> None:
        """Accumulate K^T K into G and K^T V into B (bit-exact int64)."""
        q_g, q_b = self._quantize_pair_stats(K, V)
        self._apply_raw_delta(q_g, q_b, add=True)

    def remove_pairs(self, K: np.ndarray, V: np.ndarray) -> None:
        """Subtract K^T K from G and K^T V from B (bit-exact int64)."""
        q_g, q_b = self._quantize_pair_stats(K, V)
        self._apply_raw_delta(q_g, q_b, add=False)

    def add_document_pairs(self, doc_id: str, K: np.ndarray, V: np.ndarray) -> None:
        """Add key-value pairs for a document and track its exact contribution."""
        if not doc_id:
            raise ValueError("doc_id must be a non-empty string")
        if doc_id in self._doc_contributions:
            raise ValueError(
                f"Document {doc_id!r} already exists; remove it before re-adding"
            )
        q_g, q_b = self._quantize_pair_stats(K, V)
        self._apply_raw_delta(q_g, q_b, add=True)
        self._doc_contributions[doc_id] = (q_g, q_b)

    def remove_document(self, doc_id: str) -> bool:
        """Bit-exact 1-line unlearning of a previously added document."""
        contrib = self._doc_contributions.get(doc_id)
        if contrib is None:
            return False
        q_g, q_b = contrib
        self._apply_raw_delta(q_g, q_b, add=False)
        del self._doc_contributions[doc_id]
        return True

    def solve(self) -> np.ndarray:
        """
        Solve M* = B^T (G + lam * I)^{-1} via Cholesky decomposition.
        Result is cached per self._version.
        """
        if self._cached_M is not None and self._cached_version == self._version:
            return self._cached_M.copy()

        g_float = self.G.to_float()
        # Symmetrize to guard against 1-ULP asymmetry from floating-point matmul.
        g_sym = 0.5 * (g_float + g_float.T)
        a_mat = g_sym + self.lam * np.eye(self.dim, dtype=np.float64)
        b_float = self.B.to_float()

        # Solve (G + lam I) M*^T = B  <=>  L L^T M*^T = B
        chol = np.linalg.cholesky(a_mat)
        y_mat = np.linalg.solve(chol, b_float)
        m_t = np.linalg.solve(chol.T, y_mat)
        m_star = m_t.T

        self._cached_M = m_star
        self._cached_version = self._version
        return m_star.copy()

    def predict(self, query_vec: np.ndarray) -> np.ndarray:
        """Return closed-form hop-2 PRF vector v_hat = M* q."""
        q = np.asarray(query_vec, dtype=np.float64)
        m_star = self.solve()
        if q.ndim == 1:
            if q.shape[0] != self.dim:
                raise ValueError(
                    f"Expected query_vec of dim {self.dim}, got {q.shape}"
                )
            return m_star @ q
        if q.ndim == 2:
            if q.shape[1] != self.dim:
                raise ValueError(
                    f"Expected query_vec of shape (n, {self.dim}), got {q.shape}"
                )
            return q @ m_star.T
        raise ValueError(
            f"query_vec must be 1D or 2D, got {q.ndim}D with shape {q.shape}"
        )

    @staticmethod
    def ephemeral_delta(
        K: np.ndarray,
        V: np.ndarray,
        lam: float = 1.0,
        beta: Optional[float] = None,
        epochs: int = 500,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """
        Run ephemeral per-query delta-rule:
            M_{t+1} = M_t + beta * [(V^T - M_t K^T) K - lam * M_t]
        Does not touch any persistent memory state; converges to ridge solution M*.
        """
        if lam <= 0.0 or not np.isfinite(lam):
            raise ValueError(f"lam must be finite and > 0, got {lam}")
        if epochs < 0:
            raise ValueError(f"epochs must be non-negative, got {epochs}")
        if tol < 0.0 or not np.isfinite(tol):
            raise ValueError(f"tol must be finite and >= 0, got {tol}")

        k_arr, v_arr = _as_2d_pair(K, V)
        dim = k_arr.shape[1]

        if beta is None:
            g_mat = k_arr.T @ k_arr
            g_sym = 0.5 * (g_mat + g_mat.T)
            eigs = np.linalg.eigvalsh(g_sym)
            mu_min = max(lam, float(eigs[0]) + lam)
            mu_max = max(mu_min, float(eigs[-1]) + lam)
            step = 2.0 / (mu_min + mu_max)
        else:
            step = float(beta)
            if step <= 0.0 or not np.isfinite(step):
                raise ValueError(f"beta must be finite and > 0, got {beta}")

        m_curr = np.zeros((dim, dim), dtype=np.float64)
        for _ in range(epochs):
            grad_step = (v_arr.T - m_curr @ k_arr.T) @ k_arr - lam * m_curr
            m_next = m_curr + step * grad_step
            if np.max(np.abs(m_next - m_curr)) <= tol:
                m_curr = m_next
                break
            m_curr = m_next

        return m_curr

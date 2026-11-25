"""
Unit tests for src/wave_cms/fast_weights.py (Ticket T4 / Proposition 3 & Theorem 3*).
Verifies:
1. Proposition 3: ephemeral_delta converges to RidgeAssociativeMemory.solve() within <= 1e-6.
2. Theorem 3*: add_document_pairs(D) -> remove_document(D) restores (G, B) and M* bit-exact.
3. Ephemeral delta does not mutate persistent memory state.
4. Caching, predict(), permutation invariance, and input validation.
"""

import numpy as np
import pytest

from src.wave_cms.fast_weights import RidgeAssociativeMemory


class TestProposition3Convergence:
    def test_ephemeral_delta_converges_to_ridge_solve_default_epochs(self):
        rng = np.random.default_rng(42)
        dim = 32
        n_pairs = 24
        # Normalized keys keep condition number moderate so 500 epochs hits <= 1e-6.
        K = rng.normal(0.0, 1.0, size=(n_pairs, dim))
        K /= np.linalg.norm(K, axis=1, keepdims=True)
        V = rng.normal(0.0, 1.0, size=(n_pairs, dim))

        mem = RidgeAssociativeMemory(dim=dim, lam=1.0, scale_bits=28)
        mem.add_pairs(K, V)
        M_solve = mem.solve()

        M_delta = RidgeAssociativeMemory.ephemeral_delta(K, V, lam=1.0)
        max_err = float(np.max(np.abs(M_delta - M_solve)))
        assert max_err <= 1e-6

        q = rng.normal(0.0, 1.0, size=(dim,))
        np.testing.assert_allclose(M_delta @ q, mem.predict(q), atol=1e-6, rtol=0.0)

    def test_ephemeral_delta_with_explicit_beta(self):
        rng = np.random.default_rng(123)
        dim = 16
        K = rng.normal(0.0, 0.5, size=(12, dim))
        V = rng.normal(0.0, 1.0, size=(12, dim))
        lam = 0.5

        mem = RidgeAssociativeMemory(dim=dim, lam=lam, scale_bits=28)
        mem.add_pairs(K, V)
        M_solve = mem.solve()

        M_delta = RidgeAssociativeMemory.ephemeral_delta(
            K, V, lam=lam, beta=0.05, epochs=1000, tol=1e-10
        )
        assert float(np.max(np.abs(M_delta - M_solve))) <= 1e-6


class TestTheorem3StarUnlearning:
    def test_add_then_remove_document_bit_exact(self):
        rng = np.random.default_rng(7)
        dim = 24
        mem = RidgeAssociativeMemory(dim=dim, lam=1.0, scale_bits=28)

        K1 = rng.normal(0.0, 1.0, size=(10, dim))
        V1 = rng.normal(0.0, 1.0, size=(10, dim))
        K2 = rng.normal(0.0, 1.0, size=(8, dim))
        V2 = rng.normal(0.0, 1.0, size=(8, dim))

        mem.add_document_pairs("doc_base", K1, V1)
        g_before = mem.G.copy()
        b_before = mem.B.copy()
        m_before = mem.solve().copy()

        mem.add_document_pairs("doc_target", K2, V2)
        assert mem.G != g_before
        assert mem.B != b_before
        assert not np.array_equal(mem.solve(), m_before)

        removed = mem.remove_document("doc_target")
        assert removed is True

        assert mem.G == g_before
        assert mem.B == b_before
        assert np.array_equal(mem.G.data, g_before.data)
        assert np.array_equal(mem.B.data, b_before.data)
        assert np.array_equal(mem.solve(), m_before)

        # Removing an already-removed or unknown doc returns False without mutation.
        version_before = mem._version
        assert mem.remove_document("doc_target") is False
        assert mem.remove_document("nonexistent") is False
        assert mem._version == version_before

    def test_permutation_invariance_bit_exact(self):
        rng = np.random.default_rng(99)
        dim = 16
        docs = [
            (
                f"doc_{i}",
                rng.normal(0.0, 1.0, size=(5, dim)),
                rng.normal(0.0, 1.0, size=(5, dim)),
            )
            for i in range(6)
        ]

        mem_fwd = RidgeAssociativeMemory(dim=dim, lam=1.0, scale_bits=28)
        for doc_id, k_d, v_d in docs:
            mem_fwd.add_document_pairs(doc_id, k_d, v_d)

        mem_perm = RidgeAssociativeMemory(dim=dim, lam=1.0, scale_bits=28)
        for idx in rng.permutation(len(docs)):
            doc_id, k_d, v_d = docs[int(idx)]
            mem_perm.add_document_pairs(doc_id, k_d, v_d)

        assert mem_fwd.G == mem_perm.G
        assert mem_fwd.B == mem_perm.B
        assert np.array_equal(mem_fwd.solve(), mem_perm.solve())


class TestEphemeralIsolationAndCache:
    def test_ephemeral_delta_does_not_mutate_persistent_state(self):
        rng = np.random.default_rng(2026)
        dim = 16
        mem = RidgeAssociativeMemory(dim=dim, lam=1.0, scale_bits=28)
        K_base = rng.normal(0.0, 1.0, size=(6, dim))
        V_base = rng.normal(0.0, 1.0, size=(6, dim))
        mem.add_pairs(K_base, V_base)

        g_snap = mem.G.copy()
        b_snap = mem.B.copy()
        v_snap = mem._version
        m_snap = mem.solve().copy()

        K_eph = rng.normal(0.0, 1.0, size=(8, dim))
        V_eph = rng.normal(0.0, 1.0, size=(8, dim))
        M_eph = RidgeAssociativeMemory.ephemeral_delta(K_eph, V_eph, lam=1.0)
        assert M_eph.shape == (dim, dim)

        assert mem.G == g_snap
        assert mem.B == b_snap
        assert mem._version == v_snap
        assert np.array_equal(mem.solve(), m_snap)

    def test_solve_cache_invalidation_and_add_remove_pairs(self):
        rng = np.random.default_rng(11)
        dim = 8
        mem = RidgeAssociativeMemory(dim=dim, lam=1.0, scale_bits=28)

        m0 = mem.solve()
        assert np.array_equal(m0, np.zeros((dim, dim), dtype=np.float64))
        assert mem._cached_version == mem._version

        k = rng.normal(0.0, 1.0, size=(dim,))
        v = rng.normal(0.0, 1.0, size=(dim,))
        mem.add_pairs(k, v)
        assert mem._version == 1
        m1 = mem.solve()
        assert mem._cached_version == 1
        assert not np.array_equal(m1, m0)

        # Batch predict
        q_batch = rng.normal(0.0, 1.0, size=(4, dim))
        preds = mem.predict(q_batch)
        assert preds.shape == (4, dim)
        np.testing.assert_allclose(preds[0], mem.predict(q_batch[0]))

        mem.remove_pairs(k, v)
        assert mem._version == 2
        assert np.array_equal(mem.solve(), m0)

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            RidgeAssociativeMemory(dim=0)
        with pytest.raises(ValueError):
            RidgeAssociativeMemory(dim=8, lam=0.0)

        mem = RidgeAssociativeMemory(dim=8, lam=1.0)
        with pytest.raises(ValueError):
            mem.add_pairs(np.zeros((2, 7)), np.zeros((2, 7)))
        with pytest.raises(ValueError):
            mem.add_pairs(np.zeros((2, 8)), np.zeros((3, 8)))
        with pytest.raises(ValueError):
            mem.predict(np.zeros((7,)))

        mem.add_document_pairs("d1", np.ones((1, 8)), np.ones((1, 8)))
        with pytest.raises(ValueError):
            mem.add_document_pairs("d1", np.ones((1, 8)), np.ones((1, 8)))

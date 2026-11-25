"""
Unit tests for src/wave_cms/sufficient_stats.py (Ticket T1 / Theorem 3*).
Verifies bit-exact add/sub invertibility, permutation invariance, overflow guards,
and BM25 IDF calculation.
"""

from collections import Counter
import math

import numpy as np
import pytest

from src.wave_cms.sufficient_stats import (
    FixedPointMatrix,
    FixedPointVector,
    TermCounter,
)


class TestFixedPointVector:
    def test_add_sub_bit_exact_identity(self):
        rng = np.random.default_rng(42)
        base_docs = [rng.normal(0.0, 1.0, size=64) for _ in range(5)]
        target_doc = rng.normal(0.0, 2.5, size=64)

        vec = FixedPointVector(dim=64, scale_bits=30)
        for doc in base_docs:
            vec.add(doc)
        snapshot_before = vec.copy()

        vec.add(target_doc)
        assert not (vec == snapshot_before)
        vec.sub(target_doc)

        assert vec == snapshot_before
        assert np.array_equal(vec.data, snapshot_before.data)

    def test_permutation_invariance_10_docs_bit_exact(self):
        rng = np.random.default_rng(123)
        docs = [rng.normal(0.0, 3.0, size=32) for _ in range(10)]

        v_forward = FixedPointVector(dim=32, scale_bits=30)
        for d in docs:
            v_forward.add(d)

        perm = rng.permutation(10)
        v_perm = FixedPointVector(dim=32, scale_bits=30)
        for idx in perm:
            v_perm.add(docs[int(idx)])

        assert v_forward == v_perm
        assert np.array_equal(v_forward.data, v_perm.data)

    def test_overflow_raises_and_preserves_state(self):
        vec = FixedPointVector(dim=4, scale_bits=30)
        # Quantize overflow
        with pytest.raises(OverflowError):
            vec.quantize(np.array([1e12, 0.0, 0.0, 0.0]))
        with pytest.raises(OverflowError):
            vec.quantize(np.array([np.inf, 0.0, 0.0, 0.0]))

        # Raw addition overflow (positive and negative)
        max_val = np.iinfo(np.int64).max
        min_val = np.iinfo(np.int64).min
        vec.add_raw(np.array([max_val - 5, min_val + 5, 0, 0], dtype=np.int64))
        before = vec.copy()

        with pytest.raises(OverflowError):
            vec.add_raw(np.array([6, 0, 0, 0], dtype=np.int64))
        assert vec == before

        with pytest.raises(OverflowError):
            vec.add_raw(np.array([0, -6, 0, 0], dtype=np.int64))
        assert vec == before

        with pytest.raises(OverflowError):
            vec.sub_raw(np.array([-6, 0, 0, 0], dtype=np.int64))
        assert vec == before

        with pytest.raises(OverflowError):
            vec.sub_raw(np.array([0, 6, 0, 0], dtype=np.int64))
        assert vec == before

        with pytest.raises(OverflowError):
            vec.sub_raw(np.array([0, 0, min_val, 0], dtype=np.int64))
        assert vec == before

    def test_to_float_accuracy(self):
        vec = FixedPointVector(dim=3, scale_bits=30)
        arr = np.array([1.25, -0.5, 3.0])
        vec.add(arr)
        np.testing.assert_allclose(vec.to_float(), arr, atol=1e-8)


class TestFixedPointMatrix:
    def test_add_sub_bit_exact_identity(self):
        rng = np.random.default_rng(7)
        base_mats = [rng.normal(0.0, 1.0, size=(8, 8)) for _ in range(4)]
        target = rng.normal(0.0, 2.0, size=(8, 8))

        mat = FixedPointMatrix(rows=8, cols=8, scale_bits=30)
        for m in base_mats:
            mat.add(m)
        snapshot = mat.copy()

        mat.add(target)
        assert not (mat == snapshot)
        mat.sub(target)

        assert mat == snapshot
        assert np.array_equal(mat.data, snapshot.data)

    def test_permutation_invariance_10_docs_bit_exact(self):
        rng = np.random.default_rng(99)
        mats = [rng.normal(0.0, 1.5, size=(6, 4)) for _ in range(10)]

        m1 = FixedPointMatrix(rows=6, cols=4, scale_bits=30)
        for m in mats:
            m1.add(m)

        m2 = FixedPointMatrix(rows=6, cols=4, scale_bits=30)
        for idx in rng.permutation(10):
            m2.add(mats[int(idx)])

        assert m1 == m2
        assert np.array_equal(m1.data, m2.data)

    def test_overflow_raises(self):
        mat = FixedPointMatrix(rows=2, cols=2, scale_bits=30)
        max_val = np.iinfo(np.int64).max
        mat.add_raw(np.array([[max_val, 0], [0, 0]], dtype=np.int64))
        before = mat.copy()

        with pytest.raises(OverflowError):
            mat.add_raw(np.array([[1, 0], [0, 0]], dtype=np.int64))
        assert mat == before


class TestTermCounter:
    def test_add_sub_exact_and_zero_pruning(self):
        tc = TermCounter()
        d1 = Counter(["alpha", "beta", "alpha"])
        d2 = Counter(["beta", "gamma"])

        tc.add(d1)
        snap = tc.copy()

        tc.add(d2)
        assert tc.count("beta") == 2
        assert tc.count("gamma") == 1
        assert tc.nonzero("gamma")

        tc.sub(d2)
        assert tc == snap
        assert tc.count("gamma") == 0
        assert not tc.nonzero("gamma")
        assert "gamma" not in tc._counts

    def test_presence_and_permutation_invariance_10_docs(self):
        docs = [
            ["t1", "t1", "t2"],
            ["t2", "t3"],
            ["t1", "t4", "t4"],
            ["t5"],
            ["t1", "t2", "t5"],
            ["t6", "t6"],
            ["t3", "t7"],
            ["t8"],
            ["t1", "t9"],
            ["t10", "t2"],
        ]

        tc1 = TermCounter()
        for doc in docs:
            tc1.add_presence(doc)

        rng = np.random.default_rng(2026)
        tc2 = TermCounter()
        for idx in rng.permutation(len(docs)):
            tc2.add_presence(docs[int(idx)])

        assert tc1 == tc2
        assert tc1.count("t1") == 4
        assert tc1.count("t6") == 1

        snap = tc1.copy()
        tc1.add_presence(["t1", "t11", "t11"])
        assert tc1.count("t11") == 1
        tc1.sub_presence(["t1", "t11", "t11"])
        assert tc1 == snap
        assert not tc1.nonzero("t11")

    def test_negative_prevention(self):
        tc = TermCounter()
        tc.add(Counter({"x": 1}))
        snap = tc.copy()

        with pytest.raises(ValueError):
            tc.sub(Counter({"x": 2}))
        assert tc == snap

        with pytest.raises(ValueError):
            tc.sub_presence(["x", "missing"])
        assert tc == snap

    def test_idf_matches_bm25_formula(self):
        tc = TermCounter()
        tc.add_presence(["apple", "banana"])
        tc.add_presence(["apple", "cherry"])
        tc.add_presence(["apple"])

        n_docs = 10
        for term, df in [("apple", 3), ("banana", 1), ("unseen", 0)]:
            expected = math.log((n_docs - df + 0.5) / (df + 0.5)) + 1.0
            assert tc.idf(term, n_docs) == pytest.approx(expected, rel=1e-12, abs=1e-12)

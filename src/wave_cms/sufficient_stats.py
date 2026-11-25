"""
Bit-exact fixed-point sufficient statistics and exact integer term counters
for WaveCMS-R^2 (Theorem 3*: order-independent & bit-exact unlearning).

100% strict offline: pure Python stdlib + NumPy.
"""

from __future__ import annotations

import math
from typing import Counter, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

_INT64_INFO = np.iinfo(np.int64)
_INT64_MIN = int(_INT64_INFO.min)
_INT64_MAX = int(_INT64_INFO.max)
# 2**63 in float64 is exact; any value >= 2**63 or < -2**63 overflows int64.
_FLOAT64_UPPER_BOUND = float(2**63)
_FLOAT64_LOWER_BOUND = -float(2**63)


def _validate_scale_bits(scale_bits: int) -> int:
    sb = int(scale_bits)
    if sb < 0 or sb > 62:
        raise ValueError(f"scale_bits must be in [0, 62], got {scale_bits}")
    return sb


def _quantize_array(
    arr: np.ndarray,
    expected_shape: Tuple[int, ...],
    scale: float,
) -> np.ndarray:
    arr_f = np.asarray(arr, dtype=np.float64)
    if arr_f.shape != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape}, got {arr_f.shape}"
        )
    if not np.all(np.isfinite(arr_f)):
        raise OverflowError("Cannot quantize non-finite values (NaN or Inf)")
    scaled = np.rint(arr_f * scale)
    if np.any(scaled >= _FLOAT64_UPPER_BOUND) or np.any(scaled < _FLOAT64_LOWER_BOUND):
        raise OverflowError("Quantized value exceeds int64 range")
    return scaled.astype(np.int64)


def _validate_raw_int64(q_arr: np.ndarray, expected_shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(q_arr)
    if arr.shape != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape}, got {arr.shape}"
        )
    if arr.dtype != np.int64:
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f"Expected integer array, got {arr.dtype}")
        arr = arr.astype(np.int64)
    return arr


def _checked_add_int64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    overflow_pos = (b > 0) & (a > (_INT64_MAX - b))
    overflow_neg = (b < 0) & (a < (_INT64_MIN - b))
    if np.any(overflow_pos | overflow_neg):
        raise OverflowError("int64 addition overflow")
    return a + b


def _checked_sub_int64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b_min = b == _INT64_MIN
    overflow_min = b_min & (a >= 0)
    b_safe = np.where(b_min, 0, b)
    overflow_pos = (b_safe > 0) & (a < (_INT64_MIN + b_safe))
    overflow_neg = (b_safe < 0) & (a > (_INT64_MAX + b_safe))
    if np.any(overflow_min | overflow_pos | overflow_neg):
        raise OverflowError("int64 subtraction overflow")
    return a - b


class FixedPointVector:
    """Bit-exact fixed-point int64 accumulator for 1D vectors."""

    def __init__(self, dim: int, scale_bits: int = 30) -> None:
        if dim < 0:
            raise ValueError(f"dim must be non-negative, got {dim}")
        self.dim = int(dim)
        self.scale_bits = _validate_scale_bits(scale_bits)
        self.scale = float(1 << self.scale_bits)
        self.data = np.zeros((self.dim,), dtype=np.int64)

    def quantize(self, arr: np.ndarray) -> np.ndarray:
        return _quantize_array(arr, (self.dim,), self.scale)

    def add_raw(self, q_vec: np.ndarray) -> None:
        q = _validate_raw_int64(q_vec, (self.dim,))
        self.data = _checked_add_int64(self.data, q)

    def sub_raw(self, q_vec: np.ndarray) -> None:
        q = _validate_raw_int64(q_vec, (self.dim,))
        self.data = _checked_sub_int64(self.data, q)

    def add(self, vec: np.ndarray) -> None:
        self.add_raw(self.quantize(vec))

    def sub(self, vec: np.ndarray) -> None:
        self.sub_raw(self.quantize(vec))

    def to_float(self) -> np.ndarray:
        return self.data.astype(np.float64) / self.scale

    def copy(self) -> "FixedPointVector":
        cloned = FixedPointVector(self.dim, self.scale_bits)
        cloned.data = self.data.copy()
        return cloned

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FixedPointVector):
            return NotImplemented
        return (
            self.dim == other.dim
            and self.scale_bits == other.scale_bits
            and np.array_equal(self.data, other.data)
        )


class FixedPointMatrix:
    """Bit-exact fixed-point int64 accumulator for 2D matrices (e.g., G, B)."""

    def __init__(self, rows: int, cols: int, scale_bits: int = 30) -> None:
        if rows < 0 or cols < 0:
            raise ValueError(f"rows and cols must be non-negative, got ({rows}, {cols})")
        self.rows = int(rows)
        self.cols = int(cols)
        self.shape = (self.rows, self.cols)
        self.scale_bits = _validate_scale_bits(scale_bits)
        self.scale = float(1 << self.scale_bits)
        self.data = np.zeros(self.shape, dtype=np.int64)

    def quantize(self, arr: np.ndarray) -> np.ndarray:
        return _quantize_array(arr, self.shape, self.scale)

    def add_raw(self, q_mat: np.ndarray) -> None:
        q = _validate_raw_int64(q_mat, self.shape)
        self.data = _checked_add_int64(self.data, q)

    def sub_raw(self, q_mat: np.ndarray) -> None:
        q = _validate_raw_int64(q_mat, self.shape)
        self.data = _checked_sub_int64(self.data, q)

    def add(self, mat: np.ndarray) -> None:
        self.add_raw(self.quantize(mat))

    def sub(self, mat: np.ndarray) -> None:
        self.sub_raw(self.quantize(mat))

    def to_float(self) -> np.ndarray:
        return self.data.astype(np.float64) / self.scale

    def copy(self) -> "FixedPointMatrix":
        cloned = FixedPointMatrix(self.rows, self.cols, self.scale_bits)
        cloned.data = self.data.copy()
        return cloned

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FixedPointMatrix):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.scale_bits == other.scale_bits
            and np.array_equal(self.data, other.data)
        )


class TermCounter:
    """Exact integer term-frequency / document-frequency counter with zero-pruning."""

    def __init__(self, initial: Optional[Mapping[str, int]] = None) -> None:
        self._counts: Dict[str, int] = {}
        if initial is not None:
            self.add(initial)

    def add(self, counts: Counter[str] | Mapping[str, int]) -> None:
        updates: Dict[str, int] = {}
        for term, delta in counts.items():
            d = int(delta)
            if d < 0:
                raise ValueError(f"Cannot add negative count {d} for term {term!r}")
            if d > 0:
                updates[term] = d
        for term, d in updates.items():
            self._counts[term] = self._counts.get(term, 0) + d

    def sub(self, counts: Counter[str] | Mapping[str, int]) -> None:
        updates: Dict[str, int] = {}
        for term, delta in counts.items():
            d = int(delta)
            if d < 0:
                raise ValueError(f"Cannot subtract negative count {d} for term {term!r}")
            if d == 0:
                continue
            cur = self._counts.get(term, 0)
            if cur < d:
                raise ValueError(
                    f"Cannot subtract {d} for term {term!r}: current count is {cur}"
                )
            updates[term] = cur - d
        for term, new_val in updates.items():
            if new_val == 0:
                self._counts.pop(term, None)
            else:
                self._counts[term] = new_val

    def add_presence(self, terms: Iterable[str]) -> None:
        for term in set(terms):
            self._counts[term] = self._counts.get(term, 0) + 1

    def sub_presence(self, terms: Iterable[str]) -> None:
        unique_terms = set(terms)
        for term in unique_terms:
            if self._counts.get(term, 0) < 1:
                raise ValueError(
                    f"Cannot subtract presence for term {term!r}: current count is 0"
                )
        for term in unique_terms:
            new_val = self._counts[term] - 1
            if new_val == 0:
                del self._counts[term]
            else:
                self._counts[term] = new_val

    def count(self, term: str) -> int:
        return self._counts.get(term, 0)

    def nonzero(self, term: str) -> bool:
        return self._counts.get(term, 0) > 0

    def idf(self, term: str, n_docs: int) -> float:
        if n_docs < 0:
            raise ValueError(f"n_docs must be non-negative, got {n_docs}")
        df_t = self.count(term)
        numerator = n_docs - df_t + 0.5
        if numerator <= 0.0:
            raise ValueError(
                f"Invalid BM25 IDF domain: n_docs={n_docs}, df_t={df_t}"
            )
        return math.log(numerator / (df_t + 0.5)) + 1.0

    def copy(self) -> "TermCounter":
        cloned = TermCounter()
        cloned._counts = self._counts.copy()
        return cloned

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TermCounter):
            return NotImplemented
        return self._counts == other._counts

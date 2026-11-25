"""
Tests for T7 (w7-symbolic): src/reasoning/symbolic_solver.py
"""

from fractions import Fraction
import math
import pytest

from src.reasoning.symbolic_solver import (
    affine,
    balance_equation,
    caesar,
    extract_numbers,
    rsa_toy,
    safe_eval,
    solve_linear_system,
    solve_quadratic,
    verify_numeric_provenance,
    vigenere,
)


# ---------------------------------------------------------------------------
# 1. AST Security & DoS Protection (>= 15 malicious payloads)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        "__import__('os').system('echo pwned')",
        "__import__('os')",
        "().__class__.__base__.__subclasses__()",
        "lambda x: x + 1",
        "[x for x in range(10)]",
        "{x: x for x in range(5)}",
        "(x for x in range(5))",
        "9**9**9",
        "2**1001",
        "100000**1000",
        "open('/etc/passwd')",
        "eval('1+1')",
        "exec('import os')",
        "getattr(int, '__abs__')",
        "__builtins__",
        "x.__dict__",
        "a[0]",
        "True + 1",
        "'hello'",
    ],
)
def test_safe_eval_blocks_malicious_and_dos_payloads(payload: str) -> None:
    with pytest.raises(ValueError):
        safe_eval(payload, env={"x": 1, "a": 2})


def test_safe_eval_blocks_dunder_env_and_allows_valid_math() -> None:
    with pytest.raises(ValueError):
        safe_eval("__secret__", env={"__secret__": 42})

    assert safe_eval("2 + 3 * 4") == Fraction(14)
    assert safe_eval("1 / 3 + 1 / 6") == Fraction(1, 2)
    assert safe_eval("gcd(24, 36)") == Fraction(12)
    assert safe_eval("round(3.6)") == Fraction(4)
    assert safe_eval("abs(-7)") == Fraction(7)
    assert safe_eval("sqrt(16) + log10(100)", env={"k": 1}) == pytest.approx(6.0)
    assert safe_eval("sin(0) + cos(0)") == pytest.approx(1.0)
    assert safe_eval("2 * x + y", env={"x": 3, "y": Fraction(1, 2)}) == Fraction(13, 2)


# ---------------------------------------------------------------------------
# 2. Quadratic & Linear System Solvers
# ---------------------------------------------------------------------------

def test_solve_quadratic_exact_vieta_and_complex() -> None:
    res = solve_quadratic(2, -7, 3)
    assert res["delta"] == Fraction(25)
    assert res["nature"] == "two_real"
    assert res["x1"] == Fraction(3)
    assert res["x2"] == Fraction(1, 2)
    assert float(res["x2"]) == pytest.approx(0.5)
    assert res["S"] == Fraction(7, 2)
    assert float(res["S"]) == pytest.approx(3.5)
    assert res["P"] == Fraction(3, 2)
    assert float(res["P"]) == pytest.approx(1.5)

    # Double root: x^2 - 4x + 4 = 0
    dbl = solve_quadratic(1, -4, 4)
    assert dbl["delta"] == 0
    assert dbl["nature"] == "double_real"
    assert dbl["x1"] == Fraction(2)

    # Complex roots: x^2 + 2x + 5 = 0 -> -1 +- 2i
    cplx = solve_quadratic(1, 2, 5)
    assert cplx["delta"] == Fraction(-16)
    assert cplx["nature"] == "two_complex"
    assert cplx["x1"] == complex(-1.0, 2.0)
    assert cplx["x2"] == complex(-1.0, -2.0)


def test_solve_linear_system_exact_fraction() -> None:
    # 2x + y = 5, x - y = 1 -> x = 2, y = 1
    res = solve_linear_system([[2, 1, 5], [1, -1, 1]])
    assert res["status"] == "unique"
    assert res["solution"] == [Fraction(2), Fraction(1)]

    # Inconsistent
    inc = solve_linear_system([[1, 1, 2], [1, 1, 3]])
    assert inc["status"] == "no_solution"

    # Infinite solutions
    inf = solve_linear_system([[1, 2, 3], [2, 4, 6]])
    assert inf["status"] == "infinite"


# ---------------------------------------------------------------------------
# 3. Chemical Equation Balancer
# ---------------------------------------------------------------------------

def test_balance_chemical_equations() -> None:
    eq1 = balance_equation("Fe + O2 -> Fe2O3")
    assert eq1["coefficients"] == [4, 3, 2]
    assert eq1["balanced"] == "4Fe + 3O2 -> 2Fe2O3"

    eq2 = balance_equation("Fe + HNO3 -> Fe(NO3)3 + NO + H2O")
    assert eq2["coefficients"] == [1, 4, 1, 1, 2]
    assert eq2["balanced"] == "Fe + 4HNO3 -> Fe(NO3)3 + NO + 2H2O"

    eq3 = balance_equation("KMnO4 + HCl -> KCl + MnCl2 + Cl2 + H2O")
    assert eq3["coefficients"] == [2, 16, 2, 2, 5, 8]
    assert eq3["balanced"] == "2KMnO4 + 16HCl -> 2KCl + 2MnCl2 + 5Cl2 + 8H2O"


# ---------------------------------------------------------------------------
# 4. Classical & Basic Cryptography Round-Trips
# ---------------------------------------------------------------------------

def test_crypto_round_trips() -> None:
    msg = "WaveCMS-R2 Test 123!"

    c_enc = caesar(msg, 5)
    assert caesar(c_enc, 5, decrypt=True) == msg

    v_enc = vigenere(msg, "KEY")
    assert vigenere(v_enc, "KEY", decrypt=True) == msg

    a_enc = affine(msg, 5, 8)
    assert affine(a_enc, 5, 8, decrypt=True) == msg

    with pytest.raises(ValueError):
        affine(msg, 2, 5)  # gcd(2, 26) != 1

    # RSA toy: p=61, q=53 -> n=3233, phi=3120, e=17, d=2753
    n, e, d = 3233, 17, 2753
    plaintext = 65
    ciphertext = rsa_toy(plaintext, e, n)
    assert ciphertext == 2790
    assert rsa_toy(ciphertext, d, n) == plaintext


# ---------------------------------------------------------------------------
# 5. Numeric Extraction & Provenance Verification (§3.5)
# ---------------------------------------------------------------------------

def test_extract_numbers_and_provenance() -> None:
    text = "[1] Bước 1: Giá trị đo là 3,14 và 1,000.00 tại Ví dụ 2.1 (chiếm 50%)."
    nums = extract_numbers(text)
    assert nums == {"3.14", "1000", "50"}

    ok, ungrounded = verify_numeric_provenance(
        answer="Theo tài liệu [2], kết quả là 3.14 và 1000, tổng 1003.14.",
        evidence_texts=["[1] Bước 1: Đo được 3,14 và 1.000,00."],
        solver_trace=["safe_eval('3.14 + 1000') = 1003.14"],
    )
    assert ok is True
    assert ungrounded == []

    # Fabricated number 999 must be caught!
    ok_bad, ungrounded_bad = verify_numeric_provenance(
        answer="Kết quả là 3.14 nhưng có thêm 999.",
        evidence_texts=["Giá trị là 3,14."],
        solver_trace=[],
    )
    assert ok_bad is False
    assert ungrounded_bad == ["999"]

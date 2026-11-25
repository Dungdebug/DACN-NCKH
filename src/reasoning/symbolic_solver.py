"""
Offline Symbolic Solver & Numeric Provenance Verifier for WaveCMS-R^2 (T7).

100% offline, Python stdlib only (ast, math, fractions.Fraction, re).
Never uses eval(), exec(), or sympy.parse_expr on raw strings.
"""

from __future__ import annotations

import ast
import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, List, Optional, Set, Tuple, Union

Number = Union[Fraction, float]

# ---------------------------------------------------------------------------
# Constants & Safe AST Evaluator
# ---------------------------------------------------------------------------

_MAX_EXPONENT = 1000
_MAX_INT_DIGITS = 4000

_SAFE_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _check_int_digits(val: Union[int, Fraction]) -> None:
    """Reject integers or Fraction components exceeding _MAX_INT_DIGITS digits."""
    if isinstance(val, Fraction):
        _check_int_digits(val.numerator)
        _check_int_digits(val.denominator)
        return
    # Fast O(1) bit-length upper bound before converting to decimal string
    # log10(2) ~= 0.3010299956639812
    abs_val = abs(val)
    if abs_val.bit_length() * 0.30103 > _MAX_INT_DIGITS + 1:
        if len(str(abs_val)) > _MAX_INT_DIGITS:
            raise ValueError(
                f"Integer result exceeds maximum allowed {_MAX_INT_DIGITS} digits."
            )


def _to_number(val: Any) -> Number:
    """Normalize int/Fraction/float into Fraction or float, rejecting bools."""
    if isinstance(val, bool):
        raise ValueError("Boolean values are not allowed in arithmetic expressions.")
    if isinstance(val, Fraction):
        _check_int_digits(val)
        return val
    if isinstance(val, int):
        _check_int_digits(val)
        return Fraction(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise ValueError("NaN or Infinite float values are not allowed.")
        return val
    raise ValueError(f"Unsupported numeric type: {type(val).__name__}")


def _to_fraction(val: Union[int, float, Fraction, str]) -> Fraction:
    """Convert int, float, Fraction, or numeric string to exact Fraction."""
    if isinstance(val, bool):
        raise ValueError("Boolean values are not allowed.")
    if isinstance(val, Fraction):
        return val
    if isinstance(val, int):
        return Fraction(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise ValueError("NaN or Infinite float values are not allowed.")
        return Fraction(str(val))
    if isinstance(val, str):
        return Fraction(val.strip())
    raise ValueError(f"Cannot convert {type(val).__name__} to Fraction.")


def _safe_pow(base: Number, exp: Number) -> Number:
    """Compute base ** exp with exponent and digit-count DoS protection."""
    if abs(float(exp)) > _MAX_EXPONENT:
        raise ValueError(
            f"Exponent {exp} exceeds maximum allowed magnitude {_MAX_EXPONENT}."
        )

    if isinstance(base, Fraction) and isinstance(exp, Fraction) and exp.denominator == 1:
        e_int = exp.numerator
        if abs(e_int) > _MAX_EXPONENT:
            raise ValueError(
                f"Exponent {e_int} exceeds maximum allowed magnitude {_MAX_EXPONENT}."
            )
        # Pre-check digit growth before computing huge powers
        for comp in (abs(base.numerator), abs(base.denominator)):
            if comp > 1 and abs(e_int) * math.log10(comp) > _MAX_INT_DIGITS:
                raise ValueError(
                    f"Power result would exceed {_MAX_INT_DIGITS} digits."
                )
        if base == 0 and e_int < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power.")
        res = base ** e_int
        _check_int_digits(res)
        return res

    # Fractional or float exponent -> float evaluation
    b_f = float(base)
    e_f = float(exp)
    if b_f < 0 and not e_f.is_integer():
        raise ValueError("Negative base with non-integer exponent produces complex result.")
    res_f = b_f ** e_f
    if isinstance(res_f, complex) or math.isnan(res_f) or math.isinf(res_f):
        raise ValueError("Power evaluation produced non-real or non-finite result.")
    return res_f


def _safe_call(func_name: str, args: List[Number]) -> Number:
    """Dispatch whitelisted math functions."""
    if func_name == "sqrt":
        if len(args) != 1:
            raise ValueError("sqrt() takes exactly 1 argument.")
        val = float(args[0])
        if val < 0:
            raise ValueError("sqrt() argument must be non-negative.")
        return math.sqrt(val)

    if func_name == "log":
        if len(args) not in (1, 2):
            raise ValueError("log() takes 1 or 2 arguments.")
        f_args = [float(a) for a in args]
        return math.log(*f_args)

    if func_name == "log10":
        if len(args) != 1:
            raise ValueError("log10() takes exactly 1 argument.")
        return math.log10(float(args[0]))

    if func_name == "sin":
        if len(args) != 1:
            raise ValueError("sin() takes exactly 1 argument.")
        return math.sin(float(args[0]))

    if func_name == "cos":
        if len(args) != 1:
            raise ValueError("cos() takes exactly 1 argument.")
        return math.cos(float(args[0]))

    if func_name == "abs":
        if len(args) != 1:
            raise ValueError("abs() takes exactly 1 argument.")
        return abs(args[0])

    if func_name == "gcd":
        if not args:
            raise ValueError("gcd() requires at least 1 argument.")
        int_args: List[int] = []
        for a in args:
            if isinstance(a, Fraction) and a.denominator == 1:
                int_args.append(a.numerator)
            else:
                raise ValueError("gcd() arguments must be integers.")
        return Fraction(math.gcd(*int_args))

    if func_name == "round":
        if len(args) not in (1, 2):
            raise ValueError("round() takes 1 or 2 arguments.")
        if len(args) == 1:
            return Fraction(round(args[0]))
        ndigits = args[1]
        if not (isinstance(ndigits, Fraction) and ndigits.denominator == 1):
            raise ValueError("round() ndigits must be an integer.")
        res = round(args[0], ndigits.numerator)
        return _to_number(res)

    raise ValueError(f"Function '{func_name}' is not whitelisted.")


_ALLOWED_FUNCTIONS = frozenset({"sqrt", "log", "log10", "sin", "cos", "abs", "gcd", "round"})


def _eval_ast(node: ast.AST, env: Dict[str, Number]) -> Number:
    """Recursively evaluate a strictly whitelisted AST node."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, env)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")
        return _to_number(node.value)

    if isinstance(node, ast.Name):
        name = node.id
        if "__" in name:
            raise ValueError(f"Dunder identifier '{name}' is forbidden.")
        if name in env:
            return env[name]
        if name in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[name]
        raise ValueError(f"Unknown or disallowed identifier: '{name}'")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, env)
        right = _eval_ast(node.right, env)
        op = node.op

        if isinstance(op, ast.Pow):
            return _safe_pow(left, right)

        if isinstance(op, ast.Add):
            res = left + right
        elif isinstance(op, ast.Sub):
            res = left - right
        elif isinstance(op, ast.Mult):
            res = left * right
        elif isinstance(op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("Division by zero.")
            res = left / right
        elif isinstance(op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError("Integer division by zero.")
            res = left // right
        elif isinstance(op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("Modulo by zero.")
            res = left % right
        else:
            raise ValueError(f"Disallowed binary operator: {type(op).__name__}")

        return _to_number(res)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct calls to whitelisted functions are allowed.")
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed.")
        func_name = node.func.id
        if "__" in func_name or func_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not allowed.")
        args = [_eval_ast(arg, env) for arg in node.args]
        return _safe_call(func_name, args)

    raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def safe_eval(
    expr: str,
    env: Optional[Dict[str, Union[int, float, Fraction]]] = None,
) -> Union[Fraction, float]:
    """
    Safely evaluate a mathematical expression using a strict AST whitelist.

    Allowed AST nodes: Expression, BinOp, UnaryOp, Constant, Name, Call.
    Blocks Attribute, Subscript, Lambda, comprehensions, Import, dunders, and DoS powers.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("Expression must be a non-empty string.")

    clean_env: Dict[str, Number] = {}
    if env is not None:
        for k, v in env.items():
            if not isinstance(k, str) or "__" in k:
                raise ValueError(f"Invalid environment variable name: '{k}'")
            clean_env[k] = _to_number(v)

    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    return _eval_ast(tree, clean_env)


# ---------------------------------------------------------------------------
# Quadratic Solver
# ---------------------------------------------------------------------------

def _exact_sqrt_fraction(q: Fraction) -> Optional[Fraction]:
    """Return exact Fraction sqrt(q) if q >= 0 is a rational square, else None."""
    if q < 0:
        return None
    if q == 0:
        return Fraction(0)
    num_sqrt = math.isqrt(q.numerator)
    den_sqrt = math.isqrt(q.denominator)
    if num_sqrt * num_sqrt == q.numerator and den_sqrt * den_sqrt == q.denominator:
        return Fraction(num_sqrt, den_sqrt)
    return None


def solve_quadratic(
    a: Union[int, float, Fraction],
    b: Union[int, float, Fraction],
    c: Union[int, float, Fraction],
) -> Dict[str, Any]:
    """
    Solve ax^2 + bx + c = 0 accurately with discriminant, Vieta's formulas, and roots.
    """
    fa = _to_fraction(a)
    fb = _to_fraction(b)
    fc = _to_fraction(c)

    if fa == 0:
        raise ValueError("Coefficient 'a' must be non-zero for a quadratic equation.")

    delta = fb * fb - 4 * fa * fc
    vieta_sum = -fb / fa
    vieta_product = fc / fa

    if delta > 0:
        nature = "two_real"
        sqrt_d = _exact_sqrt_fraction(delta)
        if sqrt_d is not None:
            x1: Any = (-fb + sqrt_d) / (2 * fa)
            x2: Any = (-fb - sqrt_d) / (2 * fa)
        else:
            sd_f = math.sqrt(float(delta))
            x1 = (-float(fb) + sd_f) / (2.0 * float(fa))
            x2 = (-float(fb) - sd_f) / (2.0 * float(fa))
    elif delta == 0:
        nature = "double_real"
        x1 = -fb / (2 * fa)
        x2 = x1
    else:
        nature = "two_complex"
        real_p = float(-fb / (2 * fa))
        imag_p = abs(math.sqrt(float(-delta)) / (2.0 * float(fa)))
        x1 = complex(real_p, imag_p)
        x2 = complex(real_p, -imag_p)

    return {
        "delta": delta,
        "discriminant": delta,
        "nature": nature,
        "x1": x1,
        "x2": x2,
        "roots": [x1, x2],
        "S": vieta_sum,
        "P": vieta_product,
        "vieta_sum": vieta_sum,
        "vieta_product": vieta_product,
    }


# ---------------------------------------------------------------------------
# Linear System Solver (Exact Gaussian Elimination on Fraction)
# ---------------------------------------------------------------------------

def _rref_fraction(matrix: List[List[Fraction]]) -> Tuple[List[List[Fraction]], List[int]]:
    """Compute Reduced Row Echelon Form of a Fraction matrix and return pivot columns."""
    m = [row[:] for row in matrix]
    n_rows = len(m)
    if n_rows == 0:
        return m, []
    n_cols = len(m[0])
    pivots: List[int] = []
    r = 0

    for c in range(n_cols):
        if r >= n_rows:
            break
        pivot_row = None
        for i in range(r, n_rows):
            if m[i][c] != 0:
                pivot_row = i
                break
        if pivot_row is None:
            continue

        m[r], m[pivot_row] = m[pivot_row], m[r]
        piv_val = m[r][c]
        m[r] = [v / piv_val for v in m[r]]

        for i in range(n_rows):
            if i != r and m[i][c] != 0:
                factor = m[i][c]
                m[i] = [m[i][j] - factor * m[r][j] for j in range(n_cols)]

        pivots.append(c)
        r += 1

    return m, pivots


def solve_linear_system(
    rows: List[List[Union[int, float, Fraction]]],
) -> Dict[str, Any]:
    """
    Solve an augmented linear system [A | b] using exact Gauss-Jordan elimination on Fraction.
    """
    if not rows:
        raise ValueError("Linear system must have at least one row.")
    n_cols = len(rows[0])
    if n_cols < 2:
        raise ValueError("Each row must have at least one variable and one RHS constant.")
    if any(len(row) != n_cols for row in rows):
        raise ValueError("All rows in the augmented matrix must have the same length.")

    n_vars = n_cols - 1
    frac_mat = [[_to_fraction(val) for val in row] for row in rows]
    rref, pivots = _rref_fraction(frac_mat)

    # Check inconsistency: pivot in the RHS column (n_vars)
    if n_vars in pivots:
        return {
            "status": "no_solution",
            "solution": None,
            "rank": len(pivots) - 1,
            "rref": rref,
        }

    rank = len(pivots)
    if rank < n_vars:
        free_vars = [c for c in range(n_vars) if c not in pivots]
        return {
            "status": "infinite",
            "solution": None,
            "rank": rank,
            "free_vars": free_vars,
            "rref": rref,
        }

    solution = [rref[i][n_vars] for i in range(n_vars)]
    return {
        "status": "unique",
        "solution": solution,
        "rank": rank,
        "rref": rref,
    }


# ---------------------------------------------------------------------------
# Chemical Equation Balancer (Integer Null-Space over Fraction)
# ---------------------------------------------------------------------------

def _parse_formula(formula: str) -> Dict[str, int]:
    """Parse a chemical formula (supporting parentheses/brackets) into element counts."""
    formula = formula.strip()
    if not formula:
        raise ValueError("Empty chemical formula.")

    stack: List[Dict[str, int]] = [{}]
    i = 0
    n = len(formula)

    while i < n:
        ch = formula[i]
        if ch in ("(", "["):
            stack.append({})
            i += 1
        elif ch in (")", "]"):
            if len(stack) == 1:
                raise ValueError(f"Unmatched closing bracket in formula '{formula}'.")
            i += 1
            num_start = i
            while i < n and formula[i].isdigit():
                i += 1
            mult = int(formula[num_start:i]) if i > num_start else 1
            group = stack.pop()
            for elem, cnt in group.items():
                stack[-1][elem] = stack[-1].get(elem, 0) + cnt * mult
        elif ch.isupper():
            elem_start = i
            i += 1
            while i < n and formula[i].islower():
                i += 1
            elem = formula[elem_start:i]
            num_start = i
            while i < n and formula[i].isdigit():
                i += 1
            cnt = int(formula[num_start:i]) if i > num_start else 1
            stack[-1][elem] = stack[-1].get(elem, 0) + cnt
        else:
            raise ValueError(f"Unexpected character '{ch}' in formula '{formula}'.")

    if len(stack) != 1:
        raise ValueError(f"Unclosed bracket in formula '{formula}'.")
    return stack[0]


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def balance_equation(eq: str) -> Dict[str, Any]:
    """
    Balance a chemical equation using exact integer null-space computation over Fraction.
    """
    if not isinstance(eq, str) or not eq.strip():
        raise ValueError("Equation must be a non-empty string.")

    parts = re.split(r"\s*(?:-->|->|→|=)\s*", eq.strip())
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid chemical equation format: '{eq}'.")

    reactants = [s.strip() for s in parts[0].split("+") if s.strip()]
    products = [s.strip() for s in parts[1].split("+") if s.strip()]
    if not reactants or not products:
        raise ValueError("Both reactants and products must be non-empty.")

    species = reactants + products
    parsed_species = [_parse_formula(sp) for sp in species]

    elements = sorted({el for counts in parsed_species for el in counts})
    n_species = len(species)
    n_reactants = len(reactants)

    matrix: List[List[Fraction]] = []
    for el in elements:
        row: List[Fraction] = []
        for idx, counts in enumerate(parsed_species):
            sign = 1 if idx < n_reactants else -1
            row.append(Fraction(sign * counts.get(el, 0)))
        matrix.append(row)

    rref, pivots = _rref_fraction(matrix)
    free_cols = [c for c in range(n_species) if c not in pivots]
    if len(free_cols) != 1:
        raise ValueError(
            f"Expected unique balancing null-space (nullity=1), got nullity={len(free_cols)}."
        )

    free_col = free_cols[0]
    null_vec = [Fraction(0)] * n_species
    null_vec[free_col] = Fraction(1)
    for row_idx, piv_col in enumerate(pivots):
        null_vec[piv_col] = -rref[row_idx][free_col]

    # Scale to coprime positive integers
    lcm_den = 1
    for v in null_vec:
        lcm_den = _lcm(lcm_den, v.denominator)
    int_vec = [int(v * lcm_den) for v in null_vec]

    if all(x < 0 for x in int_vec):
        int_vec = [-x for x in int_vec]
    if any(x <= 0 for x in int_vec):
        raise ValueError("Equation has no strictly positive integer balancing.")

    g = 0
    for x in int_vec:
        g = math.gcd(g, x)
    coeffs = [x // g for x in int_vec]

    r_coeffs = coeffs[:n_reactants]
    p_coeffs = coeffs[n_reactants:]

    def _fmt(c: int, formula: str) -> str:
        return formula if c == 1 else f"{c}{formula}"

    lhs_str = " + ".join(_fmt(c, f) for c, f in zip(r_coeffs, reactants))
    rhs_str = " + ".join(_fmt(c, f) for c, f in zip(p_coeffs, products))

    return {
        "reactants": list(zip(r_coeffs, reactants)),
        "products": list(zip(p_coeffs, products)),
        "coefficients": coeffs,
        "balanced": f"{lhs_str} -> {rhs_str}",
    }


# ---------------------------------------------------------------------------
# Classical & Basic Cryptography
# ---------------------------------------------------------------------------

def _shift_char(ch: str, shift: int) -> str:
    if "A" <= ch <= "Z":
        return chr((ord(ch) - ord("A") + shift) % 26 + ord("A"))
    if "a" <= ch <= "z":
        return chr((ord(ch) - ord("a") + shift) % 26 + ord("a"))
    return ch


def caesar(text: str, shift: int, decrypt: bool = False) -> str:
    """Caesar cipher over ASCII letters (preserving case and non-letters)."""
    eff_shift = -shift if decrypt else shift
    return "".join(_shift_char(ch, eff_shift) for ch in text)


def vigenere(text: str, key: str, decrypt: bool = False) -> str:
    """Vigenère cipher over ASCII letters using alphabetic characters from key."""
    key_shifts = [ord(c.upper()) - ord("A") for c in key if "A" <= c.upper() <= "Z"]
    if not key_shifts:
        raise ValueError("Vigenère key must contain at least one ASCII letter.")

    out: List[str] = []
    k_idx = 0
    k_len = len(key_shifts)
    for ch in text:
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            s = key_shifts[k_idx % k_len]
            out.append(_shift_char(ch, -s if decrypt else s))
            k_idx += 1
        else:
            out.append(ch)
    return "".join(out)


def affine(text: str, a: int, b: int, decrypt: bool = False) -> str:
    """Affine cipher E(x) = (ax + b) mod 26; D(y) = a^-1 (y - b) mod 26."""
    if math.gcd(a, 26) != 1:
        raise ValueError(f"Multiplier 'a' ({a}) must be coprime to 26.")

    if decrypt:
        a_inv = pow(a, -1, 26)

    out: List[str] = []
    for ch in text:
        if "A" <= ch <= "Z":
            base = ord("A")
        elif "a" <= ch <= "z":
            base = ord("a")
        else:
            out.append(ch)
            continue

        x = ord(ch) - base
        y = (a_inv * (x - b)) % 26 if decrypt else (a * x + b) % 26
        out.append(chr(base + y))
    return "".join(out)


def rsa_toy(m: int, e: int, n: int) -> int:
    """Toy RSA modular exponentiation: m^e mod n."""
    if n <= 1:
        raise ValueError("Modulus 'n' must be greater than 1.")
    if e < 0:
        raise ValueError("Exponent 'e' must be non-negative.")
    if m < 0 or m >= n:
        raise ValueError("Message 'm' must satisfy 0 <= m < n.")
    return pow(m, e, n)


# ---------------------------------------------------------------------------
# Numeric Extraction & Provenance Verification (§3.5)
# ---------------------------------------------------------------------------

# Strip citation brackets like [1], [1, 2] and section/step prefixes
_BRACKET_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,;\-–]\s*\d+)*\s*\]")
_PREFIX_LABEL_RE = re.compile(
    r"(?i)\b(?:bước|ví\s*dụ|mục|phần|chương|hình|bảng|câu|bài|"
    r"step|example|section|figure|table)\s+\d+(?:[.,]\d+)*\b"
)
_NUMBER_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:[.,]\d+)*(?![A-Za-z0-9_])")


def _normalize_number_token(raw: str) -> Optional[str]:
    """Normalize Vietnamese/English decimal and thousand-separator number strings."""
    sign = ""
    s = raw.strip()
    if s.startswith("+"):
        s = s[1:]
    elif s.startswith("-"):
        sign = "-"
        s = s[1:]

    has_dot = "." in s
    has_comma = "," in s

    if has_dot and has_comma:
        # Last separator is the decimal separator
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    elif has_comma:
        parts = s.split(",")
        if len(parts) > 2:
            # Multiple commas -> thousand separators (e.g., 1,000,000)
            s = s.replace(",", "")
        else:
            int_p, frac_p = parts
            if len(frac_p) == 3 and int_p != "0" and len(int_p) <= 3:
                # English thousand separator like 1,000 or 12,500
                s = int_p + frac_p
            else:
                # Vietnamese decimal comma like 3,14 or 0,5 or 0,125
                s = f"{int_p}.{frac_p}"
    elif has_dot:
        parts = s.split(".")
        if len(parts) > 2:
            # Multiple dots -> thousand separators (e.g., 1.000.000)
            s = s.replace(".", "")

    try:
        dec = Decimal(sign + s)
    except InvalidOperation:
        return None

    if dec == 0:
        return "0"
    norm = format(dec.normalize(), "f")
    if "." in norm:
        norm = norm.rstrip("0").rstrip(".")
    return norm


def extract_numbers(text: str) -> Set[str]:
    """
    Extract and normalize numbers from text.
    Handles Vietnamese/English decimal & thousand separators, while ignoring
    citation brackets ([1]) and section/step markers (Bước 1, Ví dụ 2.1).
    """
    if not text:
        return set()

    cleaned = _BRACKET_CITATION_RE.sub(" ", text)
    cleaned = _PREFIX_LABEL_RE.sub(" ", cleaned)

    result: Set[str] = set()
    for match in _NUMBER_TOKEN_RE.finditer(cleaned):
        norm = _normalize_number_token(match.group(0))
        if norm is not None:
            result.add(norm)
    return result


def verify_numeric_provenance(
    answer: str,
    evidence_texts: List[str],
    solver_trace: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Verify Numeric Provenance Invariant (§3.5):
    Every number in `answer` must belong to Num(evidence_texts) U Num(solver_trace).
    """
    allowed: Set[str] = set()
    for text in evidence_texts:
        allowed |= extract_numbers(text)
    if solver_trace:
        for trace in solver_trace:
            allowed |= extract_numbers(trace)

    answer_nums = extract_numbers(answer)
    ungrounded = sorted(answer_nums - allowed)
    return (len(ungrounded) == 0, ungrounded)

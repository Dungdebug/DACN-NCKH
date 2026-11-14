"""
Formula and Example Patterns Recognition & Dataset (Wave 5).
Pure-Python, strict offline, zero external dependencies.

Provides:
1. FORMULA_EXAMPLE_DATASET: Rich comprehensive dataset (35+ entries) for recognizing
   mathematical formulas, cryptographic equations, and illustrated examples vs. theoretical definitions.
2. Recognition helpers:
   - is_formula_sentence(text): Distinguishes mathematical formula sentences from theoretical definitions.
   - is_example_sentence(text): Detects illustrated examples and walkthrough blocks.
   - is_definition_sentence(text): Identifies conceptual/theoretical definitions.
   - classify_sentence_type(text): Classifies a sentence into formula, example, definition, or general.
3. Core extractor:
   - extract_examples_and_formulas(passages, topic=''): Extracts complete illustrated examples
     and formulas from passages, distinguishing theoretical definitions from formulas.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple


def strip_accents(text: str) -> str:
    """Removes Vietnamese diacritics for robust matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


# ------------------------------------------------------------------------------
# FORMULA AND EXAMPLE RECOGNITION PATTERNS (REGEX & HEURISTICS)
# ------------------------------------------------------------------------------

# Patterns identifying mathematical formulas, equations, modular arithmetic
FORMULA_PATTERNS = [
    # Explicit formula / equation lead-in
    r"\b(?:ta có|trong đó|công thức(?: mã hóa| giải mã)?|biểu thức|phương trình|hàm số)\s*:\s*",
    # Variable assignment / equation with mathematical variables (e.g. zi = fi, P = C = K = Z26, ci = (pi + ki))
    r"\b[A-Za-z][A-Za-z0-9_]*\s*=\s*(?:[A-Za-z0-9_]+\s*=\s*)*[A-Za-z0-9_]+",
    r"\b[a-zA-Z][a-zA-Z0-9_]*\s*=\s*\(?[a-zA-Z0-9_]+\s*[\+\-\*\^/⊕]\s*[a-zA-Z0-9_]+",
    # Function definition with arguments (e.g. e(x) =, e_k(x) =, fi(K, ...), d(y) =)
    r"\b[a-zA-Z]\w*\s*\([a-zA-Z0-9_,\s\+\-\*\^/\.]*\)\s*=",
    # Modular arithmetic and rings
    r"\b(?:mod|modulo)\s+\d+\b",
    r"\bZ_?\d+\b",
    r"\bGF\([^\)]+\)",
    # Mathematical symbols and operators
    r"[⊕⊗≡≠≤≥∑∏√≈×·]",
    r"\bXOR\b.*(?:dòng khóa|khóa|bit|byte|zi|pi|ci|k_i|z_i)",
    r"\b(?:gcd|det|inv|mod_inverse)\s*\(",
    # Cryptographic variable specs: P = C = K = Z26
    r"\b[PCKMXY]\s*=\s*Z\d+\b",
    r"\b[a-z]i\s*=\s*[a-z]i\b",
]

# Patterns identifying illustrated examples
EXAMPLE_PATTERNS = [
    r"\b(?:ví dụ|vi du|chẳng hạn|minh họa|bài toán minh họa|xét ví dụ|áp dụng ví dụ)\b",
    r"\b(?:giả sử|gia su)\b",
    r"\b(?:xét|cho)\s+(?:bản rõ|bản mã|khóa|chuỗi|ma trận|số nguyên|thông điệp|trường hợp)\b",
    r"\b(?:bước 1|bước 2|bước 3|kết quả thu được|kết quả mã hóa|kết quả giải mã)\b",
    r"\b(?:bản rõ\s+P\s*=|bản mã\s+C\s*=|khóa\s+k\s*=|khóa\s+k1\s*=)",
    r"\bchuyển đổi\s*:\s*[A-Z]\s*\(\d+\)",
]

# Patterns identifying purely theoretical or conceptual definitions
DEFINITION_PATTERNS = [
    r"\b(?:là phương pháp|là kỹ thuật|là khái niệm|được định nghĩa là|nghĩa là|là hệ thống|là quy trình|là thuật toán|là dạng mã hóa)\b",
    r"\b(?:còn được gọi là|được hiểu là)\b",
    r"\ban ninh (?:vô điều kiện|tính toán)\s*:\s*",
    r"^(?:định nghĩa|khái niệm)\s*:",
]

_COMPILED_FORMULA_RE = [re.compile(p, re.IGNORECASE) for p in FORMULA_PATTERNS]
_COMPILED_EXAMPLE_RE = [re.compile(p, re.IGNORECASE) for p in EXAMPLE_PATTERNS]
_COMPILED_DEF_RE = [re.compile(p, re.IGNORECASE) for p in DEFINITION_PATTERNS]


# ------------------------------------------------------------------------------
# CLASSIFICATION & DISCRIMINATION HELPERS
# ------------------------------------------------------------------------------

def is_formula_sentence(text: str) -> bool:
    """
    Distinguishes mathematical formula sentences from theoretical definitions.
    Returns True for equations, variable formulas ('Ta có: zi = fi...', 'Cho P = C = K = Z26', 'mod 26'),
    and False for pure conceptual sentences ('Mã hóa đối xứng là phương pháp...').
    """
    if not text or not isinstance(text, str):
        return False

    clean = text.strip()
    if not clean:
        return False

    # Ignore slide headers and metadata
    if re.match(r"^===\s*Page\s*\d+\s*===$", clean, re.IGNORECASE):
        return False
    if re.match(r"^\d+\.\s+[A-ZÀ-Ỹ\s]+:$", clean):
        return False

    # Check for strong formula markers
    has_formula_match = any(regex.search(clean) for regex in _COMPILED_FORMULA_RE)

    # Check for mathematical symbols
    has_math_symbols = any(sym in clean for sym in [" = ", " + ", " - ", " * ", " / ", " mod ", " ⊕ ", " XOR ", "zi = ", "ci = ", "pi = ", "e(x) = ", "d(y) = ", "Z26"])

    if not (has_formula_match or has_math_symbols):
        return False

    # Guard: If sentence has strong definition cues ("là phương pháp", "là khái niệm")
    # AND lacks mathematical equation syntax (no '=', no 'mod', no '⊕'), it is a definition, not a formula.
    is_def = any(regex.search(clean) for regex in _COMPILED_DEF_RE)
    has_active_equation = any(sym in clean for sym in ["=", "mod", "⊕", "≡", "+", "-", "*", "^", "Z26", "zi", "ci"])

    if is_def and not has_active_equation:
        return False

    return True


def is_example_sentence(text: str) -> bool:
    """
    Detects whether a sentence or paragraph is an illustrated example or walkthrough.
    Matches 'Ví dụ:', 'Minh họa:', 'Cho P = "HELLO", k = 3', etc.
    """
    if not text or not isinstance(text, str):
        return False

    clean = text.strip()
    if not clean:
        return False

    return any(regex.search(clean) for regex in _COMPILED_EXAMPLE_RE)


def is_definition_sentence(text: str) -> bool:
    """
    Identifies conceptual or theoretical definitions.
    Returns True for sentences defining concepts, mechanisms, or security terms.
    """
    if not text or not isinstance(text, str):
        return False

    clean = text.strip()
    if not clean:
        return False

    # Pure equations are not definitions
    if is_formula_sentence(clean) and not any(regex.search(clean) for regex in _COMPILED_DEF_RE):
        return False

    return any(regex.search(clean) for regex in _COMPILED_DEF_RE)


def classify_sentence_type(text: str) -> str:
    """
    Classifies sentence type into:
    - 'example': Illustrated example or step-by-step application
    - 'formula': Mathematical formula, equation, or parameter specification
    - 'definition': Theoretical definition or classification
    - 'general': General descriptive sentence
    """
    if not text:
        return "general"

    if is_example_sentence(text):
        return "example"
    if is_formula_sentence(text):
        return "formula"
    if is_definition_sentence(text):
        return "definition"
    return "general"


# ------------------------------------------------------------------------------
# CORE EXTRACTOR: extract_examples_and_formulas
# ------------------------------------------------------------------------------

def extract_examples_and_formulas(
    passages: List[Dict[str, Any]],
    topic: str = "",
    include_section: bool = False,
) -> List[str]:
    """
    Extracts complete illustrated examples and mathematical formulas from passages.
    - Distinguishes theoretical definition sentences from formula sentences ('Ta có: zi = fi...', 'Cho P = C = K = Z26', 'mod 26').
    - Preserves complete illustrated example blocks ('cho ví dụ') without cutting them abruptly.
    - Filters by topic if provided (case & accent insensitive), falling back to all passages if no exact match.

    Parameters:
        passages: List of passage dicts or objects with 'text' and optional 'section_title'.
        topic: Optional topic filter (e.g. 'autokey', 'hệ mã dòng', 'cho ví dụ').
        include_section: Whether to prepend the section title if available.

    Returns:
        List of clean, extracted formula and example strings.
    """
    if not passages:
        return []

    # Normalize passages input
    if isinstance(passages, dict):
        passages = [passages]

    # Normalize topic query
    t_clean = (topic or "").strip()
    t_norm = strip_accents(t_clean.lower())

    # Detect if topic query is just an example request intent (e.g. "cho ví dụ", "ví dụ")
    is_example_intent = t_norm in {
        "cho vi du", "vi du", "cho ví dụ", "ví dụ", "minh hoa", "cho xem vi du",
        "lay vi du", "lay vi du minh hoa", "cho vi du cu the", "cho vi du ve",
    } or re.match(r"^(?:cho\s+)?(?:ví\s+dụ|vi\s+du|minh\s+họa)\b", t_norm)

    # Extract core subject keyword if query is like "cho ví dụ về hệ mã autokey"
    subject_term = ""
    if not is_example_intent and t_norm:
        subject_term = t_norm
    elif is_example_intent:
        # Strip example lead-ins: "cho ví dụ về hệ mã autokey" -> "autokey"
        m = re.sub(r"^(?:hãy\s+)?(?:cho\s+)?(?:ví\s+dụ|vi\s+du|minh\s+họa|công\s+thức|cong\s+thuc)(?:\s+về|\s+ve|\s+của|\s+cua)?\s*", "", t_norm)
        subject_term = m.strip()

    # Step 1: Select candidate passages based on topic
    candidate_passages: List[Dict[str, Any]] = []

    if subject_term and len(subject_term) > 2:
        for p in passages:
            p_text = getattr(p, "text", "") or (p.get("text", "") if isinstance(p, dict) else "")
            p_sec = getattr(p, "section_title", "") or (p.get("section_title", "") if isinstance(p, dict) else "")
            combined_norm = strip_accents(f"{p_sec} {p_text}".lower())

            # Word-level or substring match
            if subject_term in combined_norm or any(w in combined_norm for w in subject_term.split() if len(w) > 3):
                candidate_passages.append(p)

    if not candidate_passages:
        candidate_passages = passages

    extracted_items: List[str] = []
    seen_signatures: Set[str] = set()

    heading_re = re.compile(
        r"^(?:===\s*Page\s*\d+\s*===\s*)?"
        r"(?:#+\s*)?"
        r"(\d+\.|\b[IVXLCDM]+\.|\bTopic\s+\d+[:.\-–]?|\bChủ đề\s+\d+[:.\-–]?|\bBài\s+\d+[:.\-–]?|\bMục\s+\d+[:.\-–]?)\s*"
        r"([^\n:]+):?",
        re.IGNORECASE,
    )

    def _run_extraction(target_list: List[Dict[str, Any]]) -> List[str]:
        items: List[str] = []
        signatures: Set[str] = set()

        for p in target_list:
            p_text = getattr(p, "text", "") or (p.get("text", "") if isinstance(p, dict) else "")
            p_sec = getattr(p, "section_title", "") or (p.get("section_title", "") if isinstance(p, dict) else "")

            if not p_text:
                continue

            raw_lines = p_text.splitlines()
            current_example_lines: List[str] = []
            in_example_block = False

            for line in raw_lines:
                line_str = line.strip()
                if not line_str:
                    continue

                # Strip page marker
                if re.match(r"^===\s*Page\s*\d+\s*===$", line_str, re.IGNORECASE):
                    continue
                clean_line = re.sub(r"^===\s*Page\s*\d+\s*===\s*", "", line_str).strip()

                # Heading line check -> terminates example block
                if heading_re.match(clean_line):
                    if in_example_block and current_example_lines:
                        full_ex = " ".join(current_example_lines)
                        sig = strip_accents(full_ex.lower()[:80])
                        if sig not in signatures:
                            signatures.add(sig)
                            prefix = f"{p_sec}: " if (include_section and p_sec and p_sec not in full_ex) else ""
                            items.append(f"{prefix}{full_ex}")
                        current_example_lines = []
                        in_example_block = False
                    continue

                # Check if this line starts an example block
                if is_example_sentence(clean_line):
                    if in_example_block and current_example_lines:
                        full_ex = " ".join(current_example_lines)
                        sig = strip_accents(full_ex.lower()[:80])
                        if sig not in signatures:
                            signatures.add(sig)
                            prefix = f"{p_sec}: " if (include_section and p_sec and p_sec not in full_ex) else ""
                            items.append(f"{prefix}{full_ex}")
                        current_example_lines = []
                    in_example_block = True
                    current_example_lines.append(clean_line.lstrip("-*•+ "))
                    continue

                # If inside an example block, collect continuation lines
                if in_example_block:
                    if is_definition_sentence(clean_line) and not is_formula_sentence(clean_line):
                        full_ex = " ".join(current_example_lines)
                        sig = strip_accents(full_ex.lower()[:80])
                        if sig not in signatures:
                            signatures.add(sig)
                            prefix = f"{p_sec}: " if (include_section and p_sec and p_sec not in full_ex) else ""
                            items.append(f"{prefix}{full_ex}")
                        current_example_lines = []
                        in_example_block = False
                    else:
                        current_example_lines.append(clean_line.lstrip("-*•+ "))
                        continue

                # Check for standalone mathematical formula
                if is_formula_sentence(clean_line) and not is_definition_sentence(clean_line):
                    clean_formula = clean_line.lstrip("-*•+ ").strip()
                    sig = strip_accents(clean_formula.lower()[:80])
                    if sig not in signatures:
                        signatures.add(sig)
                        prefix = f"{p_sec}: " if (include_section and p_sec and p_sec not in clean_formula) else ""
                        items.append(f"{prefix}{clean_formula}")

            # Flush remaining example block at end of passage
            if in_example_block and current_example_lines:
                full_ex = " ".join(current_example_lines)
                sig = strip_accents(full_ex.lower()[:80])
                if sig not in signatures:
                    signatures.add(sig)
                    prefix = f"{p_sec}: " if (include_section and p_sec and p_sec not in full_ex) else ""
                    items.append(f"{prefix}{full_ex}")

        return items

    extracted = _run_extraction(candidate_passages)
    # If candidate filter yielded no formulas or examples, fall back to all passages
    if not extracted and candidate_passages != passages:
        extracted = _run_extraction(passages)

    return extracted


def extract_formulas(passages_or_text: Any, topic: str = "") -> List[str]:
    """Extracts only mathematical formula sentences and equations."""
    if isinstance(passages_or_text, str):
        passages = [{"text": passages_or_text}]
    elif isinstance(passages_or_text, list):
        passages = [p if isinstance(p, dict) else {"text": getattr(p, "text", str(p))} for p in passages_or_text]
    else:
        passages = [passages_or_text]

    extracted = extract_examples_and_formulas(passages, topic=topic)
    return [item for item in extracted if is_formula_sentence(item)]


def extract_examples(passages_or_text: Any, topic: str = "") -> List[str]:
    """Extracts only illustrated examples and walkthrough blocks."""
    if isinstance(passages_or_text, str):
        passages = [{"text": passages_or_text}]
    elif isinstance(passages_or_text, list):
        passages = [p if isinstance(p, dict) else {"text": getattr(p, "text", str(p))} for p in passages_or_text]
    else:
        passages = [passages_or_text]

    extracted = extract_examples_and_formulas(passages, topic=topic)
    return [item for item in extracted if is_example_sentence(item)]


# ------------------------------------------------------------------------------
# COMPREHENSIVE TRAINING / FEW-SHOT DATASET (FORMULA_EXAMPLE_DATASET)
# ------------------------------------------------------------------------------

FORMULA_EXAMPLE_DATASET: List[Dict[str, Any]] = [
    # 1. Shift / Caesar Cipher
    {
        "id": "FE_001",
        "topic": "caesar_cipher",
        "topic_vn": "Hệ mã dịch vòng (Caesar)",
        "category": "formula",
        "text": "Cho P = C = K = Z26. Hàm mã hóa: e_k(x) = (x + k) mod 26. Hàm giải mã: d_k(y) = (y - k) mod 26.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["e_k(x) = (x + k) mod 26", "d_k(y) = (y - k) mod 26", "P = C = K = Z26"],
        "variables": {"x": "ký tự bản rõ (0-25)", "y": "ký tự bản mã (0-25)", "k": "khóa dịch chuyển"},
    },
    {
        "id": "FE_002",
        "topic": "caesar_cipher",
        "topic_vn": "Hệ mã dịch vòng (Caesar)",
        "category": "example",
        "text": "Ví dụ: Cho bản rõ P = 'HELLO', khóa k = 3. Chuyển đổi: H(7)->10(K), E(4)->7(H), L(11)->14(O), L(11)->14(O), O(14)->17(R). Bản mã thu được C = 'KHOOR'.",
        "is_formula": True,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["H(7) + 3 mod 26 = 10(K)"],
        "variables": {"P": "HELLO", "k": 3, "C": "KHOOR"},
    },

    # 2. Affine Cipher
    {
        "id": "FE_003",
        "topic": "affine_cipher",
        "topic_vn": "Hệ mã Affine",
        "category": "formula",
        "text": "Cho P = C = Z26, khóa K = (a, b) với a, b thuộc Z26 và gcd(a, 26) = 1. Hàm mã hóa: e(x) = (a*x + b) mod 26. Hàm giải mã: d(y) = a^(-1)*(y - b) mod 26.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["e(x) = (a*x + b) mod 26", "d(y) = a^(-1)*(y - b) mod 26", "gcd(a, 26) = 1"],
        "variables": {"a": "hệ số nhân nhân khả nghịch", "b": "hệ số dịch chuyển", "a^(-1)": "nghịch đảo modulo 26"},
    },
    {
        "id": "FE_004",
        "topic": "affine_cipher",
        "topic_vn": "Hệ mã Affine",
        "category": "example",
        "text": "Ví dụ: Xét hệ mã Affine với khóa a = 7, b = 3. Do gcd(7, 26) = 1, tìm được a^(-1) = 15 mod 26. Với ký tự x = 'H' (7): e(7) = (7*7 + 3) mod 26 = 52 mod 26 = 0 ('A').",
        "is_formula": True,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["e(7) = (7*7 + 3) mod 26 = 0", "a^(-1) = 15 mod 26"],
        "variables": {"a": 7, "b": 3, "x": 7, "output": "A (0)"},
    },

    # 3. Vigenere Cipher
    {
        "id": "FE_005",
        "topic": "vigenere_cipher",
        "topic_vn": "Hệ mã Vigenere",
        "category": "formula",
        "text": "Cho độ dài khóa m. Khóa K = (k1, k2, ..., km). Công thức mã hóa: ci = (pi + k_(i mod m)) mod 26. Giải mã: pi = (ci - k_(i mod m)) mod 26.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["ci = (pi + k_(i mod m)) mod 26", "pi = (ci - k_(i mod m)) mod 26"],
        "variables": {"m": "độ dài khóa", "pi": "ký tự thứ i của bản rõ", "ci": "ký tự thứ i của bản mã"},
    },
    {
        "id": "FE_006",
        "topic": "vigenere_cipher",
        "topic_vn": "Hệ mã Vigenere",
        "category": "example",
        "text": "Ví dụ: Bản rõ P = 'ATTACK', khóa K = 'LEMON'. Lặp khóa thành 'LEMONL'. Ký tự đầu: A(0) + L(11) = 11 ('L'); T(19) + E(4) = 23 ('X'). Kết quả bản mã C = 'LXFOPV'.",
        "is_formula": True,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["A(0) + L(11) mod 26 = 11 (L)"],
        "variables": {"P": "ATTACK", "K": "LEMON", "C": "LXFOPV"},
    },

    # 4. Hill Cipher
    {
        "id": "FE_007",
        "topic": "hill_cipher",
        "topic_vn": "Hệ mã Hill",
        "category": "formula",
        "text": "Ma trận khóa K vuông cấp m x m trên Z26 với gcd(det(K), 26) = 1. Mã hóa: C = P * K mod 26. Giải mã: P = C * K^(-1) mod 26.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["C = P * K mod 26", "P = C * K^(-1) mod 26", "gcd(det(K), 26) = 1"],
        "variables": {"K": "ma trận khóa cấp m x m", "P": "vector bản rõ", "C": "vector bản mã"},
    },
    {
        "id": "FE_008",
        "topic": "hill_cipher",
        "topic_vn": "Hệ mã Hill",
        "category": "example",
        "text": "Ví dụ: Cho ma trận khóa K = [[3, 3], [2, 5]], det(K) = 15 - 6 = 9. Do gcd(9, 26) = 1, ma trận khả nghịch. Với vector P = [7, 4] ('HE'), C = [7*3 + 4*2, 7*3 + 4*5] mod 26 = [29 mod 26, 41 mod 26] = [3, 15] ('DP').",
        "is_formula": True,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["det(K) = 9", "C = [3, 15] mod 26"],
        "variables": {"K": "[[3, 3], [2, 5]]", "P": "[7, 4]", "C": "[3, 15] ('DP')"},
    },

    # 5. One-Time Pad (Độn một lần)
    {
        "id": "FE_009",
        "topic": "one_time_pad",
        "topic_vn": "Hệ mã hóa độn một lần (Vernam)",
        "category": "formula",
        "text": "Hệ mã hóa Vernam: ci = pi ⊕ ki và giải mã pi = ci ⊕ ki. Điều kiện: Khóa ngẫu nhiên có độ dài đúng bằng thông điệp và chỉ sử dụng duy nhất một lần.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["ci = pi ⊕ ki", "pi = ci ⊕ ki"],
        "variables": {"pi": "bit bản rõ", "ki": "bit khóa ngẫu nhiên", "ci": "bit bản mã", "⊕": "phép toán XOR bitwise"},
    },
    {
        "id": "FE_010",
        "topic": "one_time_pad",
        "topic_vn": "Hệ mã hóa độn một lần (Vernam)",
        "category": "example",
        "text": "Ví dụ: Bản rõ nhị phân P = 1011001, khóa ngẫu nhiên K = 0101101. Mã hóa từng bit bằng phép XOR: C = P ⊕ K = 1110100. Giải mã: C ⊕ K = 1110100 ⊕ 0101101 = 1011001 (P).",
        "is_formula": True,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["1011001 ⊕ 0101101 = 1110100"],
        "variables": {"P": "1011001", "K": "0101101", "C": "1110100"},
    },

    # 6. Stream Cipher (Mã hóa dòng)
    {
        "id": "FE_011",
        "topic": "stream_cipher",
        "topic_vn": "Hệ mã hóa dòng",
        "category": "formula",
        "text": "Ta có: zi = fi (K, x1, ..., xi-1). Dòng khóa z được sinh ra liên tục. Bản rõ được mã hóa từng bit hoặc byte bằng phép XOR: ci = pi ⊕ zi. Giải mã: pi = ci ⊕ zi.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["zi = fi (K, x1, ..., xi-1)", "ci = pi ⊕ zi", "pi = ci ⊕ zi"],
        "variables": {"K": "khóa bí mật", "zi": "bit/byte dòng khóa thứ i", "pi": "bản rõ", "ci": "bản mã"},
    },
    {
        "id": "FE_012",
        "topic": "stream_cipher",
        "topic_vn": "Hệ mã hóa dòng",
        "category": "example",
        "text": "Ví dụ: Xét hệ mã hóa dòng với khóa K ban đầu tạo chuỗi khóa z = (1, 0, 1, 1). Bản rõ p = (0, 1, 1, 0). Thực hiện phép XOR từng bit: c1 = 0 ⊕ 1 = 1, c2 = 1 ⊕ 0 = 1, c3 = 1 ⊕ 1 = 0, c4 = 0 ⊕ 1 = 1. Bản mã c = (1, 1, 0, 1).",
        "is_formula": True,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["c = (0, 1, 1, 0) ⊕ (1, 0, 1, 1) = (1, 1, 0, 1)"],
        "variables": {"p": "(0, 1, 1, 0)", "z": "(1, 0, 1, 1)", "c": "(1, 1, 0, 1)"},
    },

    # 7. Autokey Cipher
    {
        "id": "FE_013",
        "topic": "autokey_cipher",
        "topic_vn": "Hệ mã hóa Autokey",
        "category": "formula",
        "text": "Cho P = C = K = Z26. Khóa k = (k1, p1, p2, ..., pn-1) mở rộng từ ký tự bản rõ. Công thức mã hóa: ci = (pi + ki) mod 26. Giải mã: pi = (ci - ki) mod 26.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["P = C = K = Z26", "ci = (pi + ki) mod 26", "pi = (ci - ki) mod 26", "k = (k1, p1, p2, ...)"],
        "variables": {"k1": "khóa khởi tạo", "pi": "ký tự bản rõ", "ki": "khóa tự động lấy từ bản rõ"},
    },
    {
        "id": "FE_014",
        "topic": "autokey_cipher",
        "topic_vn": "Hệ mã hóa Autokey",
        "category": "example",
        "text": "Ví dụ: Cho bản rõ P = 'WEBDESIGN' (22, 4, 1, 3, 4, 18, 8, 6, 13), khóa k1 = 7. Dòng khóa k = (7, 22, 4, 1, 3, 4, 18, 8, 6). Tính c1 = (22+7) mod 26 = 3 ('D'), c2 = (4+22) mod 26 = 0 ('A')... Bản mã C = 'DLFGHVMOW'.",
        "is_formula": True,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["c1 = (22 + 7) mod 26 = 3 (D)", "c2 = (4 + 22) mod 26 = 0 (A)"],
        "variables": {"P": "WEBDESIGN", "k1": 7, "C": "DLFGHVMOW"},
    },

    # 8. Playfair Cipher
    {
        "id": "FE_015",
        "topic": "playfair_cipher",
        "topic_vn": "Hệ mã hóa Playfair",
        "category": "formula",
        "text": "Bảng ma trận khóa kích thước 5x5 chứa 25 chữ cái (gộp I và J). Quy tắc mã hóa digram: cùng hàng dịch phải (vị trí + 1 mod 5), cùng cột dịch xuống (vị trí + 1 mod 5), khác hàng cột tạo hình chữ nhật lấy góc đối cùng hàng.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["row_shift = (col + 1) mod 5", "col_shift = (row + 1) mod 5"],
        "variables": {"digram": "cặp 2 ký tự", "matrix": "ma trận 5x5"},
    },
    {
        "id": "FE_016",
        "topic": "playfair_cipher",
        "topic_vn": "Hệ mã hóa Playfair",
        "category": "example",
        "text": "Ví dụ: Xây dựng ma trận 5x5 với từ khóa 'MONARCHY'. Mã hóa cặp ký tự 'TH': T và H cùng tạo thành hình chữ nhật -> thay bằng 'ZM'. Mã hóa cặp 'EA': tạo hình chữ nhật -> thay bằng 'XD'.",
        "is_formula": False,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["'TH' -> 'ZM'", "'EA' -> 'XD'"],
        "variables": {"keyword": "MONARCHY", "plaintext_digram": "TH", "ciphertext_digram": "ZM"},
    },

    # 9. Permutation Cipher (Mã hóa hoán vị)
    {
        "id": "FE_017",
        "topic": "permutation_cipher",
        "topic_vn": "Hệ mã hóa hoán vị",
        "category": "formula",
        "text": "Khóa là một phép hoán vị pi trên tập {1, 2, ..., m}. Công thức mã hóa cho khối m ký tự: ci = p_(pi(i)). Giải mã: pi = c_(pi^(-1)(i)).",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["ci = p_(pi(i))", "pi = c_(pi^(-1)(i))"],
        "variables": {"pi": "hàm hoán vị vị trí", "m": "kích thước khối"},
    },
    {
        "id": "FE_018",
        "topic": "permutation_cipher",
        "topic_vn": "Hệ mã hóa hoán vị",
        "category": "example",
        "text": "Ví dụ: Xét phép hoán vị pi = [3, 1, 4, 2] trên khối 4 ký tự. Bản rõ 'HELP': vị trí 3 là 'L', vị trí 1 là 'H', vị trí 4 là 'P', vị trí 2 là 'E'. Bản mã thu được C = 'LHPE'.",
        "is_formula": True,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["pi = [3, 1, 4, 2]", "'HELP' -> 'LHPE'"],
        "variables": {"plaintext": "HELP", "permutation": "[3, 1, 4, 2]", "ciphertext": "LHPE"},
    },

    # 10. Substitution Cipher (Đơn bảng)
    {
        "id": "FE_019",
        "topic": "substitution_cipher",
        "topic_vn": "Hệ mã đơn bảng",
        "category": "formula",
        "text": "Không gian khóa K gồm tất cả 26! hoán vị của bảng chữ cái Sigma = {A..Z}. Mã hóa: c = pi(p). Kích thước không gian khóa: |K| = 26! ≈ 4.03 * 10^26.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["c = pi(p)", "|K| = 26! ≈ 4.03 * 10^26"],
        "variables": {"pi": "ánh xạ song ánh bảng chữ cái", "|K|": "không gian khóa"},
    },
    {
        "id": "FE_020",
        "topic": "substitution_cipher",
        "topic_vn": "Hệ mã đơn bảng",
        "category": "example",
        "text": "Ví dụ: Cho bảng hoán vị chữ cái thay thế: A->Q, B->W, C->E, D->R... Khi mã hóa bản rõ 'CAB', ta thay thế C->E, A->Q, B->W thu được bản mã C = 'EQW'.",
        "is_formula": False,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["'CAB' -> 'EQW'"],
        "variables": {"plaintext": "CAB", "ciphertext": "EQW"},
    },

    # 11. RSA Public-Key Cryptosystem
    {
        "id": "FE_021",
        "topic": "rsa_cipher",
        "topic_vn": "Hệ mã khóa công khai RSA",
        "category": "formula",
        "text": "Ta có: n = p * q, phi(n) = (p-1)*(q-1). Khóa công khai (e, n) với gcd(e, phi(n)) = 1. Khóa bí mật d thỏa mãn e * d ≡ 1 mod phi(n). Mã hóa: c = m^e mod n. Giải mã: m = c^d mod n.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["n = p * q", "phi(n) = (p-1)*(q-1)", "e * d ≡ 1 mod phi(n)", "c = m^e mod n", "m = c^d mod n"],
        "variables": {"p, q": "số nguyên tố lớn", "n": "modulus", "e": "số mũ công khai", "d": "số mũ bí mật"},
    },
    {
        "id": "FE_022",
        "topic": "rsa_cipher",
        "topic_vn": "Hệ mã khóa công khai RSA",
        "category": "example",
        "text": "Ví dụ: Chọn hai số nguyên tố p = 61, q = 53. Tính n = 61*53 = 3233, phi(n) = 60*52 = 3120. Chọn e = 17, tính được d = 2753 vì 17*2753 ≡ 1 mod 3120. Mã hóa thông điệp m = 65: c = 65^17 mod 3233 = 2790.",
        "is_formula": True,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["n = 3233", "phi(n) = 3120", "c = 65^17 mod 3233 = 2790"],
        "variables": {"p": 61, "q": 53, "m": 65, "c": 2790},
    },

    # 12. Diffie-Hellman Key Exchange
    {
        "id": "FE_023",
        "topic": "diffie_hellman",
        "topic_vn": "Trao đổi khóa Diffie-Hellman",
        "category": "formula",
        "text": "Cho số nguyên tố công khai p và phần tử sinh g mod p. Alice chọn số bí mật a, gửi A = g^a mod p. Bob chọn số bí mật b, gửi B = g^b mod p. Khóa bí mật chung: K = B^a mod p = A^b mod p = g^(ab) mod p.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["A = g^a mod p", "B = g^b mod p", "K = g^(ab) mod p"],
        "variables": {"p": "số nguyên tố công khai", "g": "phần tử sinh", "a, b": "khóa bí mật", "K": "khóa chung"},
    },
    {
        "id": "FE_024",
        "topic": "diffie_hellman",
        "topic_vn": "Trao đổi khóa Diffie-Hellman",
        "category": "example",
        "text": "Ví dụ: Chọn p = 23, g = 5. Alice chọn a = 6, gửi A = 5^6 mod 23 = 8. Bob chọn b = 15, gửi B = 5^15 mod 23 = 19. Alice tính K = 19^6 mod 23 = 2. Bob tính K = 8^15 mod 23 = 2. Khóa bí mật chung thu được K = 2.",
        "is_formula": True,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["A = 5^6 mod 23 = 8", "B = 5^15 mod 23 = 19", "K = 2"],
        "variables": {"p": 23, "g": 5, "a": 6, "b": 15, "K": 2},
    },

    # 13. Extended Euclidean Algorithm
    {
        "id": "FE_025",
        "topic": "extended_euclidean",
        "topic_vn": "Thuật toán Euclid mở rộng",
        "category": "formula",
        "text": "Định lý Bezout: Với hai số nguyên a, b, tồn tại hai số nguyên x, y sao cho: gcd(a, b) = a * x + b * y. Khi gcd(a, m) = 1, nghịch đảo modulo a^(-1) mod m chính là x mod m.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["gcd(a, b) = a*x + b*y", "a*x ≡ 1 mod m"],
        "variables": {"a, b": "số nguyên đầu vào", "x, y": "hệ số Bezout", "gcd": "ước chung lớn nhất"},
    },
    {
        "id": "FE_026",
        "topic": "extended_euclidean",
        "topic_vn": "Thuật toán Euclid mở rộng",
        "category": "example",
        "text": "Ví dụ: Tìm nghịch đảo của 7 mod 26 bằng Euclid mở rộng. Chia liên tiếp: 26 = 3*7 + 5; 7 = 1*5 + 2; 5 = 2*2 + 1. Đi ngược lại: 1 = 5 - 2*2 = 3*5 - 2*7 = 3*26 - 11*7. Do đó -11 ≡ 15 mod 26. Vậy 7^(-1) mod 26 = 15.",
        "is_formula": True,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["26 = 3*7 + 5", "1 = 3*26 - 11*7", "7^(-1) mod 26 = 15"],
        "variables": {"a": 7, "m": 26, "inverse": 15},
    },

    # 14. Fermat's Little Theorem
    {
        "id": "FE_027",
        "topic": "fermat_little_theorem",
        "topic_vn": "Định lý Fermat nhỏ",
        "category": "formula",
        "text": "Nếu p là số nguyên tố và a là số nguyên không chia hết cho p (gcd(a, p) = 1), thì: a^(p-1) ≡ 1 mod p.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["a^(p-1) ≡ 1 mod p", "gcd(a, p) = 1"],
        "variables": {"p": "số nguyên tố", "a": "số nguyên tố cùng nhau với p"},
    },
    {
        "id": "FE_028",
        "topic": "fermat_little_theorem",
        "topic_vn": "Định lý Fermat nhỏ",
        "category": "example",
        "text": "Ví dụ: Với p = 7 và a = 3, do gcd(3, 7) = 1: tính 3^(7-1) = 3^6 = 729. Ta có 729 = 7 * 104 + 1. Suy ra 729 ≡ 1 mod 7.",
        "is_formula": True,
        "is_example": True,
        "is_definition": False,
        "mathematical_notations": ["3^6 = 729 ≡ 1 mod 7"],
        "variables": {"a": 3, "p": 7, "result": "1 mod 7"},
    },

    # 15. Block Cipher (DES / Feistel)
    {
        "id": "FE_029",
        "topic": "des_feistel",
        "topic_vn": "Cấu trúc mạng Feistel trong DES",
        "category": "formula",
        "text": "Trong mạng Feistel 16 vòng: L_i = R_(i-1) và R_i = L_(i-1) ⊕ F(R_(i-1), K_i). Giải mã sử dụng cùng cấu trúc với thứ tự khóa đảo ngược: K16, K15, ..., K1.",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["L_i = R_(i-1)", "R_i = L_(i-1) ⊕ F(R_(i-1), K_i)"],
        "variables": {"L_i": "nửa trái khối dữ liệu", "R_i": "nửa phải khối dữ liệu", "F": "hàm làm rối round function"},
    },

    # 16. Finite Field GF(2^8) in AES
    {
        "id": "FE_030",
        "topic": "aes_field",
        "topic_vn": "Trường hữu hạn GF(2^8) trong AES",
        "category": "formula",
        "text": "Các phép toán trong AES thực hiện trên trường hữu hạn GF(2^8) với đa thức bất khả quy: m(x) = x^8 + x^4 + x^3 + x + 1 (0x11B). Phép nhân đa thức được rút gọn theo modulo m(x).",
        "is_formula": True,
        "is_example": False,
        "is_definition": False,
        "mathematical_notations": ["m(x) = x^8 + x^4 + x^3 + x + 1", "GF(2^8) mod m(x)"],
        "variables": {"m(x)": "đa thức bất khả quy", "GF(2^8)": "trường Galois 256 phần tử"},
    },

    # CONTRAST DEFINITIONS (Theoretical definitions - NEGATIVE for formulas)
    {
        "id": "FE_031",
        "topic": "symmetric_crypto_def",
        "topic_vn": "Định nghĩa mã hóa đối xứng",
        "category": "definition",
        "text": "Mã hóa đối xứng là phương pháp mật mã trong đó cả hai bên truyền thông đều sử dụng chung một khóa bí mật cho cả quá trình mã hóa và giải mã dữ liệu.",
        "is_formula": False,
        "is_example": False,
        "is_definition": True,
        "mathematical_notations": [],
        "variables": {},
    },
    {
        "id": "FE_032",
        "topic": "unconditional_security_def",
        "topic_vn": "Định nghĩa an ninh vô điều kiện",
        "category": "definition",
        "text": "An ninh vô điều kiện nghĩa là hệ thống mã hóa không thể bị phá mã dù đối phương sở hữu năng lực tính toán và thời gian vô hạn.",
        "is_formula": False,
        "is_example": False,
        "is_definition": True,
        "mathematical_notations": [],
        "variables": {},
    },
    {
        "id": "FE_033",
        "topic": "computational_security_def",
        "topic_vn": "Định nghĩa an ninh tính toán",
        "category": "definition",
        "text": "An ninh tính toán nghĩa là chi phí để phá mã vượt quá giá trị của thông tin mã hóa, hoặc thời gian phá mã vượt quá thời gian hữu dụng của thông tin.",
        "is_formula": False,
        "is_example": False,
        "is_definition": True,
        "mathematical_notations": [],
        "variables": {},
    },
    {
        "id": "FE_034",
        "topic": "brute_force_def",
        "topic_vn": "Định nghĩa thám mã vét cạn",
        "category": "definition",
        "text": "Thám mã vét cạn (Brute-force attack) là kỹ thuật tấn công thử lần lượt tất cả các khóa khả dĩ trong không gian khóa cho đến khi thu được bản rõ hợp lệ.",
        "is_formula": False,
        "is_example": False,
        "is_definition": True,
        "mathematical_notations": [],
        "variables": {},
    },
    {
        "id": "FE_035",
        "topic": "frequency_analysis_def",
        "topic_vn": "Định nghĩa phân tích tần suất",
        "category": "definition",
        "text": "Phân tích tần suất là phương pháp thám mã dựa trên việc so sánh tần suất xuất hiện của các chữ cái trong bản mã với bản rõ ngôn ngữ tự nhiên.",
        "is_formula": False,
        "is_example": False,
        "is_definition": True,
        "mathematical_notations": [],
        "variables": {},
    },
]


# ------------------------------------------------------------------------------
# DATASET QUERY & RETRIEVAL HELPERS
# ------------------------------------------------------------------------------

def get_formula_example_dataset() -> List[Dict[str, Any]]:
    """Returns the complete FORMULA_EXAMPLE_DATASET."""
    return FORMULA_EXAMPLE_DATASET


def find_entries_by_topic(topic: str) -> List[Dict[str, Any]]:
    """Filters dataset entries matching topic (case & accent insensitive)."""
    if not topic:
        return FORMULA_EXAMPLE_DATASET
    norm = strip_accents(topic.lower())
    return [
        entry for entry in FORMULA_EXAMPLE_DATASET
        if norm in strip_accents(entry["topic"].lower())
        or norm in strip_accents(entry.get("topic_vn", "").lower())
        or norm in strip_accents(entry["text"].lower())
    ]


def find_formulas_by_topic(topic: str) -> List[Dict[str, Any]]:
    """Returns formula entries matching topic."""
    return [e for e in find_entries_by_topic(topic) if e.get("is_formula")]


def find_examples_by_topic(topic: str) -> List[Dict[str, Any]]:
    """Returns example entries matching topic."""
    return [e for e in find_entries_by_topic(topic) if e.get("is_example")]

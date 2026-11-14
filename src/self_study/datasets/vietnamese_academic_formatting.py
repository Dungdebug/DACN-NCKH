"""
Vietnamese University Academic Structuring & Block Recognition Engine.
100% Offline, Pure Python stdlib, Zero External Dependencies.

Accurately recognizes structures of Vietnamese university textbooks, lecture slides,
and academic papers, including:
- Chương (Chapter)
- Phần (Part)
- Mục (Section / Subsection)
- Bài (Lesson / Exercise / Lecture)
- Định nghĩa (Definition)
- Định lý (Theorem / Proposition / Lemma)
- Tính chất (Property / Characteristic)
- Hệ quả (Corollary)
- Chứng minh (Proof / Solution)
- Nhận xét (Remark / Observation)
- Chú ý (Note / Caution)
- Ví dụ (Example / Walkthrough)

Ensures that AI and Sovereign Brain v5.5 never misses any theoretical definition,
mathematical theorem, property, or exercise in academic courseware.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Block Type Enumeration & Canonical Labels
# ---------------------------------------------------------------------------

class AcademicBlockType(str, Enum):
    """Canonical categories of Vietnamese academic structural & content blocks."""
    CHUONG = "chuong"              # Chương (Chapter)
    PHAN = "phan"                  # Phần (Part)
    MUC = "muc"                    # Mục (Section / Subsection)
    BAI = "bai"                    # Bài (Lesson / Exercise / Lecture)
    DINH_NGHIA = "dinh_nghia"      # Định nghĩa (Definition)
    DINH_LY = "dinh_ly"            # Định lý / Định lí / Mệnh đề / Bổ đề (Theorem)
    TINH_CHAT = "tinh_chat"        # Tính chất / Đặc trưng (Property)
    HE_QUA = "he_qua"              # Hệ quả (Corollary)
    CHUNG_MINH = "chung_minh"      # Chứng minh / Lời giải (Proof / Solution)
    NHAN_XET = "nhan_xet"          # Nhận xét (Remark)
    CHU_Y = "chu_y"                # Chú ý / Lưu ý (Note / Caution)
    VI_DU = "vi_du"                # Ví dụ / Thí dụ (Example)
    NOI_DUNG = "noi_dung"          # Nội dung chung / Dẫn nhập (General Content)


CANONICAL_LABELS: Dict[str, str] = {
    AcademicBlockType.CHUONG.value: "Chương",
    AcademicBlockType.PHAN.value: "Phần",
    AcademicBlockType.MUC.value: "Mục",
    AcademicBlockType.BAI.value: "Bài",
    AcademicBlockType.DINH_NGHIA.value: "Định nghĩa",
    AcademicBlockType.DINH_LY.value: "Định lý",
    AcademicBlockType.TINH_CHAT.value: "Tính chất",
    AcademicBlockType.HE_QUA.value: "Hệ quả",
    AcademicBlockType.CHUNG_MINH.value: "Chứng minh",
    AcademicBlockType.NHAN_XET.value: "Nhận xét",
    AcademicBlockType.CHU_Y.value: "Chú ý",
    AcademicBlockType.VI_DU.value: "Ví dụ",
    AcademicBlockType.NOI_DUNG.value: "Nội dung",
}


# ---------------------------------------------------------------------------
# Text Normalization & Linguistic Helpers
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """Removes Vietnamese diacritics for robust accent-insensitive matching."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFD", text)
    result = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    return result.replace("đ", "d").replace("Đ", "D")


def clean_whitespace(text: str) -> str:
    """Normalizes internal whitespace while preserving clean line breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def normalize_academic_vietnamese(text: str) -> str:
    """
    Normalizes Unicode forms (NFC), standardizes punctuation,
    and removes common OCR/PDF slide artifacts (soft hyphens, zero-width chars).
    """
    if not text:
        return ""
    # Normalize Unicode canonical composition
    text = unicodedata.normalize("NFC", text)
    # Remove soft hyphens and zero-width characters
    text = text.replace("\xad", "").replace("\u200b", "").replace("\ufeff", "")
    # Normalize dashes and quotes
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)
    return text


def has_mathematical_formula(text: str) -> bool:
    """Detects presence of mathematical equations, symbols, or notation."""
    math_indicators = [
        r"[=<>≤≥≠≈±×÷⊕⊗∑∏∫√∂∞∈∉⊂⊆∪∩∧∨¬⇒⇔∀∃]",
        r"\b(?:mod|det|gcd|dim|ker|rank|log|ln|exp|sin|cos|tan)\b",
        r"\b(?:R\^n|Z_\d+|GF\([^\)]+\)|Z\d+)\b",
        r"[\+\-\*/\^]\s*[a-zA-Z0-9_]+\s*=",
        r"\b[A-Za-z0-9_]+\s*=\s*[A-Za-z0-9_]+",
        r"\b[a-zA-Z]\([a-zA-Z0-9_,\s\+\-\*]*\)\s*=",
        r"∎|\bQ\.?E\.?D\.?\b|\(đpcm\)",
    ]
    for pattern in math_indicators:
        if re.search(pattern, text):
            return True
    return False


def extract_key_terms(title: str, content: str) -> List[str]:
    """Extracts prominent academic terms, bold markers, and quoted phrases."""
    terms: List[str] = []
    # If title exists and is concise, it's a primary term
    if title and len(title) <= 50:
        terms.append(title.strip())

    # Extract terms in quotation marks or bold
    quoted = re.findall(r'["\']([^"\']{2,40})["\']', content)
    for q in quoted:
        if q not in terms:
            terms.append(q)

    bolded = re.findall(r'\*\*([^\*]{2,40})\*\*', content)
    for b in bolded:
        clean_b = b.strip()
        if clean_b and clean_b not in terms:
            terms.append(clean_b)

    return terms[:8]


# ---------------------------------------------------------------------------
# Academic Block Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AcademicBlock:
    """A recognized structural or pedagogical block in an academic text."""
    block_type: str
    label: str
    number: str
    title: str
    content: str
    raw_text: str
    start_line: int
    end_line: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts block to dictionary representation."""
        return {
            "block_type": self.block_type,
            "label": self.label,
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "raw_text": self.raw_text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Regex Patterns for Vietnamese Academic Headers
# ---------------------------------------------------------------------------

# Ordered patterns for block header recognition
BLOCK_PATTERNS: List[Tuple[str, str, str]] = [
    (AcademicBlockType.CHUONG.value, r"^(chương|chuong)\b", "Chương"),
    (AcademicBlockType.PHAN.value, r"^(phần|phan)\b", "Phần"),
    (AcademicBlockType.BAI.value, r"^(bài|bai)(?:\s+(?:tập|toán|giảng))?\b", "Bài"),
    (AcademicBlockType.DINH_NGHIA.value, r"^(định\s+nghĩa|dinh\s+nghia)\b", "Định nghĩa"),
    (AcademicBlockType.DINH_LY.value, r"^(định\s+lý|định\s+lí|dinh\s+ly|dinh\s+li|mệnh\s+đề|menh\s+de|bổ\s+đề|bo\s+de|tiên\s+đề|tien\s+de)\b", "Định lý"),
    (AcademicBlockType.TINH_CHAT.value, r"^(tính\s+chất|tinh\s+chat|đặc\s+trưng|dac\s+trung)\b", "Tính chất"),
    (AcademicBlockType.HE_QUA.value, r"^(hệ\s+quả|he\s+qua)\b", "Hệ quả"),
    (AcademicBlockType.CHUNG_MINH.value, r"^(chứng\s+minh|chung\s+minh|lời\s+giải|loi\s+giai)\b", "Chứng minh"),
    (AcademicBlockType.NHAN_XET.value, r"^(nhận\s+xét|nhan\s+xet)\b", "Nhận xét"),
    (AcademicBlockType.CHU_Y.value, r"^(chú\s+ý|chu\s+y|lưu\s+ý|luu\s+y)\b", "Chú ý"),
    (AcademicBlockType.VI_DU.value, r"^(ví\s+dụ|vi\s+du|thí\s+dụ|thi\s+du)\b", "Ví dụ"),
    (AcademicBlockType.MUC.value, r"^(mục|muc)\b", "Mục"),
]


def parse_header_line(line: str, in_content_block: bool = False) -> Optional[Dict[str, Any]]:
    """
    Inspects a single line to determine if it defines a new academic block header.
    Returns metadata dict if matched, or None if it is continuation text.
    """
    raw = line.strip()
    if not raw:
        return None

    # Strip markdown header prefixes (#, ##, ###) and list bullets (*, -, +, >)
    clean = re.sub(r"^[#>\*\-\+\•\s]+", "", raw).strip()
    is_bold = clean.startswith("**")
    clean = re.sub(r"^\*\*|\*\*$", "", clean).strip()

    # 1. Match explicit academic keywords (Chương, Phần, Định nghĩa, Định lý, v.v.)
    for btype, pat, canonical_label in BLOCK_PATTERNS:
        m = re.match(pat, clean, re.IGNORECASE)
        if not m:
            continue

        matched_word = m.group(0)
        tail = clean[m.end():].strip()

        # Check for numeric or roman numeral indicator after keyword
        num = ""
        num_match = re.match(
            r"^([0-9IVXLCDM]+(?:\.[0-9]+)*|[A-Z]|thứ\s+[a-z0-9A-Z_]+)\b",
            tail,
            re.IGNORECASE,
        )
        if num_match:
            num = num_match.group(1)
            tail = tail[num_match.end():].strip()

        has_paren = tail.startswith(("(", "["))
        has_colon_or_punct = bool(re.match(r"^[:\.\-–—]", tail))
        is_all_caps = matched_word.isupper()
        is_standalone = (len(tail) == 0)

        # STRICT VALIDATION: Distinguish real headers from regular running text
        # Must possess at least one structural cue:
        # - Numbering (Định nghĩa 1.1, Chương 2, Bài 3)
        # - Parenthesized concept name (Định nghĩa (Không gian con):)
        # - Punctuation separator (Định nghĩa:, Chứng minh:)
        # - Markdown formatting (**Định nghĩa:**, ### Định lý)
        # - ALL-CAPS keyword (CHỨNG MINH, ĐỊNH LÝ)
        # - Standalone on line (Chứng minh\n...)
        # - Structural block keyword (Chương, Phần, Mục, Bài)
        qualifies = (
            bool(num)
            or has_paren
            or has_colon_or_punct
            or is_bold
            or (is_all_caps and len(tail) < 30)
            or is_standalone
            or (btype in (AcademicBlockType.CHUONG.value, AcademicBlockType.PHAN.value,
                          AcademicBlockType.MUC.value, AcademicBlockType.BAI.value))
        )

        if not qualifies:
            continue

        title = ""
        inline_content = ""

        # Extract parenthesized title (e.g. Định nghĩa 1.1 (Không gian vectơ): ...)
        if has_paren:
            pm = re.match(r"^[\(\[]([^\)\]]+)[\)\]]\s*[:\.\-–—]?\s*(.*)", tail)
            if pm:
                title = pm.group(1).strip()
                tail = pm.group(2).strip()
        else:
            tail = re.sub(r"^[:\.\-–—]\s*", "", tail).strip()

        tail = re.sub(r"^\*\*|\*\*$", "", tail).strip()

        if btype in (AcademicBlockType.CHUONG.value, AcademicBlockType.PHAN.value,
                      AcademicBlockType.BAI.value, AcademicBlockType.MUC.value):
            title = title or tail
            inline_content = ""
        else:
            # For leaf content blocks, differentiate concise title from inline definition sentence
            lower_tail = tail.lower()
            starters = (
                "cho ", "xét ", "giả sử ", "nếu ", "với mọi ", "với ", "mọi ",
                "tồn tại ", "ta có ", "gọi ", "khi ", "trong ", "để ", "khi và chỉ khi "
            )
            is_sentence = (
                any(lower_tail.startswith(s) for s in starters)
                or (" là " in lower_tail and len(tail) > 40)
                or tail.endswith(".")
            )
            if not title and not is_sentence and 0 < len(tail) < 60:
                title = tail
                inline_content = ""
            else:
                inline_content = tail

        return {
            "block_type": btype,
            "label": canonical_label,
            "number": num,
            "title": title,
            "inline_content": inline_content,
            "original_keyword": matched_word,
            "confidence": 1.0,
        }

    # 2. Match multi-level numbered sections (e.g. '1.1. Khái niệm không gian vectơ', '2.3.4 Thuật toán')
    sec_multidot = re.match(
        r"^((?:[0-9]+(?:\.[0-9]+)+))(?:\s*[\.\:\-–—]\s*|\s+)(.*)", clean
    )
    if sec_multidot:
        sec_num = sec_multidot.group(1).strip(" .")
        sec_title = sec_multidot.group(2).strip(" :-–—")
        return {
            "block_type": AcademicBlockType.MUC.value,
            "label": "Mục",
            "number": sec_num,
            "title": sec_title,
            "inline_content": "",
            "original_keyword": sec_num,
            "confidence": 0.95,
        }

    # 3. Match Roman numeral section headings (e.g. 'I. Tổng quan', 'II: Cơ sở lý thuyết')
    # Must have a delimiter (. : -) to distinguish from variables like V, C, D in math
    sec_roman = re.match(r"^([IVXLCDM]+)\s*[\.\:\-–—]\s+(.*)", clean)
    if sec_roman:
        sec_num = sec_roman.group(1).strip()
        sec_title = sec_roman.group(2).strip(" :-–—")
        return {
            "block_type": AcademicBlockType.MUC.value,
            "label": "Mục",
            "number": sec_num,
            "title": sec_title,
            "inline_content": "",
            "original_keyword": sec_num,
            "confidence": 0.92,
        }

    # 4. Match single-level numeric heading (e.g. '1. Giới thiệu', '2. Mục tiêu')
    # ONLY when NOT inside an active content block (to avoid treating list items like '1. u + v = ...' as sections)
    if not in_content_block:
        sec_single = re.match(r"^([0-9]+)\s*[\.\:\-–—]\s+([^\d\s].*)", clean)
        if sec_single:
            sec_num = sec_single.group(1).strip()
            sec_title = sec_single.group(2).strip(" :-–—")
            return {
                "block_type": AcademicBlockType.MUC.value,
                "label": "Mục",
                "number": sec_num,
                "title": sec_title,
                "inline_content": "",
                "original_keyword": sec_num,
                "confidence": 0.88,
            }

    return None


# ---------------------------------------------------------------------------
# Core Parsing Engine
# ---------------------------------------------------------------------------

def parse_vietnamese_academic_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Parses a Vietnamese university textbook chapter, lecture slide, or academic paper
    into structured, pedagogical blocks.

    Recognizes:
    - Chương (Chapter)
    - Phần (Part)
    - Mục (Section)
    - Bài (Lesson / Exercise)
    - Định nghĩa (Definition)
    - Định lý (Theorem / Lemma / Proposition)
    - Tính chất (Property)
    - Hệ quả (Corollary)
    - Chứng minh (Proof / Solution)
    - Nhận xét (Remark)
    - Chú ý (Note / Caution)
    - Ví dụ (Example)

    Returns:
        List of dictionaries with keys:
        - block_type: str (e.g. "dinh_nghia", "dinh_ly", "tinh_chat")
        - label: str (canonical Vietnamese label, e.g. "Định nghĩa")
        - number: str (e.g. "1.1", "2", "I", "")
        - title: str (e.g. "Không gian vectơ", "Định lý Fermat nhỏ")
        - content: str (clean content body of the block)
        - raw_text: str (full unedited block text)
        - start_line: int (1-based line number)
        - end_line: int (1-based line number)
        - metadata: dict (semantic flags, formulas indicator, key terms, confidence)
    """
    if not text or not text.strip():
        return []

    normalized_text = normalize_academic_vietnamese(text)
    lines = normalized_text.splitlines()

    blocks: List[Dict[str, Any]] = []
    current_block: Optional[Dict[str, Any]] = None

    content_types = {
        AcademicBlockType.DINH_NGHIA.value,
        AcademicBlockType.DINH_LY.value,
        AcademicBlockType.TINH_CHAT.value,
        AcademicBlockType.HE_QUA.value,
        AcademicBlockType.CHUNG_MINH.value,
        AcademicBlockType.NHAN_XET.value,
        AcademicBlockType.CHU_Y.value,
        AcademicBlockType.VI_DU.value,
    }

    def finalize_block(b: Dict[str, Any]) -> Dict[str, Any]:
        """Enriches raw block dictionary with content, metadata, and analytics."""
        content_lines = b["lines"]
        clean_content = clean_whitespace("\n".join(content_lines))
        raw_content = "\n".join(b["raw_lines"]).strip()

        btype = b["block_type"]
        title = b["title"]
        has_math = has_mathematical_formula(clean_content) or has_mathematical_formula(title)
        terms = extract_key_terms(title, clean_content)

        metadata: Dict[str, Any] = {
            "is_definition": btype == AcademicBlockType.DINH_NGHIA.value,
            "is_theorem": btype == AcademicBlockType.DINH_LY.value,
            "is_property": btype == AcademicBlockType.TINH_CHAT.value,
            "is_corollary": btype == AcademicBlockType.HE_QUA.value,
            "is_proof": btype == AcademicBlockType.CHUNG_MINH.value,
            "is_example": btype == AcademicBlockType.VI_DU.value,
            "is_remark": btype == AcademicBlockType.NHAN_XET.value,
            "is_note": btype == AcademicBlockType.CHU_Y.value,
            "is_structural": btype in (
                AcademicBlockType.CHUONG.value,
                AcademicBlockType.PHAN.value,
                AcademicBlockType.MUC.value,
                AcademicBlockType.BAI.value,
            ),
            "has_formulas": has_math,
            "key_terms": terms,
            "word_count": len(clean_content.split()) if clean_content else 0,
            "line_count": len(content_lines),
            "original_keyword": b.get("original_keyword", ""),
            "confidence": b.get("confidence", 1.0),
        }

        return {
            "block_type": btype,
            "label": b["label"],
            "number": b["number"],
            "title": title,
            "content": clean_content,
            "raw_text": raw_content,
            "start_line": b["start_line"],
            "end_line": b["end_line"],
            "metadata": metadata,
        }

    for idx, line in enumerate(lines, 1):
        in_content = (
            current_block is not None
            and current_block["block_type"] in content_types
        )
        header_info = parse_header_line(line, in_content_block=in_content)

        if header_info:
            if current_block:
                blocks.append(finalize_block(current_block))

            inline = header_info["inline_content"]
            current_block = {
                "block_type": header_info["block_type"],
                "label": header_info["label"],
                "number": header_info["number"],
                "title": header_info["title"],
                "original_keyword": header_info["original_keyword"],
                "confidence": header_info["confidence"],
                "start_line": idx,
                "end_line": idx,
                "lines": [inline] if inline else [],
                "raw_lines": [line],
            }
        else:
            if current_block:
                current_block["end_line"] = idx
                current_block["lines"].append(line)
                current_block["raw_lines"].append(line)
            elif line.strip():
                # Text before any recognized academic header (preamble / introduction)
                current_block = {
                    "block_type": AcademicBlockType.NOI_DUNG.value,
                    "label": CANONICAL_LABELS[AcademicBlockType.NOI_DUNG.value],
                    "number": "",
                    "title": "",
                    "original_keyword": "",
                    "confidence": 0.8,
                    "start_line": idx,
                    "end_line": idx,
                    "lines": [line],
                    "raw_lines": [line],
                }

    if current_block:
        blocks.append(finalize_block(current_block))

    return blocks


# ---------------------------------------------------------------------------
# Specialized Filtering & Extraction Utilities
# ---------------------------------------------------------------------------

def extract_definitions(text: str) -> List[Dict[str, Any]]:
    """Extracts all 'Định nghĩa' blocks from text."""
    blocks = parse_vietnamese_academic_blocks(text)
    return [b for b in blocks if b["block_type"] == AcademicBlockType.DINH_NGHIA.value]


def extract_theorems_and_properties(text: str) -> List[Dict[str, Any]]:
    """Extracts all 'Định lý', 'Tính chất', and 'Hệ quả' blocks from text."""
    blocks = parse_vietnamese_academic_blocks(text)
    target_types = {
        AcademicBlockType.DINH_LY.value,
        AcademicBlockType.TINH_CHAT.value,
        AcademicBlockType.HE_QUA.value,
    }
    return [b for b in blocks if b["block_type"] in target_types]


def extract_examples(text: str) -> List[Dict[str, Any]]:
    """Extracts all 'Ví dụ' blocks from text."""
    blocks = parse_vietnamese_academic_blocks(text)
    return [b for b in blocks if b["block_type"] == AcademicBlockType.VI_DU.value]


def extract_exercises(text: str) -> List[Dict[str, Any]]:
    """Extracts all 'Bài' / 'Bài tập' blocks from text."""
    blocks = parse_vietnamese_academic_blocks(text)
    return [b for b in blocks if b["block_type"] == AcademicBlockType.BAI.value]


def search_academic_blocks(
    blocks: List[Dict[str, Any]],
    query: str,
    block_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Searches parsed academic blocks by keyword with diacritic-insensitive matching.
    Optionally filters by block types.
    """
    clean_query = strip_accents(query.lower()).strip()
    if not clean_query:
        return blocks

    results: List[Dict[str, Any]] = []
    allowed_types = set(block_types) if block_types else None

    for b in blocks:
        if allowed_types and b["block_type"] not in allowed_types:
            continue

        haystack = f"{b['title']} {b['content']} {b['label']} {b['number']}"
        clean_haystack = strip_accents(haystack.lower())

        if clean_query in clean_haystack:
            results.append(b)

    return results


def extract_academic_hierarchy(text: str) -> Dict[str, Any]:
    """
    Builds a hierarchical tree from flat parsed blocks:
    Chương -> Phần -> Mục -> Leaf Blocks (Định nghĩa, Định lý, Ví dụ, v.v.).
    """
    blocks = parse_vietnamese_academic_blocks(text)

    root: Dict[str, Any] = {
        "title": "Tài liệu học tập",
        "chapters": [],
        "unassigned_blocks": [],
    }

    current_chapter: Optional[Dict[str, Any]] = None
    current_section: Optional[Dict[str, Any]] = None

    for b in blocks:
        btype = b["block_type"]

        if btype == AcademicBlockType.CHUONG.value:
            current_chapter = {
                "number": b["number"],
                "title": b["title"],
                "start_line": b["start_line"],
                "sections": [],
                "blocks": [],
            }
            root["chapters"].append(current_chapter)
            current_section = None

        elif btype in (AcademicBlockType.PHAN.value, AcademicBlockType.MUC.value):
            current_section = {
                "type": btype,
                "number": b["number"],
                "title": b["title"],
                "start_line": b["start_line"],
                "blocks": [],
            }
            if current_chapter is not None:
                current_chapter["sections"].append(current_section)
            else:
                # Top level section before chapters
                if "sections" not in root:
                    root["sections"] = []
                root["sections"].append(current_section)

        else:
            # Leaf block (Definition, Theorem, Example, etc.)
            if current_section is not None:
                current_section["blocks"].append(b)
            elif current_chapter is not None:
                current_chapter["blocks"].append(b)
            else:
                root["unassigned_blocks"].append(b)

    return root


def format_block_as_markdown(block: Dict[str, Any]) -> str:
    """Formats a block dictionary back into clean, canonical Markdown."""
    label = block.get("label", "")
    number = block.get("number", "")
    title = block.get("title", "")
    content = block.get("content", "")

    header_parts = [label]
    if number:
        header_parts.append(number)
    header = " ".join(header_parts)

    if title:
        header = f"{header} ({title})"

    btype = block.get("block_type", "")
    if btype in (AcademicBlockType.CHUONG.value, AcademicBlockType.PHAN.value):
        md = f"# {header}\n"
    elif btype == AcademicBlockType.MUC.value:
        md = f"## {header}\n"
    else:
        md = f"**{header}:**\n"

    if content:
        md += f"{content}\n"

    return md


def generate_qa_seeds_from_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforms parsed academic blocks into grounded Question-Answer candidate seeds
    for the Sovereign Brain Self-Study Engine.
    """
    qa_seeds: List[Dict[str, Any]] = []

    for b in blocks:
        btype = b["block_type"]
        title = b["title"] or b["label"]
        num = b["number"]
        content = b["content"]

        if not content or len(content) < 20:
            continue

        tag = f"{title} ({num})" if num else title

        if btype == AcademicBlockType.DINH_NGHIA.value:
            qa_seeds.append({
                "question": f"Định nghĩa của {tag} là gì?",
                "answer": content,
                "rule_type": "definition",
                "source_block": btype,
                "confidence": 1.0,
            })
        elif btype == AcademicBlockType.DINH_LY.value:
            qa_seeds.append({
                "question": f"Nội dung phát biểu của định lý {tag} là gì?",
                "answer": content,
                "rule_type": "theorem",
                "source_block": btype,
                "confidence": 0.98,
            })
        elif btype == AcademicBlockType.TINH_CHAT.value:
            qa_seeds.append({
                "question": f"{tag} có những tính chất đặc trưng nào?",
                "answer": content,
                "rule_type": "property",
                "source_block": btype,
                "confidence": 0.95,
            })
        elif btype == AcademicBlockType.HE_QUA.value:
            qa_seeds.append({
                "question": f"Hệ quả rút ra từ {tag} là gì?",
                "answer": content,
                "rule_type": "corollary",
                "source_block": btype,
                "confidence": 0.92,
            })
        elif btype == AcademicBlockType.VI_DU.value:
            qa_seeds.append({
                "question": f"Hãy nêu ví dụ minh họa cho {tag}?",
                "answer": content,
                "rule_type": "example",
                "source_block": btype,
                "confidence": 0.90,
            })

    return qa_seeds


# ---------------------------------------------------------------------------
# Rich Multi-Domain Academic Benchmark Dataset
# ---------------------------------------------------------------------------

ACADEMIC_BLOCK_BENCHMARK: List[Dict[str, Any]] = [
    {
        "id": "acad_linear_algebra_001",
        "domain": "mathematics",
        "title": "Đại số tuyến tính - Không gian vectơ và Cơ sở",
        "text": """
# CHƯƠNG 1: KHÔNG GIAN VECTƠ VÀ ÁNH XẠ TUYẾN TÍNH

Phần 1: Cấu trúc không gian vectơ
Trong phần này, chúng ta nghiên cứu các không gian vectơ tổng quát trên trường K.

1.1. Khái niệm không gian vectơ
Định nghĩa 1.1 (Không gian vectơ):
Cho tập hợp V khác rỗng và trường K (số thực R hoặc số phức C). Tập hợp V cùng với phép cộng (+) và phép nhân vô hướng (.) được gọi là không gian vectơ trên K nếu thỏa mãn 8 tiên đề:
1. u + v = v + u với mọi u, v thuộc V.
2. (u + v) + w = u + (v + w).
3. Tồn tại phần tử không 0 trong V sao cho u + 0 = u.
4. Tồn tại phần tử đối xứng -u sao cho u + (-u) = 0.
5. k(u + v) = ku + kv.
6. (k + m)u = ku + mu.
7. k(mu) = (km)u.
8. 1.u = u.

Ví dụ 1.1 (Không gian Euclide R^n):
Tập hợp các bộ n số thực R^n với phép cộng theo tọa độ và phép nhân vô hướng thông thường lập thành một không gian vectơ trên R.

Định lý 1.2 (Sự tồn tại của cơ sở):
Mọi không gian vectơ hữu hạn chiều V != {0} đều tồn tại ít nhất một cơ sở.
Chứng minh:
Vì V hữu hạn chiều nên có một hệ sinh hữu hạn S = {v1, ..., vn}.
Nếu S độc lập tuyến tính, S chính là cơ sở.
Nếu S phụ thuộc tuyến tính, ta loại bỏ dần các vectơ là tổ hợp tuyến tính của các vectơ còn lại cho đến khi thu được hệ độc lập tuyến tính tối đại. ∎

Tính chất 1.3 (Duy nhất của tọa độ):
Cho B = {e1, e2, ..., en} là một cơ sở của V. Khi đó với mọi x thuộc V, tồn tại duy nhất bộ số (c1, ..., cn) sao cho x = c1*e1 + ... + cn*en.

Hệ quả 1.4:
Mọi cơ sở của cùng một không gian vectơ hữu hạn chiều V đều có số phần tử bằng nhau, được gọi là số chiều dim(V).

Nhận xét 1:
Số chiều đặc trưng cho bậc tự do độc lập của không gian vectơ.

Chú ý:
Không được nhầm lẫn giữa hệ sinh và cơ sở. Một hệ sinh có thể phụ thuộc tuyến tính, trong khi cơ sở luôn độc lập tuyến tính.

Bài tập 1:
Chứng minh rằng tập hợp các ma trận tam giác trên cấp n lập thành một không gian con của không gian ma trận vuông cấp n.
""",
        "expected_counts": {
            "chuong": 1,
            "phan": 1,
            "muc": 1,
            "dinh_nghia": 1,
            "vi_du": 1,
            "dinh_ly": 1,
            "chung_minh": 1,
            "tinh_chat": 1,
            "he_qua": 1,
            "nhan_xet": 1,
            "chu_y": 1,
            "bai": 1,
        },
    },
    {
        "id": "acad_crypto_rsa_002",
        "domain": "information_security",
        "title": "Mật mã học - Hệ mã khóa công khai RSA",
        "text": """
# CHƯƠNG 4: HỆ MẬT MÃ KHÓA CÔNG KHAI RSA

1.1. Kiến trúc thuật toán RSA
Định nghĩa 4.1 (Hệ mật mã RSA):
Hệ mật mã RSA là bộ 5 thành phần (P, C, K, E, D), trong đó không gian bản rõ P và không gian bản mã C bằng Z_n với n = p * q (p, q là hai số nguyên tố lớn ngẫu nhiên).
Khóa công khai gồm (e, n) và khóa bí mật gồm (d, p, q) sao cho e * d ≡ 1 (mod phi(n)).

Định lý 4.1 (Định lý Euler):
Nếu gcd(a, n) = 1 thì a^phi(n) ≡ 1 (mod n), trong đó phi(n) = (p-1)*(q-1).

Chứng minh:
Theo định lý Fermat nhỏ, với số nguyên tố p ta có a^(p-1) ≡ 1 (mod p).
Áp dụng định lý số dư Trung Hoa (CRT) cho n = p * q, ta suy ra a^phi(n) ≡ 1 (mod n). ∎

Tính chất 4.2 (Tính một chiều):
Phép tính c = m^e mod n là hàm một chiều với cửa bẫy (trapdoor one-way function). Việc giải mã không thể thực hiện nếu không biết khóa bí mật d hoặc không thể phân tích n thành tích các thừa số nguyên tố.

Hệ quả 4.3:
Độ an toàn của hệ mật mã RSA phụ thuộc trực tiếp vào độ khó của bài toán phân tích số nguyên lớn thành thừa số nguyên tố.

Ví dụ 4.1:
Chọn p = 61, q = 53. Khi đó n = 61 * 53 = 3233.
phi(n) = 60 * 52 = 3120. Chọn e = 17, ta tính được d = 2753 vì 17 * 2753 ≡ 1 (mod 3120).

Nhận xét 4.2:
Trong thực tế, p và q phải có độ dài ít nhất 1024 bit (RSA-2048) để chống lại các thuật toán sàng trường số tổng quát (GNFS).

Chú ý:
Không bao giờ tái sử dụng tham số p, q hoặc chọn e quá nhỏ mà không có cơ chế đệm an toàn như OAEP.

Bài tập 4:
Cho n = 33 và e = 3. Hãy tìm khóa giải mã d tương ứng.
""",
        "expected_counts": {
            "chuong": 1,
            "muc": 1,
            "dinh_nghia": 1,
            "dinh_ly": 1,
            "chung_minh": 1,
            "tinh_chat": 1,
            "he_qua": 1,
            "vi_du": 1,
            "nhan_xet": 1,
            "chu_y": 1,
            "bai": 1,
        },
    },
    {
        "id": "acad_os_deadlock_003",
        "domain": "operating_systems",
        "title": "Hệ điều hành - Quản lý tiến trình và Bế tắc",
        "text": """
# CHƯƠNG 3: ĐỒNG BỘ HÓA TIẾN TRÌNH VÀ BẾ TẮC (DEADLOCK)

Phần A: Mô hình tiến trình
Mọi chương trình đang thực thi đều được hệ điều hành quản lý dưới dạng tiến trình.

Định nghĩa 3.1 (Bế tắc - Deadlock):
Bế tắc là trạng thái trong đó hai hay nhiều tiến trình bị chặn vĩnh viễn vì mỗi tiến trình đều đang giữ tài nguyên mà tiến trình kia cần và đang chờ tài nguyên do tiến trình khác nắm giữ.

Định lý 3.1 (Điều kiện cần của Coffman):
Bế tắc chỉ có thể xảy ra khi và chỉ khi 4 điều kiện sau đồng thời thỏa mãn:
1. Loại trừ tương hỗ (Mutual Exclusion).
2. Giữ và chờ (Hold and Wait).
3. Không ưu tiên / không thu hồi tài nguyên (No Preemption).
4. Chờ đợi vòng tròn (Circular Wait).

Tính chất 3.2:
Nếu đồ thị cấp phát tài nguyên (RAG) không có chu trình thì hệ thống không bao giờ rơi vào trạng thái bế tắc.

Hệ quả 3.3:
Phá vỡ bất kỳ một trong 4 điều kiện Coffman sẽ loại bỏ hoàn toàn khả năng xảy ra bế tắc trong hệ thống.

Ví dụ 3.1 (Bữa ăn của các triết gia):
Năm triết gia ngồi quanh một bàn tròn với 5 chiếc đũa. Nếu mỗi triết gia đồng thời cầm chiếc đũa bên trái thì hệ thống rơi vào bế tắc do chờ đợi vòng tròn.

Nhận xét:
Thuật toán Banker của Dijkstra giúp phòng tránh bế tắc nhưng đòi hỏi chi phí tính toán cao và phải biết trước số lượng tài nguyên tối đa mỗi tiến trình yêu cầu.

Chú ý:
Tránh nhầm lẫn giữa Starvation (đói tài nguyên) và Deadlock. Trong Starvation, tiến trình vẫn có thể được cấp tài nguyên sau một thời gian không xác định, còn trong Deadlock thì không bao giờ.

Bài tập 3.1:
Vẽ đồ thị RAG và chứng minh hệ thống gồm 3 tiến trình P1, P2, P3 và 2 loại tài nguyên R1, R2 có chu trình hay không.
""",
        "expected_counts": {
            "chuong": 1,
            "phan": 1,
            "dinh_nghia": 1,
            "dinh_ly": 1,
            "tinh_chat": 1,
            "he_qua": 1,
            "vi_du": 1,
            "nhan_xet": 1,
            "chu_y": 1,
            "bai": 1,
        },
    },
    {
        "id": "acad_dsa_bst_004",
        "domain": "data_structures",
        "title": "Cấu trúc dữ liệu - Cây nhị phân tìm kiếm và Cây AVL",
        "text": """
# CHƯƠNG 5: CÂY TÌM KIẾM NHỊ PHÂN VÀ CÂY CÂN BẰNG

1.1. Cây nhị phân tìm kiếm (BST)
Định nghĩa 5.1 (Cây nhị phân tìm kiếm):
Cây nhị phân tìm kiếm là một cây nhị phân trong đó mỗi nút có khóa x thỏa mãn: mọi nút ở cây con trái có khóa nhỏ hơn x, và mọi nút ở cây con phải có khóa lớn hơn x.

Định lý 5.2 (Chiều cao cây AVL):
Chiều cao h của một cây AVL có n nút luôn thỏa mãn h < 1.44 * log2(n + 2) - 0.328.

Chứng minh:
Gọi N(h) là số nút tối thiểu của cây AVL có chiều cao h. Ta có hệ thức truy hồi N(h) = N(h-1) + N(h-2) + 1. Dãy số này liên hệ chặt chẽ với dãy Fibonacci F(h+3) - 1, từ đó suy ra chặn trên logarit cho chiều cao. ∎

Tính chất 5.3 (Độ phức tạp thao tác):
Mọi thao tác tìm kiếm, chèn, và xóa trên cây AVL đều có độ phức tạp thời gian tồi nhất là O(log n).

Ví dụ 5.1 (Phép quay đơn LL):
Khi chèn một phần tử làm mất cân bằng tại nút cha với hệ số lệch +2 và cây con trái có hệ số +1, ta thực hiện phép quay phải tại nút cha.

Nhận xét 5.1:
Cây Đỏ-Đen (Red-Black Tree) nới lỏng điều kiện cân bằng so với AVL, giúp giảm số lượng phép quay khi chèn và xóa.

Chú ý:
Cần cập nhật lại chiều cao của các nút tổ tiên sau mỗi lần thực hiện phép quay cây.

Bài toán 5.2:
Cho dãy số 15, 10, 20, 8, 12, 18, 25. Hãy dựng cây BST và biểu diễn thứ tự duyệt trung thứ tự (In-order).
""",
        "expected_counts": {
            "chuong": 1,
            "muc": 1,
            "dinh_nghia": 1,
            "dinh_ly": 1,
            "chung_minh": 1,
            "tinh_chat": 1,
            "vi_du": 1,
            "nhan_xet": 1,
            "chu_y": 1,
            "bai": 1,
        },
    },
    {
        "id": "acad_calculus_continuity_005",
        "domain": "calculus",
        "title": "Giải tích toán học - Hàm liên tục và Định lý giá trị trung gian",
        "text": """
# CHƯƠNG 2: GIỚI HẠN VÀ TÍNH LIÊN TỤC CỦA HÀM SỐ

Phần 2: Tính liên tục trên đoạn
Khái niệm hàm liên tục đóng vai trò nền tảng trong toàn bộ giải tích cổ điển.

Định nghĩa 2.1 (Hàm liên tục):
Hàm số f xác định trên khoảng (a, b) được gọi là liên tục tại điểm x0 thuộc (a, b) nếu lim(x -> x0) f(x) = f(x0).

Định lý 2.2 (Định lý Bolzano-Cauchy về giá trị trung gian):
Nếu hàm số f liên tục trên đoạn [a, b] và f(a) * f(b) < 0 thì tồn tại ít nhất một điểm c thuộc (a, b) sao cho f(c) = 0.

Chứng minh:
Sử dụng phương pháp chia đôi đoạn thẳng. Đặt [a0, b0] = [a, b]. Tại mỗi bước k, chia đoạn thành hai nửa và chọn nửa đoạn có tích giá trị tại hai đầu mang dấu âm. Theo nguyên lý lồng nhau Cantor, các đoạn này thu về một điểm duy nhất c sao cho f(c) = 0. ∎

Tính chất 2.3 (Định lý Weierstrass về tính bị chặn):
Mọi hàm số liên tục trên một đoạn đóng [a, b] đều bị chặn và đạt giá trị lớn nhất cũng như giá trị nhỏ nhất trên đoạn đó.

Hệ quả 2.4:
Tập giá trị của một hàm liên tục trên đoạn đóng [a, b] là một đoạn đóng [m, M] trong đó m là giá trị nhỏ nhất và M là giá trị lớn nhất.

Ví dụ 2.1:
Phương trình x^5 - 3x - 1 = 0 có ít nhất một nghiệm thực trong khoảng (1, 2) vì f(1) = -3 < 0 và f(2) = 25 > 0.

Nhận xét:
Định lý Bolzano-Cauchy là cơ sở lý thuyết cho thuật toán tìm nghiệm xấp xỉ bằng phương pháp chia đôi (Bisection Method).

Lưu ý:
Điều kiện đoạn đóng [a, b] là bắt buộc; định lý Weierstrass không còn đúng trên khoảng mở (a, b).

Bài tập 2:
Chứng minh rằng hàm số f(x) = 1/x không đạt giá trị lớn nhất trên khoảng mở (0, 1).
""",
        "expected_counts": {
            "chuong": 1,
            "phan": 1,
            "dinh_nghia": 1,
            "dinh_ly": 1,
            "chung_minh": 1,
            "tinh_chat": 1,
            "he_qua": 1,
            "vi_du": 1,
            "nhan_xet": 1,
            "chu_y": 1,
            "bai": 1,
        },
    },
    {
        "id": "acad_networks_tcp_006",
        "domain": "computer_networks",
        "title": "Mạng máy tính - Giao thức truyền vận TCP",
        "text": """
# CHƯƠNG 6: TẦNG TRUYỀN VẬN VÀ GIAO THỨC TCP

1.1. Khái niệm dịch vụ hướng kết nối
Định nghĩa 6.1 (Giao thức TCP):
TCP (Transmission Control Protocol) là giao thức tầng truyền vận hướng kết nối, đảm bảo phân phát dữ liệu tin cậy, đúng thứ tự và có cơ chế kiểm soát tắc nghẽn cũng như điều khiển luồng.

Định lý 6.2 (Nguyên lý bắt tay ba bước):
Để thiết lập kết nối tin cậy giữa hai máy chủ qua kênh truyền không tin cậy có độ trễ thay đổi, tối thiểu phải cần 3 thông điệp (SYN, SYN-ACK, ACK).

Tính chất 6.3 (Kiểm soát tắc nghẽn AIMD):
TCP điều chỉnh kích thước cửa sổ tắc nghẽn cwnd theo nguyên lý: tăng tuyến tính khi nhận ACK (Additive Increase) và giảm cấp số nhân khi xảy ra mất gói (Multiplicative Decrease).

Hệ quả 6.4:
Cơ chế AIMD đảm bảo hệ thống hội tụ về trạng thái công bằng (fairness) và hiệu quả (efficiency) giữa các luồng cạnh tranh băng thông.

Ví dụ 6.1 (Kịch bản bắt tay TCP):
Host A gửi gói SYN với số thứ tự khởi tạo x. Host B phản hồi bằng SYN-ACK với số thứ tự y và ACK = x + 1. Host A hoàn tất kết nối bằng gói ACK với số thứ tự x + 1 và ACK = y + 1.

Nhận xét 6.1:
Thời gian trễ RTT (Round Trip Time) là nhân tố quyết định tốc độ tăng trưởng của cửa sổ cwnd trong giai đoạn Slow Start.

Chú ý:
Không nên nhầm lẫn giữa điều khiển luồng (Flow Control sử dụng rwnd) và kiểm soát tắc nghẽn (Congestion Control sử dụng cwnd).

Bài 6:
Tính thông lượng cực đại của một kết nối TCP khi RTT = 100ms và tỷ lệ mất gói p = 1%.
""",
        "expected_counts": {
            "chuong": 1,
            "muc": 1,
            "dinh_nghia": 1,
            "dinh_ly": 1,
            "tinh_chat": 1,
            "he_qua": 1,
            "vi_du": 1,
            "nhan_xet": 1,
            "chu_y": 1,
            "bai": 1,
        },
    },
    {
        "id": "acad_ai_gradient_descent_007",
        "domain": "artificial_intelligence",
        "title": "Học máy & AI - Tối ưu hóa Gradient Descent",
        "text": """
# CHƯƠNG 7: TỐI ƯU HÓA TRONG HỌC MÁY

Phần B: Các thuật toán Gradient
Nền tảng của học sâu (Deep Learning) dựa trên việc cực tiểu hóa hàm mất mát bằng giải thuật tối ưu gradient.

Định nghĩa 7.1 (Hàm mất mát Cross-Entropy):
Với bài toán phân loại đa lớp, hàm mất mát Cross-Entropy giữa phân phối nhãn thực y và xác suất dự đoán p được định nghĩa là L(y, p) = - sum(y_i * log(p_i)).

Định lý 7.1 (Định lý xấp xỉ phổ quát - Universal Approximation Theorem):
Một mạng nơ-ron truyền thẳng (feedforward) chỉ cần một lớp ẩn với số lượng nơ-ron hữu hạn và hàm kích hoạt phi tuyến tính có thể xấp xỉ bất kỳ hàm liên tục nào trên tập compact của R^n với độ chính xác tùy ý.

Tính chất 7.2 (Tính hội tụ của GD hàm lồi):
Nếu hàm mục tiêu f là lồi và gradient Lipschitz liên tục với hằng số L, giải thuật Gradient Descent với tốc độ học eta <= 1/L sẽ hội tụ về nghiệm tối ưu toàn cục với tốc độ O(1/t).

Ví dụ 7.1:
Xét hàm f(w) = w^2 - 4w + 4. Với tốc độ học eta = 0.1 và điểm bắt đầu w0 = 0, bước cập nhật thứ nhất là w1 = 0 - 0.1 * (-4) = 0.4.

Nhận xét 7.1:
Giải thuật Stochastic Gradient Descent (SGD) đưa thêm nhiễu vào gradient giúp mô hình dễ dàng thoát khỏi các cực tiểu địa phương (local minima) nông.

Lưu ý:
Lựa chọn tốc độ học eta quá lớn có thể khiến giải thuật phân kỳ (diverge), trong khi eta quá nhỏ khiến tốc độ huấn luyện chậm chạp.

Bài tập 7:
Tính đạo hàm của hàm mất mát MSE theo trọng số w trong mô hình hồi quy tuyến tính đơn biến.
""",
        "expected_counts": {
            "chuong": 1,
            "phan": 1,
            "dinh_nghia": 1,
            "dinh_ly": 1,
            "tinh_chat": 1,
            "vi_du": 1,
            "nhan_xet": 1,
            "chu_y": 1,
            "bai": 1,
        },
    },
    {
        "id": "acad_db_normalization_008",
        "domain": "database_systems",
        "title": "Cơ sở dữ liệu - Chuẩn hóa lược đồ quan hệ",
        "text": """
# CHƯƠNG 8: THIẾT KẾ CƠ SỞ DỮ LIỆU VÀ CHUẨN HÓA

1.1. Các dạng chuẩn quan hệ
Định nghĩa 8.1 (Dạng chuẩn 3 - 3NF):
Một lược đồ quan hệ R ở dạng chuẩn 2NF được gọi là ở dạng chuẩn 3NF nếu với mọi phụ thuộc hàm phi tầm thường X -> A, thì X là siêu khóa (superkey) hoặc A là thuộc tính khóa nguyên tố.

Định lý 8.2 (Bảo toàn phụ thuộc hàm):
Mọi lược đồ quan hệ R luôn có thể phân rã không mất mát thông tin thành tập các lược đồ quan hệ con đạt dạng chuẩn 3NF đồng thời bảo toàn tập phụ thuộc hàm ban đầu.

Tính chất 8.3 (Tính bất biến của BCNF):
Dạng chuẩn Boyce-Codd (BCNF) loại bỏ triệt để mọi dư thừa dữ liệu do phụ thuộc hàm gây ra, nhưng không phải lúc nào cũng bảo toàn được phụ thuộc hàm khi phân rã.

Ví dụ 8.1:
Cho quan hệ R(A, B, C) với tập phụ thuộc hàm F = {AB -> C, C -> B}. Quan hệ này ở dạng 3NF nhưng không ở dạng BCNF vì C không phải là siêu khóa.

Nhận xét:
Trong thiết kế thực tế của hệ thống quản trị CSDL quan hệ, 3NF thường được ưu tiên lựa chọn hơn BCNF để cân bằng giữa việc chống dư thừa dữ liệu và duy trì toàn vẹn dữ liệu thông qua ràng buộc phụ thuộc hàm.

Chú ý:
Phân rã không mất mát thông tin (Lossless Join Decomposition) là điều kiện tiên quyết bắt buộc; vi phạm điều kiện này sẽ sinh ra các bộ dữ liệu giả (spurious tuples).

Bài tập 8:
Tìm tất cả các khóa ứng viên của lược đồ quan hệ R(A, B, C, D, E) với F = {A -> BC, CD -> E, B -> D}.
""",
        "expected_counts": {
            "chuong": 1,
            "muc": 1,
            "dinh_nghia": 1,
            "dinh_ly": 1,
            "tinh_chat": 1,
            "vi_du": 1,
            "nhan_xet": 1,
            "chu_y": 1,
            "bai": 1,
        },
    },
]


def get_academic_benchmark_dataset() -> List[Dict[str, Any]]:
    """Returns the comprehensive multi-domain academic benchmark dataset."""
    return ACADEMIC_BLOCK_BENCHMARK


def get_academic_benchmark_by_id(sample_id: str) -> Optional[Dict[str, Any]]:
    """Finds an academic benchmark example by its identifier."""
    for item in ACADEMIC_BLOCK_BENCHMARK:
        if item["id"] == sample_id:
            return item
    return None


def validate_academic_dataset() -> bool:
    """
    Validates structural completeness and expected block counts
    across the entire academic benchmark dataset.
    """
    if len(ACADEMIC_BLOCK_BENCHMARK) < 8:
        return False

    for sample in ACADEMIC_BLOCK_BENCHMARK:
        if not sample.get("id") or not sample.get("text"):
            return False

        parsed_blocks = parse_vietnamese_academic_blocks(sample["text"])
        if not parsed_blocks:
            return False

        type_counts: Dict[str, int] = {}
        for b in parsed_blocks:
            bt = b["block_type"]
            type_counts[bt] = type_counts.get(bt, 0) + 1

        expected = sample.get("expected_counts", {})
        for req_type, min_count in expected.items():
            if type_counts.get(req_type, 0) < min_count:
                return False

    return True

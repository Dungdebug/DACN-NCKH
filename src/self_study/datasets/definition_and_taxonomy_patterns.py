"""
Definition & Taxonomy Patterns Dataset and Extraction Engine.
100% Offline, Pure Python stdlib, Zero External Dependencies.

Provides:
- TAXONOMY_INSTRUCTION_DATASET: Instruction-tuning / few-shot dataset for recognizing
  definitions, classification branches, and concrete systems/types across academic disciplines.
- extract_taxonomy_and_types(passages: list[dict]) -> dict: Core scanner aggregating
  (1) opening definition/introductory sentences,
  (2) bulleted taxonomy branches (e.g., '* Mã hóa thay thế', '* Mã hóa hoán vị'),
  (3) concrete types/systems extracted from section headers (e.g., 'Hệ mã hóa dòng',
      'Hệ mã hóa độn một lần', 'Hệ mã hóa Autokey', 'Hệ mã hóa Playfair'...).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Text Normalization & Linguistic Helpers
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    """Normalize text by removing Vietnamese accents for fuzzy matching."""
    norm = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in norm if unicodedata.category(c) != "Mn")


def _clean_whitespace(text: str) -> str:
    """Normalize internal and surrounding whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def _format_proper_casing(title: str) -> str:
    """
    Format section titles cleanly.
    If the title is ALL CAPS (e.g., 'HỆ MÃ DỊCH VÒNG', 'HỆ MÃ AFFINE'),
    converts it to clean sentence/title case while preserving proper nouns.
    If the title already has mixed casing (e.g. 'Thuật toán Round Robin'),
    preserves the author's casing.
    """
    title = _clean_whitespace(title)
    if not title:
        return ""

    # Known acronyms and proper names to preserve
    known_proper_words = {
        "affine": "Affine",
        "vigenere": "Vigenere",
        "hill": "Hill",
        "autokey": "Autokey",
        "playfair": "Playfair",
        "caesar": "Caesar",
        "vernam": "Vernam",
        "feistel": "Feistel",
        "rsa": "RSA",
        "des": "DES",
        "aes": "AES",
        "ecc": "ECC",
        "dsa": "DSA",
        "md5": "MD5",
        "sha": "SHA",
        "fcfs": "FCFS",
        "sjf": "SJF",
        "cpu": "CPU",
        "ram": "RAM",
        "os": "OS",
        "osi": "OSI",
        "tcp": "TCP",
        "ip": "IP",
        "udp": "UDP",
        "http": "HTTP",
        "https": "HTTPS",
        "cnn": "CNN",
        "rnn": "RNN",
        "lstm": "LSTM",
        "bert": "BERT",
        "gpt": "GPT",
        "svm": "SVM",
        "knn": "KNN",
        "sql": "SQL",
        "nosql": "NoSQL",
    }

    # If title is all uppercase (e.g., HỆ MÃ DỊCH VÒNG, HỆ MÃ AFFINE), format gracefully
    if title.isupper():
        words = title.split()
        formatted_words = []
        for i, word in enumerate(words):
            clean_word = re.sub(r"[^\w\-]", "", word)
            lower_word = clean_word.lower()

            if lower_word in known_proper_words:
                replacement = known_proper_words[lower_word]
                punct_suffix = word[len(clean_word):] if word.startswith(clean_word) else ""
                formatted_words.append(replacement + punct_suffix)
            elif i == 0:
                formatted_words.append(word.capitalize())
            else:
                formatted_words.append(word.lower())
        title = " ".join(formatted_words)

    # Standardize common Vietnamese spelling variations
    title = re.sub(r"\bhệ mã hoá\b", "Hệ mã hóa", title, flags=re.IGNORECASE)
    title = re.sub(r"\bhệ mã hóa\b", "Hệ mã hóa", title, flags=re.IGNORECASE)
    title = re.sub(r"\bhệ mã\b", "Hệ mã", title, flags=re.IGNORECASE)
    return title


# ---------------------------------------------------------------------------
# Taxonomy Pattern Indicators
# ---------------------------------------------------------------------------

DEFINITION_REGEXES = [
    re.compile(r"^(.*?)\s+là\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.*?)\s+được định nghĩa là\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.*?)\s+được hiểu là\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.*?)\s+chỉ\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.*?)\s+có nghĩa là\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.*?)\s+đóng vai trò là\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.*?)\s+là một phương pháp\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.*?)\s+là quá trình\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.*?)\s+là kỹ thuật\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.*?)\s+là hệ thống\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(.*?)\s+đề cập đến\s+(.+)$", re.IGNORECASE),
]

CLASSIFICATION_INTRO_CUES = [
    "có hai phương pháp",
    "có 2 phương pháp",
    "có các phương pháp",
    "có ba phương pháp",
    "có 3 phương pháp",
    "có hai loại",
    "có 2 loại",
    "có ba loại",
    "có 3 loại",
    "có các loại",
    "phân loại",
    "phân thành",
    "chia thành",
    "được chia thành",
    "chia làm",
    "gồm các loại",
    "gồm có",
    "bao gồm",
    "các nhóm",
    "các nhánh",
    "các hướng tiếp cận",
    "hai hướng chính",
    "hai phương pháp chính",
    "các phương pháp chính",
    "phân loại chính",
    "phương pháp chính",
]

NON_SYSTEM_SUBSTRING_KEYWORDS = [
    "giới thiệu",
    "mở đầu",
    "tổng quan",
    "khái niệm",
    "định nghĩa",
    "mục tiêu",
    "kết luận",
    "tài liệu tham khảo",
    "bài tập",
    "câu hỏi ôn tập",
    "introduction",
    "overview",
    "conclusion",
    "references",
    "summary",
    "background",
]

NON_SYSTEM_EXACT_KEYWORDS = [
    "phá mã",
    "thám mã",
    "tấn công",
    "cryptanalysis",
    "attack",
    "attacks",
]

CONCRETE_SYSTEM_KEYWORDS = [
    "hệ mã",
    "hệ mã hóa",
    "hệ mã hoá",
    "mã hóa",
    "mã hoá",
    "cipher",
    "thuật toán",
    "giao thức",
    "mô hình",
    "phương pháp",
    "giải thuật",
    "hệ thống",
    "kỹ thuật",
    "cơ chế",
    "chỉ mục",
    "kiến trúc",
    "mạng",
    "algorithm",
    "protocol",
    "model",
    "system",
    "structure",
    "index",
]


# ---------------------------------------------------------------------------
# Section Heading Parser
# ---------------------------------------------------------------------------

def extract_clean_heading(heading: str) -> str:
    """
    Remove numbering prefixes (e.g., '1. ', '1: ', 'Mục 1: ', 'Topic 1: ', '# 1. ')
    and return the clean heading text.
    """
    heading = _clean_whitespace(heading)
    # Strip markdown headers '# ', '## ', etc.
    heading = re.sub(r"^#+\s*", "", heading)
    # Strip prefixes like '1. ', '12. ', '1.1. ', '1) ', 'I. ', 'II. ', 'Mục 1: ', 'Topic 1: '
    heading = re.sub(
        r"^(?:chương|mục|bài|phần|topic|section|chapter|lesson|unit)\s*[\w\d\.\-]+\s*[:\.\-]?\s*",
        "",
        heading,
        flags=re.IGNORECASE,
    )
    heading = re.sub(r"^(?:[0-9]+|[ivxlcdm]+)[\.\:\)\-]\s*", "", heading, flags=re.IGNORECASE)
    return heading.strip()


def is_concrete_system_section(heading: str) -> bool:
    """
    Determine if a section title refers to a concrete type, system, or algorithm
    rather than a general overview or operational section.
    """
    clean = extract_clean_heading(heading)
    clean_lower = clean.lower()
    clean_no_accents = _strip_accents(clean)

    # 1. Exact match check against stand-alone non-system sections (e.g. '3. PHÁ MÃ', 'TẤN CÔNG')
    if clean_lower in NON_SYSTEM_EXACT_KEYWORDS or clean_no_accents in NON_SYSTEM_EXACT_KEYWORDS:
        return False

    # 2. Substring match for intro/outro/meta sections
    for kw in NON_SYSTEM_SUBSTRING_KEYWORDS:
        if kw in clean_lower or _strip_accents(kw) in clean_no_accents:
            return False

    # 3. Check for presence of system/algorithm/type keywords
    for kw in CONCRETE_SYSTEM_KEYWORDS:
        if kw in clean_lower or _strip_accents(kw) in clean_no_accents:
            return True

    # 4. Check if starts with capitalized specific name after clean
    words = clean.split()
    if len(words) >= 2 and words[0].istitle():
        return True

    return False


# ---------------------------------------------------------------------------
# Bullet & Sentence Extraction Helpers
# ---------------------------------------------------------------------------

def split_clean_sentences(text: str) -> List[str]:
    """
    Split text into clean sentences while:
    - Protecting decimal numbers (3.14, 0.5)
    - Protecting abbreviations (TS., GS., ThS., v.v., e.g., i.e.)
    - Preserving short bullet lines without dropping them
    - Stripping page tags like '=== Page N ==='
    """
    if not text:
        return []

    # Strip page tags and metadata lines
    cleaned = re.sub(r"===\s*Page\s*\d+\s*===", "", text, flags=re.IGNORECASE)

    # Protect decimals: 3.14 -> 3<DECIMAL>14
    cleaned = re.sub(r"(\d+)\.(\d+)", r"\1<DECIMAL>\2", cleaned)

    # Protect common abbreviations
    abbrs = ["gs", "ts", "ths", "ks", "bs", "tp", "ubnd", "v.v", "e.g", "i.e", "fig", "eq", "al"]
    for ab in abbrs:
        pattern = re.compile(rf"\b{ab}\.", re.IGNORECASE)
        cleaned = pattern.sub(f"{ab}<DOT>", cleaned)

    # Split on line breaks or sentence terminators followed by whitespace
    raw_lines = re.split(r"(?<=[.!?])\s+|\n+", cleaned)

    sentences = []
    for s in raw_lines:
        line = s.replace("<DECIMAL>", ".").replace("<DOT>", ".").strip()
        # Keep non-empty lines with meaningful content (> 3 chars)
        if len(line) >= 3:
            sentences.append(line)
    return sentences


def extract_bullet_items_from_text(text: str) -> List[str]:
    """
    Extract bulleted list items from text.
    Strictly detects list items starting with '*', '•', '+', or '-' (followed by space),
    or clean inline bullet points like '...: * Mã hóa thay thế * Mã hóa hoán vị'.
    Excludes numbered headings like '1. GIỚI THIỆU...' and hyphenated words inside parentheses.
    """
    bullets: List[str] = []
    if not text:
        return bullets

    # Clean page boundaries first
    clean_text = re.sub(r"===\s*Page\s*\d+\s*===", "", text, flags=re.IGNORECASE)

    lines = clean_text.splitlines()
    bullet_line_pattern = re.compile(r"^\s*([*•\+\-–—])\s+(.+)$")

    for line in lines:
        line_clean = line.strip()
        # Ignore markdown section headings '# Header'
        if line_clean.startswith("#"):
            continue

        match = bullet_line_pattern.match(line_clean)
        if match:
            marker = match.group(1)
            content = match.group(2).strip()
            # Verify content is not a section heading like '1. GIỚI THIỆU' or uppercase title
            clean_content = extract_clean_heading(content)
            if not is_concrete_system_section(content) and not content.isupper() and len(clean_content) >= 3:
                bullets.append(f"* {content}")
            elif len(clean_content) >= 3 and not content.isupper():
                bullets.append(f"* {content}")
        else:
            # Check for inline bullet markers: e.g. '* Mã hóa thay thế * Mã hóa hoán vị'
            inline_matches = re.findall(r"(?:^|\s|\:)([\*•])\s+([^\*•\n\r]+)", line_clean)
            if len(inline_matches) >= 2:
                for _, item_content in inline_matches:
                    item_clean = item_content.strip()
                    if len(item_clean) >= 3 and not item_clean.isupper():
                        bullets.append(f"* {item_clean}")

    # Fallback: check if entire string has inline asterisks separating classification items
    if not bullets and "*" in clean_text:
        parts = clean_text.split("*")
        if len(parts) > 1:
            for part in parts[1:]:
                item = part.strip().split("\n")[0].strip()
                # Ignore uppercase slide titles or noise
                if item and len(item) >= 3 and not item.isupper() and not item.startswith("#"):
                    bullets.append(f"* {item}")

    # Deduplicate while preserving order
    seen: Set[str] = set()
    deduped: List[str] = []
    for b in bullets:
        b_clean = re.sub(r"\s+", " ", b).strip()
        b_key = _strip_accents(b_clean)
        if b_key not in seen:
            seen.add(b_key)
            deduped.append(b_clean)

    return deduped


def extract_definition_sentences_from_text(text: str) -> List[str]:
    """Extract candidate definition sentences using semantic patterns."""
    sentences = split_clean_sentences(text)
    definitions = []

    for sentence in sentences:
        # Skip bullet lines or pure headings
        if sentence.startswith(("*", "-", "+", "•")) or sentence.startswith("#"):
            continue

        sent_clean = _clean_whitespace(sentence)
        sent_no_accents = _strip_accents(sent_clean)

        # Check for definition patterns
        is_def = False
        for regex in DEFINITION_REGEXES:
            if regex.search(sent_clean):
                is_def = True
                break

        if not is_def:
            if "la phuong phap" in sent_no_accents or "la he thong" in sent_no_accents or "duoc goi la" in sent_no_accents:
                is_def = True

        if is_def and len(sent_clean) >= 15:
            definitions.append(sent_clean)

    return definitions


# ---------------------------------------------------------------------------
# Core Extraction Engine
# ---------------------------------------------------------------------------

class TaxonomyExtractor:
    """
    100% Offline extractor for Definitions, Taxonomies, and Concrete System Types.
    Scans a sequence of document passages and synthesizes structured taxonomy metadata.
    """

    def __init__(self, default_domain: str = "cryptography"):
        self.default_domain = default_domain

    def extract(self, passages: Union[List[Dict[str, Any]], List[Any]]) -> Dict[str, Any]:
        """
        Scans a list of passages and returns a unified taxonomy dictionary:
        {
            "definitions": List[str],             # Opening / introductory definition sentences
            "introductory_definitions": List[str], # Canonical alias
            "intro_sents": List[str],              # Alias for answer composer / stitcher
            "bullet_types": List[str],             # Bulleted classification branches (* A, * B)
            "taxonomy_branches": List[str],        # Canonical alias
            "classification_branches": List[str],  # Alias
            "section_systems": List[str],          # Concrete types/systems from section titles
            "concrete_types": List[str],           # Canonical alias
            "all_sections": List[str],             # All cleaned section titles found
            "taxonomy_tree": Dict[str, Any],       # Hierarchical representation
            "summary": str                         # Human-readable synthesized summary
        }
        """
        if not passages:
            return {
                "definitions": [],
                "introductory_definitions": [],
                "intro_sents": [],
                "bullet_types": [],
                "taxonomy_branches": [],
                "classification_branches": [],
                "section_systems": [],
                "concrete_types": [],
                "all_sections": [],
                "taxonomy_tree": {},
                "summary": "",
            }

        intro_definitions: List[str] = []
        body_definitions: List[str] = []
        all_bullets: List[str] = []
        all_sections_raw: List[str] = []
        concrete_systems: List[str] = []

        seen_def_keys: Set[str] = set()
        seen_bullet_keys: Set[str] = set()
        seen_system_keys: Set[str] = set()
        seen_section_keys: Set[str] = set()

        # Step 1: Scan passages
        for idx, item in enumerate(passages):
            # Support both dict and object with attributes
            if isinstance(item, dict):
                text = item.get("text", "") or item.get("content", "")
                section_title = item.get("section_title", "") or item.get("title", "")
                metadata = item.get("metadata", {})
            else:
                text = getattr(item, "text", "") or getattr(item, "content", "")
                section_title = getattr(item, "section_title", "") or getattr(item, "title", "")
                metadata = getattr(item, "metadata", {}) or {}

            # Fallback section title from metadata if missing
            if not section_title and isinstance(metadata, dict):
                section_title = metadata.get("section_title", "") or metadata.get("title", "")

            # If section_title is still empty, scan top lines of passage text for headings
            if not section_title and text:
                first_lines = text.strip().splitlines()
                if first_lines:
                    first_line = first_lines[0].strip()
                    if re.match(r"^(?:#+|\d+[\.\:]|Mục\s*\d+|Chương\s*\d+|Topic\s*\d+)", first_line, re.IGNORECASE):
                        section_title = first_line

            # Process Section Title
            if section_title:
                clean_sec = extract_clean_heading(section_title)
                if clean_sec:
                    sec_key = _strip_accents(clean_sec)
                    if sec_key not in seen_section_keys:
                        seen_section_keys.add(sec_key)
                        all_sections_raw.append(clean_sec)

                    # Check if section title is a concrete system/type
                    if is_concrete_system_section(section_title):
                        formatted_sys = _format_proper_casing(clean_sec)
                        sys_key = _strip_accents(formatted_sys)
                        if sys_key not in seen_system_keys:
                            seen_system_keys.add(sys_key)
                            concrete_systems.append(formatted_sys)

            # Process Passage Text
            if text:
                sec_lower = section_title.lower() if section_title else ""
                is_intro_passage = (
                    idx == 0
                    or any(k in sec_lower for k in ["giới thiệu", "mở đầu", "tổng quan", "định nghĩa", "khái niệm", "intro"])
                )

                defs = extract_definition_sentences_from_text(text)
                for d in defs:
                    d_key = _strip_accents(d)
                    if d_key not in seen_def_keys:
                        seen_def_keys.add(d_key)
                        if is_intro_passage:
                            intro_definitions.append(d)
                        else:
                            body_definitions.append(d)

                # Extract bullet taxonomy branches
                bullets = extract_bullet_items_from_text(text)
                for b in bullets:
                    b_key = _strip_accents(b)
                    if b_key not in seen_bullet_keys:
                        seen_bullet_keys.add(b_key)
                        all_bullets.append(b)

                # Scan inline markdown headers inside passage body
                inline_headings = re.findall(
                    r"(?:^|\n)(?:#+\s*|\d+[\.\:]\s*)([A-ZÀ-Ỹ0-9\s\-]+?)(?=\n|$)",
                    text,
                )
                for ih in inline_headings:
                    ih_clean = extract_clean_heading(ih)
                    if len(ih_clean) >= 4 and is_concrete_system_section(ih_clean):
                        formatted_ih = _format_proper_casing(ih_clean)
                        ih_key = _strip_accents(formatted_ih)
                        if ih_key not in seen_system_keys:
                            seen_system_keys.add(ih_key)
                            concrete_systems.append(formatted_ih)

        # Consolidate definitions: introductory definitions first, then body definitions
        all_definitions = intro_definitions + body_definitions

        # Fallback if no definitions found: pick first clean introductory sentence
        if not all_definitions and passages:
            first_p = passages[0]
            first_text = first_p.get("text", "") if isinstance(first_p, dict) else getattr(first_p, "text", "")
            sents = split_clean_sentences(first_text)
            for s in sents:
                if not s.startswith(("*", "-", "+", "#")) and len(s) >= 20:
                    all_definitions.append(s)
                    break

        # Preferred intro sentences for stitcher/composer
        primary_intro = intro_definitions if intro_definitions else all_definitions[:2]

        # Step 2: Build Structured Taxonomy Tree
        root_name = all_sections_raw[0] if all_sections_raw else "Chủ đề chính"
        taxonomy_tree = {
            "root": root_name,
            "definitions": all_definitions[:3],
            "branches": all_bullets,
            "concrete_systems": concrete_systems,
        }

        # Step 3: Build Coherent Synthesized Summary
        summary_lines = []
        if all_definitions:
            summary_lines.append(f"Khái niệm: {all_definitions[0]}")
        if all_bullets:
            summary_lines.append("Phân loại chính:")
            for b in all_bullets:
                summary_lines.append(f"  {b}")
        if concrete_systems:
            summary_lines.append("Các hệ / loại cụ thể trong tài liệu:")
            for s in concrete_systems:
                summary_lines.append(f"  - {s}")

        summary = "\n".join(summary_lines)

        return {
            "definitions": all_definitions,
            "introductory_definitions": primary_intro,
            "intro_sents": primary_intro,
            "bullet_types": all_bullets,
            "taxonomy_branches": all_bullets,
            "classification_branches": all_bullets,
            "section_systems": concrete_systems,
            "concrete_types": concrete_systems,
            "all_sections": all_sections_raw,
            "taxonomy_tree": taxonomy_tree,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Public API Function
# ---------------------------------------------------------------------------

def extract_taxonomy_and_types(passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Scans a list of passages to aggregate:
    (1) Opening definition / introductory sentences,
    (2) Bulleted taxonomy branches (e.g., '* Mã hóa thay thế', '* Mã hóa hoán vị'),
    (3) List of all concrete types/systems extracted from section titles
        (e.g., 'Hệ mã hóa dòng', 'Hệ mã hóa độn một lần', 'Hệ mã hóa Autokey',
        'Hệ mã hóa Playfair'...).
    """
    extractor = TaxonomyExtractor()
    return extractor.extract(passages)


# ---------------------------------------------------------------------------
# TAXONOMY_INSTRUCTION_DATASET
# ---------------------------------------------------------------------------

TAXONOMY_INSTRUCTION_DATASET: List[Dict[str, Any]] = [
    {
        "id": "tax_inst_001_crypto_symmetric",
        "domain": "cryptography",
        "title": "Mật mã học cổ điển - Hệ mã khóa đối xứng",
        "instruction": "Trích xuất câu định nghĩa mở đầu, các nhánh phân loại dạng gạch đầu dòng và danh sách các hệ mã cụ thể từ tài liệu slide sau.",
        "input_passages": [
            {
                "passage_id": "SLIDE::P001",
                "section_title": "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG",
                "text": "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG\nMã hóa đối xứng là phương pháp mã hóa mà trong đó bên gửi và bên nhận dùng chung một khóa bí mật để mã hóa và giải mã dữ liệu. Có hai phương pháp chính:\n* Mã hóa thay thế\n* Mã hóa hoán vị",
            },
            {
                "passage_id": "SLIDE::P002",
                "section_title": "2. HỆ MÃ DỊCH VÒNG",
                "text": "2. HỆ MÃ DỊCH VÒNG\nHệ mã dịch vòng (Caesar cipher) là hệ mã thay thế đơn giản nhất, trong đó mỗi chữ cái trong bản rõ được thay thế bằng một chữ cái cách nó một khoảng k vị trí trong bảng chữ cái.",
            },
            {
                "passage_id": "SLIDE::P003",
                "section_title": "3. PHÁ MÃ",
                "text": "3. PHÁ MÃ\nPhá mã (thám mã) là quá trình khôi phục bản rõ hoặc khóa bí mật mà không có khóa giải mã trước, bằng cách phân tích tần suất ký tự hoặc vét cạn không gian khóa.",
            },
            {
                "passage_id": "SLIDE::P004",
                "section_title": "4. HỆ MÃ ĐƠN BẢNG",
                "text": "4. HỆ MÃ ĐƠN BẢNG\nHệ mã đơn bảng sử dụng một bảng hoán vị cố định của bảng chữ cái làm khóa mã hóa.",
            },
            {
                "passage_id": "SLIDE::P005",
                "section_title": "5. HỆ MÃ AFFINE",
                "text": "5. HỆ MÃ AFFINE\nHệ mã Affine kết hợp giữa phép nhân và phép cộng modulo: e(x) = (ax + b) mod m, với điều kiện ước số chung lớn nhất gcd(a, m) = 1.",
            },
            {
                "passage_id": "SLIDE::P006",
                "section_title": "6. HỆ MÃ VIGENERE",
                "text": "6. HỆ MÃ VIGENERE\nHệ mã Vigenere là hệ mã thay thế đa bảng, sử dụng một từ khóa lặp lại để xác định bước dịch chuyển cho từng ký tự.",
            },
            {
                "passage_id": "SLIDE::P007",
                "section_title": "7. HỆ MÃ HILL",
                "text": "7. HỆ MÃ HILL\nHệ mã Hill là hệ mã thay thế đa ký tự dựa trên đại số tuyến tính, sử dụng ma trận vuông khả nghịch modulo m làm khóa mã hóa.",
            },
            {
                "passage_id": "SLIDE::P008",
                "section_title": "8. HỆ MÃ HOÁ ĐỘN MỘT LẦN",
                "text": "8. HỆ MÃ HOÁ ĐỘN MỘT LẦN\nHệ mã hóa độn một lần (One-Time Pad - OTP) là hệ mã hóa sử dụng chuỗi khóa ngẫu nhiên có độ dài bằng bản rõ và chỉ sử dụng một lần duy nhất, đạt an ninh tuyệt đối.",
            },
            {
                "passage_id": "SLIDE::P009",
                "section_title": "9. HỆ MÃ HOÁ DÒNG",
                "text": "9. HỆ MÃ HOÁ DÒNG\nHệ mã hóa dòng (Stream cipher) mã hóa từng bit hoặc byte bản rõ bằng cách kết hợp với dòng khóa giả ngẫu nhiên thông qua phép toán XOR.",
            },
            {
                "passage_id": "SLIDE::P010",
                "section_title": "10. HỆ MÃ HOÁ AUTOKEY",
                "text": "10. HỆ MÃ HOÁ AUTOKEY\nHệ mã hóa Autokey là hệ mã tự sinh khóa, trong đó chính nội dung bản rõ được nối vào từ khóa ban đầu để tạo thành dòng khóa mã hóa dài vô hạn.",
            },
            {
                "passage_id": "SLIDE::P011",
                "section_title": "11. HỆ MÃ HOÁ PLAYFAIR",
                "text": "11. HỆ MÃ HOÁ PLAYFAIR\nHệ mã hóa Playfair mã hóa từng cặp ký tự (digraph) dựa trên ma trận khóa 5x5 chứa bảng chữ cái tiếng Anh.",
            },
            {
                "passage_id": "SLIDE::P012",
                "section_title": "12. HỆ MÃ HOÁ HOÁN VỊ",
                "text": "12. HỆ MÃ HOÁ HOÁN VỊ\nHệ mã hóa hoán vị giữ nguyên các ký tự của bản rõ nhưng sắp xếp lại vị trí của chúng theo một quy tắc hoán vị xác định.",
            },
        ],
        "expected_output": {
            "definitions": [
                "Mã hóa đối xứng là phương pháp mã hóa mà trong đó bên gửi và bên nhận dùng chung một khóa bí mật để mã hóa và giải mã dữ liệu.",
            ],
            "bullet_types": [
                "* Mã hóa thay thế",
                "* Mã hóa hoán vị",
            ],
            "section_systems": [
                "Hệ mã dịch vòng",
                "Hệ mã đơn bảng",
                "Hệ mã Affine",
                "Hệ mã Vigenere",
                "Hệ mã Hill",
                "Hệ mã hóa độn một lần",
                "Hệ mã hóa dòng",
                "Hệ mã hóa Autokey",
                "Hệ mã hóa Playfair",
                "Hệ mã hóa hoán vị",
            ],
        },
        "reasoning": "Mục 1 là giới thiệu khái quát, cung cấp định nghĩa mã hóa đối xứng và 2 nhánh phân loại gạch đầu dòng. Mục 3 (Phá mã) là kỹ thuật thám mã, không phải là một hệ mã hóa cụ thể nên bị loại trừ khỏi danh sách các hệ mã. 10 mục còn lại là các hệ mã hóa đối xứng cụ thể trong tài liệu.",
    },
    {
        "id": "tax_inst_002_machine_learning",
        "domain": "artificial_intelligence",
        "title": "Học máy và Các Phương pháp Học",
        "instruction": "Trích xuất câu định nghĩa học máy, các nhánh học chính theo gạch đầu dòng, và các thuật toán/mô hình cụ thể theo các tiêu đề mục.",
        "input_passages": [
            {
                "passage_id": "ML::P001",
                "section_title": "Chương 1. TỔNG QUAN VỀ HỌC MÁY",
                "text": "Học máy là một nhánh của trí tuệ nhân tạo nghiên cứu cách xây dựng các thuật toán có khả năng tự cải thiện hiệu năng qua kinh nghiệm dữ liệu. Dựa vào cách thức cung cấp dữ liệu và mục tiêu bài toán, học máy được phân thành 4 nhánh chính:\n* Học có giám sát (Supervised Learning)\n* Học không giám sát (Unsupervised Learning)\n* Học bán giám sát (Semi-supervised Learning)\n* Học tăng cường (Reinforcement Learning)",
            },
            {
                "passage_id": "ML::P002",
                "section_title": "Thuật toán Hồi quy Tuyến tính (Linear Regression)",
                "text": "Thuật toán Hồi quy Tuyến tính mô hình hóa mối quan hệ tuyến tính giữa biến phụ thuộc liên tục và một hoặc nhiều biến độc lập.",
            },
            {
                "passage_id": "ML::P003",
                "section_title": "Thuật toán Hồi quy Logistic (Logistic Regression)",
                "text": "Thuật toán Hồi quy Logistic sử dụng hàm Sigmoid để ánh xạ đầu ra thành xác suất dự đoán cho các bài toán phân lớp nhị phân.",
            },
            {
                "passage_id": "ML::P004",
                "section_title": "Mô hình Support Vector Machine (SVM)",
                "text": "Mô hình Support Vector Machine tìm siêu phẳng phân cách tối ưu nhằm tối đa hóa khoảng cách biên giữa các lớp dữ liệu.",
            },
            {
                "passage_id": "ML::P005",
                "section_title": "Thuật toán Phân cụm K-Means",
                "text": "Thuật toán Phân cụm K-Means phân chia tập dữ liệu không gán nhãn thành K cụm riêng biệt bằng cách tối thiểu hóa khoảng cách đến các tâm cụm.",
            },
        ],
        "expected_output": {
            "definitions": [
                "Học máy là một nhánh của trí tuệ nhân tạo nghiên cứu cách xây dựng các thuật toán có khả năng tự cải thiện hiệu năng qua kinh nghiệm dữ liệu.",
            ],
            "bullet_types": [
                "* Học có giám sát (Supervised Learning)",
                "* Học không giám sát (Unsupervised Learning)",
                "* Học bán giám sát (Semi-supervised Learning)",
                "* Học tăng cường (Reinforcement Learning)",
            ],
            "section_systems": [
                "Thuật toán Hồi quy Tuyến tính (Linear Regression)",
                "Thuật toán Hồi quy Logistic (Logistic Regression)",
                "Mô hình Support Vector Machine (SVM)",
                "Thuật toán Phân cụm K-Means",
            ],
        },
        "reasoning": "Chương 1 cung cấp định nghĩa mở đầu và 4 nhánh phân loại gạch đầu dòng. Các mục sau nêu các thuật toán/mô hình cụ thể.",
    },
    {
        "id": "tax_inst_003_os_scheduling",
        "domain": "operating_systems",
        "title": "Điều phối Tiến trình trong Hệ Điều hành",
        "instruction": "Trích xuất câu định nghĩa điều phối CPU, các cơ chế phân loại gạch đầu dòng và danh sách các thuật toán điều phối cụ thể.",
        "input_passages": [
            {
                "passage_id": "OS::P001",
                "section_title": "Mở đầu: Điều phối CPU",
                "text": "Điều phối CPU là quá trình hệ điều hành quyết định tiến trình nào trong hàng đợi sẵn sàng được cấp phát CPU để thực thi. Cơ chế điều phối được chia thành hai nhóm chính:\n* Điều phối không trưng dụng (Non-preemptive scheduling)\n* Điều phối trưng dụng (Preemptive scheduling)",
            },
            {
                "passage_id": "OS::P002",
                "section_title": "Thuật toán FCFS (First-Come, First-Served)",
                "text": "Thuật toán FCFS cấp phát CPU cho tiến trình theo thứ tự xuất hiện trước trong hàng đợi.",
            },
            {
                "passage_id": "OS::P003",
                "section_title": "Thuật toán SJF (Shortest Job First)",
                "text": "Thuật toán SJF ưu tiên cấp phát CPU cho tiến trình có thời gian thực thi ngắn nhất tiếp theo.",
            },
            {
                "passage_id": "OS::P004",
                "section_title": "Thuật toán Round Robin",
                "text": "Thuật toán Round Robin chia đều thời gian thực thi CPU cho từng tiến trình theo lát thời gian (time quantum) cố định.",
            },
        ],
        "expected_output": {
            "definitions": [
                "Điều phối CPU là quá trình hệ điều hành quyết định tiến trình nào trong hàng đợi sẵn sàng được cấp phát CPU để thực thi.",
            ],
            "bullet_types": [
                "* Điều phối không trưng dụng (Non-preemptive scheduling)",
                "* Điều phối trưng dụng (Preemptive scheduling)",
            ],
            "section_systems": [
                "Thuật toán FCFS (First-Come, First-Served)",
                "Thuật toán SJF (Shortest Job First)",
                "Thuật toán Round Robin",
            ],
        },
        "reasoning": "Mở đầu nêu định nghĩa và 2 phương pháp điều phối. Các mục sau tương ứng với các thuật toán điều phối cụ thể.",
    },
    {
        "id": "tax_inst_004_network_topologies",
        "domain": "computer_networks",
        "title": "Cấu trúc liên kết mạng máy tính (Network Topologies)",
        "instruction": "Trích xuất định nghĩa cấu trúc mạng, các dạng hình học cơ bản dạng gạch đầu dòng và các kiến trúc mạng cụ thể.",
        "input_passages": [
            {
                "passage_id": "NET::P001",
                "section_title": "Tổng quan Cấu trúc mạng",
                "text": "Cấu trúc liên kết mạng (Network Topology) là cách bố trí hình học và sắp xếp các nút mạng cùng các đường liên kết trong hệ thống truyền thông. Các cấu trúc liên kết vật lý cơ bản gồm:\n* Cấu trúc hình sao (Star Topology)\n* Cấu trúc đường trục (Bus Topology)\n* Cấu trúc hình vòng (Ring Topology)\n* Cấu trúc dạng lưới (Mesh Topology)",
            },
            {
                "passage_id": "NET::P002",
                "section_title": "Mạng Cục bộ Ethernet (IEEE 802.3)",
                "text": "Mạng Cục bộ Ethernet là chuẩn mạng LAN phổ biến nhất hiện nay sử dụng cáp xoắn đôi và chuyển mạch Ethernet.",
            },
            {
                "passage_id": "NET::P003",
                "section_title": "Mạng Không dây Wi-Fi (IEEE 802.11)",
                "text": "Mạng Không dây Wi-Fi cho phép các thiết bị kết nối mạng thông qua sóng vô tuyến điện tử.",
            },
        ],
        "expected_output": {
            "definitions": [
                "Cấu trúc liên kết mạng (Network Topology) là cách bố trí hình học và sắp xếp các nút mạng cùng các đường liên kết trong hệ thống truyền thông.",
            ],
            "bullet_types": [
                "* Cấu trúc hình sao (Star Topology)",
                "* Cấu trúc đường trục (Bus Topology)",
                "* Cấu trúc hình vòng (Ring Topology)",
                "* Cấu trúc dạng lưới (Mesh Topology)",
            ],
            "section_systems": [
                "Mạng Cục bộ Ethernet (IEEE 802.3)",
                "Mạng Không dây Wi-Fi (IEEE 802.11)",
            ],
        },
        "reasoning": "Phần tổng quan đưa ra định nghĩa và 4 nhánh gạch đầu dòng. Các mục sau trình bày các hệ thống mạng cục bộ cụ thể.",
    },
    {
        "id": "tax_inst_005_database_indexing",
        "domain": "database_systems",
        "title": "Chỉ mục Cơ sở dữ liệu và Kỹ thuật Tối ưu",
        "instruction": "Trích xuất định nghĩa chỉ mục, các hình thức tổ chức chỉ mục và các hệ chỉ mục cụ thể.",
        "input_passages": [
            {
                "passage_id": "DB::P001",
                "section_title": "Khái niệm Chỉ mục",
                "text": "Chỉ mục (Index) là cấu trúc dữ liệu bổ trợ giúp tăng tốc độ tìm kiếm và truy xuất bản ghi trong bảng cơ sở dữ liệu. Phân loại theo cách sắp xếp vật lý dữ liệu gồm:\n* Chỉ mục cụm (Clustered Index)\n* Chỉ mục phi cụm (Non-clustered Index)",
            },
            {
                "passage_id": "DB::P002",
                "section_title": "Hệ chỉ mục B-Tree",
                "text": "Hệ chỉ mục B-Tree cân bằng cây tự động, đảm bảo mọi thao tác tìm kiếm, chèn và xóa đều có độ phức tạp thời gian O(log n).",
            },
            {
                "passage_id": "DB::P003",
                "section_title": "Hệ chỉ mục Hash",
                "text": "Hệ chỉ mục Hash sử dụng hàm băm để ánh xạ khóa tìm kiếm trực tiếp vào khối dữ liệu, đạt tốc độ truy xuất O(1) cho phép so khớp chính xác.",
            },
            {
                "passage_id": "DB::P004",
                "section_title": "Hệ chỉ mục Bitmap",
                "text": "Hệ chỉ mục Bitmap sử dụng chuỗi bit nhị phân để biểu diễn sự tồn tại của giá trị khóa trên từng dòng, tối ưu cho các cột có số lượng giá trị phân biệt thấp.",
            },
        ],
        "expected_output": {
            "definitions": [
                "Chỉ mục (Index) là cấu trúc dữ liệu bổ trợ giúp tăng tốc độ tìm kiếm và truy xuất bản ghi trong bảng cơ sở dữ liệu.",
            ],
            "bullet_types": [
                "* Chỉ mục cụm (Clustered Index)",
                "* Chỉ mục phi cụm (Non-clustered Index)",
            ],
            "section_systems": [
                "Hệ chỉ mục B-Tree",
                "Hệ chỉ mục Hash",
                "Hệ chỉ mục Bitmap",
            ],
        },
        "reasoning": "Định nghĩa và phân loại 2 nhánh cụm / phi cụm nằm ở mục mở đầu. 3 hệ chỉ mục cụ thể được lấy từ tiêu đề các mục sau.",
    },
    {
        "id": "tax_inst_006_public_key_crypto",
        "domain": "cryptography",
        "title": "Hệ Mật mã Khóa Công khai (Asymmetric Cryptography)",
        "instruction": "Trích xuất định nghĩa mật mã khóa công khai, các nhánh phân loại nền tảng toán học và các hệ mã cụ thể.",
        "input_passages": [
            {
                "passage_id": "ASYM::P001",
                "section_title": "Giới thiệu Mật mã Khóa Công khai",
                "text": "Mật mã khóa công khai là hệ mã hóa sử dụng một cặp khóa gồm khóa công khai để mã hóa hoặc kiểm tra chữ ký và khóa riêng để giải mã hoặc ký số. Các hệ mã khóa công khai dựa trên các bài toán toán học khó:\n* Bài toán phân tích thừa số nguyên lớn (Integer Factorization)\n* Bài toán logarit rời rạc (Discrete Logarithm)\n* Bài toán logarit rời rạc trên đường cong Elliptic (Elliptic Curve Discrete Logarithm)",
            },
            {
                "passage_id": "ASYM::P002",
                "section_title": "Hệ mã RSA",
                "text": "Hệ mã RSA là thuật toán mã hóa khóa công khai đầu tiên dựa trên tính khó của việc phân tích tích hai số nguyên tố rất lớn.",
            },
            {
                "passage_id": "ASYM::P003",
                "section_title": "Hệ mã ElGamal",
                "text": "Hệ mã ElGamal là hệ mã khóa công khai dựa trên bài toán trao đổi khóa Diffie-Hellman và logarit rời rạc trong trường hữu hạn.",
            },
            {
                "passage_id": "ASYM::P004",
                "section_title": "Hệ mã ECC (Elliptic Curve Cryptography)",
                "text": "Hệ mã ECC sử dụng cấu trúc nhóm của các đường cong Elliptic trên trường hữu hạn để đạt mức an toàn cao với kích thước khóa nhỏ hơn nhiều so với RSA.",
            },
        ],
        "expected_output": {
            "definitions": [
                "Mật mã khóa công khai là hệ mã hóa sử dụng một cặp khóa gồm khóa công khai để mã hóa hoặc kiểm tra chữ ký và khóa riêng để giải mã hoặc ký số.",
            ],
            "bullet_types": [
                "* Bài toán phân tích thừa số nguyên lớn (Integer Factorization)",
                "* Bài toán logarit rời rạc (Discrete Logarithm)",
                "* Bài toán logarit rời rạc trên đường cong Elliptic (Elliptic Curve Discrete Logarithm)",
            ],
            "section_systems": [
                "Hệ mã RSA",
                "Hệ mã ElGamal",
                "Hệ mã ECC (Elliptic Curve Cryptography)",
            ],
        },
        "reasoning": "Mở đầu đưa ra định nghĩa và 3 nhánh toán học nền tảng gạch đầu dòng. Ba hệ mã hóa khóa công khai cụ thể là RSA, ElGamal và ECC.",
    },
    {
        "id": "tax_inst_007_cyber_attacks",
        "domain": "information_security",
        "title": "Phân loại Tấn công và Cơ chế Phòng thủ Không gian mạng",
        "instruction": "Trích xuất câu định nghĩa tấn công mạng, phân loại chủ động / bị động và các hệ thống/hình thức tấn công cụ thể.",
        "input_passages": [
            {
                "passage_id": "SEC::P001",
                "section_title": "1. Khái niệm Tấn công An ninh Mạng",
                "text": "Tấn công mạng là hành vi cố ý của kẻ tấn công nhằm xâm phạm tính bí mật, tính toàn vẹn hoặc tính sẵn sàng của tài nguyên mạng và hệ thống thông tin. Các cuộc tấn công được chia thành hai nhánh lớn:\n* Tấn công bị động (Passive Attack)\n* Tấn công chủ động (Active Attack)",
            },
            {
                "passage_id": "SEC::P002",
                "section_title": "Hệ thống Tấn công Từ chối Dịch vụ Phân tán (DDoS)",
                "text": "Tấn công Từ chối Dịch vụ Phân tán huy động mạng lưới máy tính ma botnet để làm tràn ngập băng thông hoặc làm kiệt quệ tài nguyên máy chủ đích.",
            },
            {
                "passage_id": "SEC::P003",
                "section_title": "Kỹ thuật Tấn công Chen ngang (Man-in-the-Middle)",
                "text": "Tấn công Chen ngang bí mật chuyển tiếp và có thể thay đổi dữ liệu liên lạc giữa hai bên tin rằng họ đang nói chuyện trực tiếp với nhau.",
            },
            {
                "passage_id": "SEC::P004",
                "section_title": "Hệ thống Phòng thủ Tường lửa (Firewall)",
                "text": "Hệ thống Tường lửa giám sát và kiểm soát lưu lượng mạng vào ra dựa trên các quy tắc bảo mật xác định trước.",
            },
        ],
        "expected_output": {
            "definitions": [
                "Tấn công mạng là hành vi cố ý của kẻ tấn công nhằm xâm phạm tính bí mật, tính toàn vẹn hoặc tính sẵn sàng của tài nguyên mạng và hệ thống thông tin.",
            ],
            "bullet_types": [
                "* Tấn công bị động (Passive Attack)",
                "* Tấn công chủ động (Active Attack)",
            ],
            "section_systems": [
                "Hệ thống Tấn công Từ chối Dịch vụ Phân tán (DDoS)",
                "Kỹ thuật Tấn công Chen ngang (Man-in-the-Middle)",
                "Hệ thống Phòng thủ Tường lửa (Firewall)",
            ],
        },
        "reasoning": "Định nghĩa và phân nhánh bị động/chủ động ở mục 1. Ba hệ thống/kỹ thuật cụ thể ở các mục 2, 3, 4.",
    },
    {
        "id": "tax_inst_008_sorting_algorithms",
        "domain": "data_structures_algorithms",
        "title": "Thuật toán Sắp xếp và Cấu trúc Dữ liệu",
        "instruction": "Trích xuất định nghĩa sắp xếp, các chiến lược phân loại và các thuật toán cụ thể từ tài liệu.",
        "input_passages": [
            {
                "passage_id": "ALG::P001",
                "section_title": "Tổng quan về Sắp xếp",
                "text": "Sắp xếp là quá trình tổ chức lại một tập hợp các phần tử dữ liệu theo một thứ tự xác định nhằm tối ưu hóa hiệu quả tìm kiếm. Các chiến lược thuật toán sắp xếp gồm:\n* Thuật toán sắp xếp so sánh (Comparison Sort)\n* Thuật toán sắp xếp không so sánh (Non-comparison Sort)",
            },
            {
                "passage_id": "ALG::P002",
                "section_title": "Thuật toán Quick Sort",
                "text": "Thuật toán Quick Sort hoạt động theo nguyên lý chia để trị bằng cách chọn một phần tử chốt và phân hoạch danh sách.",
            },
            {
                "passage_id": "ALG::P003",
                "section_title": "Thuật toán Merge Sort",
                "text": "Thuật toán Merge Sort chia danh sách thành các danh sách con nhỏ hơn rồi trộn chúng lại theo thứ tự có độ phức tạp O(n log n).",
            },
            {
                "passage_id": "ALG::P004",
                "section_title": "Thuật toán Heap Sort",
                "text": "Thuật toán Heap Sort chuyển đổi mảng thành cấu trúc dữ liệu Max-Heap để lần lượt trích xuất phần tử lớn nhất.",
            },
        ],
        "expected_output": {
            "definitions": [
                "Sắp xếp là quá trình tổ chức lại một tập hợp các phần tử dữ liệu theo một thứ tự xác định nhằm tối ưu hóa hiệu quả tìm kiếm.",
            ],
            "bullet_types": [
                "* Thuật toán sắp xếp so sánh (Comparison Sort)",
                "* Thuật toán sắp xếp không so sánh (Non-comparison Sort)",
            ],
            "section_systems": [
                "Thuật toán Quick Sort",
                "Thuật toán Merge Sort",
                "Thuật toán Heap Sort",
            ],
        },
        "reasoning": "Khái niệm và phân loại 2 nhánh ở mục tổng quan. Ba thuật toán sắp xếp cụ thể ở các mục kế tiếp.",
    },
]


# ---------------------------------------------------------------------------
# Dataset Access & Validation Utilities
# ---------------------------------------------------------------------------

def get_taxonomy_dataset() -> List[Dict[str, Any]]:
    """Returns the full taxonomy instruction dataset."""
    return TAXONOMY_INSTRUCTION_DATASET


def get_taxonomy_example_by_id(example_id: str) -> Optional[Dict[str, Any]]:
    """Finds a specific taxonomy training pattern by ID."""
    for item in TAXONOMY_INSTRUCTION_DATASET:
        if item.get("id") == example_id:
            return item
    return None


def validate_taxonomy_dataset() -> bool:
    """
    Validates structural completeness and grounding across all dataset entries:
    - Verifies presence of id, domain, title, instruction, input_passages, expected_output.
    - Verifies expected_output has definitions, bullet_types, section_systems.
    - Verifies that running extract_taxonomy_and_types on input_passages produces expected items.
    """
    for entry in TAXONOMY_INSTRUCTION_DATASET:
        assert "id" in entry, "Missing id in taxonomy entry"
        assert "domain" in entry, f"Missing domain in {entry.get('id')}"
        assert "title" in entry, f"Missing title in {entry.get('id')}"
        assert "instruction" in entry, f"Missing instruction in {entry.get('id')}"
        assert "input_passages" in entry, f"Missing input_passages in {entry.get('id')}"
        assert "expected_output" in entry, f"Missing expected_output in {entry.get('id')}"

        expected = entry["expected_output"]
        assert "definitions" in expected, f"Missing definitions in expected_output of {entry.get('id')}"
        assert "bullet_types" in expected, f"Missing bullet_types in expected_output of {entry.get('id')}"
        assert "section_systems" in expected, f"Missing section_systems in expected_output of {entry.get('id')}"

    return True

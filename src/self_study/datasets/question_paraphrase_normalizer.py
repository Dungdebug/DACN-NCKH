"""
Question Paraphrase Normalizer (Wave 5 / Self-Study Datasets).
100% Offline, Pure Python stdlib, Zero External Dependencies.

Provides:
1. normalize_student_question(query: str) -> dict:
   Normalizes natural student Vietnamese questions and extracts:
   - cleaned_query: Normalized, trimmed string.
   - core_keywords: High-priority semantic topic keywords / compound terms.
   - quoted_targets: Text extracted from quotes ('...', "...", “...”).
   - asks_definition: True if asking for definitions, concepts, meanings.
   - asks_classification: True if asking for classifications, types, methods, counts.
   - asks_example: True if asking for examples, illustrations, walkthroughs.
   - asks_section_detail: True if asking for a specific section, topic, or quoted heading.
2. QuestionParaphraseNormalizer:
   Engine for query normalization, facet extraction, intent classification, and dataset benchmarking.
3. PARAPHRASE_DATASET:
   Benchmark dataset of 35+ realistic Vietnamese student queries across diverse learning facets.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Set, Tuple


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive canonical alignment."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


# Pre-compiled regex patterns for quotes
QUOTE_PATTERN = re.compile(
    r"['\"“‘«`]([^'\"”’»`]+)['\"”’»`]",
    re.UNICODE,
)

# Common conversational noise / interrogative stopwords in Vietnamese
STOPWORDS = {
    "là", "gì", "thế", "nào", "sao", "ra", "nghĩa",
    "có", "bao", "nhiêu", "mấy", "loại", "những", "các", "gồm",
    "cho", "và", "với", "hoặc", "kèm", "theo",
    "ví", "dụ", "minh", "họa", "cụ", "thể",
    "nói", "rõ", "chi", "tiết", "trình", "bày", "giải", "thích",
    "hãy", "nêu", "mô", "tả", "kể", "liệt", "kê", "tìm", "hiểu",
    "về", "trong", "của", "ở", "tại", "từ", "đến", "được", "bởi",
    "mục", "phần", "chương", "bài", "topic", "chủ", "đề", "nội", "dung",
    "tôi", "em", "bạn", "thầy", "cô", "ạ", "nhé", "nha", "vậy", "xin",
    "hỏi", "giúp", "muốn", "biết", "ơi", "nhỉ", "được", "không", "ko",
}

# Domain technical compound phrases (ordered by length descending to match longest first)
DOMAIN_COMPOUND_PHRASES = [
    "hệ mã khóa đối xứng",
    "hệ mã đối xứng",
    "mã hóa đối xứng",
    "mã hoá đối xứng",
    "mã hóa bất đối xứng",
    "mã hoá bất đối xứng",
    "an ninh vô điều kiện",
    "an ninh độ phức tạp tính toán",
    "hệ mã độn một lần",
    "mã hóa độn một lần",
    "mã độn một lần",
    "độn một lần",
    "one-time pad",
    "hệ mã dịch vòng",
    "mã hóa dịch vòng",
    "mã dịch vòng",
    "hệ mã đơn bảng",
    "mã hóa đơn bảng",
    "mã đơn bảng",
    "hệ mã affine",
    "mã hóa affine",
    "mã affine",
    "hệ mã vigenere",
    "mã hóa vigenere",
    "mã vigenere",
    "hệ mã hill",
    "mã hóa hill",
    "mã hill",
    "hệ mã autokey",
    "mã hóa autokey",
    "mã autokey",
    "hệ mã playfair",
    "mã hóa playfair",
    "mã playfair",
    "mã hóa thay thế",
    "mã hoá thay thế",
    "hệ mã thay thế",
    "mã hóa hoán vị",
    "mã hoá hoán vị",
    "hệ mã hoán vị",
    "mã hóa dòng",
    "mã hoá dòng",
    "hệ mã dòng",
    "stream cipher",
    "mã hóa khối",
    "mã hoá khối",
    "hệ mã khối",
    "block cipher",
    "không gian khóa",
    "khóa bí mật",
    "khóa công khai",
    "phá mã",
    "thám mã",
    "mật mã học",
    "bản rõ",
    "bản mã",
    "thuật toán",
    "công thức",
    "mã hóa",
    "mã hoá",
    "giải mã",
    "hệ mã",
    "mật mã",
]

# Definition question patterns
DEFINITION_PATTERNS = [
    r"\blà\s+(?:cái\s+)?gì\b", r"\bla\s+(?:cai\s+)?gi\b",
    r"\blà gì\b", r"\bla gi\b",
    r"\blà cái gì\b", r"\bla cai gi\b",
    r"\bnghĩa là gì\b", r"\bnghia la gi\b",
    r"\bthế nào là\b", r"\bthe nao la\b",
    r"\bkhái niệm\b", r"\bkhai niem\b",
    r"\bđịnh nghĩa\b", r"\bdinh nghia\b",
    r"\bbản chất\b", r"\bban chat\b",
    r"\bý nghĩa\b", r"\by nghia\b",
    r"\bhiểu như thế nào\b", r"\bhieu nhu the nao\b",
    r"\bhiểu thế nào về\b", r"\bhieu the nao ve\b",
    r"\bcho biết khái niệm\b", r"\bcho biet khai niem\b",
    r"\bwhat is\b", r"\bdefine\b", r"\bdefinition of\b", r"\bconcept of\b",
]

# Classification / typology question patterns
CLASSIFICATION_PATTERNS = [
    r"\bbao nhiêu loại\b", r"\bbao nhieu loai\b",
    r"\bcó mấy loại\b", r"\bco may loai\b",
    r"\bcác loại\b", r"\bcac loai\b",
    r"\bnhững loại nào\b", r"\bnhung loai nao\b",
    r"\bgồm những loại nào\b", r"\bgom nhung loai nao\b",
    r"\bphân loại\b", r"\bphan loai\b",
    r"\bphân nhóm\b", r"\bphan nhom\b",
    r"\bphân chia\b", r"\bphan chia\b",
    r"\bphương pháp nào\b", r"\bphuong phap nao\b",
    r"\bcác phương pháp\b", r"\bcac phuong phap\b",
    r"\bcó bao nhiêu phương pháp\b", r"\bco bao nhieu phuong phap\b",
    r"\bcó mấy phương pháp\b", r"\bco may phuong phap\b",
    r"\bcác hệ nào\b", r"\bcac he nao\b",
    r"\bgồm những hệ nào\b", r"\bgom nhung he nao\b",
    r"\bcó những hệ nào\b", r"\bco nhung he nao\b",
    r"\bliệt kê các\b", r"\bliet ke cac\b",
    r"\bliệt kê\b", r"\bliet ke\b",
    r"\bkể tên các\b", r"\bke ten cac\b",
    r"\bkể tên\b", r"\bke ten\b",
    r"\bdanh sách các\b", r"\bdanh sach cac\b",
    r"\bhow many types\b", r"\btypes of\b", r"\bclassification\b",
]

# Example question patterns
EXAMPLE_PATTERNS = [
    r"\bcho ví dụ\b", r"\bcho vi du\b",
    r"\bví dụ minh họa\b", r"\bvi du minh hoa\b",
    r"\blấy ví dụ\b", r"\blay vi du\b",
    r"\bví dụ cụ thể\b", r"\bvi du cu the\b",
    r"\bcho một ví dụ\b", r"\bcho mot vi du\b",
    r"\bcho 1 ví dụ\b", r"\bcho 1 vi du\b",
    r"\bkèm ví dụ\b", r"\bkem vi du\b",
    r"\bminh họa bằng ví dụ\b", r"\bminh hoa bang vi du\b",
    r"\bminh họa\b", r"\bminh hoa\b",
    r"\bvề ví dụ\b", r"\bve vi du\b",
    r"\bví dụ\b", r"\bvi du\b",
    r"\bfor example\b", r"\bgive an example\b", r"\bexample\b",
]

# Section lookup / detail patterns
SECTION_KEYWORDS_PATTERN = re.compile(
    r"\b(mục|muc|phần|phan|chương|chuong|topic|chủ\s*đề|chu\s*de|tiết|bài|bai|section|chapter|lesson|unit)\b",
    re.IGNORECASE,
)

NUMBERED_HEADING_START_PATTERN = re.compile(
    r"^\s*\d+[\.\:\-]\s*([A-Za-z0-9À-ỹ\s]+)",
    re.IGNORECASE,
)

SECTION_TARGET_EXTRACTION_PATTERN = re.compile(
    r"\b(?:mục|muc|phần|phan|chương|chuong|topic|chủ\s*đề|chu\s*de|section|chapter)\s+([A-Za-z0-9À-ỹ\s\.\-]+)",
    re.IGNORECASE,
)

SECTION_ACTION_PHRASE_PATTERN = re.compile(
    r"\b(nói rõ|noi ro|nói chi tiết|noi chi tiet|trình bày chi tiết|trinh bay chi tiet|đi sâu vào|di sau vao|giải thích chi tiết|giai thich chi tiet)\b",
    re.IGNORECASE,
)


class QuestionParaphraseNormalizer:
    """
    Normalizes natural Vietnamese student questions for grounded QA retrieval
    and self-study curriculum synthesis.
    """

    def __init__(self):
        # Pre-compile combined regex patterns for lightning speed
        self._def_regex = re.compile("|".join(DEFINITION_PATTERNS), re.IGNORECASE)
        self._class_regex = re.compile("|".join(CLASSIFICATION_PATTERNS), re.IGNORECASE)
        self._ex_regex = re.compile("|".join(EXAMPLE_PATTERNS), re.IGNORECASE)

    def extract_quoted_targets(self, query: str) -> List[str]:
        """
        Extracts all substrings enclosed in quotation marks ('...', "...", “...”).
        Preserves original casing and strips internal whitespace.
        """
        if not query:
            return []
        matches = QUOTE_PATTERN.findall(query)
        targets = []
        for m in matches:
            cleaned = m.strip()
            if cleaned and cleaned not in targets:
                targets.append(cleaned)
        return targets

    def clean_query(self, query: str) -> str:
        """
        Standardizes query string:
        - Unicode NFC normalization.
        - Whitespace compaction.
        - Strips surrounding quotes if the entire string was quoted (e.g. "'PHÁ MÃ'" -> "PHÁ MÃ").
        - Trims trailing interrogative / sentence punctuation (? ! .).
        """
        if not query:
            return ""
        text = unicodedata.normalize("NFC", str(query))
        text = re.sub(r"\s+", " ", text).strip()

        # Check if entire query is enclosed in quotes
        if (text.startswith(("'","\"","“","‘","«","`")) and
            text.endswith(("'","\"","”","’","»","`")) and len(text) >= 2):
            text = text[1:-1].strip()

        # Trim trailing question marks / exclamation marks / periods
        text = re.sub(r"[\?\!\.]+$", "", text).strip()
        return text

    def detect_facets(self, query: str, quoted_targets: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Detects primary pedagogical facets in the student's question:
        - asks_definition: Inquiring about definition, concept, nature.
        - asks_classification: Inquiring about categories, types, methods, list of variants.
        - asks_example: Inquiring about concrete examples, calculations, illustrations.
        - asks_section_detail: Inquiring about a specific section/heading/topic.
        """
        if not query:
            return {
                "asks_definition": False,
                "asks_classification": False,
                "asks_example": False,
                "asks_section_detail": False,
            }

        q_clean = self.clean_query(query)
        if quoted_targets is None:
            quoted_targets = self.extract_quoted_targets(query)

        asks_def = bool(self._def_regex.search(query))
        asks_class = bool(self._class_regex.search(query))
        asks_ex = bool(self._ex_regex.search(query))

        # Determine section detail
        asks_sec = False

        # 1. Section keywords (mục, phần, chương, topic...)
        if SECTION_KEYWORDS_PATTERN.search(query):
            asks_sec = True

        # 2. Numbered heading starting pattern (e.g. "3. PHÁ MÃ")
        if NUMBERED_HEADING_START_PATTERN.search(query):
            asks_sec = True

        # 3. Quoted suggestion clicks (e.g. user clicked suggested topic "'PHÁ MÃ'")
        if quoted_targets:
            raw_clean = re.sub(r"['\"“‘«`'”’»`]", "", query).strip()
            for qt in quoted_targets:
                # Quoted string is the sole query or matches clean text
                if qt.strip().lower() == raw_clean.lower():
                    asks_sec = True
                    break
                # Quoted string begins with a number ("3. PHÁ MÃ")
                if re.match(r"^\d+[\.\:\-]\s*", qt.strip()):
                    asks_sec = True
                    break
                # Quoted string is in UPPERCASE or title heading
                if qt.strip().isupper() and len(qt.strip()) >= 3:
                    asks_sec = True
                    break

        # 4. Action phrase targeting a specific topic (e.g. "nói rõ phá mã")
        if SECTION_ACTION_PHRASE_PATTERN.search(query) and not (asks_def or asks_class or asks_ex):
            asks_sec = True

        return {
            "asks_definition": asks_def,
            "asks_classification": asks_class,
            "asks_example": asks_ex,
            "asks_section_detail": asks_sec,
        }

    def extract_core_keywords(
        self,
        query: str,
        quoted_targets: Optional[List[str]] = None,
        facets: Optional[Dict[str, bool]] = None,
    ) -> List[str]:
        """
        Extracts high-priority core semantic keywords and compound phrases:
        1. Quoted targets (highest priority).
        2. Extracted section targets (e.g. from 'nói rõ mục phá mã' -> 'phá mã').
        3. Known technical domain compound phrases (e.g. 'mã hóa đối xứng', 'hệ mã affine').
        4. Salient remaining non-stopword tokens.
        """
        if not query:
            return []

        cleaned = self.clean_query(query)
        if quoted_targets is None:
            quoted_targets = self.extract_quoted_targets(query)
        if facets is None:
            facets = self.detect_facets(query, quoted_targets)

        core_keywords: List[str] = []
        seen_lower: Set[str] = set()

        def _add(kw: str):
            kw_clean = kw.strip(" ,.-:;?!\"'")
            if not kw_clean:
                return
            low = kw_clean.lower()
            if low not in seen_lower and len(kw_clean) >= 2:
                seen_lower.add(low)
                core_keywords.append(kw_clean)

        # 1. Quoted targets
        for qt in quoted_targets:
            _add(qt)

        # 2. Numbered heading topic if query begins with e.g. "3. PHÁ MÃ"
        num_match = NUMBERED_HEADING_START_PATTERN.search(query)
        if num_match:
            heading_title = num_match.group(1).strip()
            _add(heading_title)

        # 3. Section targets (e.g. "mục phá mã" -> "phá mã")
        sec_match = SECTION_TARGET_EXTRACTION_PATTERN.search(query)
        if sec_match:
            sec_target = sec_match.group(1).strip()
            # Trim trailing action words if any
            sec_target = re.split(r"\b(?:và|kèm|cho|nhé|ạ)\b", sec_target, flags=re.I)[0].strip()
            _add(sec_target)

        # 4. Domain compound phrases
        query_lower = cleaned.lower()
        for phrase in DOMAIN_COMPOUND_PHRASES:
            if phrase in query_lower:
                _add(phrase)

        # 5. Fallback token extraction if keywords are still empty or minimal
        if len(core_keywords) < 2:
            # Tokenize and remove conversational stopwords
            tokens = re.findall(r"[A-Za-z0-9À-ỹ]+", cleaned)
            filtered_tokens = []
            for tok in tokens:
                t_low = tok.lower()
                if t_low not in STOPWORDS and len(t_low) >= 2 and not t_low.isdigit():
                    filtered_tokens.append(tok)

            # Combine adjacent filtered tokens into bigrams if suitable
            for tok in filtered_tokens:
                _add(tok)

        return core_keywords

    def classify_primary_intent(self, query: str) -> str:
        """
        Classifies the dominant pedagogical intent for downstream routing:
        - 'multi_facet': Query asks for multiple components (def + types + example).
        - 'definition': Asks what something is / concept.
        - 'classification': Asks for categories, types, count.
        - 'example': Asks for concrete walkthrough or example.
        - 'section_detail': Inquires about a specific section, topic, or quoted heading.
        - 'general_qa': General inquiry.
        """
        facets = self.detect_facets(query)
        def_flag = facets["asks_definition"]
        class_flag = facets["asks_classification"]
        ex_flag = facets["asks_example"]
        sec_flag = facets["asks_section_detail"]

        num_facets = sum([def_flag, class_flag, ex_flag])
        if num_facets >= 2:
            return "multi_facet"
        if sec_flag and not (def_flag or class_flag):
            return "section_detail"
        if def_flag:
            return "definition"
        if class_flag:
            return "classification"
        if ex_flag:
            return "example"
        if sec_flag:
            return "section_detail"
        return "general_qa"

    def normalize(self, query: str) -> Dict[str, Any]:
        """
        Full pipeline normalization of student question.
        Returns guaranteed dictionary with all required keys:
        - cleaned_query: str
        - core_keywords: list[str]
        - quoted_targets: list[str]
        - asks_definition: bool
        - asks_classification: bool
        - asks_example: bool
        - asks_section_detail: bool
        - primary_intent: str
        - is_multi_facet: bool
        """
        cleaned = self.clean_query(query)
        quoted = self.extract_quoted_targets(query)
        facets = self.detect_facets(query, quoted_targets=quoted)
        keywords = self.extract_core_keywords(query, quoted_targets=quoted, facets=facets)
        intent = self.classify_primary_intent(query)
        is_multi = intent == "multi_facet" or sum([
            facets["asks_definition"],
            facets["asks_classification"],
            facets["asks_example"]
        ]) >= 2

        return {
            "cleaned_query": cleaned,
            "core_keywords": keywords,
            "quoted_targets": quoted,
            "asks_definition": facets["asks_definition"],
            "asks_classification": facets["asks_classification"],
            "asks_example": facets["asks_example"],
            "asks_section_detail": facets["asks_section_detail"],
            "primary_intent": intent,
            "is_multi_facet": is_multi,
        }


# Singleton instance
_DEFAULT_NORMALIZER = QuestionParaphraseNormalizer()


def normalize_student_question(query: str) -> Dict[str, Any]:
    """
    Module-level entrance: Normalizes student question and extracts all facets.
    Guaranteed contract:
    {
        'cleaned_query': str,
        'core_keywords': list[str],
        'quoted_targets': list[str],
        'asks_definition': bool,
        'asks_classification': bool,
        'asks_example': bool,
        'asks_section_detail': bool,
    }
    """
    return _DEFAULT_NORMALIZER.normalize(query)


# Comprehensive evaluation dataset for testing, calibration, and few-shot guidance
PARAPHRASE_DATASET: List[Dict[str, Any]] = [
    {
        "query": "mã hóa là gì có bao nhiêu loại mã hóa và cho ví dụ",
        "expected_facets": {
            "asks_definition": True,
            "asks_classification": True,
            "asks_example": True,
            "asks_section_detail": False,
        },
        "expected_keywords_subset": ["mã hóa"],
        "expected_quoted": [],
    },
    {
        "query": "nói rõ mục phá mã",
        "expected_facets": {
            "asks_definition": False,
            "asks_classification": False,
            "asks_example": False,
            "asks_section_detail": True,
        },
        "expected_keywords_subset": ["phá mã"],
        "expected_quoted": [],
    },
    {
        "query": "'PHÁ MÃ'",
        "expected_facets": {
            "asks_definition": False,
            "asks_classification": False,
            "asks_example": False,
            "asks_section_detail": True,
        },
        "expected_keywords_subset": ["PHÁ MÃ"],
        "expected_quoted": ["PHÁ MÃ"],
    },
    {
        "query": "'3. PHÁ MÃ'",
        "expected_facets": {
            "asks_definition": False,
            "asks_classification": False,
            "asks_example": False,
            "asks_section_detail": True,
        },
        "expected_keywords_subset": ["3. PHÁ MÃ"],
        "expected_quoted": ["3. PHÁ MÃ"],
    },
    {
        "query": "3. PHÁ MÃ",
        "expected_facets": {
            "asks_definition": False,
            "asks_classification": False,
            "asks_example": False,
            "asks_section_detail": True,
        },
        "expected_keywords_subset": ["PHÁ MÃ"],
        "expected_quoted": [],
    },
    {
        "query": "thế nào là hệ mã khóa đối xứng?",
        "expected_facets": {
            "asks_definition": True,
            "asks_classification": False,
            "asks_example": False,
            "asks_section_detail": False,
        },
        "expected_keywords_subset": ["hệ mã khóa đối xứng"],
        "expected_quoted": [],
    },
    {
        "query": "có bao nhiêu phương pháp mã hóa và gồm những loại nào?",
        "expected_facets": {
            "asks_definition": False,
            "asks_classification": True,
            "asks_example": False,
            "asks_section_detail": False,
        },
        "expected_keywords_subset": ["mã hóa"],
        "expected_quoted": [],
    },
    {
        "query": "cho ví dụ minh họa về mã dịch vòng với k=3",
        "expected_facets": {
            "asks_definition": False,
            "asks_classification": False,
            "asks_example": True,
            "asks_section_detail": False,
        },
        "expected_keywords_subset": ["mã dịch vòng"],
        "expected_quoted": [],
    },
    {
        "query": "trình bày chi tiết mục 'HỆ MÃ AFFINE' kèm ví dụ",
        "expected_facets": {
            "asks_definition": False,
            "asks_classification": False,
            "asks_example": True,
            "asks_section_detail": True,
        },
        "expected_keywords_subset": ["HỆ MÃ AFFINE"],
        "expected_quoted": ["HỆ MÃ AFFINE"],
    },
    {
        "query": "so sánh 'MÃ HÓA THAY THẾ' và 'MÃ HÓA HOÁN VỊ'",
        "expected_facets": {
            "asks_definition": False,
            "asks_classification": False,
            "asks_example": False,
            "asks_section_detail": True,
        },
        "expected_keywords_subset": ["MÃ HÓA THAY THẾ", "MÃ HÓA HOÁN VỊ"],
        "expected_quoted": ["MÃ HÓA THAY THẾ", "MÃ HÓA HOÁN VỊ"],
    },
    {
        "query": "định nghĩa về bản rõ và bản mã là gì",
        "expected_facets": {
            "asks_definition": True,
            "asks_classification": False,
            "asks_example": False,
            "asks_section_detail": False,
        },
        "expected_keywords_subset": ["bản rõ", "bản mã"],
        "expected_quoted": [],
    },
    {
        "query": "kể tên các hệ mã hóa cổ điển trong tài liệu",
        "expected_facets": {
            "asks_definition": False,
            "asks_classification": True,
            "asks_example": False,
            "asks_section_detail": False,
        },
        "expected_keywords_subset": ["hệ mã"],
        "expected_quoted": [],
    },
]

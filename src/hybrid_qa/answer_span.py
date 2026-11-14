"""
Answer Span Extraction & Self-Study Question Tail — shared, deterministic utilities.

Two hard product requirements are enforced here:

1. PRECISION (no over-capture):
   When the system answers a question it must return *only* the span that answers it,
   never a whole paragraph or an unrelated trailing clause. `extract_answer_span`
   classifies the question intent and cuts the minimal clause that satisfies it.

2. QUESTION TAIL (always end on a question):
   Every explanatory answer must terminate with at least one follow-up question.
   `ensure_trailing_question` is the single chokepoint that guarantees this,
   regardless of which answer path produced the body.

100% offline, zero external calls.
"""

import re
import unicodedata
from typing import Any, List, Optional, Sequence, Tuple
from src.utils.vietnamese_nlp import VietnameseNLP

__all__ = [
    "strip_accents",
    "split_sentences",
    "split_clauses",
    "detect_answer_type",
    "extract_answer_span",
    "select_best_sentence",
    "relevance_score",
    "generate_followup_questions",
    "build_question_block",
    "ends_with_question",
    "strip_question_block",
    "ensure_trailing_question",
    "QUESTION_BLOCK_HEADER",
]

QUESTION_BLOCK_HEADER = "💡 **Câu hỏi tự học tiếp theo:**"


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive matching."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


# Stopwords are stored accent-stripped because tokenisation strips accents before
# comparing; keeping diacritics here would silently disable the whole filter.
_RAW_STOPWORDS = (
    "và", "của", "các", "những", "trong", "để", "với", "cho", "tại", "là",
    "có", "được", "khi", "này", "đó", "theo", "về", "như", "một", "nhiều",
    "thì", "mà", "từ", "ở", "ra", "đã", "sẽ", "đang", "bởi", "do", "bằng",
    "gì", "nào", "sao", "thế", "đâu", "mấy", "ai", "bao", "nhiêu", "vậy",
    # English
    "the", "is", "are", "was", "were", "of", "and", "in", "to", "for", "with",
    "on", "a", "an", "what", "which", "how", "when", "where", "why", "who",
    "does", "do", "did", "that", "this", "it", "be", "as", "by", "from",
)

STOPWORDS = {strip_accents(w) for w in _RAW_STOPWORDS}
_STOPWORD_STEMS = STOPWORDS

CITATION_ARTIFACT_RE = re.compile(r"\[?[A-Z0-9_\-]+::P\d+\]?")

# Markers that mean "the sentence that follows states the answer".
DEFINITION_CUES = (
    "là ", "được hiểu là", "được định nghĩa", "nghĩa là", "được gọi là",
    "refers to", "is defined as", "means ",
)

LIMITATION_CUES = (
    "hạn chế", "nhược điểm", "thách thức", "khó khăn", "chưa", "tồn tại",
    "limitation", "drawback", "challenge",
)

NUMBER_RE = re.compile(
    r"(?<!\w)(?:\d+\s*[-–—]\s*\d+"                    # ranges such as 2024-2030
    r"|\d+(?:[.,]\d+)?\s*(?:%|phần\s*trăm|token[s]?|ms|giây|phút|giờ|"
    r"gb|mb|kb|mw|kw|triệu|tỷ|nghìn|đồng|usd|vnd)?)(?!\w)",
    re.IGNORECASE,
)

# Prepositions that introduce the answer payload of a definition/entity question.
PAYLOAD_CUES = (
    "thuộc lĩnh vực", "thuộc về", "thuộc", "bao gồm", "gồm", "được gọi là",
    "được hiểu là", "được định nghĩa là", "được xác định là", "nghĩa là",
    "hướng tới", "tập trung vào", "liên quan đến", "về việc", "về",
)

# Subset of cues whose payload is a pure value (no subject), safe to cut aggressively.
VALUE_CUES = (
    "thuộc lĩnh vực", "thuộc về", "thuộc", "bao gồm", "gồm",
    "được gọi là", "được hiểu là", "được định nghĩa là", "được xác định là",
)

# Words that typically open a new clause; used to stop an over-long payload.
CLAUSE_BREAKERS = {
    "khẳng", "cho", "được", "với", "trong", "nhằm", "để", "khi", "nếu",
    "tuy", "nhưng", "và", "hoặc", "theo", "bởi", "do", "vì", "nên",
    "which", "that", "with", "while", "where", "when",
}

# Words that introduce a quantified value; the numeric span starts here.
NUMBER_LEFT_CUES = (
    "giai đoạn", "thời gian", "thời điểm", "từ năm", "vào năm", "năm",
    "tổng mức", "tổng", "mức", "đạt", "chiếm", "giảm", "tăng", "khoảng",
    "chỉ", "còn", "với", "trong", "là", "chi phí", "ngân sách", "hiệu suất",
    "tỷ lệ", "tỉ lệ", "số lượng", "độ trễ", "dung lượng", "kích thước",
)

MAX_SPAN_WORDS = 22
DEFAULT_PAYLOAD_WORDS = 7


def _clean(text: str) -> str:
    text = CITATION_ARTIFACT_RE.sub("", text or "")
    return re.sub(r"[ \t]+", " ", text).strip()


def split_sentences(text: str, min_len: int = 6) -> List[str]:
    """
    Splits into sentences on terminal punctuation and line breaks.

    Markdown heading markers are stripped rather than dropping the line: a passage
    frequently begins with "## 1. Kiến trúc ..." followed by the real content, and
    discarding it would lose the most relevant sentence in the document.
    """
    cleaned = _clean(text)
    raw = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    out: List[str] = []
    for piece in raw:
        candidate = piece.strip()
        candidate = re.sub(r"^#{1,6}\s*", "", candidate)
        candidate = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if len(candidate) >= min_len:
            out.append(candidate)
    return out


def split_clauses(sentence: str) -> List[str]:
    """
    Splits a sentence into minimal clauses on punctuation and coordinators.

    Decimal commas ("30,9%") and thousands separators are protected: splitting there
    would truncate a numeric answer in half.
    """
    protected = re.sub(r"(?<=\d),(?=\d)", "\u0000", sentence)
    parts = re.split(r"[,;:()]|\s+(?:và|hoặc|nhưng|song|while|but|and)\s+", protected)
    return [p.replace("\u0000", ",").strip(" .") for p in parts if len(p.replace("\u0000", ",").strip(" .")) >= 3]


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"\w+", strip_accents(text.lower())) if w not in _STOPWORD_STEMS and len(w) > 1}


def _has_cue(text: str, cues: Sequence[str]) -> bool:
    """
    Word-boundary cue matching on the accent-stripped text.

    Substring matching is not safe here: "chưa" (limitation cue) is a prefix of
    "chuẩn hóa", which would falsely turn an unrelated sentence into a limitation.
    """
    normalized = strip_accents(text.lower())
    for cue in cues:
        normalized_cue = strip_accents(cue.lower()).strip()
        if not normalized_cue:
            continue
        if re.search(r"(?<!\w)" + re.escape(normalized_cue) + r"(?!\w)", normalized):
            return True
    return False


def detect_answer_type(query: str) -> str:
    """
    Classifies the expected answer shape so the span cutter can be strict.

    Order matters: a query like "hạn chế của hệ thống là gì" is a *limitation*
    question even though it ends with "là gì".

    Returns one of: number, limitation, definition, entity, generic.
    """
    q = strip_accents(query.lower())
    if any(k in q for k in ["bao nhieu", "may ", "ty le", "ti le", "phan tram", "how many",
                            "how much", "bao lau", "thoi gian", "chi phi", "do tre",
                            "so luong", "tong so", "nam nao", "khi nao", "tu nam", "den nam"]):
        return "number"
    if any(k in q for k in ["han che", "nhuoc diem", "thach thuc", "kho khan",
                            "limitation", "drawback", "weakness", "rui ro"]):
        return "limitation"
    if any(k in q for k in ["la gi", "dinh nghia", "khai niem", "nghia la", "what is", "define"]):
        return "definition"
    if any(k in q for k in ["ai ", "cua ai", "ten gi", "goi la gi", "who "]):
        return "entity"
    return "generic"


_TRAILING_DANGLING = re.compile(
    r"\s+(và|hoặc|nhưng|của|với|cho|trong|theo|vào|đến|tại|trên|như|khi|để|sang|về|and|of|with|to|in|on)$",
    re.IGNORECASE,
)

# Accent-stripped function words used to stop a numeric window.
_TRAILING_DANGLING_STEMS = {
    strip_accents(w)
    for w in ("và", "hoặc", "của", "với", "cho", "trong", "theo", "vào", "đến",
              "tại", "trên", "như", "khi", "để", "sang", "về", "so", "vậy",
              "and", "of", "with", "to", "in", "on")
}


def _trim_span(span: str) -> str:
    """Removes leading connectives and trailing dangling words (repeat until stable)."""
    span = span.strip(" -–—:;,.")
    span = re.sub(
        r"^(và|hoặc|nhưng|song|do đó|vì vậy|tuy nhiên|ngoài ra|theo đó|"
        r"bên cạnh đó|and|but|however|therefore|also)\s+",
        "",
        span,
        flags=re.IGNORECASE,
    )
    previous = None
    while previous != span:
        previous = span
        span = _TRAILING_DANGLING.sub("", span).strip(" ,.;:-")
    return span.strip()


def _cap_words(span: str, max_words: int = MAX_SPAN_WORDS) -> str:
    words = span.split()
    if len(words) <= max_words:
        return span
    return " ".join(words[:max_words]).rstrip(",;:") + "…"


def _numeric_span(clause: str, start: int, end: int, query: str = "") -> str:
    """
    Builds the minimal span around a numeric match.

    The window walks backwards from the number while words still look like a
    measurement head ("chi phí", "giai đoạn", "giảm"), and forwards only across
    the unit, so unrelated trailing words are never included.
    """
    words = clause.split()
    offsets = []
    cursor = 0
    for word in words:
        idx = clause.find(word, cursor)
        if idx == -1:
            idx = cursor
        offsets.append((idx, idx + len(word)))
        cursor = idx + len(word)

    first_word = 0
    last_word = len(words) - 1
    for i, (w_start, w_end) in enumerate(offsets):
        if w_end <= start:
            first_word = i + 1
        if w_start < end:
            last_word = i
        else:
            break

    # Extend left while the preceding words still describe the quantity.
    left_bound = first_word
    while left_bound > 0 and (first_word - left_bound) < 3:
        candidate = re.sub(r"\W", "", words[left_bound - 1].lower())
        if candidate in {re.sub(r"\W", "", c.lower().split()[0]) for c in NUMBER_LEFT_CUES}:
            left_bound -= 1
        elif candidate.isdigit():
            left_bound -= 1
        else:
            break

    left = " ".join(words[left_bound:first_word])
    numeric_text = clause[start:end].strip()
    tail_words = words[last_word + 1:]

    # A bare year or year range is a complete answer: never attach a trailing clause.
    is_bare_year = bool(
        re.fullmatch(r"(19|20)\d{2}", numeric_text)
        or re.fullmatch(r"(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}", numeric_text)
    )

    right_parts: List[str] = []
    if not is_bare_year:
        # 1. A range ("30,9% đến 39,2%", "2021 - 2030") is answered as a whole.
        range_connectors = {"đến", "tới", "to", "-", "–", "—"}
        consumed = 0
        while consumed < 3 and consumed < len(tail_words):
            normalized = re.sub(r"\W", "", tail_words[consumed].lower())
            if normalized in {re.sub(r"\W", "", c) for c in range_connectors} or tail_words[consumed] in range_connectors:
                right_parts.append(tail_words[consumed])
                consumed += 1
                continue
            if NUMBER_RE.match(tail_words[consumed]) and right_parts:
                right_parts.append(tail_words[consumed])
                consumed += 1
                break
            break

        # 2. Then at most two qualifying words (the measured object), never function words.
        extra = 0
        while consumed < len(tail_words) and extra < 4:
            word = tail_words[consumed]
            normalized = re.sub(r"\W", "", word.lower())
            if normalized in CLAUSE_BREAKERS or normalized in _TRAILING_DANGLING_STEMS:
                break
            right_parts.append(word)
            consumed += 1
            extra += 1
            if word.endswith((".", ";", ",")):
                break

    span = _trim_span(" ".join(p for p in (left, numeric_text, " ".join(right_parts)) if p))
    return span


def _score_numeric(query: str, clause: str, numeric_text: str) -> float:
    """Rates how likely a numeric occurrence is the one the question asks for."""
    q = strip_accents(query.lower())
    text = numeric_text.lower()
    score = 0.0

    asks_time = any(k in q for k in ["nam nao", "tu nam", "den nam", "thoi gian", "bao lau",
                                     "thoi diem", "ky ke hoach", "giai doan", "when", "how long"])
    asks_percent = any(k in q for k in ["phan tram", "ty le", "ti le", "hieu suat", "percent", "%"])
    asks_money = any(k in q for k in ["chi phi", "ngan sach", "von", "gia tri", "cost", "budget"])

    is_range = bool(re.search(r"\d\s*[-–—]\s*\d", text))
    is_year = bool(re.fullmatch(r"(19|20)\d{2}\s*[-–—]?\s*((19|20)\d{2})?", text.strip()))
    is_percent = "%" in text or "phan tram" in text

    if asks_time and (is_range or is_year):
        score += 6.0
    if asks_percent and is_percent:
        score += 6.0
    if asks_money and (is_percent or re.search(r"\d", text)):
        score += 2.0
    if asks_time and is_percent and not is_range:
        score -= 2.0
    if asks_percent and is_range:
        score -= 2.0

    q_tokens = _tokens(query)
    score += len(q_tokens & _tokens(clause)) * 0.5
    return score


def _payload_after_cue(sentence: str, max_words: int = DEFAULT_PAYLOAD_WORDS) -> str:
    """
    Returns the short answer payload that follows a prepositional cue.

    Example:
        "Nội dung văn kiện nghiên cứu thuộc lĩnh vực Trí tuệ Nhân tạo khẳng định ..."
        -> "Trí tuệ Nhân tạo"

    Only cues that introduce a pure *value* are used here ("thuộc lĩnh vực", "bao gồm").
    Definitional cues ("là") are intentionally excluded because their payload must keep
    the subject ("X là một hệ thống ..." -> "X là một hệ thống ...", not "một hệ thống ...").
    """
    lowered = sentence.lower()
    best_idx = -1
    best_cue_len = 0
    for cue in VALUE_CUES:
        idx = lowered.find(cue)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx = idx
            best_cue_len = len(cue)
    if best_idx == -1:
        return ""

    remainder = sentence[best_idx + best_cue_len:].strip(" :,.-")
    words = remainder.split()
    payload: List[str] = []
    for word in words:
        normalized = re.sub(r"\W", "", word.lower())
        if normalized in CLAUSE_BREAKERS and len(payload) >= 2:
            break
        if word.endswith((".", ";")) and len(payload) >= 2:
            payload.append(word.rstrip(".;"))
            break
        payload.append(word)
        if len(payload) >= max_words:
            break

    span = _trim_span(" ".join(payload))
    return span if len(span.split()) >= 2 else ""


def extract_answer_span(
    query: str,
    sentence: str,
    max_words: int = MAX_SPAN_WORDS,
) -> str:
    """
    Cuts the minimal clause inside `sentence` that actually answers `query`.

    This is the guard against "lấy quá" (over-capture): the returned string is the
    shortest contiguous span that satisfies the detected answer type, never the
    whole sentence when a shorter clause is sufficient.
    """
    sentence = _clean(sentence)
    if not sentence:
        return ""

    answer_type = detect_answer_type(query)
    q_tokens = _tokens(query)
    clauses = split_clauses(sentence) or [sentence]

    # ---- number: keep the numeric phrase and its immediate head noun ----
    if answer_type == "number":
        best_num: Optional[Tuple[float, str]] = None
        for clause in clauses:
            for match in NUMBER_RE.finditer(clause):
                numeric_text = match.group(0).strip()
                span = _numeric_span(clause, match.start(), match.end())
                score = _score_numeric(query, clause, numeric_text)
                if best_num is None or score > best_num[0]:
                    best_num = (score, span)
        if best_num is not None:
            return _cap_words(best_num[1], max_words)

    # ---- definition: keep "<subject> là <predicate head>", trimmed to a full clause ----
    if answer_type == "definition":
        for cue in DEFINITION_CUES:
            idx = sentence.lower().find(cue)
            if idx != -1:
                subject = sentence[: idx + len(cue)].strip()
                predicate = sentence[idx + len(cue):]
                # Keep 1-2 predicate clauses: enough to be a real definition, not a paragraph.
                pred_clauses = split_clauses(predicate)[:2]
                span = _trim_span(subject + " " + ", ".join(pred_clauses))
                if len(span.split()) >= 3:
                    return _cap_words(span, max_words)

    # ---- payload cue: "<X> thuộc/bao gồm/là <ANSWER>" -> keep only <ANSWER> ----
    # Runs for definition, entity, and generic intents: it removes the topic
    # preamble that makes answers over-long.
    if answer_type in ("definition", "entity", "generic"):
        payload = _payload_after_cue(sentence)
        if payload:
            return _cap_words(payload, max_words)

    # ---- limitation: clause that carries the limitation cue ----
    if answer_type == "limitation":
        for clause in clauses:
            if _has_cue(clause, LIMITATION_CUES):
                return _cap_words(_trim_span(clause), max_words)

    # ---- generic / entity: highest-overlap clause, shortened if possible ----
    best_clause = ""
    best_score = -1.0
    for clause in clauses:
        overlap = len(q_tokens & _tokens(clause))
        # Prefer clauses that carry content, but do not reward raw length.
        score = overlap * 2.0 - len(clause.split()) * 0.05
        if score > best_score:
            best_score = score
            best_clause = clause

    span = _trim_span(best_clause) if best_clause else _trim_span(sentence)
    return _cap_words(span, max_words)


def relevance_score(query: str, sentence: str, extra: str = "") -> float:
    """
    Lexical relevance between the question and a candidate sentence, in [0, 1].

    Accent-insensitive so Vietnamese questions match documents with/without diacritics.
    `extra` (typically the section heading) counts too: a section titled "Hạn chế"
    legitimately answers "hệ thống có hạn chế gì" even when the sentence itself only
    repeats the topic. Used as a refusal gate: below threshold the system refuses
    instead of guessing.
    """
    # 1. Alias + N matching: if query has alias+N and passage section/text has any alias+N with the same N, rel = 1.0
    alias_re = re.compile(
        r"\b(?:topic|chủ đề|chu de|bài|bai|mục|muc|chương|chuong|phần|phan|section|chapter|lesson|unit)\s*(\d+)\b",
        re.IGNORECASE,
    )
    q_aliases = set(alias_re.findall(query or ""))
    if q_aliases:
        passage_text = f"{sentence or ''} {extra or ''}"
        p_aliases = set(alias_re.findall(passage_text))
        if q_aliases & p_aliases:
            return 1.0

    # 2. Quoted phrase matching: e.g. 'PHÁ MÃ' or "1. GIỚI THIỆU"
    quoted_matches = re.findall(r"['\"]([^'\"]{2,60})['\"]", query or "")
    for qm in quoted_matches:
        qm_clean = strip_accents(qm.strip().lower())
        haystack_raw = strip_accents(f"{sentence or ''} {extra or ''}".lower())
        if qm_clean in haystack_raw:
            return 1.0

    # 3. Direct section title matching: query refers directly to section title
    if extra:
        extra_clean = strip_accents(extra.strip().lower())
        extra_core = re.sub(r"^(?:\d+(?:\.\d+)*\s*[:.\-–]?\s*)", "", extra_clean).strip()
        query_clean = strip_accents((query or "").lower())
        if extra_core and len(extra_core) >= 4 and (extra_core in query_clean or query_clean in extra_core):
            return 0.95

    # 4. Remove meta-question wrapper words before computing lexical coverage
    META_QUESTION_TOKENS = {
        "yeu", "cau", "chi", "tiet", "cu", "the", "lien", "quan", "den", "noi", "dung",
        "cach", "trien", "khai", "duoc", "mo", "ta", "ra", "sao", "gom", "nhung", "gi",
        "chinh", "trong", "muc", "khai", "niem", "nguyen", "ly", "dac", "diem", "nhu",
        "the", "nao", "cho", "biet", "giai", "thich", "trinh", "bay", "vi", "du", "bao",
        "nhieu", "loai", "cac", "nhung", "mot", "he", "thong"
    }
    raw_q_tokens = _tokens(query)
    filtered_q_tokens = {t for t in raw_q_tokens if t not in META_QUESTION_TOKENS}
    q_tokens = filtered_q_tokens if filtered_q_tokens else raw_q_tokens

    if not q_tokens:
        return 0.0
    haystack = _tokens(sentence) | _tokens(extra or "")
    if not haystack:
        return 0.0
    return min(1.0, len(q_tokens & haystack) / float(len(q_tokens)))


def select_best_sentence(query: str, sentences: Sequence[str], section_titles: Optional[Sequence[str]] = None) -> str:
    """
    Picks the sentence with the strongest alignment to the query.

    `section_titles`, when supplied, must be parallel to `sentences`. A section whose
    heading matches the question intent gets a boost.
    """
    META_QUESTION_TOKENS = {
        "yeu", "cau", "chi", "tiet", "cu", "the", "lien", "quan", "den", "noi", "dung",
        "cach", "trien", "khai", "duoc", "mo", "ta", "ra", "sao", "gom", "nhung", "gi",
        "chinh", "trong", "muc", "khai", "niem", "nguyen", "ly", "dac", "diem", "nhu",
        "the", "nao", "cho", "biet", "giai", "thich", "trinh", "bay", "vi", "du", "bao",
        "nhieu", "loai"
    }
    raw_q_tokens = _tokens(query)
    filtered_q_tokens = {t for t in raw_q_tokens if t not in META_QUESTION_TOKENS}
    q_tokens = filtered_q_tokens if filtered_q_tokens else raw_q_tokens
    q_type = detect_answer_type(query)

    q_low = strip_accents(query.lower())
    is_general_concept = any(w in q_low for w in ["la gi", "gioi thieu", "khai niem", "bao nhieu loai", "cac loai", "phuong phap", "phan loai"])
    quoted_matches = [strip_accents(qm.strip().lower()) for qm in re.findall(r"['\"]([^'\"]{2,60})['\"]", query or "")]

    best, best_score = "", -1.0
    for index, sentence in enumerate(sentences):
        sentence_clean = _clean(sentence)
        if len(sentence_clean) < 6:
            continue
        s_norm = strip_accents(sentence_clean.lower())
        score = len(q_tokens & _tokens(sentence_clean)) * 2.0

        # Quoted entity match bonus
        for qm in quoted_matches:
            if qm in s_norm:
                score += 8.0

        if q_type == "number" and NUMBER_RE.search(sentence_clean):
            score += 4.0
        if q_type == "limitation" and _has_cue(sentence_clean, LIMITATION_CUES):
            score += 4.5
        if q_type == "definition" and _has_cue(sentence_clean, DEFINITION_CUES):
            score += 4.0

        # Preference for overview/definition/classification when asking general questions
        if is_general_concept:
            if any(term in s_norm for term in ["gioi thieu", "khai niem", "dinh nghia", "phuong phap pho bien", "bao gom", "co 2 phuong phap", "co 2 loai", "khoa doi xung"]):
                score += 5.0
            if sentence_clean.strip().startswith(("*", "-", "•", "+")):
                score += 3.0
            # Deprioritize isolated formulas when asking for concept
            if any(term in s_norm for term in ["zi = fi", "= z26", "= (pi +", "mod 26", "ta co:"]):
                score -= 5.0

        if section_titles is not None and index < len(section_titles):
            title = section_titles[index] or ""
            t_norm = strip_accents(title.lower())
            for qm in quoted_matches:
                if qm in t_norm:
                    score += 6.0
            if q_type == "limitation" and _has_cue(title, LIMITATION_CUES):
                score += 3.5
            if q_type == "definition" and _has_cue(title, DEFINITION_CUES):
                score += 1.5
            if is_general_concept and any(term in t_norm for term in ["1. ", "gioi thieu", "khai niem"]):
                score += 4.0
            score += len(q_tokens & _tokens(title)) * 0.5

        if re.match(r"^\d+(?:\.\d+)*\s+[A-ZÀ-Ỹ]", sentence_clean) and not is_general_concept:
            score -= 1.5
        if score > best_score:
            best_score, best = score, sentence_clean
    return best


def _meaningful_numbers(context: str, limit: int = 3) -> List[str]:
    """
    Returns numeric expressions worth asking about.

    Bare 1-2 digit integers are discarded: they are almost always list indices or
    heading numbers ("2. Phương pháp"), and quoting them yields nonsensical questions.
    """
    found: List[str] = []
    for match in NUMBER_RE.finditer(_clean(context)):
        text = match.group(0).strip()
        digits = re.sub(r"\D", "", text)
        if len(digits) <= 2 and not re.search(r"[%]|phần trăm", text, re.IGNORECASE):
            continue
        if text not in found:
            found.append(text)
        if len(found) >= limit:
            break
    return found


def _salient_phrase(context: str) -> str:
    """
    Extracts the most informative multi-word phrase from the context.

    Single-token candidates are rejected when a two-word phrase exists, because a
    lone token is frequently a fragment of a compound ("thống" from "hệ thống") that
    would make the generated question meaningless.
    """
    words = re.findall(r"[A-Za-zÀ-Ỹà-ỹ][\w\-]{1,}", _clean(context))

    def is_content(word: str) -> bool:
        return len(word) > 2 and word.lower() not in STOPWORDS

    # Prefer proper-noun-like phrases: capitalised tokens, or token + capitalised token.
    bigrams: dict = {}
    for i in range(len(words) - 1):
        first, second = words[i], words[i + 1]
        if not (is_content(first) and is_content(second)):
            continue
        if not (first[0].isupper() or second[0].isupper()):
            continue
        phrase = f"{first} {second}"
        bigrams[phrase] = bigrams.get(phrase, 0) + 1

    if bigrams:
        return sorted(bigrams.items(), key=lambda kv: (-kv[1], -len(kv[0])))[0][0]

    # Fall back to the longest content-bearing single token.
    singles = [w for w in words if is_content(w)]
    if not singles:
        return ""
    return max(singles, key=len)


def _clean_topic(text: str, max_words: int = 12) -> str:
    """Normalises a section heading or query into a short, quotable topic label."""
    cleaned = (text or "").strip()
    # Strip nested template prefixes/suffixes to avoid recursion
    cleaned = re.sub(r"^(?:nội dung trọng tâm|điểm nổi bật|nội dung chính|tổng quan|tóm tắt)\s*(?:liên quan đến|về)?\s*['\"]?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"['\"]?\s*(?:gồm những gì|như thế nào|ra sao|là gì)\??$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:cho tôi biết|hãy cho biết|cho biết|tìm hiểu về|tóm tắt|về|câu hỏi về)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(#{1,4}\s*)?(\d+(?:\.\d+)*\.?\s*)?", "", cleaned.strip())
    topic = _clean(cleaned)
    topic = topic.strip("'\"` ").strip()

    # Drop unbalanced opening or closing parentheses
    if topic.count("(") > topic.count(")"):
        topic = re.sub(r"\([^\)]*$", "", topic).strip()
    elif topic.count(")") > topic.count("("):
        topic = topic.replace(")", "").strip()

    words = topic.split()
    if words:
        topic = " ".join(words[:max_words])
    else:
        topic = "nội dung này"
    return topic.rstrip(" (,-–—:;")


def _norm_topic_key(text: str) -> str:
    """Normalizes a topic/heading label for diversity deduplication."""
    cleaned = strip_accents(re.sub(r"[^\w\s]", " ", (text or "").lower()))
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_topic_num_key(text: str) -> Optional[str]:
    """Extracts canonical 'topic N' / 'muc N' / 'chuong N' key if present."""
    norm = _norm_topic_key(text)
    m = re.search(r"\b(topic|chu de|muc|bai|section|chuong|phan)\s*(\d+)\b", norm)
    return f"{m.group(1)} {m.group(2)}" if m else None


def _clean_heading_title(raw_title: str, max_words: int = 10) -> str:
    """Cleans a topic/section title without mid-word truncation or body-text bleed."""
    cleaned = _clean(raw_title or "")
    # Stop before body-text clause separators or page markers
    cleaned = re.split(r"===\s*Page|\s+[–—-]\s+(?=[a-zà-ỹA-ZÀ-Ỹ])|[.!?;]\s+", cleaned, maxsplit=1)[0].strip()
    if cleaned.count("(") > cleaned.count(")"):
        cleaned = re.sub(r"\([^\)]*$", "", cleaned).strip()
    elif cleaned.count(")") > cleaned.count("("):
        cleaned = cleaned.replace(")", "").strip()
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
    return cleaned.strip(" .,:;!?-–—(\"'`…")


def _is_covered_topic(label: str, used_keys: set) -> bool:
    """True when `label` targets a topic/heading already in `used_keys`."""
    num_key = _extract_topic_num_key(label)
    if num_key and num_key in used_keys:
        return True
    norm = _norm_topic_key(label)
    if not norm:
        return False
    if norm in used_keys:
        return True
    label_tokens = {w for w in norm.split() if len(w) > 2 and w not in STOPWORDS}
    if len(label_tokens) >= 2:
        for existing in used_keys:
            ex_tokens = {w for w in existing.split() if len(w) > 2 and w not in STOPWORDS}
            if len(ex_tokens) >= 2:
                overlap = len(label_tokens & ex_tokens) / min(len(label_tokens), len(ex_tokens))
                if overlap >= 0.8:
                    return True
    return False


def _mark_used_topic(label: str, used_keys: set) -> None:
    num_key = _extract_topic_num_key(label)
    if num_key:
        used_keys.add(num_key)
    norm = _norm_topic_key(label)
    if norm:
        used_keys.add(norm)


def _clean_snippet_with_balanced_parens(text: str, max_words: int = 8) -> str:
    """Caps `text` at `max_words` while ensuring parentheses are never left unclosed."""
    words = _clean(text).split()
    if not words:
        return ""
    snippet = " ".join(words[:max_words]).rstrip(".,:;!?…")
    if snippet.count("(") > snippet.count(")"):
        # Try extending up to 3 words if that closes the parenthetical cleanly
        extended = False
        for extra in range(1, 4):
            if max_words + extra > len(words):
                break
            cand = " ".join(words[: max_words + extra]).rstrip(".,:;!?…")
            if cand.count("(") == cand.count(")"):
                snippet = cand
                extended = True
                break
        if not extended:
            snippet = re.sub(r"\s*\([^\)]*$", "", snippet).strip()
    elif snippet.count(")") > snippet.count("("):
        snippet = snippet.replace(")", "").strip()
    return snippet.rstrip(".,:;!?…-–—(")


def _noun_context_for_number(
    num: str,
    ctx_sents: Sequence[str],
    used_sent_indices: set,
) -> Tuple[str, int]:
    """Finds an unused sentence containing `num` and returns (noun_context, sent_idx)."""
    for idx, sent in enumerate(ctx_sents):
        if idx in used_sent_indices or num not in sent:
            continue
        cleaned = _clean(sent)
        content_words = [
            w for w in re.findall(r"[A-Za-zÀ-Ỹà-ỹ][\w\-]{1,}", cleaned)
            if strip_accents(w.lower()) not in STOPWORDS
        ]
        if len(content_words) < 2:
            continue
        snippet = _clean_snippet_with_balanced_parens(cleaned, 8)
        return snippet, idx
    return "", -1


def generate_followup_questions(
    query: str,
    context: str = "",
    section_title: str = "",
    count: int = 3,
) -> List[str]:
    """
    Produces fluent, natural, grounded follow-up questions tied to the answered topic and document.
    Prioritizes topic/entity/sentence diversity and context-anchored retrieval.
    Guarantees exactly `count` questions when context is non-empty.
    """
    if count <= 0:
        return []

    q_low = (query or "").lower()
    ctx_low = (context or "").lower()
    norm_query = _norm_topic_key(query)
    query_num_key = _extract_topic_num_key(query)

    # Prefer clean section_title over query for Topic label
    sec_clean = (section_title or "").strip()
    if sec_clean and strip_accents(sec_clean.lower()) not in {"phan noi dung", "noi dung", "toan van", "tong quan", "section", "doc"}:
        topic = _clean_topic(sec_clean, max_words=12)
    else:
        topic = _clean_topic(query, max_words=12)

    ctx_sents = [s.strip() for s in split_sentences(context) if len(s.strip().split()) >= 4]
    used_topic_keys: set = set()
    used_entities: set = set()
    used_sent_indices: set = set()

    candidates: List[str] = []
    deferred_candidates: List[str] = []

    # 1. Distinct Topics / Sections from Document Context (e.g., "Topic 1", "Chủ đề 2", "Mục 3", "Chương 1")
    topic_matches = re.findall(
        r"\b(Topic|Chủ đề|Mục|Bài|Section|Chương|Phần)\s*(\d+)(?:\s*[:\-\.]\s*([^\n\.!\?;]+))?",
        context,
        re.IGNORECASE,
    )
    for tm in topic_matches:
        pfx, num_str = tm[0], tm[1]
        title_part = _clean_heading_title(tm[2], max_words=10) if tm[2] else ""
        label = f"{pfx} {num_str}"
        num_key = _extract_topic_num_key(label)
        if num_key and num_key in used_topic_keys:
            continue
        _mark_used_topic(label, used_topic_keys)
        if title_part:
            _mark_used_topic(title_part, used_topic_keys)
            q_text = f"Yêu cầu hoặc nội dung cụ thể của {pfx} {num_str} ({title_part}) là gì?"
        else:
            q_text = f"Nội dung chi tiết của {pfx} {num_str} trong tài liệu là gì?"

        # Mark sentence(s) that open with this heading so fallback won't repeat the same topic
        for idx, sent in enumerate(ctx_sents):
            if num_key and _extract_topic_num_key(sent[:40]) == num_key:
                used_sent_indices.add(idx)

        if query_num_key and num_key == query_num_key:
            deferred_candidates.append(q_text)
        else:
            candidates.append(q_text)

    # 2. Distinct section headings or key headers from context
    for line in context.splitlines():
        line_s = line.strip()
        has_marker = bool(re.match(r"^(?:#{1,4}\s+|\d+(?:\.\d+)*\.?\s+)", line_s))
        if not has_marker and (line_s.endswith((".", "!", "?")) or len(line_s.split()) > 10):
            continue
        h_m = re.match(r"^(?:#{1,4}\s*)?(?:\d+(?:\.\d+)*\.?\s*)?([A-ZÀ-Ỹ][^\n:]{4,100})$", line_s)
        if h_m:
            sec_txt = _clean_heading_title(h_m.group(1), max_words=10)
            if (
                len(sec_txt) >= 4
                and not any(g in sec_txt.lower() for g in ["nguồn tham khảo", "tài liệu", "đoạn", "bài tập", "references", "phụ lục"])
                and not _is_covered_topic(sec_txt, used_topic_keys)
            ):
                _mark_used_topic(sec_txt, used_topic_keys)
                for idx, sent in enumerate(ctx_sents):
                    if _norm_topic_key(sec_txt) in _norm_topic_key(sent) and len(sent.split()) <= 12:
                        used_sent_indices.add(idx)
                candidates.append(f"Mục tiêu và phạm vi của '{sec_txt}' được trình bày ra sao?")
        if len(candidates) >= count + 3:
            break

    # 3. Distinct concept definitions ("là", "đóng vai trò là", "được định nghĩa là", "được gọi là")
    for def_m in re.finditer(
        r"([A-ZÀ-Ỹa-zà-ỹ0-9_\- ]{2,40})\s+(?:là|đóng vai trò là|được định nghĩa là|được gọi là)\s+([^.!?\n]{8,80})",
        context,
    ):
        raw_t = def_m.group(1)
        term = VietnameseNLP.clean_extracted_term(raw_t)
        norm_term = _norm_topic_key(term)
        if (
            1 <= len(term.split()) <= 5
            and norm_term
            and norm_term not in used_entities
            and norm_term != norm_query
            and not _is_covered_topic(term, used_topic_keys)
            and not any(w in term.lower() for w in ["điều này", "nó", "chúng", "đây", "tài liệu", "nội dung này", "nhận được", "thu được", "tìm được", "sau đây", "nêu trên", "chỉ", "cũng"])
        ):
            used_entities.add(norm_term)
            candidates.append(f"Vai trò và nguyên lý hoạt động của '{term}' là gì?")
        if len(candidates) >= count + 4:
            break

    # 4. Identity, Roles, Workplace — ONLY when context literally contains them
    if any(k in ctx_low for k in ["trần hữu trang", "công tác tại", "tổ trưởng", "đơn vị công tác"]):
        candidates.append("Đơn vị công tác cụ thể của tác giả là gì?")
        candidates.append("Kết quả phấn đấu và thành tích đạt được ra sao?")
        candidates.append("Các nhiệm vụ và công việc cụ thể được giao gồm những gì?")

    # 5. Ideology, Study, Guidelines — ONLY when context literally contains them
    if any(k in ctx_low for k in ["hồ chí minh", "chỉ thị 05", "chỉ thị 07", "đảng viên"]):
        candidates.append("Phương hướng học tập theo tư tưởng Bác gồm những điểm nào?")
        candidates.append("Kế hoạch thực hiện chuyên đề năm 2026 được đề ra như thế nào?")
        candidates.append("Trách nhiệm của người Đảng viên được tài liệu nêu ra sao?")

    # 6. Innovation, Performance, Improvements — ONLY when context literally contains them
    if any(k in ctx_low for k in ["sáng kiến", "cải tiến quy trình", "năng suất lao động"]):
        candidates.append("Những sáng kiến cải tiến quy trình làm việc gồm những gì?")
        candidates.append("Các giải pháp khắc phục hạn chế, thiếu sót của bản thân là gì?")

    # 7. Dates / Years from context (anchored with sentence noun context when available)
    years = re.findall(r"\b(202[0-9]|2030)\b", context)
    if years:
        yr = years[0]
        yr_ctx, yr_idx = _noun_context_for_number(yr, ctx_sents, used_sent_indices)
        if yr_ctx:
            used_sent_indices.add(yr_idx)
            candidates.append(f"Mục tiêu hoặc kế hoạch năm {yr} liên quan đến '{yr_ctx}' là gì?")
        else:
            candidates.append(f"Mục tiêu hoặc kế hoạch trong năm {yr} là gì?")

    # 8. Percentages or Metrics from context — MUST include noun context from source sentence for BM25
    for num in _meaningful_numbers(context):
        if "%" in num or "phần trăm" in num.lower():
            num_ctx, num_idx = _noun_context_for_number(num, ctx_sents, used_sent_indices)
            if num_ctx:
                used_sent_indices.add(num_idx)
                candidates.append(f"Số liệu {num} liên quan đến '{num_ctx}' được giải thích như thế nào?")

    # 9. Orthogonal aspect questions for single `topic` — ONLY when context has signal for that aspect
    deferred_sent_indices: set = set()
    if topic and topic != "nội dung này" and not _is_covered_topic(topic, used_topic_keys):
        unused_sent_count = sum(1 for i in range(len(ctx_sents)) if i not in used_sent_indices)
        topic_aspects: List[str] = []
        if any(k in ctx_low for k in ["là ", "vai trò", "khái niệm", "chức năng", "định nghĩa", "cơ chế", "nguyên lý", "hoạt động"]):
            topic_aspects.append(f"Khái niệm và vai trò chính của '{topic}' gồm những gì?")
        if any(k in ctx_low for k in ["ví dụ", "minh họa", "công thức", "quy trình", "các bước", "mô hình", "thuật toán"]):
            topic_aspects.append(f"Ví dụ hoặc quy trình áp dụng cụ thể của '{topic}' được mô tả như thế nào?")
        if any(k in ctx_low for k in LIMITATION_CUES):
            topic_aspects.append(f"Những hạn chế hoặc thách thức của '{topic}' được tài liệu nêu ra là gì?")
        elif any(k in ctx_low for k in ["ứng dụng", "lợi ích", "ưu điểm", "hiệu quả", "áp dụng"]):
            topic_aspects.append(f"Ứng dụng và ưu điểm nổi bật của '{topic}' gồm những gì?")
        if topic_aspects:
            _mark_used_topic(topic, used_topic_keys)
            max_aspects = (
                len(topic_aspects)
                if (len(candidates) == 0 and len(ctx_sents) >= 4 and unused_sent_count >= 4)
                else 1
            )
            candidates.extend(topic_aspects[:max_aspects])
            # Defer the lead body sentence right under `topic` so Fallback 1 picks subsequent sentences first
            for idx, sent in enumerate(ctx_sents):
                if idx not in used_sent_indices:
                    if idx == 0 or _is_covered_topic(sent, {_norm_topic_key(topic)}):
                        deferred_sent_indices.add(idx)
                        if idx == 0 and idx + 1 < len(ctx_sents) and _is_covered_topic(sent, {_norm_topic_key(topic)}):
                            deferred_sent_indices.add(idx + 1)
                    break

    # 10. Only suggest "biện pháp/giải pháp" if context literally mentions them
    if any(k in ctx_low for k in ["biện pháp", "giải pháp"]):
        candidates.append("Những biện pháp chính được tài liệu đề xuất là gì?")

    # Deduplicate and filter out near-duplicates of the current query
    clean_q = re.sub(r"[^\w\s]", "", q_low).strip()
    seen: set = set()
    unique: List[str] = []

    def _try_add(question: str) -> bool:
        clean_cand = re.sub(r"[^\w\s]", "", question.lower()).strip()
        if not clean_cand or clean_cand == clean_q or clean_cand in seen:
            return False
        seen.add(clean_cand)
        unique.append(question)
        return True

    for question in candidates:
        _try_add(question)
        if len(unique) >= count:
            break

    # Fallback 1: Prefer extracting from distinct unused sentences (`ctx_sents`) with real keywords
    if len(unique) < count and context.strip():
        ordered_sent_indices = [
            i for i in range(len(ctx_sents))
            if i not in used_sent_indices and i not in deferred_sent_indices
        ] + [
            i for i in range(len(ctx_sents))
            if i not in used_sent_indices and i in deferred_sent_indices
        ]
        for idx in ordered_sent_indices:
            if len(unique) >= count:
                break
            sent = ctx_sents[idx]
            if len(sent.split()) <= 10 and _is_covered_topic(sent, used_topic_keys):
                continue
            raw_snippet = _clean_snippet_with_balanced_parens(sent, 8)
            snippet = VietnameseNLP.clean_sentence_snippet(raw_snippet, 8)
            clean_snip = re.sub(r"[^\w\s]", "", snippet.lower()).strip()
            # Must have at least 2 meaningful words and not end in dangling prepositions
            if clean_snip and len(snippet.split()) >= 2 and clean_snip not in seen:
                cand_q = f"Thông tin chi tiết về '{snippet}' được trình bày như thế nào?"
                if _try_add(cand_q):
                    seen.add(clean_snip)
                    used_sent_indices.add(idx)

        # Fallback 2: Deferred candidates (e.g. topic already in query when no other topics remain)
        for dc in deferred_candidates:
            if len(unique) >= count:
                break
            _try_add(dc)

        # Fallback 3: Orthogonal single-topic fallbacks ONLY if still below `count`
        topic_fallbacks = [
            f"Khái niệm và vai trò của '{topic}' gồm những gì?",
            f"Nguyên lý và đặc điểm của '{topic}' như thế nào?",
            f"Nội dung chi tiết của mục '{topic}' gồm những điểm nào?",
            f"Ý nghĩa và phạm vi áp dụng của '{topic}' được nêu ra sao?",
        ]
        for tf in topic_fallbacks:
            if len(unique) >= count:
                break
            _try_add(tf)

    return unique[:count]


def build_question_block(questions: Sequence[str]) -> str:
    """Renders the mandatory trailing question block."""
    items = [q.strip() for q in questions if q and q.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return f"{QUESTION_BLOCK_HEADER}\n{items[0]}"
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(items, start=1))
    return f"{QUESTION_BLOCK_HEADER}\n{numbered}"


def ends_with_question(text: str) -> bool:
    """True when the last non-empty line is a question ending in '?'."""
    for line in reversed([l.strip() for l in (text or "").splitlines() if l.strip()]):
        return line.rstrip("*)` ").endswith("?")
    return False


def strip_question_block(text: str, header: str = QUESTION_BLOCK_HEADER) -> str:
    """
    Removes any existing question block so callers can re-attach it at the end.

    The block spans from its header line up to the next markdown heading (usually
    "### Nguồn tham khảo") or the end of the text. Content after that heading is
    preserved, which lets the caller put sources before the question tail.
    """
    body = text or ""
    if header not in body:
        return body.rstrip()

    lines = body.splitlines()
    kept: List[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(header):
            skipping = True
            continue
        if skipping:
            if stripped.startswith("### "):
                skipping = False
                kept.append(line)
            continue
        kept.append(line)

    cleaned = "\n".join(kept).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def ensure_trailing_question(
    answer: str,
    questions: Sequence[str],
    header: str = QUESTION_BLOCK_HEADER,
) -> str:
    """
    Guarantees the answer text terminates with a question block.

    Idempotent: if the answer already ends on a question, it is returned unchanged.
    """
    body = (answer or "").rstrip()
    if ends_with_question(body):
        return body

    block = build_question_block(questions)
    if not block:
        return body

    if header in body:
        # A question block exists but is not last (e.g. sources appended after it).
        # Move it to the end instead of duplicating it.
        head, _, _tail = body.partition(header)
        body = head.rstrip()

    return f"{body}\n\n{block}" if body else block


def generate_followup_questions_llm(
    query: str,
    context: str = "",
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    count: int = 3,
) -> List[str]:
    """
    LLM-based Follow-up Question Generation.
    Uses prompt-instructed neural generation to formulate fluent, context-grounded questions.
    Smoothly falls back to Vietnamese NLP rule-based extraction if model is unavailable.
    """
    if model is not None and tokenizer is not None and context.strip():
        try:
            import torch
            clean_ctx = context[:800].replace("\n", " ").strip()
            prompt = (
                f"Dựa vào nội dung câu hỏi và đoạn trích tài liệu sau:\n"
                f"Câu hỏi: {query}\n"
                f"Tài liệu: {clean_ctx}\n\n"
                f"Hãy đóng vai một người học đang nghiên cứu, đặt đúng {count} câu hỏi gợi ý ngắn gọn, "
                f"hoàn chỉnh ngữ pháp tiếng Việt để đào sâu thêm vấn đề. "
                f"Tuyệt đối không dùng các cụm từ cắt dở dang, mỗi câu hỏi trên một dòng:\n1."
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            if hasattr(model, "device"):
                input_ids = input_ids.to(model.device)

            model.eval()
            with torch.no_grad():
                gen_ids = input_ids
                for _ in range(80):
                    out = model(gen_ids)
                    logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if tokenizer.eos_token_id and next_tok.item() == tokenizer.eos_token_id:
                        break
                    gen_ids = torch.cat([gen_ids, next_tok], dim=1)

            decoded = tokenizer.decode(gen_ids[0][input_ids.size(1):], skip_special_tokens=True).strip()
            # Parse lines into questions
            lines = [l.strip() for l in decoded.split("\n") if l.strip()]
            q_list = []
            for line in lines:
                clean_l = re.sub(r"^\d+[\.\:\)\-]\s*", "", line).strip()
                if clean_l and len(clean_l.split()) >= 4 and clean_l.endswith("?"):
                    q_list.append(clean_l)
                elif clean_l and len(clean_l.split()) >= 4 and not clean_l.endswith((":", ".")):
                    q_list.append(f"{clean_l}?")
                if len(q_list) >= count:
                    break

            if len(q_list) >= count:
                return q_list
        except Exception:
            pass

    # Fallback to enhanced rule-based generation with Vietnamese NLP cleaning
    return generate_followup_questions(query=query, context=context, count=count)
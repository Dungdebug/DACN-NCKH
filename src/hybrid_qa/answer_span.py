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
from typing import List, Optional, Sequence, Tuple

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


def split_sentences(text: str, min_len: int = 18) -> List[str]:
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
    q_tokens = _tokens(query)
    if not q_tokens:
        return 0.0
    haystack = _tokens(sentence) | _tokens(extra or "")
    if not haystack:
        return 0.0
    return len(q_tokens & haystack) / float(len(q_tokens))


def select_best_sentence(query: str, sentences: Sequence[str], section_titles: Optional[Sequence[str]] = None) -> str:
    """
    Picks the sentence with the strongest alignment to the query.

    `section_titles`, when supplied, must be parallel to `sentences`. A section whose
    heading matches the question intent (e.g. asking for "hạn chế" inside a section
    titled "Hạn chế và Thách thức") gets a boost, which is what makes limitation and
    definition lookups land in the right part of the document.
    """
    q_tokens = _tokens(query)
    q_type = detect_answer_type(query)
    best, best_score = "", -1.0
    for index, sentence in enumerate(sentences):
        sentence = _clean(sentence)
        if len(sentence) < 18:
            continue
        score = len(q_tokens & _tokens(sentence)) * 2.0
        if q_type == "number" and NUMBER_RE.search(sentence):
            score += 4.0
        if q_type == "limitation" and _has_cue(sentence, LIMITATION_CUES):
            score += 4.5
        if q_type == "definition" and _has_cue(sentence, DEFINITION_CUES):
            score += 4.0
        if section_titles is not None and index < len(section_titles):
            title = section_titles[index] or ""
            if q_type == "limitation" and _has_cue(title, LIMITATION_CUES):
                score += 3.5
            if q_type == "definition" and _has_cue(title, DEFINITION_CUES):
                score += 1.5
            score += len(q_tokens & _tokens(title)) * 0.5
        if re.match(r"^\d+(?:\.\d+)*\s+[A-ZÀ-Ỹ]", sentence):
            score -= 1.5
        if score > best_score:
            best_score, best = score, sentence
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


def _clean_topic(text: str, max_words: int = 9) -> str:
    """Normalises a section heading or query into a short, quotable topic label."""
    topic = re.sub(r"^(#{1,4}\s*)?(\d+(?:\.\d+)*\.?\s*)?", "", (text or "").strip())
    topic = _clean(topic)
    words = topic.split()
    return " ".join(words[:max_words]) if words else "nội dung này"


def generate_followup_questions(
    query: str,
    context: str = "",
    section_title: str = "",
    count: int = 3,
) -> List[str]:
    """
    Produces fluent, natural, grounded follow-up questions tied to the answered topic and document.
    Anchored on real document content and entities so they stay directly answerable.
    """
    q_low = (query or "").lower()
    ctx_low = (context or "").lower()
    raw_topic = section_title or query
    topic = _clean_topic(raw_topic)
    if strip_accents(topic.lower()) in {"phan noi dung", "noi dung", "toan van", "tong quan"}:
        topic = _clean_topic(query)

    candidates: List[str] = []

    # 1. Identity, Roles, Workplace
    if any(k in q_low or k in ctx_low for k in ["chức vụ", "tổ trưởng", "công tác", "nghề", "làm việc", "tên là", "trần hữu trang", "đơn vị"]):
        candidates.append("Đơn vị công tác cụ thể của tác giả là gì?")
        candidates.append("Kết quả phấn đấu và thành tích đạt được ra sao?")
        candidates.append("Các nhiệm vụ trọng tâm được giao gồm những gì?")

    # 2. Ideology, Study, Guidelines
    if any(k in q_low or k in ctx_low for k in ["học tập", "tư tưởng", "đạo đức", "chỉ thị", "chính trị", "hồ chí minh", "phong cách"]):
        candidates.append("Nội dung học tập theo tư tưởng Bác gồm những điểm chính nào?")
        candidates.append("Kế hoạch thực hiện chuyên đề năm 2026 được đề ra như thế nào?")
        candidates.append("Trách nhiệm của người Đảng viên được tài liệu nêu ra sao?")

    # 3. Innovation, Performance, Improvements
    if any(k in q_low or k in ctx_low for k in ["kết quả", "sáng kiến", "cải tiến", "quy trình", "công nghệ", "năng suất", "phấn đấu"]):
        candidates.append("Những sáng kiến cải tiến quy trình làm việc gồm những gì?")
        candidates.append("Các biện pháp khắc phục hạn chế, thiếu sót của bản thân là gì?")

    # 4. Dates / Years
    years = re.findall(r"\b(202[0-9]|2030)\b", context)
    if years:
        candidates.append(f"Mục tiêu hoặc kế hoạch trong năm {years[0]} là gì?")

    # 5. Percentages or Metrics
    for num in _meaningful_numbers(context):
        if "%" in num or "phần trăm" in num.lower():
            candidates.append(f"Chỉ số {num} phản ánh kết quả hoặc tiêu chí nào?")

    candidates.append(f"Nội dung trọng tâm liên quan đến '{topic}' gồm những gì?")
    candidates.append("Những biện pháp chính được tài liệu đề xuất là gì?")

    # Deduplicate and filter out near-duplicates of the current query
    clean_q = re.sub(r"[^\w\s]", "", q_low).strip()
    seen: set = set()
    unique: List[str] = []
    for question in candidates:
        clean_cand = re.sub(r"[^\w\s]", "", question.lower()).strip()
        if clean_cand == clean_q or clean_cand in seen:
            continue
        seen.add(clean_cand)
        unique.append(question)
        if len(unique) >= max(1, count):
            break
    return unique


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
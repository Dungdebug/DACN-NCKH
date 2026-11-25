"""
WaveCMS-R² Nonconformity Signal Extraction & Mondrian Group Assignment (Ticket T5a).

100% Strict Offline, Pure Python stdlib & NumPy, Zero External APIs.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

SIGNAL_NAMES: Tuple[str, ...] = (
    "margin",
    "entropy",
    "lolo_drop",
    "salient_cov",
    "negation_mismatch",
    "answer_type_ok",
    "pmi",
    "bm25_max",
)

R3_SIGNAL_NAMES: Tuple[str, ...] = SIGNAL_NAMES + (
    "cross_retriever_agree",
    "sent_margin",
    "idf_query_cov",
    "analyzer_conf",
)

MONDRIAN_GROUPS: Tuple[str, ...] = (
    "VI-encyclopedia",
    "EN-encyclopedia",
    "EN-reading",
    "EN-longpaper",
)

_VI_DIACRITICS_RE = re.compile(
    r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]+(?:[.'/-][0-9A-Za-zÀ-ỹ]+)*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])\d+(?:[.,]\d+)?%?(?![A-Za-z0-9_])")
_YEAR_RE = re.compile(r"\b(1\d{3}|2\d{3})\b")

_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "doing", "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "what",
    "which", "who", "whom", "whose", "this", "that", "these", "those",
    "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "it", "its", "they", "them", "their", "he", "him", "his", "she", "her",
    "we", "our", "you", "your", "i", "me", "my",
    "là", "của", "và", "có", "trong", "cho", "với", "từ", "đến", "về",
    "khi", "nào", "đâu", "ai", "gì", "sao", "như", "thế", "này", "kia",
    "đó", "được", "bị", "đã", "đang", "sẽ", "các", "những", "một", "nhiều",
}

_QUESTION_STARTERS: Set[str] = {
    "what", "when", "where", "who", "whom", "whose", "which", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "shall", "should", "has", "have", "had", "in", "on", "at", "by",
    "for", "from", "to", "with", "during", "the", "a", "an",
    "ai", "khi", "ở", "tại", "bao", "mấy", "vì", "tại", "cái", "nào",
}

_NEGATION_CUES: Set[str] = {
    "not", "n't", "never", "no", "none", "neither", "nor", "cannot", "can't",
    "won't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't", "shouldn't", "wouldn't",
    "couldn't", "without", "fail", "failed", "fails", "unable", "impossible",
    "false", "deny", "denied", "denies", "refuse", "refused", "lack", "lacks",
    "không", "chưa", "chẳng", "không phải", "không thể", "trái ngược",
    "phủ nhận", "thiếu",
}

_ANTONYM_PAIRS: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (("increase", "increases", "increased", "increasing", "tăng"),
     ("decrease", "decreases", "decreased", "decreasing", "giảm")),
    (("rise", "rises", "rose", "risen", "rising"),
     ("fall", "falls", "fell", "fallen", "falling")),
    (("grow", "grows", "grew", "grown", "expand", "expands", "expanded"),
     ("shrink", "shrinks", "shrank", "shrunk", "contract", "contracts", "contracted")),
    (("before", "earlier", "prior", "trước"),
     ("after", "later", "subsequent", "sau")),
    (("first", "initial", "start", "started", "begin", "began"),
     ("last", "final", "end", "ended", "finish", "finished")),
    (("largest", "biggest", "greatest", "lớn nhất"),
     ("smallest", "tiniest", "least", "nhỏ nhất")),
    (("most", "maximum", "max", "highest", "higher", "more", "nhiều nhất", "cao nhất"),
     ("least", "minimum", "min", "lowest", "lower", "less", "fewer", "ít nhất", "thấp nhất")),
    (("major", "primary"), ("minor", "secondary")),
    (("true", "correct", "right", "đúng"), ("false", "incorrect", "wrong", "sai")),
    (("always", "luôn", "luôn luôn"), ("never", "không bao giờ")),
    (("include", "includes", "included", "including"),
     ("exclude", "excludes", "excluded", "excluding")),
    (("allow", "allows", "allowed", "permit", "cho phép"),
     ("forbid", "forbids", "forbade", "prohibit", "prohibits", "ban", "banned", "cấm")),
    (("accept", "accepted", "approve", "approved"),
     ("reject", "rejected", "decline", "declined")),
    (("win", "wins", "won", "success", "successful"),
     ("lose", "loses", "lost", "failure", "failed")),
    (("positive", "same", "similar", "giống"),
     ("negative", "different", "opposite", "khác")),
)

_MONTHS: Set[str] = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "tháng", "ngày", "năm", "thế kỷ", "century", "decade", "bc", "ad", "bce", "ce",
}

_NUMBER_WORDS: Set[str] = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "hundred", "thousand", "million",
    "billion", "trillion", "percent", "percentage", "dollar", "dollars",
    "euro", "euros", "kg", "km", "m", "cm", "mm", "hours", "minutes", "seconds",
    "years", "days", "months", "một", "hai", "ba", "bốn", "năm", "sáu",
    "bảy", "tám", "chín", "mười", "trăm", "nghìn", "triệu", "tỷ", "phần trăm",
}

_PERSON_ROLE_CUES: Set[str] = {
    "president", "king", "queen", "emperor", "prime", "minister", "senator",
    "governor", "mayor", "professor", "doctor", "dr", "author", "writer",
    "scientist", "physicist", "chemist", "mathematician", "engineer", "artist",
    "director", "founder", "ceo", "leader", "general", "captain", "player",
    "tác giả", "giáo sư", "tiến sĩ", "bác sĩ", "nhà khoa học", "kỹ sư",
    "chủ tịch", "thủ tướng", "vua", "hoàng đế", "người sáng lập",
}

_LOCATION_CUES: Set[str] = {
    "city", "town", "village", "country", "nation", "state", "province",
    "county", "region", "continent", "island", "ocean", "sea", "river",
    "lake", "mountain", "valley", "street", "avenue", "road", "building",
    "university", "college", "school", "hospital", "museum", "park", "airport",
    "capital", "north", "south", "east", "west", "thành phố", "tỉnh",
    "quốc gia", "nước", "làng", "xã", "huyện", "quận", "sông", "núi",
    "biển", "đảo", "đại học", "trường", "bệnh viện", "thủ đô",
}

_CAUSAL_CUES: Set[str] = {
    "because", "since", "due", "therefore", "thus", "hence", "order",
    "prevent", "avoid", "cause", "caused", "causes", "enable", "allow",
    "vì", "do", "bởi", "nhờ", "nhằm", "để", "nên", "vì vậy", "bởi vì",
}


def assign_group(
    doc_meta: Optional[Dict[str, Any]] = None,
    query: str = "",
    doc_length_chars: int = 0,
) -> str:
    """Assign a Mondrian calibration group using only deployment-time metadata.

    Strictly ignores any hidden ground-truth fields (e.g., ``answerable``,
    ``gold_answer``, ``is_impossible``, ``answers``, ``label``).
    """
    meta = doc_meta or {}

    lang_raw = str(meta.get("language") or meta.get("lang") or "").strip().lower()
    if lang_raw in {"vi", "vn", "vietnamese", "tiếng việt"}:
        return "VI-encyclopedia"

    if lang_raw not in {"en", "eng", "english"}:
        title = str(meta.get("title") or "")
        source_hint = str(meta.get("source") or "")
        if _VI_DIACRITICS_RE.search(f"{query} {title} {source_hint}"):
            return "VI-encyclopedia"

    # English document sub-grouping by deployment-time source/type/length
    descriptor = " ".join(
        str(meta.get(k) or "")
        for k in ("source", "dataset", "doc_type", "type", "domain", "genre", "category")
    ).lower()

    longpaper_tokens = (
        "qasper", "arxiv", "longpaper", "long_paper", "paper", "research",
        "scientific", "academic", "pdf", "thesis", "journal", "proceedings",
    )
    if any(tok in descriptor for tok in longpaper_tokens) or doc_length_chars >= 12000:
        return "EN-longpaper"

    reading_tokens = (
        "squad", "reading", "comprehension", "passage", "race", "newsqa",
        "narrative", "story", "hotpot",
    )
    if any(tok in descriptor for tok in reading_tokens):
        return "EN-reading"

    encyclopedia_tokens = ("wiki", "wikipedia", "encyclopedia", "handbook", "kb")
    if any(tok in descriptor for tok in encyclopedia_tokens):
        return "EN-encyclopedia"

    if 0 < doc_length_chars < 2500:
        return "EN-reading"

    return "EN-encyclopedia"


def _tokenize_lower(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _content_tokens(text: str) -> Set[str]:
    return {t for t in _tokenize_lower(text) if t not in _STOPWORDS and len(t) > 1}


def _token_idf(tok: str, idf_map: Optional[Mapping[str, float]] = None) -> float:
    if idf_map is not None and tok in idf_map:
        val = idf_map[tok]
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            return max(0.0, float(val))
    return math.log(1.0 + len(tok))


def _idf_weighted_cov(
    query_tokens: Set[str],
    target_tokens: Set[str],
    idf_map: Optional[Mapping[str, float]] = None,
) -> float:
    if not query_tokens:
        return 0.0
    total = 0.0
    matched = 0.0
    for tok in query_tokens:
        w = _token_idf(tok, idf_map)
        total += w
        if tok in target_tokens or (len(tok) >= 5 and any(t.startswith(tok[:5]) for t in target_tokens)):
            matched += w
    if total <= 0.0:
        return 0.0
    return float(np.clip(matched / total, 0.0, 1.0))


def _split_sentences(evidence_texts: Sequence[str]) -> List[str]:
    sents: List[str] = []
    for text in evidence_texts:
        for part in _SENTENCE_SPLIT_RE.split(text or ""):
            s = part.strip()
            if s:
                sents.append(s)
    return sents


def _extract_best_evidence_sentence(
    query: str,
    candidate_answer: str,
    evidence_texts: Sequence[str],
    idf_map: Optional[Mapping[str, float]] = None,
) -> Tuple[str, float, float]:
    """Select the evidence sentence most aligned with query and candidate_answer."""
    sentences = _split_sentences(evidence_texts)
    if not sentences:
        return "", 0.0, 0.0

    q_tokens = _content_tokens(query)
    ans_tokens = {t for t in _tokenize_lower(candidate_answer) if t not in _STOPWORDS}

    best_sent = sentences[0]
    best_score = -1.0
    scores: List[float] = []

    for sent in sentences:
        sent_tokens = {t for t in _tokenize_lower(sent) if t not in _STOPWORDS}
        q_cov = _idf_weighted_cov(q_tokens, sent_tokens, idf_map)
        ans_cov = _idf_weighted_cov(ans_tokens, sent_tokens, idf_map)
        if q_tokens and ans_tokens:
            score = 0.65 * q_cov + 0.35 * ans_cov
        elif q_tokens:
            score = q_cov
        else:
            score = ans_cov
        scores.append(score)
        if score > best_score:
            best_score = score
            best_sent = sent

    sorted_scores = sorted(scores, reverse=True)
    top1 = float(np.clip(sorted_scores[0], 0.0, 1.0))
    top2 = float(np.clip(sorted_scores[1], 0.0, 1.0)) if len(sorted_scores) > 1 else 0.0
    return best_sent, top1, top2


def _extract_evidence_texts(evidence: Sequence[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for item in evidence or ():
        if isinstance(item, dict):
            txt = item.get("text") or item.get("content") or item.get("passage") or ""
            texts.append(str(txt))
        elif isinstance(item, str):
            texts.append(item)
    return texts


def _extract_evidence_scores(evidence: Sequence[Dict[str, Any]]) -> List[float]:
    scores: List[float] = []
    for item in evidence or ():
        if not isinstance(item, dict):
            continue
        for key in ("alpha", "weight", "score", "bm25", "bm25_score"):
            val = item.get(key)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                scores.append(float(val))
                break
    return scores


def _top1_index(evidence: Sequence[Dict[str, Any]]) -> int:
    if not evidence:
        return 0
    best_idx = 0
    best_score = -math.inf
    found_score = False
    for idx, item in enumerate(evidence):
        if not isinstance(item, dict):
            continue
        for key in ("alpha", "weight", "score", "bm25", "bm25_score"):
            val = item.get(key)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                found_score = True
                if float(val) > best_score:
                    best_score = float(val)
                    best_idx = idx
                break
    return best_idx if found_score else 0


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits
    shifted = logits - np.max(logits)
    exp_vals = np.exp(np.clip(shifted, -60.0, 60.0))
    total = float(np.sum(exp_vals))
    if total <= 0.0 or not math.isfinite(total):
        return np.full_like(logits, 1.0 / logits.size, dtype=np.float64)
    return exp_vals / total


def _compute_margin_and_entropy(
    routing: Optional[Dict[str, Any]],
    evidence: Sequence[Dict[str, Any]],
) -> Tuple[float, float]:
    rt = routing or {}

    margin_val: Optional[float] = None
    entropy_val: Optional[float] = None

    if isinstance(rt.get("margin"), (int, float)) and math.isfinite(float(rt["margin"])):
        margin_val = float(rt["margin"])
    if isinstance(rt.get("entropy"), (int, float)) and math.isfinite(float(rt["entropy"])):
        entropy_val = float(np.clip(float(rt["entropy"]), 0.0, 1.0))

    probs_arr: Optional[np.ndarray] = None
    logits_arr: Optional[np.ndarray] = None

    for key in ("alphas", "weights", "probs", "alpha"):
        seq = rt.get(key)
        if isinstance(seq, (list, tuple, np.ndarray)) and len(seq) > 0:
            arr = np.asarray(seq, dtype=np.float64)
            arr = np.where(np.isfinite(arr) & (arr >= 0.0), arr, 0.0)
            s = float(np.sum(arr))
            if s > 0.0:
                probs_arr = arr / s
                break

    for key in ("logits", "scores", "z"):
        seq = rt.get(key)
        if isinstance(seq, (list, tuple, np.ndarray)) and len(seq) > 0:
            arr = np.asarray(seq, dtype=np.float64)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            logits_arr = arr
            if probs_arr is None:
                probs_arr = _softmax(arr)
            break

    if margin_val is None:
        ref = logits_arr if logits_arr is not None else probs_arr
        if ref is None:
            ev_scores = _extract_evidence_scores(evidence)
            if ev_scores:
                ref = np.asarray(ev_scores, dtype=np.float64)
        if ref is None or ref.size == 0:
            margin_val = 0.0
        elif ref.size == 1:
            margin_val = float(abs(ref[0]))
        else:
            sorted_ref = np.sort(ref)[::-1]
            margin_val = float(sorted_ref[0] - sorted_ref[1])

    if entropy_val is None:
        if probs_arr is None:
            ev_weights: List[float] = []
            for item in evidence or ():
                if isinstance(item, dict):
                    w = item.get("alpha", item.get("weight"))
                    if isinstance(w, (int, float)) and math.isfinite(float(w)) and float(w) >= 0.0:
                        ev_weights.append(float(w))
            if ev_weights and sum(ev_weights) > 0.0:
                arr = np.asarray(ev_weights, dtype=np.float64)
                probs_arr = arr / float(np.sum(arr))

        if probs_arr is None or probs_arr.size == 0:
            entropy_val = 1.0
        elif probs_arr.size == 1:
            entropy_val = 0.0
        else:
            nz = probs_arr[probs_arr > 1e-12]
            h = float(-np.sum(nz * np.log(nz))) if nz.size > 0 else 0.0
            h_norm = h / math.log(probs_arr.size)
            entropy_val = float(np.clip(h_norm, 0.0, 1.0))

    return float(margin_val), float(entropy_val)


def _compute_lolo_drop(
    query: str,
    candidate_answer: str,
    evidence: Sequence[Dict[str, Any]],
    routing: Optional[Dict[str, Any]],
) -> float:
    rt = routing or {}
    for key in ("lolo_drop", "lolo_score", "causal_drop"):
        val = rt.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            return float(val)

    for key in ("lolo_scores", "causal_scores"):
        seq = rt.get(key)
        if isinstance(seq, dict) and seq:
            vals = [float(v) for v in seq.values() if isinstance(v, (int, float)) and math.isfinite(float(v))]
            if vals:
                return max(vals)
        elif isinstance(seq, (list, tuple, np.ndarray)) and len(seq) > 0:
            vals = [float(v) for v in seq if isinstance(v, (int, float)) and math.isfinite(float(v))]
            if vals:
                return max(vals)

    texts = _extract_evidence_texts(evidence)
    if not texts:
        return 0.0

    target_tokens = _content_tokens(f"{query} {candidate_answer}")
    if not target_tokens:
        return 0.0

    full_tokens: Set[str] = set()
    for t in texts:
        full_tokens.update(_content_tokens(t))
    full_cov = len(target_tokens & full_tokens) / len(target_tokens)

    drop_idx = _top1_index(evidence)
    loo_tokens: Set[str] = set()
    for idx, t in enumerate(texts):
        if idx != drop_idx:
            loo_tokens.update(_content_tokens(t))
    loo_cov = len(target_tokens & loo_tokens) / len(target_tokens)

    return float(max(0.0, full_cov - loo_cov))


def _compute_salient_cov(
    query: str,
    candidate_answer: str,
    evidence: Sequence[Dict[str, Any]],
    best_sent: Optional[str] = None,
) -> float:
    texts = _extract_evidence_texts(evidence)
    top1_text = texts[_top1_index(evidence)] if texts else ""
    focus_text = best_sent if (best_sent and best_sent.strip()) else top1_text
    support_text = f"{candidate_answer} {focus_text}"
    support_lower = support_text.lower()
    support_tokens = set(_tokenize_lower(support_text))
    support_numbers = {m.group(0).replace(",", "") for m in _NUMBER_RE.finditer(support_text)}

    weighted_items: List[Tuple[str, float, str]] = []
    seen: Set[str] = set()

    # 1. Numbers and years (highest weight to catch SQuAD v2 number swaps)
    for m in _NUMBER_RE.finditer(query or ""):
        num = m.group(0).replace(",", "")
        if num not in seen:
            seen.add(num)
            weighted_items.append((num, 3.0, "number"))

    # 2. Proper nouns / capitalized entities / acronyms (catch entity swaps)
    raw_tokens = list(_WORD_RE.finditer(query or ""))
    for idx, m in enumerate(raw_tokens):
        tok = m.group(0)
        low = tok.lower()
        if low in seen or low in _STOPWORDS:
            continue
        is_acronym = len(tok) >= 2 and tok.isupper() and any(c.isalpha() for c in tok)
        is_capitalized = (
            len(tok) >= 2
            and tok[0].isupper()
            and (idx > 0 or low not in _QUESTION_STARTERS)
        )
        if is_acronym or is_capitalized:
            seen.add(low)
            weighted_items.append((low, 2.5, "entity"))

    # 3. Rare / longest content keywords
    content = sorted(
        (t for t in _content_tokens(query) if t not in seen),
        key=lambda w: (-len(w), w),
    )
    for tok in content[:6]:
        seen.add(tok)
        weight = 1.0 + 0.1 * min(len(tok), 10)
        weighted_items.append((tok, weight, "keyword"))

    if not weighted_items:
        return 1.0 if (candidate_answer or "").strip() else 0.0

    total_weight = sum(w for _, w, _ in weighted_items)
    matched_weight = 0.0
    for item, weight, kind in weighted_items:
        if kind == "number":
            if item in support_numbers:
                matched_weight += weight
        elif kind == "entity":
            if item in support_tokens or re.search(rf"\b{re.escape(item)}\b", support_lower):
                matched_weight += weight
        else:
            if (
                item in support_tokens
                or re.search(rf"\b{re.escape(item)}\b", support_lower)
                or (len(item) >= 5 and any(t.startswith(item[:5]) for t in support_tokens))
            ):
                matched_weight += weight

    return float(np.clip(matched_weight / total_weight, 0.0, 1.0))


def _has_negation(text: str) -> bool:
    # Strip trailing Vietnamese yes/no question particles ("... không?", "... chưa?")
    cleaned = re.sub(
        r"\b(hay\s+không|phải\s+không|được\s+không|không|chưa)\s*[?？]*\s*$",
        "",
        (text or "").strip(),
        flags=re.IGNORECASE,
    )
    lower = f" {cleaned.lower()} "
    if any(cue in lower for cue in ("không phải", "không thể", "trái ngược", "phủ nhận")):
        return True
    tokens = set(re.findall(r"[0-9A-Za-zÀ-ỹ']+", lower))
    if tokens & _NEGATION_CUES:
        return True
    return any(tok.endswith("n't") for tok in tokens)


def _score_negation_against_text(query: str, text: str) -> float:
    q_tokens = set(_tokenize_lower(query))
    q_lower = (query or "").lower()
    ev_tokens = set(_tokenize_lower(text))
    ev_lower = (text or "").lower()

    antonym_hit = False
    for side_a, side_b in _ANTONYM_PAIRS:
        q_has_a = any((" " in w and w in q_lower) or w in q_tokens for w in side_a)
        q_has_b = any((" " in w and w in q_lower) or w in q_tokens for w in side_b)
        ev_has_a = any((" " in w and w in ev_lower) or w in ev_tokens for w in side_a)
        ev_has_b = any((" " in w and w in ev_lower) or w in ev_tokens for w in side_b)

        if (q_has_a and ev_has_b and not ev_has_a) or (q_has_b and ev_has_a and not ev_has_b):
            antonym_hit = True
            break

    polarity_mismatch = _has_negation(query) != _has_negation(text)

    if antonym_hit and polarity_mismatch:
        return 1.0
    if antonym_hit:
        return 0.9
    if polarity_mismatch:
        return 0.8
    return 0.0


def _compute_negation_mismatch(
    query: str,
    evidence: Sequence[Dict[str, Any]],
    best_sent: Optional[str] = None,
) -> float:
    texts = _extract_evidence_texts(evidence)
    if not texts and not best_sent:
        return 0.0

    target = best_sent if (best_sent and best_sent.strip()) else texts[_top1_index(evidence)]
    score = _score_negation_against_text(query, target)

    q_tokens = _content_tokens(query)
    if q_tokens and texts:
        min_overlap = min(2, len(q_tokens))
        for sent in _split_sentences(texts):
            if len(_content_tokens(sent) & q_tokens) >= min_overlap:
                score = max(score, _score_negation_against_text(query, sent))
    return score


def _compute_answer_type_ok(query: str, candidate_answer: str) -> float:
    ans = (candidate_answer or "").strip()
    if not ans:
        return 0.0

    q_low = (query or "").strip().lower()
    ans_low = ans.lower()
    ans_tokens = _tokenize_lower(ans)
    ans_token_set = set(ans_tokens)
    has_digit = bool(re.search(r"\d", ans))
    has_year = bool(_YEAR_RE.search(ans))
    is_bare_number = bool(re.fullmatch(r"[\d.,%$\s+-]+", ans))

    # 1. Temporal questions: when / what year / what date / năm nào / khi nào
    if re.search(r"\b(when|what year|what date|which year)\b", q_low) or any(
        k in q_low for k in ("năm nào", "khi nào", "ngày nào", "thời gian nào")
    ):
        if has_year or has_digit or bool(ans_token_set & _MONTHS):
            return 1.0
        return 0.0

    # 2. Quantity questions: how many / how much / bao nhiêu / mấy
    if re.search(r"\b(how many|how much)\b", q_low) or any(
        k in q_low for k in ("bao nhiêu", "mấy")
    ):
        if has_digit or bool(ans_token_set & _NUMBER_WORDS):
            return 1.0
        return 0.0

    # 3. Person questions: who / whom / whose / ai
    if re.search(r"\b(who|whom|whose)\b", q_low) or re.search(r"\b(ai|người nào)\b", q_low):
        if is_bare_number:
            return 0.0
        has_cap_entity = any(w[:1].isupper() for w in re.findall(r"[A-Za-zÀ-ỹ]+", ans))
        has_role = bool(ans_token_set & _PERSON_ROLE_CUES) or any(
            c in ans_low for c in ("tác giả", "giáo sư", "tiến sĩ", "nhà khoa học", "chủ tịch")
        )
        return 1.0 if (has_cap_entity or has_role) else 0.2

    # 4. Location questions: where / ở đâu / nơi nào
    if re.search(r"\bwhere\b", q_low) or any(
        k in q_low for k in ("ở đâu", "nơi nào", "chỗ nào", "địa điểm nào")
    ):
        if is_bare_number:
            return 0.0
        has_cap_entity = any(w[:1].isupper() for w in re.findall(r"[A-Za-zÀ-ỹ]+", ans))
        has_loc_cue = bool(ans_token_set & _LOCATION_CUES) or any(
            c in ans_low for c in ("thành phố", "quốc gia", "đại học", "bệnh viện", "thủ đô")
        )
        return 1.0 if (has_cap_entity or has_loc_cue) else 0.2

    # 5. Explanation questions: why / tại sao / vì sao
    if re.search(r"\bwhy\b", q_low) or any(k in q_low for k in ("tại sao", "vì sao", "lý do")):
        has_causal = bool(ans_token_set & _CAUSAL_CUES) or any(
            c in ans_low for c in ("vì vậy", "bởi vì", "due to", "in order to")
        )
        if has_causal or len(ans_tokens) >= 4:
            return 1.0
        return 0.2

    return 1.0


def _compute_bm25_max(
    evidence: Sequence[Dict[str, Any]],
    routing: Optional[Dict[str, Any]],
) -> float:
    scores: List[float] = []
    rt = routing or {}

    if isinstance(rt.get("bm25_max"), (int, float)) and math.isfinite(float(rt["bm25_max"])):
        scores.append(float(rt["bm25_max"]))
    if isinstance(rt.get("bm25_scores"), (list, tuple, np.ndarray)):
        for v in rt["bm25_scores"]:
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                scores.append(float(v))

    for item in evidence or ():
        if not isinstance(item, dict):
            continue
        for key in ("bm25", "bm25_score", "score"):
            val = item.get(key)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                scores.append(float(val))
                break

    if not scores:
        return 0.0

    max_score = max(scores)
    x = max(-60.0, min(60.0, max_score / 5.0))
    return float(1.0 / (1.0 + math.exp(-x)))


def _pair_agree(text_a: str, text_b: str) -> float:
    a = {t for t in _tokenize_lower(text_a) if t not in _STOPWORDS}
    b = {t for t in _tokenize_lower(text_b) if t not in _STOPWORDS}
    if not a or not b:
        return 0.0
    inter = len(a & b)
    jaccard = inter / len(a | b)
    overlap = inter / min(len(a), len(b))
    return float(np.clip(0.5 * jaccard + 0.5 * overlap, 0.0, 1.0))


def _compute_sent_margin(
    sentence_scores: Optional[Sequence[float]],
    top1_sent_score: float,
    top2_sent_score: float,
) -> float:
    if sentence_scores is not None:
        valid = sorted(
            (float(s) for s in sentence_scores if isinstance(s, (int, float)) and math.isfinite(float(s))),
            reverse=True,
        )
        if not valid:
            return 0.0
        if len(valid) == 1:
            return float(np.clip(valid[0], 0.0, 1.0))
        return float(np.clip(valid[0] - valid[1], 0.0, 1.0))
    return float(np.clip(top1_sent_score - top2_sent_score, 0.0, 1.0))


def extract_signals(
    query: str,
    candidate_answer: str,
    evidence: List[Dict[str, Any]],
    routing: Optional[Dict[str, Any]] = None,
    pmi: Optional[float] = None,
    bm25_best_sentence: Optional[str] = None,
    idf_map: Optional[Mapping[str, float]] = None,
    analyzer_conf: Optional[float] = None,
    sentence_scores: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Extract the 8 WaveCMS-R² + 4 WaveCMS-R³ nonconformity signals plus ``pmi_missing``."""
    texts = _extract_evidence_texts(evidence)
    best_sent, top1_sent_score, top2_sent_score = _extract_best_evidence_sentence(
        query, candidate_answer, texts, idf_map=idf_map
    )

    margin, entropy = _compute_margin_and_entropy(routing, evidence)
    lolo_drop = _compute_lolo_drop(query, candidate_answer, evidence, routing)
    salient_cov = _compute_salient_cov(query, candidate_answer, evidence, best_sent=best_sent)
    negation_mismatch = _compute_negation_mismatch(query, evidence, best_sent=best_sent)
    answer_type_ok = _compute_answer_type_ok(query, candidate_answer)
    pmi_val = float(pmi) if pmi is not None and math.isfinite(float(pmi)) else 0.0
    pmi_missing = 1.0 if pmi is None else 0.0
    bm25_max = _compute_bm25_max(evidence, routing)

    if bm25_best_sentence is not None:
        agree_ans = _pair_agree(candidate_answer, bm25_best_sentence)
        agree_sent = _pair_agree(best_sent, bm25_best_sentence) if best_sent else 0.0
        cross_retriever_agree = max(agree_ans, agree_sent)
    else:
        ev0_best, _, _ = _extract_best_evidence_sentence(
            query, candidate_answer, [texts[0]] if texts else (), idf_map=idf_map
        )
        cross_retriever_agree = _pair_agree(candidate_answer, ev0_best)

    sent_margin = _compute_sent_margin(sentence_scores, top1_sent_score, top2_sent_score)
    idf_query_cov = _idf_weighted_cov(
        _content_tokens(query),
        _content_tokens(f"{best_sent} {candidate_answer}"),
        idf_map=idf_map,
    )

    if analyzer_conf is not None and math.isfinite(float(analyzer_conf)):
        analyzer_conf_val = float(np.clip(float(analyzer_conf), 0.0, 1.0))
    else:
        analyzer_conf_val = float(
            np.clip(0.45 * salient_cov + 0.30 * answer_type_ok + 0.25 * (1.0 - negation_mismatch), 0.0, 1.0)
        )

    return {
        "margin": margin,
        "entropy": entropy,
        "lolo_drop": lolo_drop,
        "salient_cov": salient_cov,
        "negation_mismatch": negation_mismatch,
        "answer_type_ok": answer_type_ok,
        "pmi": pmi_val,
        "bm25_max": bm25_max,
        "pmi_missing": pmi_missing,
        "cross_retriever_agree": cross_retriever_agree,
        "sent_margin": sent_margin,
        "idf_query_cov": idf_query_cov,
        "analyzer_conf": analyzer_conf_val,
    }


__all__ = [
    "SIGNAL_NAMES",
    "R3_SIGNAL_NAMES",
    "MONDRIAN_GROUPS",
    "assign_group",
    "extract_signals",
]

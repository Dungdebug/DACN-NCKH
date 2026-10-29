"""
Token-Efficient Answering Engine ("Concise Evidence Mode") — Task 10.
Implements exploratory answering modes (Balanced, Concise, Minimal) designed to reduce
output token footprint while strictly preserving citations, key facts, and refusal integrity.

RULES:
- Never omit citations or key facts
- Never fabricate external knowledge
- Never convert a refusal into an answer
- No hidden chain-of-thought
- Explicitly marked as exploratory product feature (does NOT modify Phase 4.x benchmark)
"""

from typing import Tuple, Optional, List, Dict, Any
import re


class AnswerLengthMode:
    BALANCED = "balanced"
    CONCISE = "concise"
    MINIMAL = "minimal"
    DETAILED = "detailed"


STOPWORDS_VI_EN = {
    "là", "của", "và", "các", "có", "trong", "được", "cho", "những", "với",
    "khi", "ở", "nào", "gì", "sao", "thế", "này", "đó", "ra", "vào", "từ",
    "theo", "đã", "sẽ", "đang", "như", "thì", "ai", "nhiêu", "bao", "nơi",
    "mỗi", "lại", "qua", "bởi", "do", "để", "rằng", "về", "hay", "hoặc",
    "hãy", "cho", "tôi", "biết", "hỏi", "trả", "lời", "tài", "liệu", "nêu",
    "the", "is", "are", "a", "an", "of", "in", "to", "for", "with", "on",
    "at", "by", "from", "and", "or", "what", "which", "how", "when", "where", "who",
}

NUMERIC_QUESTION_CUES = (
    "bao nhiêu", "mấy", "tỷ lệ", "phần trăm", "kích thước", "tham số",
    "chiều dài", "tốc độ", "độ trễ", "thời gian", "năm nào", "khi nào",
    "ngân sách", "chi phí", "mức", "số lượng", "how many", "how much", "rate", "size",
)


def _extract_query_keywords(question: str) -> List[str]:
    cleaned = re.sub(r"[^\w\s]", " ", (question or "").lower())
    return [w for w in cleaned.split() if len(w) > 1 and w not in STOPWORDS_VI_EN]


def strip_passage_artifacts(text: str) -> str:
    """
    Removes internal passage IDs, page markers, markdown heading hashes,
    and decorative noise from passage text so they never leak into user answers.
    """
    if not text:
        return ""
    s = text
    # Strip === Page X === markers
    s = re.sub(r"===\s*Page\s*\d+\s*===", " ", s, flags=re.IGNORECASE)
    # Strip leaked citation/passage tags like [DOC001::P002] or _DOC_013::P000]
    s = re.sub(r"\[?[A-Za-z0-9_\-]+::P\d+\]?", " ", s)
    s = re.sub(r"\[DOC_[A-Za-z0-9_\-]+\]", " ", s)
    # Strip TOC dotted lines
    s = re.sub(r"[\.\-_·]{4,}\s*\d*", " ", s)
    # Convert markdown headings into clean sentence breaks if followed by newline
    lines = []
    for line in s.split("\n"):
        l_strip = line.strip()
        if not l_strip:
            continue
        # Skip standalone markdown headers that have no sentence punctuation or content
        if l_strip.startswith("#"):
            header_text = l_strip.lstrip("#").strip()
            # If header is just a short label like "1. Giới thiệu", skip it as a standalone sentence
            if len(header_text.split()) <= 6 and ":" not in header_text:
                continue
            l_strip = header_text
        # Strip markdown bold/italic markers for clean reading
        l_strip = re.sub(r"\*{1,3}([^\*]+)\*{1,3}", r"\1", l_strip)
        lines.append(l_strip)

    joined = " ".join(lines)
    return re.sub(r"\s+", " ", joined).strip()


def split_clean_sentences(text: str) -> List[str]:
    """
    Splits passage text into clean, complete sentences WITHOUT breaking:
    - Decimal numbers (e.g. 134.5, 743.86, 3.03, 1.541)
    - Section numbers (e.g. 1.1, 2.3)
    - Common Vietnamese/English abbreviations (TS., PGS., Tp., Dr., Fig., Eq.)
    """
    cleaned = strip_passage_artifacts(text)
    if not cleaned:
        return []

    # Protect decimals and thousands separators (e.g. 134.5, 1.541)
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DEC_DOT>", cleaned)
    # Protect common abbreviations
    protected = re.sub(
        r"\b(TS|PGS|GS|ThS|KS|CN|BS|Tp|TP|Dr|Mr|Mrs|Ms|Prof|Fig|Eq|Sec|Vol|No|vs|al)\.",
        r"\1<ABBR_DOT>",
        protected,
        flags=re.IGNORECASE,
    )
    # Protect standalone numbering at start of clauses like "1. " or "2. "
    protected = re.sub(r"(?:^|(?<=\s))(\d+)\.(?=\s+[A-ZÀ-Ỹ])", r"\1<NUM_DOT>", protected)

    raw_parts = re.split(r"(?<=[.!?])\s+", protected)
    sentences: List[str] = []
    for part in raw_parts:
        restored = (
            part.replace("<DEC_DOT>", ".")
            .replace("<ABBR_DOT>", ".")
            .replace("<NUM_DOT>", ".")
            .strip()
        )
        # Clean leading bullet markers
        restored = re.sub(r"^[\-\•\*]\s+", "", restored).strip()
        if len(restored) >= 12 and any(c.isalpha() for c in restored):
            sentences.append(restored)

    if not sentences and len(cleaned) >= 5:
        sentences.append(cleaned)

    return sentences


def score_sentence_relevance(
    sentence: str,
    question: str,
    query_keywords: List[str],
    position_idx: int = 0,
) -> float:
    """
    Scores how well a candidate sentence answers the question:
    - Lexical keyword overlap & Jaccard similarity
    - Exact phrase/bigram bonus
    - Numeric/unit boost when question asks for numbers/metrics
    - Penalizes noisy/fragmentary strings
    """
    s_lower = sentence.lower()
    q_lower = (question or "").lower()
    s_words = set(re.sub(r"[^\w\s]", " ", s_lower).split())
    q_set = set(query_keywords)

    if not s_words:
        return -10.0

    matched_kw = q_set & s_words
    overlap_count = len(matched_kw)
    coverage = overlap_count / max(1, len(q_set))
    jaccard = overlap_count / max(1, len(q_set | (s_words - STOPWORDS_VI_EN)))

    # Bigram match bonus
    bigram_bonus = 0.0
    if len(query_keywords) >= 2:
        for i in range(len(query_keywords) - 1):
            bg = f"{query_keywords[i]} {query_keywords[i + 1]}"
            if bg in s_lower:
                bigram_bonus += 2.5

    # Numeric intent alignment
    wants_number = any(cue in q_lower for cue in NUMERIC_QUESTION_CUES)
    has_number = bool(re.search(r"\d+", sentence))
    numeric_bonus = 3.0 if (wants_number and has_number) else (0.5 if has_number else 0.0)

    # Definitional alignment ("là gì", "vai trò", "nhiệm vụ", "hoạt động")
    def_bonus = 0.0
    if any(w in q_lower for w in ("là gì", "vai trò", "nhiệm vụ", "chức năng", "định nghĩa", "what is")):
        if any(w in s_lower for w in ("là ", "đảm nhiệm", "vai trò", "nhiệm vụ", "giúp ", "cho phép", "bao gồm", "được thiết kế")):
            def_bonus = 2.0

    # Slight preference for informative sentence length (10-45 words)
    word_cnt = len(sentence.split())
    len_penalty = 0.0
    if word_cnt < 5:
        len_penalty = 2.0
    elif word_cnt > 65:
        len_penalty = 1.5

    pos_bias = max(0.0, 0.3 - position_idx * 0.04)

    return (
        overlap_count * 3.0
        + coverage * 4.0
        + jaccard * 5.0
        + bigram_bonus
        + numeric_bonus
        + def_bonus
        + pos_bias
        - len_penalty
    )


def compress_evidence_context(
    question: str,
    evidence_passages: List[Dict[str, Any]],
    max_sentences_per_passage: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    """
    Query-Guided Sub-Passage Context Pruner:
    Instead of feeding full 256-token passages containing surrounding filler/boilerplate
    into the LLM prompt, extracts only the top query-relevant sentences per passage while
    preserving exact passage citation IDs.
    """
    if not evidence_passages:
        return "", {
            "raw_context_tokens": 0,
            "compressed_context_tokens": 0,
            "tokens_saved": 0,
            "compression_ratio_pct": 0.0,
        }

    q_keywords = _extract_query_keywords(question)
    raw_blocks = []
    compressed_blocks = []

    for p in evidence_passages:
        pid = p.get("passage_id", "DOC::P000")
        raw_txt = p.get("text", "")
        raw_blocks.append(f"[{pid}] {raw_txt}")

        sents = split_clean_sentences(raw_txt)
        if len(sents) <= max_sentences_per_passage:
            clean_joined = " ".join(sents) if sents else strip_passage_artifacts(raw_txt)
            compressed_blocks.append(f"[{pid}] {clean_joined}")
            continue

        scored = [
            (score_sentence_relevance(s, question, q_keywords, idx), idx, s)
            for idx, s in enumerate(sents)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_picked = sorted(scored[:max_sentences_per_passage], key=lambda x: x[1])
        pruned_text = " ".join(item[2] for item in top_picked)
        compressed_blocks.append(f"[{pid}] {pruned_text}")

    raw_full = "\n\n".join(raw_blocks)
    comp_full = "\n\n".join(compressed_blocks)

    raw_tok = int(len(raw_full.split()) * 1.35)
    comp_tok = int(len(comp_full.split()) * 1.35)
    saved = max(0, raw_tok - comp_tok)
    ratio = round((saved / raw_tok) * 100.0, 2) if raw_tok > 0 else 0.0

    return comp_full, {
        "raw_context_tokens": raw_tok,
        "compressed_context_tokens": comp_tok,
        "tokens_saved": saved,
        "compression_ratio_pct": ratio,
    }


def synthesize_accurate_grounded_answer(
    question: str,
    citations_data: List[Dict[str, Any]],
    route: str = "DIRECT_LOOKUP",
    answer_mode: str = "balanced",
) -> str:
    """
    Synthesizes a clean, complete, and strictly document-grounded answer without
    noise, page markers, broken decimal numbers, or irrelevant opening sentences.
    Uses Complementary Coverage Selection to combine the primary salient sentence
    with supporting sentences when the question has multiple aspects.
    """
    if not citations_data:
        return "Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn."

    q_keywords = _extract_query_keywords(question)
    q_set = set(q_keywords)
    mode = (answer_mode or "balanced").lower()

    # Collect and score all clean sentences across retrieved citations
    all_candidates = []
    for c_idx, c in enumerate(citations_data):
        raw_text = c.get("text", "")
        cit_label = c.get("citation_label", f"[{c_idx + 1}]")
        doc_title = c.get("document_title", c.get("document_id", "Tài liệu"))
        sents = split_clean_sentences(raw_text)
        for s_idx, sent in enumerate(sents):
            rel_score = score_sentence_relevance(sent, question, q_keywords, s_idx)
            # Boost by passage retrieval rank
            passage_rank_bonus = max(0.0, 2.0 - c_idx * 0.6)
            total_score = rel_score + passage_rank_bonus
            s_words = set(re.sub(r"[^\w\s]", " ", sent.lower()).split())
            covered = q_set & s_words
            all_candidates.append({
                "score": total_score,
                "sentence": sent,
                "citation_label": cit_label,
                "document_title": doc_title,
                "passage_idx": c_idx,
                "sent_idx": s_idx,
                "covered_kw": covered,
            })

    if not all_candidates:
        fallback = strip_passage_artifacts(citations_data[0].get("text", ""))[:200]
        cit_1 = citations_data[0].get("citation_label", "[1]")
        return f"{fallback} {cit_1}".strip()

    all_candidates.sort(key=lambda x: x["score"], reverse=True)

    def _ensure_period(s: str) -> str:
        s = s.strip()
        if not s:
            return s
        return s if s[-1] in ".!?" else f"{s}."

    # Route 1: Document Summary
    if route == "DOCUMENT_SUMMARY":
        summary_points = []
        seen_sents = set()
        for c_idx, c in enumerate(citations_data[:3]):
            c_cands = [x for x in all_candidates if x["passage_idx"] == c_idx]
            if c_cands:
                best_s = c_cands[0]["sentence"]
                if best_s not in seen_sents:
                    seen_sents.add(best_s)
                    summary_points.append(f"- {_ensure_period(best_s)} {c.get('citation_label', '')}".strip())
        if not summary_points:
            best = all_candidates[0]
            summary_points.append(f"- {_ensure_period(best['sentence'])} {best['citation_label']}")
        return "Tài liệu này tập trung vào các nội dung trọng tâm sau:\n" + "\n".join(summary_points)

    # Route 2: Cross-Document Comparison
    if route == "CROSS_DOCUMENT_COMPARISON":
        comp_points = []
        seen_docs = set()
        for cand in all_candidates:
            d_title = cand["document_title"]
            if d_title in seen_docs:
                continue
            seen_docs.add(d_title)
            comp_points.append(
                f"- **{d_title}**: {_ensure_period(cand['sentence'])} {cand['citation_label']}"
            )
            if len(comp_points) >= 3:
                break
        return "Khi đối chiếu giữa các tài liệu, có các điểm phân biệt chính:\n" + "\n".join(comp_points)

    # Route 3: Direct / Section / Multi-Passage Lookup with Complementary Coverage
    primary = all_candidates[0]
    selected = [primary]
    covered_so_far = set(primary["covered_kw"])

    max_sents = 1 if mode in ("minimal", "concise") else (3 if mode == "detailed" else 2)

    if max_sents > 1:
        for cand in all_candidates[1:]:
            if len(selected) >= max_sents:
                break
            # Avoid near-duplicate sentences
            if any(cand["sentence"] in s["sentence"] or s["sentence"] in cand["sentence"] for s in selected):
                continue
            new_kw = cand["covered_kw"] - covered_so_far
            # Pick complementary sentence if it covers remaining query keywords or has high relevance score
            if new_kw or (cand["score"] >= primary["score"] * 0.65 and cand["score"] > 2.0):
                selected.append(cand)
                covered_so_far.update(cand["covered_kw"])

    # Sort selected sentences that come from the same passage by their natural reading order
    primary_sent = _ensure_period(selected[0]["sentence"])
    primary_cit = selected[0]["citation_label"]

    if mode == "minimal":
        return f"{primary_sent} {primary_cit}".strip()

    if mode == "concise":
        return f"{primary_sent} {primary_cit}".strip()

    # Balanced or Detailed mode: combine primary and complementary sentences smoothly
    parts = [f"{primary_sent} {primary_cit}"]
    for extra in selected[1:]:
        ext_sent = _ensure_period(extra["sentence"])
        ext_cit = extra["citation_label"]
        if ext_cit == primary_cit:
            parts.append(ext_sent)
        else:
            parts.append(f"{ext_sent} {ext_cit}")

    combined = " ".join(parts)
    if route == "SECTION_LOOKUP":
        return f"Tại mục được tra cứu, tài liệu ghi nhận: {combined}"
    if mode == "detailed":
        return f"Trong tài liệu đã cung cấp, thông tin chi tiết như sau: {combined}"
    return f"Dựa trên tài liệu đã cung cấp, {combined}"


def build_token_efficient_prompt(
    question: str,
    context_text: str = "",
    answer_mode: str = AnswerLengthMode.BALANCED,
    language: str = "en",
) -> Tuple[str, int]:
    """
    Constructs a prompt tailored to the requested answer length mode.
    Returns:
        (prompt_string, max_answer_tokens)
    """
    mode = (answer_mode or AnswerLengthMode.BALANCED).lower()

    if language in ("vi", "vietnamese"):
        if mode == AnswerLengthMode.MINIMAL:
            max_tokens = 24
            if context_text:
                instruction = (
                    "Dựa vào ngữ cảnh sau, trả lời câu hỏi bằng một cụm từ hoặc một câu tối thiểu cực ngắn. "
                    "BẮT BUỘC giữ nguyên trích dẫn [DOC...::P...]. Không giải thích thêm."
                )
            else:
                instruction = (
                    "Dựa vào bộ nhớ, trả lời câu hỏi thật ngắn gọn (tối đa 1 câu ngắn). Không giải thích thêm."
                )
        elif mode == AnswerLengthMode.CONCISE:
            max_tokens = 40
            if context_text:
                instruction = (
                    "Dựa vào các đoạn trích sau, hãy trả lời câu hỏi trực tiếp, súc tích trong 1-2 câu ngắn, đủ sự kiện chính. "
                    "BẮT BUỘC ghi rõ mã trích dẫn [DOC...::P...]. Không dùng từ đệm dư thừa."
                )
            else:
                instruction = (
                    "Dựa vào bộ nhớ, trả lời câu hỏi trực tiếp và súc tích trong 1-2 câu ngắn, đủ ý chính."
                )
        else:  # balanced
            max_tokens = 64
            if context_text:
                instruction = (
                    "Dựa trên các đoạn trích sau, hãy trả lời câu hỏi một cách ngắn gọn, chính xác bằng tiếng Việt. "
                    "Kèm theo trích dẫn nguồn."
                )
            else:
                instruction = (
                    "Dựa trên kiến thức trong bộ nhớ, hãy trả lời câu hỏi sau bằng tiếng Việt:"
                )

        if context_text:
            prompt = (
                f"{instruction}\n\n"
                f"Ngữ cảnh:\n{context_text}\n\n"
                f"Câu hỏi: {question}\n"
                f"Trả lời:"
            )
        else:
            prompt = (
                f"{instruction}\n\n"
                f"Câu hỏi: {question}\n"
                f"Trả lời:"
            )

    else:  # English
        if mode == AnswerLengthMode.MINIMAL:
            max_tokens = 20
            if context_text:
                instruction = (
                    "Based on the context, provide a minimal single-clause answer with source citation [DOC...::P...]. "
                    "Zero conversational filler."
                )
            else:
                instruction = "Answer minimally from memory in a single phrase. No filler."
        elif mode == AnswerLengthMode.CONCISE:
            max_tokens = 36
            if context_text:
                instruction = (
                    "Based on the context, provide a direct, fact-dense answer in 1-2 sentences with citations. "
                    "Avoid preamble and redundant phrases."
                )
            else:
                instruction = "Answer directly and concisely from memory in 1-2 sentences."
        else:  # balanced
            max_tokens = 64
            if context_text:
                instruction = "Based on the following context, answer the question accurately with citations."
            else:
                instruction = "Answer the following question based on your memory."

        if context_text:
            prompt = (
                f"{instruction}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )
        else:
            prompt = (
                f"{instruction}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )

    return prompt, max_tokens


def postprocess_concise_answer(
    raw_answer: str,
    citations: List[str],
    answer_mode: str,
    refused: bool,
) -> str:
    """
    Ensures that concise/minimal answers remain grammatically sound and contain citations.
    """
    if refused or not raw_answer:
        return raw_answer

    ans = raw_answer.strip()

    # Clean repetitive loop endings common in small 135M backbones
    lines = [l.strip() for l in ans.split("\n") if l.strip()]
    if lines:
        ans = lines[0]

    # Ensure citation presence if citations list is non-empty and not in text
    if citations:
        has_cit = any(cit in ans for cit in citations)
        if not has_cit:
            ans = f"{ans} [{citations[0]}]"

    return ans

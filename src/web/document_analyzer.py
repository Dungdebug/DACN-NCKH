"""
Phase 5.9 Extension — Document Intelligence & Auto Question Generator.
Extracts structured executive summaries, key takeaways, and generates
diverse, relevant, document-grounded questions from hierarchical SA-CMS structures.
100% Local Offline, Zero External API, Zero Network Egress.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path


def strip_accents(s: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive matching."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def _is_one_edit_away(a: str, b: str) -> bool:
    """Returns True if two tokens (>= 3 chars) differ by exactly 1 edit (sub/ins/del)."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1 or la < 3 or lb < 3:
        return False
    if la == lb:
        return sum(1 for ca, cb in zip(a, b) if ca != cb) == 1
    if la < lb:
        a, b = b, a
    i = j = diffs = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            diffs += 1
            if diffs > 1:
                return False
            i += 1
        else:
            i += 1
            j += 1
    return True


def _fuzzy_token_overlap(q_tokens: Set[str], s_tokens: Set[str]) -> Tuple[int, int]:
    """Returns (exact_hits, fuzzy_hits) where fuzzy_hits require context corroboration."""
    if not q_tokens or not s_tokens:
        return 0, 0
    exact_hits = len(q_tokens & s_tokens)
    fuzzy_hits = 0
    if exact_hits > 0 or len(q_tokens) == 1:
        for qt in q_tokens - s_tokens:
            if len(qt) >= 3 and any(_is_one_edit_away(qt, st) for st in s_tokens):
                fuzzy_hits += 1
    return exact_hits, fuzzy_hits


def _extract_compound_concept_parts(query: str) -> List[str]:
    """Splits compound concept/comparison queries ('A và B là gì', 'phân biệt A với B') into parts."""
    q = (query or "").strip().rstrip("?!.")
    q = re.sub(
        r"^(?:hãy\s+|cho\s+biết\s+|trình\s+bày\s+|giải\s+thích\s+|nêu\s+)?(?:khái\s+niệm\s+|định\s+nghĩa\s+|so\s+sánh\s+|phân\s+biệt\s+)?(?:giữa\s+)?",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip()
    q = re.sub(
        r"\s+(?:là\s+gì|như\s+thế\s+nào|ra\s+sao|có\s+gì\s+khác\s+nhau|khác\s+nhau\s+như\s+thế\s+nào|là\s+hai\s+(?:chất|khái\s+niệm|loại)\s+khác\s+nhau|khác\s+nhau\s+ở\s+điểm\s+nào|khác\s+gì\s+nhau)$",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip()
    parts = [p.strip() for p in re.split(r"\s+(?:và|với|so\s+với|cùng\s+với)\s+|,\s*", q, flags=re.IGNORECASE) if len(p.strip()) >= 2]
    return parts if len(parts) >= 2 else []


def _build_grounded_reasoning(
    query: str,
    evidence_items: List[Tuple[str, str]],
    is_comparison: bool,
    has_sub_queries: bool,
) -> str:
    """Synthesizes entity-grounded reasoning strictly from extracted frames (zero canned fluff)."""
    if len(evidence_items) < 2:
        return ""
    try:
        from src.brain.grounded_explainer import extract_frame, get_grounded_explainer
        from src.brain.neural_discourse_scorer import NeuralDiscourseScorer

        scorer = NeuralDiscourseScorer.get_instance()
        raw_sents = [s for s, _ in evidence_items]
        scores = scorer.score_clauses(query, raw_sents)
        frames = [
            extract_frame(
                sentence=sent,
                role=sc.role,
                salience=sc.salience,
                citation_label=lbl,
            )
            for (sent, lbl), sc in zip(evidence_items, scores)
        ]
        synth = get_grounded_explainer().build_entity_grounded_synthesis(
            query=query,
            frames=frames,
            is_comparison=is_comparison,
        )
        if synth:
            return synth
    except Exception:
        pass
    return ""


class DocumentAnalyzer:
    """
    Analyzes document structures to synthesize key points and generate
    insightful, grounded questions tailored to the document's content.
    """

    STOPWORDS_VI = {
        "và", "của", "các", "những", "trong", "để", "với", "cho", "tại", "là",
        "có", "được", "khi", "này", "đó", "theo", "về", "như", "một", "nhiều",
        "thì", "mà", "từ", "ở", "ra", "đã", "sẽ", "đang", "bởi", "do", "bằng",
    }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Removes citation markers and excess whitespace."""
        clean = re.sub(r"\[?[A-Z0-9_\-]+::P\d+\]?", "", text)
        clean = re.sub(r"_[A-Z0-9_]+::P\d+\]?", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @classmethod
    def _split_clean_sentences(cls, text: str) -> List[str]:
        """Splits text into meaningful sentences."""
        clean = cls._clean_text(text)
        raw_sents = re.split(r"(?<=[.!?])\s+|\n+", clean)
        sentences = []
        for s in raw_sents:
            s_stripped = s.strip()
            # Ignore numbered headers like "1. ", "1.1 "
            s_stripped = re.sub(r"^\d+(\.\d+)*\.?\s*", "", s_stripped)
            if len(s_stripped) >= 20 and not s_stripped.startswith("#"):
                sentences.append(s_stripped)
        return sentences

    @classmethod
    def extract_key_points_and_summary(
        cls,
        raw_text: str,
        title: str = "Tài liệu",
        max_points: int = 4,
    ) -> Dict[str, Any]:
        """
        Extracts high-level executive summary and 3-4 key takeaways.
        """
        all_sentences = cls._split_clean_sentences(raw_text)
        if not all_sentences:
            return {
                "overview": f"Tài liệu {title} đã được nạp thành công vào hệ thống bộ nhớ tham số SA-CMS.",
                "key_points": [f"Nội dung tài liệu '{title}' bao gồm các phần kiến trúc và dữ kiện kỹ thuật liên quan."],
            }

        key_points = []
        try:
            from src.brain.textrank_salience import select_key_points
            key_points = select_key_points(all_sentences, k=max_points)
        except Exception:
            key_points = []

        if not key_points:
            # Fallback to legacy heuristic scoring + distinct prefix selection
            scored_sentences = []
            for s in all_sentences:
                score = 0.0
                s_lower = s.lower()
                # Informational keywords
                if any(k in s_lower for k in ["mục tiêu", "nghiên cứu", "đề xuất", "hệ thống", "kiến trúc", "mô hình"]):
                    score += 3.0
                if any(k in s_lower for k in ["kết quả", "hiệu năng", "đạt", "đánh giá", "tối ưu"]):
                    score += 2.5
                if any(k in s_lower for k in ["hạn chế", "thách thức", "vấn đề", "khó khăn"]):
                    score += 2.0
                if re.search(r"\d+(\.\d+)?\s*(%|token|ms|gb|mb|triệu|nghìn)", s_lower):
                    score += 2.0
                # Length sanity check (prefer 40 - 200 chars)
                if 40 <= len(s) <= 220:
                    score += 1.0

                scored_sentences.append((score, s))

            scored_sentences.sort(key=lambda x: x[0], reverse=True)

            # Pick top distinct points
            seen = set()
            for _, s in scored_sentences:
                first_few = " ".join(s.split()[:4]).lower()
                if first_few not in seen:
                    seen.add(first_few)
                    key_points.append(s)
                if len(key_points) >= max_points:
                    break

            if not key_points:
                key_points = all_sentences[:max_points]

        lead = key_points[0] if key_points else "Tài liệu kỹ thuật chuyên ngành."
        overview = f"Tài liệu '{title}' tập trung vào các nội dung trọng tâm: {lead}"

        return {
            "overview": overview,
            "key_points": key_points,
        }

    @classmethod
    def build_document_knowledge_blueprint(
        cls,
        raw_text: str,
        title: str = "Tài liệu",
    ) -> str:
        """
        Synthesizes a comprehensive VIP Multi-Tier Document Knowledge Blueprint:
        - L1 TL;DR & L2 Executive Synthesis (via ProgressiveSummarizer)
        - Key structural points & headings
        - Extracted quantitative metrics & benchmarks (% / MB / GB / ms / parameters)
        Uses zero raw '###' or single-asterisk '*italic*' so frontend renders cleanly.
        """
        summary_data = cls.extract_key_points_and_summary(raw_text, title=title, max_points=6)
        overview = summary_data.get("overview", "")
        key_points = summary_data.get("key_points", [])

        l1_tldr = overview
        l2_exec = ""
        try:
            from src.brain.progressive_summarizer import ProgressiveSummarizer
            ps = ProgressiveSummarizer().generate_digest(raw_text, doc_title=title)
            if ps.l1_tldr:
                l1_tldr = ps.l1_tldr
            if ps.l2_executive:
                l2_exec = ps.l2_executive
        except Exception:
            pass

        # Extract section headings if present
        headings = []
        for line in raw_text.splitlines():
            l_str = line.strip()
            if l_str.startswith("#"):
                h_clean = l_str.lstrip("#").strip()
                if h_clean and h_clean not in headings:
                    headings.append(h_clean)

        # Extract quantitative metrics
        metrics_found = []
        for m in re.finditer(
            r"\b(\d+(?:[.,]\d+)?\s*(?:%|(?:MB|GB|KB|ms|s|tokens?|triệu(?:\s+tham\s+số)?|tỷ|tầng|lớp)\b))",
            raw_text,
            re.IGNORECASE,
        ):
            val_str = m.group(1).strip()
            if val_str not in metrics_found:
                metrics_found.append(val_str)

        blocks = [
            f"📖 **BẢN ĐỒ TRI THỨC TÀI LIỆU: {title}**",
            f"🎯 **Tóm lược cốt lõi (TL;DR):** {l1_tldr}",
        ]
        if l2_exec and l2_exec.strip() != l1_tldr.strip():
            blocks.append(f"🧭 **Tổng hợp điều hành:**\n{l2_exec}")

        if headings:
            h_badges = " • ".join(f"`{h}`" for h in headings[:6])
            blocks.append(f"🗂️ **Cấu trúc chương mục:** {h_badges}")

        if key_points:
            pts_str = "\n".join(f"- {pt}" for pt in key_points[:6])
            blocks.append(f"📌 **Các luận điểm & phát hiện trọng tâm:**\n{pts_str}")

        if metrics_found:
            m_badges = ", ".join(f"**{m}**" for m in metrics_found[:8])
            blocks.append(f"📊 **Chỉ số & thông số định lượng nổi bật:** {m_badges}")

        return "\n\n".join(blocks)

    @classmethod
    def generate_suggested_questions(
        cls,
        raw_text: str,
        title: str = "Tài liệu",
        max_questions: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Automatically generates diverse, relevant, document-grounded questions.
        Covers definitions, mechanisms, limitations, quantitative facts, and overviews.
        """
        questions = []
        seen_queries = set()

        def add_q(q_text: str, category: str, sample_ans: str = ""):
            clean_q = q_text.strip()
            if clean_q and clean_q.lower() not in seen_queries:
                seen_queries.add(clean_q.lower())
                questions.append({
                    "question": clean_q,
                    "category": category,
                    "sample_answer_excerpt": sample_ans[:160] if sample_ans else "",
                })

        # Pattern 0: Specific Topics / Modules (e.g. "Topic 1: ...", "Topic 11: ...", "Chủ đề 2:")
        topic_pattern = re.compile(
            r"^\s*(Topic|Chủ đề|Bài|Mục|Section)\s*(\d+)[:\-\.\s]+([^\n\.\?]{3,65})",
            re.IGNORECASE | re.MULTILINE,
        )
        for tm in topic_pattern.finditer(raw_text):
            pfx, num_str, t_name = tm.group(1).title(), tm.group(2), tm.group(3).strip()
            add_q(f"Yêu cầu và nội dung chính của {pfx} {num_str} ({t_name[:45]}) là gì?", "Chủ đề bài học", tm.group(0))
            if len(questions) >= 3:
                break

        # Pattern 1: Section-based Questions (Level 2 SA-CMS structure - highest fidelity)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for line in lines:
            # Matches: "## 1. Quy hoạch Điện VIII", "### 2. Kiến trúc", "1. Đặt vấn đề"
            h_match = re.match(r"^(#{1,4}\s*)?(\d+(\.\d+)*\.?\s*)?([A-ZÀ-Ỹ][^\n:]{4,55})$", line)
            if h_match:
                sec_name = h_match.group(4).strip()
                # Skip generic headers
                if any(g in sec_name.lower() for g in ["tài liệu tham khảo", "mục lục", "references", "phụ lục", "overview"]):
                    continue
                q = f"Thông tin cốt lõi được trình bày tại mục '{sec_name}' là gì?"
                add_q(q, "Tra cứu Chương mục", sec_name)
            if len(questions) >= 3:
                break

        # Pattern 2: Concept & Definitions ("là gì", "được định nghĩa")
        def_pattern = re.compile(
            r"([A-ZÀ-Ỹa-zà-ỹ0-9_\-\s]{3,35})\s+(là\s+gì|là\s+một|được\s+(xác định|hiểu|định nghĩa)\s+là|đóng\s+vai\s+trò\s+là)\s+([^.!?]+)",
            re.IGNORECASE,
        )
        for m in def_pattern.finditer(raw_text):
            subject = m.group(1).strip()
            subject = re.sub(r"^(ở|tại|theo|trong|với|khi)\s+", "", subject, flags=re.IGNORECASE).strip()
            if 2 <= len(subject.split()) <= 6 and not subject.startswith("#"):
                q = f"{subject} là gì và có vai trò như thế nào?"
                add_q(q, "Định nghĩa & Khái niệm", m.group(0))
            if len(questions) >= 4:
                break

        # Pattern 3: Limitations / Problems ("Hạn chế", "Nhược điểm", "Thách thức")
        limit_pattern = re.compile(
            r"(hạn\s+chế|nhược\s+điểm|thách\s+thức|khó\s+khăn|vấn\s+đề)\s+(ở\s+đây|của\s+hệ\s+thống|chính|được\s+xác\s+định|trong)?\s*[:\s]+([^.!?]+)",
            re.IGNORECASE,
        )
        for m in limit_pattern.finditer(raw_text):
            excerpt = m.group(0).strip()
            add_q(f"Những hạn chế hoặc thách thức nào được chỉ rõ trong '{title}'?", "Hạn chế & Thách thức", excerpt)
            break

        # Pattern 4: Mechanism / Architecture / Solutions ("mô hình", "kiến trúc", "phương pháp", "giải pháp")
        mech_pattern = re.compile(
            r"(hệ\s+thống|mô\s+hình|kiến\s+trúc|phương\s+pháp|thuật\s+toán)\s+([A-ZÀ-Ỹa-zà-ỹ0-9_\-\s]{4,30})\s+(hoạt\s+động|kết\s+hợp|áp\s+dụng|đạt\s+được|bao\s+gồm)",
            re.IGNORECASE,
        )
        for m in mech_pattern.finditer(raw_text):
            name = m.group(2).strip()
            if 2 <= len(name.split()) <= 6:
                q = f"Kiến trúc hoặc cơ chế hoạt động của {name} như thế nào?"
                add_q(q, "Kiến trúc & Cơ chế", m.group(0))
            if len(questions) >= 5:
                break

        # Pattern 5: Quantitative Metrics / Numbers ("%", "ms", "MB", "GB", "tokens")
        num_pattern = re.compile(
            r"([^.!?\n]{15,90}\b\d+(\.\d+)?\s*(%|token|ms|gb|mb|mw|triệu|nghìn)\b[^.!?\n]{10,60})",
            re.IGNORECASE,
        )
        for m in num_pattern.finditer(raw_text):
            excerpt = m.group(1).strip()
            add_q(f"Tài liệu ghi nhận những kết quả hoặc số liệu định lượng cụ thể nào?", "Số liệu & Định lượng", excerpt)
            break

        # Fallback questions with guaranteed document relevance
        defaults = [
            (f"Tài liệu '{title}' tóm tắt những nội dung chính nào?", "Tổng quan"),
            (f"Các giải pháp kỹ thuật chính được đề xuất trong '{title}' là gì?", "Giải pháp"),
            (f"Mục tiêu trọng tâm của '{title}' được xác định như thế nào?", "Mục tiêu"),
        ]
        for def_q, def_cat in defaults:
            if len(questions) < max_questions:
                add_q(def_q, def_cat)

        return questions[:max_questions]

    @classmethod
    def find_best_grounded_answer(
        cls,
        query: str,
        evidence_passages: List[Any],
        answer_mode: str = "balanced",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Locates the exact span answering the query from the evidence passages.

        Two invariants are enforced:
        1. Precision — the returned core is the minimal clause that answers the
           question (answering "bao nhiêu %" never returns the whole paragraph).
        2. Question tail — every answer ends with a self-study question block.
        """
        from src.hybrid_qa.answer_span import (
            ensure_trailing_question,
            extract_answer_span,
            generate_followup_questions,
            relevance_score,
            select_best_sentence,
            split_sentences,
            strip_question_block,
        )

        MIN_RELEVANCE = 0.34

        from src.self_study.persona.academic_persona import (
            AcademicIntent,
            get_academic_persona_engine,
        )
        persona = get_academic_persona_engine()
        persona_intent = persona.classify_intent(query)
        if persona_intent in (
            AcademicIntent.GREETING,
            AcademicIntent.FAREWELL,
            AcademicIntent.GRATITUDE,
            AcademicIntent.SYSTEM_CAPABILITY,
        ):
            return persona.handle_social_query(query, intent=persona_intent), []

        if not evidence_passages:
            return "Không tìm thấy đủ thông tin bằng chứng trong tài liệu để trả lời câu hỏi này.", []

        # Collect every sentence from every evidence passage.
        sentence_pool: List[str] = []
        sentence_sections: List[str] = []
        for passage in evidence_passages:
            passage_text = getattr(passage, "text", "") if hasattr(passage, "text") else str(passage)
            passage_section = getattr(passage, "section_title", "") or ""
            for sentence in split_sentences(passage_text):
                sentence_pool.append(sentence)
                sentence_sections.append(passage_section)

        from src.web.token_optimizer import expand_query_semantics
        exp_query, _ = expand_query_semantics(query)

        best_sentence = select_best_sentence(exp_query, sentence_pool, sentence_sections)
        best_passage_idx = 0
        if best_sentence:
            for idx, passage in enumerate(evidence_passages):
                passage_text = getattr(passage, "text", "") if hasattr(passage, "text") else str(passage)
                if best_sentence[:40] in cls._clean_text(passage_text):
                    best_passage_idx = idx
                    break
        elif sentence_pool:
            best_sentence = sentence_pool[0]

        # Relevance gate: refuse instead of answering from an unrelated passage.
        matched_section = ""
        if best_sentence in sentence_pool:
            matched_section = sentence_sections[sentence_pool.index(best_sentence)]
        combined_pool_text = " ".join(sentence_pool[:4])
        combined_sec_text = " ".join(sentence_sections[:4])
        rel = max(
            relevance_score(query, best_sentence, extra=matched_section),
            relevance_score(exp_query, best_sentence, extra=matched_section),
            relevance_score(query, combined_pool_text, extra=combined_sec_text) * 0.85,
            relevance_score(exp_query, combined_pool_text, extra=combined_sec_text) * 0.85,
        )
        if rel < MIN_RELEVANCE and sentence_pool:
            q_norm_rel = strip_accents((query or "").lower())
            q_rel_tokens = {
                t for t in re.findall(r"[a-z0-9_]+", q_norm_rel)
                if len(t) >= 2 and t not in {
                    "la", "gi", "cua", "va", "voi", "cac", "nhung", "mot", "cho",
                    "biet", "ve", "co", "duoc", "trong", "nay", "nao", "nhu", "the",
                }
            }
            if q_rel_tokens:
                for idx_p, s_p in enumerate(sentence_pool):
                    sec_p = sentence_sections[idx_p] if idx_p < len(sentence_sections) else ""
                    s_rel_tokens = set(re.findall(r"[a-z0-9_]+", strip_accents(f"{s_p} {sec_p}".lower())))
                    ex_h, fz_h = _fuzzy_token_overlap(q_rel_tokens, s_rel_tokens)
                    fuzzy_rel = min(1.0, (ex_h + 0.85 * fz_h) / float(len(q_rel_tokens)))
                    if fuzzy_rel > rel:
                        rel = fuzzy_rel
                        best_sentence = s_p
                        matched_section = sec_p
        if not best_sentence or rel < MIN_RELEVANCE:
            return "Không tìm thấy đủ thông tin bằng chứng trong tài liệu để trả lời câu hỏi này.", []

        # Deduplicate citations: one citation per distinct document & section.
        dedup_citations = []
        seen_keys = set()
        passage_to_cit_label: List[str] = []
        for idx, p in enumerate(evidence_passages, start=1):
            doc_id = getattr(p, "document_id", "DOC")
            doc_title = getattr(p, "document_title", doc_id) or doc_id
            sec_title = getattr(p, "section_title", "Phần nội dung") or "Phần nội dung"
            para_idx = getattr(p, "paragraph_index", idx)

            key = (doc_id, sec_title, para_idx)
            if key not in seen_keys:
                seen_keys.add(key)
                cit_label = f"[{len(dedup_citations) + 1}]"
                dedup_citations.append({
                    "citation_index": len(dedup_citations) + 1,
                    "citation_label": cit_label,
                    "passage_id": getattr(p, "passage_id", f"PASSAGE_{idx}"),
                    "document_id": doc_id,
                    "document_title": doc_title,
                    "section_title": sec_title,
                    "paragraph_index": para_idx,
                    "score": round(float(getattr(p, "score", 1.0)), 4),
                    "text": getattr(p, "text", "")[:400],
                })
                passage_to_cit_label.append(cit_label)
            else:
                # Find existing label for this key
                matched_lbl = "[1]"
                for dc in dedup_citations:
                    if (dc["document_id"], dc["section_title"], dc["paragraph_index"]) == key:
                        matched_lbl = dc["citation_label"]
                        break
                passage_to_cit_label.append(matched_lbl)

        # Map each sentence in sentence_pool to its passage citation label
        sentence_cit_labels: List[str] = []
        for p_idx, passage in enumerate(evidence_passages):
            passage_text = getattr(passage, "text", "") if hasattr(passage, "text") else str(passage)
            p_lbl = passage_to_cit_label[p_idx] if p_idx < len(passage_to_cit_label) else "[1]"
            for _ in split_sentences(passage_text):
                sentence_cit_labels.append(p_lbl)

        # Check compound query decomposition so multi-part questions cover all sub-queries
        sub_query_sentences: List[Tuple[str, str]] = []
        try:
            from src.reasoning.query_decomposer import QueryDecomposer
            decomp = QueryDecomposer().decompose(query)
            if decomp.is_compound and len(decomp.sub_queries) >= 2:
                for sq in decomp.sub_queries:
                    sq_exp, _ = expand_query_semantics(sq.text)
                    sq_best = select_best_sentence(sq_exp, sentence_pool, sentence_sections)
                    if sq_best and sq_best in sentence_pool:
                        s_idx = sentence_pool.index(sq_best)
                        s_lbl = sentence_cit_labels[s_idx] if s_idx < len(sentence_cit_labels) else "[1]"
                        if all(sq_best != existing_s for existing_s, _ in sub_query_sentences):
                            sub_query_sentences.append((sq_best, s_lbl))
                if sub_query_sentences:
                    best_sentence = sub_query_sentences[0][0]
        except Exception:
            pass

        cit_1 = dedup_citations[0]["citation_label"] if dedup_citations else "[1]"
        if best_sentence in sentence_pool:
            b_idx = sentence_pool.index(best_sentence)
            if b_idx < len(sentence_cit_labels):
                cit_1 = sentence_cit_labels[b_idx]

        q_norm = strip_accents((query or "").lower())
        general_section_markers = (
            "noi dung chi tiet", "thong tin cot loi", "yeu cau cua muc",
            "tong quan", "tom tat", "gioi thieu chung", "noi dung chinh",
            "noi dung cua muc", "trinh bay muc", "bao gom nhung gi",
            "toan bo", "lay het", "day du", "tat ca", "co gi",
            "phan chinh", "y chinh",
        )
        is_general_section_query = any(m in q_norm for m in general_section_markers)
        aspect_concept = any(k in q_norm for k in ("khai niem", "dinh nghia", "la gi", "y nghia", "vai tro", "muc dich"))
        aspect_mechanism = any(k in q_norm for k in (
            "nguyen ly", "co che", "cong thuc", "dac diem", "tinh chat",
            "cach hoat dong", "hoat dong nhu the nao", "hoat dong the nao",
            "quy trinh", "thuat toan", "ky thuat", "trien khai",
        ))
        aspect_example = any(k in q_norm for k in ("vi du", "minh hoa", "bai toan", "truong hop", "ap dung"))
        aspect_eval = any(k in q_norm for k in (
            "uu diem", "nhuoc diem", "han che", "loi ich", "ket qua",
            "so lieu", "hieu nang", "hieu qua", "do chinh xac", "toc do", "bao nhieu", "%",
        ))
        aspect_reasoning = any(k in q_norm for k in (
            "tai sao", "vi sao", "nguyen nhan", "ban chat", "nguyen tac", "bao toan", "ly do",
        ))
        has_specific_aspect = aspect_concept or aspect_mechanism or aspect_example or aspect_eval or aspect_reasoning

        q_intent_tokens = {
            t for t in re.findall(r"[a-z0-9_]+", q_norm)
            if len(t) >= 2 and t not in {
                "la", "gi", "cua", "muc", "trong", "nay", "nao", "nhu", "the", "cho",
                "biet", "ve", "co", "duoc", "cac", "nhung", "mot", "va", "voi",
            }
        }
        exp_norm = strip_accents((exp_query or "").lower())
        exp_tokens = {
            t for t in re.findall(r"[a-z0-9_]+", exp_norm)
            if len(t) >= 2 and t not in q_intent_tokens
        }

        def _is_heading_sent(sent: str, sec: str = "") -> bool:
            st = (sent or "").strip()
            if not st:
                return True
            if re.match(
                r"^\s*(?:#{1,6}\s+|(?:\b(?:topic|chủ đề|chu de|bài|bai|mục|muc|chương|chuong|phần|phan|section|chapter|lesson|unit)\s*\d+[:.\-–]?)|(?:\d+(?:\.\d+)*\.\s+[A-ZÀ-Ỹ]))",
                st,
                re.IGNORECASE,
            ):
                return True
            letters = [ch for ch in st if ch.isalpha()]
            if len(letters) >= 4 and all(ch.isupper() for ch in letters) and "=" not in st:
                return True
            if sec and (
                st.lower() == sec.strip().lower()
                or (len(st) >= 4 and st.rstrip(".:") in sec.strip())
            ):
                return True
            return False

        def _score_sentence_for_tokens(
            sent: str,
            target_tokens: Set[str],
            extra_tokens: Optional[Set[str]] = None,
            subject_phrase_norm: str = "",
        ) -> float:
            s_norm = strip_accents(sent.lower())
            s_tokens = set(re.findall(r"[a-z0-9_]+", s_norm))
            ex_hits, fz_hits = _fuzzy_token_overlap(target_tokens, s_tokens)
            score = ex_hits * 2.5 + fz_hits * 1.8
            if subject_phrase_norm:
                subj_split = re.split(r":|\s+la\s+", s_norm, maxsplit=1)
                subj_part = subj_split[0] if len(subj_split) > 1 else s_norm[:len(subject_phrase_norm) + 16]
                if s_norm.startswith(subject_phrase_norm) or subject_phrase_norm in subj_part:
                    score += 10.0
                else:
                    subj_tokens = set(re.findall(r"[a-z0-9_]+", subj_part))
                    s_ex, s_fz = _fuzzy_token_overlap(target_tokens, subj_tokens)
                    score += s_ex * 3.0 + s_fz * 2.0
            if extra_tokens:
                score += len(extra_tokens & s_tokens) * 0.75
            is_ex_sent = (
                any(c in s_norm for c in ("vi du", "minh hoa", "gia su", "cho p =", "ta co", "chang han", "de bai", "bai tap", "buoc 1", "buoc 2", "loi giai", "huong dan giai"))
                or bool(re.match(r"^\s*(?:vi du|bai|buoc)\s*\d", s_norm))
            )
            if aspect_concept:
                def_cues = (
                    "la chat", "duoc dinh nghia la", "nhuong electron", "nhan electron",
                    "la qua trinh", "la phuong trinh", "la dinh luat", "la he thuc",
                    " la ", "duoc dinh nghia", "khai niem", "phuong phap", "he thong",
                    "dung de", "nham", "bao gom", "la dai luong", "la phan ung",
                )
                if not is_ex_sent and any(c in s_norm for c in def_cues):
                    score += 8.0
                if not aspect_example and (is_ex_sent or any(c in s_norm for c in ("cho p =", "ta co:", "mod 26"))):
                    score -= 8.0
            if aspect_mechanism:
                mech_cues = ("nguyen ly", "co che", "hoat dong", "cong thuc", "dac diem", "tinh chat", "quy trinh", "buoc", "thuat toan", "thong qua", "dua tren", "=", "mod ")
                mech_hits = sum(1 for c in mech_cues if c in s_norm)
                if mech_hits:
                    score += min(9.0, 4.5 + mech_hits * 2.0)
                if "cong thuc" in q_norm and ("cong thuc" in s_norm or "=" in s_norm):
                    score += 4.0
            if aspect_example:
                if is_ex_sent or any(c in s_norm for c in ("ap dung", "bai toan", "truong hop")) or re.search(r"\d", s_norm):
                    score += 6.5
            if aspect_eval:
                if any(c in s_norm for c in ("uu diem", "loi ich", "nhuoc diem", "han che", "kho khan", "rui ro", "ket qua", "dat ", "cai thien", "tang", "giam", "%")) or re.search(r"\d+(?:[.,]\d+)?\s*%", s_norm):
                    score += 6.5
            if aspect_reasoning:
                reason_cues = (
                    "nhuong", "nhan", "bao toan", "ban chat", "nguyen tac",
                    "do ", "vi ", "nen ", "nho ", "dan den", "boi vi", "=",
                )
                reason_hits = sum(1 for c in reason_cues if c in s_norm)
                if reason_hits and not is_ex_sent:
                    score += min(10.0, 4.0 + reason_hits * 2.0)
                if not aspect_example and is_ex_sent:
                    score -= 6.0
            return score

        q_subject_norm = re.sub(
            r"^(?:hay\s+|cho\s+biet\s+|trinh\s+bay\s+|giai\s+thich\s+|neu\s+)?(?:khai\s+niem\s+|dinh\s+nghia\s+)?",
            "",
            q_norm.strip().rstrip("?!."),
        ).strip()
        q_subject_norm = re.sub(
            r"\s+(?:la\s+gi|nhu\s+the\s+nao|ra\s+sao)$",
            "",
            q_subject_norm,
        ).strip()
        if len(q_subject_norm.split()) > 6 or any(sep in f" {q_subject_norm} " for sep in (" va ", " voi ", " so voi ")):
            q_subject_norm = ""

        def _score_body_sentence(sent: str) -> float:
            return _score_sentence_for_tokens(
                sent,
                q_intent_tokens,
                exp_tokens,
                subject_phrase_norm=q_subject_norm,
            )

        # Compound concept decomposition fallback (e.g. "chất thử và chất oxi hóa là gì", "phân biệt A với B")
        if len(sub_query_sentences) < 2:
            compound_parts = _extract_compound_concept_parts(query)
            if len(compound_parts) >= 2 and sentence_pool:
                sub_query_sentences = []
                for part in compound_parts:
                    p_norm = strip_accents(part.lower())
                    p_tokens = {
                        t for t in re.findall(r"[a-z0-9_]+", p_norm)
                        if len(t) >= 2 and t not in {"la", "gi", "cua", "va", "voi", "cac", "nhung", "mot"}
                    }
                    if not p_tokens:
                        continue
                    best_p_sent = ""
                    best_p_score = -1e9
                    best_p_idx = -1
                    for idx_s, s_cand in enumerate(sentence_pool):
                        if _is_heading_sent(s_cand, sentence_sections[idx_s] if idx_s < len(sentence_sections) else ""):
                            continue
                        sc = _score_sentence_for_tokens(s_cand, p_tokens, subject_phrase_norm=p_norm)
                        if sc > best_p_score:
                            best_p_score = sc
                            best_p_sent = s_cand
                            best_p_idx = idx_s
                    if best_p_sent and best_p_score > 0:
                        s_lbl = sentence_cit_labels[best_p_idx] if 0 <= best_p_idx < len(sentence_cit_labels) else "[1]"
                        if all(best_p_sent != existing_s for existing_s, _ in sub_query_sentences):
                            sub_query_sentences.append((best_p_sent, s_lbl))
                if sub_query_sentences:
                    best_sentence = sub_query_sentences[0][0]
                    cit_1 = sub_query_sentences[0][1]

        # Never leave best_sentence as a bare heading when evidence_passages contain real body sentences
        # (unless this is a general section query whose own passage already has body sentences to append),
        # and for concept/reasoning queries prefer the highest-scoring body sentence across all passages.
        is_heading_now = _is_heading_sent(best_sentence, matched_section)
        best_p_obj = evidence_passages[best_passage_idx]
        best_p_txt = getattr(best_p_obj, "text", "") if hasattr(best_p_obj, "text") else str(best_p_obj)
        best_p_has_body = any(
            s.strip() and not _is_heading_sent(s, matched_section)
            for s in split_sentences(best_p_txt)
        )
        should_replace_heading = is_heading_now and not best_p_has_body
        if (
            (should_replace_heading or ((aspect_concept or aspect_reasoning) and not aspect_example and not is_general_section_query))
            and len(sub_query_sentences) < 2
            and sentence_pool
        ):
            body_pool = [
                (idx_s, s_cand)
                for idx_s, s_cand in enumerate(sentence_pool)
                if not _is_heading_sent(s_cand, sentence_sections[idx_s] if idx_s < len(sentence_sections) else "")
            ]
            if body_pool:
                top_idx, top_sent = max(
                    body_pool,
                    key=lambda item: (_score_body_sentence(item[1]), -item[0]),
                )
                if should_replace_heading or _score_body_sentence(top_sent) > _score_body_sentence(best_sentence):
                    best_sentence = top_sent
                    if top_idx < len(sentence_cit_labels):
                        cit_1 = sentence_cit_labels[top_idx]
                    for idx_p, passage in enumerate(evidence_passages):
                        p_txt = getattr(passage, "text", "") if hasattr(passage, "text") else str(passage)
                        if best_sentence[:40] in cls._clean_text(p_txt):
                            best_passage_idx = idx_p
                            matched_section = getattr(passage, "section_title", "") or matched_section
                            break

        # Check if best_sentence is a heading or equals section_title
        is_heading = _is_heading_sent(best_sentence, matched_section)

        best_passage = evidence_passages[best_passage_idx]
        best_passage_text = getattr(best_passage, "text", "") if hasattr(best_passage, "text") else str(best_passage)
        passage_sents = split_sentences(best_passage_text)

        body_sents = [
            s.strip() for s in passage_sents
            if s.strip() and not _is_heading_sent(s, matched_section)
        ]
        if is_heading:
            if not is_general_section_query and len(body_sents) >= 2 and has_specific_aspect:
                ranked_body = sorted(
                    enumerate(body_sents),
                    key=lambda item: (_score_body_sentence(item[1]), -item[0]),
                    reverse=True,
                )
                best_sentence = ranked_body[0][1]
                is_heading = False
        elif has_specific_aspect and not is_general_section_query and len(body_sents) >= 2:
            ranked_body = sorted(
                enumerate(body_sents),
                key=lambda item: (_score_body_sentence(item[1]), -item[0]),
                reverse=True,
            )
            top_body = ranked_body[0][1]
            if _score_body_sentence(top_body) >= _score_body_sentence(best_sentence) + 3.0:
                best_sentence = top_body

        if is_heading:
            # If best sentence is a heading, answer with heading plus next 2-3 body sentences
            idx_in_passage = -1
            for i, s in enumerate(passage_sents):
                if best_sentence[:40] in s or s[:40] in best_sentence:
                    idx_in_passage = i
                    break
            if idx_in_passage >= 0:
                heading_part = passage_sents[idx_in_passage].strip()
                next_sents = [s.strip() for s in passage_sents[idx_in_passage + 1 : idx_in_passage + 4] if s.strip()]
                if next_sents:
                    h_end = "" if heading_part.endswith((".", "!", "?", ":")) else "."
                    answer_core = f"{heading_part}{h_end} {' '.join(next_sents)}"
                else:
                    answer_core = heading_part
            else:
                answer_core = best_sentence
        else:
            # Precision cut: minimal answering span in concise mode, full rich sentence in balanced/detailed
            answer_core = extract_answer_span(exp_query, best_sentence) if best_sentence else ""
            if (
                not answer_core
                or len(answer_core.split()) < 4
                or answer_core.strip().endswith(":")
                or ":" in best_sentence
                or (answer_mode != "concise" and len(answer_core) < len(best_sentence) * 0.75)
            ):
                answer_core = best_sentence

        if answer_core and not answer_core.endswith((".", "!", "?")):
            answer_core += "."

        # Follow-up questions grounded in the evidence that was actually used.
        target_sec_name = dedup_citations[0]["section_title"] if dedup_citations else "chương mục"
        all_context_text = "\n".join(
            getattr(p, "text", "") if hasattr(p, "text") else str(p)
            for p in evidence_passages[:3]
        )
        follow_ups = generate_followup_questions(
            query=query,
            context=all_context_text,
            section_title=target_sec_name,
        )

        if answer_mode == "concise":
            body = f"{answer_core} {cit_1}".strip()
            active_citations = [c for c in dedup_citations if c["citation_label"] == cit_1]
            if not active_citations:
                active_citations = dedup_citations[:1]
        else:
            doc_display_title = dedup_citations[0]["document_title"] if dedup_citations else "tài liệu"
            is_comparison = bool(re.search(
                r"\b(?:so sánh|đối chiếu|khác biệt|khác nhau|so với|compare|comparison|versus|\bvs\b)\b",
                query,
                re.IGNORECASE,
            ))
            asks_full_section = any(
                k in q_norm
                for k in (
                    "toan bo", "lay het", "day du", "tat ca", "co gi lay het",
                    "trong muc", "trong phan", "trong chuong", "trong bai",
                    "co nhung gi", "gom nhung gi", "bao gom nhung gi",
                )
            )
            asks_key_points = any(
                k in q_norm
                for k in (
                    "phan chinh", "lay phan chinh", "y chinh", "lay y chinh",
                    "rut ra y chinh", "noi dung chinh", "diem chinh", "tom tat", "trong tam",
                )
            )
            has_sub_queries = bool(
                sub_query_sentences
                and len(sub_query_sentences) >= 2
                and not (asks_key_points and not is_comparison)
                and not (asks_full_section and not is_comparison)
            )

            if is_comparison:
                # Comparison queries: support cross-passage delta reasoning with compressed grounded statements
                from src.brain.grounded_explainer import compress_clause

                c_core = compress_clause(answer_core, preserve_examples=bool(aspect_example)) or answer_core
                evidence_items: List[Tuple[str, str]] = [(c_core, cit_1)]
                cross_parts = [f"{c_core} {cit_1}"]
                for sq_s, sq_lbl in sub_query_sentences:
                    if sq_s.strip() not in answer_core:
                        s_comp = compress_clause(sq_s, preserve_examples=bool(aspect_example)) or sq_s.strip()
                        s_fmt = s_comp if s_comp.endswith((".", "!", "?")) else f"{s_comp}."
                        cross_parts.append(f"{s_fmt} {sq_lbl}")
                        evidence_items.append((s_fmt, sq_lbl))

                delta_bullet = None
                all_used_pairs = [(best_sentence, cit_1)] + [(s, sentence_cit_labels[i]) for i, s in enumerate(sentence_pool) if s != best_sentence]
                unit_buckets: Dict[str, List[Tuple[float, str, str]]] = {}
                for u_sent, u_lbl in all_used_pairs:
                    for nm in re.finditer(
                        r"(\d+(?:[.,]\d+)?)\s*(%|(?:mb|gb|ms|s|tokens?|triệu|tỷ)\b)",
                        u_sent,
                        re.IGNORECASE,
                    ):
                        raw_num = nm.group(1).replace(",", ".")
                        unit_key = nm.group(2).lower()
                        try:
                            num_val = float(raw_num)
                        except ValueError:
                            continue
                        unit_buckets.setdefault(unit_key, [])
                        if all(abs(num_val - existing[0]) > 1e-6 for existing in unit_buckets[unit_key]):
                            unit_buckets[unit_key].append((num_val, nm.group(0).strip(), u_lbl))

                for unit_key, entries in unit_buckets.items():
                    if len(entries) >= 2:
                        v1, raw1, lbl1 = entries[0]
                        v2, raw2, lbl2 = entries[1]
                        delta_val = round(abs(v1 - v2), 2)
                        delta_str = f"{delta_val:g}"
                        unit_disp = "%" if unit_key == "%" else f" {entries[0][1].split()[-1] if ' ' in entries[0][1] else unit_key}"
                        cit_pair = lbl1 if lbl1 == lbl2 else f"{lbl1} {lbl2}"
                        delta_bullet = (
                            f"- **Đối chiếu định lượng tự động**: Mức chênh lệch giữa hai mốc số liệu được ghi nhận là **{delta_str}{unit_disp}** ({raw1} so với {raw2}). {cit_pair}"
                        )
                        break

                if len(evidence_items) >= 2:
                    reasoning_txt = _build_grounded_reasoning(query, evidence_items, is_comparison=True, has_sub_queries=has_sub_queries)
                    core_bullets = "\n".join(f"- {s} {lbl}" for s, lbl in evidence_items)
                    if reasoning_txt:
                        body = f"**1. Điểm đối chiếu trực tiếp:**\n{core_bullets}\n\n💡 **Liên hệ & Giải thích từ tài liệu:** {reasoning_txt}"
                    else:
                        body = f"**1. Điểm đối chiếu trực tiếp:**\n{core_bullets}"
                else:
                    body = " ".join(cross_parts)
                if delta_bullet:
                    body = f"{body}\n\n{delta_bullet}"
                active_citations = dedup_citations
            elif has_sub_queries:
                # Decomposed compound queries: compress clauses & include answers for all sub-queries + entity synthesis
                from src.brain.grounded_explainer import compress_clause

                evidence_items = []
                for sq_s, sq_lbl in sub_query_sentences:
                    s_comp = compress_clause(sq_s, preserve_examples=bool(aspect_example)) or sq_s.strip()
                    s_fmt = s_comp if s_comp.endswith((".", "!", "?")) else f"{s_comp}."
                    if all(s_fmt != ex_s for ex_s, _ in evidence_items):
                        evidence_items.append((s_fmt, sq_lbl))
                if not evidence_items:
                    c_core = compress_clause(answer_core, preserve_examples=bool(aspect_example)) or answer_core
                    evidence_items.append((c_core, cit_1))
                if len(evidence_items) >= 2:
                    reasoning_txt = _build_grounded_reasoning(query, evidence_items, is_comparison=False, has_sub_queries=True)
                    core_bullets = "\n".join(f"- {s} {lbl}" for s, lbl in evidence_items)
                    if reasoning_txt:
                        body = f"{core_bullets}\n\n💡 **Liên hệ & Giải thích từ tài liệu:** {reasoning_txt}"
                    else:
                        body = core_bullets
                else:
                    body = " ".join(f"{s} {lbl}" for s, lbl in evidence_items)
                active_citations = dedup_citations
            else:
                is_reasoning_cue = any(
                    k in q_norm
                    for k in (
                        "phan biet", "khac nhau", "giong nhau", "tai sao", "vi sao",
                        "nguyen nhan", "co che", "nhu the nao", "the nao", "vai tro",
                        "tac dong", "dan den", "suy ra", "ap dung", "he qua", "lien he",
                        "ban chat", "nguyen tac", "bao toan", "giai thich", "de hieu",
                    )
                )

                if asks_full_section and not asks_key_points:
                    # Exhaustive section extraction across all evidence passages sharing matched_section
                    full_sents: List[str] = []
                    seen_full: Set[str] = set()
                    target_passages = [
                        p for p in evidence_passages
                        if not matched_section or (getattr(p, "section_title", "") or "").strip() == matched_section.strip()
                    ] or [best_passage]
                    for tp in target_passages:
                        tp_text = getattr(tp, "text", "") if hasattr(tp, "text") else str(tp)
                        for s in split_sentences(tp_text):
                            s_clean = s.strip()
                            if s_clean and s_clean not in seen_full:
                                seen_full.add(s_clean)
                                full_sents.append(s_clean if s_clean.endswith((".", "!", "?")) else f"{s_clean}.")
                    if not full_sents:
                        full_sents = [answer_core]
                    body = "\n".join(f"- {s} {cit_1}" for s in full_sents) if len(full_sents) > 1 else f"{full_sents[0]} {cit_1}"
                    active_citations = [c for c in dedup_citations if c["citation_label"] == cit_1] or dedup_citations[:1]
                elif asks_key_points:
                    # Concise 3-5 salient takeaways across matching section passages
                    kp_passages = [
                        p for p in evidence_passages
                        if not matched_section or (getattr(p, "section_title", "") or "").strip() == matched_section.strip()
                    ] or [best_passage]
                    kp_raw = "\n".join(
                        getattr(tp, "text", "") if hasattr(tp, "text") else str(tp)
                        for tp in kp_passages
                    )
                    kp_data = cls.extract_key_points_and_summary(
                        kp_raw,
                        title=matched_section or doc_display_title,
                        max_points=5,
                    )
                    kp_list = kp_data.get("key_points") or [answer_core]
                    kp_lines = []
                    for kp in kp_list[:5]:
                        kp_clean = kp.strip()
                        if kp_clean:
                            if not kp_clean.endswith((".", "!", "?")):
                                kp_clean += "."
                            kp_lines.append(f"- {kp_clean} {cit_1}")
                    body = f"**Các ý chính trọng tâm ({matched_section or doc_display_title}):**\n" + "\n".join(kp_lines)
                    active_citations = [c for c in dedup_citations if c["citation_label"] == cit_1] or dedup_citations[:1]
                elif is_heading:
                    # Preserves clean single-passage prose for bare heading lookups (e.g. 'cho tôi biết topic 49')
                    supporting_sents = []
                    for s in passage_sents:
                        s_clean = s.strip()
                        if s_clean and s_clean not in answer_core and not answer_core.startswith(s_clean):
                            if s_clean not in supporting_sents:
                                supporting_sents.append(s_clean)
                                if len(supporting_sents) >= 2:
                                    break
                    evidence_items = [(answer_core, cit_1)] + [
                        (s if s.endswith((".", "!", "?")) else f"{s}.", cit_1) for s in supporting_sents
                    ]
                    body = " ".join(f"{s} {lbl}" for s, lbl in evidence_items)
                    active_citations = [c for c in dedup_citations if c["citation_label"] == cit_1] or dedup_citations[:1]
                else:
                    # Grounded Discourse Brain: neural salience + MMR role selection + clause compression
                    from src.brain.grounded_explainer import get_grounded_explainer

                    cand_items: List[Dict[str, Any]] = []
                    anchor_text = best_sentence.strip() if best_sentence else answer_core.strip()
                    anchor_fmt = anchor_text if anchor_text.endswith((".", "!", "?")) else f"{anchor_text}."
                    cand_items.append({
                        "sentence": anchor_fmt,
                        "citation_label": cit_1,
                        "section_title": matched_section,
                        "pos_ratio": 0.0,
                        "rule_score": _score_body_sentence(anchor_fmt),
                        "is_anchor": True,
                        "passage_id": dedup_citations[0].get("passage_id", "P1") if dedup_citations else "P1",
                    })
                    sent_cursor = 0
                    for p_idx, p in enumerate(evidence_passages):
                        p_txt = getattr(p, "text", "") if hasattr(p, "text") else str(p)
                        p_sec = getattr(p, "section_title", "") or ""
                        p_cit = passage_to_cit_label[p_idx] if p_idx < len(passage_to_cit_label) else f"[{p_idx+1}]"
                        p_id = getattr(p, "passage_id", f"P{p_idx+1}")
                        p_sents = split_sentences(p_txt)
                        for s_i, s in enumerate(p_sents):
                            s_clean = s.strip()
                            if not s_clean or s_clean in anchor_fmt or _is_heading_sent(s_clean, p_sec):
                                continue
                            s_fmt = s_clean if s_clean.endswith((".", "!", "?")) else f"{s_clean}."
                            cand_items.append({
                                "sentence": s_fmt,
                                "citation_label": p_cit,
                                "section_title": p_sec,
                                "pos_ratio": s_i / max(1, len(p_sents) - 1),
                                "rule_score": _score_body_sentence(s_fmt),
                                "is_anchor": False,
                                "passage_id": p_id,
                            })
                            sent_cursor += 1

                    source_raw_texts = [
                        getattr(p, "text", "") if hasattr(p, "text") else str(p)
                        for p in evidence_passages
                    ]
                    explainer = get_grounded_explainer()
                    body, used_lbls, _ = explainer.explain(
                        query=query,
                        candidates=cand_items,
                        source_texts=source_raw_texts,
                        answer_mode=answer_mode,
                        preserve_examples=bool(aspect_example),
                        include_synthesis=bool(is_reasoning_cue or answer_mode == "detailed"),
                        is_comparison=False,
                    )
                    if not body:
                        body = f"{answer_core} {cit_1}"
                        used_lbls = {cit_1}
                    active_citations = [c for c in dedup_citations if c["citation_label"] in used_lbls]
                    if not active_citations:
                        active_citations = dedup_citations[:1]

            # Multi-facet synthesis across evidence passages (Concept + Concrete Systems + Examples)
            q_norm = strip_accents((query or "").lower())
            asks_types = any(k in q_norm for k in [
                "bao nhieu loai", "cac loai", "nhung loai", "co may loai", "phan loai",
                "cac he", "cac he ma", "he ma", "phuong phap", "may phuong phap",
            ])
            asks_example = any(k in q_norm for k in ["vi du", "cho vi du", "minh hoa", "bai toan"])
            asks_formula = any(k in q_norm for k in ["cong thuc", "bieu thuc", "he thuc", "tinh toan", "phuong trinh"])
            asks_concept = any(k in q_norm for k in ["la gi", "khai niem", "gioi thieu", "tong quan"])
            is_multi_facet = (asks_concept and (asks_types or asks_example)) or (asks_types and asks_example)

            multi_facet_bullets: List[str] = []
            used_citation_labels: Set[str] = {c["citation_label"] for c in active_citations}

            if is_multi_facet or asks_types:
                other_systems: List[Tuple[str, str]] = []
                for p_idx, p in enumerate(evidence_passages):
                    if p_idx == best_passage_idx:
                        continue
                    p_text = getattr(p, "text", "") if hasattr(p, "text") else str(p)
                    p_sec = getattr(p, "section_title", "") or ""
                    p_cit = passage_to_cit_label[p_idx] if p_idx < len(passage_to_cit_label) else f"[{p_idx+1}]"
                    for cand in [p_sec] + p_text.splitlines():
                        cand_clean = cand.strip().rstrip(".:")
                        m = re.match(r"^(?:\d+\.)?\s*(Hệ mã\s+[^\n:]+|HỆ MÃ\s+[^\n:]+|Mã hóa\s+[^\n:]+)", cand_clean, re.IGNORECASE)
                        if m:
                            raw_name = m.group(1).strip()
                            norm_name = strip_accents(raw_name.lower())
                            if not any(strip_accents(existing.lower()) in norm_name or norm_name in strip_accents(existing.lower()) for existing, _ in other_systems):
                                other_systems.append((raw_name, p_cit))
                                used_citation_labels.add(p_cit)
                if other_systems:
                    sys_list_str = ", ".join(f"{name} {lbl}" for name, lbl in other_systems)
                    multi_facet_bullets.append(f"- **Các hệ mã hóa cụ thể trong tài liệu bao gồm:** {sys_list_str}.")

            if is_multi_facet or asks_example or asks_formula or answer_mode == "detailed":
                from src.web.formula_example_scanner import (
                    format_example_blocks_markdown,
                    format_formula_blocks_markdown,
                    scan_passages_for_formulas_and_examples,
                )

                scanned_formulas, scanned_examples = scan_passages_for_formulas_and_examples(
                    evidence_passages,
                    passage_cit_labels=passage_to_cit_label,
                )
                body_norm = strip_accents(body.lower())

                if (asks_formula or answer_mode == "detailed" or is_multi_facet) and scanned_formulas:
                    fresh_formulas = [
                        fb for fb in scanned_formulas
                        if strip_accents(fb.expression.lower()) not in body_norm
                    ]
                    target_formulas = fresh_formulas if fresh_formulas else (scanned_formulas if (asks_formula or len(scanned_formulas) >= 2) else [])
                    if target_formulas:
                        multi_facet_bullets.append(format_formula_blocks_markdown(target_formulas))
                        for fb in target_formulas[:5]:
                            used_citation_labels.add(fb.citation_label)

                if (asks_example or answer_mode == "detailed" or is_multi_facet) and scanned_examples:
                    fresh_examples = [
                        eb for eb in scanned_examples
                        if strip_accents(eb.title.lower()) not in body_norm
                        or any(strip_accents(st.lower()) not in body_norm for st in eb.steps)
                    ]
                    if fresh_examples:
                        multi_facet_bullets.append(format_example_blocks_markdown(fresh_examples))
                        for eb in fresh_examples[:2]:
                            used_citation_labels.add(eb.citation_label)
                elif is_multi_facet or asks_example:
                    examples_found: List[Tuple[str, str]] = []
                    for p_idx, p in enumerate(evidence_passages):
                        p_text = getattr(p, "text", "") if hasattr(p, "text") else str(p)
                        p_cit = passage_to_cit_label[p_idx] if p_idx < len(passage_to_cit_label) else f"[{p_idx+1}]"
                        for line in p_text.splitlines():
                            line_clean = line.strip().lstrip("-*• ")
                            line_norm = strip_accents(line_clean.lower())
                            if any(marker in line_norm for marker in ["cho p =", "vi du", "cong thuc ma hoa", "ta co: zi =", "p = c = k"]):
                                if line_clean and not any(line_clean in ex for ex, _ in examples_found):
                                    examples_found.append((line_clean, p_cit))
                                    used_citation_labels.add(p_cit)
                                    if len(examples_found) >= 2:
                                        break
                        if len(examples_found) >= 2:
                            break
                    if examples_found:
                        ex_list_str = "; ".join(f"{ex} {lbl}" for ex, lbl in examples_found)
                        multi_facet_bullets.append(f"- **Ví dụ và công thức minh họa:** {ex_list_str}.")

            if multi_facet_bullets:
                body = f"{body}\n\n" + "\n".join(multi_facet_bullets)
                active_citations = [c for c in dedup_citations if c["citation_label"] in used_citation_labels]
                if not active_citations:
                    active_citations = dedup_citations

            if answer_mode == "detailed":
                # Defense Strategist integration (Wave 3)
                try:
                    from src.brain.defense_strategist import DefenseStrategist
                    strategist = DefenseStrategist()
                    plan = strategist.plan(query)
                    if plan.archetype != "GENERIC" and plan.confidence >= 0.35:
                        cand_dicts = [
                            {
                                "sentence": s,
                                "citation_label": lbl,
                                "section_title": sec,
                            }
                            for s, lbl, sec in zip(sentence_pool, sentence_cit_labels, sentence_sections)
                        ]
                        arg_block = strategist.assemble_argument(plan, cand_dicts)
                        if arg_block and "- **" in arg_block:
                            body = f"{body}\n\n{arg_block}"
                except Exception:
                    pass

                sec_list = sorted({c["section_title"] for c in dedup_citations if c.get("section_title")})
                sec_summary = ", ".join(f"`{s}`" for s in sec_list[:3]) if sec_list else f"**{doc_display_title}**"
                cit_badges = " ".join(c["citation_label"] for c in dedup_citations[:3])
                detailed_synth = ""
                try:
                    from src.brain.grounded_explainer import extract_frame, get_grounded_explainer
                    detail_cands = [
                        extract_frame(sentence=s, role="MECHANISM", salience=1.0, citation_label=lbl)
                        for s, lbl in zip(sentence_pool[:4], sentence_cit_labels[:4])
                        if len(s.split()) >= 5
                    ]
                    if len(detail_cands) >= 2:
                        detailed_synth = get_grounded_explainer().build_entity_grounded_synthesis(
                            query=query,
                            frames=detail_cands,
                            is_comparison=False,
                        )
                except Exception:
                    detailed_synth = ""
                if not detailed_synth:
                    detailed_synth = (
                        f"Các dữ kiện trọng tâm từ {sec_summary} trong **{doc_display_title}** {cit_badges} "
                        f"liên kết chặt chẽ và toàn bộ chi tiết được đối chiếu trực tiếp từ văn bản gốc."
                    )
                body = (
                    f"{body}\n\n"
                    f"💡 **Tổng hợp & Đúc kết chuyên sâu:**\n"
                    f"{detailed_synth}"
                )

        dedup_citations = active_citations

        # Sources first, then the mandatory trailing question block.
        sources_lines = [
            f"{c['citation_label']} **{c['document_title']}** — {c['section_title']} — Đoạn {c['paragraph_index']}"
            for c in dedup_citations[:3]
        ]

        body = strip_question_block(body)
        if sources_lines:
            body = f"{body}\n\n### Nguồn tham khảo\n" + "\n\n".join(sources_lines)

        full_answer = ensure_trailing_question(body, follow_ups)
        return full_answer, dedup_citations

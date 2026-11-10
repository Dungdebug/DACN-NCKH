"""
Phase 5.9 Extension — Document Intelligence & Auto Question Generator.
Extracts structured executive summaries, key takeaways, and generates
diverse, relevant, document-grounded questions from hierarchical SA-CMS structures.
100% Local Offline, Zero External API, Zero Network Egress.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


def strip_accents(s: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive matching."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


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

        # Select sentences with strong informational markers
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
        key_points = []
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
                q = f"Nội dung trọng tâm được trình bày tại mục '{sec_name}' là gì?"
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
        rel = max(
            relevance_score(query, best_sentence, extra=matched_section),
            relevance_score(exp_query, best_sentence, extra=matched_section),
        )
        if not best_sentence or rel < MIN_RELEVANCE:
            return "Không tìm thấy đủ thông tin bằng chứng trong tài liệu để trả lời câu hỏi này.", []

        # Deduplicate citations: one citation per distinct document & section.
        dedup_citations = []
        seen_keys = set()
        for idx, p in enumerate(evidence_passages, start=1):
            doc_id = getattr(p, "document_id", "DOC")
            doc_title = getattr(p, "document_title", doc_id) or doc_id
            sec_title = getattr(p, "section_title", "Phần nội dung") or "Phần nội dung"
            para_idx = getattr(p, "paragraph_index", idx)

            key = (doc_id, sec_title, para_idx)
            if key not in seen_keys:
                seen_keys.add(key)
                dedup_citations.append({
                    "citation_index": len(dedup_citations) + 1,
                    "citation_label": f"[{len(dedup_citations) + 1}]",
                    "passage_id": getattr(p, "passage_id", f"PASSAGE_{idx}"),
                    "document_id": doc_id,
                    "document_title": doc_title,
                    "section_title": sec_title,
                    "paragraph_index": para_idx,
                    "score": round(float(getattr(p, "score", 1.0)), 4),
                    "text": getattr(p, "text", "")[:400],
                })

        cit_1 = dedup_citations[0]["citation_label"] if dedup_citations else "[1]"

        # Precision cut: minimal answering span, fallback to best_sentence if span is fragmented or cuts key-values
        answer_core = extract_answer_span(exp_query, best_sentence) if best_sentence else ""
        if not answer_core or len(answer_core.split()) < 4 or answer_core.strip().endswith(":") or ":" in best_sentence:
            answer_core = best_sentence
        elif not answer_core:
            answer_core = best_sentence

        if answer_core and not answer_core.endswith((".", "!", "?")):
            answer_core += "."

        # Follow-up questions grounded in the evidence that was actually used.
        target_sec_name = dedup_citations[0]["section_title"] if dedup_citations else "chương mục"
        source_context = getattr(evidence_passages[best_passage_idx], "text", "") if evidence_passages else ""
        follow_ups = generate_followup_questions(
            query=query,
            context=source_context,
            section_title=target_sec_name,
        )

        if answer_mode == "concise":
            body = f"{answer_core} {cit_1}".strip()
        else:
            parts = [f"**Nội dung cốt lõi:**\n{answer_core} {cit_1}"]

            supporting = [
                s for s in split_sentences(source_context)
                if s and s != best_sentence
            ][:2]
            if supporting:
                parts.append(f"📌 **Bối cảnh bổ sung:**\n{' '.join(supporting)}")

            body = "\n\n".join(parts)

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

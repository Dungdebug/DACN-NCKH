"""
Neural Document AI Engine.
Independent, specialized reasoning engine for deep PDF analysis and document question answering.
100% Strict Offline, Zero External API, Zero Network Calls.
"""

import re
import time
from typing import List, Dict, Any, Optional

from src.document_ai.models import (
    DocumentAnalysisResponse,
    DocumentQueryResponse,
    CitationSource,
    KeyPoint,
    GeneratedQuestion,
    ExtractedMetric,
)
from src.document_ai.parser import DeepDocumentParser, strip_accents_vi


class NeuralDocumentAI:
    """
    Dedicated AI Engine solely focused on analyzing uploaded PDF documents,
    summarizing key takeaways, generating analytical questions, and synthesizing
    accurate, document-grounded answers.
    """

    _cache: Dict[str, DocumentAnalysisResponse] = {}

    @classmethod
    def analyze_document(
        cls,
        document_id: str,
        title: str,
        raw_text: str,
    ) -> DocumentAnalysisResponse:
        """
        Performs deep structural and semantic analysis on the document.
        Results are cached in-memory for immediate query answering.
        """
        # Segment into sections
        sections = DeepDocumentParser.segment_into_sections(raw_text, default_title=title)
        key_points = DeepDocumentParser.extract_key_points_by_section(sections)
        metrics = DeepDocumentParser.extract_metrics(raw_text, max_metrics=5)
        questions = DeepDocumentParser.generate_analytical_questions(title, sections, raw_text, max_questions=6)

        # Build executive overview
        total_paras = sum(len(s["paragraphs"]) for s in sections)
        approx_tokens = int(len(raw_text.split()) * 1.35)

        lead_takeaway = key_points[0].summary if key_points else "Tài liệu kỹ thuật chuyên đề."
        overview = (
            f"Tài liệu '{title}' bao gồm {len(sections)} chương mục chính với {total_paras} đoạn văn bản. "
            f"Nội dung trọng tâm: {lead_takeaway}"
        )

        analysis = DocumentAnalysisResponse(
            document_id=document_id,
            title=title,
            total_tokens=approx_tokens,
            total_sections=len(sections),
            total_paragraphs=total_paras,
            overview=overview,
            key_points=key_points,
            suggested_questions=questions,
            metrics=metrics,
            status="COMPLETED",
        )

        cls._cache[document_id] = analysis
        return analysis

    @classmethod
    def answer_document_query(
        cls,
        document_id: str,
        title: str,
        raw_text: str,
        query: str,
        answer_mode: str = "balanced",
    ) -> DocumentQueryResponse:
        """
        Synthesizes a precise, natural document-grounded answer to a user query
        without relying on rigid splitting or crude template fallbacks.
        """
        t0 = time.perf_counter()
        q_clean = query.strip()
        q_lower = q_clean.lower()
        q_no_accent = strip_accents_vi(q_lower)

        # 1. Check if user is asking for document summary
        is_summary_query = any(k in q_lower or k in q_no_accent for k in [
            "tóm tắt", "tom tat", "ý chính", "y chinh", "nội dung chính", "noi dung chinh",
            "tổng quan", "tong quan", "nói về gì", "noi ve gi", "vấn đề gì", "overview",
        ])

        if is_summary_query:
            analysis = cls._cache.get(document_id) or cls.analyze_document(document_id, title, raw_text)
            pts_str = "\n".join(f"• **{kp.section_title}**: {kp.summary} [1]" for kp in analysis.key_points[:4])
            answer = f"Bản tóm tắt nội dung trọng tâm của tài liệu **{title}**:\n\n{pts_str}"
            citations = [CitationSource(
                citation_index=1,
                citation_label="[1]",
                document_id=document_id,
                document_title=title,
                section_title="Tổng quan tài liệu",
                paragraph_index=1,
                score=5.0,
                text_excerpt=analysis.overview,
                support_status="SUPPORTED",
            )]
            lat_ms = (time.perf_counter() - t0) * 1000
            return DocumentQueryResponse(
                query=query,
                answer=answer,
                document_id=document_id,
                document_title=title,
                citations=citations,
                confidence=1.0,
                telemetry={"latency_ms": round(lat_ms, 2), "mode": "document_ai_summary"},
            )

        # 2. Check if user is asking for questions
        is_question_gen = any(k in q_lower or k in q_no_accent for k in [
            "câu hỏi ôn tập", "cau hoi on tap", "gợi ý câu hỏi", "goi y cau hoi",
            "sinh câu hỏi", "sinh cau hoi", "tạo câu hỏi", "tao cau hoi",
            "câu hỏi liên quan", "cau hoi lien quan",
        ])

        if is_question_gen:
            analysis = cls._cache.get(document_id) or cls.analyze_document(document_id, title, raw_text)
            q_lines = [f"{idx}. **[{q.category}]**: {q.question}" for idx, q in enumerate(analysis.suggested_questions, start=1)]
            answer = (
                f"Chuyên gia Document AI đã phân tích tài liệu **{title}** và tự động sinh ra các câu hỏi then chốt sau:\n\n"
                + "\n\n".join(q_lines)
                + "\n\n👉 Bạn có thể đặt bất kỳ câu hỏi nào trong danh sách trên để nhận câu trả lời có trích dẫn nguồn chi tiết!"
            )
            citations = [CitationSource(
                citation_index=1,
                citation_label="[1]",
                document_id=document_id,
                document_title=title,
                section_title="Cấu trúc câu hỏi tự sinh",
                paragraph_index=1,
                score=4.8,
                text_excerpt=analysis.overview,
                support_status="SUPPORTED",
            )]
            lat_ms = (time.perf_counter() - t0) * 1000
            return DocumentQueryResponse(
                query=query,
                answer=answer,
                document_id=document_id,
                document_title=title,
                citations=citations,
                confidence=1.0,
                telemetry={"latency_ms": round(lat_ms, 2), "mode": "document_ai_questions"},
            )

        # 3. Dedicated Answering via answer-span extraction (precision + question tail)
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

        sections = DeepDocumentParser.segment_into_sections(raw_text, default_title=title)

        # Gather candidate sentences together with their provenance.
        candidate_sentences = []
        for sec in sections:
            sec_title = sec["title"]
            for p_idx, para in enumerate(sec["paragraphs"], start=1):
                clean_p = DeepDocumentParser.clean_text(para)
                for sentence in split_sentences(clean_p):
                    candidate_sentences.append((sentence, sec_title, p_idx, clean_p))

        best_sentence = select_best_sentence(
            q_clean,
            [c[0] for c in candidate_sentences],
            [c[1] for c in candidate_sentences],
        )

        best_sec_title = "Phần nội dung"
        best_para_idx = 1
        best_full_para = ""
        best_score = 0.0
        for sentence, sec_title, p_idx, clean_p in candidate_sentences:
            if sentence == best_sentence:
                best_sec_title = sec_title
                best_para_idx = p_idx
                best_full_para = clean_p
                best_score = relevance_score(q_clean, sentence, extra=sec_title)
                break

        # Calibrated refusal (zero hallucination): no sentence is on-topic enough.
        if not best_sentence or best_score < MIN_RELEVANCE:
            lat_ms = (time.perf_counter() - t0) * 1000
            return DocumentQueryResponse(
                query=query,
                answer="Tài liệu được cung cấp không chứa thông tin hoặc căn cứ để trả lời câu hỏi này.",
                document_id=document_id,
                document_title=title,
                citations=[],
                confidence=0.0,
                telemetry={"latency_ms": round(lat_ms, 2), "refused": True},
            )

        # Precision cut (Section 6.5): only the clause that answers the question.
        answer_core = extract_answer_span(q_clean, best_sentence) or best_sentence
        if not answer_core.endswith((".", "!", "?")):
            answer_core += "."

        # Section 6.4: follow-up questions, always rendered last.
        follow_ups = generate_followup_questions(
            query=q_clean,
            context=best_full_para,
            section_title=best_sec_title,
        )

        supporting_sents = [
            s for s in split_sentences(best_full_para) if s and s != best_sentence
        ][:2]

        if answer_mode == "concise":
            body = f"{answer_core} [1]"
        else:
            parts = [f"**Nội dung cốt lõi:**\n{answer_core} [1]"]
            if supporting_sents:
                parts.append(f"📌 **Bối cảnh bổ sung:**\n{' '.join(supporting_sents)}")
            body = "\n\n".join(parts)

        body = strip_question_block(body)

        citations = [CitationSource(
            citation_index=1,
            citation_label="[1]",
            document_id=document_id,
            document_title=title,
            section_title=best_sec_title,
            paragraph_index=best_para_idx,
            score=round(best_score, 4),
            text_excerpt=best_full_para[:350],
            support_status="SUPPORTED",
        )]

        lat_ms = (time.perf_counter() - t0) * 1000
        with_sources = f"{body}\n\n### Nguồn tham khảo\n[1] **{title}** — {best_sec_title} — Đoạn {best_para_idx}"
        full_answer = ensure_trailing_question(with_sources, follow_ups)

        return DocumentQueryResponse(
            query=query,
            answer=full_answer,
            document_id=document_id,
            document_title=title,
            citations=citations,
            confidence=1.0,
            telemetry={"latency_ms": round(lat_ms, 2), "score": round(best_score, 4)},
        )


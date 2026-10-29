"""
Phase 5.2 — Question Router & Answer Strategy Engine.
Classifies user questions into structured routing strategies to drive evidence-first QA.

STRATEGIES:
- DIRECT_LOOKUP: Specific factual queries, definitions, entities.
- SECTION_LOOKUP: Chapter/section/heading specific queries.
- PAGE_LOOKUP: Specific page/location queries.
- DOCUMENT_SUMMARY: Holistic document overview / synthesis.
- CROSS_DOCUMENT_COMPARISON: Comparing concepts across multiple documents.
- MULTI_PASSAGE_SYNTHESIS: Multi-aspect queries requiring synthesis across multiple passages.
- UNANSWERABLE: Out-of-domain questions with no document grounding (refusal required).
- INSUFFICIENT_EVIDENCE: Ambiguous or weak matches failing relevance threshold.

STRICT RULE: NO TRAINING, NO GRADIENT UPDATES, FROZEN INFERENCE ONLY.
"""

import re
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple


class QuestionRoute(str, Enum):
    DIRECT_LOOKUP = "DIRECT_LOOKUP"
    SECTION_LOOKUP = "SECTION_LOOKUP"
    PAGE_LOOKUP = "PAGE_LOOKUP"
    DOCUMENT_SUMMARY = "DOCUMENT_SUMMARY"
    CROSS_DOCUMENT_COMPARISON = "CROSS_DOCUMENT_COMPARISON"
    MULTI_PASSAGE_SYNTHESIS = "MULTI_PASSAGE_SYNTHESIS"
    UNANSWERABLE = "UNANSWERABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class QuestionRouter:
    """Routes queries to appropriate retrieval, synthesis, and grounding strategies."""

    # Out-of-domain patterns (world knowledge not in academic/enterprise docs)
    OUT_OF_DOMAIN_PATTERNS = [
        r"\bthủ đô\b", r"\bcapital of\b",
        r"\bthời tiết\b", r"\bweather\b",
        r"\bbóng đá\b", r"\bfootball\b", r"\bworld cup\b",
        r"\btổng thống\b", r"\bpresident of\b",
        r"\bgiá vàng\b", r"\bgold price\b",
        r"\bxổ số\b", r"\blottery\b",
        r"\bmón ăn\b", r"\brecipe\b",
    ]

    # Document summary patterns
    SUMMARY_PATTERNS = [
        r"\btóm tắt\b", r"\bsummary\b", r"\bsummarize\b",
        r"\bnội dung chính\b", r"\bkhái quát\b", r"\btổng quan\b",
        r"\btài liệu này nói về\b", r"\btài liệu này viết về\b",
        r"\bwhat is this document about\b", r"\boverview\b",
        r"\bcho tôi biết nội dung\b", r"\bnói về vấn đề gì\b",
    ]

    # Section lookup patterns
    SECTION_PATTERNS = [
        r"\bchương\s+\d+\b", r"\bchương\s+[ivxcdm]+\b", r"\bchapter\s+\d+\b",
        r"\bphần\s+\d+\b", r"\bphần\s+[ivxcdm]+\b", r"\bsection\s+\d+\b",
        r"\bmục\s+\d+(\.\d+)*\b", r"\bđiều\s+\d+\b", r"\bkhoản\s+\d+\b",
    ]

    # Page lookup patterns
    PAGE_PATTERNS = [
        r"\btrang\s+\d+\b", r"\btrang số\s+\d+\b", r"\bpage\s+\d+\b",
    ]

    # Cross-document comparison patterns
    COMPARISON_PATTERNS = [
        r"\bso sánh\b", r"\bcompare\b", r"\bcomparison\b",
        r"\bkhác nhau\b", r"\bgiống nhau\b", r"\bđiểm khác\b",
        r"\bgiữa các tài liệu\b", r"\btrong các tài liệu\b",
        r"\bacross documents\b", r"\bbetween documents\b",
        r"\bkhác biệt giữa\b", r"\bđối chiếu\b",
    ]

    # Multi-passage synthesis patterns
    SYNTHESIS_PATTERNS = [
        r"\bnguyên nhân\b", r"\bcause\b", r"\breason\b",
        r"\bưu nhược điểm\b", r"\bpros and cons\b", r"\badvantages\b",
        r"\bcác bước\b", r"\bquy trình\b", r"\bsteps\b", r"\bworkflow\b",
        r"\bphân tích\b", r"\banalyze\b", r"\bđánh giá\b",
        r"\bcó nói gì về\b", r"\bđề cập gì về\b", r"\bdoes it mention\b",
    ]

    @classmethod
    def classify_intent(
        cls,
        query: str,
        num_attached_documents: int = 1,
    ) -> Tuple[QuestionRoute, Dict[str, Any]]:
        """
        Classifies query intent into QuestionRoute and metadata clues.
        """
        q_norm = query.lower().strip()

        # 1. Check Out-of-domain obvious queries
        for pat in cls.OUT_OF_DOMAIN_PATTERNS:
            if re.search(pat, q_norm, re.IGNORECASE):
                return QuestionRoute.UNANSWERABLE, {
                    "matched_pattern": pat,
                    "reason": "out_of_domain_query",
                    "explanation": "Câu hỏi thuộc tri thức thế giới ngoài tài liệu.",
                }

        # 2. Check Comparison across documents
        if num_attached_documents > 1:
            for pat in cls.COMPARISON_PATTERNS:
                if re.search(pat, q_norm, re.IGNORECASE):
                    return QuestionRoute.CROSS_DOCUMENT_COMPARISON, {
                        "matched_pattern": pat,
                        "documents_scope": num_attached_documents,
                        "explanation": "So sánh thông tin giữa các tài liệu.",
                    }

        # 3. Check Page lookup (specific page)
        for pat in cls.PAGE_PATTERNS:
            match = re.search(pat, q_norm, re.IGNORECASE)
            if match:
                return QuestionRoute.PAGE_LOOKUP, {
                    "matched_pattern": pat,
                    "target_page": match.group(0),
                    "explanation": "Tra cứu thông tin theo trang cụ thể.",
                }

        # 4. Check Section lookup (specific chapter / section)
        for pat in cls.SECTION_PATTERNS:
            match = re.search(pat, q_norm, re.IGNORECASE)
            if match:
                return QuestionRoute.SECTION_LOOKUP, {
                    "matched_pattern": pat,
                    "target_section": match.group(0),
                    "explanation": "Tra cứu thông tin theo chương/mục cụ thể.",
                }

        # 5. Check Document summary
        for pat in cls.SUMMARY_PATTERNS:
            if re.search(pat, q_norm, re.IGNORECASE):
                return QuestionRoute.DOCUMENT_SUMMARY, {
                    "matched_pattern": pat,
                    "explanation": "Yêu cầu tóm tắt/khái quát nội dung tài liệu.",
                }

        # 6. Check Multi-passage synthesis
        for pat in cls.SYNTHESIS_PATTERNS:
            if re.search(pat, q_norm, re.IGNORECASE):
                return QuestionRoute.MULTI_PASSAGE_SYNTHESIS, {
                    "matched_pattern": pat,
                    "explanation": "Tổng hợp thông tin từ nhiều đoạn khác nhau.",
                }

        # 7. Default to Direct Lookup (Definitions, specific facts)
        return QuestionRoute.DIRECT_LOOKUP, {
            "explanation": "Tra cứu trực tiếp định nghĩa hoặc dữ kiện cụ thể.",
        }

    @classmethod
    def evaluate_evidence_grounding(
        cls,
        route: QuestionRoute,
        evidence_passages: List[Any],
        score_threshold: float = 3.0,
        min_evidence: int = 1,
    ) -> Tuple[QuestionRoute, bool, str]:
        """
        Validates whether retrieved evidence is sufficient for the classified route.
        Returns (effective_route, is_grounded, explanation_vi).
        """
        if route == QuestionRoute.UNANSWERABLE:
            return (
                QuestionRoute.UNANSWERABLE,
                False,
                "Tài liệu bạn cung cấp không có thông tin để trả lời câu hỏi này.",
            )

        if not evidence_passages:
            return (
                QuestionRoute.UNANSWERABLE,
                False,
                "Tài liệu không đề cập đến thông tin này.",
            )

        # Check top evidence score
        top_score = getattr(evidence_passages[0], "score", 0.0)

        # In Document Summary mode, even broader passages are acceptable if relevance exists
        if route == QuestionRoute.DOCUMENT_SUMMARY:
            return (
                QuestionRoute.DOCUMENT_SUMMARY,
                True,
                "Đã tổng hợp được ngữ cảnh toàn diện từ tài liệu.",
            )

        if top_score < score_threshold:
            return (
                QuestionRoute.INSUFFICIENT_EVIDENCE,
                False,
                "Thông tin trong tài liệu chưa đủ để xác nhận câu trả lời.",
            )

        if len(evidence_passages) < min_evidence:
            return (
                QuestionRoute.INSUFFICIENT_EVIDENCE,
                False,
                "Bằng chứng trong tài liệu quá mỏng để đưa ra kết luận chắc chắn.",
            )

        return (
            route,
            True,
            "Đã tìm thấy bằng chứng xác thực trong tài liệu.",
        )

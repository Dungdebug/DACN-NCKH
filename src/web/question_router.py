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
        r"\btóm tắt\b", r"\btóm lược\b", r"\btóm gọn\b", r"\btổng kết\b",
        r"\bsummary\b", r"\bsummarize\b", r"\boverview\b",
        r"\bnội dung chính\b", r"\bkhái quát\b", r"\btổng quan\b", r"\bđại ý\b",
        r"\bý chính\b", r"\bđiểm chính\b", r"\bđiểm nổi bật\b", r"\btrọng tâm\b",
        r"\bgạch đầu dòng\b", r"\bcho vài ý\b", r"\bnói chung là\b",
        r"\bkể cái gì\b", r"\bviết cái gì\b", r"\bnói cái gì\b",
        r"\bý nào quan trọng\b", r"\bcó gì quan trọng\b",
        r"\bnhững ý quan trọng\b", r"\bthông tin chính\b",
        r"\btài liệu này nói về\b", r"\btài liệu này viết về\b",
        r"\bwhat is this document about\b",
        r"\bcho tôi biết nội dung\b", r"\bnói về vấn đề gì\b",
        r"\btrong này có gì\b", r"\bcó gì trong này\b",
        r"\bfile này có gì\b", r"\btài liệu này có gì\b",
        r"\btài liệu có gì\b", r"\bfile có gì\b",
        r"\bnói về cái gì\b", r"\bviết về cái gì\b",
        r"\bnội dung là gì\b", r"\bnội dung của tài liệu\b",
        r"\bgiới thiệu tài liệu\b", r"\btrong file có gì\b",
        r"\bfile này nói gì\b", r"\bfile này viết gì\b",
        r"\bfile này kể gì\b", r"\btài liệu này kể gì\b",
    ]

    # Section lookup patterns
    SECTION_PATTERNS = [
        r"\bchương\s+\d+\b", r"\bchương\s+[ivxcdm]+\b", r"\bchapter\s+\d+\b",
        r"\bphần\s+\d+\b", r"\bphần\s+[ivxcdm]+\b", r"\bsection\s+\d+\b",
        r"\bmục\s+\d+(\.\d+)*\b", r"\bđiều\s+\d+\b", r"\bkhoản\s+\d+\b",
        r"\btopic\s*\d+\b", r"\bchủ\s*đề\s*\d+\b", r"\bbài\s*\d+\b",
        r"\blesson\s*\d+\b", r"\bunit\s*\d+\b",
    ]

    # Explicit scope patterns: topic aliases + N or quoted span after "liên quan đến/về"
    TOPIC_SCOPE_PATTERN = re.compile(
        r"\b(topic|chủ\s*đề|bài|mục|chương|phần|section|chapter|lesson|unit)\s*(\d+)\b",
        re.IGNORECASE,
    )
    QUOTED_SCOPE_PATTERN = re.compile(
        r"(?:liên\s+quan\s+đến|liên\s+quan\s+về|về)\s+['\"“]([^'\"”]+)['\"”]",
        re.IGNORECASE,
    )
    NUMBERED_HEADING_PATTERN = re.compile(
        r"^\s*(\d+)\.\s*([A-Za-zÀ-ỹ\s]+)",
        re.IGNORECASE,
    )
    GENERAL_QUOTED_PATTERN = re.compile(
        r"['\"“]([^'\"”]{2,80})['\"”]",
        re.IGNORECASE,
    )

    META_QUESTION_STOPWORDS = {
        "yêu", "cầu", "chi", "tiết", "cụ", "thể", "liên", "quan", "đến", "nội", "dung",
        "triển", "khai", "mô", "tả", "trình", "bày", "giải", "thích", "chính", "mục",
        "gồm", "những", "gì", "bao", "nhiêu", "loại", "cách", "thức", "cho", "biết",
        "hãy", "nêu", "tìm", "hiểu", "về", "là", "và", "của", "trong", "có", "các",
        "được", "ra", "sao", "thế", "nào", "khái", "niệm", "nguyên", "lý", "đặc", "điểm",
        "ví", "dụ", "như", "nào", "ở", "tại", "với", "theo", "hỏi", "đáp", "bài",
    }

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

        # 4. Scope Check BEFORE SUMMARY: any alias+N or quoted span or numbered heading
        topic_match = cls.TOPIC_SCOPE_PATTERN.search(q_norm)
        if topic_match:
            alias_name = topic_match.group(1).title()
            num = topic_match.group(2)
            target = f"{alias_name} {num}"
            return QuestionRoute.SECTION_LOOKUP, {
                "matched_pattern": cls.TOPIC_SCOPE_PATTERN.pattern,
                "target_section": target,
                "scope_type": "topic_alias",
                "explanation": f"Tra cứu thông tin theo {target} cụ thể.",
            }

        numbered_match = cls.NUMBERED_HEADING_PATTERN.match(query.strip())
        if numbered_match:
            target = query.strip()
            return QuestionRoute.SECTION_LOOKUP, {
                "matched_pattern": cls.NUMBERED_HEADING_PATTERN.pattern,
                "target_section": target,
                "scope_type": "numbered_heading",
                "explanation": f"Tra cứu thông tin theo mục '{target}' cụ thể.",
            }

        quoted_match = cls.QUOTED_SCOPE_PATTERN.search(query)
        if not quoted_match:
            quoted_match = cls.GENERAL_QUOTED_PATTERN.search(query)

        if quoted_match:
            target = quoted_match.group(1).strip()
            return QuestionRoute.SECTION_LOOKUP, {
                "matched_pattern": getattr(quoted_match, "re", cls.GENERAL_QUOTED_PATTERN).pattern,
                "target_section": target,
                "scope_type": "quoted_scope",
                "explanation": f"Tra cứu thông tin theo phạm vi '{target}'.",
            }

        # Check general Section lookup patterns
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

        # 7. Check Exploratory / Conversational summary phrases when documents are attached
        if num_attached_documents > 0:
            exploratory_phrases = [
                r"\bnói\s+sơ\b", r"\bsơ\s+qua\b", r"\bnói\s+qua\b",
                r"\bkể\s+(sơ|qua|nghe)\b", r"\bkể\s+cái\s+gì\b", r"\bkể\s+gì\b",
                r"\btrong\s+(này|file|đây|tài liệu)\s+có\s+gì\b",
                r"\bcó\s+gì\s+trong\s+(này|file|đây|tài liệu)\b",
                r"\b(bài|file|tài liệu)\s+này\s+(viết|nói|kể|chứa)\s+về\s+(cái\s+)?gì\b",
                r"\b(bài|file|tài liệu)\s+này\s+(viết|nói|kể)\s+(cái\s+)?gì\b",
                r"\bgiúp\s+(tớ|mình|tôi|em)\s+hiểu\b",
                r"\bhiểu\s+(sơ\s+)?về\s+tài\s+liệu\b",
                r"\bđọc\s+hộ\b", r"\bđọc\s+giúp\b",
                r"\bcho\s+(vài\s+)?gạch\s+đầu\s+dòng\b",
                r"\bý\s+(nào\s+)?quan\s+trọng\b",
                r"\b(điểm|ý)\s+chính\b",
                r"\btóm\s+(lược|gọn|tắt)\b",
                r"\bnói\s+chung\b",
                r"\bquan\s+trọng\s+nhất\b",
            ]
            for ep in exploratory_phrases:
                if re.search(ep, q_norm, re.IGNORECASE):
                    return QuestionRoute.DOCUMENT_SUMMARY, {
                        "matched_pattern": ep,
                        "explanation": "Câu hỏi thăm dò/khái quát nội dung về tài liệu đang ghim.",
                    }

        # 8. Default to Direct Lookup (Definitions, specific facts)
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

    @classmethod
    def _extract_query_terms(cls, query: str) -> List[str]:
        """
        Extract core informative terms from query, stripping meta-question boilerplate.
        If query contains a quoted entity, tokens from that entity are prioritized as #1 target.
        """
        if not query or not query.strip():
            return []

        # 1. Check for quoted entity
        quoted_match = cls.GENERAL_QUOTED_PATTERN.search(query)
        priority_terms: List[str] = []
        if quoted_match:
            quoted_text = quoted_match.group(1).strip()
            raw_quoted_tokens = re.findall(r"[\w]+", quoted_text.lower(), re.UNICODE)
            priority_terms = [t for t in raw_quoted_tokens if t not in cls.META_QUESTION_STOPWORDS and (len(t) >= 2 or t.isdigit())]
            if not priority_terms:
                priority_terms = [t for t in raw_quoted_tokens if len(t) >= 2 or t.isdigit()]

        # 2. Extract remaining words from query
        raw_tokens = re.findall(r"[\w]+", query.lower(), re.UNICODE)
        core_terms: List[str] = []
        for t in raw_tokens:
            if t not in cls.META_QUESTION_STOPWORDS and (len(t) >= 2 or t.isdigit()):
                if t not in priority_terms and t not in core_terms:
                    core_terms.append(t)

        final_terms = priority_terms + core_terms
        return final_terms if final_terms else [t for t in raw_tokens if len(t) >= 2 or t.isdigit()]

    @classmethod
    def extract_query_terms(cls, query: str) -> List[str]:
        return cls._extract_query_terms(query)


# Module-level alias
_extract_query_terms = QuestionRouter._extract_query_terms
extract_query_terms = QuestionRouter.extract_query_terms


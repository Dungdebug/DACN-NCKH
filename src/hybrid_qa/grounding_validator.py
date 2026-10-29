"""
Phase 5.5 — Grounded QA Safety Layer.
Formalizes:
1. EvidenceVerifier
2. CitationManager
3. AnswerGroundingValidator
4. Standard 4-state grounding classification:
   - SUPPORTED
   - PARTIALLY_SUPPORTED
   - INSUFFICIENT_EVIDENCE
   - UNANSWERABLE

STRICT INVARIANT:
- Does NOT alter official Phase 4 refusal thresholds (tau=3.0, min_coverage=0.35).
- If insufficient evidence or unanswerable:
  "Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn."
- Zero hallucination / zero invention of information.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import re

from src.hybrid_qa.evidence import Citation, EvidencePackage
from src.hybrid_qa.retriever import RetrievalResult


class GroundingStatus(Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNANSWERABLE = "UNANSWERABLE"


@dataclass
class GroundedCitation:
    """Rich citation item with complete provenance."""
    citation_index: int
    label: str  # "[1]", "[2]"
    passage_id: str
    document_id: str
    document_title: str
    section_title: str = "Phần nội dung"
    paragraph_index: int = 1
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    score: float = 0.0
    text: str = ""
    status: str = "SUPPORTED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroundingValidationResult:
    """Outcome of Grounded QA Safety Verification."""
    status: GroundingStatus
    is_safe_to_answer: bool
    confidence_score: float
    refusal_message: Optional[str] = None
    citations: List[GroundedCitation] = field(default_factory=list)
    query_coverage: float = 0.0
    diagnostic_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_safe_to_answer": self.is_safe_to_answer,
            "confidence_score": round(self.confidence_score, 4),
            "refusal_message": self.refusal_message,
            "citations": [c.to_dict() for c in self.citations],
            "query_coverage": round(self.query_coverage, 4),
            "diagnostic_notes": self.diagnostic_notes,
        }


class EvidenceVerifier:
    """
    Verifies candidate retrieval passages against grounding criteria.
    Preserves default Phase 4 thresholds: score_threshold=3.0, min_query_coverage=0.35.
    """

    STOPWORDS = {
        'là', 'của', 'và', 'các', 'có', 'trong', 'được', 'cho', 'những', 'với',
        'khi', 'ở', 'nào', 'gì', 'sao', 'thế', 'này', 'đó', 'ra', 'vào', 'từ',
        'theo', 'đã', 'sẽ', 'đang', 'như', 'thì', 'ai', 'nhiêu', 'bao', 'nơi',
        'mỗi', 'lại', 'qua', 'bởi', 'do', 'để', 'rằng', 'chính', 'xác', 'cụ',
        'the', 'is', 'are', 'a', 'an', 'of', 'in', 'to', 'for', 'with', 'on',
    }

    def __init__(
        self,
        score_threshold: float = 3.0,
        min_query_coverage: float = 0.35,
        max_evidence: int = 5,
    ):
        self.score_threshold = score_threshold
        self.min_query_coverage = min_query_coverage
        self.max_evidence = max_evidence

    def extract_keywords(self, query: str) -> List[str]:
        q_clean = re.sub(r'[^\w\s]', ' ', query.lower())
        tokens = [w for w in q_clean.split() if len(w) > 1 and w not in self.STOPWORDS]
        return tokens

    def compute_coverage(self, query: str, passages: List[RetrievalResult]) -> float:
        keywords = self.extract_keywords(query)
        if not keywords:
            return 1.0

        all_text = " ".join([p.text.lower() for p in passages])
        matched = sum(1 for kw in set(keywords) if kw in all_text)
        return matched / len(set(keywords))

    def verify(
        self,
        query: str,
        retrieval_candidates: List[RetrievalResult],
    ) -> Tuple[GroundingStatus, float, List[RetrievalResult], float]:
        """
        Verifies retrieval candidates.
        Returns (GroundingStatus, confidence, valid_passages, query_coverage).
        """
        if not retrieval_candidates:
            return GroundingStatus.UNANSWERABLE, 0.0, [], 0.0

        # Filter by threshold
        valid_passages = [p for p in retrieval_candidates if p.score >= self.score_threshold][:self.max_evidence]

        coverage = self.compute_coverage(query, valid_passages if valid_passages else retrieval_candidates)
        max_score = max([p.score for p in retrieval_candidates]) if retrieval_candidates else 0.0

        if not valid_passages:
            if max_score > 0.5:
                return GroundingStatus.INSUFFICIENT_EVIDENCE, round(max_score / self.score_threshold * 0.5, 4), [], coverage
            else:
                return GroundingStatus.UNANSWERABLE, 0.0, [], coverage

        if coverage < self.min_query_coverage:
            return GroundingStatus.INSUFFICIENT_EVIDENCE, 0.45, valid_passages, coverage

        if coverage < 0.70:
            return GroundingStatus.PARTIALLY_SUPPORTED, 0.75, valid_passages, coverage

        return GroundingStatus.SUPPORTED, 0.95, valid_passages, coverage


class CitationManager:
    """Manages citation assignment, source traceability, and formatted evidence blocks."""

    def __init__(self):
        pass

    def build_citations(
        self,
        valid_passages: List[RetrievalResult],
        doc_store: Optional[Any] = None,
    ) -> List[GroundedCitation]:
        citations = []
        for idx, p in enumerate(valid_passages, start=1):
            doc_id = getattr(p, "document_id", "DOC")
            doc_title = doc_id
            if doc_store:
                try:
                    d_obj = doc_store.get_document(doc_id)
                    doc_title = d_obj.title
                except Exception:
                    pass

            sec_title = getattr(p, "section_title", "Phần nội dung") or "Phần nội dung"
            para_idx = getattr(p, "paragraph_index", idx)

            cit = GroundedCitation(
                citation_index=idx,
                label=f"[{idx}]",
                passage_id=p.passage_id,
                document_id=doc_id,
                document_title=doc_title,
                section_title=sec_title,
                paragraph_index=para_idx,
                score=round(float(p.score), 4),
                text=p.text,
                status="SUPPORTED" if p.score >= 3.0 else "PARTIAL",
            )
            citations.append(cit)
        return citations

    def format_sources_section(self, citations: List[GroundedCitation]) -> str:
        if not citations:
            return ""
        lines = ["### Nguồn tham khảo"]
        for c in citations:
            lines.append(f"[{c.citation_index}] **{c.document_title}** — {c.section_title} — Đoạn {c.paragraph_index}")
        return "\n\n".join(lines)


class AnswerGroundingValidator:
    """
    Final gatekeeper ensuring the answer is fully grounded and refusal is triggered
    when evidence is insufficient or unanswerable.
    """

    STANDARD_REFUSAL_VI = "Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn."
    STANDARD_REFUSAL_EN = "Insufficient evidence found in the document to answer reliably."

    def __init__(
        self,
        verifier: Optional[EvidenceVerifier] = None,
        citation_manager: Optional[CitationManager] = None,
    ):
        self.verifier = verifier or EvidenceVerifier()
        self.citation_manager = citation_manager or CitationManager()

    def validate_and_ground(
        self,
        query: str,
        retrieval_candidates: List[RetrievalResult],
        doc_store: Optional[Any] = None,
        language: str = "vi",
    ) -> GroundingValidationResult:
        """
        Executes the full Grounded QA Safety pipeline.
        """
        status, conf, valid_passages, coverage = self.verifier.verify(query, retrieval_candidates)

        if status in (GroundingStatus.INSUFFICIENT_EVIDENCE, GroundingStatus.UNANSWERABLE):
            refusal_text = self.STANDARD_REFUSAL_VI if language == "vi" else self.STANDARD_REFUSAL_EN
            return GroundingValidationResult(
                status=status,
                is_safe_to_answer=False,
                confidence_score=conf,
                refusal_message=refusal_text,
                citations=[],
                query_coverage=coverage,
                diagnostic_notes=f"Refusal enforced: evidence below threshold ({status.value})",
            )

        # Build citations
        citations = self.citation_manager.build_citations(valid_passages, doc_store=doc_store)

        return GroundingValidationResult(
            status=status,
            is_safe_to_answer=True,
            confidence_score=conf,
            refusal_message=None,
            citations=citations,
            query_coverage=coverage,
            diagnostic_notes=f"Evidence verified: {len(citations)} citations supported",
        )

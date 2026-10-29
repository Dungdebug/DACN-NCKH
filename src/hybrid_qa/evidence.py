"""
Evidence Selector & Citation System.

Selects supporting evidence passages from retrieval results,
assigns citations to answer claims, and provides full traceability:
    answer → cited passage → document → version.

Design:
- Evidence is selected by score threshold + top-k.
- Citation objects carry full provenance (passage_id, document_id, version).
- No claim of "faithful" without external judge evaluation.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from src.hybrid_qa.retriever import RetrievalResult


@dataclass
class Citation:
    """A citation linking a claim to a source passage."""
    passage_id: str      # e.g. "DOC001::P003"
    document_id: str     # e.g. "DOC001"
    text: str            # The cited passage text
    score: float         # Retrieval confidence score
    section_title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidencePackage:
    """Complete evidence package for answer generation."""
    query: str
    evidence_passages: List[Citation]
    has_sufficient_evidence: bool
    max_evidence_score: float
    total_passages_searched: int
    query_coverage: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_passages"] = [c.to_dict() if isinstance(c, Citation) else c
                                  for c in self.evidence_passages]
        return d


class EvidenceSelector:
    """
    Selects evidence passages from retrieval results using configurable policy.
    
    Policy parameters:
    - score_threshold: minimum BM25/similarity score to consider as evidence.
    - max_evidence: maximum number of evidence passages to include.
    - min_evidence: if fewer than this many passages pass threshold,
                    mark as insufficient evidence.
    - min_query_coverage: minimum fraction of query content words covered by evidence.
    """

    STOPWORDS = {
        'là', 'của', 'và', 'các', 'có', 'trong', 'được', 'cho', 'những', 'với',
        'khi', 'ở', 'nào', 'gì', 'sao', 'thế', 'này', 'đó', 'ra', 'vào', 'từ',
        'theo', 'đã', 'sẽ', 'đang', 'như', 'thì', 'ai', 'nhiêu', 'bao', 'nơi',
        'mỗi', 'lại', 'qua', 'bởi', 'do', 'để', 'rằng', 'chính', 'xác', 'cụ',
        'the', 'is', 'are', 'a', 'an', 'of', 'in', 'to', 'for', 'with', 'on',
        'at', 'by', 'from', 'and', 'or', 'what', 'which', 'how', 'when', 'where', 'who'
    }

    def __init__(
        self,
        score_threshold: float = 1.0,
        max_evidence: int = 5,
        min_evidence: int = 1,
        min_query_coverage: float = 0.0,
        dynamic_top_k: bool = False,
        dominance_ratio: float = 0.55,
    ):
        self.score_threshold = score_threshold
        self.max_evidence = max_evidence
        self.min_evidence = min_evidence
        self.min_query_coverage = min_query_coverage
        self.dynamic_top_k = dynamic_top_k
        self.dominance_ratio = dominance_ratio

    def _tokenize_content(self, text: str) -> List[str]:
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w for w in text.split() if len(w) > 1 and w not in self.STOPWORDS]

    def _apply_dynamic_top_k(
        self,
        query_tokens: set,
        candidates: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Dynamic Top-K Early-Stopping & Marginal Coverage Pruning:
        Instead of blindly feeding all k passages into the prompt, retains the top passage
        and only includes subsequent passages if they:
        1) Contribute new query token coverage (marginal gain > 0), OR
        2) Have a competitive BM25 score (>= dominance_ratio * top_score) and non-zero query overlap.
        Stops early once cumulative query coverage >= 0.85 and subsequent scores drop sharply.
        """
        if len(candidates) <= 1 or not query_tokens:
            return candidates

        top_score = candidates[0].score
        selected = [candidates[0]]
        covered_tokens = set(self._tokenize_content(candidates[0].text)) & query_tokens

        for cand in candidates[1:]:
            cand_tokens = set(self._tokenize_content(cand.text)) & query_tokens
            new_tokens = cand_tokens - covered_tokens
            current_cov = len(covered_tokens) / max(1, len(query_tokens))
            score_ratio = (cand.score / top_score) if top_score > 0 else 0.0

            # If we already covered >= 85% of query terms and score drops below dominance ratio, stop early
            if current_cov >= 0.85 and score_ratio < self.dominance_ratio and not new_tokens:
                break

            # Keep passage if it brings new query terms or has strong relevance score with query overlap
            if new_tokens or (score_ratio >= self.dominance_ratio and len(cand_tokens) > 0):
                selected.append(cand)
                covered_tokens.update(cand_tokens)

        return selected

    def select_evidence(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        dynamic_top_k: Optional[bool] = None,
    ) -> EvidencePackage:
        """
        Select evidence from retrieval results.
        Returns an EvidencePackage with selected citations, coverage, and sufficiency flag.
        """
        # Filter by threshold
        qualified = [r for r in retrieval_results if r.score >= self.score_threshold]

        # Take top-k
        qualified = qualified[:self.max_evidence]

        q_tokens = set(self._tokenize_content(query))
        use_dynamic = self.dynamic_top_k if dynamic_top_k is None else dynamic_top_k
        pre_prune_count = len(qualified)
        if use_dynamic and len(qualified) > 1:
            qualified = self._apply_dynamic_top_k(q_tokens, qualified)

        # Build citations
        citations = []
        for r in qualified:
            citations.append(Citation(
                passage_id=r.passage_id,
                document_id=r.document_id,
                text=r.text,
                score=r.score,
                section_title=r.section_title,
                metadata=r.metadata,
            ))

        # Compute query term coverage
        if q_tokens and citations:
            matched = set()
            for c in citations:
                p_tokens = set(self._tokenize_content(c.text))
                matched.update(q_tokens & p_tokens)
            coverage = len(matched) / len(q_tokens)
        elif citations:
            coverage = 1.0
        else:
            coverage = 0.0

        has_sufficient = len(citations) >= self.min_evidence and (coverage >= self.min_query_coverage)
        max_score = max((c.score for c in citations), default=0.0)

        return EvidencePackage(
            query=query,
            evidence_passages=citations,
            has_sufficient_evidence=has_sufficient,
            max_evidence_score=max_score,
            total_passages_searched=len(retrieval_results),
            query_coverage=coverage,
            metadata={
                "dynamic_top_k_applied": bool(use_dynamic),
                "candidates_above_threshold": pre_prune_count,
                "selected_count": len(citations),
            },
        )


class CitationChecker:
    """
    Verifies that citations in an answer are traceable to actual passages.
    
    Does NOT make faithfulness claims — only verifies structural traceability:
    - cited passage_id exists in the evidence package
    - cited passage_id resolves to a real document passage
    """

    @staticmethod
    def verify_citations(
        cited_passage_ids: List[str],
        evidence: EvidencePackage,
        all_passage_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Verify that cited passage IDs are valid.
        
        Returns:
            {
                "all_valid": bool,
                "verified": [passage_id, ...],
                "invalid": [passage_id, ...],
                "in_evidence": [passage_id, ...],
                "in_corpus_not_evidence": [passage_id, ...],
            }
        """
        evidence_ids = {c.passage_id for c in evidence.evidence_passages}
        corpus_ids = set(all_passage_ids) if all_passage_ids else evidence_ids

        verified = []
        invalid = []
        in_evidence = []
        in_corpus_not_evidence = []

        for pid in cited_passage_ids:
            if pid in evidence_ids:
                verified.append(pid)
                in_evidence.append(pid)
            elif pid in corpus_ids:
                verified.append(pid)
                in_corpus_not_evidence.append(pid)
            else:
                invalid.append(pid)

        return {
            "all_valid": len(invalid) == 0,
            "verified": verified,
            "invalid": invalid,
            "in_evidence": in_evidence,
            "in_corpus_not_evidence": in_corpus_not_evidence,
        }

    @staticmethod
    def trace_citation(
        passage_id: str,
        document_store: Any,
        expected_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Trace a citation passage_id through the complete structural chain:
            answer → citation_id → passage_id → paragraph_id → section_id → document_id → document_version
        """
        if not document_store:
            return {
                "valid": False,
                "citation_id": passage_id,
                "passage_id": passage_id,
                "paragraph_id": None,
                "section_id": None,
                "section_title": "",
                "document_id": None,
                "document_version": None,
                "version_matched": False,
                "error": "No document store provided",
            }

        if "::" not in passage_id:
            return {
                "valid": False,
                "citation_id": passage_id,
                "passage_id": passage_id,
                "paragraph_id": None,
                "section_id": None,
                "section_title": "",
                "document_id": None,
                "document_version": None,
                "version_matched": False,
                "error": f"Invalid passage ID format: {passage_id}",
            }

        doc_id = passage_id.split("::")[0]
        try:
            doc = document_store.get_document(doc_id)
        except Exception:
            return {
                "valid": False,
                "citation_id": passage_id,
                "passage_id": passage_id,
                "paragraph_id": None,
                "section_id": None,
                "section_title": "",
                "document_id": doc_id,
                "document_version": None,
                "version_matched": False,
                "error": f"Document {doc_id} not found in store",
            }

        target_passage = None
        for p in doc.passages:
            if p.passage_id == passage_id:
                target_passage = p
                break

        if target_passage is None:
            return {
                "valid": False,
                "citation_id": passage_id,
                "passage_id": passage_id,
                "paragraph_id": None,
                "section_id": None,
                "section_title": "",
                "document_id": doc_id,
                "document_version": doc.version,
                "version_matched": False,
                "error": f"Passage {passage_id} not found in document {doc_id}",
            }

        metadata = target_passage.metadata or {}
        paragraph_id = metadata.get("paragraph_id", "PARA001")
        section_id = metadata.get("section_id", "SEC001")
        doc_version = metadata.get("document_version", doc.version)

        version_matched = True
        error = None
        if expected_version is not None and doc_version != expected_version:
            version_matched = False
            error = f"Version mismatch: expected v{expected_version}, but citation points to v{doc_version}"

        return {
            "valid": version_matched,
            "citation_id": passage_id,
            "passage_id": passage_id,
            "paragraph_id": paragraph_id,
            "section_id": section_id,
            "section_title": target_passage.section_title,
            "document_id": doc_id,
            "document_version": doc_version,
            "version_matched": version_matched,
            "error": error,
        }


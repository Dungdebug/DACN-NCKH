"""
Unit tests for Grounded QA Safety Layer (EvidenceVerifier, CitationManager, AnswerGroundingValidator).
"""

import pytest
from src.hybrid_qa.grounding_validator import (
    GroundingStatus,
    EvidenceVerifier,
    CitationManager,
    AnswerGroundingValidator,
)
from src.hybrid_qa.retriever import RetrievalResult


def test_grounding_validator_supported():
    validator = AnswerGroundingValidator()
    candidates = [
        RetrievalResult(passage_id="P1", document_id="D1", text="Kiến trúc SA-CMS lưu trữ thông tin", score=4.5),
        RetrievalResult(passage_id="P2", document_id="D1", text="Cấu trúc đa quy mô thời gian", score=3.8),
    ]

    res = validator.validate_and_ground("Kiến trúc SA-CMS lưu trữ thông tin", candidates)
    assert res.is_safe_to_answer is True
    assert res.status in (GroundingStatus.SUPPORTED, GroundingStatus.PARTIALLY_SUPPORTED)
    assert len(res.citations) == 2
    assert res.refusal_message is None


def test_grounding_validator_refusal_below_threshold():
    validator = AnswerGroundingValidator()
    candidates = [
        RetrievalResult(passage_id="P1", document_id="D1", text="Ngẫu nhiên", score=1.8),
    ]

    res = validator.validate_and_ground("Câu hỏi bất kỳ", candidates)
    assert res.is_safe_to_answer is False
    assert res.status == GroundingStatus.INSUFFICIENT_EVIDENCE
    assert res.refusal_message == "Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn."
    assert len(res.citations) == 0


def test_grounding_validator_refusal_unanswerable():
    validator = AnswerGroundingValidator()
    res = validator.validate_and_ground("Thủ đô nước Pháp", [])
    assert res.is_safe_to_answer is False
    assert res.status == GroundingStatus.UNANSWERABLE
    assert res.refusal_message == "Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn."


def test_citation_manager_formatting():
    manager = CitationManager()
    valid_passages = [
        RetrievalResult(passage_id="P01", document_id="DOC01", text="Nội dung trích dẫn", score=4.2),
    ]
    citations = manager.build_citations(valid_passages)
    assert len(citations) == 1
    assert citations[0].label == "[1]"

    formatted = manager.format_sources_section(citations)
    assert "### Nguồn tham khảo" in formatted
    assert "[1]" in formatted

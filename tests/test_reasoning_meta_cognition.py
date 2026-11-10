"""
Unit Tests for MetaCognitiveEvaluator & Epistemic Calibration (Ticket 4.2).
Validates:
1. Segregation of verified facts vs tentative/hedged claims.
2. Calibrated confidence tiers (VERY_HIGH, HIGH, MODERATE, LOW).
3. Explicit knowledge gap identification for unmentioned query aspects.
4. Refusal handling on missing citations.
5. Report dictionary serialization.
"""

import pytest
from src.reasoning.meta_cognition import (
    MetaCognitiveEvaluator,
    MetaCognitiveReport,
)


@pytest.fixture
def sample_citations():
    return [
        {
            "passage_id": "doc1::p1",
            "document_id": "de_cuong_doc",
            "citation_label": "[1]",
            "text": "Mô hình SA-CMS đạt tỷ lệ giảm 87,5% chi phí token trên GPU RTX 3050. Theo lý thuyết, phương pháp có thể mở rộng lên quy mô 100 tài liệu.",
        }
    ]


def test_evaluate_response_verified_vs_tentative(sample_citations):
    evaluator = MetaCognitiveEvaluator()
    query = "Hiệu năng của SA-CMS như thế nào?"
    answer = (
        "Mô hình SA-CMS đạt tỷ lệ giảm 87,5% chi phí token trên GPU RTX 3050 [1]. "
        "Theo lý thuyết, phương pháp có thể mở rộng lên quy mô 100 tài liệu [1]."
    )

    report = evaluator.evaluate_response(query, answer, sample_citations, raw_confidence=0.85)

    assert isinstance(report, MetaCognitiveReport)
    assert report.is_refusal is False
    assert len(report.verified_facts) >= 1
    assert len(report.uncertain_claims) >= 1

    # First sentence is verified
    assert "87,5%" in report.verified_facts[0]["statement"]
    assert report.verified_facts[0]["epistemic_tag"] == "[Chắc chắn]"

    # Second sentence has modal hedge "có thể"
    assert "có thể" in report.uncertain_claims[0]["statement"].lower()
    assert "Giả thuyết" in report.uncertain_claims[0]["epistemic_tag"]


def test_calibrated_confidence_tiers(sample_citations):
    evaluator = MetaCognitiveEvaluator()
    query = "Tỷ lệ giảm token của SA-CMS là bao nhiêu?"
    pure_verified = "Mô hình SA-CMS đạt tỷ lệ giảm 87,5% chi phí token trên GPU RTX 3050 [1]."

    report_high = evaluator.evaluate_response(query, pure_verified, sample_citations, raw_confidence=0.95)
    assert report_high.confidence_tier in ("VERY_HIGH", "HIGH")
    assert report_high.overall_confidence >= 0.70


def test_knowledge_gaps_detection(sample_citations):
    evaluator = MetaCognitiveEvaluator()
    # Query asks about quantum computing and supercomputer, which are absent in evidence
    query = "SA-CMS chạy trên siêu máy tính lượng tử quantum supercomputer như thế nào?"
    answer = "Mô hình SA-CMS đạt tỷ lệ giảm 87,5% chi phí token trên GPU RTX 3050 [1]."

    report = evaluator.evaluate_response(query, answer, sample_citations, raw_confidence=0.7)
    assert len(report.knowledge_gaps) >= 1
    gap = report.knowledge_gaps[0].lower()
    assert "chưa công bố" in gap or "khoảng trống" in gap or "quantum" in gap or "lượng tử" in gap


def test_refusal_on_empty_citations():
    evaluator = MetaCognitiveEvaluator()
    report = evaluator.evaluate_response("Câu hỏi bất kỳ?", "", [])
    assert report.is_refusal is True
    assert report.overall_confidence == 0.0
    assert report.confidence_tier == "LOW"


def test_report_serialization(sample_citations):
    evaluator = MetaCognitiveEvaluator()
    query = "SA-CMS là gì?"
    answer = "Mô hình SA-CMS đạt tỷ lệ giảm 87,5% chi phí token [1]."

    report = evaluator.evaluate_response(query, answer, sample_citations)
    d = report.to_dict()

    assert isinstance(d, dict)
    assert "overall_confidence" in d
    assert "confidence_tier" in d
    assert "verified_facts" in d
    assert "uncertain_claims" in d
    assert "knowledge_gaps" in d
    assert "formatted_scholarly_response" in d

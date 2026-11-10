"""
Unit tests for Curiosity-Driven Active Ingestion (Ticket 7.5).
Verifies calculation of curiosity scores, hard example mining,
curriculum curation, and gradient prioritization for SA-CMS Levels 2 & 3.
"""

import pytest
from dataclasses import dataclass
from src.self_study.curiosity_trainer import (
    CuriosityDrivenIngestionEngine,
    CuriosityScore,
    CuriosityIngestionReport,
)


@dataclass
class MockQAPair:
    question: str
    answer: str
    passage_id: str
    document_id: str
    rule_name: str
    confidence: float


@pytest.fixture
def sample_qa_pairs():
    return [
        MockQAPair(
            question="SA-CMS là gì?",
            answer="SA-CMS là hệ thống bộ nhớ liên tục căn theo cấu trúc văn bản.",
            passage_id="p1",
            document_id="DOC_1",
            rule_name="rule_definition",
            confidence=0.95,
        ),
        MockQAPair(
            question="Kiến trúc SA-CMS giảm bao nhiêu phần trăm chi phí token so với RAG và trong điều kiện nào?",
            answer="SA-CMS giảm 87,5% chi phí token trong điều kiện văn bản dài trên 2048 tokens nhưng tốn 15% thời gian nạp ban đầu.",
            passage_id="p2",
            document_id="DOC_1",
            rule_name="rule_metric",
            confidence=0.80,
        ),
        MockQAPair(
            question="Tại sao Self-Attention gặp vấn đề chi phí bậc hai O(N^2)?",
            answer="Nguyên nhân cốt lõi do cơ chế Self-Attention phải tính toán ma trận tương quan giữa mọi cặp token.",
            passage_id="p3",
            document_id="DOC_1",
            rule_name="rule_cause",
            confidence=0.88,
        ),
        MockQAPair(
            question="Mô hình SmolLM2-135M có mấy tham số?",
            answer="Mô hình có 135 triệu tham số.",
            passage_id="p4",
            document_id="DOC_1",
            rule_name="rule_number",
            confidence=0.90,
        ),
    ]


def test_compute_curiosity_score(sample_qa_pairs):
    engine = CuriosityDrivenIngestionEngine()
    scored = engine.score_curiosity(sample_qa_pairs)
    
    assert len(scored) == len(sample_qa_pairs)
    # The multi-clause complex question with numbers and conditions should have higher curiosity than simple definition
    pair_metric = next(s for s in scored if "87,5%" in s.pair.answer)
    pair_def = next(s for s in scored if "SA-CMS là gì" in s.pair.question)
    assert pair_metric.curiosity_score > pair_def.curiosity_score


def test_curate_curriculum_budgeting(sample_qa_pairs):
    engine = CuriosityDrivenIngestionEngine()
    report = engine.curate_curriculum(sample_qa_pairs, max_budget=3)
    
    assert len(report.curated_pairs) == 3
    assert report.high_curiosity_count >= 1
    assert report.mean_curiosity_score > 0.0
    # Top hardest QA should be included in budget
    questions = [p.question for p in report.curated_pairs]
    assert any("chi phí token" in q or "Self-Attention" in q for q in questions)


def test_empty_pairs_handling():
    engine = CuriosityDrivenIngestionEngine()
    report = engine.curate_curriculum([], max_budget=5)
    
    assert report.total_evaluated == 0
    assert len(report.curated_pairs) == 0
    assert report.mean_curiosity_score == 0.0


def test_prioritized_training_weights(sample_qa_pairs):
    engine = CuriosityDrivenIngestionEngine()
    weights = engine.get_gradient_step_weights(sample_qa_pairs)
    
    assert len(weights) == len(sample_qa_pairs)
    # Harder questions should receive higher weight (e.g. >= 1.5) compared to basic definition
    assert max(weights) > min(weights)

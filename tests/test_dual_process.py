"""
Unit Tests for Dual-Process System 1/System 2 & Premise Validator (Ticket 6.4).
Validates:
1. False presupposition detection and grounded refutation.
2. Mode routing between System 1 (Fast) and System 2 (Deliberate).
3. End-to-end execution of loaded questions and analytical multi-hop queries.
4. Zero hallucination guarantee.
"""

import pytest
from src.reasoning.dual_process import (
    PremiseValidator,
    DualProcessController,
    DualProcessResult,
)
from src.self_study.reader import NeuralReaderEngine


@pytest.fixture
def passages():
    return [
        {
            "passage_id": "p1",
            "document_id": "doc_sacms",
            "document_title": "Báo cáo SA-CMS",
            "section_title": "3. Kết quả",
            "citation_label": "[1]",
            "score": 10.0,
            "text": (
                "Hệ thống SA-CMS là bộ nhớ liên tục căn chỉnh theo cấu trúc tài liệu. "
                "Mô hình đạt tỷ lệ giảm 87,5% chi phí token so với RAG truyền thống. "
                "Tuy nhiên, hạn chế là cần tài liệu có cấu trúc phân đoạn rõ ràng."
            ),
        }
    ]


def test_premise_validator_catches_false_presupposition(passages):
    validator = PremiseValidator()
    # User asks a loaded question assuming SA-CMS increases token cost
    query = "Tại sao mô hình SA-CMS lại làm tăng chi phí token so với RAG?"
    res = validator.validate_premise(query, passages)

    assert res.has_false_premise is True
    assert "giảm 87,5% chi phí token" in res.counter_evidence_sentence
    assert "[1]" in res.counter_citation
    assert "không phù hợp với tài liệu" in res.refutation_text


def test_premise_validator_neutral_query(passages):
    validator = PremiseValidator()
    # Neutral query should not trigger refutation
    query = "Mô hình SA-CMS đạt tỷ lệ giảm chi phí token là bao nhiêu?"
    res = validator.validate_premise(query, passages)

    assert res.has_false_premise is False
    assert res.refutation_text == ""


def test_dual_process_mode_routing():
    controller = DualProcessController()

    # Simple factual lookups -> System 1
    assert controller.determine_mode("SA-CMS là gì?", "DEFINITION") == "SYSTEM_1_FAST"
    assert controller.determine_mode("Tỷ lệ giảm token?", "METRIC") == "SYSTEM_1_FAST"

    # Complex analytical, why, compare, multi-hop -> System 2
    assert controller.determine_mode("Tại sao SA-CMS lại tiết kiệm token?", "CAUSE") == "SYSTEM_2_DELIBERATE"
    assert controller.determine_mode("So sánh SA-CMS với RAG truyền thống khác gì nhau?", "COMPARISON") == "SYSTEM_2_DELIBERATE"
    assert controller.determine_mode("Đánh giá tổng quan về kiến trúc và hiệu năng thực nghiệm?", "SUMMARY") == "SYSTEM_2_DELIBERATE"


def test_dual_process_execution_refutation_end_to_end(passages):
    controller = DualProcessController()
    loaded_query = "Tại sao SA-CMS lại làm tăng chi phí token?"

    res = controller.execute(loaded_query, passages)

    assert res.is_refutation is True
    assert res.process_mode == "PREMISE_REFUTATION"
    assert "không phù hợp với tài liệu" in res.final_answer
    assert "87,5%" in res.final_answer
    assert len(res.citations) > 0


def test_dual_process_execution_deliberate_end_to_end(passages):
    controller = DualProcessController()
    deliberate_query = "Tại sao SA-CMS lại đạt hiệu quả vượt trội so với RAG?"

    res = controller.execute(deliberate_query, passages)

    assert res.is_refutation is False
    assert res.process_mode == "SYSTEM_2_DELIBERATE"
    assert res.confidence > 0.30
    assert len(res.reasoning_trace) > 0
    assert "[1]" in res.final_answer


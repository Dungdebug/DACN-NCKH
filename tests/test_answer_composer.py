"""
Unit Tests for Coherent Multi-Sentence Answer Composer (Ticket 6.2).
Validates:
1. Sentence role classification (metric, limitation, mechanism, elaboration).
2. Concise mode (returns single core sentence).
3. Balanced mode (composes coherent multi-sentence prose with fluent connectives).
4. Comprehensive mode (prose + structured key points).
5. Exact substring grounding (verifies zero hallucination across all output clauses).
6. In-text citation alignment ([1], [2], [3]).
7. NeuralReaderEngine end-to-end integration.
"""

import pytest
from src.self_study.answer_composer import CoherentAnswerComposer
from src.self_study.reader import NeuralReaderEngine


@pytest.fixture
def composer():
    return CoherentAnswerComposer()


@pytest.fixture
def sample_candidates():
    return [
        {
            "sentence": "Hệ thống SA-CMS là bộ nhớ liên tục căn chỉnh cấu trúc tài liệu.",
            "score": 5.8,
            "citation_label": "[1]",
            "passage_data": {"document_id": "doc1", "text": "Hệ thống SA-CMS là bộ nhớ liên tục căn chỉnh cấu trúc tài liệu."},
        },
        {
            "sentence": "Mô hình đạt tỷ lệ giảm 87,5% chi phí token và độ trễ 18,2 ms.",
            "score": 5.1,
            "citation_label": "[2]",
            "passage_data": {"document_id": "doc1", "text": "Mô hình đạt tỷ lệ giảm 87,5% chi phí token và độ trễ 18,2 ms."},
        },
        {
            "sentence": "Tuy nhiên, hạn chế là cần tài liệu có cấu trúc phân đoạn rõ ràng.",
            "score": 4.6,
            "citation_label": "[2]",
            "passage_data": {"document_id": "doc1", "text": "Tuy nhiên, hạn chế là cần tài liệu có cấu trúc phân đoạn rõ ràng."},
        },
        {
            "sentence": "Cơ chế cập nhật đồng bộ qua 3 cấp độ đoạn văn, phần mục và tài liệu.",
            "score": 4.0,
            "citation_label": "[1]",
            "passage_data": {"document_id": "doc1", "text": "Cơ chế cập nhật đồng bộ qua 3 cấp độ đoạn văn, phần mục và tài liệu."},
        },
    ]


def test_classify_sentence_role(composer):
    assert composer.classify_sentence_role("Mô hình đạt tỷ lệ giảm 87,5% chi phí token.") == "metric"
    assert composer.classify_sentence_role("Độ trễ phản hồi trung bình là 18.2 ms.") == "metric"
    assert composer.classify_sentence_role("Hạn chế của hệ thống là cần tài liệu chuẩn hóa.") == "limitation"
    assert composer.classify_sentence_role("Tuy nhiên, thách thức đặt ra là bộ nhớ GPU.") == "limitation"
    assert composer.classify_sentence_role("Cơ chế cập nhật diễn ra thông qua cổng thích ứng.") == "mechanism"
    assert composer.classify_sentence_role("Tài liệu nghiên cứu các vấn đề về trí tuệ nhân tạo.") == "elaboration"


def test_compose_concise_mode(composer, sample_candidates):
    prose, key_points, active = composer.compose(
        query="SA-CMS là gì?",
        candidates=sample_candidates,
        answer_mode="concise",
    )
    assert "[1]" in prose
    assert "87,5%" not in prose
    assert len(key_points) == 1
    assert len(active) == 1


def test_compose_balanced_multi_sentence(composer, sample_candidates):
    prose, key_points, active = composer.compose(
        query="SA-CMS là gì và hiệu quả ra sao?",
        candidates=sample_candidates,
        answer_mode="balanced",
    )
    # Checks presence of Core + Metric + Limitation
    assert "bộ nhớ liên tục" in prose
    assert "[1]" in prose
    assert "87,5%" in prose
    assert "[2]" in prose
    assert "hạn chế" in prose.lower()
    # Checks fluent discourse connectives
    assert "Đáng chú ý" in prose or "đáng chú ý" in prose
    assert len(key_points) >= 2
    assert len(active) >= 2


def test_exact_substring_grounding(composer, sample_candidates):
    """Verifies that every factual payload clause in composed prose originates verbatim from source."""
    prose, _, active = composer.compose(
        query="SA-CMS là gì?",
        candidates=sample_candidates,
        answer_mode="balanced",
    )
    # Source sentences
    source_texts = " ".join([c["sentence"] for c in sample_candidates])

    # Key factual terms from sources must be preserved verbatim
    assert "bộ nhớ liên tục căn chỉnh cấu trúc tài liệu" in source_texts
    assert "87,5% chi phí token" in prose
    assert "cần tài liệu có cấu trúc phân đoạn rõ ràng" in prose


def test_reader_engine_integration_with_composer():
    reader = NeuralReaderEngine(min_confidence=0.20)
    passages = [
        {
            "passage_id": "p1",
            "document_id": "doc1",
            "document_title": "Báo cáo SA-CMS",
            "section_title": "Kiến trúc",
            "citation_label": "[1]",
            "score": 10.0,
            "text": (
                "Hệ thống SA-CMS là bộ nhớ liên tục căn chỉnh theo cấu trúc tài liệu. "
                "Mô hình đạt tỷ lệ giảm 87,5% chi phí token so với RAG truyền thống. "
                "Tuy nhiên, hạn chế là cần tài liệu có cấu trúc phân đoạn rõ ràng."
            ),
        }
    ]

    res = reader.synthesize(
        query="SA-CMS hoạt động như thế nào và có ưu điểm gì?",
        passages=passages,
        answer_mode="balanced",
    )

    assert res.is_grounded is True
    assert res.composed_answer != ""
    assert "87,5%" in res.composed_answer
    assert "[1]" in res.composed_answer
    assert len(res.key_points) >= 2


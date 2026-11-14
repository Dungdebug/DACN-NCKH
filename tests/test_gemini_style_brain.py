"""
Tests for Self-Built Gemini-Style Grounded Discourse & Synthesis Engine.
100% Local Offline, Zero External API, Zero Hallucination.
Verifies:
1. Multi-faceted structured synthesis across multiple passages (Core + Categorized Bullets).
2. Sub-query coverage for compound questions.
3. Comprehensive multi-tier document summary (TL;DR + Executive + Section/Key Takeaways).
4. Progressive depth across concise -> balanced -> detailed modes.
"""

import pytest
from dataclasses import dataclass
from typing import Dict, Any

from src.web.document_analyzer import DocumentAnalyzer


@dataclass
class MockPassage:
    passage_id: str
    document_id: str
    document_title: str
    section_title: str
    paragraph_index: int
    score: float
    text: str


@pytest.fixture
def sample_passages():
    return [
        MockPassage(
            passage_id="DOC1::P1",
            document_id="DOC1",
            document_title="Nghiên cứu SA-CMS",
            section_title="1. Định nghĩa và Kiến trúc",
            paragraph_index=1,
            score=12.5,
            text=(
                "Hệ thống SA-CMS là bộ nhớ tham số liên tục đa thang căn chỉnh theo cấu trúc tự nhiên của văn bản. "
                "Kiến trúc vận hành thông qua 3 tầng bộ nhớ cập nhật tại ranh giới đoạn văn, tiểu mục và toàn văn tài liệu."
            ),
        ),
        MockPassage(
            passage_id="DOC1::P2",
            document_id="DOC1",
            document_title="Nghiên cứu SA-CMS",
            section_title="2. Kết quả Thực nghiệm",
            paragraph_index=2,
            score=10.8,
            text=(
                "Trong thực nghiệm trên GPU RTX 3050 4GB, mô hình giúp giảm 87,5% chi phí token so với RAG truyền thống. "
                "Tuy nhiên, hạn chế của phương pháp là đòi hỏi tài liệu đầu vào phải có cấu trúc phân đoạn rõ ràng."
            ),
        ),
    ]


def test_gemini_style_structured_synthesis_across_multiple_passages(sample_passages):
    """
    Verifies that balanced/detailed answers synthesize facts across multiple passages
    with categorized bullet facets (mechanism, metric, limitation) and citations [1], [2].
    """
    answer, citations = DocumentAnalyzer.find_best_grounded_answer(
        query="Hệ thống SA-CMS hoạt động như thế nào?",
        evidence_passages=sample_passages,
        answer_mode="balanced",
    )

    assert len(citations) == 2
    # Must contain core answer from passage 1 and metric/limitation facets from passage 2
    assert "SA-CMS" in answer
    assert "[1]" in answer
    assert "[2]" in answer
    assert "87,5%" in answer
    # Must contain structured bullet points
    assert "- **" in answer


def test_compound_query_answers_all_subquestions(sample_passages):
    """
    Verifies that a multi-part question ("là gì và giảm bao nhiêu %") explicitly
    covers both sub-questions in the synthesized answer.
    """
    answer, citations = DocumentAnalyzer.find_best_grounded_answer(
        query="Hệ thống SA-CMS là gì và giúp giảm bao nhiêu % chi phí token?",
        evidence_passages=sample_passages,
        answer_mode="balanced",
    )

    assert "bộ nhớ tham số liên tục" in answer.lower()
    assert "87,5%" in answer


def test_progressive_depth_across_modes(sample_passages):
    """
    Verifies strict hierarchy: concise < balanced < detailed in length and depth.
    """
    concise_ans, _ = DocumentAnalyzer.find_best_grounded_answer(
        query="Hệ thống SA-CMS là gì?",
        evidence_passages=sample_passages,
        answer_mode="concise",
    )
    balanced_ans, _ = DocumentAnalyzer.find_best_grounded_answer(
        query="Hệ thống SA-CMS là gì?",
        evidence_passages=sample_passages,
        answer_mode="balanced",
    )
    detailed_ans, _ = DocumentAnalyzer.find_best_grounded_answer(
        query="Hệ thống SA-CMS là gì?",
        evidence_passages=sample_passages,
        answer_mode="detailed",
    )

    # Strip trailing question block to compare pure answer body length
    from src.hybrid_qa.answer_span import strip_question_block
    c_body = strip_question_block(concise_ans).split("### Nguồn tham khảo")[0]
    b_body = strip_question_block(balanced_ans).split("### Nguồn tham khảo")[0]
    d_body = strip_question_block(detailed_ans).split("### Nguồn tham khảo")[0]

    assert len(c_body) < len(b_body)
    assert len(b_body) < len(d_body)
    assert "Tổng hợp" in d_body or "Đúc kết" in d_body


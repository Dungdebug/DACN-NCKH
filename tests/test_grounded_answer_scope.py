"""
Unit & Integration Tests for Grounded Answer Scope & Followup Generation (Wave 4 Fixes).
Validates:
1. Scoped topic lookups (e.g. "topic 49", "cho tôi biết chủ đề 47") produce plain prose:
   - Zero robotic intent intros ("Dựa trên tài liệu..., nội dung được phân tích...").
   - Zero "📌 Phân tích đa chiều" blocks.
   - Zero leakage from adjacent topics (Topic 48 never leaks into Topic 49).
2. Heading query behavior:
   - When the best sentence is a heading, the answer core includes the heading plus the next 2-3 body sentences.
3. Relevance gate with alias+N:
   - "cho tôi biết chủ đề 47" matches "Topic 47" with relevance 1.0 (no false refusal).
4. generate_followup_questions:
   - Always returns exactly `count` questions when context is non-empty.
   - Zero SUMMARY trigger words ("trọng tâm", "điểm nổi bật", "nội dung chính", "tóm tắt", "tổng quan").
   - Doc-specific templates ("biện pháp", "trần hữu trang") are dropped when context lacks those keywords.
   - _clean_topic trims to <= 12 words and drops unbalanced "(".
   - Prefers clean section_title for Topic labels.
"""

import pytest
from dataclasses import dataclass
from typing import List

from src.hybrid_qa.answer_span import (
    _clean_topic,
    generate_followup_questions,
    relevance_score,
)
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
def topic_speak_passages():
    """Mock passages resembling Topic_speak.docx with distinct IELTS topics."""
    return [
        MockPassage(
            passage_id="P_TOPIC_48",
            document_id="DOC_SPEAK",
            document_title="Topic_speak.docx",
            section_title="Topic 48: Describe a bad service at a restaurant",
            paragraph_index=1,
            score=7.2,
            text=(
                "Topic 48: Describe a bad service at a restaurant you visited.\n"
                "You should say: when it happened, which restaurant it was, what went wrong, "
                "and explain why you were dissatisfied with the service. "
                "Although the problem was eventually resolved, I did not have a good experience at that restaurant."
            ),
        ),
        MockPassage(
            passage_id="P_TOPIC_49",
            document_id="DOC_SPEAK",
            document_title="Topic_speak.docx",
            section_title="Topic 49: Describe an outdoor sport you played for the first time",
            paragraph_index=2,
            score=9.5,
            text=(
                "Topic 49: Describe an outdoor sport you played for the first time.\n"
                "You should say: what sport it was, where you played it, who you played it with, "
                "and explain how you felt after playing it for the first time. "
                "I played basketball at a local court with my high school classmates. "
                "It was exhausting but exhilarating, and it helped me build teamwork skills."
            ),
        ),
    ]


def test_relevance_gate_alias_number_matching():
    """Verifies that 'cho tôi biết chủ đề 47' matches 'Topic 47' with rel = 1.0 (prevents false refusal)."""
    query = "cho tôi biết chủ đề 47"
    passage_sent = "Topic 47: Describe a success you had in your life."
    sec_title = "Topic 47: Describe a success"

    rel = relevance_score(query, passage_sent, extra=sec_title)
    assert rel == 1.0

    # Unrelated topic number must not get 1.0
    rel_diff = relevance_score("cho tôi biết chủ đề 47", "Topic 48: Bad service", extra="Topic 48")
    assert rel_diff < 1.0


def test_scoped_topic_answer_isolation_and_no_robotic_intro(topic_speak_passages):
    """
    Verifies:
    1. Answer to 'cho tôi biết topic 49' contains NO robotic intro ('Dựa trên tài liệu...').
    2. Answer contains NO '📌 Phân tích đa chiều' block.
    3. Content from Topic 48 ('restaurant', 'bad service') is NEVER included in Topic 49 answer.
    4. Since best_sentence is a heading, heading + next body sentences are returned together.
    """
    answer, citations = DocumentAnalyzer.find_best_grounded_answer(
        query="cho tôi biết topic 49",
        evidence_passages=topic_speak_passages,
        answer_mode="balanced",
    )

    body = answer.split("### Nguồn tham khảo")[0]

    # No robotic intros
    assert "Dựa trên tài liệu" not in body
    assert "nội dung được phân tích và tổng hợp như sau" not in body

    # No multi-faceted breakdown block
    assert "📌" not in body
    assert "Phân tích đa chiều" not in body

    # Zero leakage from Topic 48
    assert "restaurant" not in body.lower()
    assert "bad service" not in body.lower()

    # Topic 49 content preserved
    assert "Topic 49" in body
    assert "outdoor sport" in body.lower()
    assert "basketball" in body.lower() or "teamwork" in body.lower()

    # Citations contain only the relevant passage
    assert len(citations) == 1
    assert citations[0]["passage_id"] == "P_TOPIC_49"


def test_clean_topic_helper():
    """Verifies _clean_topic word cap and unbalanced parenthesis removal."""
    # Trims to <= 12 words
    long_raw = "Nội dung trọng tâm liên quan đến việc triển khai các phương pháp nghiên cứu khoa học công nghệ năm 2026 trong môi trường đại học"
    cleaned = _clean_topic(long_raw, max_words=12)
    assert len(cleaned.split()) <= 12
    assert "nội dung trọng tâm" not in cleaned.lower()

    # Drops unbalanced opening parenthesis
    unbalanced_open = "Topic 1: Describe a time (when you were ill"
    cl_open = _clean_topic(unbalanced_open)
    assert "(" not in cl_open
    assert "when you were ill" not in cl_open or cl_open.endswith("Topic 1: Describe a time")

    # Strips summary trigger words
    assert "trọng tâm" not in _clean_topic("Nội dung trọng tâm về SA-CMS").lower()
    assert "tổng quan" not in _clean_topic("Tổng quan về mô hình").lower()


def test_generate_followup_questions_guarantees_exact_count_and_no_summary_triggers():
    """
    Verifies:
    1. Returns exactly `count` questions when context is non-empty.
    2. Zero summary trigger words ('trọng tâm', 'điểm nổi bật', 'nội dung chính', 'tóm tắt', 'tổng quan').
    3. No 'biện pháp' question when context doesn't mention biện pháp/giải pháp.
    """
    context = (
        "Topic 49: Describe an outdoor sport you played for the first time.\n"
        "I played basketball at a local court with my high school classmates. "
        "It was exhausting but exhilarating, and it helped me build teamwork skills."
    )

    questions = generate_followup_questions(
        query="cho tôi biết topic 49",
        context=context,
        section_title="Topic 49: Outdoor sport",
        count=3,
    )

    # Exactly 3 questions
    assert len(questions) == 3

    # Zero summary trigger words
    for q in questions:
        q_low = q.lower()
        assert "trọng tâm" not in q_low
        assert "điểm nổi bật" not in q_low
        assert "nội dung chính" not in q_low
        assert "tóm tắt" not in q_low
        assert "tổng quan" not in q_low
        # No ungrounded 'biện pháp'
        assert "biện pháp" not in q_low
        # No doc-specific workplace questions for IELTS topic
        assert "đơn vị công tác" not in q_low
        assert "tác giả" not in q_low

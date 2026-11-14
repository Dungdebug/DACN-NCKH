"""
Unit and Integration Tests for Conversational Brain Enhancements & Bug Fixes:
1. Follow-up questions have NO hardcoded fake questions (only real topics/entities).
2. Numbered reference resolution ("câu 2", "trả lời câu 2") resolves to suggested questions.
3. Pasted question cleanup (strips '1. **[Category]**:').
4. Exact topic number boosting (Topic 1 vs Topic 50).
5. Semantic alias mapping ('chủ đề 47' -> 'Topic 47').
6. Non-refusal and structured generation on Document Summary.
"""

import pytest
from src.hybrid_qa.answer_span import generate_followup_questions, _clean_topic
from src.web.session_manager import SessionManager
from src.hybrid_qa.document_store import DocumentStore, DocumentRecord
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.retriever import BM25Retriever
from src.web.token_optimizer import expand_query_semantics, score_sentence_relevance


def test_followup_questions_no_hardcoded_measures_on_academic_text():
    academic_ctx = (
        "Cơ chế Bộ nhớ Đa quy mô (SA-CMS) hoạt động dựa trên các ma trận trọng số khả vi. "
        "Cổng hòa trộn g_t điều tiết tỷ lệ biểu diễn giữa tầng chú ý và bộ nhớ tham số."
    )
    questions = generate_followup_questions(
        query="cơ chế bộ nhớ là gì",
        context=academic_ctx,
        section_title="Kiến trúc SA-CMS",
        count=4,
    )
    # Must NOT have the fake hardcoded question about "biện pháp"
    assert not any("biện pháp chính" in q.lower() for q in questions)
    assert len(questions) >= 1
    # Clean topic must not be nested
    for q in questions:
        assert "nội dung trọng tâm liên quan đến 'nội dung trọng tâm" not in q.lower()


def test_followup_questions_extracts_real_topics_from_speaking_document():
    speaking_ctx = (
        "Topic 1: Describe an interesting video that you watched on social media.\n"
        "Topic 2: Describe an outdoor sport you would like to try.\n"
        "Topic 3: Describe a traditional celebration in your country."
    )
    questions = generate_followup_questions(
        query="cho tôi biết topic 1",
        context=speaking_ctx,
        section_title="Chủ đề Nói",
        count=3,
    )
    assert any("topic" in q.lower() for q in questions)
    assert not any("biện pháp chính" in q.lower() for q in questions)


def test_resolve_conversation_context_handles_numbered_queries(tmp_path):
    mgr = SessionManager(storage_dir=str(tmp_path))
    session = mgr.create_session(title="Test Conversational Context")

    # Turn 1: user asks, assistant responds with 2 suggested questions
    session.add_message(role="user", content="Tài liệu có những chủ đề gì?")
    session.add_message(
        role="assistant",
        content="Dưới đây là các chủ đề.",
        suggested_questions=[
            "Yêu cầu của Topic 1 là gì?",
            "Nội dung chi tiết của Topic 2 được mô tả như thế nào?",
        ],
    )

    # Turn 2: User says "câu 2"
    resolved_q2 = mgr.resolve_conversation_context(session.session_id, "câu 2")
    assert resolved_q2 == "Nội dung chi tiết của Topic 2 được mô tả như thế nào?"

    # Turn 2 alternate: User says "trả lời câu 1"
    resolved_q1 = mgr.resolve_conversation_context(session.session_id, "trả lời câu 1")
    assert resolved_q1 == "Yêu cầu của Topic 1 là gì?"


def test_resolve_conversation_context_cleans_copy_pasted_prefixes(tmp_path):
    mgr = SessionManager(storage_dir=str(tmp_path))
    session = mgr.create_session(title="Test Pasted Cleaning")
    session.add_message(role="user", content="Hello")

    pasted = "2. **[Chủ đề bài học]**: Yêu cầu và nội dung chính của Topic 11 là gì?"
    cleaned = mgr.resolve_conversation_context(session.session_id, pasted)
    assert cleaned == "Yêu cầu và nội dung chính của Topic 11 là gì?"
    assert "**[Chủ đề" not in cleaned


def test_semantic_alias_chude_expands_to_topic():
    exp_text, exp_tokens = expand_query_semantics("cho tôi biết chủ đề 47")
    assert "topic" in exp_text.lower()
    assert "47" in exp_text.lower()


def test_exact_topic_number_boost_prefers_target_topic():
    q = "cho tôi biết topic 1"
    s_target = "Topic 1: Describe an interesting video that you watched on social media."
    s_other = "Topic 50: Describe an interesting historical event that you know about."

    score_target = score_sentence_relevance(s_target, q, ["topic", "1"])
    score_other = score_sentence_relevance(s_other, q, ["topic", "1"])

    assert score_target > score_other + 10.0


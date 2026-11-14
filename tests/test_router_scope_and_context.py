"""
Tests for QuestionRouter Scope Check and SessionManager Anchor Carry-Over:
Verifies:
- Scoped questions with topic alias + N or quoted text route to SECTION_LOOKUP (not DOCUMENT_SUMMARY)
- Pure document summary triggers DOCUMENT_SUMMARY only when no specific scope exists
- 'mục 3' is treated as a document section lookup, NOT hijacked as a question/bullet index
- Multi-turn conversation accurately carries over topic anchor (e.g. 'topic 49') for anaphoric queries
"""

import pytest
from src.web.question_router import QuestionRouter, QuestionRoute
from src.web.session_manager import SessionManager, extract_topic_anchor


def test_question_router_topic_scope_before_summary():
    """Verify that queries with topic aliases + N route to SECTION_LOOKUP even if containing summary trigger words."""
    router = QuestionRouter()

    # Even though "trọng tâm" is a summary trigger, scoped to 'topic 1' must route to SECTION_LOOKUP
    route, meta = router.classify_intent("Nội dung trọng tâm liên quan đến 'topic 1' gồm những gì?")
    assert route == QuestionRoute.SECTION_LOOKUP
    assert "Topic 1" in meta.get("target_section", "")

    # "cho tôi biết chủ đề 47"
    route, meta = router.classify_intent("cho tôi biết chủ đề 47")
    assert route == QuestionRoute.SECTION_LOOKUP
    assert "47" in meta.get("target_section", "")

    # "cho tôi biết topic 49"
    route, meta = router.classify_intent("cho tôi biết topic 49")
    assert route == QuestionRoute.SECTION_LOOKUP
    assert "Topic 49" in meta.get("target_section", "")

    # "Điểm nổi bật của Topic 50" (contains "điểm nổi bật" summary cue)
    route, meta = router.classify_intent("Điểm nổi bật của Topic 50 là gì?")
    assert route == QuestionRoute.SECTION_LOOKUP
    assert "Topic 50" in meta.get("target_section", "")

    # "cho tôi biết mục 3"
    route, meta = router.classify_intent("cho tôi biết mục 3")
    assert route == QuestionRoute.SECTION_LOOKUP
    assert "Mục 3" in meta.get("target_section", "")

    # "bài 5 nói về điều gì"
    route, meta = router.classify_intent("bài 5 nói về điều gì")
    assert route == QuestionRoute.SECTION_LOOKUP
    assert "Bài 5" in meta.get("target_section", "")


def test_question_router_quoted_scope_before_summary():
    """Verify that queries with quoted target after 'liên quan đến/về' route to SECTION_LOOKUP."""
    router = QuestionRouter()

    route, meta = router.classify_intent("Tóm tắt nội dung liên quan đến 'kiến trúc SA-CMS'")
    assert route == QuestionRoute.SECTION_LOOKUP
    assert meta.get("target_section") == "kiến trúc SA-CMS"

    route, meta = router.classify_intent("Điểm chính về \"Nested Learning\" gồm những gì?")
    assert route == QuestionRoute.SECTION_LOOKUP
    assert meta.get("target_section") == "Nested Learning"


def test_question_router_pure_summary_without_scope():
    """Verify that pure summary questions without scope still route to DOCUMENT_SUMMARY."""
    router = QuestionRouter()

    route, _ = router.classify_intent("Tóm tắt nội dung tài liệu này")
    assert route == QuestionRoute.DOCUMENT_SUMMARY

    route, _ = router.classify_intent("Nội dung chính của tài liệu là gì?")
    assert route == QuestionRoute.DOCUMENT_SUMMARY

    route, _ = router.classify_intent("Tổng quan các điểm nổi bật trong file")
    assert route == QuestionRoute.DOCUMENT_SUMMARY


def test_session_manager_muc_not_hijacked_as_index(tmp_path):
    """Verify that 'cho biết mục 3' is NOT hijacked into suggested question or bullet #3."""
    mgr = SessionManager(storage_dir=str(tmp_path))
    session = mgr.create_session(title="Test Muc")

    # Add assistant message with 5 suggested questions
    session.add_message(role="user", content="Tài liệu có gì?")
    session.add_message(
        role="assistant",
        content="Nội dung tóm tắt...",
        suggested_questions=[
            "Câu hỏi 1",
            "Câu hỏi 2",
            "Câu hỏi 3 về SA-CMS",
            "Câu hỏi 4",
            "Câu hỏi 5",
        ],
    )

    # User asks for "cho biết mục 3" (referencing document section 3)
    resolved = mgr.resolve_conversation_context(session.session_id, "cho biết mục 3")
    # Must NOT be replaced with suggested question #3
    assert resolved == "cho biết mục 3"


def test_session_manager_topic_anchor_carry_over(tmp_path):
    """Verify that follow-up queries with anaphoric cues inherit the prior topic anchor."""
    mgr = SessionManager(storage_dir=str(tmp_path))
    session = mgr.create_session(title="Test Topic Anchor")

    # Turn 1
    session.add_message(role="user", content="cho tôi biết topic 49")
    session.add_message(
        role="assistant",
        content="Topic 49: Describe an outdoor sport you would like to try in the future...",
    )

    # Turn 2: "giải thích thêm chi tiết" -> carry over (topic 49)
    resolved_1 = mgr.resolve_conversation_context(session.session_id, "giải thích thêm chi tiết")
    assert "topic 49" in resolved_1.lower()
    assert "(topic 49)" in resolved_1.lower()

    # Turn 2b: "cho ví dụ cụ thể"
    resolved_2 = mgr.resolve_conversation_context(session.session_id, "cho ví dụ cụ thể")
    assert "(topic 49)" in resolved_2.lower()

    # Turn 2c: "tại sao lại như vậy"
    resolved_3 = mgr.resolve_conversation_context(session.session_id, "tại sao lại như vậy")
    assert "(topic 49)" in resolved_3.lower()

    # Turn 2d: "còn về từ vựng thì sao"
    resolved_4 = mgr.resolve_conversation_context(session.session_id, "còn về từ vựng thì sao")
    assert "topic 49" in resolved_4.lower()


def test_session_manager_topic_anchor_not_overwritten_when_explicit(tmp_path):
    """Verify that when user specifies a new topic, the old anchor is NOT carried over."""
    mgr = SessionManager(storage_dir=str(tmp_path))
    session = mgr.create_session(title="Test Anchor Switch")

    session.add_message(role="user", content="cho tôi biết topic 49")
    session.add_message(role="assistant", content="Topic 49 details...")

    # User explicitly asks for topic 50
    resolved = mgr.resolve_conversation_context(session.session_id, "cho tôi biết topic 50")
    assert "topic 49" not in resolved.lower()
    assert resolved == "cho tôi biết topic 50"


def test_session_manager_quoted_anchor_carry_over(tmp_path):
    """Verify quoted anchor carry-over across turns."""
    mgr = SessionManager(storage_dir=str(tmp_path))
    session = mgr.create_session(title="Test Quoted Anchor")

    session.add_message(role="user", content="Nội dung liên quan đến 'SA-CMS' là gì?")
    session.add_message(role="assistant", content="SA-CMS là...")

    resolved = mgr.resolve_conversation_context(session.session_id, "nội dung này có ưu điểm gì")
    assert "SA-CMS" in resolved


def test_question_router_numbered_heading_and_extract_terms():
    """Verify that '3. PHÁ MÃ' routes to SECTION_LOOKUP and _extract_query_terms strips boilerplate."""
    router = QuestionRouter()
    route, meta = router.classify_intent("3. PHÁ MÃ")
    assert route == QuestionRoute.SECTION_LOOKUP
    assert meta.get("target_section") == "3. PHÁ MÃ"

    terms = router._extract_query_terms("Yêu cầu và chi tiết cụ thể liên quan đến 'PHÁ MÃ' gồm những gì?")
    assert terms == ["phá", "mã"]

    terms_natural = router._extract_query_terms("phá mã trong hệ mã khóa đối xứng là gì")
    assert "phá" in terms_natural
    assert "mã" in terms_natural
    assert "gì" not in terms_natural
    assert "trong" not in terms_natural


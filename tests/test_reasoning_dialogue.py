"""
Unit Tests for ParametricDialogueTracker & Anaphora Resolution (Ticket 4.1).
Validates:
1. Dynamic entity extraction across turns.
2. Pronoun coreference resolution ("nó", "chúng").
3. Demonstrative noun phrase resolution ("phương pháp này", "mô hình này").
4. Elliptic question resolution ("Thế còn...").
5. Compact context prompt generation with high token efficiency.
"""

import pytest
from src.reasoning.dialogue_tracker import (
    ParametricDialogueTracker,
    DialogueSessionState,
    DialogueTurn,
)


def test_dialogue_entity_extraction():
    tracker = ParametricDialogueTracker()
    session_id = "test_sess_01"

    resolved, state = tracker.resolve_query(session_id, "Mô hình SA-CMS có ưu điểm gì so với RAG?")
    assert state.active_subject in ("SA-CMS", "RAG")
    assert "sa_cms" in state.entity_registry or "rag" in state.entity_registry


def test_anaphora_resolution_pronoun():
    tracker = ParametricDialogueTracker()
    session_id = "test_sess_02"

    # Turn 1: Introduce subject
    r1, s1 = tracker.resolve_query(session_id, "Hệ thống SA-CMS là gì?")
    tracker.record_turn(session_id, "Hệ thống SA-CMS là gì?", r1, bot_summary="SA-CMS là bộ nhớ liên tục.")
    assert s1.active_subject == "SA-CMS"

    # Turn 2: Pronoun "Nó"
    r2, s2 = tracker.resolve_query(session_id, "Nó giảm bao nhiêu chi phí token?")
    assert "SA-CMS" in r2
    assert "nó" not in r2.lower()


def test_anaphora_resolution_demonstrative():
    tracker = ParametricDialogueTracker()
    session_id = "test_sess_03"

    # Turn 1
    r1, s1 = tracker.resolve_query(session_id, "Kiến trúc Hope-Attention được thiết kế thế nào?")
    tracker.record_turn(session_id, "Kiến trúc Hope-Attention được thiết kế thế nào?", r1)
    assert s1.active_subject == "Hope-Attention"

    # Turn 2: "mô hình này"
    r2, s2 = tracker.resolve_query(session_id, "Ưu điểm của mô hình này là gì?")
    assert "Hope-Attention" in r2
    assert "mô hình này" not in r2.lower()


def test_elliptic_question_resolution():
    tracker = ParametricDialogueTracker()
    session_id = "test_sess_04"

    # Turn 1
    r1, s1 = tracker.resolve_query(session_id, "SA-CMS đạt tỷ lệ nén bao nhiêu?")
    tracker.record_turn(session_id, "SA-CMS đạt tỷ lệ nén bao nhiêu?", r1)

    # Turn 2: Ellipsis "Thế còn độ trễ thì sao?"
    r2, s2 = tracker.resolve_query(session_id, "Thế còn độ trễ thì sao?")
    assert "độ trễ" in r2.lower()
    assert "SA-CMS" in r2


def test_compact_context_prompt_token_savings():
    tracker = ParametricDialogueTracker()
    session_id = "test_sess_05"

    tracker.resolve_query(session_id, "SA-CMS là gì?")
    tracker.record_turn(session_id, "SA-CMS là gì?", "SA-CMS là gì?", bot_summary="Bộ nhớ liên tục.")

    tracker.resolve_query(session_id, "Độ trễ bao nhiêu?")
    tracker.record_turn(session_id, "Độ trễ bao nhiêu?", "Độ trễ của SA-CMS là bao nhiêu?", bot_summary="18.2 ms.")

    prompt = tracker.get_compact_context_prompt(session_id, max_turns=2)
    assert "SA-CMS" in prompt
    assert "Q1:" in prompt or "Q2:" in prompt
    # Ultra-compact length
    assert len(prompt.split()) < 30


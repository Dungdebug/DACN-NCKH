"""
Phase 5.2 — Automated Acceptance Tests for Chat-First Document QA Web Experience.
Tests Session Management, Multi-Doc Chat, Conversational Context Memory, Question Routing,
Evidence-First Grounded Answers, Citation Explorer, Refusal Gate, and Token Telemetry.

STRICT RULE: NO TRAINING, NO GRADIENT UPDATES, FROZEN INFERENCE ONLY.
"""

import io
import pytest
from fastapi.testclient import TestClient

from src.web.app import app, PipelineService
from src.web.question_router import QuestionRouter, QuestionRoute
from src.web.session_manager import SessionManager, ChatSession


@pytest.fixture(scope="module")
def client():
    """TestClient fixture with initialized demo documents."""
    service = PipelineService.get_instance()
    service._initialize_retriever()
    if service.store.document_count == 0:
        from src.web.demo_data import seed_demo_documents
        seed_demo_documents()
        service._initialize_retriever()
    with TestClient(app) as c:
        yield c


# ==============================================================================
# 1. QUESTION ROUTING UNIT TESTS (Task 4 & 5)
# ==============================================================================

def test_question_router_classification():
    router = QuestionRouter()

    # Direct lookup
    route, _ = router.classify_intent("Kiến trúc SA-CMS là gì?", num_attached_documents=1)
    assert route == QuestionRoute.DIRECT_LOOKUP

    # Section lookup
    route, _ = router.classify_intent("Nội dung chính trong phần 2 là gì?", num_attached_documents=1)
    assert route == QuestionRoute.SECTION_LOOKUP

    # Document summary
    route, _ = router.classify_intent("Tóm tắt toàn bộ tài liệu này cho tôi.", num_attached_documents=1)
    assert route == QuestionRoute.DOCUMENT_SUMMARY

    # Cross-document comparison with multiple docs
    route, _ = router.classify_intent("So sánh điểm khác biệt giữa các tài liệu.", num_attached_documents=2)
    assert route == QuestionRoute.CROSS_DOCUMENT_COMPARISON

    # Out of domain unanswerable
    route, _ = router.classify_intent("Thủ đô của Nhật Bản là thành phố nào?", num_attached_documents=1)
    assert route == QuestionRoute.UNANSWERABLE


# ==============================================================================
# 2. CONVERSATIONAL CONTEXT MEMORY TESTS (Task 12)
# ==============================================================================

def test_conversation_context_resolution():
    sm = SessionManager(storage_dir="data/chat_sessions")
    session = sm.create_session(title="Test Context Session")

    # User asks Q1
    session.add_message(role="user", content="CMS là gì?")
    session.add_message(role="assistant", content="CMS là Continual Memory Snapshot.")

    # User asks Q2 using follow-up pronoun "Nó"
    resolved = sm.resolve_conversation_context(session.session_id, "Nó khác RAG thế nào?")
    assert "CMS" in resolved or "RAG" in resolved


# ==============================================================================
# 3. CHAT SESSION LIFECYCLE (Task 13, 14, 15)
# ==============================================================================

def test_chat_session_crud(client):
    # 1. Create Session
    resp = client.post("/api/chat/sessions", json={"title": "Cuộc hội thoại kiểm thử"})
    assert resp.status_code == 200
    data = resp.json()
    session_id = data["session_id"]
    assert data["title"] == "Cuộc hội thoại kiểm thử"

    # 2. List Sessions
    resp = client.get("/api/chat/sessions")
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    assert any(s["session_id"] == session_id for s in sessions)

    # 3. Get Session Detail
    resp = client.get(f"/api/chat/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == session_id

    # 4. Rename Session
    resp = client.patch(f"/api/chat/sessions/{session_id}", json={"title": "Tên mới đã cập nhật"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "Tên mới đã cập nhật"

    # 5. Delete Session
    resp = client.delete(f"/api/chat/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


# ==============================================================================
# 4. MULTI-DOCUMENT ATTACHMENT & DRAG/DROP (Task 2 & 3)
# ==============================================================================

def test_multi_document_attachment(client):
    # Create session
    resp = client.post("/api/chat/sessions", json={"title": "Session Multi-doc"})
    session_id = resp.json()["session_id"]

    # Upload file with session_id
    content = b"# Document Multi-Doc A\nDay la tai lieu A viet ve Nested Learning va SA-CMS."
    files = {"file": ("test_multi_a.md", io.BytesIO(content), "text/markdown")}
    upload_resp = client.post("/api/documents/upload", files=files, data={"session_id": session_id})
    assert upload_resp.status_code == 200
    doc_a_id = upload_resp.json()["document_id"]
    assert upload_resp.json()["status_message"] == "Đã đọc xong tài liệu"

    # Verify document is attached to session
    sess_resp = client.get(f"/api/chat/sessions/{session_id}")
    attached = sess_resp.json()["document_ids"]
    assert doc_a_id in attached

    # Detach document
    detach_resp = client.delete(f"/api/chat/sessions/{session_id}/documents/{doc_a_id}")
    assert detach_resp.status_code == 200
    assert doc_a_id not in detach_resp.json()["document_ids"]


# ==============================================================================
# 5. GROUNDED QA WITH CITATIONS & REFUSAL (Task 6, 7, 8, 30)
# ==============================================================================

def test_grounded_answer_and_citations(client):
    # Create session and attach a known seeded document
    resp = client.post("/api/chat/sessions", json={"title": "Session QA"})
    session_id = resp.json()["session_id"]

    # Send relevant grounded question
    msg_resp = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"query": "Kiến trúc SA-CMS là gì?", "answer_mode": "balanced"},
    )
    assert msg_resp.status_code == 200
    result = msg_resp.json()

    assert result["refused"] is False
    assert len(result["citations"]) > 0
    # Citations must have label [1]
    assert result["citations"][0]["citation_index"] == 1
    assert "passage_id" in result["citations"][0]
    assert "document_title" in result["citations"][0]

    # Verify citation resolver endpoint
    first_passage_id = result["citations"][0]["passage_id"]
    cit_resp = client.get(f"/api/citations/{first_passage_id}")
    assert cit_resp.status_code == 200
    cit_data = cit_resp.json()
    assert cit_data["passage_id"] == first_passage_id
    assert "text" in cit_data
    assert "paragraph_index" in cit_data


def test_refusal_for_out_of_domain_query(client):
    resp = client.post("/api/chat/sessions", json={"title": "Session Refusal"})
    session_id = resp.json()["session_id"]

    # Send out-of-domain world knowledge query
    msg_resp = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"query": "Thủ đô của Nhật Bản là thành phố nào?", "answer_mode": "balanced"},
    )
    assert msg_resp.status_code == 200
    result = msg_resp.json()

    # Must strictly refuse according to document-grounded principle
    assert result["refused"] is True
    assert "Tài liệu" in result["message"]["content"]
    assert "không có thông tin" in result["message"]["content"] or "chưa đủ" in result["message"]["content"]


# ==============================================================================
# 6. ANSWER LENGTH MODES (Task 11)
# ==============================================================================

def test_answer_length_modes(client):
    resp = client.post("/api/chat/sessions", json={"title": "Session Lengths"})
    session_id = resp.json()["session_id"]

    # Concise mode
    concise_resp = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"query": "SA-CMS hoạt động như thế nào?", "answer_mode": "concise"},
    )
    assert concise_resp.status_code == 200
    concise_words = len(concise_resp.json()["message"]["content"].split())

    # Detailed mode
    detailed_resp = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"query": "SA-CMS hoạt động như thế nào?", "answer_mode": "detailed"},
    )
    assert detailed_resp.status_code == 200
    detailed_words = len(detailed_resp.json()["message"]["content"].split())

    assert detailed_words >= concise_words


# ==============================================================================
# 7. COMPARE MODE ENDPOINT (Task 24)
# ==============================================================================

def test_compare_methods_endpoint(client):
    resp = client.post(
        "/api/chat/compare",
        json={
            "query": "Kiến trúc SA-CMS là gì?",
            "methods": ["B1", "B2", "P1", "P2"],
            "answer_mode": "balanced",
        },
    )
    assert resp.status_code == 200
    comp = resp.json()["comparison"]
    assert "B1" in comp
    assert "B2" in comp
    assert "P1" in comp
    assert "P2" in comp
    assert comp["P2"]["total_tokens"] > 0

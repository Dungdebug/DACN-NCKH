"""
Tests for clean output, strict document isolation, and follow-up consistency in src/web/app.py.
Validates:
1. Clean Output: Zero robotic System-2 preamble ('🧠 **Tiến trình nơ-ron suy ngẫm') in returned content;
   reasoning trace resides in diagnostics['reasoning_trace'].
2. Strict Document Isolation: Sessions with no attached document and no title match refuse cleanly
   ('NO_ATTACHED_DOCUMENT') with a clear prompt to attach a document, preventing cross-document leakage.
3. Follow-up Questions Consistency: diagnostics['suggested_questions'] matches the trailing rendered list.
4. Re-chunking Persistence: PipelineService rechunk_all_documents executes cleanly.
"""

import pytest
from fastapi.testclient import TestClient
from src.web.app import app, PipelineService


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def service():
    return PipelineService.get_instance()


def test_clean_output_no_robotic_system2_preamble(client):
    """Verifies assistant content does not contain robotic System-2 headers."""
    # Create session with DEMO_DOC_001 attached
    sess_res = client.post("/api/chat/sessions", json={"title": "Test Clean Output Session"})
    assert sess_res.status_code == 200
    sess_id = sess_res.json()["session_id"]

    client.post(f"/api/chat/sessions/{sess_id}/documents", json={"document_id": "DEMO_DOC_001"})

    # Send a query in detailed/balanced mode
    msg_res = client.post(
        f"/api/chat/sessions/{sess_id}/messages",
        json={"query": "Kiến trúc bộ nhớ liên tục hoạt động như thế nào?", "answer_mode": "detailed"},
    )
    assert msg_res.status_code == 200
    data = msg_res.json()

    content = data.get("answer") or data.get("content") or data.get("message", {}).get("content", "")
    # Must NOT contain robotic System-2 analytical banner in user-facing content
    assert "🧠 **Tiến trình nơ-ron suy ngẫm" not in content
    assert "Tiến trình nơ-ron suy ngẫm (System 2):" not in content
    assert "chuỗi phân tích nơ-ron đa chặng" not in content

    # Diagnostics should preserve reasoning trace if generated
    diag = data.get("diagnostics", {})
    assert "reasoning_trace" in diag


def test_strict_document_isolation_no_attached_doc(client):
    """Verifies sessions with no attached docs cleanly refuse rather than leaking other docs."""
    sess_res = client.post("/api/chat/sessions", json={"title": "Isolated Empty Session"})
    assert sess_res.status_code == 200
    sess_id = sess_res.json()["session_id"]

    # Query with generic query that does not name any document
    msg_res = client.post(
        f"/api/chat/sessions/{sess_id}/messages",
        json={"query": "Các nhiệm vụ trọng tâm và mục tiêu phấn đấu trong năm 2026 là gì?"},
    )
    assert msg_res.status_code == 200
    data = msg_res.json()

    # Must be cleanly refused with NO_ATTACHED_DOCUMENT
    assert data.get("refused") is True
    assert data.get("refusal_reason") == "NO_ATTACHED_DOCUMENT"
    answer_text = data.get("answer") or data.get("content", "")
    assert "đính kèm tài liệu" in answer_text.lower()
    # Must have zero citations and zero leaked evidence
    assert len(data.get("citations", [])) == 0
    assert len(data.get("evidence", [])) == 0


def test_followup_questions_consistency(client):
    """Verifies diagnostics.suggested_questions matches the rendered trailing question list."""
    sess_res = client.post("/api/chat/sessions", json={"title": "Test Followup Consistency"})
    sess_id = sess_res.json()["session_id"]
    client.post(f"/api/chat/sessions/{sess_id}/documents", json={"document_id": "DEMO_DOC_001"})

    msg_res = client.post(
        f"/api/chat/sessions/{sess_id}/messages",
        json={"query": "Tỷ lệ giảm token của SA-CMS là bao nhiêu?", "answer_mode": "balanced"},
    )
    assert msg_res.status_code == 200
    data = msg_res.json()

    if not data.get("refused"):
        suggested = data.get("suggested_questions", [])
        diag_suggested = data.get("diagnostics", {}).get("suggested_questions", [])
        # Both must match exactly
        assert suggested == diag_suggested
        if suggested:
            content = data.get("answer") or data.get("content", "")
            # At least one question from suggested should be in content trailing block
            assert any(q.split("?")[0] in content for q in suggested)


def test_rechunk_all_documents_persists_cleanly(service):
    """Verifies that PipelineService.rechunk_all_documents executes without error."""
    # Should run smoothly over all stored documents
    service.rechunk_all_documents()
    assert service.store.document_count > 0

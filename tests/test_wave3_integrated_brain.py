"""
Tests for Wave 3 Integrated Brain:
- End-to-end DefenseStrategist integration inside DocumentAnalyzer.find_best_grounded_answer()
- Deep Epistemic Calibration (EpistemicCalibrator.calibrate_deep) wired into chat endpoints
- Backward-compatible retrieval with query decomposition and focus entities
"""

import pytest
from dataclasses import dataclass
from fastapi.testclient import TestClient

from src.web.app import app
from src.web.document_analyzer import DocumentAnalyzer


@dataclass
class MockPassage:
    text: str
    section_title: str
    document_id: str = "DOC_TEST_01"
    document_title: str = "Nghiên cứu SA-CMS"
    paragraph_index: int = 1
    score: float = 0.95
    passage_id: str = "P_01"


def test_defense_strategist_wired_in_find_best_grounded_answer():
    """Verify that DefenseStrategist synthesizes defense blocks in detailed mode for non-generic queries."""
    passages = [
        MockPassage(
            text="Lý do lựa chọn SA-CMS thay vì Transformer thuần túy là đề xuất cơ chế phân tầng bộ nhớ liên cấp nhằm giải quyết bài toán tràn bộ nhớ GPU.",
            section_title="Phương pháp và Kiến trúc đề xuất",
            paragraph_index=1,
            passage_id="P_01",
        ),
        MockPassage(
            text="So với Transformer truyền thống, mô hình baseline cũ tiêu tốn O(N^2) tài nguyên tính toán.",
            section_title="Nghiên cứu liên quan và baseline",
            paragraph_index=2,
            passage_id="P_02",
        ),
        MockPassage(
            text="Kết quả thực nghiệm cho thấy phương pháp đạt mức giảm 42% VRAM và tăng tốc 1.8x trên RTX 3050.",
            section_title="Thực nghiệm và Kết quả đo đạc",
            paragraph_index=3,
            passage_id="P_03",
        ),
        MockPassage(
            text="Tuy nhiên, sự đánh đổi là chi phí thời gian khởi tạo cấu trúc chỉ mục ban đầu.",
            section_title="Thảo luận và Đánh đổi",
            paragraph_index=4,
            passage_id="P_04",
        ),
    ]

    defense_query = "Tại sao lại chọn SA-CMS mà không phải Transformer thuần túy?"

    # In detailed mode: Defense argument block must be present and precede deep synthesis
    detailed_answer, citations = DocumentAnalyzer.find_best_grounded_answer(
        query=defense_query,
        evidence_passages=passages,
        answer_mode="detailed",
    )

    assert "🎓 **Lập luận bảo vệ:**" in detailed_answer
    assert "💡 **Tổng hợp & Đúc kết chuyên sâu:**" in detailed_answer

    # Verify order: Defense block comes before Deep Synthesis
    pos_defense = detailed_answer.find("🎓 **Lập luận bảo vệ:**")
    pos_synthesis = detailed_answer.find("💡 **Tổng hợp & Đúc kết chuyên sâu:**")
    assert pos_defense < pos_synthesis, "Defense argument block must appear before deep synthesis block!"

    # In balanced mode: Defense block is omitted (reserved for detailed/defense queries)
    balanced_answer, _ = DocumentAnalyzer.find_best_grounded_answer(
        query=defense_query,
        evidence_passages=passages,
        answer_mode="balanced",
    )
    assert "🎓 **Lập luận bảo vệ:**" not in balanced_answer


def test_deep_epistemic_calibration_in_e2e_chat():
    """Verify that EpistemicCalibrator.calibrate_deep is wired into the chat endpoint response & diagnostics."""
    client = TestClient(app)

    # 1. Create a session and attach document
    resp_session = client.post("/api/chat/sessions", json={"title": "Test Wave 3 Integrated"})
    assert resp_session.status_code == 200
    session_id = resp_session.json()["session_id"]
    client.post(f"/api/chat/sessions/{session_id}/documents", json={"document_id": "DEMO_DOC_001"})

    # 2. Ask question in detailed mode
    resp_chat = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={
            "query": "Quy định về thời gian công tác là gì và các biện pháp thực hiện ra sao?",
            "answer_mode": "detailed",
            "top_k": 3,
        },
    )
    assert resp_chat.status_code == 200
    data = resp_chat.json()

    assert "answer" in data
    assert "diagnostics" in data
    # Diagnostics must contain epistemic_calibration
    assert "epistemic_calibration" in data["diagnostics"]
    ep_diag = data["diagnostics"]["epistemic_calibration"]
    assert "uncertainty_score" in ep_diag
    assert "epistemic_label" in ep_diag
    assert "evidence_quality" in ep_diag
    assert "claim_support" in ep_diag
    assert "conflicts" in ep_diag


def test_focus_entities_and_diversify_retrieval_resilience():
    """Verify that retrieval handles queries with focus entities and diversification flags cleanly."""
    client = TestClient(app)

    # Create session
    resp_session = client.post("/api/chat/sessions", json={"title": "Test Focus Entity"})
    assert resp_session.status_code == 200
    session_id = resp_session.json()["session_id"]

    # Query containing multiple focus entities: SA-CMS, BM25, RTX 3050
    resp_chat = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={
            "query": "So sánh SA-CMS và BM25 trên GPU RTX 3050",
            "answer_mode": "detailed",
            "top_k": 3,
        },
    )
    assert resp_chat.status_code == 200
    data = resp_chat.json()
    assert "answer" in data
    assert len(data["answer"]) > 10

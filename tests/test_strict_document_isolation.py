"""
Tests for Strict Document Boundary & Zero Cross-Contamination Retrieval.
Ensures when a document is attached/locked, queries are answered strictly from that document,
and questions regarding other documents are strictly refused without leakage.
"""

import io
import pytest
from fastapi.testclient import TestClient

from src.web.app import app, PipelineService
from src.web.session_manager import SessionManager


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    test_session_dir = tmp_path_factory.mktemp("test_sessions_isolation")
    service = PipelineService.get_instance()
    orig_sm = service.session_manager
    service.session_manager = SessionManager(storage_dir=str(test_session_dir))
    service._initialize_retriever()
    with TestClient(app) as c:
        yield c
    service.session_manager = orig_sm


def test_zero_leakage_between_isolated_documents(client):
    service = PipelineService.get_instance()

    # 1. Upload Doc Alpha: About Solar Energy project in Ninh Thuan
    content_alpha = (
        "Dự án điện mặt trời tại Ninh Thuận có công suất thiết kế 450 MW. "
        "Tổng mức đầu tư cho dự án là 5000 tỷ đồng và vận hành từ năm 2024."
    )
    res_a = client.post(
        "/api/documents/upload",
        files={"file": ("ninh_thuan_solar.txt", io.BytesIO(content_alpha.encode("utf-8")), "text/plain")},
        data={"title": "Điện Mặt Trời Ninh Thuận"}
    )
    assert res_a.status_code == 200
    doc_alpha_id = res_a.json()["document_id"]

    # 2. Upload Doc Beta: About AI Coffee sorting machine in Da Lat
    content_beta = (
        "Máy phân loại cà phê bằng trí tuệ nhân tạo tại Đà Lạt đạt độ chính xác 99.2%. "
        "Năng suất phân loại đạt 15 tấn mỗi ngày và sử dụng camera cảm biến quang phổ."
    )
    res_b = client.post(
        "/api/documents/upload",
        files={"file": ("da_lat_coffee.txt", io.BytesIO(content_beta.encode("utf-8")), "text/plain")},
        data={"title": "Máy Phân Loại Cà Phê Đà Lạt"}
    )
    assert res_b.status_code == 200
    doc_beta_id = res_b.json()["document_id"]

    # 3. Create Session A locked exclusively to Doc Alpha
    sess_a_resp = client.post("/api/chat/sessions", json={"title": "Session Alpha", "document_ids": [doc_alpha_id]})
    assert sess_a_resp.status_code == 200
    sess_a_id = sess_a_resp.json()["session_id"]

    # Ask Doc Alpha relevant question
    q_alpha_resp = client.post(
        f"/api/chat/sessions/{sess_a_id}/messages",
        json={"query": "Công suất thiết kế của dự án điện mặt trời Ninh Thuận là bao nhiêu?"}
    )
    assert q_alpha_resp.status_code == 200
    res_q_a = q_alpha_resp.json()
    assert res_q_a["refused"] is False
    assert len(res_q_a["citations"]) > 0
    # Every citation MUST be from doc_alpha_id
    for cit in res_q_a["citations"]:
        assert cit["document_id"] == doc_alpha_id

    # 4. Ask Session A a question from Doc Beta (cross-document attack!)
    # Should be strictly REFUSED, not answering about coffee sorting!
    q_cross_resp = client.post(
        f"/api/chat/sessions/{sess_a_id}/messages",
        json={"query": "Năng suất phân loại cà phê bằng trí tuệ nhân tạo là bao nhiêu tấn mỗi ngày?"}
    )
    assert q_cross_resp.status_code == 200
    res_q_cross = q_cross_resp.json()
    assert res_q_cross["refused"] is True
    # Citations must be 0
    assert len(res_q_cross["citations"]) == 0
    # Message must indicate that the document does not contain this info
    assert "Điện Mặt Trời Ninh Thuận" in res_q_cross["message"]["content"] or "không chứa thông tin" in res_q_cross["message"]["content"]

    # 5. Create Session B locked exclusively to Doc Beta
    sess_b_resp = client.post("/api/chat/sessions", json={"title": "Session Beta", "document_ids": [doc_beta_id]})
    assert sess_b_resp.status_code == 200
    sess_b_id = sess_b_resp.json()["session_id"]

    # Ask Doc Beta relevant question
    q_beta_resp = client.post(
        f"/api/chat/sessions/{sess_b_id}/messages",
        json={"query": "Năng suất phân loại cà phê đạt bao nhiêu tấn mỗi ngày?"}
    )
    assert q_beta_resp.status_code == 200
    res_q_b = q_beta_resp.json()
    assert res_q_b["refused"] is False
    assert len(res_q_b["citations"]) > 0
    for cit in res_q_b["citations"]:
        assert cit["document_id"] == doc_beta_id

    # Ask Session B a question from Doc Alpha (should be strictly refused)
    q_beta_cross = client.post(
        f"/api/chat/sessions/{sess_b_id}/messages",
        json={"query": "Tổng mức đầu tư cho dự án điện mặt trời là bao nhiêu?"}
    )
    assert q_beta_cross.status_code == 200
    assert q_beta_cross.json()["refused"] is True
    assert len(q_beta_cross.json()["citations"]) == 0


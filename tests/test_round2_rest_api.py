"""
Test suite for Phase 5.13 Standardized REST APIs:
- POST /documents
- GET /documents
- DELETE /documents/{id}
- POST /chat
- GET /chat/{id}
- POST /retrieve
- GET /evidence/{id}
- GET /research/trace
- GET /research/metrics
"""

import pytest
from fastapi.testclient import TestClient

from src.web.app import app, PipelineService


@pytest.fixture(scope="module")
def client():
    service = PipelineService.get_instance()
    service._initialize_retriever()
    if service.store.document_count == 0:
        from src.web.demo_data import seed_demo_documents
        seed_demo_documents()
        service._initialize_retriever()
    with TestClient(app) as c:
        yield c


def test_standardized_documents_api(client):
    # 1. GET /documents
    resp = client.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert "documents" in data
    assert "total" in data

    # 2. POST /documents
    post_resp = client.post("/documents", json={
        "title": "API Test Doc",
        "text": "Kiến trúc bộ nhớ SA-CMS hỗ trợ hỏi đáp chuẩn xác.",
        "category": "Test"
    })
    assert post_resp.status_code == 200
    doc_id = post_resp.json()["document_id"]

    # 3. DELETE /documents/{id}
    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200


def test_standardized_retrieve_api(client):
    resp = client.post("/retrieve", json={
        "query": "Kiến trúc SA-CMS",
        "top_k": 3
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "candidates" in data
    assert "latency_ms" in data


def test_standardized_chat_api(client):
    resp = client.post("/chat", json={
        "query": "Kiến trúc SA-CMS là gì?",
        "answer_mode": "concise"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data
    assert "session_id" in data
    session_id = data["session_id"]

    # GET /chat/{id}
    chat_resp = client.get(f"/chat/{session_id}")
    assert chat_resp.status_code == 200
    assert chat_resp.json()["session_id"] == session_id


def test_standardized_research_trace_api(client):
    resp = client.get("/research/trace")
    assert resp.status_code == 200
    data = resp.json()
    assert "memory_residual_norm" in data
    assert "level_contributions" in data
    assert "heatmap" in data


def test_standardized_research_metrics_api(client):
    resp = client.get("/research/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "official_phase_4" in data
    assert "round_2_extension" in data
    assert data["official_phase_4"]["status"] == "FROZEN_LOCKED"
    assert data["round_2_extension"]["status"] == "ROUND_2_ACTIVE"
    assert data["round_2_extension"]["robustness"]["passed_conditions"] == 8

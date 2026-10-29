"""
Unit & Integration Tests for Phase 5.1 Document Intelligence Web Platform.
Validates FastAPI endpoints, DocumentStore integration, file parser security,
citation explorer, refusal gate, method comparison, and token-efficiency telemetry.

TARGET: 100% PASS, ZERO TRAINING.
"""

import pytest
import json
import io
from fastapi.testclient import TestClient

from src.web.app import app
from src.web.file_parser import sanitize_filename, validate_file, extract_text_from_file
from src.web.token_optimizer import build_token_efficient_prompt, AnswerLengthMode


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# ==============================================================================
# 1. HEALTH & METADATA TESTS
# ==============================================================================

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["phase"] == "5.1"
    assert "document_count" in data
    assert "indexed_passages" in data


def test_serve_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SA-CMS" in response.text


# ==============================================================================
# 2. DOCUMENT WORKSPACE & STRUCTURAL VIEW TESTS
# ==============================================================================

def test_list_documents(client):
    response = client.get("/api/documents?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)
    if data["documents"]:
        first = data["documents"][0]
        assert "document_id" in first
        assert "title" in first
        assert "version" in first
        assert "word_count" in first
        assert "token_estimate" in first


def test_create_and_delete_raw_document(client):
    doc_id = f"TEST_DOC_{int(pytest.__version__.replace('.', '')[:4])}"
    payload = {
        "title": "Kiểm thử Nền tảng Web Phase 5.1",
        "text": "# Phần 1. Giới thiệu\n\nNội dung đoạn văn đầu tiên kiểm tra parsing.\n\n# Phần 2. Thử nghiệm\n\nĐoạn văn thứ hai kiểm tra ranh giới.",
        "document_id": doc_id,
        "category": "Test",
    }
    create_res = client.post("/api/documents/raw", json=payload)
    assert create_res.status_code == 200
    create_data = create_res.json()
    assert create_data["success"] is True
    assert create_data["document_id"] == doc_id
    assert create_data["passage_count"] >= 1

    # Check detail
    detail_res = client.get(f"/api/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["title"] == payload["title"]

    # Check Structure View (Task 4)
    struct_res = client.get(f"/api/documents/{doc_id}/structure")
    assert struct_res.status_code == 200
    struct_data = struct_res.json()
    assert "hierarchy" in struct_data
    assert "sa_cms_schedule_boundaries" in struct_data
    assert struct_data["total_paragraphs"] >= 2

    # Clean up (Delete)
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True


def test_memory_visualization_endpoint(client):
    # Retrieve demo doc
    response = client.get("/api/documents?limit=5")
    docs = response.json()["documents"]
    if docs:
        doc_id = docs[0]["document_id"]
        mem_res = client.get(f"/api/memory/{doc_id}")
        assert mem_res.status_code == 200
        mem_data = mem_res.json()
        assert "timescales" in mem_data
        assert "level_1" in mem_data["timescales"]
        assert "level_2" in mem_data["timescales"]
        assert "level_3" in mem_data["timescales"]
        assert mem_data["total_trainable_parameters"] == 5314752


# ==============================================================================
# 3. CHAT, REFUSAL, AND CITATION TESTS
# ==============================================================================

def test_chat_hybrid_mode_answerable(client):
    # Query answerable demo document item
    payload = {
        "query": "Tầng 1 của hệ thống CMS đảm nhiệm vai trò gì?",
        "mode": "hybrid",
        "answer_mode": "balanced",
        "top_k": 3,
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "telemetry" in data
    assert data["telemetry"]["total_tokens"] > 0
    assert data["telemetry"]["total_latency_ms"] >= 0


def test_chat_hybrid_mode_unanswerable_refusal(client):
    # Query unanswerable / out-of-domain question to trigger refusal controller
    payload = {
        "query": "Nhiệt độ sôi của dung dịch plutonium hexafluoride trong chân không vũ trụ là bao nhiêu độ C?",
        "mode": "hybrid",
        "answer_mode": "balanced",
        "top_k": 3,
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["refused"] is True
    assert data["refusal_reason"] in (
        "no_evidence",
        "insufficient_evidence",
        "retrieval_threshold",
        "insufficient_coverage",
        "insufficient_evidence_coverage",
    )
    assert len(data["refusal_message"]) > 0


def test_citation_resolver(client):
    # Ingest query to get a valid citation passage ID
    payload = {
        "query": "Đường sắt cao tốc",
        "mode": "hybrid",
        "answer_mode": "balanced",
        "top_k": 3,
    }
    res = client.post("/api/chat", json=payload)
    data = res.json()
    citations = data.get("citations", [])
    if citations:
        passage_id = citations[0]
        cit_res = client.get(f"/api/citations/{passage_id}")
        assert cit_res.status_code == 200
        cit_data = cit_res.json()
        assert cit_data["passage_id"] == passage_id
        assert "document_title" in cit_data
        assert "text" in cit_data
        assert cit_data["support_status"] == "SUPPORTED"


# ==============================================================================
# 4. METHOD COMPARISON & TOKEN EFFICIENCY TESTS
# ==============================================================================

def test_method_comparison_endpoint(client):
    payload = {
        "query": "Dự án đường sắt cao tốc Bắc Nam",
        "methods": ["B1", "B2", "P1", "P2"],
        "answer_mode": "balanced",
    }
    response = client.post("/api/chat/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "comparison" in data
    comp = data["comparison"]
    for m in ["B1", "B2", "P1", "P2"]:
        assert m in comp
        assert "input_tokens" in comp[m]
        assert "output_tokens" in comp[m]
        assert "latency_ms" in comp[m]


def test_token_efficiency_analytics(client):
    res_analytics = client.get("/api/analytics")
    assert res_analytics.status_code == 200
    a_data = res_analytics.json()
    assert "reductions" in a_data
    assert "tradeoffs" in a_data

    res_eff = client.get("/api/token-efficiency")
    assert res_eff.status_code == 200
    e_data = res_eff.json()
    assert "reductions" in e_data
    assert "recent_experiments" in e_data


def test_history_endpoint(client):
    response = client.get("/api/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# ==============================================================================
# 5. SECURITY & FILE PARSING TESTS (Task 21)
# ==============================================================================

def test_filename_sanitization():
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("My Document (Final) 2026.pdf") == "My_Document__Final__2026.pdf"


def test_file_validation_constraints():
    # Unsupported extension
    is_valid, err = validate_file("malicious.exe", b"binarycontent")
    assert is_valid is False
    assert "Unsupported file type" in err

    # Empty file
    is_valid, err = validate_file("empty.txt", b"")
    assert is_valid is False
    assert "empty" in err

    # Valid TXT
    is_valid, err = validate_file("notes.txt", b"Valid text contents")
    assert is_valid is True
    assert err is None


def test_extract_text_txt():
    raw_text, meta = extract_text_from_file("report.txt", b"# Heading\n\nParagraph text here.")
    assert "Heading" in raw_text
    assert meta["extension"] == ".txt"
    assert meta["word_count"] > 0


def test_token_optimizer_prompt_building():
    p_bal, max_bal = build_token_efficient_prompt("Test question?", "Context text", AnswerLengthMode.BALANCED, "vi")
    p_con, max_con = build_token_efficient_prompt("Test question?", "Context text", AnswerLengthMode.CONCISE, "vi")
    p_min, max_min = build_token_efficient_prompt("Test question?", "Context text", AnswerLengthMode.MINIMAL, "vi")

    assert max_min < max_con < max_bal
    assert "ngắn" in p_min or "tối thiểu" in p_min

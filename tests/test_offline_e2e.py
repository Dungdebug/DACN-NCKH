"""
Offline E2E Verification Suite for SA-CMS Research Prototype.
Enforces strict hardware socket blocking against non-localhost addresses to guarantee
100% offline autonomy without any external network, cloud LLM, or remote service.
"""

import os
import socket
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent

# Enforce offline flags
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SA_CMS_OFFLINE_MODE"] = "1"

from src.web.app import app, PipelineService
from src.web.local_model_loader import LocalModelLoader
from src.web.session_manager import SessionManager


@pytest.fixture(autouse=True)
def block_external_networks(monkeypatch):
    """
    Strict socket interceptor:
    Permits only localhost / loopback addresses.
    Raises ConnectionRefusedError on any attempt to contact external IP/DNS.
    """
    orig_connect = socket.socket.connect

    def guarded_connect(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        allowed = ("127.0.0.1", "localhost", "::1")
        if host not in allowed:
            raise ConnectionRefusedError(
                f"STRICT OFFLINE ENFORCEMENT: External network access to '{host}' is strictly forbidden."
            )
        return orig_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(scope="module")
def client():
    """Test client for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_local_model_loader_offline():
    """Verify local model loader loads SmolLM2 and SA-CMS without network calls."""
    loader = LocalModelLoader.get_instance()
    model, tokenizer = loader.load_offline_model(prefer_gpu=False)

    assert loader.model_loaded is True
    assert model is not None
    assert tokenizer is not None
    assert "SmolLM2" in loader.config["model"]["backbone_name"]
    assert loader.active_checkpoint_status == "CURRENT_VALID_CHECKPOINT"


def test_system_status_endpoint(client):
    """Verify GET /api/system/status reflects 100% offline readiness."""
    res = client.get("/api/system/status")
    assert res.status_code == 200
    data = res.json()

    assert data["model"] == "READY"
    assert data["tokenizer"] == "READY"
    assert "READY" in data["sa_cms_checkpoint"]
    assert data["offline_mode"] == "ACTIVE"
    assert data["network"] == "DISABLED / NOT USED"


def test_offline_document_ingestion_and_retrieval(client):
    """Verify local text extraction, structure parsing, and BM25 indexation."""
    doc_payload = {
        "title": "Tài liệu Nghiên cứu SA-CMS Mẫu Offline",
        "text": "# Chương 1: Kiến trúc Bộ nhớ\n\nKiến trúc SA-CMS áp dụng cơ chế căn chỉnh theo cấu trúc tài liệu gồm 3 cấp độ: đoạn văn, chương mục và toàn văn tài liệu.\n\n# Chương 2: Hiệu năng Thực nghiệm\n\nHiệu năng thu hồi logit trên bài toán needle MK-NIAH tăng 175% dưới điều kiện không truy xuất.",
        "category": "Research",
    }
    res = client.post("/api/documents/raw", json=doc_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    doc_id = data["document_id"]

    # Verify retrieval
    ret_res = client.post("/api/retrieval/search", json={"query": "needle MK-NIAH tăng 175%", "document_ids": [doc_id], "top_k": 3})
    assert ret_res.status_code == 200
    hits = ret_res.json()["passages"]
    assert len(hits) > 0
    assert any(h["document_id"] == doc_id for h in hits)


def test_offline_direct_chat_endpoint(client):
    """Verify POST /chat using ONLY local model resources with citations."""
    payload = {
        "query": "Kiến trúc SA-CMS gồm mấy cấp độ?",
        "mode": "hybrid",
        "answer_mode": "balanced",
    }
    res = client.post("/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "answer" in data
    assert len(data["answer"]) > 10
    assert data["refused"] is False
    assert len(data["citations"]) > 0


def test_offline_unanswerable_refusal(client):
    """Verify refusal trigger when evidence is absent without cloud AI assistance."""
    payload = {
        "query": "Thời tiết ngày mai tại thủ đô Paris dự báo thế nào?",
        "mode": "hybrid",
        "answer_mode": "balanced",
    }
    res = client.post("/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["refused"] is True
    assert "Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn." in data["answer"]
    assert len(data["citations"]) == 0


def test_offline_session_flow_and_citations(client):
    """Verify multi-turn session chat with local evidence citation inspection."""
    # 1. Create session
    sess_res = client.post("/api/chat/sessions", json={"title": "Kiểm thử Phiên Offline"})
    assert sess_res.status_code == 200
    session_id = sess_res.json()["session_id"]

    # 2. Ask question
    msg_res = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"query": "Hiệu năng trên bài toán MK-NIAH tăng bao nhiêu phần trăm?"},
    )
    assert msg_res.status_code == 200
    msg_data = msg_res.json()
    assert msg_data["refused"] is False
    assert len(msg_data["citations"]) > 0

    # 3. Inspect citation locally
    cit = msg_data["citations"][0]
    passage_id = cit["passage_id"]
    doc_id = cit["document_id"]
    cit_res = client.get(f"/api/documents/{doc_id}/citations/{passage_id}")
    assert cit_res.status_code == 200
    assert cit_res.json()["passage_id"] == passage_id


def test_offline_history_persistence_across_restart(client):
    """Verify chat sessions survive backend reboot and reload from local disk."""
    sess_res = client.post("/api/chat/sessions", json={"title": "Phiên Lưu Trữ Bền Vững"})
    session_id = sess_res.json()["session_id"]

    client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"query": "Tin nhắn thử nghiệm kiểm tra tính bền vững."},
    )

    # Re-initialize SessionManager to simulate application restart
    new_manager = SessionManager(storage_dir="data/chat_sessions")
    reloaded_sess = new_manager.get_session(session_id)

    assert reloaded_sess is not None
    assert reloaded_sess.session_id == session_id
    assert len(reloaded_sess.messages) >= 2  # user + assistant
    assert reloaded_sess.messages[0]["content"] == "Tin nhắn thử nghiệm kiểm tra tính bền vững."


def test_checkpoint_switch_endpoint(client):
    """Verify switching checkpoints without altering neural architecture."""
    res = client.post("/api/checkpoints/switch?checkpoint_name=cms_3lvl_seed_42.pt")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["active_checkpoint"] == "cms_3lvl_seed_42.pt"

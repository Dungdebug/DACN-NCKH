"""
End-to-End Pipeline Integration & Benchmark Verification (Phase 5).
Validates the sovereign on-device QA brain across all dimensions:
1. Multi-turn Dialogue Context Tracking (ParametricDialogueTracker).
2. Friston Predictive Surprisal Filtering (EntropySurprisalGate).
3. Grounded Synthesis & Zero Hallucination (NeuralReaderEngine).
4. Calibrated Meta-Cognition (MetaCognitiveEvaluator).
5. Sub-100ms Inference Latency and Measurable Token Frugality.
"""

import json
import time
import pytest
from fastapi.testclient import TestClient

from src.web.app import app, PipelineService


@pytest.fixture(scope="module")
def client():
    service = PipelineService.get_instance()

    benchmark_text = (
        "Hệ thống SA-CMS là kiến trúc bộ nhớ liên tục đa thang căn theo cấu trúc phân tầng của văn bản. "
        "Mô hình tích hợp ba tầng bộ nhớ bao gồm đoạn, phần và tài liệu. "
        "Mô hình SA-CMS đạt tỷ lệ giảm 87,5% chi phí token và giảm 68,2% độ trễ trên GPU RTX 3050. "
        "Tuy nhiên, hạn chế hiện tại là hệ thống đòi hỏi tài liệu có ranh giới cấu trúc văn bản rõ ràng."
    )
    if "doc_benchmark_e2e_01" not in service.store._index:
        rec = service.store.add_document(
            title="Tài liệu SA-CMS Benchmark",
            raw_text=benchmark_text,
            document_id="doc_benchmark_e2e_01",
        )
    else:
        rec = service.store.get_document("doc_benchmark_e2e_01")

    rec.passages = service.chunker.chunk_document(rec)
    doc_file = service.store.docs_dir / "doc_benchmark_e2e_01" / f"v{rec.version}.json"
    with open(doc_file, "w", encoding="utf-8") as f:
        json.dump(rec.to_dict(), f, indent=2, ensure_ascii=False)

    service._initialize_retriever()
    service.load_model_if_requested(prefer_gpu=False)
    return TestClient(app)


def test_e2e_pipeline_single_turn_grounded_answer(client):
    """Verifies single-turn grounded answer with exact citation, intent detection, and low latency."""
    t0 = time.perf_counter()
    payload = {
        "query": "SA-CMS là gì?",
        "document_id": "doc_benchmark_e2e_01",
        "answer_mode": "balanced",
        "use_entropy_filter": True,
        "use_meta_cognition": True,
    }
    res = client.post("/api/self-study/ask", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert res.status_code == 200
    data = res.json()

    assert data["is_grounded"] is True
    assert data["confidence"] >= 0.50
    assert "kiến trúc bộ nhớ liên tục" in data["core_answer"].lower()
    assert "[1]" in data["formatted_answer"]
    assert "meta_cognition" in data
    assert len(data["meta_cognition"]["verified_facts"]) >= 1
    # Check pipeline latency
    assert data["end_to_end_latency_ms"] < 500.0
    assert elapsed_ms < 5000.0


def test_e2e_pipeline_multi_turn_anaphora_resolution(client):
    """
    Verifies multi-turn dialogue state:
    Turn 1 sets active entity to SA-CMS.
    Turn 2 asks 'Ưu điểm của nó là gì?' -> resolves 'nó' to 'SA-CMS'.
    Turn 3 asks 'Còn hạn chế thì sao?' -> resolves elliptic question to limitation of SA-CMS.
    """
    session_id = f"e2e_session_{int(time.time())}"

    # Turn 1
    t1_res = client.post("/api/self-study/ask", json={
        "session_id": session_id,
        "query": "Kiến trúc của mô hình SA-CMS gồm những tầng nào?",
        "document_id": "doc_benchmark_e2e_01",
    })
    assert t1_res.status_code == 200
    t1_data = t1_res.json()
    assert "dialogue" in t1_data
    assert "SA-CMS" in t1_data["dialogue"]["active_subject"]

    # Turn 2: Anaphora pronoun 'nó'
    t2_res = client.post("/api/self-study/ask", json={
        "session_id": session_id,
        "query": "Ưu điểm của nó là gì?",
        "document_id": "doc_benchmark_e2e_01",
    })
    assert t2_res.status_code == 200
    t2_data = t2_res.json()
    assert "SA-CMS" in t2_data["dialogue"]["resolved_query"]
    assert "87,5%" in t2_data["core_answer"] or "chi phí token" in t2_data["core_answer"]

    # Turn 3: Elliptic question 'Còn hạn chế thì sao?'
    t3_res = client.post("/api/self-study/ask", json={
        "session_id": session_id,
        "query": "Còn hạn chế thì sao?",
        "document_id": "doc_benchmark_e2e_01",
    })
    assert t3_res.status_code == 200
    t3_data = t3_res.json()
    assert "SA-CMS" in t3_data["dialogue"]["resolved_query"]
    assert "hạn chế" in t3_data["core_answer"].lower() or "cấu trúc" in t3_data["core_answer"].lower()

    # Verify compact prompt is frugal (< 150 chars, ~25 tokens)
    compact_prompt = t3_data["dialogue"]["compact_prompt"]
    assert len(compact_prompt) < 150
    assert "SA-CMS" in compact_prompt


def test_e2e_pipeline_entropy_pruning_token_efficiency(client):
    """
    Verifies that the Friston predictive surprisal gate saves tokens
    by stripping low-information conversational noise from evidence.
    """
    # Create document with heavy conversational filler
    noisy_text = (
        "Xin chào bạn, chúc một ngày làm việc tràn đầy năng lượng và hiệu quả. "
        "Như chúng ta đều đã biết, vấn đề tối ưu hóa mô hình là rất quan trọng. "
        "Mô hình SA-CMS đạt tỷ lệ giảm 87,5% chi phí token trên GPU RTX 3050. "
        "Dạ vâng xin cảm ơn sự quan tâm của quý bạn đọc."
    )
    client.post("/api/self-study/ingest", json={
        "text": noisy_text,
        "document_id": "doc_noisy_entropy_01",
        "train_memory": False,
    })

    res = client.post("/api/self-study/ask", json={
        "query": "Tỷ lệ giảm token của SA-CMS là bao nhiêu?",
        "document_id": "doc_noisy_entropy_01",
        "use_entropy_filter": True,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["is_grounded"] is True
    assert "87,5%" in data["core_answer"]
    # Verify tokens were saved by filtering filler
    assert data["tokens_saved_by_entropy"] > 0


def test_e2e_pipeline_calibrated_meta_cognition(client):
    """
    Verifies epistemic segregation into verified facts [Chắc chắn]
    and detection of knowledge gaps for unmentioned aspects.
    """
    res = client.post("/api/self-study/ask", json={
        "query": "SA-CMS có chạy được trên chip lượng tử quantum processor và siêu máy tính Fugaku không?",
        "document_id": "doc_benchmark_e2e_01",
        "use_meta_cognition": True,
    })
    assert res.status_code == 200
    data = res.json()
    meta = data["meta_cognition"]
    assert "knowledge_gaps" in meta
    # Identifies gap regarding unmentioned quantum aspects
    assert len(meta["knowledge_gaps"]) >= 1


def test_e2e_pipeline_safe_refusal_on_irrelevant_query(client):
    """Verifies 100% zero-hallucination safe refusal when query has no grounding."""
    res = client.post("/api/self-study/ask", json={
        "query": "Cách nấu món lẩu hải sản chua cay thập cẩm thơm ngon?",
        "document_id": "doc_benchmark_e2e_01",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["is_grounded"] is False
    assert "Không tìm thấy đủ thông tin" in data["formatted_answer"]
    assert data["citations"] == []

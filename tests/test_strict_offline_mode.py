"""
Tests for 100% Strict Offline Mode Guarantee (Stage 5 — Task 5.1).
Simulates absolute network disconnection by monkeypatching socket connection calls.
Verifies that:
1. SA-CMS backbone model & tokenizer load from local files with local_files_only=True.
2. Ingestion of structured documents proceeds without network.
3. Hybrid QA pipeline answers grounded questions with citations and clean text.
4. Out-of-domain queries correctly trigger standard refusal.
5. Mode B (Memory-only) forward pass executes successfully without network.
"""

import os
import sys
import socket
import pytest
from pathlib import Path

# Force offline environment before importing libraries
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SA_CMS_OFFLINE_MODE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.web.local_model_loader import LocalModelLoader
from src.hybrid_qa.pipeline import HybridQAPipeline, QAMode
from src.hybrid_qa.document_store import DocumentStore
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.retriever import BM25Retriever
from src.hybrid_qa.evidence import EvidenceSelector
from src.hybrid_qa.refusal import RefusalController


@pytest.fixture(autouse=True)
def enforce_network_isolation(monkeypatch):
    """
    Strict socket monkeypatch: Any attempt to connect to a non-loopback network host
    will immediately raise an OSError, simulating complete physical internet disconnection.
    """
    orig_connect = socket.socket.connect

    def blocked_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        allowed_hosts = ("127.0.0.1", "localhost", "::1")
        if host not in allowed_hosts:
            raise OSError(f"STRICT OFFLINE TEST: Outbound network connection to {host} is strictly prohibited.")
        return orig_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    yield


def test_strict_offline_model_and_tokenizer_loading():
    """Verify local model and tokenizer load strictly from local storage."""
    loader = LocalModelLoader.get_instance()
    model, tokenizer = loader.load_offline_model(prefer_gpu=False)

    assert model is not None, f"Local model failed to load: {loader.model_error}"
    assert tokenizer is not None, "Local tokenizer failed to load."
    assert loader.model_loaded is True
    assert loader.model_error is None

    # Verify model is frozen
    for p in model.backbone.parameters():
        assert not p.requires_grad, "Backbone must be 100% frozen."

    # Verify CMS memory module exists
    assert hasattr(model, "cms")
    assert model.cms is not None


def test_strict_offline_qa_pipeline_grounding_and_refusal(tmp_path):
    """
    Verifies full lifecycle under strict offline socket blockage:
    - Ingest document
    - Retrieve via BM25
    - Parametric memory pass with gating & norms
    - Clean extractive grounded QA
    - Refusal on ungrounded query
    - Memory-only mode
    """
    store_dir = tmp_path / "offline_store"
    store = DocumentStore(store_dir=str(store_dir))
    chunker = DocumentChunker(chunk_size=256, chunk_overlap=32)
    retriever = BM25Retriever(k1=1.5, b=0.75)
    evidence_selector = EvidenceSelector(score_threshold=3.0, max_evidence=5, min_evidence=1)
    refusal_controller = RefusalController(min_evidence_score=3.0, min_evidence_count=1)

    loader = LocalModelLoader.get_instance()
    model, tokenizer = loader.load_offline_model(prefer_gpu=False)

    doc_text = (
        "# 1. Giới thiệu Kiến trúc SA-CMS\n\n"
        "Nghiên cứu Nested Learning đề xuất hệ thống bộ nhớ liên tục (SA-CMS). "
        "Mô hình backbone được lựa chọn là SmolLM2-135M với 134.5 triệu tham số. "
        "Toàn bộ mô hình backbone được giữ đóng băng 100% trong quá trình suy luận.\n\n"
        "# 2. Cấu hình Phần cứng và Siêu tham số\n\n"
        "Hệ thống hoạt động với 3 tầng bộ nhớ: Tầng 1 đoạn văn, Tầng 2 phần mục, Tầng 3 tài liệu. "
        "Tốc độ học cơ sở được cố định ở mức 0.01 với thuật toán tối ưu online."
    )

    doc_record = store.add_document(
        title="Nghiên cứu Kiến trúc SA-CMS",
        raw_text=doc_text,
        document_id="DOC_OFFLINE_001",
    )
    passages = chunker.chunk_document(doc_record)
    doc_record.passages = passages
    retriever.build_index(passages)

    pipeline = HybridQAPipeline(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        evidence_selector=evidence_selector,
        refusal_controller=refusal_controller,
        chunker=chunker,
        document_store=store,
        device=loader.device,
    )

    # 1. Answerable Query (Mode C Hybrid)
    res_answerable = pipeline.answer_question(
        question="Mô hình backbone được lựa chọn là gì và có bao nhiêu tham số?",
        mode=QAMode.HYBRID,
        language="vi",
    )

    assert not res_answerable.refused, "Query should be answered with evidence."
    assert len(res_answerable.citations) > 0, "Citations must be present."
    assert "SmolLM2-135M" in res_answerable.answer or "134.5 triệu" in res_answerable.answer
    assert not "_DOC_" in res_answerable.answer, "Raw token tags must be stripped."
    assert "memory_diagnostics" in res_answerable.metadata
    mem_diag = res_answerable.metadata["memory_diagnostics"]
    assert mem_diag["has_cms"] is True
    assert "g1" in mem_diag and "g2" in mem_diag and "g3" in mem_diag
    assert "l1_norm" in mem_diag and "l2_norm" in mem_diag and "l3_norm" in mem_diag

    # 2. Out-of-domain Unanswerable Query (Refusal Branch)
    res_unanswerable = pipeline.answer_question(
        question="Chi phí sản xuất tàu ngầm hạt nhân lớp Typhoon là bao nhiêu đô la Mỹ?",
        mode=QAMode.HYBRID,
        language="vi",
    )

    assert res_unanswerable.refused is True, "Out-of-domain query must be refused."
    assert res_unanswerable.refusal_message == "Tài liệu được cung cấp không chứa thông tin để trả lời câu hỏi này."
    assert res_unanswerable.answer == "", "Answer must be empty when refused."
    assert len(res_unanswerable.citations) == 0, "No citations when refused."

    # 3. Memory-Only Mode (Mode B)
    res_memory = pipeline.answer_question(
        question="Mô hình SA-CMS có mấy tầng bộ nhớ?",
        mode=QAMode.MEMORY,
        language="vi",
    )
    assert res_memory.mode == "memory"
    assert "memory_diagnostics" in res_memory.metadata
    assert res_memory.metadata["memory_diagnostics"]["has_cms"] is True

"""
Comprehensive Unit Test Suite for Advanced RAG Mechanisms:
1. Multi-level Persistent Memory & Parent-Child Retriever
2. Context Compression & Token Minimization (LLMLingua-style Entity Protection)
3. Re-ranking with Cross-Encoder (Joint Contextual Self-Attention Scorer)
4. Query Transformation (Query Expansion & HyDE: Hypothetical Document Embeddings)
5. Graph RAG & Knowledge Hypergraph (Entity-Relation Extraction & Multi-Hop BFS)
"""

import numpy as np
import pytest

from src.hybrid_qa.document_store import DocumentRecord, Passage
from src.hybrid_qa.parent_child_retriever import ParentChildRetriever
from src.hybrid_qa.context_compression import ContextCompressor
from src.hybrid_qa.reranker import CrossEncoderReranker
from src.hybrid_qa.retriever import RetrievalResult
from src.hybrid_qa.hyde import HyDEEngine
from src.reasoning.hypergraph import ConceptHypergraph, RelationType


# ==============================================================================
# TEST 1: PARENT-CHILD RETRIEVER (Structure-Aware Micro-to-Macro Retrieval)
# ==============================================================================

def test_parent_child_retriever():
    text = (
        "1. KIẾN TRÚC MÔ HÌNH HỌC SÂU.\n"
        "Transformer là kiến trúc nền tảng dựa hoàn toàn trên cơ chế Self-Attention. "
        "Mô hình SmolLM2-135M kế thừa cấu trúc này với 134.5 triệu tham số để chạy nhẹ nhàng.\n\n"
        "2. CƠ CHẾ BỘ NHỚ LIÊN TỤC.\n"
        "SA-CMS bổ sung bộ nhớ đa tầng giúp mô hình ghi nhớ tài liệu dài mà không vượt ngưỡng VRAM."
    )
    doc = DocumentRecord(
        document_id="DOC_ARCH",
        title="Kiến trúc AI",
        raw_text=text,
        version=1,
    )

    retriever = ParentChildRetriever()
    p_cnt, c_cnt = retriever.index_document(doc)
    assert p_cnt >= 2
    assert c_cnt >= 2

    # Query matches child sentence specifically, but retriever yields the entire parent block
    hits = retriever.retrieve("SmolLM2 134.5 triệu tham số", top_k_parents=1)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.passage_id.startswith("DOC_ARCH::PARENT_")
    assert "Self-Attention" in hit.text
    assert "134.5 triệu tham số" in hit.text
    assert hit.metadata.get("retrieval_mode") == "parent_child"


# ==============================================================================
# TEST 2: CONTEXT COMPRESSION (Token Minimization & Entity Protection)
# ==============================================================================

def test_context_compression():
    compressor = ContextCompressor(target_compression_ratio=0.4)
    raw_context = (
        "Như chúng ta đã biết, hệ thống mật mã AES-256 được ban hành vào năm 2001 "
        "bởi tổ chức NIST với tỷ lệ an toàn đạt 99.9%."
    )
    res = compressor.compress_passage(raw_context, query="AES-256 an toàn")
    assert res.tokens_saved > 0
    assert res.compression_ratio_pct > 0.0

    # Critical entities must never be removed
    comp_text = res.compressed_text
    assert "AES-256" in comp_text
    assert "2001" in comp_text
    assert "99.9%" in comp_text or "99" in comp_text


# ==============================================================================
# TEST 3: RE-RANKING WITH CROSS-ENCODER
# ==============================================================================

def test_cross_encoder_reranker():
    reranker = CrossEncoderReranker()
    c1 = RetrievalResult(
        passage_id="P1",
        document_id="D1",
        text="Ẩm thực Việt Nam với phở bò và bánh mì nổi tiếng thế giới.",
        score=5.0,  # initially higher
    )
    c2 = RetrievalResult(
        passage_id="P2",
        document_id="D1",
        text="Mô hình SmolLM2-135M có cấu hình 134.5 triệu tham số chạy trên GPU RTX 3050.",
        score=3.0,  # initially lower
    )

    # Cross-encoder re-evaluates deep query-passage joint semantics
    reranked = reranker.rerank("SmolLM2 có bao nhiêu tham số trên GPU?", [c1, c2], top_k=1)
    assert len(reranked) == 1
    # P2 must be boosted to top 1
    assert reranked[0].passage_id == "P2"
    assert "cross_encoder_score" in reranked[0].metadata


# ==============================================================================
# TEST 4: QUERY TRANSFORMATION (HyDE: Hypothetical Document Embeddings)
# ==============================================================================

def test_hyde_query_transformation():
    hyde = HyDEEngine(blend_weight=0.7)
    query = "Hệ thống mật mã RSA là gì?"
    blended_vector, hypo_doc = hyde.encode_transformed_query(query, language="vi")

    assert blended_vector.shape == (384,)
    assert abs(np.linalg.norm(blended_vector) - 1.0) < 1e-4
    assert len(hypo_doc) > 20
    assert "RSA" in hypo_doc or "khái niệm" in hypo_doc or "thuật toán" in hypo_doc


# ==============================================================================
# TEST 5: GRAPH RAG (Concept Hypergraph & Multi-hop Reasoning - Verified Existing)
# ==============================================================================

def test_graph_rag_existing_hypergraph():
    graph = ConceptHypergraph()
    # Add nodes and causal relationship: A causes B, B leads to C
    graph.add_edge(
        source="Bi-Encoder",
        target="Fast Vector Retrieval",
        relation_type=RelationType.CAUSAL,
        evidence_text="Bi-Encoder cho phép mã hóa nhanh độc lập để truy xuất vector dưới 1 mili-giây.",
        document_id="DOC_A",
    )
    graph.add_edge(
        source="Fast Vector Retrieval",
        target="Low Latency Chatbot",
        relation_type=RelationType.SOLUTION,
        evidence_text="Truy xuất vector nhanh giúp giảm độ trễ phản hồi của hệ thống chatbot.",
        document_id="DOC_B",
    )

    # BFS multi-hop pathfinding across documents
    edges = graph.find_path("Bi-Encoder", "Low Latency Chatbot", max_hops=3)
    assert edges is not None
    assert len(edges) == 2
    docs = {e.document_id for e in edges}
    assert "DOC_A" in docs
    assert "DOC_B" in docs

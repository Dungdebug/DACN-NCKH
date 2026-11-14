"""
Comprehensive Unit Test Suite verifying the 5 Core RAG Mechanisms:
1. Document Chunking (Semantic Splitting & Fixed-size with Overlap)
2. Text Embedding (Bi-Encoder all-MiniLM-L6-v2, 384-d vectors, L2 norm, Cosine Similarity)
3. Information Retrieval (Hybrid Search combining HNSW Vector Search & BM25 via RRF)
4. Text Generation (Autoregressive Token Generation with Causal Self-Attention on SLMs)
5. Precision Control & Prompt Engineering (Strict Grounding, Anti-Hallucination, Citation)
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.document_store import DocumentRecord, Passage
from src.hybrid_qa.embedding import BiEncoderEmbedder
from src.hybrid_qa.vector_search import HNSWIndex, cosine_similarity, cosine_distance
from src.hybrid_qa.hybrid_retriever import HybridRetriever
from src.hybrid_qa.prompt_template import (
    GroundedPromptTemplate,
    STRICT_GROUNDING_VI,
    ANTI_HALLUCINATION_VI,
    CITATION_VI,
)
from src.inference.generator import TextGenerator, SUPPORTED_SLMS


# ==============================================================================
# TEST 1: DOCUMENT CHUNKING (Semantic Splitting & Fixed-size with Overlap)
# ==============================================================================

def test_document_chunking_semantic_and_fixed():
    sample_text = (
        "=== Page 1 ===\n"
        "1. TỔNG QUAN HỆ THỐNG MÃ HOÁ\n"
        "Hệ thống mã hoá đối xứng sử dụng cùng một khóa bí mật cho cả mã hoá và giải mã. "
        "Thuật toán AES và DES là hai ví dụ tiêu biểu trong lĩnh vực bảo mật thông tin.\n\n"
        "2. HỆ MÃ KHOÁ CÔNG KHAI\n"
        "Mã hoá bất đối xứng sử dụng một cặp khóa gồm khóa công khai và khóa bí mật. "
        "Thuật toán RSA là giải pháp hàng đầu cho chữ ký số và xác thực điện tử."
    )
    doc = DocumentRecord(document_id="DOC_TEST_001", title="Mật Mã Học", raw_text=sample_text, version=1)

    # 1. Semantic Splitting (theo câu, tiêu đề và ranh giới đoạn)
    chunker_semantic = DocumentChunker(chunk_size=120, chunk_overlap=20, strategy="semantic")
    passages_semantic = chunker_semantic.chunk_document(doc)
    assert len(passages_semantic) >= 2
    # Heading boundaries must be preserved cleanly
    section_titles = [p.section_title for p in passages_semantic if p.section_title]
    assert any("TỔNG QUAN" in t or "1." in t for t in section_titles)
    assert any("HỆ MÃ KHOÁ" in t or "2." in t for t in section_titles)

    # 2. Token-aware Fixed-size with Overlap
    chunker_token_fixed = DocumentChunker(
        chunk_size=300,
        chunk_overlap=50,
        strategy="fixed",
        use_tokens=True,
    )
    passages_token = chunker_token_fixed.chunk_document(doc)
    assert len(passages_token) >= 1
    assert passages_token[0].passage_id.startswith("DOC_TEST_001::P")


# ==============================================================================
# TEST 2: TEXT EMBEDDING (Bi-Encoder all-MiniLM-L6-v2 & Cosine Similarity)
# ==============================================================================

def test_bi_encoder_embedding():
    embedder = BiEncoderEmbedder(dimension=384)
    assert embedder.dimension == 384

    texts = [
        "Hệ thống mật mã khoá công khai RSA",
        "Thuật toán mã hoá bất đối xứng và chữ ký số",
        "Công thức làm món bánh mì nướng bơ tỏi",
    ]
    vectors = embedder.encode(texts)
    assert vectors.shape == (3, 384)

    # Check L2 Unit Norm
    for i in range(3):
        norm = np.linalg.norm(vectors[i])
        assert abs(norm - 1.0) < 1e-4

    # Cosine similarity between related security texts vs baking recipe
    sim_related = BiEncoderEmbedder.compute_similarity(vectors[0], vectors[1])
    sim_unrelated = BiEncoderEmbedder.compute_similarity(vectors[0], vectors[2])
    assert isinstance(sim_related, float)
    assert isinstance(sim_unrelated, float)


# ==============================================================================
# TEST 3: INFORMATION RETRIEVAL (HNSW Vector Search & Hybrid RRF Search)
# ==============================================================================

def test_hnsw_vector_and_hybrid_search():
    # 1. Test HNSW Vector Index directly
    hnsw = HNSWIndex(dimension=8, m=4, m0=8)
    v1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    v2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    v3 = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=float)

    hnsw.insert("DOC1::P001", v1)
    hnsw.insert("DOC1::P002", v2)
    hnsw.insert("DOC1::P003", v3)

    results = hnsw.search(v1, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "DOC1::P001"
    assert abs(results[0][1] - 1.0) < 1e-4
    assert results[1][0] == "DOC1::P003"

    # 2. Test Hybrid Search (HNSW Dense + BM25 Sparse via RRF)
    hybrid_retriever = HybridRetriever(vector_weight=0.5, rrf_k=60)
    p1 = Passage(
        passage_id="DOC_NET::P001",
        document_id="DOC_NET",
        text="Giao thức truyền thông bảo mật SSL/TLS bảo vệ dữ liệu trên Internet.",
        start_char=0,
        end_char=68,
        section_title="Giao thức Mạng",
    )
    p2 = Passage(
        passage_id="DOC_NET::P002",
        document_id="DOC_NET",
        text="Mô hình ngôn ngữ nhỏ SLM như SmolLM2-135M và Phi-3 chạy tối ưu trên thiết bị biên.",
        start_char=70,
        end_char=150,
        section_title="Mô hình Ngôn ngữ",
    )
    hybrid_retriever.build_index([p1, p2])

    hits = hybrid_retriever.retrieve("SmolLM2 mô hình ngôn ngữ", top_k=2)
    assert len(hits) == 2
    # p2 matches both keywords and semantics
    assert hits[0].passage_id == "DOC_NET::P002"
    assert "fusion" in hits[0].metadata


# ==============================================================================
# TEST 4: TEXT GENERATION (Autoregressive Causal Self-Attention on SLMs)
# ==============================================================================

def test_autoregressive_generation_and_slm_profiles():
    # Verify SLM Profiles catalog
    assert "smollm2_135m" in SUPPORTED_SLMS
    assert "qwen2_5_1_5b" in SUPPORTED_SLMS
    assert "phi3_mini" in SUPPORTED_SLMS
    assert "llama3_2_1b" in SUPPORTED_SLMS

    # Verify Autoregressive Loop
    class DummyCausalLM(nn.Module):
        def __init__(self, vocab_size=50):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, 16)
            self.head = nn.Linear(16, vocab_size)

        def forward(self, input_ids):
            x = self.emb(input_ids)
            logits = self.head(x)
            return logits

    dummy_model = DummyCausalLM()
    generator = TextGenerator(dummy_model)
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)

    # Greedy autoregression
    gen_tokens = generator.generate(
        prompt,
        max_new_tokens=5,
        temperature=0.0,
        repetition_penalty=1.1,
    )
    assert gen_tokens.shape == (1, 8)
    assert gen_tokens[0, :3].tolist() == [1, 2, 3]


# ==============================================================================
# TEST 5: PROMPT ENGINEERING (Strict Grounding, Anti-Hallucination, Citation)
# ==============================================================================

def test_grounded_prompt_template_constraints():
    question = "Hệ mã AES sử dụng kích thước khoá bao nhiêu bit?"
    passages = [
        {
            "passage_id": "CRYPTO::P005",
            "section_title": "Chuẩn Mã Hoá Nâng Cao AES",
            "text": "Tiêu chuẩn mã hoá AES hỗ trợ các kích thước khoá 128, 192 và 256 bit.",
        }
    ]

    # 1. Plain Text Format
    plain_prompt = GroundedPromptTemplate.build_prompt(
        question=question,
        context_passages=passages,
        language="vi",
        format_style="plain",
    )
    assert STRICT_GROUNDING_VI in plain_prompt
    assert ANTI_HALLUCINATION_VI in plain_prompt
    assert CITATION_VI in plain_prompt
    assert "[CRYPTO::P005]" in plain_prompt
    assert question in plain_prompt

    # 2. ChatML Format (for Qwen-2.5-1.5B)
    chatml_prompt = GroundedPromptTemplate.build_prompt(
        question=question,
        context_passages=passages,
        language="vi",
        format_style="chatml",
    )
    assert "<|im_start|>system" in chatml_prompt
    assert "<|im_start|>user" in chatml_prompt
    assert "<|im_start|>assistant" in chatml_prompt

    # 3. Llama-3 Format
    llama_prompt = GroundedPromptTemplate.build_prompt(
        question=question,
        context_passages=passages,
        language="vi",
        format_style="llama3",
    )
    assert "<|begin_of_text|><|start_header_id|>system" in llama_prompt
    assert "<|start_header_id|>assistant" in llama_prompt

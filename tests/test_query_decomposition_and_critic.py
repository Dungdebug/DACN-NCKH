"""
Tests for Trụ cột 2 & 3:
- Neural Query Decomposer (src/reasoning/query_decomposer.py)
- Reflective Grounding Critic (src/hybrid_qa/grounding_critic.py)
"""

import pytest

from src.reasoning.query_decomposer import QueryDecomposer, SubQuery, DecompositionResult
from src.hybrid_qa.grounding_critic import GroundingCritic, ClaimVerificationResult


def test_query_decomposer_single_query():
    """Simple query should not be decomposed."""
    decomposer = QueryDecomposer()
    res = decomposer.decompose("SmolLM2 có bao nhiêu tham số?")
    assert res.is_compound is False
    assert len(res.sub_queries) == 1
    assert res.sub_queries[0].text == "SmolLM2 có bao nhiêu tham số?"


def test_query_decomposer_conjunction_and():
    """Compound query with 'và' / 'đồng thời' should be split cleanly."""
    decomposer = QueryDecomposer()
    query = "Nhiệm vụ của người đảng viên là gì và các biện pháp thực hiện ra sao?"
    res = decomposer.decompose(query)
    assert res.is_compound is True
    assert len(res.sub_queries) == 2
    assert "nhiệm vụ của người đảng viên" in res.sub_queries[0].text.lower()
    assert "biện pháp thực hiện" in res.sub_queries[1].text.lower()


def test_query_decomposer_comparison():
    """Comparison queries should be split into target entities."""
    decomposer = QueryDecomposer()
    query = "So sánh sự khác biệt giữa mô hình SA-CMS và baseline CMS"
    res = decomposer.decompose(query)
    assert res.is_compound is True
    assert len(res.sub_queries) >= 2
    assert any("sa-cms" in q.text.lower() for q in res.sub_queries)
    assert any("baseline cms" in q.text.lower() for q in res.sub_queries)


def test_grounding_critic_supported_claims():
    """Critic should verify claims when they are directly supported by citations."""
    critic = GroundingCritic()
    citations = [
        {
            "citation_index": 1,
            "document_title": "Báo cáo AI",
            "text": "Mô hình SmolLM2 có 135 triệu tham số và được huấn luyện trên 2 nghìn tỷ token.",
        }
    ]
    answer = "Mô hình SmolLM2 có 135 triệu tham số."
    report = critic.verify_answer(answer, citations)
    assert report.is_faithful is True
    assert report.faithfulness_score >= 0.8
    assert len(report.unsupported_claims) == 0


def test_grounding_critic_detects_unsupported_hallucination():
    """Critic should flag claims with unsupported numbers/entities not found in citations."""
    critic = GroundingCritic()
    citations = [
        {
            "citation_index": 1,
            "document_title": "Báo cáo AI",
            "text": "Mô hình SmolLM2 có 135 triệu tham số.",
        }
    ]
    # Answer hallucinates 99.9% accuracy and 500ms latency not in citation
    answer = "SmolLM2 có 135 triệu tham số. Độ chính xác đạt 99.9% với độ trễ 500ms."
    report = critic.verify_answer(answer, citations)
    assert len(report.unsupported_claims) >= 1
    # Reflective correction should prune or flag the unsupported claim
    corrected = critic.correct_answer(answer, report)
    assert "99.9%" not in corrected or "không có trong tài liệu" in corrected.lower()


def test_query_decomposition_in_e2e_session():
    """E2E test: verifies compound query decomposes and retrieves evidence without errors."""
    from fastapi.testclient import TestClient
    from src.web.app import app

    client = TestClient(app)

    # 1. Create session
    create_resp = client.post("/api/chat/sessions", json={"title": "Test Decomposer"})
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    # 2. Ask compound query
    chat_resp = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={
            "query": "Quy định về thời gian công tác là gì và các biện pháp thực hiện gồm những gì?",
            "answer_mode": "balanced",
            "top_k": 3,
        },
    )
    assert chat_resp.status_code == 200
    data = chat_resp.json()
    assert "answer" in data
    assert len(data["answer"]) > 10


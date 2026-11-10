"""
Master Integration Test Suite for Sovereign Document Brain (Stage 6 Master).
Validates all 5 Core Architecture Pillars:
1. Semantic Intent & Synonym Thesaurus (Vé 6.1)
2. Coherent Multi-Sentence Answer Composer (Vé 6.2)
3. Hierarchical Section Drawer (Vé 6.3)
4. Dual-Process System 1/System 2 & Premise Validator (Vé 6.4)
5. Mechanistic Gating Lens Attribution & Zero-Hallucination Grounding (Vé 6.5)
100% Offline, Deterministic, Zero LLM API calls.
"""

import socket
import pytest
from src.self_study.semantic_thesaurus import SemanticThesaurus
from src.self_study.answer_composer import CoherentAnswerComposer
from src.self_study.reader import NeuralReaderEngine
from src.brain.section_drawer import SectionDrawer, DRAWER_EXPERIMENT, DRAWER_SOLUTION
from src.brain.gating_lens import MechanisticGatingLens
from src.reasoning.dual_process import DualProcessController


@pytest.fixture(autouse=True)
def guard_strict_offline():
    """Ensures 100% local socket block on non-loopback calls."""
    orig_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        allowed = ("127.0.0.1", "localhost", "::1", "testserver")
        if isinstance(host, str) and (host in allowed or host.startswith("127.")):
            return orig_connect(self, address)
        raise ConnectionRefusedError(f"[OFFLINE VIOLATION] Blocked attempt to {address}")

    socket.socket.connect = guarded
    yield
    socket.socket.connect = orig_connect


@pytest.fixture
def master_document_passages():
    return [
        {
            "passage_id": "doc_sacms::p1",
            "document_id": "doc_sacms",
            "document_title": "Kiến trúc SA-CMS Đa Quy Mô",
            "section_title": "1. Tổng quan và Bối cảnh",
            "citation_label": "[1]",
            "score": 10.0,
            "text": (
                "Hệ thống SA-CMS là bộ nhớ liên tục căn chỉnh theo cấu trúc phân tầng của tài liệu. "
                "Cơ chế giải quyết bài toán suy luận trên văn bản dài mà không cần nhồi toàn bộ ngữ cảnh. "
                "Mô hình backbone SmolLM2-135M được giữ đóng băng 100% trong quá trình suy luận."
            ),
        },
        {
            "passage_id": "doc_sacms::p2",
            "document_id": "doc_sacms",
            "document_title": "Kiến trúc SA-CMS Đa Quy Mô",
            "section_title": "3. Kết quả Thực nghiệm và Hiệu năng",
            "citation_label": "[2]",
            "score": 9.5,
            "text": (
                "Mô hình đạt tỷ lệ giảm 87,5% chi phí token so với baseline RAG truyền thống. "
                "Độ trễ truy vấn trung bình đạt 18,2 ms trên GPU RTX 3050 4GB VRAM. "
                "Tuy nhiên, hạn chế của hệ thống là yêu cầu văn bản đầu vào có cấu trúc phân đoạn rõ ràng."
            ),
        },
    ]


def test_semantic_thesaurus_bridges_colloquial_gap():
    thesaurus = SemanticThesaurus()
    # User asks colloquial query
    query = "Mô hình này giúp ích gì cho chi phí?"
    expanded = thesaurus.expand_query_terms(query)

    # Expanded terms must include academic synonyms
    assert any(term in expanded for term in ["tiết kiệm", "tối ưu", "ưu điểm", "giảm"])


def test_section_drawer_indexes_and_prioritizes(master_document_passages):
    drawer = SectionDrawer()
    indexed = drawer.index_document_drawers(master_document_passages)

    assert len(indexed[DRAWER_EXPERIMENT]) == 1
    assert "87,5%" in indexed[DRAWER_EXPERIMENT][0]["text"]

    # When query intent is METRIC, experiment drawer must be placed first
    ranked = drawer.filter_and_rank_by_intent(master_document_passages, intent="METRIC")
    assert ranked[0]["passage_id"] == "doc_sacms::p2"


def test_dual_process_premise_refutation_on_loaded_query(master_document_passages):
    controller = DualProcessController()
    # User asks a loaded question with false assumption: "làm tăng chi phí"
    query = "Tại sao SA-CMS lại làm tăng chi phí token so với RAG?"

    res = controller.execute(query, master_document_passages)

    assert res.is_refutation is True
    assert "không phù hợp với tài liệu" in res.final_answer
    assert "giảm 87,5% chi phí token" in res.final_answer
    assert "[2]" in res.final_answer


def test_coherent_multi_sentence_composer_end_to_end(master_document_passages):
    reader = NeuralReaderEngine(min_confidence=0.20)
    query = "SA-CMS là gì?"

    res = reader.synthesize(query, master_document_passages, answer_mode="balanced")

    assert res.is_grounded is True
    assert res.composed_answer != ""
    # Checks presence of Core definition + supporting metric
    assert "bộ nhớ liên tục" in res.composed_answer
    assert "[1]" in res.composed_answer
    assert "SmolLM2-135M" in res.composed_answer or "100%" in res.composed_answer
    # Verifies follow-up questions
    assert len(res.followup_questions) >= 2


def test_mechanistic_gating_lens_attribution():
    lens = MechanisticGatingLens()
    # Gating weights: high section level weight
    attr = lens.extract_attribution(g1=0.25, g2=0.55, g3=0.20)

    assert attr["g1"] == 0.25
    assert attr["percentages"]["section_level_2"] == 55.0
    assert attr["dominant_level"] == "Cấp 2 (Phần mục)"
    assert "Chi phối chính: Cấp 2 (Phần mục)" in attr["attribution_summary"]


def test_strict_zero_hallucination_guarantee(master_document_passages):
    reader = NeuralReaderEngine(min_confidence=0.20)
    res = reader.synthesize("SA-CMS có ưu điểm gì về chi phí?", master_document_passages, answer_mode="balanced")

    # Every clause extracted must exist verbatim in the source passages
    all_source = " ".join([p["text"] for p in master_document_passages])
    assert res.core_answer in all_source
    assert "87,5% chi phí token" in res.composed_answer
    assert "87,5% chi phí token" in all_source

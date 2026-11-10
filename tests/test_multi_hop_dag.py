"""
Unit tests for Neuro-Symbolic Multi-Hop Evidence DAG (Ticket 7.1).
Verifies evidence graph construction, directed edge formation,
optimal reasoning path search, and multi-hop narrative composition.
"""

import pytest
from src.reasoning.multi_hop_dag import (
    MultiHopEvidenceDAG,
    EvidenceNode,
    EvidenceEdge,
    EvidenceChain,
)


@pytest.fixture
def sample_passages():
    return [
        {
            "passage_id": "p_problem",
            "document_id": "DOC_TEST",
            "section_title": "1. Đặt vấn đề",
            "citation_label": "[1]",
            "text": (
                "Các mô hình LLM truyền thống gặp vấn đề chi phí token cực lớn khi xử lý tài liệu dài. "
                "Nguyên nhân cốt lõi là do cơ chế Self-Attention phải tính toán ma trận tương quan bậc hai O(N^2)."
            ),
        },
        {
            "passage_id": "p_solution",
            "document_id": "DOC_TEST",
            "section_title": "2. Phương pháp SA-CMS",
            "citation_label": "[2]",
            "text": (
                "Để giải quyết vấn đề này, kiến trúc SA-CMS đề xuất cập nhật bộ nhớ liên tục căn theo ranh giới cấu trúc văn bản. "
                "Cơ chế này sử dụng Hope-Attention kết hợp phân tầng Level 1, Level 2 và Level 3 nhằm nén thông tin trực tiếp vào trọng số tham số."
            ),
        },
        {
            "passage_id": "p_experiment",
            "document_id": "DOC_TEST",
            "section_title": "3. Kết quả thực nghiệm",
            "citation_label": "[3]",
            "text": (
                "Kết quả thực nghiệm trên mô hình SmolLM2-135M chứng minh SA-CMS giúp giảm 87,5% chi phí token so với RAG tiêu chuẩn. "
                "Tuy nhiên, phương pháp này đòi hỏi chi phí tính toán ban đầu lớn hơn trong giai đoạn nạp văn bản (ingestion)."
            ),
        },
    ]


def test_build_dag_nodes_and_roles(sample_passages):
    dag = MultiHopEvidenceDAG()
    graph = dag.build_dag(sample_passages, query="SA-CMS hoạt động như thế nào và đạt hiệu quả gì?")
    
    assert len(graph.nodes) >= 3
    # Check that roles are assigned
    roles = {node.role for node in graph.nodes.values()}
    assert "mechanism" in roles or "cause" in roles
    assert "metric" in roles or "limitation" in roles


def test_directed_edge_formation(sample_passages):
    dag = MultiHopEvidenceDAG()
    graph = dag.build_dag(sample_passages, query="Cơ chế của SA-CMS và kết quả giảm chi phí token")
    
    # Must have directed edges linking mechanism/solution to experiment/metric
    assert len(graph.edges) > 0
    edge_types = {e.edge_type for e in graph.edges}
    assert any(et in ("quantifies", "causes_or_leads_to", "elaborates") for et in edge_types)


def test_find_optimal_chain_2_hops(sample_passages):
    dag = MultiHopEvidenceDAG()
    graph = dag.build_dag(sample_passages, query="SA-CMS giúp giảm chi phí bao nhiêu?")
    chain = dag.find_optimal_evidence_chain(graph, max_hops=2)
    
    assert chain is not None
    assert len(chain.nodes) >= 2
    # The chain should contain both the mechanism and the metric
    roles = [n.role for n in chain.nodes]
    assert any(r in ("mechanism", "cause") for r in roles)
    assert any(r in ("metric", "impact") for r in roles)


def test_find_optimal_chain_3_hops(sample_passages):
    dag = MultiHopEvidenceDAG()
    graph = dag.build_dag(sample_passages, query="Tại sao cần SA-CMS, cơ chế hoạt động ra sao và kết quả đạt được là gì?")
    chain = dag.find_optimal_evidence_chain(graph, max_hops=3)
    
    assert chain is not None
    assert len(chain.nodes) >= 2
    assert chain.cumulative_score > 0.0


def test_compose_chain_narrative(sample_passages):
    dag = MultiHopEvidenceDAG()
    graph = dag.build_dag(sample_passages, query="SA-CMS giải quyết vấn đề gì và đạt kết quả thực nghiệm thế nào?")
    chain = dag.find_optimal_evidence_chain(graph, max_hops=3)
    narrative = dag.compose_chain_narrative(chain)
    
    assert narrative is not None
    assert len(narrative) > 50
    # Must contain citations
    assert "[" in narrative and "]" in narrative
    # Must contain quantitative metric if present
    assert "87,5%" in narrative or "SA-CMS" in narrative


def test_graceful_fallback_single_node():
    single_passage = [
        {
            "passage_id": "p1",
            "document_id": "DOC_1",
            "section_title": "Khái niệm",
            "citation_label": "[1]",
            "text": "Bộ nhớ SA-CMS được cấu hình 3 cấp độ để lưu trữ tri thức văn bản.",
        }
    ]
    dag = MultiHopEvidenceDAG()
    graph = dag.build_dag(single_passage, query="Bộ nhớ SA-CMS là gì?")
    chain = dag.find_optimal_evidence_chain(graph, max_hops=3)
    narrative = dag.compose_chain_narrative(chain)
    
    assert chain is not None
    assert len(chain.nodes) == 1
    assert "SA-CMS" in narrative
    assert "[1]" in narrative


"""
Unit Tests for ConceptHypergraph & Multi-Hop Path Reasoning (Ticket 2.1).
Validates:
1. Node canonicalization and adjacency indexing.
2. Syntactic relation extraction across 5 semantic patterns.
3. Breadth-First Search (BFS) multi-hop pathfinding across document boundaries.
4. Logical synthesis generation with exact citation provenance.
5. Disconnected components handling (safe None return).
"""

import pytest
from src.reasoning.hypergraph import (
    ConceptHypergraph,
    RelationType,
    ConceptNode,
    RelationEdge,
    MultiHopChain,
)


def test_hypergraph_add_nodes_and_edges():
    graph = ConceptHypergraph()
    n1 = graph.add_node("Mô hình SA-CMS", node_type="method")
    n2 = graph.add_node("Bộ nhớ liên tục", node_type="entity")

    assert n1.canonical_id == "mo_hinh_sa_cms"
    assert n2.canonical_id == "bo_nho_lien_tuc"
    assert len(graph.nodes) == 2

    edge = graph.add_edge(
        source="Mô hình SA-CMS",
        target="Bộ nhớ liên tục",
        relation_type=RelationType.PART_OF,
        evidence_text="SA-CMS bao gồm bộ nhớ liên tục đa mức.",
        document_id="doc1",
    )

    assert edge is not None
    assert edge.relation_type == RelationType.PART_OF
    assert len(graph.edges) == 1
    assert len(graph.adj_list["mo_hinh_sa_cms"]) == 1

    # Self-loops must be rejected
    self_loop = graph.add_edge(
        source="Mô hình SA-CMS",
        target="Mô hình SA-CMS",
        relation_type=RelationType.CAUSAL,
        evidence_text="self loop test",
        document_id="doc1",
    )
    assert self_loop is None


def test_hypergraph_syntactic_relation_extraction():
    graph = ConceptHypergraph()
    sentences = [
        "Mô hình SA-CMS giải quyết bài toán suy luận văn bản dài.",
        "Bộ nhớ đa tầng dẫn đến việc tiết kiệm chi phí token.",
        "Kiến trúc hệ thống bao gồm ba mức bộ nhớ liên tục.",
        "SA-CMS vượt trội hơn mô hình RAG truyền thống.",
        "Tỷ lệ nén dữ liệu đạt mức 87,5% tổng số token.",
    ]

    count = graph.extract_from_sentences(
        sentences=sentences,
        document_id="de_cuong_doc",
        section_title="Phương pháp & Kết quả",
    )

    assert count >= 4
    stats = graph.get_statistics()
    assert stats["total_nodes"] >= 6
    assert stats["total_edges"] >= 4
    assert stats["relations_by_type"]["solution"] >= 1
    assert stats["relations_by_type"]["causal"] >= 1


def test_hypergraph_multi_hop_bfs_pathfinding():
    graph = ConceptHypergraph()

    # Document 1 (Chính sách): SA-CMS tối ưu hóa Chi phí vận hành
    graph.add_edge(
        source="SA-CMS",
        target="Chi phí vận hành",
        relation_type=RelationType.SOLUTION,
        evidence_text="SA-CMS tối ưu hóa chi phí vận hành hệ thống.",
        document_id="doc_policy",
    )

    # Document 2 (Kỹ thuật): Chi phí vận hành dẫn đến Ngân sách nghiên cứu
    graph.add_edge(
        source="Chi phí vận hành",
        target="Ngân sách nghiên cứu",
        relation_type=RelationType.CAUSAL,
        evidence_text="Chi phí vận hành giảm dẫn đến ngân sách nghiên cứu được bảo toàn.",
        document_id="doc_tech",
    )

    # Document 3 (Quản lý): Ngân sách nghiên cứu tạo điều kiện cho Mở rộng quy mô
    graph.add_edge(
        source="Ngân sách nghiên cứu",
        target="Mở rộng quy mô",
        relation_type=RelationType.CAUSAL,
        evidence_text="Ngân sách nghiên cứu dồi dào tạo điều kiện cho mở rộng quy mô phòng lab.",
        document_id="doc_management",
    )

    # Find 3-hop path: SA-CMS -> Mở rộng quy mô
    path = graph.find_path("SA-CMS", "Mở rộng quy mô", max_hops=4)
    assert path is not None
    assert len(path) == 3

    assert path[0].source_id == "sa_cms"
    assert path[0].target_id == "chi_phi_van_hanh"
    assert path[1].source_id == "chi_phi_van_hanh"
    assert path[1].target_id == "ngan_sach_nghien_cuu"
    assert path[2].source_id == "ngan_sach_nghien_cuu"
    assert path[2].target_id == "mo_rong_quy_mo"


def test_hypergraph_multi_hop_synthesis():
    graph = ConceptHypergraph()

    graph.add_edge(
        source="SA-CMS",
        target="Chi phí phần cứng",
        relation_type=RelationType.SOLUTION,
        evidence_text="SA-CMS giải quyết bài toán chi phí phần cứng đắt đỏ.",
        document_id="doc_a",
    )
    graph.add_edge(
        source="Chi phí phần cứng",
        target="Khả năng triển khai thực tế",
        relation_type=RelationType.CAUSAL,
        evidence_text="Chi phí phần cứng thấp dẫn đến khả năng triển khai thực tế tại bệnh viện.",
        document_id="doc_b",
    )

    chain = graph.synthesize_multi_hop_deduction(
        query="SA-CMS tác động thế nào đến triển khai thực tế?",
        start_concept="SA-CMS",
        target_concept="Khả năng triển khai thực tế",
    )

    assert chain is not None
    assert chain.hop_count == 2
    assert "doc_a" in chain.documents_involved
    assert "doc_b" in chain.documents_involved
    assert len(chain.reasoning_steps) == 2
    assert "Bước 1:" in chain.reasoning_steps[0]
    assert "Bước 2:" in chain.reasoning_steps[1]
    assert len(chain.citations) == 2


def test_hypergraph_no_path_returns_none():
    graph = ConceptHypergraph()
    graph.add_node("Cụm A", node_type="entity")
    graph.add_node("Cụm B", node_type="entity")

    path = graph.find_path("Cụm A", "Cụm B")
    assert path is None

    chain = graph.synthesize_multi_hop_deduction(
        query="Liên hệ giữa A và B?",
        start_concept="Cụm A",
        target_concept="Cụm B",
    )
    assert chain is None


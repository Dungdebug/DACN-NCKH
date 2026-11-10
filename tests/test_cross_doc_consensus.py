"""
Unit tests for Cross-Document Consensus & Divergence Matrix (Ticket 7.3).
Verifies detection of shared consensus, numerical divergence,
contradiction scanning across multiple documents, and Markdown table output.
"""

import pytest
from src.reasoning.cross_doc_consensus import (
    CrossDocConsensusEngine,
    ConsensusPoint,
    DivergencePoint,
    CrossDocAnalysisReport,
)


@pytest.fixture
def multi_doc_corpus():
    return [
        {
            "document_id": "DOC_A",
            "title": "Báo cáo Kỹ thuật SA-CMS 2024",
            "raw_text": (
                "Kiến trúc SA-CMS sử dụng cơ chế cập nhật bộ nhớ liên tục căn theo cấu trúc. "
                "Thực nghiệm cho thấy SA-CMS giúp giảm 87,5% chi phí token trên tác vụ tài liệu dài. "
                "Mô hình SmolLM2-135M chạy ổn định trên phần cứng GPU 4GB VRAM."
            ),
        },
        {
            "document_id": "DOC_B",
            "title": "Đánh giá Độc lập Baseline 2025",
            "raw_text": (
                "Nghiên cứu này cũng xác nhận SA-CMS sử dụng cơ chế cập nhật bộ nhớ liên tục căn theo cấu trúc. "
                "Tuy nhiên, khi thử nghiệm trên tập dữ liệu tổng quát, SA-CMS chỉ giảm 75% chi phí token. "
                "Phương pháp đòi hỏi tài nguyên tính toán lớn trong pha ingestion."
            ),
        },
    ]


def test_detect_consensus_point(multi_doc_corpus):
    engine = CrossDocConsensusEngine()
    report = engine.analyze_cross_documents(multi_doc_corpus)
    
    assert len(report.consensus_points) >= 1
    consensus = report.consensus_points[0]
    assert "cập nhật bộ nhớ liên tục" in consensus.statement.lower()
    assert "DOC_A" in consensus.document_supports
    assert "DOC_B" in consensus.document_supports


def test_detect_numerical_divergence(multi_doc_corpus):
    engine = CrossDocConsensusEngine()
    report = engine.analyze_cross_documents(multi_doc_corpus)
    
    assert len(report.divergence_points) >= 1
    div = report.divergence_points[0]
    assert div.divergence_type == "numerical_difference"
    assert "87,5%" in div.claim_a
    assert "75%" in div.claim_b


def test_markdown_summary_table(multi_doc_corpus):
    engine = CrossDocConsensusEngine()
    report = engine.analyze_cross_documents(multi_doc_corpus)
    
    md = report.summary_markdown
    assert "| Điểm đối chiếu |" in md or "### Bảng đối chiếu" in md
    assert "87,5%" in md
    assert "75%" in md
    assert "Đồng thuận" in md or "Phân kỳ" in md


def test_single_document_graceful_handling():
    single_doc = [
        {
            "document_id": "DOC_SOLO",
            "title": "Tài liệu duy nhất",
            "raw_text": "Hệ thống hoạt động với bộ nhớ liên tục đơn lẻ.",
        }
    ]
    engine = CrossDocConsensusEngine()
    report = engine.analyze_cross_documents(single_doc)
    
    assert len(report.consensus_points) == 0
    assert len(report.divergence_points) == 0
    assert "Cần tối thiểu 2 tài liệu" in report.summary_markdown


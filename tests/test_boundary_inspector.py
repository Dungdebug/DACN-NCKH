"""
Unit tests for Counterfactual Prober & Boundary Inspector (Ticket 7.2).
Verifies boundary condition detection, trade-off clause extraction,
absolute assumption challenging, and advisory synthesis.
"""

import pytest
from src.reasoning.boundary_inspector import (
    BoundaryInspector,
    BoundaryClause,
    BoundaryAdvisory,
)


@pytest.fixture
def sample_passages_with_boundaries():
    return [
        {
            "passage_id": "p1",
            "document_id": "DOC_BOUNDARY",
            "section_title": "Phương pháp và Giới hạn",
            "citation_label": "[1]",
            "text": (
                "Kiến trúc SA-CMS giúp giảm 87,5% chi phí token. "
                "Tuy nhiên, phương pháp này chỉ phát huy hiệu quả tối đa trong điều kiện văn bản dài trên 2048 tokens. "
                "Đối với các đoạn văn bản quá ngắn dưới 512 tokens, chi phí cập nhật bộ nhớ có thể vượt qua lợi ích tiết kiệm."
            ),
        },
        {
            "passage_id": "p2",
            "document_id": "DOC_BOUNDARY",
            "section_title": "Thực nghiệm phần cứng",
            "citation_label": "[2]",
            "text": (
                "Thực nghiệm được thực hiện trên GPU NVIDIA RTX 3050 4GB VRAM. "
                "Đổi lại, việc huấn luyện Level 3 đòi hỏi thời gian nạp ban đầu (ingestion latency) lâu hơn 15% so với BM25 thuần."
            ),
        },
    ]


def test_extract_boundary_clauses(sample_passages_with_boundaries):
    inspector = BoundaryInspector()
    clauses = inspector.extract_boundaries(sample_passages_with_boundaries)
    
    assert len(clauses) >= 2
    categories = {c.category for c in clauses}
    assert "scope_limit" in categories or "trade_off" in categories
    assert any("2048 tokens" in c.condition_text or "dưới 512 tokens" in c.condition_text for c in clauses)


def test_detect_absolute_overgeneralization_query():
    inspector = BoundaryInspector()
    assert inspector.is_absolute_query("SA-CMS luôn luôn giảm chi phí token trong mọi trường hợp phải không?") is True
    assert inspector.is_absolute_query("Mô hình này hoàn toàn không có bất kỳ nhược điểm nào đúng không?") is True
    assert inspector.is_absolute_query("SA-CMS giảm bao nhiêu phần trăm chi phí token?") is False


def test_generate_counter_boundary_advisory(sample_passages_with_boundaries):
    inspector = BoundaryInspector()
    query = "SA-CMS luôn luôn hiệu quả trong mọi trường hợp văn bản phải không?"
    advisory = inspector.inspect_and_advise(sample_passages_with_boundaries, query)
    
    assert advisory.has_boundary is True
    assert advisory.challenges_absolute is True
    assert "Không thể khẳng định trong mọi trường hợp" in advisory.advisory_text or "chỉ phát huy hiệu quả" in advisory.advisory_text
    assert "[1]" in advisory.advisory_text or "[2]" in advisory.advisory_text


def test_attach_advisory_to_base_answer(sample_passages_with_boundaries):
    inspector = BoundaryInspector()
    query = "Hiệu quả của SA-CMS như thế nào?"
    base_answer = "SA-CMS giúp giảm 87,5% chi phí token [1]."
    enhanced_answer = inspector.enhance_answer_with_boundaries(base_answer, sample_passages_with_boundaries, query)
    
    assert "87,5%" in enhanced_answer
    assert "Lưu ý về điều kiện biên" in enhanced_answer or "ràng buộc" in enhanced_answer
    assert "[1]" in enhanced_answer


def test_no_boundary_fallback():
    clean_passages = [
        {
            "passage_id": "p_clean",
            "document_id": "DOC_CLEAN",
            "section_title": "Giới thiệu",
            "citation_label": "[1]",
            "text": "Bộ nhớ SA-CMS được cấu hình gồm 3 cấp độ căn theo ranh giới cấu trúc văn bản.",
        }
    ]
    inspector = BoundaryInspector()
    query = "Bộ nhớ SA-CMS gồm mấy cấp độ?"
    base_answer = "Bộ nhớ SA-CMS gồm 3 cấp độ [1]."
    enhanced_answer = inspector.enhance_answer_with_boundaries(base_answer, clean_passages, query)
    
    # Should not invent fake boundaries if text has none
    assert enhanced_answer == base_answer


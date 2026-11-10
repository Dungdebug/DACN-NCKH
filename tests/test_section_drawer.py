"""
Unit Tests for Hierarchical Section Drawer (Ticket 6.3).
Validates:
1. Academic section title classification into 4 canonical drawers.
2. Ambiguous title fallback with representative text tokens.
3. Multi-passage drawer indexing.
4. Intent-guided drawer prioritization.
"""

import pytest
from src.brain.section_drawer import (
    SectionDrawer,
    DRAWER_PROBLEM,
    DRAWER_SOLUTION,
    DRAWER_EXPERIMENT,
    DRAWER_CONCLUSION,
    DRAWER_GENERIC,
)


@pytest.fixture
def drawer():
    return SectionDrawer()


def test_categorize_section_titles(drawer):
    assert drawer.categorize_section("1. Giới thiệu và Động lực Nghiên cứu") == DRAWER_PROBLEM
    assert drawer.categorize_section("2. Kiến trúc Hệ thống và Thuật toán SA-CMS") == DRAWER_SOLUTION
    assert drawer.categorize_section("3. Kết quả Thực nghiệm và Đánh giá Hiệu năng") == DRAWER_EXPERIMENT
    assert drawer.categorize_section("4. Kết luận, Hạn chế và Thảo luận") == DRAWER_CONCLUSION
    assert drawer.categorize_section("Introduction & Related Work") == DRAWER_PROBLEM
    assert drawer.categorize_section("Experimental Setup and Results") == DRAWER_EXPERIMENT


def test_categorize_generic_section_with_text_sample(drawer):
    # Title has no keywords ("Mục 3.2"), but text sample mentions benchmarks
    sec = drawer.categorize_section(
        section_title="Mục 3.2",
        text_sample="Mô hình đạt tỷ lệ giảm 87,5% chi phí token trên GPU RTX 3050 với độ trễ 18.2 ms.",
    )
    assert sec == DRAWER_EXPERIMENT


def test_index_document_drawers(drawer):
    passages = [
        {"passage_id": "p1", "section_title": "1. Tổng quan", "text": "Đề tài nghiên cứu bối cảnh mô hình ngôn ngữ lớn."},
        {"passage_id": "p2", "section_title": "2. Phương pháp", "text": "Thiết kế kiến trúc bộ nhớ đa tầng SA-CMS."},
        {"passage_id": "p3", "section_title": "3. Kết quả thực nghiệm", "text": "Tiết kiệm 87,5% token và giữ độ trễ 18,2ms."},
        {"passage_id": "p4", "section_title": "4. Thảo luận và Hạn chế", "text": "Ràng buộc về tài liệu có phân đoạn rõ ràng."},
    ]

    indexed = drawer.index_document_drawers(passages)
    assert len(indexed[DRAWER_PROBLEM]) == 1
    assert len(indexed[DRAWER_SOLUTION]) == 1
    assert len(indexed[DRAWER_EXPERIMENT]) == 1
    assert len(indexed[DRAWER_CONCLUSION]) == 1
    assert indexed[DRAWER_EXPERIMENT][0]["passage_id"] == "p3"


def test_filter_and_rank_by_intent(drawer):
    passages = [
        {"passage_id": "p1", "section_title": "1. Giới thiệu", "text": "Giới thiệu bối cảnh bài toán."},
        {"passage_id": "p2", "section_title": "2. Kiến trúc", "text": "Cơ chế hoạt động của bộ nhớ."},
        {"passage_id": "p3", "section_title": "3. Kết quả", "text": "Số liệu đo đạc thực nghiệm."},
        {"passage_id": "p4", "section_title": "4. Hạn chế", "text": "Hạn chế của mô hình."},
    ]

    # For METRIC intent, experiment drawer (p3) must be first
    metric_ranked = drawer.filter_and_rank_by_intent(passages, intent="METRIC")
    assert metric_ranked[0]["passage_id"] == "p3"

    # For LIMITATION intent, conclusion drawer (p4) must be first
    lim_ranked = drawer.filter_and_rank_by_intent(passages, intent="LIMITATION")
    assert lim_ranked[0]["passage_id"] == "p4"

    # For MECHANISM intent, solution drawer (p2) must be first
    mech_ranked = drawer.filter_and_rank_by_intent(passages, intent="MECHANISM")
    assert mech_ranked[0]["passage_id"] == "p2"


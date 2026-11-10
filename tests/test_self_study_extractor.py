"""
Unit and Integration Tests for Self-Study QA Extractor (Proposal Section 6.4).
100% Strict Offline, Zero External API, Pure Python Test Suite.
"""

import pytest
from pathlib import Path

from src.self_study.types import QARuleType, SelfStudyQAPair, SelfStudyReport
from src.self_study.extractor import QAPairExtractor, normalize_text_key


def test_self_study_types_grounding_validation():
    """Validates that SelfStudyQAPair strictly enforces substring grounding."""
    # Valid pair
    pair = SelfStudyQAPair(
        question="SA-CMS là gì?",
        answer="bộ nhớ đa thang",
        span_start=10,
        span_end=25,
        source_text="SA-CMS là bộ nhớ đa thang căn theo cấu trúc.",
        document_id="DOC_001",
        level=1,
        rule_type=QARuleType.DEFINITION,
    )
    assert pair.question == "SA-CMS là gì?"
    assert pair.answer in pair.source_text
    assert pair.to_dict()["rule_type"] == "definition"

    # Auto-append question mark
    pair2 = SelfStudyQAPair(
        question="SA-CMS là gì",
        answer="bộ nhớ đa thang",
        span_start=10,
        span_end=25,
        source_text="SA-CMS là bộ nhớ đa thang căn theo cấu trúc.",
        document_id="DOC_001",
    )
    assert pair2.question.endswith("?")

    # Violation: answer not in source text must raise ValueError
    with pytest.raises(ValueError, match="Grounding violation"):
        SelfStudyQAPair(
            question="SA-CMS là gì?",
            answer="thực thể hoàn toàn không có trong đoạn",
            span_start=0,
            span_end=10,
            source_text="Văn bản gốc hoàn toàn khác.",
            document_id="DOC_001",
        )


def test_extractor_definition_rule():
    """Tests definition pattern extraction."""
    extractor = QAPairExtractor()
    text = (
        "Chương 1: Khái niệm\n\n"
        "SA-CMS là bộ nhớ đa thang căn theo cấu trúc văn bản phân cấp.\n\n"
        "Hope-Attention chính là kiến trúc nơ-ron nền tảng."
    )
    report = extractor.extract_from_text(text, document_id="TEST_DEF")
    assert report.total_pairs >= 1

    def_pairs = [p for p in report.pairs if p.rule_type == QARuleType.DEFINITION]
    assert len(def_pairs) >= 1
    found_sa_cms = any("SA-CMS" in p.question for p in def_pairs)
    assert found_sa_cms
    for p in def_pairs:
        assert p.answer in p.source_text


def test_extractor_mechanism_rule():
    """Tests mechanism pattern extraction."""
    extractor = QAPairExtractor()
    text = (
        "Chương 2: Cơ chế\n\n"
        "Cổng hòa trộn hoạt động nhờ tổng có trọng số của các mức bộ nhớ.\n\n"
        "Mô hình vận hành dựa trên cơ chế lan truyền ngược."
    )
    report = extractor.extract_from_text(text, document_id="TEST_MEC")
    mec_pairs = [p for p in report.pairs if p.rule_type == QARuleType.MECHANISM]
    assert len(mec_pairs) >= 1
    for p in mec_pairs:
        assert p.answer in p.source_text
        assert "hoạt động như thế nào?" in p.question


def test_extractor_metric_rule():
    """Tests quantitative metric extraction."""
    extractor = QAPairExtractor()
    text = (
        "Chương 3: Thực nghiệm\n\n"
        "Điểm số F1 đạt 0.154 trên tập kiểm chuẩn.\n\n"
        "Hệ thống tiêu thụ 1.2 GB bộ nhớ VRAM khi suy luận."
    )
    report = extractor.extract_from_text(text, document_id="TEST_MET")
    met_pairs = [p for p in report.pairs if p.rule_type == QARuleType.METRIC]
    assert len(met_pairs) >= 1
    for p in met_pairs:
        assert p.answer in p.source_text
        assert p.question.endswith("?")


def test_extractor_enumeration_rule():
    """Tests enumeration pattern extraction."""
    extractor = QAPairExtractor()
    text = (
        "Chương 4: Kiến trúc\n\n"
        "Hệ thống SA-CMS bao gồm: Paragraph, Section và Document.\n\n"
        "Mô hình chia thành: ba mức bộ nhớ độc lập."
    )
    report = extractor.extract_from_text(text, document_id="TEST_ENUM")
    enum_pairs = [p for p in report.pairs if p.rule_type == QARuleType.ENUMERATION]
    assert len(enum_pairs) >= 1
    for p in enum_pairs:
        assert p.answer in p.source_text
        assert "bao gồm những thành phần nào?" in p.question


def test_extractor_limitation_rule():
    """Tests limitation and challenge extraction."""
    extractor = QAPairExtractor()
    text = (
        "Chương 5: Đánh giá\n\n"
        "Hạn chế của phương pháp là chưa hỗ trợ bảng biểu phức tạp.\n\n"
        "Điểm yếu của baseline là hiện tượng quên lãng thảm khốc."
    )
    report = extractor.extract_from_text(text, document_id="TEST_LIM")
    lim_pairs = [p for p in report.pairs if p.rule_type == QARuleType.LIMITATION]
    assert len(lim_pairs) >= 1
    for p in lim_pairs:
        assert p.answer in p.source_text
        assert "Hạn chế của" in p.question or "Điểm yếu của" in p.question


def test_extractor_comparison_rule():
    """Tests comparative relationship extraction."""
    extractor = QAPairExtractor()
    text = (
        "Chương 6: Đối sánh\n\n"
        "SA-CMS tốt hơn baseline trên bài toán ngữ cảnh dài.\n\n"
        "Mô hình đề xuất quên ít hơn adapter một mức."
    )
    report = extractor.extract_from_text(text, document_id="TEST_COMP")
    comp_pairs = [p for p in report.pairs if p.rule_type == QARuleType.COMPARISON]
    assert len(comp_pairs) >= 1
    for p in comp_pairs:
        assert p.answer in p.source_text


def test_extractor_deduplication():
    """Tests that identical sentences do not produce duplicate questions."""
    extractor = QAPairExtractor()
    text = (
        "Mục 1: Mở đầu\n\n"
        "SA-CMS là bộ nhớ đa thang căn theo cấu trúc.\n\n"
        "SA-CMS là bộ nhớ đa thang căn theo cấu trúc.\n\n"
        "SA-CMS là bộ nhớ đa thang căn theo cấu trúc."
    )
    report = extractor.extract_from_text(text, document_id="TEST_DEDUP")
    sa_cms_q = [p for p in report.pairs if "SA-CMS là gì?" in p.question]
    assert len(sa_cms_q) == 1


def test_extractor_on_real_de_cuong_document():
    """Integration test extracting grounded QA pairs from actual de_cuong_text.txt."""
    de_cuong_path = Path("docs/de_cuong_text.txt")
    if not de_cuong_path.exists():
        pytest.skip("docs/de_cuong_text.txt not found")

    text = de_cuong_path.read_text(encoding="utf-8")
    extractor = QAPairExtractor()
    report = extractor.extract_from_text(text, document_id="DE_CUONG_REAL")

    # Proposal is ~350 lines, should yield dozens of solid pairs
    assert report.total_pairs >= 10
    assert report.extraction_time_ms > 0

    # Every single pair must strictly be grounded
    for pair in report.pairs:
        assert pair.answer in pair.source_text
        assert len(pair.question.split()) >= 3
        assert pair.question.endswith("?")
        assert pair.level in (1, 2, 3)

    # Check diverse coverage
    rules_found = {p.rule_type for p in report.pairs}
    assert len(rules_found) >= 3

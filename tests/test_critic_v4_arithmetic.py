"""
Tests for Sovereign Brain 4.0 - GroundingCritic Upgrades:
- Derived Arithmetic Verification (_is_derived_number_supported)
- Cross-Citation Metric Contrast Detection (detect_metric_contrasts)
"""

import pytest
from src.hybrid_qa.grounding_critic import GroundingCritic, ClaimVerificationResult, CriticReport


def test_is_derived_number_supported_subtraction():
    """Verifies that difference |a - b| is recognized as supported."""
    critic = GroundingCritic()
    citation_text = "Mô hình Alpha đạt độ chính xác 89.6% trong khi Mô hình Beta đạt 71.2%."

    # Exact difference 89.6 - 71.2 = 18.4
    assert critic._is_derived_number_supported("18.4%", citation_text) is True
    assert critic._is_derived_number_supported("18.4", citation_text) is True

    # Within 1e-2 tolerance (e.g. 18.41)
    assert critic._is_derived_number_supported("18.41%", citation_text) is True

    # Out of tolerance / hallucinated
    assert critic._is_derived_number_supported("18.5%", citation_text) is False
    assert critic._is_derived_number_supported("99.9%", citation_text) is False

    # Edge cases: text with insufficient numbers
    assert critic._is_derived_number_supported("18.4%", "Chỉ có một số 89.6%") is False
    assert critic._is_derived_number_supported("18.4%", "Không có số nào") is False


def test_is_derived_number_supported_addition():
    """Verifies that sum (a + b) is recognized as supported."""
    critic = GroundingCritic()
    citation_text = "Phần cứng tiêu thụ 15.5W ở chế độ chờ và thêm 24.5W khi tính toán."

    # 15.5 + 24.5 = 40.0
    assert critic._is_derived_number_supported("40.0", citation_text) is True
    assert critic._is_derived_number_supported("40.0W", citation_text) is True
    assert critic._is_derived_number_supported("50.0", citation_text) is False


def test_verify_answer_with_derived_difference_faithful():
    """
    End-to-end verification: 89.6% and 71.2% in citations -> 18.4% in answer
    must be verified as faithful, not flagged as hallucinated!
    """
    critic = GroundingCritic()
    citations = [
        {
            "citation_index": 1,
            "document_title": "Báo cáo Benchmark",
            "text": "Mô hình A đạt độ chính xác 89.6% trong khi Mô hình B đạt 71.2%.",
        }
    ]
    answer = "Mô hình A có độ chính xác cao hơn Mô hình B là 18.4%."
    report = critic.verify_answer(answer, citations)

    assert report.is_faithful is True
    assert report.faithfulness_score >= 0.8
    assert len(report.unsupported_claims) == 0
    assert len(report.claims) == 1
    assert report.claims[0].is_supported is True
    assert len(report.claims[0].unsupported_entities) == 0


def test_verify_answer_with_derived_sum_faithful():
    """End-to-end verification: 12.3s and 17.7s in citation -> 30.0s in answer is faithful."""
    critic = GroundingCritic()
    citations = [
        {
            "citation_index": 1,
            "document_title": "Chi phí tính toán",
            "text": "Giai đoạn tiền xử lý tốn 12.3s và giai đoạn suy luận tốn 17.7s.",
        }
    ]
    answer = "Giai đoạn tiền xử lý và suy luận tốn tổng thời gian là 30.0s."
    report = critic.verify_answer(answer, citations)

    assert report.is_faithful is True
    assert report.faithfulness_score >= 0.8
    assert len(report.unsupported_claims) == 0


def test_verify_answer_with_unsupported_hallucinated_difference():
    """Answer with a hallucinated difference (e.g. 35.0% instead of 18.4%) must be flagged."""
    critic = GroundingCritic()
    citations = [
        {
            "citation_index": 1,
            "document_title": "Báo cáo Benchmark",
            "text": "Mô hình A đạt độ chính xác 89.6% trong khi Mô hình B đạt 71.2%.",
        }
    ]
    answer = "Mô hình A có độ chính xác cao hơn Mô hình B là 35.0%."
    report = critic.verify_answer(answer, citations)

    assert report.is_faithful is False
    assert len(report.unsupported_claims) >= 1
    assert any("35.0" in ent for ent in report.claims[0].unsupported_entities)


def test_detect_metric_contrasts_percentages():
    """Scans across citations for % metrics and calculates delta."""
    critic = GroundingCritic()
    citations = [
        {
            "citation_index": 1,
            "document_title": "Nghiên cứu A",
            "text": "Mô hình Alpha đạt độ chính xác 89.6%.",
        },
        {
            "citation_index": 2,
            "document_title": "Nghiên cứu B",
            "text": "Mô hình Beta đạt độ chính xác 71.2%.",
        },
    ]

    contrasts = critic.detect_metric_contrasts(citations)
    assert len(contrasts) == 1
    item = contrasts[0]
    assert item["unit"] == "%"
    assert item["values"] == [89.6, 71.2]
    assert item["delta"] == 18.4
    assert item["citations"] == [1, 2]


def test_detect_metric_contrasts_multiple_units():
    """Scans citations with latency (ms) and token counts (tokens)."""
    critic = GroundingCritic()
    citations = [
        {
            "citation_index": 1,
            "document_title": "Hệ thống v1",
            "text": "Độ trễ xử lý là 500ms với chi phí 1500 tokens.",
        },
        {
            "citation_index": 2,
            "document_title": "Hệ thống v2",
            "text": "Độ trễ xử lý tối ưu còn 200ms với chi phí 500 tokens.",
        },
    ]

    contrasts = critic.detect_metric_contrasts(citations)
    assert len(contrasts) == 2

    contrast_by_unit = {c["unit"]: c for c in contrasts}
    assert "ms" in contrast_by_unit
    assert "tokens" in contrast_by_unit

    ms_contrast = contrast_by_unit["ms"]
    assert ms_contrast["values"] == [500.0, 200.0]
    assert ms_contrast["delta"] == 300.0
    assert ms_contrast["citations"] == [1, 2]

    tok_contrast = contrast_by_unit["tokens"]
    assert tok_contrast["values"] == [1500.0, 500.0]
    assert tok_contrast["delta"] == 1000.0
    assert tok_contrast["citations"] == [1, 2]


def test_detect_metric_contrasts_hardware_units():
    """Scans citations for mb, gb, and s units."""
    critic = GroundingCritic()
    citations = [
        {
            "citation_index": 1,
            "document_title": "Cấu hình Server 1",
            "text": "Bộ nhớ chiếm 1024 mb, dung lượng đĩa 4 gb, thời gian nạp 15 s.",
        },
        {
            "citation_index": 2,
            "document_title": "Cấu hình Server 2",
            "text": "Bộ nhớ chiếm 256 mb, dung lượng đĩa 1 gb, thời gian nạp 5 s.",
        },
    ]

    contrasts = critic.detect_metric_contrasts(citations)
    assert len(contrasts) == 3

    by_unit = {c["unit"]: c for c in contrasts}
    assert by_unit["mb"]["delta"] == 768.0
    assert by_unit["mb"]["citations"] == [1, 2]

    assert by_unit["gb"]["delta"] == 3.0
    assert by_unit["gb"]["citations"] == [1, 2]

    assert by_unit["s"]["delta"] == 10.0
    assert by_unit["s"]["citations"] == [1, 2]


def test_detect_metric_contrasts_single_or_empty_citations():
    """Single citation or empty citations should produce empty contrast list."""
    critic = GroundingCritic()

    assert critic.detect_metric_contrasts([]) == []
    assert critic.detect_metric_contrasts([{"text": "Chỉ có 1 tài liệu 89.6%"}]) == []
    assert critic.detect_metric_contrasts([
        {"citation_index": 1, "text": "Không có đơn vị nào"},
        {"citation_index": 2, "text": "Cũng không có đơn vị nào"},
    ]) == []


def test_derived_arithmetic_vietnamese_comma_notation():
    """Verifies that numbers using Vietnamese comma decimals are supported."""
    critic = GroundingCritic()
    citations = [
        {
            "citation_index": 1,
            "document_title": "Báo cáo Benchmark",
            "text": "Mô hình A đạt độ chính xác 89,6% trong khi Mô hình B đạt 71,2%.",
        }
    ]
    answer = "Mô hình A có độ chính xác cao hơn Mô hình B là 18,4%."
    report = critic.verify_answer(answer, citations)

    assert report.is_faithful is True
    assert len(report.unsupported_claims) == 0
    # Correct answer should not alter faithful answer
    corrected = critic.correct_answer(answer, report)
    assert corrected == answer


def test_detect_metric_contrasts_vietnamese_unit_alias():
    """Verifies that 'phần trăm' maps correctly to '%' in contrast detection."""
    critic = GroundingCritic()
    citations = [
        {
            "citation_index": 1,
            "document_title": "Khảo sát 1",
            "text": "Tỷ lệ hồi đáp đạt 85.5 phần trăm.",
        },
        {
            "citation_index": 2,
            "document_title": "Khảo sát 2",
            "text": "Tỷ lệ hồi đáp đạt 60.5 phần trăm.",
        },
    ]

    contrasts = critic.detect_metric_contrasts(citations)
    assert len(contrasts) == 1
    assert contrasts[0]["unit"] == "%"
    assert contrasts[0]["delta"] == 25.0
    assert contrasts[0]["values"] == [85.5, 60.5]


def test_detect_metric_contrasts_across_sections_same_doc():
    """Verifies contrast detection across different sections of the same document."""
    critic = GroundingCritic()
    citations = [
        {
            "citation_index": 1,
            "document_title": "Báo cáo Thống nhất",
            "section_title": "Chương 1",
            "text": "Độ chính xác ban đầu là 65.0%.",
        },
        {
            "citation_index": 2,
            "document_title": "Báo cáo Thống nhất",
            "section_title": "Chương 5",
            "text": "Độ chính xác sau tối ưu là 95.0%.",
        },
    ]

    contrasts = critic.detect_metric_contrasts(citations)
    assert len(contrasts) == 1
    assert contrasts[0]["unit"] == "%"
    assert contrasts[0]["delta"] == 30.0
    assert contrasts[0]["citations"] == [1, 2]


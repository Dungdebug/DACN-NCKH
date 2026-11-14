"""
Wave 2 Unit tests for Epistemic Uncertainty & Bayesian Confidence Calibrator (Ticket W2-D).
Tests:
1. No-diacritic queries get low semantic drift via folded unigram+bigram matching.
2. Evidence quality accounts for mean coverage and source diversity saturation.
3. Polarity conflict detection identifies opposing claims (tăng/giảm, cao/thấp, có/không, cải thiện/suy giảm, nhanh/chậm).
4. Detected conflicts raise uncertainty in calibrate_deep.
5. Unsupported answer text gets downgraded to [Giả thuyết] with low claim_support.
6. Supported answer text maintains [Chắc chắn] epistemic label.
7. Backward compatibility of CalibratedEpistemicReport dataclass defaults with calibrate().
"""

import pytest
from src.brain.epistemic_calibrator import (
    EpistemicCalibrator,
    CalibratedEpistemicReport,
    strip_accents,
)


def test_strip_accents():
    assert strip_accents("Hạn chế của mô hình") == "Han che cua mo hinh"
    assert strip_accents("Độ trễ và tốc độ") == "Do tre va toc do"
    assert strip_accents("") == ""


def test_no_diacritic_query_gets_low_drift():
    calibrator = EpistemicCalibrator()
    query_no_accents = "han che cua mo hinh"
    evidence_with_accents = [
        "Hạn chế lớn nhất của mô hình là kích thước bộ nhớ và độ trễ xử lý.",
    ]

    # Old calculate_semantic_drift failed to match because of accents -> drift = 1.0
    old_drift = calibrator.calculate_semantic_drift(query_no_accents, evidence_with_accents)
    assert old_drift == 1.0

    # New calculate_folded_semantic_drift matches unigrams and bigrams folded -> low drift
    folded_drift = calibrator.calculate_folded_semantic_drift(query_no_accents, evidence_with_accents)
    assert folded_drift <= 0.20
    assert folded_drift < old_drift


def test_evidence_quality_coverage_and_diversity():
    calibrator = EpistemicCalibrator()
    query = "chi phí token và độ trễ"

    # Case 1: Empty evidence
    assert calibrator.calculate_evidence_quality(query, []) == 0.0

    # Case 2: 3 distinct passages with high query overlap
    distinct_evidence = [
        "Thực nghiệm cho thấy chi phí token và độ trễ giảm đáng kể trên SmolLM2.",
        "Độ trễ suy luận và chi phí token được tối ưu hóa xuống dưới 5ms.",
        "Mô hình đạt cân bằng tối ưu giữa chi phí token và độ trễ phản hồi.",
    ]
    quality_high = calibrator.calculate_evidence_quality(query, distinct_evidence)
    assert quality_high >= 0.90

    # Case 3: 3 identical passages (source diversity penalized)
    redundant_evidence = [
        "Mô hình đạt cân bằng tối ưu giữa chi phí token và độ trễ phản hồi.",
        "Mô hình đạt cân bằng tối ưu giữa chi phí token và độ trễ phản hồi.",
        "Mô hình đạt cân bằng tối ưu giữa chi phí token và độ trễ phản hồi.",
    ]
    quality_redundant = calibrator.calculate_evidence_quality(query, redundant_evidence)
    assert quality_redundant < quality_high
    assert quality_redundant <= 0.40


def test_detect_polarity_conflicts_all_dimensions():
    calibrator = EpistemicCalibrator()

    # Dimension 1: tăng / giảm
    ev_tang_giam = [
        "Thực nghiệm xác nhận chi phí token tăng 25% khi kích hoạt toàn văn.",
        "Thực nghiệm xác nhận chi phí token giảm 25% khi tối ưu hóa bộ nhớ.",
    ]
    conflicts_1 = calibrator.detect_polarity_conflicts(ev_tang_giam)
    assert len(conflicts_1) == 1
    assert "tăng" in conflicts_1[0]["polarity_dimension"] or "tang_giam" in conflicts_1[0]["polarity_dimension"]
    assert "token" in conflicts_1[0]["shared_keywords"] or "phi" in conflicts_1[0]["shared_keywords"]

    # Dimension 2: cao / thấp
    ev_cao_thap = [
        "Độ chính xác mô hình đạt mức rất cao trên tập kiểm thử.",
        "Độ chính xác mô hình lại ở mức rất thấp trong môi trường thực tế.",
    ]
    conflicts_2 = calibrator.detect_polarity_conflicts(ev_cao_thap)
    assert len(conflicts_2) == 1
    assert "cao" in conflicts_2[0]["polarity_dimension"]

    # Dimension 3: có / không
    ev_co_khong = [
        "Hệ thống có cơ chế bộ nhớ dài hạn liên tục.",
        "Hệ thống không có cơ chế bộ nhớ dài hạn liên tục.",
    ]
    conflicts_3 = calibrator.detect_polarity_conflicts(ev_co_khong)
    assert len(conflicts_3) == 1
    assert "có" in conflicts_3[0]["polarity_dimension"]

    # Dimension 4: cải thiện / suy giảm
    ev_cai_thien_suy_giam = [
        "Hiệu năng tổng thể được cải thiện rõ rệt sau khi fine-tuning.",
        "Hiệu năng tổng thể bị suy giảm rõ rệt sau khi fine-tuning.",
    ]
    conflicts_4 = calibrator.detect_polarity_conflicts(ev_cai_thien_suy_giam)
    assert len(conflicts_4) == 1
    assert "cải thiện" in conflicts_4[0]["polarity_dimension"]

    # Dimension 5: nhanh / chậm
    ev_nhanh_cham = [
        "Tốc độ xử lý câu hỏi của thuật toán rất nhanh.",
        "Tốc độ xử lý câu hỏi của thuật toán rất chậm khi dữ liệu lớn.",
    ]
    conflicts_5 = calibrator.detect_polarity_conflicts(ev_nhanh_cham)
    assert len(conflicts_5) == 1
    assert "nhanh" in conflicts_5[0]["polarity_dimension"]


def test_detect_polarity_conflicts_negative_cases():
    calibrator = EpistemicCalibrator()

    # Case A: Same polarity (both positive - agreement)
    ev_agree = [
        "Thuật toán giúp tăng độ chính xác phân loại.",
        "Phương pháp mới tiếp tục tăng độ chính xác phân loại câu hỏi.",
    ]
    assert calibrator.detect_polarity_conflicts(ev_agree) == []

    # Case B: Opposite polarity but different topics (< 2 shared content keywords)
    ev_unrelated = [
        "Tốc độ xử lý của mô hình rất nhanh.",
        "Chi phí bộ nhớ RAM giảm đáng kể.",
    ]
    assert calibrator.detect_polarity_conflicts(ev_unrelated) == []


def test_conflict_raises_uncertainty_in_calibrate_deep():
    calibrator = EpistemicCalibrator()
    query = "Hiệu năng của mô hình SA-CMS như thế nào?"

    # Baseline consistent evidence
    consistent_evidence = [
        "Mô hình SA-CMS cải thiện hiệu năng xử lý đáng kể.",
        "Hiệu năng xử lý của SA-CMS được nâng cao trên SmolLM2.",
    ]
    report_consistent = calibrator.calibrate_deep(
        query=query,
        evidence_texts=consistent_evidence,
        gating_weights=(0.1, 0.8, 0.1),
    )
    assert report_consistent.epistemic_label == "[Chắc chắn]"
    assert len(report_consistent.conflicts) == 0

    # Conflicting evidence
    conflicting_evidence = [
        "Mô hình SA-CMS cải thiện hiệu năng xử lý đáng kể.",
        "Mô hình SA-CMS suy giảm hiệu năng xử lý nghiêm trọng.",
    ]
    report_conflicting = calibrator.calibrate_deep(
        query=query,
        evidence_texts=conflicting_evidence,
        gating_weights=(0.1, 0.8, 0.1),
    )

    # Uncertainty must strictly increase due to conflict term
    assert len(report_conflicting.conflicts) >= 1
    assert report_conflicting.uncertainty_score > report_consistent.uncertainty_score


def test_unsupported_answer_downgrades_to_hypothesis():
    calibrator = EpistemicCalibrator()
    query = "SA-CMS giảm bao nhiêu chi phí token?"
    evidence = [
        "SA-CMS giảm 87,5% chi phí token trên SmolLM2.",
        "Số liệu giảm 87,5% được xác nhận trong thực nghiệm.",
    ]

    # Without answer or with well-supported answer -> [Chắc chắn]
    supported_ans = "SA-CMS giảm 87,5% chi phí token theo thực nghiệm."
    report_supported = calibrator.calibrate_deep(
        query=query,
        evidence_texts=evidence,
        gating_weights=(0.05, 0.90, 0.05),
        answer_text=supported_ans,
    )
    assert report_supported.claim_support >= 0.80
    assert report_supported.epistemic_label == "[Chắc chắn]"

    # Unsupported hallucinated answer -> must downgrade to [Giả thuyết]
    unsupported_ans = (
        "Mô hình được áp dụng trực tiếp để điều khiển tàu vũ trụ hạ cánh sao Hỏa. "
        "Ngoài ra nó còn chữa khỏi bệnh cúm mùa cho toàn bộ nhân loại."
    )
    report_unsupported = calibrator.calibrate_deep(
        query=query,
        evidence_texts=evidence,
        gating_weights=(0.05, 0.90, 0.05),
        answer_text=unsupported_ans,
    )
    assert report_unsupported.claim_support < 0.50
    assert report_unsupported.epistemic_label == "[Giả thuyết]"
    assert report_unsupported.uncertainty_score >= 0.70
    assert "hạ bậc" in report_unsupported.calibration_explanation or "thiếu hỗ trợ" in report_unsupported.calibration_explanation


def test_calibrate_deep_empty_evidence_fallback():
    calibrator = EpistemicCalibrator()
    report = calibrator.calibrate_deep(
        query="Truy vấn bất kỳ",
        evidence_texts=[],
        answer_text="Câu trả lời bất kỳ",
    )
    assert report.epistemic_label == "[Giả thuyết]"
    assert report.uncertainty_score == 1.0
    assert report.evidence_quality == 0.0
    assert report.claim_support == 0.0
    assert report.conflicts == []


def test_backward_compatibility_calibrate_defaults():
    calibrator = EpistemicCalibrator()
    # Call original calibrate() method
    report = calibrator.calibrate(
        query="SA-CMS giảm bao nhiêu phần trăm chi phí token?",
        evidence_texts=[
            "Thực nghiệm chứng minh SA-CMS giảm 87,5% chi phí token.",
            "Số liệu giảm 87,5% chi phí token được xác nhận trên SmolLM2-135M.",
        ],
        gating_weights=(0.10, 0.85, 0.05),
    )
    # Default fields on CalibratedEpistemicReport are present and have default values
    assert hasattr(report, "evidence_quality")
    assert report.evidence_quality == 0.0
    assert hasattr(report, "conflicts")
    assert report.conflicts == []
    assert hasattr(report, "claim_support")
    assert report.claim_support == 1.0
    assert report.epistemic_label == "[Chắc chắn]"

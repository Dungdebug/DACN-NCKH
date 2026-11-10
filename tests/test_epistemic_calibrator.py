"""
Unit tests for Epistemic Uncertainty & Bayesian Confidence Calibrator (Ticket 7.4).
Verifies calculation of Gating Entropy H(g), Evidence Density, Semantic Drift,
and calibrated labels ([Chắc chắn] / [Có khả năng cao] / [Giả thuyết]).
"""

import pytest
from src.brain.epistemic_calibrator import (
    EpistemicCalibrator,
    CalibratedEpistemicReport,
)


def test_high_certainty_calibration():
    calibrator = EpistemicCalibrator()
    # Crisp gating (Level 2 dominates), high evidence overlap, 3 supporting passages
    report = calibrator.calibrate(
        query="SA-CMS giảm bao nhiêu phần trăm chi phí token?",
        evidence_texts=[
            "Thực nghiệm chứng minh SA-CMS giảm 87,5% chi phí token.",
            "Số liệu giảm 87,5% chi phí token được xác nhận trên SmolLM2-135M.",
        ],
        gating_weights=(0.10, 0.85, 0.05),
    )
    
    assert report.epistemic_label == "[Chắc chắn]"
    assert report.uncertainty_score <= 0.35
    assert report.gating_entropy < 0.65
    assert "chắc chắn" in report.calibration_explanation.lower()


def test_moderate_certainty_calibration():
    calibrator = EpistemicCalibrator()
    # Moderate gating focus, evidence with good overlap
    report = calibrator.calibrate(
        query="Thuật toán huấn luyện có tiêu tốn nhiều bộ nhớ RAM không?",
        evidence_texts=[
            "Quá trình huấn luyện đòi hỏi thời gian tính toán nhưng bộ nhớ RAM được tối ưu hóa.",
        ],
        gating_weights=(0.20, 0.60, 0.20),
    )
    
    assert report.epistemic_label in ("[Có khả năng cao]", "[Chắc chắn]")
    assert 0.20 <= report.uncertainty_score <= 0.65


def test_high_uncertainty_sparse_evidence():
    calibrator = EpistemicCalibrator()
    # Maximal entropy (1/3 each), low keyword overlap
    report = calibrator.calibrate(
        query="Tác giả dự định áp dụng mô hình cho xe tự hành vào năm 2030 đúng không?",
        evidence_texts=[
            "Tài liệu tập trung vào nghiên cứu chatbot văn bản cho sinh viên.",
        ],
        gating_weights=(0.333, 0.333, 0.334),
    )
    
    assert report.epistemic_label == "[Giả thuyết]"
    assert report.uncertainty_score > 0.55
    assert "bất định" in report.calibration_explanation.lower() or "giả thuyết" in report.calibration_explanation.lower()


def test_zero_evidence_fallback():
    calibrator = EpistemicCalibrator()
    report = calibrator.calibrate(
        query="Câu hỏi không có bất kỳ bằng chứng nào",
        evidence_texts=[],
        gating_weights=(0.33, 0.33, 0.34),
    )
    
    assert report.epistemic_label == "[Giả thuyết]"
    assert report.uncertainty_score == 1.0

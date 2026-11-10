"""
Unit Tests for CrossDocumentContradictionScanner (Ticket 2.2).
Validates:
1. Numerical claims extraction and decimal normalization.
2. Numerical discrepancy detection across distinct documents.
3. Polarity conflict detection (affirmative vs negative assertions).
4. Consistent documents handling (zero false alarms).
5. Single-document boundary handling.
"""

import pytest
from src.reasoning.cross_doc_scanner import (
    CrossDocumentContradictionScanner,
    ContradictionType,
    ContradictionAlert,
    ContradictionScanReport,
)


def test_extract_numerical_claims():
    scanner = CrossDocumentContradictionScanner()
    text = (
        "Hệ thống đạt tỷ lệ giảm 87,5% chi phí token so với RAG. "
        "Độ trễ trung bình duy trì 18.2 ms trong suốt phiên thử nghiệm."
    )
    claims = scanner.extract_numerical_claims(text, document_id="doc1", section_title="Kết quả")
    assert len(claims) >= 2
    assert any(c["value"] == 87.5 and c["unit"] == "%" for c in claims)
    assert any(c["value"] == 18.2 and c["unit"] == "ms" for c in claims)


def test_numerical_contradiction_detection():
    scanner = CrossDocumentContradictionScanner(tolerance_ratio=0.05)

    doc_a = {
        "document_id": "bao_cao_quy_1",
        "title": "Báo cáo Q1",
        "text": "Mô hình đạt tỷ lệ giảm 87,5% chi phí token trong thử nghiệm ban đầu.",
    }
    doc_b = {
        "document_id": "bao_cao_quy_2",
        "title": "Báo cáo Q2",
        "text": "Mô hình đạt tỷ lệ giảm 45,0% chi phí token do dữ liệu quá dài.",
    }

    report = scanner.scan_documents([doc_a, doc_b])
    assert report.total_documents == 2
    assert report.contradictions_found >= 1

    alert = report.alerts[0]
    assert alert.contradiction_type == ContradictionType.NUMERICAL_DISCREPANCY
    assert alert.severity == "HIGH"
    assert "87.5" in alert.explanation
    assert "45.0" in alert.explanation
    assert alert.claim_a["document_id"] != alert.claim_b["document_id"]


def test_polarity_contradiction_detection():
    scanner = CrossDocumentContradictionScanner()

    doc_a = {
        "document_id": "nghien_cuu_ly_thuyet",
        "title": "Nghiên cứu lý thuyết",
        "text": "Triển khai mô hình trên GPU 4GB là hoàn toàn khả thi với cấu hình nén.",
    }
    doc_b = {
        "document_id": "thuc_nghiem_phong_lab",
        "title": "Thực nghiệm phòng lab",
        "text": "Triển khai mô hình trên GPU 4GB là chưa khả thi khi chạy tải cao.",
    }

    report = scanner.scan_documents([doc_a, doc_b])
    assert report.contradictions_found >= 1

    pol_alerts = [a for a in report.alerts if a.contradiction_type == ContradictionType.POLARITY_CONFLICT]
    assert len(pol_alerts) >= 1
    alert = pol_alerts[0]
    assert alert.severity == "HIGH"
    assert "khẳng định" in alert.explanation or "xung đột" in alert.explanation.lower()


def test_no_contradiction_on_consistent_documents():
    scanner = CrossDocumentContradictionScanner(tolerance_ratio=0.05)

    doc_a = {
        "document_id": "doc_a",
        "title": "Tài liệu A",
        "text": "Mô hình đạt tỷ lệ giảm 87,5% chi phí token theo đo đạc chuẩn.",
    }
    doc_b = {
        "document_id": "doc_b",
        "title": "Tài liệu B",
        "text": "Hệ thống đạt tỷ lệ giảm 88,0% chi phí token trong thử nghiệm lặp lại.",  # delta < 1%, within 5% tolerance
    }

    report = scanner.scan_documents([doc_a, doc_b])
    assert report.contradictions_found == 0


def test_single_document_scan():
    scanner = CrossDocumentContradictionScanner()
    doc_single = {
        "document_id": "doc_single",
        "title": "Tài liệu duy nhất",
        "text": "Hệ thống đạt tỷ lệ giảm 87,5% chi phí token.",
    }
    report = scanner.scan_documents([doc_single])
    assert report.total_documents == 1
    assert report.contradictions_found == 0
    assert len(report.alerts) == 0


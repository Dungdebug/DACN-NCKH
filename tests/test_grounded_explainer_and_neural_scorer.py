"""
Tests for Self-Built WaveCMS Discourse Brain (NeuralDiscourseScorer + GroundedExplainer).

Verifies:
1. Local PyTorch NeuralDiscourseHead checkpoint loads offline (<15ms CPU), classifies roles,
   and assigns higher salience to real explanatory clauses than keyword-matched headings/fillers.
2. GroundedExplainer compresses keyword-heavy text into concise, easy-to-understand output
   passing GroundingCritic + verify_numeric_provenance (0% hallucination, zero canned fluff).
3. Entity-grounded comparison & causal synthesis use real entities instead of generic templates.
4. End-to-end coexistence of Dung's 'explain' mode and Kiet's 'full' / 'key_points' modes
   without cross-section leakage or duplicate answers.
5. Safe refusal on out-of-document questions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.brain.grounded_explainer import GroundedExplainer, compress_clause, get_grounded_explainer
from src.brain.neural_discourse_scorer import DEFAULT_CHECKPOINT_PATH, NeuralDiscourseScorer
from src.hybrid_qa.grounding_critic import GroundingCritic
from src.reasoning.symbolic_solver import verify_numeric_provenance
from src.web.app import app
from src.web.document_analyzer import DocumentAnalyzer
from src.web.question_router import QuestionRouter


ROOT_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT_DIR / "results" / "discourse_scorer_metrics.json"


@dataclass
class _MockPassage:
    passage_id: str
    document_id: str
    document_title: str
    section_title: str
    paragraph_index: int
    score: float
    text: str


BANNED_CANNED_PHRASES = (
    "dữ kiện bổ sung trực tiếp cho nhau",
    "dữ kiện mở đầu nêu cơ sở vận hành",
    "xác định rõ đặc trưng riêng của từng khái niệm",
    "nội dung trên được tổng hợp đối chiếu trực tiếp từ",
)


def test_neural_discourse_scorer_role_and_salience():
    """Verify trained PyTorch discourse scorer checkpoint, val quality metrics, CPU latency, and noise filtering."""
    assert DEFAULT_CHECKPOINT_PATH.exists(), f"Missing trained checkpoint at {DEFAULT_CHECKPOINT_PATH}"
    assert METRICS_PATH.exists(), f"Missing training metrics JSON at {METRICS_PATH}"

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics["val_role_macro_f1"] >= 0.80
    assert metrics["val_salience_f1"] >= 0.82
    assert metrics["avg_cpu_inference_ms"] < 15.0

    scorer = NeuralDiscourseScorer(checkpoint_path=DEFAULT_CHECKPOINT_PATH)
    assert scorer.has_trained_weights is True

    query = "Hệ thống WaveCMS hoạt động như thế nào và có hạn chế gì?"
    clauses = [
        "# PHẦN 2: KIẾN TRÚC HỆ THỐNG WAVECMS",                                         # hard-neg heading
        "Bảng dưới đây trình bày tóm tắt về hệ thống WaveCMS.",                           # hard-neg filler
        "WaveCMS là hệ thống bộ nhớ liên tục đa phân giải căn chỉnh theo cấu trúc văn bản.",  # DEFINITION
        "Cơ chế định tuyến Sparse-LSE giúp giảm độ phức tạp truy vấn từ tuyến tính xuống O(log N).",  # MECHANISM / CAUSE
        "Độ trễ trung bình khi tắt PMI đạt 28.19 ms trên GTX 1650 Ti.",                   # METRIC_EXAMPLE
        "Tuy nhiên, hạn chế của WaveCMS là đòi hỏi tài liệu phải có phân đoạn cấu trúc rõ ràng.",  # LIMITATION_CONDITION
    ]

    t0 = time.perf_counter()
    scored = scorer.score_clauses(query, clauses)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 15.0

    # Real explanations beat keyword-matched heading/filler noise in salience
    heading_sc, filler_sc = scored[0], scored[1]
    real_scores = scored[2:]
    assert heading_sc.is_filler is True
    assert filler_sc.is_filler is True
    assert min(s.salience for s in real_scores) > max(heading_sc.salience, filler_sc.salience)

    # Discourse roles cleanly identified
    assert scored[2].role == "DEFINITION"
    assert scored[4].role == "METRIC_EXAMPLE"
    assert scored[5].role == "LIMITATION_CONDITION"


def test_grounded_explainer_no_raw_keyword_dump_and_high_faithfulness():
    """Verify verbose noisy keyword-rich passage becomes concise, understandable, and 100% grounded."""
    noisy_passage_text = (
        "## 2. Tổng quan hệ thống WaveCMS\n"
        "Như đã trình bày ở trên, phần này nói về WaveCMS.\n"
        "WaveCMS là hệ thống bộ nhớ lai phân cấp sử dụng biến đổi Wavelet Haar để tổ chức tri thức theo cây (xem mục 1 để biết thêm lịch sử nghiên cứu dài dòng không liên quan tới phần kỹ thuật hiện tại).\n"
        "Ngoài ra, cơ chế định tuyến Sparse-LSE hoạt động bằng cách chọn nhánh cây có xác suất log-sum-exp cao nhất nhằm giảm độ phức tạp xuống O(log N).\n"
        "Trong thực nghiệm đánh giá trên 64 tài liệu, WaveCMS đạt routing hit@1 = 0.70 và độ trễ 28.19 ms.\n"
        "Ví dụ như ta giả sử có nhiều tập dữ liệu nhỏ khác nhau trong phòng thí nghiệm thử nghiệm.\n"
        "Bảng dưới đây trình bày chi tiết các thông số của WaveCMS.\n"
        "Tuy nhiên, hạn chế của WaveCMS là đòi hỏi cấu trúc chương mục đầu vào rõ ràng để phân tách nhánh chính xác."
    )

    raw_passages = [
        _MockPassage(
            passage_id="P_WC_1",
            document_id="DOC_WC",
            document_title="WaveCMS Architecture",
            section_title="2. Tổng quan hệ thống WaveCMS",
            paragraph_index=1,
            score=11.0,
            text=noisy_passage_text,
        )
    ]

    answer, citations = DocumentAnalyzer.find_best_grounded_answer(
        query="Giải thích ngắn gọn dễ hiểu về WaveCMS là gì và hoạt động ra sao?",
        evidence_passages=raw_passages,
        answer_mode="balanced",
    )

    body = answer.split("### Nguồn tham khảo")[0].strip()
    body_low = body.lower()

    # Concise compared to raw dump & noise phrases filtered out
    assert len(body) < 0.75 * len(noisy_passage_text), f"Expected concise body, got len {len(body)} vs {len(noisy_passage_text)}:\n{body}"
    assert "như đã trình bày" not in body_low
    assert "bảng dưới đây trình bày" not in body_low
    assert "lịch sử nghiên cứu dài dòng" not in body_low
    for canned in BANNED_CANNED_PHRASES:
        assert canned not in body_low, f"Found canned boilerplate '{canned}' in:\n{body}"

    # GroundingCritic and numeric provenance invariance
    critic_cits = [{"citation_index": c["citation_index"], "text": noisy_passage_text} for c in citations]
    report = GroundingCritic().verify_answer(body, critic_cits)
    assert report.is_faithful is True, f"GroundingCritic rejected output: {report.unsupported_claims}"

    ok_nums, ungrounded = verify_numeric_provenance(body, [noisy_passage_text])
    assert ok_nums is True, f"Ungrounded numbers detected: {ungrounded}"


def test_entity_grounded_comparison_and_causal_explanation():
    """Verify comparison and causal queries produce real entity reasoning without canned boilerplate."""
    chem_passages = [
        _MockPassage(
            passage_id="CH_1",
            document_id="DOC_CHEM",
            document_title="Phản ứng Oxi hóa - Khử",
            section_title="1. Định nghĩa chất khử và chất oxi hóa",
            paragraph_index=1,
            score=12.0,
            text=(
                "Chất khử là chất nhường electron trong phản ứng hóa học và có số oxi hóa tăng lên. "
                "Chất oxi hóa là chất nhận electron trong phản ứng hóa học và có số oxi hóa giảm xuống."
            ),
        ),
        _MockPassage(
            passage_id="CH_2",
            document_id="DOC_CHEM",
            document_title="Phản ứng Oxi hóa - Khử",
            section_title="2. Nguyên tắc bảo toàn electron",
            paragraph_index=2,
            score=11.0,
            text="Vì tổng số electron chất khử nhường luôn bằng tổng số electron chất oxi hóa nhận nên phương trình phản ứng phải bảo toàn electron.",
        ),
    ]

    comp_ans, _ = DocumentAnalyzer.find_best_grounded_answer(
        query="Phân biệt chất khử và chất oxi hóa khác nhau thế nào?",
        evidence_passages=chem_passages,
        answer_mode="balanced",
    )
    comp_body = comp_ans.split("### Nguồn tham khảo")[0].lower()
    assert "chất khử" in comp_body and "nhường electron" in comp_body
    assert "chất oxi hóa" in comp_body and "nhận electron" in comp_body
    for canned in BANNED_CANNED_PHRASES:
        assert canned not in comp_body

    causal_ans, _ = DocumentAnalyzer.find_best_grounded_answer(
        query="Giải thích tại sao phản ứng oxi hóa khử phải bảo toàn electron?",
        evidence_passages=chem_passages,
        answer_mode="balanced",
    )
    causal_body = causal_ans.split("### Nguồn tham khảo")[0].lower()
    assert "electron" in causal_body and ("nhường" in causal_body or "nhận" in causal_body)
    for canned in BANNED_CANNED_PHRASES:
        assert canned not in causal_body


def test_e2e_three_modes_coexist_dung_and_kiet():
    """
    End-to-end API check ensuring Kiet's full/key_points modes and Dung's explain mode
    all work harmoniously on the same document with zero cross-section bleed.
    """
    doc_title = "Kien Truc Phan Tang W8 E2E Explain"
    doc_text = (
        "# PHẦN 1: BỐI CẢNH VÀ BÀI TOÁN\n"
        "- Các mô hình cũ đời 1995 thường gặp tràn ngữ cảnh khi văn bản dài hơn 512 token.\n"
        "- Chi phí tìm kiếm tuyến tính O(N) làm chậm thời gian phản hồi.\n\n"
        "# PHẦN 2: KIẾN TRÚC GIẢI THÍCH WAVECMS\n"
        "- WaveCMS phân rã tài liệu theo cấu trúc mục bằng biến đổi Wavelet Haar để lưu trữ gọn nhẹ.\n"
        "- Cơ chế Sparse-LSE giúp định tuyến truy xuất trong thời gian O(log N) với độ chính xác định tuyến cao.\n"
        "- Nhờ thống kê đủ số nguyên int64 nên hệ thống xóa tri thức với sai số nhỏ hơn 5.55e-17 trong 87.2 ms.\n"
        "- Trên thực nghiệm 64 tài liệu, mô hình đạt routing hit@1 = 0.70.\n"
        "- Tuy nhiên, hệ thống đòi hỏi tài liệu đầu vào phải có đề mục phân đoạn rõ ràng.\n"
        "- Ngoài ra, bảng tham số chi tiết ghi nhận dung lượng bộ nhớ chiếm 743 MB VRAM.\n"
    )

    assert QuestionRouter.classify_section_granularity("Giải thích ngắn gọn dễ hiểu phần 2") == "explain"
    assert QuestionRouter.classify_section_granularity("Toàn bộ nội dung của phần 2 có gì? Có gì lấy hết") == "full"
    assert QuestionRouter.classify_section_granularity("Lấy phần chính của phần 2") == "key_points"

    client = TestClient(app)
    res_doc = client.post("/api/documents/raw", json={"title": doc_title, "text": doc_text})
    assert res_doc.status_code == 200, res_doc.text
    doc_id = res_doc.json()["document_id"]

    res_sess = client.post("/api/chat/sessions", json={"title": "W8 Explain Coexistence", "document_ids": [doc_id]})
    assert res_sess.status_code == 200, res_sess.text
    sid = res_sess.json()["session_id"]

    try:
        # 1. Kiet full section
        r_full = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": "Toàn bộ nội dung của phần 2 có gì? Có gì lấy hết", "answer_mode": "balanced"},
        )
        assert r_full.status_code == 200
        p_full = r_full.json()
        assert p_full["refused"] is False
        full_body = p_full["answer"].split("### Nguồn tham khảo")[0]
        assert "743 mb vram" in full_body.lower()
        assert "5.55e-17" in full_body.lower()
        assert "1995" not in full_body.lower()

        # 2. Kiet key points
        r_kp = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": "Lấy phần chính của phần 2", "answer_mode": "balanced"},
        )
        assert r_kp.status_code == 200
        kp_body = r_kp.json()["answer"].split("### Nguồn tham khảo")[0]
        assert len(kp_body) < len(full_body)
        assert "1995" not in kp_body.lower()

        # 3. Dung grounded concise explanation of Section 2
        r_exp = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": "Giải thích ngắn gọn dễ hiểu phần 2", "answer_mode": "balanced"},
        )
        assert r_exp.status_code == 200
        p_exp = r_exp.json()
        assert p_exp["refused"] is False
        exp_body = p_exp["answer"].split("### Nguồn tham khảo")[0]
        assert len(exp_body) < len(full_body)
        assert "1995" not in exp_body.lower()
        assert "**2. suy luận" not in exp_body.lower()
        assert "góc nhìn đa độ phân giải" not in exp_body.lower()
        assert p_exp["diagnostics"]["discourse_explainer"]["engine"] == "WaveCMS-Discourse-SelfBuilt"
        assert p_exp["diagnostics"]["discourse_explainer"]["neural_scorer_active"] is True

        # 4. Dung concise explanation of whole document
        r_doc_exp = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": "Giải thích ngắn gọn dễ hiểu tài liệu này nói gì?", "answer_mode": "balanced"},
        )
        assert r_doc_exp.status_code == 200
        p_doc_exp = r_doc_exp.json()
        assert p_doc_exp["refused"] is False
        assert "wavecms" in p_doc_exp["answer"].lower()
    finally:
        client.delete(f"/api/chat/sessions/{sid}")
        client.delete(f"/api/documents/{doc_id}")


def test_explainer_never_hallucinates_out_of_doc():
    """Verify out-of-domain query triggers safe refusal instead of fabricated explanation."""
    client = TestClient(app)
    res_doc = client.post(
        "/api/documents/raw",
        json={
            "title": "Tai Lieu Hoa Hoc Co Ban",
            "text": "## 1. Phản ứng trao đổi ion\nPhản ứng trao đổi ion chỉ xảy ra khi tạo thành chất kết tủa, chất điện li yếu hoặc chất khí.",
        },
    )
    doc_id = res_doc.json()["document_id"]
    res_sess = client.post("/api/chat/sessions", json={"title": "OutOfDoc Test", "document_ids": [doc_id]})
    sid = res_sess.json()["session_id"]

    try:
        r_oob = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": "Giải thích ngắn gọn dễ hiểu cách nấu phở bò Hà Nội trong tài liệu", "answer_mode": "balanced"},
        )
        assert r_oob.status_code == 200
        assert r_oob.json()["refused"] is True
    finally:
        client.delete(f"/api/chat/sessions/{sid}")
        client.delete(f"/api/documents/{doc_id}")

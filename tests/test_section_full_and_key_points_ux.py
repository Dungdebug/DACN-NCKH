"""
Regression tests for Full-Section Extraction, Key-Points Extraction,
and Single-Unified-Answer UX (no duplicate answer blocks):
1. Exhaustive section lookup ('Toàn bộ nội dung của phần 2 có gì? Có gì lấy hết', 'Trong mục 2.1 có gì?'):
   - Returns 100% of every line, bullet, formula, and child subsection (2.1, 2.2, Ví dụ 2.1).
   - Zero leakage from Section 1.
   - Single cohesive answer block (no duplicate '**2. Suy luận & Liên hệ**' split).
2. Key-points extraction ('Lấy phần chính của phần 2', 'Lấy ý chính của phần 2', 'Lấy phần chính của tài liệu này'):
   - Returns concise bulleted key takeaways, strictly shorter than full section dump.
3. No duplicate answer parts across fast | deep | broad | deep_broad modes:
   - No '**Góc nhìn đa độ phân giải:**' duplicate block injected into answer body
     (while `reasoning_trace['facets']` still contains all 5 valid facets for broad/deep_broad).
   - Frontend `renderReasoningTraceHtml` skips trace box in `fast` mode and defaults `isOpen = false`.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.web.app import app
from src.web.question_router import QuestionRouter, QuestionRoute


MULTI_SECTION_DOC_TITLE = "Kien Truc Bo Nho Va Phan Ung W8 Test"
MULTI_SECTION_DOC_TEXT = """# PHẦN 1: CƠ SỞ LÝ THUYẾT CHUNG

## 1.1. Tổng quan lịch sử
- Bộ nhớ cổ điển chỉ lưu trữ theo khối kích thước cố định năm 1998.
- Các mô hình cũ thường mất ngữ cảnh khi văn bản vượt quá 512 token.

## 1.2. Hạn chế cần khắc phục
- Độ phức tạp truy xuất tăng tuyến tính O(N) khi số đoạn tăng nhanh.

# PHẦN 2: KIẾN TRÚC WAVECMS-R3 VÀ CƠ CHẾ VẬN HÀNH

## 2.1. Quy trình 3 bước xử lý
- Bước 1: Phân rã tài liệu theo cấu trúc chương mục bằng biến đổi Wavelet Haar nhị phân.
- Bước 2: Định tuyến Sparse-LSE giúp độ phức tạp truy vấn giảm còn O(log N) với độ khớp 100%.
- Bước 3: Kiểm soát rủi ro ảo giác bằng Mondrian Conformal Risk Control ở mức epsilon 0.10.

## 2.2. Đặc điểm nổi bật và thành phần phụ trợ
- Xóa tri thức chuẩn bit int64 với sai số nhỏ hơn 5.55e-17 trong 87.2 ms.
- Hệ thống bảo toàn 100% thông tin gốc của phần không bị xóa.
- Hỗ trợ 4 chế độ tư duy: Nhanh, Suy nghĩ Sâu, Tư duy Rộng và Sâu + Rộng.
- Độ trễ trung bình khi tắt PMI đạt 28.19 ms trên GTX 1650 Ti.

## Ví dụ 2.1
Giả sử có 64 tài liệu với 1598 nút lá, bộ định tuyến Sparse-LSE đạt routing hit@1 = 0.70 trong khi mean_key sụp về 0.00.
"""


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def attached_session(client: TestClient):
    res = client.post(
        "/api/documents/raw",
        json={"title": MULTI_SECTION_DOC_TITLE, "text": MULTI_SECTION_DOC_TEXT},
    )
    assert res.status_code == 200, res.text
    doc_id = res.json()["document_id"]

    sess_res = client.post(
        "/api/chat/sessions",
        json={"title": "W8 Section Granularity UX Test", "document_ids": [doc_id]},
    )
    assert sess_res.status_code == 200, sess_res.text
    sid = sess_res.json()["session_id"]

    yield client, sid, doc_id

    client.delete(f"/api/chat/sessions/{sid}")
    client.delete(f"/api/documents/{doc_id}")


def test_router_distinguishes_full_vs_key_points_granularity():
    router = QuestionRouter()

    r_full, m_full = router.classify_intent("Toàn bộ nội dung của phần 2 có gì? Có gì lấy hết")
    assert r_full == QuestionRoute.SECTION_LOOKUP
    assert m_full["section_granularity"] == "full"

    r_sub, m_sub = router.classify_intent("Trong mục 2.1 có gì?")
    assert r_sub == QuestionRoute.SECTION_LOOKUP
    assert m_sub["section_granularity"] == "full"

    r_sec_kp, m_sec_kp = router.classify_intent("Lấy phần chính của phần 2")
    assert r_sec_kp == QuestionRoute.SECTION_LOOKUP
    assert m_sec_kp["section_granularity"] == "key_points"

    r_doc_kp, m_doc_kp = router.classify_intent("Lấy phần chính của tài liệu này", num_attached_documents=1)
    assert r_doc_kp == QuestionRoute.DOCUMENT_SUMMARY
    assert m_doc_kp["section_granularity"] == "key_points"

    r_doc_yi, m_doc_yi = router.classify_intent("Lấy ý chính của tài liệu", num_attached_documents=1)
    assert r_doc_yi == QuestionRoute.DOCUMENT_SUMMARY
    assert m_doc_yi["section_granularity"] == "key_points"


def test_full_section_lossless_extraction_no_truncation_no_bleed(attached_session):
    c, sid, _ = attached_session
    res = c.post(
        f"/api/chat/sessions/{sid}/messages",
        json={"query": "Toàn bộ nội dung của phần 2 có gì? Có gì lấy hết", "answer_mode": "balanced"},
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["refused"] is False
    ans = payload["answer"]
    ans_lower = ans.lower()

    # Must contain 100% of Section 2 bullets across 2.1, 2.2, and Ví dụ 2.1
    expected_facts = [
        "wavelet haar",
        "sparse-lse",
        "mondrian conformal risk control",
        "5.55e-17",
        "bảo toàn 100%",
        "4 chế độ tư duy",
        "28.19 ms",
        "1598 nút lá",
    ]
    for fact in expected_facts:
        assert fact in ans_lower, f"Missing expected fact {fact!r} from full section 2:\n{ans}"

    # Zero leakage from Section 1
    assert "năm 1998" not in ans_lower
    assert "512 token" not in ans_lower

    # Single cohesive answer (no second reasoning/synthesis split)
    assert "**2. suy luận" not in ans_lower
    assert "góc nhìn đa độ phân giải" not in ans_lower


def test_sub_section_lookup_isolated_and_complete(attached_session):
    c, sid, _ = attached_session
    res = c.post(
        f"/api/chat/sessions/{sid}/messages",
        json={"query": "Trong mục 2.1 có gì?", "answer_mode": "balanced"},
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["refused"] is False
    ans_lower = payload["answer"].lower()

    assert "bước 1:" in ans_lower and "wavelet haar" in ans_lower
    assert "bước 2:" in ans_lower and "sparse-lse" in ans_lower
    assert "bước 3:" in ans_lower and "mondrian" in ans_lower
    # Isolated to 2.1: must not bleed 2.2 or Phần 1
    assert "năm 1998" not in ans_lower
    assert "5.55e-17" not in ans_lower


def test_nonexistent_section_refuses_cleanly(attached_session):
    c, sid, _ = attached_session
    res = c.post(
        f"/api/chat/sessions/{sid}/messages",
        json={"query": "Toàn bộ nội dung phần 99 có gì?", "answer_mode": "balanced"},
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["refused"] is True


def test_key_points_section_and_document_concise_and_grounded(attached_session):
    c, sid, _ = attached_session

    full_res = c.post(
        f"/api/chat/sessions/{sid}/messages",
        json={"query": "Toàn bộ nội dung của phần 2 có gì? Có gì lấy hết", "answer_mode": "balanced"},
    )
    assert full_res.status_code == 200
    full_body = full_res.json()["answer"].split("### Nguồn tham khảo")[0]

    for q_kp in ("Lấy phần chính của phần 2", "Lấy ý chính của phần 2"):
        kp_res = c.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": q_kp, "answer_mode": "balanced"},
        )
        assert kp_res.status_code == 200, kp_res.text
        kp_payload = kp_res.json()
        assert kp_payload["refused"] is False
        kp_body = kp_payload["answer"].split("### Nguồn tham khảo")[0]
        assert "- " in kp_body
        assert len(kp_body) < len(full_body), f"Key points should be more concise than full dump for {q_kp!r}"
        assert "**2. suy luận" not in kp_body.lower()
        assert "năm 1998" not in kp_body.lower()

    doc_kp_res = c.post(
        f"/api/chat/sessions/{sid}/messages",
        json={"query": "Lấy phần chính của tài liệu này", "answer_mode": "balanced"},
    )
    assert doc_kp_res.status_code == 200, doc_kp_res.text
    doc_kp_payload = doc_kp_res.json()
    assert doc_kp_payload["refused"] is False
    doc_kp_body = doc_kp_payload["answer"].split("### Nguồn tham khảo")[0]
    assert "- " in doc_kp_body
    assert "**BẢN ĐỒ TRI THỨC TÀI LIỆU:" not in doc_kp_body


def test_no_duplicate_answer_parts_across_thinking_modes_and_frontend_collapsed(attached_session):
    c, sid, _ = attached_session

    for mode in ("fast", "deep", "broad", "deep_broad"):
        res = c.post(
            f"/api/chat/sessions/{sid}/messages",
            json={
                "query": "Cơ chế Sparse-LSE và Mondrian Conformal Risk Control hoạt động như thế nào?",
                "answer_mode": "balanced",
                "thinking_mode": mode,
            },
        )
        assert res.status_code == 200, res.text
        payload = res.json()
        assert payload["refused"] is False
        assert "**Góc nhìn đa độ phân giải:**" not in payload["answer"]
        trace = payload.get("reasoning_trace") or {}
        if mode == "fast":
            assert trace.get("steps") == []
        if mode in ("broad", "deep_broad"):
            assert len(trace.get("facets") or []) == 5

    # Verify frontend skips trace panel in fast mode and collapses panel by default in non-fast modes
    repo_root = Path(__file__).resolve().parent.parent
    app_js = (repo_root / "src" / "web" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert 'if (mode === "fast" || (!mode && steps.length === 0 && facets.length === 0))' in app_js
    assert "const isOpen = false;" in app_js

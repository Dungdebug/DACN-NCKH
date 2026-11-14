"""
End-to-end regression test verifying that auto-generated follow-up questions
(`suggested_questions`) produce valid, non-refused, and distinct answer bodies
across multiple document types via FastAPI TestClient(app).
"""

import re
import pytest
from fastapi.testclient import TestClient

from src.web.app import app


CRYPTO_SLIDE_TEXT = """=== Page 1 ===
1. HỆ MÃ DỊCH VÒNG:
- Không gian bản rõ P, bản mã C và khóa K đều thuộc vành số nguyên Z26.
- Hàm mã hóa: e_k(x) = (x + k) mod 26.
- Hàm giải mã: d_k(y) = (y - k) mod 26.
- Ví dụ kinh điển là mật mã Caesar với khóa dịch chuyển k = 3, biến A thành D.

=== Page 2 ===
2. HỆ MÃ AFFINE:
- Hàm mã hóa: e(x) = (a * x + b) mod 26 với điều kiện gcd(a, 26) = 1.
- Hàm giải mã: d(y) = a^(-1) * (y - b) mod 26.
- Khóa k = (a, b) thuộc không gian Z26 x Z26 với 312 khóa khả dụng.

=== Page 3 ===
3. HỆ MÃ VIGENERE:
- Hệ mã thay thế đa bảng chữ cái sử dụng từ khóa lặp lại độ dài m: K = (k1, k2, ..., km).
- Phép cộng modulo 26 cho từng ký tự tương ứng giúp làm phẳng phân bố tần suất chữ cái.
- Khắc phục điểm yếu phân tích tần suất đơn bảng nhưng vẫn có thể bị phá bằng phương pháp Kasiski.

=== Page 4 ===
4. AN NINH HỆ MÃ:
- An ninh vô điều kiện: Không thể phá mã dù đối phương có tài nguyên tính toán vô hạn (ví dụ One-Time Pad).
- An ninh tính toán: Chi phí phá mã vượt quá giá trị thông tin hoặc thời gian hữu dụng của dữ liệu.
"""

SHORT_SINGLE_SECTION_TEXT = """=== Page 1 ===
1. GIAO THỨC HTTP VÀ CƠ CHẾ HOẠT ĐỘNG:
- HTTP (Hypertext Transfer Protocol) là giao thức truyền tải siêu văn bản ở tầng ứng dụng theo mô hình Client-Server.
- Giao thức hoạt động theo cơ chế Request-Response phi trạng thái (stateless), sử dụng các phương thức phổ biến như GET, POST, PUT, DELETE.
- Ví dụ: Khi trình duyệt gửi yêu cầu GET /index.html, máy chủ phản hồi mã trạng thái 200 OK kèm nội dung trang web.
- Phiên bản HTTPS bổ sung lớp mã hóa TLS/SSL để bảo vệ tính toàn vẹn và bảo mật dữ liệu trên đường truyền.
"""


def _extract_answer_core(answer_text: str) -> str:
    """Strip trailing suggested questions, synthesis, and citation blocks to compare actual answer bodies."""
    text = (answer_text or "").strip()
    for marker in (
        "### Nguồn tham khảo",
        "💡 **Câu hỏi tự học tiếp theo:**",
        "💡 **Tổng hợp & Đúc kết chuyên sâu:**",
        "💡 **Câu hỏi gợi ý",
        "**Câu hỏi gợi ý",
        "Câu hỏi gợi ý tiếp theo:",
        "📚 **Nguồn trích dẫn",
        "**Nguồn trích dẫn",
        "Nguồn trích dẫn:",
    ):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()
    return re.sub(r"\s+", " ", text).strip()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.mark.parametrize(
    "case_name, doc_payload, opening_query, min_distinct_cores",
    [
        (
            "sa_cms_multi",
            {"existing_doc_id": "DEMO_DOC_001"},
            "Kiến trúc bộ nhớ liên tục hoạt động như thế nào?",
            3,
        ),
        (
            "crypto_slide_multi",
            {
                "document_id": "DOC_TEST_CRYPTO_FOLLOWUP",
                "title": "Slide Mật Mã Cổ Điển Test Followup",
                "text": CRYPTO_SLIDE_TEXT,
                "category": "Cryptography",
            },
            "Hệ mã dịch vòng hoạt động như thế nào?",
            3,
        ),
        (
            "short_single_section",
            {
                "document_id": "DOC_TEST_SHORT_FOLLOWUP",
                "title": "Giao Thức HTTP Ngắn",
                "text": SHORT_SINGLE_SECTION_TEXT,
                "category": "Networks",
            },
            "Giao thức HTTP là gì?",
            2,
        ),
    ],
)
def test_followup_questions_produce_distinct_grounded_answers(
    client, case_name, doc_payload, opening_query, min_distinct_cores
):
    created_doc_id = None
    session_id = None

    try:
        if "existing_doc_id" in doc_payload:
            doc_id = doc_payload["existing_doc_id"]
        else:
            doc_res = client.post("/api/documents/raw", json=doc_payload)
            assert doc_res.status_code == 200, f"[{case_name}] Failed to create raw doc: {doc_res.text}"
            doc_id = doc_res.json().get("document_id") or doc_payload["document_id"]
            created_doc_id = doc_id

        sess_res = client.post("/api/chat/sessions", json={"title": f"Followup Test {case_name}"})
        assert sess_res.status_code == 200
        session_id = sess_res.json()["session_id"]

        attach_res = client.post(
            f"/api/chat/sessions/{session_id}/documents",
            json={"document_id": doc_id},
        )
        assert attach_res.status_code == 200

        open_res = client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={"query": opening_query, "answer_mode": "balanced"},
        )
        assert open_res.status_code == 200
        open_data = open_res.json()
        assert open_data.get("refused") is False, f"[{case_name}] Opening query refused: {open_data}"

        suggested = open_data.get("suggested_questions", [])
        assert len(suggested) == 3, f"[{case_name}] Expected 3 suggested_questions, got {suggested}"
        assert len({q.strip().lower() for q in suggested}) == 3, (
            f"[{case_name}] Suggested questions are not distinct: {suggested}"
        )

        cores = []
        for idx, q in enumerate(suggested):
            follow_res = client.post(
                f"/api/chat/sessions/{session_id}/messages",
                json={"query": q, "answer_mode": "balanced"},
            )
            assert follow_res.status_code == 200
            follow_data = follow_res.json()
            assert follow_data.get("refused") is False, (
                f"[{case_name}] Follow-up #{idx + 1} '{q}' was refused: {follow_data.get('refusal_reason')}"
            )
            raw_answer = follow_data.get("answer") or follow_data.get("content") or ""
            core = _extract_answer_core(raw_answer)
            assert core, f"[{case_name}] Empty answer core for follow-up '{q}'"
            cores.append(core)

        distinct_count = len(set(cores))
        assert distinct_count >= min_distinct_cores, (
            f"[{case_name}] Expected >= {min_distinct_cores} distinct answer cores, "
            f"got {distinct_count}.\nQuestions: {suggested}\nCores: {cores}"
        )
    finally:
        if session_id:
            client.delete(f"/api/chat/sessions/{session_id}")
        if created_doc_id:
            client.delete(f"/api/documents/{created_doc_id}")

"""
Regression tests reproducing the 2 real-world bug scenarios reported by the user:
- Scenario 1 (Image 1): Multi-doc session (Math + Chemistry attached), typo query
  "chất thử và chất oxi hóa là gì". Must NOT leak hardcoded crypto text
  ("hệ mã khóa đối xứng cổ điển"), must NOT drift to Math ("ax^2 + bx + c"),
  must cover both chất khử and chất oxi hóa with grounded reasoning, and must
  NOT emit false-positive "Lưu ý đối soát" warnings.
- Scenario 2 (Image 2): Single-doc locked session ("Bài 2" Chemistry), asking
  "chất thử và chất oxi hóa là gì" and follow-up reasoning
  "tại sao trong phản ứng oxi hóa khử phải bảo toàn electron". Must NOT leak
  crypto text, must NOT falsely flag valid numbers (25, 24.79, 0.6, 48.4),
  and must provide grounded definitions and reasoning.
"""

import pytest
from fastapi.testclient import TestClient

from src.web.app import app


MATH_DOC_TITLE = "Bo Tai Lieu Mau Co Cau Truc Toan Dai So"
MATH_DOC_TEXT = """# PHẦN 1: ĐẠI SỐ - PHƯƠNG TRÌNH BẬC HAI

## 1.1. Định nghĩa và công thức nghiệm
Phương trình bậc hai một ẩn có dạng tổng quát là ax^2 + bx + c = 0 (với a != 0).
Biệt thức Delta được tính theo công thức Delta = b^2 - 4ac.
Nếu Delta > 0 thì phương trình có hai nghiệm phân biệt x1, x2.

## 1.2. Hệ thức Vi-ét
Nếu phương trình ax^2 + bx + c = 0 (a != 0) có hai nghiệm x1, x2 thì theo hệ thức Vi-ét:
- Tổng hai nghiệm: x1 + x2 = -b / a
- Tích hai nghiệm: x1 * x2 = c / a

## Ví dụ 1.1
Giải phương trình bậc hai: 2x^2 - 7x + 3 = 0.
Ta có Delta = (-7)^2 - 4 * 2 * 3 = 49 - 24 = 25 > 0.
Do đó phương trình có hai nghiệm phân biệt x1 = 3 và x2 = 0.5.
"""

CHEM_DOC_TITLE = "Bài 2"
CHEM_DOC_TEXT = """# Bài 2: PHẢN ỨNG OXI HÓA - KHỬ VÀ BẢO TOÀN ELECTRON

## 2.1. Khái niệm cơ bản
- Chất khử (chất bị oxi hóa) là chất nhường electron, có số oxi hóa tăng sau phản ứng.
- Chất oxi hóa (chất bị khử) là chất nhận electron, có số oxi hóa giảm sau phản ứng.
- Quá trình oxi hóa (sự oxi hóa) là quá trình nhường electron; quá trình khử (sự khử) là quá trình nhận electron.
- Phản ứng oxi hóa - khử là phản ứng hóa học trong đó có sự chuyển electron giữa các chất tham gia, nghĩa là quá trình nhường và nhận electron luôn diễn ra đồng thời.

## 2.2. Nguyên tắc bảo toàn electron
Trong một phản ứng oxi hóa - khử, tổng số mol electron do chất khử nhường luôn bằng tổng số mol electron do chất oxi hóa nhận:
$\\sum n(e\\text{ nhường}) = \\sum n(e\\text{ nhận})$
Nhờ nguyên tắc bảo toàn electron, ta không cần cân bằng toàn bộ phương trình phức tạp mà vẫn thiết lập được mối liên hệ định lượng giữa chất khử và chất oxi hóa.

## Ví dụ 2.1
Đề bài: Hòa tan hoàn toàn 11,2 gam Fe vào dung dịch HNO3 loãng dư, thu được V lít khí NO (sản phẩm khử duy nhất, đo ở ĐKC 25°C, 1 bar) và dung dịch chứa m gam muối Fe(NO3)3.
Bước 1: Tính số mol Fe: n(Fe) = 11,2 / 56 = 0,2 mol. Quá trình oxi hóa: Fe -> Fe3+ + 3e, suy ra tổng số mol electron nhường là 0,2 * 3 = 0,6 mol e.
Bước 2: Áp dụng bảo toàn electron: số mol electron nhận để tạo NO (N+5 + 3e -> N+2) bằng 0,6 mol e, nên n(NO) = 0,6 / 3 = 0,2 mol.
Bước 3: Thể tích khí NO ở ĐKC (25°C, 1 bar, 1 mol khí chiếm 24,79 lít): V = 0,2 * 24,79 = 4,958 lít NO.
Bước 4: Khối lượng muối Fe(NO3)3 thu được là m = 0,2 * 242 = 48,4 gam.
"""


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _create_temp_doc(client: TestClient, title: str, text: str) -> str:
    res = client.post("/api/documents/raw", json={"title": title, "text": text})
    assert res.status_code == 200, f"Failed to create doc '{title}': {res.text}"
    data = res.json()
    assert data.get("document_id")
    return data["document_id"]


def _cleanup_resources(client: TestClient, session_ids: list[str], doc_ids: list[str]) -> None:
    for sid in session_ids:
        client.delete(f"/api/chat/sessions/{sid}")
    for did in doc_ids:
        client.delete(f"/api/documents/{did}")


def test_scenario_1_multi_doc_typo_query_grounded_no_crypto_leak(client: TestClient):
    """
    Kịch bản Ảnh 1: Đính kèm cả Tài liệu 1 (Toán) + Tài liệu 2 (Hóa),
    hỏi 'chất thử và chất oxi hóa là gì'.
    """
    doc_ids: list[str] = []
    session_ids: list[str] = []
    try:
        math_id = _create_temp_doc(client, MATH_DOC_TITLE, MATH_DOC_TEXT)
        doc_ids.append(math_id)
        chem_id = _create_temp_doc(client, CHEM_DOC_TITLE, CHEM_DOC_TEXT)
        doc_ids.append(chem_id)

        sess_res = client.post(
            "/api/chat/sessions",
            json={"title": "Test Scenario 1 Multi-Doc", "document_ids": [math_id, chem_id]},
        )
        assert sess_res.status_code == 200, sess_res.text
        sid = sess_res.json()["session_id"]
        session_ids.append(sid)

        msg_res = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": "chất thử và chất oxi hóa là gì", "mode": "hybrid"},
        )
        assert msg_res.status_code == 200, msg_res.text
        payload = msg_res.json()
        assert payload.get("refused") is False

        answer = payload.get("answer", "")
        answer_lower = answer.lower()

        # 1. Tuyệt đối không leak nội dung mật mã hay lạc sang Toán
        assert "hệ mã khóa đối xứng cổ điển" not in answer_lower
        assert "ax^2 + bx + c" not in answer_lower
        assert "phương trình bậc hai" not in answer_lower

        # 2. Phải nêu đúng trọng tâm cả chất khử (nhường electron) và chất oxi hóa (nhận electron)
        assert "chất khử" in answer_lower
        assert "nhường electron" in answer_lower
        assert "chất oxi hóa" in answer_lower
        assert "nhận electron" in answer_lower

        # 3. Phải có phần suy luận / liên hệ logic ngắn dựa trên tài liệu
        assert ("suy luận" in answer_lower) or ("liên hệ" in answer_lower)

        # 4. Không xuất hiện cảnh báo đối soát sai
        assert "lưu ý đối soát" not in answer_lower
        assert "không có trong tài liệu đối chiếu" not in answer_lower
    finally:
        _cleanup_resources(client, session_ids, doc_ids)


def test_scenario_2_locked_chem_doc_definition_and_reasoning(client: TestClient):
    """
    Kịch bản Ảnh 2: Khóa độc quyền 'Bài 2', hỏi 'chất thử và chất oxi hóa là gì'
    và hỏi thêm câu suy luận 'tại sao trong phản ứng oxi hóa khử phải bảo toàn electron'.
    """
    doc_ids: list[str] = []
    session_ids: list[str] = []
    try:
        chem_id = _create_temp_doc(client, CHEM_DOC_TITLE, CHEM_DOC_TEXT)
        doc_ids.append(chem_id)

        sess_res = client.post(
            "/api/chat/sessions",
            json={"title": "Test Scenario 2 Locked Chem", "document_ids": [chem_id]},
        )
        assert sess_res.status_code == 200, sess_res.text
        sid = sess_res.json()["session_id"]
        session_ids.append(sid)

        # Câu 1: Định nghĩa với lỗi gõ nhẹ 'chất thử'
        res1 = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": "chất thử và chất oxi hóa là gì", "mode": "hybrid"},
        )
        assert res1.status_code == 200, res1.text
        p1 = res1.json()
        assert p1.get("refused") is False
        ans1 = p1.get("answer", "")
        ans1_lower = ans1.lower()

        assert "hệ mã khóa đối xứng cổ điển" not in ans1_lower
        assert "chất khử" in ans1_lower
        assert "nhường electron" in ans1_lower
        assert "chất oxi hóa" in ans1_lower
        assert "nhận electron" in ans1_lower
        assert "lưu ý đối soát" not in ans1_lower
        assert "không có trong tài liệu đối chiếu" not in ans1_lower
        for num in ("25", "24.79", "0.6", "48.4"):
            assert f"chi tiết {num}" not in ans1_lower

        # Câu 2: Suy luận tại sao phải bảo toàn electron
        res2 = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={
                "query": "tại sao trong phản ứng oxi hóa khử phải bảo toàn electron",
                "mode": "hybrid",
            },
        )
        assert res2.status_code == 200, res2.text
        p2 = res2.json()
        assert p2.get("refused") is False
        ans2 = p2.get("answer", "")
        ans2_lower = ans2.lower()

        assert "hệ mã khóa đối xứng cổ điển" not in ans2_lower
        assert "bảo toàn electron" in ans2_lower
        assert ("nhường" in ans2_lower) and ("nhận" in ans2_lower)
        assert "lưu ý đối soát" not in ans2_lower
        assert "không có trong tài liệu đối chiếu" not in ans2_lower
        for num in ("25", "24.79", "0.6", "48.4"):
            assert f"chi tiết {num}" not in ans2_lower
    finally:
        _cleanup_resources(client, session_ids, doc_ids)

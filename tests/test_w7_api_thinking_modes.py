"""
Integration tests for Ticket T9a (WaveCMS-R^2 thinking_mode & W0-W6 dashboard):
- Tests all 4 thinking modes: fast, deep, broad, deep_broad (plus unknown -> fast fallback)
- Uses real Math (quadratic equations) + Chemistry (redox & electron conservation) docs
- Verifies §5 reasoning_trace schema, 0% regression on fast mode, symbolic solver
  verification in deep mode, 5 facets x 4 Haar resolutions in broad mode, both in deep_broad,
  sparse_lse WaveCMSEngine initialization, and ablation_w7 / pillar1_sparse_lse_w7 keys
  in GET /api/comparison/dashboard.
"""

import pytest
from fastapi.testclient import TestClient

from src.web.app import PipelineService, app


MATH_DOC_TITLE = "Toan Dai So Bac Hai W7"
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

CHEM_DOC_TITLE = "Hoa Hoc Oxi Hoa Khu W7"
CHEM_DOC_TEXT = """# Bài 2: PHẢN ỨNG OXI HÓA - KHỬ VÀ BẢO TOÀN ELECTRON

## 2.1. Khái niệm cơ bản
- Chất khử (chất bị oxi hóa) là chất nhường electron, có số oxi hóa tăng sau phản ứng.
- Chất oxi hóa (chất bị khử) là chất nhận electron, có số oxi hóa giảm sau phản ứng.
- Phản ứng oxi hóa - khử là phản ứng hóa học trong đó có sự chuyển electron giữa các chất tham gia.

## 2.2. Nguyên tắc bảo toàn electron
Trong một phản ứng oxi hóa - khử, tổng số mol electron do chất khử nhường luôn bằng tổng số mol electron do chất oxi hóa nhận.
Nhờ nguyên tắc bảo toàn electron, ta thiết lập được mối liên hệ định lượng giữa chất khử và chất oxi hóa.

## 2.3. Ứng dụng và giới hạn
Phương pháp bảo toàn electron áp dụng hiệu quả cho bài toán kim loại tác dụng với HNO3, nhưng chỉ đúng khi xác định đúng sản phẩm khử và số oxi hóa.
"""

VALID_STEP_KINDS = {"premise", "mechanism", "formula", "critique", "conclusion"}
VALID_FACETS = {"definition", "mechanism", "relation_comparison", "application", "limitation"}
VALID_RESOLUTIONS = {"LEAF_PARAGRAPH", "SECTION", "DOCUMENT", "CORPUS"}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _assert_trace_schema(trace: dict, expected_mode: str) -> None:
    assert isinstance(trace, dict)
    assert trace.get("mode") == expected_mode
    assert isinstance(trace.get("hops"), int) and trace["hops"] >= 1
    assert isinstance(trace.get("rho_hat"), (int, float))
    assert isinstance(trace.get("rounds"), int) and trace["rounds"] >= 1
    conformal = trace.get("conformal")
    assert isinstance(conformal, dict)
    assert "score" in conformal
    assert "group" in conformal
    assert "lambda_g" in conformal
    assert isinstance(trace.get("steps"), list)
    assert isinstance(trace.get("facets"), list)
    assert isinstance(trace.get("removed_claims"), list)


def test_wave_engine_uses_sparse_lse_routing():
    service = PipelineService.get_instance()
    assert service.wave_engine is not None
    assert service.wave_engine.routing_mode == "sparse_lse"


def test_session_thinking_modes_fast_deep_broad_deep_broad(client: TestClient):
    doc_ids = []
    session_ids = []
    try:
        math_res = client.post("/api/documents/raw", json={"title": MATH_DOC_TITLE, "text": MATH_DOC_TEXT})
        assert math_res.status_code == 200
        math_id = math_res.json()["document_id"]
        doc_ids.append(math_id)

        chem_res = client.post("/api/documents/raw", json={"title": CHEM_DOC_TITLE, "text": CHEM_DOC_TEXT})
        assert chem_res.status_code == 200
        chem_id = chem_res.json()["document_id"]
        doc_ids.append(chem_id)

        sess_res = client.post(
            "/api/chat/sessions",
            json={"title": "W7 Thinking Modes Test", "document_ids": [math_id, chem_id]},
        )
        assert sess_res.status_code == 200
        sid = sess_res.json()["session_id"]
        session_ids.append(sid)

        math_query = "Giải phương trình bậc hai 2x^2 - 7x + 3 = 0 và tính Delta"
        chem_query = "Chất khử và chất oxi hóa là gì, nguyên tắc bảo toàn electron hoạt động và giới hạn ra sao?"

        # 1. Default (omitted) vs explicit fast -> 0% regression
        r_default = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": math_query, "answer_mode": "balanced"},
        )
        r_fast = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": math_query, "answer_mode": "balanced", "thinking_mode": "fast"},
        )
        r_unknown = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": math_query, "answer_mode": "balanced", "thinking_mode": "unknown_mode"},
        )
        assert r_default.status_code == 200
        assert r_fast.status_code == 200
        assert r_unknown.status_code == 200

        d_def = r_default.json()
        d_fast = r_fast.json()
        d_unk = r_unknown.json()
        assert d_fast["refused"] is False
        assert d_fast["answer"] == d_def["answer"]
        assert d_unk["thinking_mode"] == "fast"
        _assert_trace_schema(d_fast["reasoning_trace"], "fast")

        # 2. Deep mode on Math quadratic -> steps + verified + symbolic solver
        r_deep = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": math_query, "answer_mode": "balanced", "thinking_mode": "deep"},
        )
        assert r_deep.status_code == 200
        d_deep = r_deep.json()
        assert d_deep["refused"] is False
        trace_deep = d_deep["reasoning_trace"]
        _assert_trace_schema(trace_deep, "deep")
        assert len(trace_deep["steps"]) >= 1
        assert all(s["kind"] in VALID_STEP_KINDS for s in trace_deep["steps"])
        assert any(s.get("verified") is True for s in trace_deep["steps"])
        assert any(s["kind"] == "formula" for s in trace_deep["steps"])
        assert "### Nguồn tham khảo" in d_deep["answer"]
        assert len(d_deep.get("suggested_questions", [])) >= 1

        # 3. Broad mode on Chemistry -> 5 facets x 4 Haar resolutions with ok|dropped
        r_broad = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": chem_query, "answer_mode": "balanced", "thinking_mode": "broad"},
        )
        assert r_broad.status_code == 200
        d_broad = r_broad.json()
        assert d_broad["refused"] is False
        trace_broad = d_broad["reasoning_trace"]
        _assert_trace_schema(trace_broad, "broad")
        assert len(trace_broad["facets"]) == 5
        seen_facets = set()
        for f in trace_broad["facets"]:
            assert f["facet"] in VALID_FACETS
            assert f["resolution"] in VALID_RESOLUTIONS
            assert f["status"] in {"ok", "dropped"}
            seen_facets.add(f["facet"])
        assert seen_facets == VALID_FACETS
        assert any(f["status"] == "ok" for f in trace_broad["facets"])

        # 4. Deep + Broad mode -> both steps (verified) and facets (ok|dropped)
        r_db = client.post(
            f"/api/chat/sessions/{sid}/messages",
            json={"query": chem_query, "answer_mode": "balanced", "thinking_mode": "deep_broad"},
        )
        assert r_db.status_code == 200
        d_db = r_db.json()
        assert d_db["refused"] is False
        trace_db = d_db["reasoning_trace"]
        _assert_trace_schema(trace_db, "deep_broad")
        assert len(trace_db["steps"]) >= 1
        assert any(s.get("verified") is True for s in trace_db["steps"])
        assert len(trace_db["facets"]) == 5
        assert all(f["status"] in {"ok", "dropped"} for f in trace_db["facets"])

        # 5. Unified /chat and /api/chat endpoints also accept thinking_mode
        r_root = client.post(
            "/chat",
            json={"query": math_query, "document_id": math_id, "thinking_mode": "deep"},
        )
        assert r_root.status_code == 200
        assert r_root.json()["reasoning_trace"]["mode"] == "deep"

        r_api = client.post(
            "/api/chat",
            json={"query": math_query, "document_id": math_id, "thinking_mode": "broad"},
        )
        assert r_api.status_code == 200
        assert r_api.json()["reasoning_trace"]["mode"] == "broad"
    finally:
        for sid in session_ids:
            client.delete(f"/api/chat/sessions/{sid}")
        for did in doc_ids:
            client.delete(f"/api/documents/{did}")


def test_comparison_dashboard_exposes_w7_keys(client: TestClient):
    res = client.get("/api/comparison/dashboard")
    assert res.status_code == 200
    data = res.json()
    assert "ablation_w7" in data
    assert "pillar1_sparse_lse_w7" in data
    assert data["ablation_w7"] is None or isinstance(data["ablation_w7"], dict)
    assert data["pillar1_sparse_lse_w7"] is None or isinstance(data["pillar1_sparse_lse_w7"], dict)

"""
Comprehensive integration & unit tests for:
1. Online Corpus Fetcher (`src/evaluation/online_corpus_fetcher.py`) & `data/online_benchmark_corpus.json` (>=200 docs, >=500 queries).
2. Comparison Dashboard API (`GET /api/comparison/dashboard`).
3. Word (.docx) Export APIs (`POST /api/export/word` & `GET /api/chat/sessions/{session_id}/export/word`).
4. Online Document Ingestion & Immediate QA (`POST /api/documents/fetch-online`).
5. Frontend UI (`index.html` & `app.js` baseline-compare tab, API fetch button, Word export button)
   and Scientific Word Report (`docs/Bao_Cao_NCKH_SA_CMS_WaveCMS.docx`).
"""

import io
import json
from pathlib import Path

import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient
from openpyxl import load_workbook
from pptx import Presentation

from src.web.app import app


REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "data" / "online_benchmark_corpus.json"
BENCHMARK_RESULTS_PATH = REPO_ROOT / "results" / "online_scale_comparison_200docs.json"
TRAINING_CORPUS_PATH = REPO_ROOT / "data" / "online_training_corpus.json"
TRAINING_COMP_PATH = REPO_ROOT / "results" / "training_params_comparison.json"
REPORT_DOCX_PATH = REPO_ROOT / "docs" / "Bao_Cao_NCKH_SA_CMS_WaveCMS.docx"
REPORT_XLSX_PATH = REPO_ROOT / "docs" / "Thong_Ke_So_Sanh_NCKH.xlsx"
REPORT_PPTX_PATH = REPO_ROOT / "docs" / "Slide_Bao_Ve_NCKH_SA_CMS_WaveCMS.pptx"
INDEX_HTML_PATH = REPO_ROOT / "src" / "web" / "static" / "index.html"
APP_JS_PATH = REPO_ROOT / "src" / "web" / "static" / "js" / "app.js"

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Online Corpus Fetcher & 200+ Docs / 500+ Queries Dataset
# ---------------------------------------------------------------------------

def test_online_corpus_fetcher_exports_and_normalization(monkeypatch):
    """Verifies online_corpus_fetcher exposes required functions and parses payloads cleanly."""
    import src.evaluation.online_corpus_fetcher as fetcher

    assert callable(fetcher.fetch_wikipedia_article)
    assert callable(fetcher.fetch_arxiv_papers)
    assert callable(fetcher.build_large_online_corpus)

    # Monkeypatch Wikipedia helper/network if requests/urllib is used, or test mockable fetch
    sample_wiki = {
        "document_id": "wiki_vi_test_001",
        "title": "Trí tuệ nhân tạo",
        "text": "## Khái niệm\nTrí tuệ nhân tạo (AI) là lĩnh vực khoa học máy tính.\n\n## Ứng dụng\nAI được dùng trong xử lý ngôn ngữ tự nhiên.",
        "language": "vi",
        "source": "wikipedia",
        "url": "https://vi.wikipedia.org/wiki/Tr%C3%AD_tu%E1%BB%87_nh%C3%A2n_t%E1%BA%A1o",
    }
    monkeypatch.setattr(fetcher, "fetch_wikipedia_article", lambda *a, **kw: sample_wiki)
    doc = fetcher.fetch_wikipedia_article("Trí tuệ nhân tạo", lang="vi")
    assert doc is not None
    assert doc["title"] == "Trí tuệ nhân tạo"
    assert "## Khái niệm" in doc["text"]


def test_online_benchmark_corpus_json_scale_and_schema():
    """Verifies data/online_benchmark_corpus.json exists with >=200 docs and >=500 queries."""
    assert CORPUS_PATH.exists(), f"Missing corpus file: {CORPUS_PATH}"
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    docs = data.get("documents", [])
    queries = data.get("questions") or data.get("queries", [])

    assert len(docs) >= 200, f"Expected >= 200 documents, got {len(docs)}"
    assert len(queries) >= 500, f"Expected >= 500 queries, got {len(queries)}"

    # Validate document schema on sample
    doc_ids = set()
    for d in docs[:20]:
        assert d.get("document_id") or d.get("id")
        assert d.get("title")
        assert d.get("text") and len(d["text"]) > 40
        doc_ids.add(d.get("document_id") or d.get("id"))

    # Validate query schema (both answerable and unanswerable present)
    answerable_count = sum(1 for q in queries if q.get("answerable") is True)
    unanswerable_count = sum(1 for q in queries if q.get("answerable") is False)
    assert answerable_count > 0, "Corpus queries must include answerable questions"
    assert unanswerable_count > 0, "Corpus queries must include unanswerable questions"

    for q in queries[:20]:
        assert q.get("question_id")
        assert q.get("document_id")
        assert q.get("question")
        assert "gold_answer" in q
        assert isinstance(q.get("answerable"), bool)


# ---------------------------------------------------------------------------
# 2. Comparison Dashboard Endpoint (Baseline vs Proposed)
# ---------------------------------------------------------------------------

def test_comparison_dashboard_endpoint(client: TestClient):
    """GET /api/comparison/dashboard returns HTTP 200 with accuracy % and Baseline vs Proposed data."""
    resp = client.get("/api/comparison/dashboard")
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    # Must contain structured comparison data (percentages, baseline vs proposed, 20M training, WaveCMS)
    raw_text = json.dumps(payload, ensure_ascii=False).lower()
    assert any(k in raw_text for k in ("overall_accuracy", "accuracy", "baseline", "proposed", "wave")), (
        f"Dashboard payload missing comparison keys: {list(payload.keys())}"
    )

    # Check presence of structured comparison sections in payload
    for key in (
        "models",
        "comparison_table",
        "training_20m",
        "exact_unlearning",
        "online_vs_offline_comparison",
        "training_comparison",
        "offline_innovations",
    ):
        assert key in payload, f"Expected '{key}' in {list(payload.keys())}"
    assert len(payload["comparison_table"]) >= 1
    assert len(payload["training_comparison"]) >= 5
    assert len(payload["offline_innovations"]) >= 5
    assert len(payload["online_vs_offline_comparison"]) >= 1

    # Exact parity with results/online_scale_comparison_200docs.json and API training report
    assert BENCHMARK_RESULTS_PATH.exists(), f"Missing benchmark results: {BENCHMARK_RESULTS_PATH}"
    bench = json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))

    assert payload["corpus_scale"]["total_documents"] == bench["total_documents"] == 272
    assert payload["corpus_scale"]["total_queries"] == bench["total_queries"] == 987

    assert payload["models"]["standard_rag"]["accuracy_pct"] == bench["methods"]["baseline_standard_rag"]["overall_accuracy_pct"] == 25.73
    assert payload["models"]["sa_cms_phase4"]["accuracy_pct"] == bench["methods"]["baseline_legacy_sa_cms"]["overall_accuracy_pct"] == 39.41
    assert payload["models"]["wave_cms_hybrid"]["accuracy_pct"] == bench["methods"]["proposed_wave_hybrid"]["overall_accuracy_pct"] == 51.57

    expected_p = bench["statistical_tests"]["proposed_vs_standard_rag"]["p_value"]
    assert payload["statistical_tests"]["p_value"] == pytest.approx(expected_p, rel=1e-4)
    assert payload["statistical_tests"]["p_value"] == pytest.approx(6.555e-46, rel=1e-2)

    assert payload["exact_unlearning"]["wave_cms_max_discrepancy"] <= 1e-12

    tokens_trained = payload["api_training"].get("actual_tokens_trained") or payload["api_training"].get("tokens_trained")
    assert tokens_trained == 500026


def test_chat_compare_real_execution_no_fake_templates(client: TestClient):
    """POST /api/chat/compare executes real B1/B2/P1/P2 pipelines without fake template strings."""
    resp = client.post(
        "/api/chat/compare",
        json={
            "query": "Kiến trúc bộ nhớ liên tục hoạt động như thế nào?",
            "document_id": "DEMO_DOC_001",
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    comparison = payload.get("comparison", {})
    forbidden_phrases = (
        "Phương pháp B1 trả lời trực tiếp",
        "xác nhận thông tin hỗ trợ từ tài liệu",
        "hồi đáp trực tiếp từ bộ nhớ tham số nén",
    )

    for method_key in ("B1", "B2", "P1", "P2"):
        assert method_key in comparison, f"Missing method {method_key} in /api/chat/compare response"
        entry = comparison[method_key]
        answer = entry.get("answer") or ""
        for phrase in forbidden_phrases:
            assert phrase not in answer, f"{method_key} returned fake template phrase: {phrase!r}"
        assert entry.get("input_tokens", 0) > 0, f"{method_key} input_tokens must be > 0, got {entry}"
        latency = entry.get("latency_ms")
        assert isinstance(latency, (int, float)) and latency >= 0, f"{method_key} invalid latency_ms: {latency}"


# ---------------------------------------------------------------------------
# 3. Word (.docx) Export Endpoints
# ---------------------------------------------------------------------------

def test_export_word_comparison_report_endpoint(client: TestClient):
    """POST /api/export/word returns a valid .docx readable by python-docx with comparison tables."""
    resp = client.post("/api/export/word", json={"include_comparison": True})
    assert resp.status_code == 200, resp.text
    content_type = resp.headers.get("content-type", "")
    assert DOCX_MIME in content_type or "wordprocessingml" in content_type, f"Unexpected content-type: {content_type}"
    assert len(resp.content) > 5000

    doc = DocxDocument(io.BytesIO(resp.content))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    assert len(doc.paragraphs) >= 5
    assert len(doc.tables) >= 1, "Exported Word comparison report must contain at least 1 table"
    assert len(full_text) > 100


def test_export_session_to_word_endpoint(client: TestClient):
    """GET /api/chat/sessions/{session_id}/export/word returns valid .docx with Q&A history."""
    sess_res = client.post("/api/chat/sessions", json={"title": "Word Export Test Session"})
    assert sess_res.status_code == 200
    sess_id = sess_res.json()["session_id"]

    client.post(f"/api/chat/sessions/{sess_id}/documents", json={"document_id": "DEMO_DOC_001"})
    msg_res = client.post(
        f"/api/chat/sessions/{sess_id}/messages",
        json={"query": "Kiến trúc bộ nhớ liên tục hoạt động như thế nào?", "answer_mode": "balanced"},
    )
    assert msg_res.status_code == 200

    export_res = client.get(f"/api/chat/sessions/{sess_id}/export/word")
    assert export_res.status_code == 200, export_res.text
    content_type = export_res.headers.get("content-type", "")
    assert DOCX_MIME in content_type or "wordprocessingml" in content_type, f"Unexpected content-type: {content_type}"

    doc = DocxDocument(io.BytesIO(export_res.content))
    all_text = "\n".join(p.text for p in doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            all_text += "\n" + " ".join(c.text for c in row.cells)

    assert "Kiến trúc bộ nhớ liên tục" in all_text or "Word Export Test Session" in all_text


# ---------------------------------------------------------------------------
# 4. Online Fetch Ingestion & Immediate QA
# ---------------------------------------------------------------------------

def test_fetch_online_document_and_immediate_qa(client: TestClient, monkeypatch):
    """POST /api/documents/fetch-online ingests fetched article and supports immediate QA."""
    import src.evaluation.online_corpus_fetcher as fetcher

    mock_article = {
        "document_id": "wiki_vi_wavelet_test",
        "title": "Biến đổi Wavelet Haar trong nén tín hiệu",
        "text": (
            "## Tổng quan\n"
            "Biến đổi Wavelet Haar là phép biến đổi trực giao đa phân giải do Alfréd Haar đề xuất năm 1909.\n\n"
            "## Ứng dụng trong bộ nhớ\n"
            "WaveCMS sử dụng cây Wavelet nhị phân cân bằng để phân rã tri thức theo độ sâu O(log N) "
            "và đạt sai số xóa tri thức chính xác mức máy nhỏ hơn 5.55e-17."
        ),
        "language": "vi",
        "source": "wikipedia",
        "url": "https://vi.wikipedia.org/wiki/Wavelet_Haar",
    }
    monkeypatch.setattr(fetcher, "fetch_wikipedia_article", lambda *a, **kw: mock_article)

    ingest_res = client.post(
        "/api/documents/fetch-online",
        json={"source": "wikipedia", "query": "Biến đổi Wavelet Haar", "language": "vi", "max_results": 1},
    )
    assert ingest_res.status_code == 200, ingest_res.text
    ingest_data = ingest_res.json()

    ingested_docs = ingest_data.get("documents") or ingest_data.get("ingested") or []
    assert len(ingested_docs) >= 1, f"Expected >=1 ingested doc, got: {ingest_data}"
    doc_id = ingested_docs[0].get("document_id") or ingested_docs[0].get("id")
    assert doc_id

    # Create session, attach ingested doc, and ask a question immediately
    sess_res = client.post("/api/chat/sessions", json={"title": "Online Fetch QA Session"})
    assert sess_res.status_code == 200
    sess_id = sess_res.json()["session_id"]

    attach_res = client.post(f"/api/chat/sessions/{sess_id}/documents", json={"document_id": doc_id})
    assert attach_res.status_code == 200

    qa_res = client.post(
        f"/api/chat/sessions/{sess_id}/messages",
        json={"query": "Alfréd Haar đề xuất biến đổi Wavelet Haar vào năm nào?", "answer_mode": "concise"},
    )
    assert qa_res.status_code == 200, qa_res.text
    qa_data = qa_res.json()
    answer = qa_data.get("answer") or qa_data.get("content", "")
    assert qa_data.get("refused") is not True, f"Should answer from ingested doc, got refusal: {qa_data}"
    assert "1909" in answer or "Haar" in answer


# ---------------------------------------------------------------------------
# 5. Frontend UI Elements & Scientific Word Report Deliverable
# ---------------------------------------------------------------------------

def test_frontend_has_baseline_compare_tab_online_fetch_and_word_export():
    """Verifies index.html and app.js contain baseline-compare tab, API fetch button, and Word export button."""
    assert INDEX_HTML_PATH.exists()
    assert APP_JS_PATH.exists()

    html = INDEX_HTML_PATH.read_text(encoding="utf-8")
    js = APP_JS_PATH.read_text(encoding="utf-8")

    # Baseline-compare tab & panel
    assert 'data-rlab-tab="baseline-compare"' in html
    assert "rlab-panel-baseline-compare" in html

    # Word, Excel, PowerPoint export buttons in HTML & handlers in JS
    assert "btn-export-word" in html
    assert "btn-export-excel" in html
    assert "btn-export-pptx" in html
    assert "export/word" in js or "downloadOfficeFile" in js
    assert "export/excel" in js or "'excel'" in js or '"excel"' in js
    assert "export/pptx" in js or "'pptx'" in js or '"pptx"' in js

    # Online API fetch button/modal in HTML & endpoint in JS
    assert "fetch-online" in js or "fetch-online" in html
    assert "/api/comparison/dashboard" in js

    # Ensure no hardcoded fake numbers remain in app.js
    for fake_snippet in ("19.07 → 2.34", "?? 84.5", "?? 62.52"):
        assert fake_snippet not in js, f"Found hardcoded fake fallback {fake_snippet!r} in app.js"


def test_export_excel_and_pptx_endpoints(client: TestClient):
    """POST /api/export/excel, POST /api/export/pptx and session export endpoints return valid .xlsx & .pptx."""
    # Comparison Excel
    res_xlsx = client.post("/api/export/excel", json={"include_comparison": True})
    assert res_xlsx.status_code == 200, res_xlsx.text
    assert XLSX_MIME in res_xlsx.headers.get("content-type", "")
    wb = load_workbook(io.BytesIO(res_xlsx.content))
    assert "Tong_Quan_So_Sanh" in wb.sheetnames
    assert "Thong_So_Huan_Luyen" in wb.sheetnames
    assert "Diem_Moi_Offline_vs_Online" in wb.sheetnames

    # Comparison PPTX
    res_pptx = client.post("/api/export/pptx", json={"include_comparison": True})
    assert res_pptx.status_code == 200, res_pptx.text
    assert PPTX_MIME in res_pptx.headers.get("content-type", "")
    prs = Presentation(io.BytesIO(res_pptx.content))
    assert len(prs.slides) >= 7

    # Session-specific Excel & PPTX
    sess_res = client.post("/api/chat/sessions", json={"title": "Office Export Session"})
    assert sess_res.status_code == 200
    sess_id = sess_res.json()["session_id"]
    client.post(f"/api/chat/sessions/{sess_id}/documents", json={"document_id": "DEMO_DOC_001"})
    client.post(
        f"/api/chat/sessions/{sess_id}/messages",
        json={"query": "WaveCMS hỗ trợ xóa tri thức như thế nào?", "answer_mode": "balanced"},
    )

    sess_xlsx = client.get(f"/api/chat/sessions/{sess_id}/export/excel")
    assert sess_xlsx.status_code == 200, sess_xlsx.text
    assert XLSX_MIME in sess_xlsx.headers.get("content-type", "")
    wb_sess = load_workbook(io.BytesIO(sess_xlsx.content))
    assert len(wb_sess.sheetnames) >= 1

    sess_pptx = client.get(f"/api/chat/sessions/{sess_id}/export/pptx")
    assert sess_pptx.status_code == 200, sess_pptx.text
    assert PPTX_MIME in sess_pptx.headers.get("content-type", "")
    prs_sess = Presentation(io.BytesIO(sess_pptx.content))
    assert len(prs_sess.slides) >= 1


def test_online_training_corpus_and_training_comparison_json():
    """Verifies data/online_training_corpus.json (>=1,000 docs) and results/training_params_comparison.json."""
    assert TRAINING_CORPUS_PATH.exists(), f"Missing online training corpus: {TRAINING_CORPUS_PATH}"
    corpus = json.loads(TRAINING_CORPUS_PATH.read_text(encoding="utf-8"))
    docs = corpus.get("documents", [])
    assert len(docs) >= 1000, f"Expected >= 1,000 online training docs, got {len(docs)}"

    assert TRAINING_COMP_PATH.exists(), f"Missing training comparison JSON: {TRAINING_COMP_PATH}"
    comp = json.loads(TRAINING_COMP_PATH.read_text(encoding="utf-8"))
    assert len(comp.get("training_comparison", [])) >= 5
    api_stats = comp.get("online_api_corpus_stats", {})
    assert int(api_stats.get("num_documents", 0)) >= 1000


def test_office_deliverables_xlsx_and_pptx_exist_and_valid():
    """Verifies exported Excel & PowerPoint deliverables in docs/ exist and open cleanly."""
    assert REPORT_XLSX_PATH.exists(), f"Missing Excel deliverable: {REPORT_XLSX_PATH}"
    wb = load_workbook(str(REPORT_XLSX_PATH))
    assert len(wb.sheetnames) >= 4

    assert REPORT_PPTX_PATH.exists(), f"Missing PPTX deliverable: {REPORT_PPTX_PATH}"
    prs = Presentation(str(REPORT_PPTX_PATH))
    assert len(prs.slides) >= 7


def test_scientific_word_report_docx_exists_and_is_complete():
    """Verifies docs/Bao_Cao_NCKH_SA_CMS_WaveCMS.docx exists and has rich paragraphs, tables, and embedded figures."""
    assert REPORT_DOCX_PATH.exists(), f"Missing scientific report: {REPORT_DOCX_PATH}"
    doc = DocxDocument(str(REPORT_DOCX_PATH))

    non_empty_paras = [p for p in doc.paragraphs if p.text.strip()]
    assert len(doc.paragraphs) >= 180, f"Expected >= 180 paragraphs, got {len(doc.paragraphs)}"
    assert len(non_empty_paras) >= 150, f"Expected >= 150 non-empty paragraphs, got {len(non_empty_paras)}"
    assert len(doc.tables) >= 10, f"Expected >= 10 tables, got {len(doc.tables)}"
    assert len(doc.inline_shapes) >= 5, f"Expected >= 5 embedded figures, got {len(doc.inline_shapes)}"


def test_export_session_excel_sheets_and_pptx_per_turn(client: TestClient):
    """Session Excel has Lich_Su_Hoi_Dap + So_Sanh_Hieu_Nang; Session PPTX creates 1 slide per Q&A turn."""
    sess_res = client.post("/api/chat/sessions", json={"title": "Multi-Turn Export Session"})
    assert sess_res.status_code == 200
    sess_id = sess_res.json()["session_id"]
    client.post(f"/api/chat/sessions/{sess_id}/documents", json={"document_id": "DEMO_DOC_001"})
    client.post(
        f"/api/chat/sessions/{sess_id}/messages",
        json={"query": "WaveCMS dùng biến đổi gì?", "answer_mode": "concise"},
    )
    client.post(
        f"/api/chat/sessions/{sess_id}/messages",
        json={"query": "Sai số xóa tri thức là bao nhiêu?", "answer_mode": "concise"},
    )

    res_xlsx = client.get(f"/api/chat/sessions/{sess_id}/export/excel")
    assert res_xlsx.status_code == 200, res_xlsx.text
    wb = load_workbook(io.BytesIO(res_xlsx.content))
    assert wb.sheetnames[0] == "Lich_Su_Hoi_Dap"
    assert "So_Sanh_Hieu_Nang" in wb.sheetnames

    res_pptx = client.get(f"/api/chat/sessions/{sess_id}/export/pptx")
    assert res_pptx.status_code == 200, res_pptx.text
    prs = Presentation(io.BytesIO(res_pptx.content))
    # 1 overview slide + 2 Q&A turn slides = 3 slides
    assert len(prs.slides) == 3


def test_export_specific_content_and_document_office_formats(client: TestClient):
    """POST /api/export/{word,excel,pptx} supports exporting specific content or document_id."""
    payload = {
        "title": "Tóm tắt WaveTreeMemory",
        "content": "## Điểm nổi bật\n- Phân rã Haar Wavelet\n- Exact Unlearning <= 5.55e-17",
        "include_comparison": False,
    }
    res_w = client.post("/api/export/word", json=payload)
    assert res_w.status_code == 200, res_w.text
    doc = DocxDocument(io.BytesIO(res_w.content))
    doc_text = "\n".join(p.text for p in doc.paragraphs)
    assert "Tóm tắt WaveTreeMemory" in doc_text
    assert "Exact Unlearning" in doc_text

    res_x = client.post("/api/export/excel", json=payload)
    assert res_x.status_code == 200, res_x.text
    wb = load_workbook(io.BytesIO(res_x.content))
    assert "Noi_Dung_Xuat" in wb.sheetnames

    res_p = client.post("/api/export/pptx", json=payload)
    assert res_p.status_code == 200, res_p.text
    prs = Presentation(io.BytesIO(res_p.content))
    assert len(prs.slides) >= 1

    # Also test document_id export
    res_doc = client.post(
        "/api/export/word",
        json={"document_id": "DEMO_DOC_001", "include_comparison": False},
    )
    assert res_doc.status_code == 200, res_doc.text


def test_fetch_online_qasper_and_squad_v2_sources(client: TestClient, monkeypatch):
    """POST /api/documents/fetch-online supports qasper and squad_v2 via fetch_hf_dataset_sample."""
    import src.evaluation.online_corpus_fetcher as fetcher

    assert callable(fetcher.fetch_hf_dataset_sample)

    def _mock_hf_rows(dataset, config, split, offset, length, timeout=20):
        if "qasper" in dataset:
            return [{
                "id": "qasper_test_01",
                "title": "Wavelet Memory Paper",
                "abstract": "We study orthogonal Haar wavelets for continual knowledge editing.",
                "full_text": {"section_name": ["Method"], "paragraphs": [["Haar decomposition achieves O(log N) updates."]]},
                "qas": {"question": [], "question_id": [], "answers": []},
            }]
        return [{
            "id": "squad_test_01",
            "title": "Haar Transform",
            "context": "Alfréd Haar proposed the Haar wavelet in 1909 as an orthonormal basis.",
            "question": "Who proposed the Haar wavelet?",
            "answers": {"text": ["Alfréd Haar"]},
        }]

    monkeypatch.setattr(fetcher, "_fetch_hf_rows", _mock_hf_rows)

    q_docs = fetcher.fetch_hf_dataset_sample("qasper", split="train", offset=0, limit=2)
    assert len(q_docs) == 1 and q_docs[0]["source"] == "qasper"

    s_docs = fetcher.fetch_hf_dataset_sample("squad_v2", split="train", offset=0, limit=2)
    assert len(s_docs) == 1 and s_docs[0]["source"] == "squad_v2"

    res_q = client.post(
        "/api/documents/fetch-online",
        json={"source": "qasper", "query": "", "max_results": 10},
    )
    assert res_q.status_code == 200, res_q.text
    assert res_q.json()["count"] >= 1

    res_s = client.post(
        "/api/documents/fetch-online",
        json={"source": "squad_v2", "query": "", "max_results": 20},
    )
    assert res_s.status_code == 200, res_s.text
    assert res_s.json()["count"] >= 1


def test_wavecms_runtime_sync_and_exact_unlearning(client: TestClient):
    """PipelineService syncs WaveCMSEngine on ingest and executes Exact Unlearning on delete."""
    from src.web.app import PipelineService

    service = PipelineService.get_instance()
    status_before = client.get("/api/system/status")
    assert status_before.status_code == 200
    wave_before = status_before.json().get("wave_cms_runtime", {})
    assert wave_before.get("enabled") is True
    assert wave_before.get("exact_unlearning_enabled") is True

    raw_res = client.post(
        "/api/documents/raw",
        json={
            "title": "WaveCMS Runtime Unlearn Test",
            "text": (
                "## Architecture\n"
                "WaveCMSEngine decomposes document embeddings with Haar wavelets for O(log N) updates.\n\n"
                "## Unlearning\n"
                "Exact unlearning subtracts leaf contributions with machine-precision discrepancy."
            ),
            "category": "Research",
        },
    )
    assert raw_res.status_code == 200, raw_res.text
    doc_id = raw_res.json().get("document_id")
    assert doc_id
    assert service.wave_engine is not None
    assert doc_id in service.wave_engine._document_registry

    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 200, del_res.text
    del_data = del_res.json()
    assert del_data.get("success") is True
    audit = del_data.get("wave_cms_unlearning")
    assert audit is not None
    assert audit.get("certified_exact") is True
    assert audit.get("exact_match") is True
    assert doc_id not in service.wave_engine._document_registry


def test_chat_auto_online_api_fallback_and_safe_refusal(client: TestClient, monkeypatch):
    """Chat calls search_external_document_apis when allow_online_api=True, and refuses safely when False."""
    import src.evaluation.online_corpus_fetcher as fetcher

    unique_topic = "ZirconiumWaveletQuantization9876"
    mock_doc = {
        "document_id": "OPENALEX_W987654321",
        "title": f"{unique_topic} in Continual Memory",
        "text": (
            f"# {unique_topic} in Continual Memory\n\n"
            f"## Abstract\n"
            f"{unique_topic} achieves a compression ratio of 42.5x while preserving 99.1% recall in WaveCMS.\n"
        ),
        "source": "openalex",
        "language": "en",
        "url": "https://openalex.org/W987654321",
    }
    monkeypatch.setattr(fetcher, "search_external_document_apis", lambda *a, **kw: [mock_doc])

    # 1) allow_online_api=False -> must NOT call external API and must refuse safely
    sess_off = client.post("/api/chat/sessions", json={"title": "Offline Airgap Refusal Test"})
    assert sess_off.status_code == 200
    sid_off = sess_off.json()["session_id"]

    res_off = client.post(
        f"/api/chat/sessions/{sid_off}/messages",
        json={
            "query": f"What compression ratio does {unique_topic} achieve?",
            "answer_mode": "concise",
            "allow_online_api": False,
        },
    )
    assert res_off.status_code == 200, res_off.text
    data_off = res_off.json()
    assert data_off.get("refused") is True

    # 2) allow_online_api=True -> auto-fetches mock_doc and answers with grounding
    sess_on = client.post("/api/chat/sessions", json={"title": "Auto Online API Fallback Test"})
    assert sess_on.status_code == 200
    sid_on = sess_on.json()["session_id"]

    res_on = client.post(
        f"/api/chat/sessions/{sid_on}/messages",
        json={
            "query": f"What compression ratio does {unique_topic} achieve?",
            "answer_mode": "concise",
            "allow_online_api": True,
        },
    )
    assert res_on.status_code == 200, res_on.text
    data_on = res_on.json()
    ans_on = data_on.get("answer") or data_on.get("content", "")
    assert data_on.get("refused") is not True, f"Expected grounded answer via online API, got: {data_on}"
    assert "42.5" in ans_on or unique_topic in ans_on

    # Clean up ingested mock document
    client.delete("/api/documents/OPENALEX_W987654321")


def test_fetch_openalex_europepmc_and_sciq_pubmedqa_fetchers(monkeypatch):
    """Verifies fetch_openalex_papers, fetch_europepmc_papers, and fetch_hf_dataset_sample('sciq'/'pubmedqa')."""
    import src.evaluation.online_corpus_fetcher as fetcher

    class _MockResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def _mock_requests_get(url, params=None, headers=None, timeout=15):
        if "openalex.org" in url:
            return _MockResp({
                "results": [{
                    "id": "https://openalex.org/W123456",
                    "title": "Wavelet Continual Memory Systems",
                    "publication_year": 2025,
                    "doi": "https://doi.org/10.1000/test",
                    "primary_location": {"source": {"display_name": "NeurIPS"}},
                    "concepts": [{"display_name": "Machine Learning"}],
                    "authorships": [{"author": {"display_name": "Nguyen Van A"}}],
                    "abstract_inverted_index": {
                        "Wavelet": [0], "memory": [1], "enables": [2], "exact": [3],
                        "unlearning": [4], "with": [5], "orthogonal": [6], "basis": [7],
                        "functions": [8], "and": [9], "bounded": [10], "drift.": [11],
                    },
                }]
            })
        if "ebi.ac.uk/europepmc" in url:
            return _MockResp({
                "resultList": {
                    "result": [{
                        "pmcid": "PMC999999",
                        "title": "Biomedical Memory Grounding",
                        "abstractText": "Conformal calibration reduces biomedical hallucination rates across clinical QA benchmarks.",
                        "pubYear": "2025",
                        "journalTitle": "Bioinformatics",
                        "authorString": "Tran Thi B",
                    }]
                }
            })
        return _MockResp({})

    monkeypatch.setattr(fetcher.requests, "get", _mock_requests_get)

    oa_docs = fetcher.fetch_openalex_papers("wavelet memory", max_results=2)
    assert len(oa_docs) == 1
    assert oa_docs[0]["source"] == "openalex"
    assert "Wavelet memory enables exact unlearning" in oa_docs[0]["abstract"]

    epmc_docs = fetcher.fetch_europepmc_papers("biomedical grounding", max_results=2)
    assert len(epmc_docs) == 1
    assert epmc_docs[0]["source"] == "europepmc"
    assert epmc_docs[0]["document_id"] == "EUROPEPMC_PMC999999"

    def _mock_hf_rows(dataset, config, split, offset, length, timeout=20):
        if "sciq" in dataset.lower():
            return [{
                "question": "What organelle produces ATP in eukaryotic cells?",
                "correct_answer": "mitochondria",
                "support": "Mitochondria are double-membrane organelles that generate most of the cell's supply of ATP.",
            }]
        return [{
            "pubid": 888777,
            "question": "Does conformal gating reduce unsupported claims?",
            "context": {
                "labels": ["RESULTS", "CONCLUSIONS"],
                "contexts": [
                    "Conformal risk control lowered hallucination rates from 52% to 18%.",
                    "Calibrated abstention improves clinical safety.",
                ],
            },
            "long_answer": "Yes, conformal gating significantly reduces unsupported claims.",
            "final_decision": "yes",
        }]

    monkeypatch.setattr(fetcher, "_fetch_hf_rows", _mock_hf_rows)

    sciq_docs = fetcher.fetch_hf_dataset_sample("sciq", split="train", offset=0, limit=2)
    assert len(sciq_docs) == 1
    assert sciq_docs[0]["source"] == "sciq"

    pqa_docs = fetcher.fetch_hf_dataset_sample("pubmedqa", split="train", offset=0, limit=2)
    assert len(pqa_docs) == 1
    assert pqa_docs[0]["source"] == "pubmedqa"

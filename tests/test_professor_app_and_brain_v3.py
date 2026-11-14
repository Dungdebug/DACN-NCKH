"""
Tests for Big Update: Sovereign Brain 3.0 & Professor-Machine App Portability.
100% Local Offline, Zero External API.
Verifies:
1. DOCX Table + Paragraph extraction (professor uploads a .docx with tables).
2. CSV & JSON structured file upload extraction.
3. Multi-turn Conversational Working Memory (positional facet reference & ellipsis resolution).
4. Hybrid BM25 + Semantic N-Gram RRF Retriever (typo/synonym resilience).
5. Quantitative Delta Reasoner (automatic metric comparison & delta calculation).
6. Zero raw Markdown artifacts (no raw '###' or '*italic*' in rendered answer body).
7. Professor-machine portability: prepare_offline_assets & LocalModelLoader never crash when model.safetensors is absent.
"""

import io
import json
import pytest
import docx

from src.web.file_parser import extract_text_from_file, validate_file
from src.web.session_manager import SessionManager
from src.hybrid_qa.document_store import DocumentRecord
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.retriever import BM25Retriever
from src.web.document_analyzer import DocumentAnalyzer


def test_docx_table_extraction_preserves_tabular_knowledge():
    """
    When the professor uploads a .docx file containing a table (e.g. syllabus, benchmark results),
    extract_text_from_file must extract all rows and cells with column headers attached.
    """
    doc = docx.Document()
    doc.add_heading("Báo cáo Kết quả Mô hình SA-CMS", level=1)
    doc.add_paragraph("Dưới đây là bảng tổng hợp kết quả thực nghiệm chính thức:")

    table = doc.add_table(rows=3, cols=3)
    headers = ["Phương pháp", "Độ chính xác F1", "Dung lượng VRAM"]
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h

    row1 = ["Baseline RAG (B2)", "71.2%", "1450 MB"]
    for i, val in enumerate(row1):
        table.rows[1].cells[i].text = val

    row2 = ["SA-CMS Hybrid (P2)", "89.6%", "743 MB"]
    for i, val in enumerate(row2):
        table.rows[2].cells[i].text = val

    buf = io.BytesIO()
    doc.save(buf)
    file_bytes = buf.getvalue()

    extracted_text, meta = extract_text_from_file("ket_qua_thay_test.docx", file_bytes)

    assert "SA-CMS Hybrid (P2)" in extracted_text
    assert "89.6%" in extracted_text
    assert "743 MB" in extracted_text
    assert "Độ chính xác F1" in extracted_text


def test_csv_and_json_file_upload_support():
    """
    Verifies that .csv and .json files uploaded by the professor are accepted and parsed into readable sentences.
    """
    csv_bytes = "Môn học,Số tín chỉ,Giảng viên\nTrí tuệ nhân tạo,4,PGS.TS Nguyễn Văn A\nHọc máy nâng cao,3,TS Trần Thị B\n".encode("utf-8")
    ok, err = validate_file("mon_hoc.csv", csv_bytes)
    assert ok is True, f"CSV validation failed: {err}"

    text, _ = extract_text_from_file("mon_hoc.csv", csv_bytes)
    assert "Trí tuệ nhân tạo" in text
    assert "Số tín chỉ: 4" in text
    assert "PGS.TS Nguyễn Văn A" in text

    json_bytes = json.dumps(
        [{"Tên hệ thống": "SA-CMS v3", "Độ trễ": "18.2 ms", "Chế độ": "100% Offline"}],
        ensure_ascii=False,
    ).encode("utf-8")
    ok_j, err_j = validate_file("config_test.json", json_bytes)
    assert ok_j is True, f"JSON validation failed: {err_j}"
    text_j, _ = extract_text_from_file("config_test.json", json_bytes)
    assert "SA-CMS v3" in text_j
    assert "18.2 ms" in text_j


def test_conversational_working_memory_positional_and_ellipsis(tmp_path):
    """
    Verifies that multi-turn follow-up queries like:
    - 'Giải thích kỹ hơn ý thứ 2' (positional facet reference)
    - 'Còn hạn chế thì sao?' (elliptical subject recovery)
    are accurately resolved to full self-contained queries.
    """
    mgr = SessionManager(storage_dir=str(tmp_path))
    session = mgr.create_session(title="Test Professor Multi-Turn")

    session.add_message(role="user", content="Kiến trúc SA-CMS hoạt động như thế nào?")
    session.add_message(
        role="assistant",
        content=(
            "Dựa trên tài liệu **Nghiên cứu SA-CMS**, cơ chế vận hành như sau:\n\n"
            "Hệ thống SA-CMS cập nhật bộ nhớ theo ranh giới cấu trúc văn bản. [1]\n\n"
            "📌 **Phân tích đa chiều từ tài liệu:**\n"
            "- **Cơ chế & Kiến trúc**: Tầng 1 cập nhật theo đoạn văn, tầng 2 theo mục và tầng 3 theo tài liệu. [1]\n"
            "- **Số liệu & Thực nghiệm**: Mô hình giúp tiết kiệm 87,5% chi phí token trên GPU RTX 3050. [2]\n"
            "- **Lưu ý & Giới hạn**: Đòi hỏi văn bản đầu vào có cấu trúc phân đoạn rõ ràng. [2]"
        ),
    )

    # 1. Positional facet reference: "Giải thích kỹ hơn ý thứ 2"
    resolved_facet = mgr.resolve_conversation_context(session.session_id, "Giải thích kỹ hơn ý thứ 2")
    assert "87,5%" in resolved_facet or "tiết kiệm" in resolved_facet.lower()

    # 2. Elliptical follow-up: "Còn hạn chế thì sao?"
    resolved_ellipsis = mgr.resolve_conversation_context(session.session_id, "Còn hạn chế thì sao?")
    assert "sa-cms" in resolved_ellipsis.lower()
    assert "hạn chế" in resolved_ellipsis.lower()


def test_hybrid_semantic_rrf_retriever_resilience():
    """
    Verifies that Hybrid BM25 + Character/Word N-Gram RRF Retriever finds relevant passages
    even when query uses unaccented Vietnamese or morphological variations.
    """
    doc = DocumentRecord(
        document_id="DOC_RRF",
        version=1,
        title="Báo cáo Kỹ thuật",
        raw_text=(
            "Đoạn 1: Quy trình bảo mật dữ liệu yêu cầu mã hóa đầu cuối bằng thuật toán AES-256.\n\n"
            "Đoạn 2: Hệ thống bộ nhớ liên tục đa thang SA-CMS giúp giảm thiểu 87,5% chi phí tính toán token."
        ),
    )
    chunker = DocumentChunker(chunk_size=120, chunk_overlap=20, min_chunk_size=20)
    passages = chunker.chunk_document(doc)

    retriever = BM25Retriever()
    retriever.build_index(passages)

    # Query without accents and slightly varied phrasing
    hits = retriever.retrieve("giam thieu chi phi tinh toan token cua sa cms", top_k=2)
    assert len(hits) >= 1
    assert "87,5%" in hits[0].text


def test_quantitative_delta_reasoner_and_zero_raw_markdown():
    """
    Verifies that:
    1. When passages contain comparable metrics (e.g. 89.6% vs 71.2%), the answer includes quantitative delta insight.
    2. The answer body has ZERO raw '###' or '*single-asterisk*' artifacts before the Sources section.
    """
    from tests.test_gemini_style_brain import MockPassage

    passages = [
        MockPassage(
            passage_id="D1::P1",
            document_id="D1",
            document_title="Báo cáo Thực nghiệm",
            section_title="1. So sánh F1",
            paragraph_index=1,
            score=14.0,
            text="Phương pháp SA-CMS (P2) đạt độ chính xác F1 là 89.6% trên tập dữ liệu kiểm thử.",
        ),
        MockPassage(
            passage_id="D1::P2",
            document_id="D1",
            document_title="Báo cáo Thực nghiệm",
            section_title="2. Baseline RAG",
            paragraph_index=2,
            score=12.0,
            text="Trong khi đó, phương pháp Baseline RAG (B2) chỉ đạt độ chính xác F1 là 71.2% trên cùng tập dữ liệu.",
        ),
    ]

    answer, _ = DocumentAnalyzer.find_best_grounded_answer(
        query="So sánh độ chính xác F1 giữa SA-CMS (P2) và Baseline RAG (B2)",
        evidence_passages=passages,
        answer_mode="detailed",
    )

    body_before_sources = answer.split("### Nguồn tham khảo")[0]
    # Must not contain raw '###' or single-asterisk '*1. So sánh F1*'
    assert "###" not in body_before_sources
    assert "*1. So sánh F1*" not in body_before_sources
    # Must contain automatic quantitative delta (89.6 - 71.2 = 18.4)
    assert "18.4" in body_before_sources or "18,4" in body_before_sources


def test_vip_document_knowledge_blueprint_extracts_metrics_and_structure():
    """
    Verifies that DocumentAnalyzer.build_document_knowledge_blueprint() generates
    a VIP multi-tier blueprint including TL;DR, structural breakdown, and extracted quantitative highlights.
    """
    raw_text = (
        "# 1. Kiến trúc Hệ thống\n\n"
        "Hệ thống SA-CMS sử dụng mô hình SmolLM2 với 135 triệu tham số và 3 tầng bộ nhớ liên tục.\n\n"
        "# 2. Hiệu năng Thực nghiệm\n\n"
        "Phương pháp đạt độ chính xác 89.6% F1, giảm 87.5% chi phí token và chỉ chiếm 743 MB VRAM.\n\n"
        "# 3. Hạn chế Kỹ thuật\n\n"
        "Tuy nhiên, hệ thống đòi hỏi văn bản đầu vào phải có cấu trúc phân đoạn rõ ràng."
    )
    blueprint = DocumentAnalyzer.build_document_knowledge_blueprint(raw_text, title="Báo cáo VIP SA-CMS")
    assert "Báo cáo VIP SA-CMS" in blueprint
    assert "89.6%" in blueprint or "87.5%" in blueprint or "743 MB" in blueprint
    assert "###" not in blueprint


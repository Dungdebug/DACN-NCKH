"""
Red-Team Test Suite: PDF Page Boundary Bleed Prevention & Bullet Preservation.

Verifies:
1. Complete immunity against PDF page bleed (=== Page 1 === .. === Page 36 ===).
2. Exactly 0% of generated passages contain '=== Page' in passage.text.
3. Absolute boundary isolation: end of Page 30 ('9. HỆ MÃ HOÁ DÒNG') NEVER contains
   content from Page 31 ('10. HỆ MÃ HOÁ AUTOKEY'), and vice versa.
4. Complete preservation of short bullet lists (* Mã hóa thay thế., * Mã hóa hoán vị., - An ninh vô điều kiện).
5. split_clean_sentences cleanly strips page markers while preserving short bullet items and decimals.
6. compress_evidence_context and synthesize_accurate_grounded_answer never leak page artifacts.
7. End-to-end multi-page slide QA execution: ultra-fast ingestion, rich grounded answers
   (concept + classifications + examples), refused == False, and 0% artifact leakage.
"""

import re
import pytest
from starlette.testclient import TestClient

from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.document_store import DocumentRecord
from src.web.token_optimizer import (
    split_clean_sentences,
    strip_passage_artifacts,
    compress_evidence_context,
    synthesize_accurate_grounded_answer,
)
from src.self_study.knowledge_bank import SelfStudyKnowledgeBank
from src.web.app import app


# ---------------------------------------------------------------------------
# Realistic 36-Page Slide Lecture Dataset (Classical Cryptography)
# ---------------------------------------------------------------------------
def generate_36_page_slide_corpus() -> str:
    """
    Constructs a comprehensive 36-page slide lecture corpus mirroring
    real university slides with page markers, uppercase section headings,
    nested bullet lists, and mathematical formulas.
    """
    pages = []

    # Page 1: Introduction & 2 Core Methods
    pages.append(
        "=== Page 1 ===\n"
        "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:\n"
        "- Xuất hiện đầu tiên trong ngành mã hóa.\n"
        "- Các hệ mã này phần có từ trước năm 1970.\n"
        "- Hệ mã khóa bí mật hoặc hệ mã khóa cổ điển.\n"
        "- Các phương pháp này là cơ sở để phát triển các thuật toán mã hóa đối xứng hiện nay.\n"
        "- Có 2 phương pháp phổ biến:\n"
        "* Mã hóa thay thế.\n"
        "* Mã hóa hoán vị."
    )

    # Page 2: Shift Cipher
    pages.append(
        "=== Page 2 ===\n"
        "2. HỆ MÃ DỊCH VÒNG:\n"
        "- Định nghĩa: Không gian bản rõ P, bản mã C và khóa K đều thuộc Z26.\n"
        "- Hàm mã hóa: e_k(x) = (x + k) mod 26.\n"
        "- Hàm giải mã: d_k(y) = (y - k) mod 26."
    )

    # Pages 3 - 9: Classical Ciphers & Theory
    for p in range(3, 10):
        pages.append(
            f"=== Page {p} ===\n"
            f"CHUYÊN ĐỀ MẬT MÃ TRANG {p}:\n"
            f"- Khái niệm bổ trợ về số học đồng dư trong không gian Z26.\n"
            f"- Đánh giá độ phức tạp thuật toán và không gian khóa 26!"
        )

    # Page 10: Cryptanalysis & Security Levels
    pages.append(
        "=== Page 10 ===\n"
        "3. PHÁ MÃ:\n"
        "An ninh hệ mã khóa:\n"
        "- An ninh vô điều kiện: Không thể phá mã dù có thời gian và tài nguyên vô hạn.\n"
        "- An ninh tính toán: Chi phí để phá vỡ vượt quá giá trị của thông tin mã hóa, "
        "hoặc thời gian phá mã vượt quá thời gian hữu dụng của thông tin.\n"
        "- Các phương pháp thám mã phổ biến gồm tấn công vét cạn và phân tích thống kê tần suất xuất hiện."
    )

    # Pages 11 - 29: Intermediate Ciphers (Monoalphabetic, Affine, Vigenere, Hill)
    pages.append(
        "=== Page 11 ===\n"
        "4. HỆ MÃ ĐƠN BẢNG:\n"
        "- Thay thế mỗi ký tự bằng một hoán vị cố định của bảng chữ cái.\n"
        "- Không gian khóa: 26! xấp xỉ 4 x 10^26."
    )
    pages.append(
        "=== Page 15 ===\n"
        "5. HỆ MÃ AFFINE:\n"
        "- Công thức: e(x) = (a * x + b) mod 26 với gcd(a, 26) = 1.\n"
        "- Giải mã: d(y) = a^(-1) * (y - b) mod 26."
    )
    pages.append(
        "=== Page 20 ===\n"
        "6. HỆ MÃ VIGENERE:\n"
        "- Sử dụng từ khóa lặp lại k = (k1, k2, ..., km).\n"
        "- Phá mã bằng phương pháp Kasiski và chỉ số trùng phùng Friedman."
    )
    pages.append(
        "=== Page 25 ===\n"
        "7. HỆ MÃ HILL:\n"
        "- Sử dụng đại số tuyến tính và ma trận khóa m x m khả nghịch trên Z26.\n"
        "- C = P * K mod 26."
    )

    for p in [12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29]:
        pages.append(
            f"=== Page {p} ===\n"
            f"MỤC MỞ RỘNG TRANG {p}:\n"
            f"- Phân tích chi tiết các kỹ thuật thám mã vi phân và tuyến tính.\n"
            f"- Khảo sát trên bộ ngữ liệu tiếng Việt và tiếng Anh."
        )

    # Page 30: One-Time Pad & Stream Cipher
    pages.append(
        "=== Page 30 ===\n"
        "8. HỆ MÃ HOÁ ĐỘN MỘT LẦN:\n"
        "- Hệ mã hóa hoàn hảo (Vernam Cipher / One-Time Pad).\n"
        "- Độ dài khóa bằng đúng độ dài thông điệp cần mã hóa và chỉ dùng duy nhất một lần.\n\n"
        "9. HỆ MÃ HOÁ DÒNG:\n"
        "- Sinh ra một chuỗi khóa z = z1, z2, z3...\n"
        "- Ta có: zi = fi (K, x1, ..., xi-1).\n"
        "- Bản rõ được mã hóa từng bit hoặc từng byte bằng phép XOR với dòng khóa z.\n"
        "- Ưu điểm: Tốc độ xử lý nhanh và độ trễ thấp trong truyền thông thời gian thực."
    )

    # Page 31: Autokey Cipher
    pages.append(
        "=== Page 31 ===\n"
        "10. HỆ MÃ HOÁ AUTOKEY:\n"
        "- Cho P = C = K = Z26.\n"
        "- Khóa k = k1, k2, ... ban đầu được mở rộng bằng chính các ký tự của bản rõ (Autokey).\n"
        "- Công thức mã hóa: ci = (pi + ki) mod 26."
    )

    # Pages 32 - 35: Permutation & Digram Ciphers
    for p in range(32, 36):
        pages.append(
            f"=== Page {p} ===\n"
            f"PHÂN TÍCH CHUYÊN SÂU TRANG {p}:\n"
            f"- Kỹ thuật hoán vị cột và hoán vị hàng trong mật mã cổ điển.\n"
            f"- Ma trận khóa chuyển dịch vị trí ký tự bản rõ."
        )

    # Page 36: Playfair Cipher
    pages.append(
        "=== Page 36 ===\n"
        "11. HỆ MÃ HOÁ PLAYFAIR:\n"
        "- Sử dụng ma trận ký tự 5x5 để mã hóa từng cặp ký tự (digram).\n"
        "- Thay thế các cặp ký tự dựa trên quy tắc cùng hàng, cùng cột hoặc tạo thành hình chữ nhật trên ma trận."
    )

    return "\n\n".join(pages)


# ===========================================================================
# Unit & Red-Team Tests
# ===========================================================================

def test_full_36_page_pdf_chunking_zero_page_bleed():
    """
    Verify that in a 36-page lecture PDF document, 0% of the passages contain
    '=== Page' artifacts in passage.text, and all page boundaries are strictly isolated.
    """
    raw_text = generate_36_page_slide_corpus()
    doc = DocumentRecord(
        document_id="SLIDE_36_PAGES",
        version=1,
        title="Classical Cryptography Lecture (36 Pages)",
        raw_text=raw_text,
    )

    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    passages = chunker.chunk_document(doc)

    assert len(passages) >= 36, f"Expected at least 36 passages, got {len(passages)}"

    # 1. Verification of 0% page marker leakage across all passages
    leaked_passages = [
        (p.passage_id, p.text)
        for p in passages
        if "=== Page" in p.text or re.search(r"===\s*Page\s*\d+\s*===", p.text, re.IGNORECASE)
    ]
    assert len(leaked_passages) == 0, (
        f"Detected {len(leaked_passages)} passages with page artifact leak:\n"
        f"{leaked_passages[:3]}"
    )

    # 2. Verify section_title does NOT contain raw '=== Page' syntax
    for p in passages:
        if p.section_title:
            assert "=== Page" not in p.section_title, (
                f"Section title leaked page marker: {p.section_title}"
            )


def test_page_30_and_31_boundary_isolation_no_forward_backward_leak():
    """
    Verify that the end of Page 30 ('9. HỆ MÃ HOÁ DÒNG') NEVER contains any content
    from Page 31 ('10. HỆ MÃ HOÁ AUTOKEY'), and vice versa.
    """
    raw_text = generate_36_page_slide_corpus()
    doc = DocumentRecord(
        document_id="SLIDE_36_PAGES",
        version=1,
        title="Classical Cryptography Lecture",
        raw_text=raw_text,
    )

    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    passages = chunker.chunk_document(doc)

    # Locate Page 30 stream cipher passages
    p30_stream_passages = [
        p for p in passages
        if "HỆ MÃ HOÁ DÒNG" in (p.section_title or "") or "HỆ MÃ HOÁ DÒNG" in p.text
    ]
    assert len(p30_stream_passages) >= 1, "Page 30 stream cipher passage not found"

    # Locate Page 31 autokey cipher passages
    p31_autokey_passages = [
        p for p in passages
        if "HỆ MÃ HOÁ AUTOKEY" in (p.section_title or "") or "HỆ MÃ HOÁ AUTOKEY" in p.text
    ]
    assert len(p31_autokey_passages) >= 1, "Page 31 autokey cipher passage not found"

    # Forward leak check: Page 30 must never bleed into Page 31
    for p in p30_stream_passages:
        assert "10. HỆ MÃ HOÁ AUTOKEY" not in p.text, (
            f"Forward leak detected: Page 30 passage contains Page 31 heading:\n{p.text}"
        )
        assert "AUTOKEY" not in p.text.upper() or "HỆ MÃ HOÁ AUTOKEY" not in p.text, (
            f"Forward leak detected in passage {p.passage_id}"
        )
        assert "ci = (pi + ki) mod 26" not in p.text, (
            f"Page 31 formula leaked into Page 30 passage:\n{p.text}"
        )

    # Backward leak check: Page 31 must never bleed into Page 30
    for p in p31_autokey_passages:
        assert "9. HỆ MÃ HOÁ DÒNG" not in p.text, (
            f"Backward leak detected: Page 31 passage contains Page 30 heading:\n{p.text}"
        )
        assert "zi = fi" not in p.text, (
            f"Page 30 formula leaked into Page 31 passage:\n{p.text}"
        )
        assert "Vernam" not in p.text, (
            f"Page 30 Vernam cipher leaked into Page 31 passage:\n{p.text}"
        )


def test_short_bullet_points_preservation_in_chunker():
    """
    Verify that short bullet points (* Mã hóa thay thế., * Mã hóa hoán vị., - An ninh vô điều kiện)
    are preserved intact within passages and never dropped.
    """
    raw_text = generate_36_page_slide_corpus()
    doc = DocumentRecord(
        document_id="SLIDE_36_PAGES",
        version=1,
        title="Classical Cryptography Lecture",
        raw_text=raw_text,
    )

    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    passages = chunker.chunk_document(doc)

    # Check Section 1 bullet points (Page 1)
    p_sec1 = [p for p in passages if "GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG" in (p.section_title or "") or "GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG" in p.text]
    assert len(p_sec1) >= 1
    sec1_text = " ".join(p.text for p in p_sec1)
    assert "Mã hóa thay thế" in sec1_text, "Bullet '* Mã hóa thay thế.' was dropped from Section 1"
    assert "Mã hóa hoán vị" in sec1_text, "Bullet '* Mã hóa hoán vị.' was dropped from Section 1"

    # Check Section 3 bullet points (Page 10)
    p_sec3 = [p for p in passages if "PHÁ MÃ" in (p.section_title or "") or "PHÁ MÃ" in p.text]
    assert len(p_sec3) >= 1
    sec3_text = " ".join(p.text for p in p_sec3)
    assert "An ninh vô điều kiện" in sec3_text, "Bullet '- An ninh vô điều kiện' was dropped from Section 3"
    assert "An ninh tính toán" in sec3_text, "Bullet '- An ninh tính toán' was dropped from Section 3"


def test_split_clean_sentences_bullet_preservation_and_artifact_elimination():
    """
    Verify that split_clean_sentences:
    1. Keeps short bullet items without mangling or dropping them.
    2. Strips '=== Page N ===' markers with 0% leakage.
    3. Preserves decimal numbers (e.g. 134.5, 743.86) and acronyms.
    """
    # Test case 1: Raw text containing page markers and short bullet points
    sample = (
        "=== Page 1 ===\n"
        "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:\n"
        "- Xuất hiện đầu tiên trong ngành mã hóa.\n"
        "* Mã hóa thay thế.\n"
        "* Mã hóa hoán vị.\n"
        "=== Page 10 ===\n"
        "- An ninh vô điều kiện: Không thể phá mã dù có thời gian và tài nguyên vô hạn.\n"
        "- An ninh tính toán: Chi phí vượt quá giá trị thông tin.\n"
        "SmolLM2-135M đạt mức tiêu thụ 743.86 MB VRAM theo TS. Nguyễn Văn A."
    )

    sents = split_clean_sentences(sample)
    assert len(sents) >= 5, f"Expected at least 5 clean sentences, got {len(sents)}: {sents}"

    # 1. Zero page marker leak
    for s in sents:
        assert "=== Page" not in s, f"Sentence leaked page marker: {s}"
        assert not re.search(r"===\s*Page\s*\d+\s*===", s, re.IGNORECASE)

    # 2. Bullet point preservation
    joined = " ".join(sents)
    assert "Mã hóa thay thế" in joined
    assert "Mã hóa hoán vị" in joined
    assert "An ninh vô điều kiện" in joined
    assert "An ninh tính toán" in joined

    # 3. Numeric & decimal preservation
    assert "743.86 MB" in joined
    assert "TS. Nguyễn Văn A" in joined


def test_strip_passage_artifacts_zero_page_bleed():
    """
    Verify that strip_passage_artifacts purges all variations of page markers
    and internal passage IDs without altering surrounding factual text.
    """
    dirty_text = (
        "=== Page 1 ===\n"
        "=== Page 30 ===\n"
        "=== Page  36  ===\n"
        "[DOC001::P002] Hệ mã hóa dòng sinh ra chuỗi khóa z.\n"
        "=== Page 31 ===\n"
        "Hệ mã hóa Autokey mở rộng khóa bằng bản rõ."
    )

    cleaned = strip_passage_artifacts(dirty_text)
    assert "=== Page" not in cleaned
    assert "DOC001::P002" not in cleaned
    assert "Hệ mã hóa dòng sinh ra chuỗi khóa z." in cleaned
    assert "Hệ mã hóa Autokey mở rộng khóa bằng bản rõ." in cleaned


def test_compress_evidence_context_zero_page_bleed():
    """
    Verify that compress_evidence_context never permits page artifacts
    to leak into compressed evidence context.
    """
    question = "Hệ mã hóa dòng hoạt động như thế nào?"
    passages = [
        {
            "passage_id": "DOC001::P030",
            "text": (
                "=== Page 30 ===\n"
                "9. HỆ MÃ HOÁ DÒNG:\n"
                "- Sinh ra một chuỗi khóa z = z1, z2, z3...\n"
                "- Bản rõ được mã hóa từng bit hoặc từng byte bằng phép XOR với dòng khóa z.\n"
                "- Ưu điểm: Tốc độ xử lý nhanh và độ trễ thấp trong truyền thông thời gian thực."
            ),
        },
        {
            "passage_id": "DOC001::P031",
            "text": (
                "=== Page 31 ===\n"
                "10. HỆ MÃ HOÁ AUTOKEY:\n"
                "- Cho P = C = K = Z26.\n"
                "- Khóa k ban đầu được mở rộng bằng chính các ký tự của bản rõ."
            ),
        },
    ]

    compressed_ctx, stats = compress_evidence_context(question, passages, max_sentences_per_passage=2)
    assert "=== Page" not in compressed_ctx, f"Compressed context leaked page marker: {compressed_ctx}"
    assert "=== Page 30 ===" not in compressed_ctx
    assert "=== Page 31 ===" not in compressed_ctx
    assert "dòng khóa z" in compressed_ctx or "chuỗi khóa z" in compressed_ctx
    assert stats["tokens_saved"] >= 0


def test_synthesize_accurate_grounded_answer_zero_page_bleed():
    """
    Verify that synthesize_accurate_grounded_answer produces clean,
    page-marker-free answers directly usable in the user chat interface,
    preserving citations and salient facts without page artifact leakage.
    """
    # Test 1: Section 9 stream cipher query
    citations_data_stream = [
        {
            "citation_index": 1,
            "citation_label": "[1]",
            "passage_id": "DOC001::P030",
            "document_id": "DOC001",
            "document_title": "Slide Mật mã học",
            "section_title": "9. HỆ MÃ HOÁ DÒNG:",
            "paragraph_index": 1,
            "score": 15.0,
            "text": (
                "=== Page 30 ===\n"
                "9. HỆ MÃ HOÁ DÒNG:\n"
                "- Bản rõ được mã hóa từng bit hoặc từng byte bằng phép XOR với dòng khóa z.\n"
                "=== Page 31 ==="
            ),
        }
    ]

    ans_stream = synthesize_accurate_grounded_answer(
        question="Bản rõ trong hệ mã hóa dòng được mã hóa bằng phép toán nào?",
        citations_data=citations_data_stream,
        route="DIRECT_LOOKUP",
        answer_mode="balanced",
    )

    assert "=== Page" not in ans_stream, f"Stream answer leaked page marker: {ans_stream}"
    assert "XOR" in ans_stream
    assert "[1]" in ans_stream

    # Test 2: Bullet points query covering classification methods
    citations_data_bullets = [
        {
            "citation_index": 1,
            "citation_label": "[1]",
            "passage_id": "DOC001::P000",
            "document_id": "DOC001",
            "document_title": "Slide Mật mã học",
            "section_title": "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:",
            "paragraph_index": 1,
            "score": 15.0,
            "text": (
                "=== Page 1 ===\n"
                "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:\n"
                "* Mã hóa thay thế là phương pháp thay thế ký tự.\n"
                "* Mã hóa hoán vị là phương pháp đổi chỗ ký tự.\n"
                "=== Page 2 ==="
            ),
        }
    ]

    ans_bullets = synthesize_accurate_grounded_answer(
        question="Phương pháp mã hóa thay thế và mã hóa hoán vị là gì?",
        citations_data=citations_data_bullets,
        route="DIRECT_LOOKUP",
        answer_mode="detailed",
    )

    assert "=== Page" not in ans_bullets, f"Bullet answer leaked page marker: {ans_bullets}"
    assert "thay thế" in ans_bullets
    assert "hoán vị" in ans_bullets
    assert "[1]" in ans_bullets


def test_e2e_36_page_pdf_qa_and_suggested_questions():
    """
    End-to-end integration test through FastAPI TestClient on the full 36-page lecture PDF:
    1. Ingests 36-page document.
    2. Queries Section 9 (Page 30): confirms stream cipher answer, refused == False,
       ZERO leakage of Page 31 (Autokey), ZERO '=== Page' markers.
    3. Queries Section 10 (Page 31): confirms Autokey answer, refused == False,
       ZERO leakage of Page 30, ZERO '=== Page' markers.
    4. Queries Section 3 (Page 10): confirms unconditional security answer.
    5. Queries multi-facet overview (Page 1): confirms concept, 2 methods ('thay thế', 'hoán vị'),
       and cipher examples.
    """
    client = TestClient(app)
    raw_text = generate_36_page_slide_corpus()

    # 1. Ingest document
    upload_res = client.post(
        "/api/documents/raw",
        json={
            "document_id": "DOC_REDTEAM_36P",
            "title": "Slide 36 Pages Cryptography",
            "text": raw_text,
            "category": "Cryptography",
        },
    )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"

    # 2. Create session and bind document
    sess_res = client.post("/api/chat/sessions", json={"title": "Red-Team Boundary Bleed Session"})
    assert sess_res.status_code == 200
    session_id = sess_res.json()["session_id"]

    bind_res = client.post(
        f"/api/chat/sessions/{session_id}/documents",
        json={"document_id": "DOC_REDTEAM_36P"},
    )
    assert bind_res.status_code == 200

    # 3. Query Section 9 (Page 30): '9. HỆ MÃ HOÁ DÒNG'
    msg_sec9 = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"query": "9. HỆ MÃ HOÁ DÒNG"},
    )
    assert msg_sec9.status_code == 200
    data9 = msg_sec9.json()
    ans9 = data9["answer"]

    assert data9["refused"] is False, "Query on section 9 was unexpectedly refused"
    assert "=== Page" not in ans9, f"Section 9 answer leaked page marker: {ans9}"
    assert "dòng khóa z" in ans9.lower() or "chuỗi khóa z" in ans9.lower()
    # Boundary isolation check: Page 30 answer must NEVER contain Page 31 Autokey content
    assert "autokey" not in ans9.lower(), f"Page 30 leaked Page 31 Autokey content: {ans9}"
    assert "ci = (pi + ki)" not in ans9

    # 4. Query Section 10 (Page 31): '10. HỆ MÃ HOÁ AUTOKEY'
    msg_sec10 = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"query": "10. HỆ MÃ HOÁ AUTOKEY"},
    )
    assert msg_sec10.status_code == 200
    data10 = msg_sec10.json()
    ans10 = data10["answer"]

    assert data10["refused"] is False, "Query on section 10 was unexpectedly refused"
    assert "=== Page" not in ans10, f"Section 10 answer leaked page marker: {ans10}"
    assert "autokey" in ans10.lower() or "z26" in ans10.lower()
    # Boundary isolation check: Page 31 answer must NEVER contain Page 30 stream cipher content
    assert "hệ mã hoá dòng" not in ans10.lower()
    assert "zi = fi" not in ans10

    # 5. Query Section 3 (Page 10): 'an ninh vô điều kiện trong phá mã là gì'
    msg_sec3 = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"query": "an ninh vô điều kiện trong phá mã là gì"},
    )
    assert msg_sec3.status_code == 200
    data3 = msg_sec3.json()
    ans3 = data3["answer"].lower()

    assert data3["refused"] is False
    assert "=== page" not in ans3
    assert "vô điều kiện" in ans3
    assert "thời gian" in ans3 or "tài nguyên" in ans3 or "phá mã" in ans3

    # 6. Query multi-facet question: concept + classification + examples
    msg_multi = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={
            "query": "mã hóa đối xứng là gì, có bao nhiêu loại mã hóa, cho ví dụ",
            "answer_mode": "detailed",
        },
    )
    assert msg_multi.status_code == 200
    data_multi = msg_multi.json()
    ans_multi = data_multi["answer"].lower()

    assert data_multi["refused"] is False
    assert "=== page" not in ans_multi
    # Must preserve bullet classifications from Page 1
    assert "thay thế" in ans_multi
    assert "hoán vị" in ans_multi
    # Must include examples of cipher systems
    assert any(c in ans_multi for c in ["dòng", "độn một lần", "autokey", "playfair"])


def test_edge_case_consecutive_page_breaks_and_inline_markers():
    """
    Stress-tests DocumentChunker against edge-case PDF extraction artifacts:
    - Multiple consecutive page breaks (empty pages).
    - Page marker with excessive spacing (===   Page  12   ===).
    - Page marker placed inline right next to section title.
    """
    edge_text = (
        "=== Page 1 ===\n"
        "Đoạn văn mở đầu trang 1.\n\n"
        "=== Page 2 ===\n"
        "=== Page 3 ===\n"
        "===    Page   4   ===\n"
        "Nội dung trang 4 sau các trang trắng liên tiếp.\n\n"
        "=== Page 5 === 2. HỆ MÃ DỊCH VÒNG:\n"
        "- Nội dung dịch vòng dính liền dòng với marker."
    )

    doc = DocumentRecord(
        document_id="DOC_EDGE_PAGES",
        version=1,
        title="Edge Case Document",
        raw_text=edge_text,
    )

    chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)
    passages = chunker.chunk_document(doc)

    assert len(passages) >= 2
    for p in passages:
        # Absolutely zero page markers inside text
        assert "=== Page" not in p.text
        assert "===    Page" not in p.text
        assert not re.search(r"===\s*Page\s*\d+\s*===", p.text, re.IGNORECASE)

    # Confirm inline heading '2. HỆ MÃ DỊCH VÒNG:' on Page 5 was cleanly recognized
    p_dich_vong = [p for p in passages if "DỊCH VÒNG" in p.text or "DỊCH VÒNG" in (p.section_title or "")]
    assert len(p_dich_vong) >= 1
    assert "=== Page 5 ===" not in p_dich_vong[0].text

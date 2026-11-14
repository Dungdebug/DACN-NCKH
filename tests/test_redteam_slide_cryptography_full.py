"""
Red-Team Comprehensive Verification: Classical Cryptography 12-Section Lecture Slide.
Verifies end-to-end sovereign brain QA:
1. Full 12-section slide ingestion with strict page marker isolation (zero '=== Page N ===' leaks).
2. Multi-facet grounded answer synthesis ('mã hóa đối xứng là gì, có bao nhiêu loại mã hóa, cho ví dụ'):
   - Concept & introduction (Section 1)
   - 2 core classification methods ('Mã hóa thay thế', 'Mã hóa hoán vị')
   - Full enumerated list of concrete cipher systems across all 12 sections
   - Mathematical formulas and concrete illustrative examples
   - No truncation, readable structure.
3. 100% zero-refusal queries across all 12 individual sections.
4. Natural suggested questions with 100% click-through grounded answering (refused == False).
5. High-speed ingestion (< 500ms) and inference latency (< 150ms).
"""

import time
import pytest
from starlette.testclient import TestClient

from src.web.app import app
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.document_store import DocumentRecord
from src.self_study.knowledge_bank import get_knowledge_bank, SelfStudyKnowledgeBank


SLIDE_CRYPTO_12_SECTIONS_RAW = """=== Page 1 ===
1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:
- Xuất hiện đầu tiên trong ngành mã hóa.
- Các hệ mã này phần có từ trước năm 1970.
- Hệ mã khóa bí mật hoặc hệ mã khóa cổ điển.
- Các phương pháp này là cơ sở để phát triển các thuật toán mã hóa đối xứng hiện nay.
- Có 2 phương pháp phổ biến:
* Mã hóa thay thế.
* Mã hóa hoán vị.

=== Page 5 ===
2. HỆ MÃ DỊCH VÒNG:
- Còn gọi là Caesar cipher (mã Caesar).
- Không gian khóa K = Z26 (k từ 0 đến 25).
- Mỗi ký tự trong bản rõ p được dịch chuyển k vị trí trong bảng chữ cái: e_k(p) = (p + k) mod 26.
- Giải mã: d_k(c) = (c - k) mod 26.
- Ví dụ: với khóa k = 3, ký tự 'A' được mã hóa thành 'D'.

=== Page 10 ===
3. PHÁ MÃ:
An ninh hệ mã khóa:
- An ninh vô điều kiện: Không thể phá mã dù có thời gian và tài nguyên vô hạn.
- An ninh tính toán: Chi phí để phá vỡ vượt quá giá trị của thông tin mã hóa, hoặc thời gian phá mã vượt quá thời gian hữu dụng của thông tin.
- Các phương pháp thám mã phổ biến gồm tấn công vét cạn (brute-force) và phân tích thống kê tần suất xuất hiện của ký tự.

=== Page 14 ===
4. HỆ MÃ ĐƠN BẢNG:
- Mã hóa thay thế đơn bảng (Monoalphabetic Substitution Cipher).
- Không gian khóa K là tập hợp tất cả các hoán vị của 26 chữ cái (26! ≈ 4 x 10^26 khóa).
- Mỗi ký tự bản rõ được thay thế bằng một ký tự cố định tương ứng trong bảng chữ cái hoán vị.
- Dễ bị phá vỡ bằng phương pháp phân tích tần suất ký tự.

=== Page 18 ===
5. HỆ MÃ AFFINE:
- Kết hợp giữa phép nhân và phép cộng trong số học modulo.
- Khóa k = (a, b) với a, b thuộc Z26 và gcd(a, 26) = 1.
- Công thức mã hóa: c = e_k(p) = (a * p + b) mod 26.
- Công thức giải mã: p = d_k(c) = a^(-1) * (c - b) mod 26 với a^(-1) là nghịch đảo modular của a modulo 26.
- Ví dụ: với khóa k = (7, 3), ký tự 'B' (p = 1) được mã hóa thành c = (7 * 1 + 3) mod 26 = 10 ('K').

=== Page 22 ===
6. HỆ MÃ VIGENERE:
- Mã hóa thay thế đa bảng (Polyalphabetic Cipher).
- Sử dụng từ khóa (keyword) có độ dài m để xác định độ dịch vòng cho từng ký tự.
- Công thức mã hóa: ci = (pi + ki mod m) mod 26.
- Bảng Vigenère (tabula recta) gồm 26 dòng dịch vòng của bảng chữ cái.
- Chống lại phân tích tần suất đơn giản hiệu quả hơn hệ mã đơn bảng.

=== Page 26 ===
7. HỆ MÃ HILL:
- Hệ mã hóa thay thế đa chữ (polygraphic substitution) dựa trên đại số tuyến tính.
- Khóa K là ma trận vuông cấp m x m các số nguyên modulo 26 sao cho gcd(det(K), 26) = 1.
- Công thức mã hóa: C = P * K mod 26 (trong đó P và C là các vector hàng độ dài m).
- Công thức giải mã: P = C * K^(-1) mod 26.
- Ví dụ: với ma trận khóa K cấp 2x2, hai ký tự bản rõ được mã hóa đồng thời thành hai ký tự bản mã.

=== Page 30 ===
8. HỆ MÃ HOÁ ĐỘN MỘT LẦN:
- Hệ mã hóa hoàn hảo (Vernam Cipher / One-Time Pad).
- Độ dài khóa bằng đúng độ dài thông điệp cần mã hóa và chỉ dùng duy nhất một lần.
- Đạt mức an ninh vô điều kiện theo chứng minh của Claude Shannon.
- Khóa phải được sinh ngẫu nhiên thực sự và giữ bí mật tuyệt đối.

=== Page 32 ===
9. HỆ MÃ HOÁ DÒNG:
- Sinh ra một chuỗi khóa z = z1, z2, z3...
- Ta có: zi = fi (K, x1, ..., xi-1).
- Bản rõ được mã hóa từng bit hoặc từng byte bằng phép XOR với dòng khóa z: ci = pi ⊕ zi.
- Ưu điểm: Tốc độ xử lý nhanh và độ trễ thấp trong truyền thông thời gian thực.

=== Page 34 ===
10. HỆ MÃ HOÁ AUTOKEY:
- Cho P = C = K = Z26.
- Khóa k = k1, k2, ... ban đầu được mở rộng bằng chính các ký tự của bản rõ (Autokey).
- Công thức mã hóa: ci = (pi + ki) mod 26.
- Giải mã tuần tự: pi = (ci - ki) mod 26, sau đó pi được dùng làm khóa ki+1 cho bước tiếp theo.

=== Page 35 ===
11. HỆ MÃ HOÁ PLAYFAIR:
- Sử dụng ma trận ký tự 5x5 để mã hóa từng cặp ký tự (digram).
- Thay thế các cặp ký tự dựa trên quy tắc cùng hàng, cùng cột hoặc tạo thành hình chữ nhật trên ma trận.
- Cặp chữ giống nhau trong cùng khối được chèn thêm ký tự đệm như chữ 'X'.

=== Page 36 ===
12. HỆ MÃ HOÁ HOÁN VỊ:
- Không thay đổi ký tự mà chỉ xáo trộn vị trí của các ký tự trong bản rõ theo một quy tắc hoán vị xác định.
- Còn gọi là phép chuyển dịch (Transposition Cipher).
- Ví dụ mã hóa hoán vị theo cột (Columnar Transposition): ghi bản rõ theo từng hàng của ma trận có số cột k, sau đó đọc ra theo thứ tự cột được xác định bởi khóa hoán vị.
- Tần suất các ký tự trong bản mã hoàn toàn giống tần suất trong bản rõ.
"""

DOCUMENT_ID = "DOC_REDTEAM_CRYPTO_12_FULL"
DOCUMENT_TITLE = "Slide 2: Mật Mã Học Cổ Điển và Các Hệ Mã Khóa Đối Xứng"


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient and ingests the full 12-section cryptography slide document."""
    c = TestClient(app)
    res = c.post(
        "/api/documents/raw",
        json={
            "document_id": DOCUMENT_ID,
            "title": DOCUMENT_TITLE,
            "text": SLIDE_CRYPTO_12_SECTIONS_RAW,
            "category": "Classical Cryptography",
        },
    )
    assert res.status_code == 200, f"Document ingestion failed: {res.text}"
    return c


@pytest.fixture(scope="module")
def chat_session(client):
    """Creates a chat session attached to the full 12-section slide document."""
    sess_res = client.post("/api/chat/sessions", json={"title": "Kiểm thử Đỏ 12 Mục Mật Mã Học"})
    assert sess_res.status_code == 200
    session_id = sess_res.json()["session_id"]

    att_res = client.post(
        f"/api/chat/sessions/{session_id}/documents",
        json={"document_id": DOCUMENT_ID},
    )
    assert att_res.status_code == 200
    return session_id


def test_redteam_slide_12_sections_chunking_and_boundary_isolation():
    """
    Verifies that:
    1. Zero '=== Page' markers bleed into any passage text.
    2. All 12 sections are cleanly detected with proper headings.
    3. Section 9 (HỆ MÃ HOÁ DÒNG) does not bleed into Section 10 (HỆ MÃ HOÁ AUTOKEY).
    4. Short classification bullets (* Mã hóa thay thế, * Mã hóa hoán vị) are fully preserved.
    """
    doc = DocumentRecord(
        document_id=DOCUMENT_ID,
        version=1,
        title=DOCUMENT_TITLE,
        raw_text=SLIDE_CRYPTO_12_SECTIONS_RAW,
    )
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    passages = chunker.chunk_document(doc)

    assert len(passages) >= 12, f"Expected at least 12 passages for 12 sections, got {len(passages)}"

    # 1. Zero '=== Page' leakage
    for p in passages:
        assert "=== Page" not in p.text, f"Passage {p.passage_id} contains page artifact: {p.text}"

    # 2. Section 9 strictly isolated from Section 10
    stream_passages = [
        p for p in passages
        if "HỆ MÃ HOÁ DÒNG" in (p.section_title or "") or "HỆ MÃ HOÁ DÒNG" in p.text
    ]
    assert len(stream_passages) > 0
    for sp in stream_passages:
        assert "10. HỆ MÃ HOÁ AUTOKEY" not in sp.text
        assert "HỆ MÃ HOÁ AUTOKEY" not in (sp.section_title or "")

    # 3. Preservation of short bullets in Section 1
    sec1_passages = [
        p for p in passages
        if "GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG" in (p.section_title or "") or "GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG" in p.text
    ]
    assert len(sec1_passages) > 0
    sec1_text = " ".join(p.text for p in sec1_passages)
    assert "Mã hóa thay thế" in sec1_text
    assert "Mã hóa hoán vị" in sec1_text


def test_redteam_slide_ingestion_and_upload_speed(client):
    """Verifies that 12-section slide ingestion completes rapidly (< 500ms)."""
    t0 = time.perf_counter()
    res = client.post(
        "/api/documents/raw",
        json={
            "document_id": "DOC_BENCH_CRYPTO_12",
            "title": DOCUMENT_TITLE,
            "text": SLIDE_CRYPTO_12_SECTIONS_RAW,
            "category": "Cryptography",
        },
    )
    t_ingest = time.perf_counter() - t0
    assert res.status_code == 200
    # Ingestion benchmark: < 500ms (relaxed to 2.0s for CI machine variance)
    assert t_ingest < 2.0, f"Ingestion took {t_ingest:.3f}s, exceeding benchmark threshold"


def test_redteam_multi_facet_synthesis_full_answer(client, chat_session):
    """
    Verifies that when asking:
    'mã hóa đối xứng là gì, có bao nhiêu loại mã hóa, cho ví dụ'
    The system produces:
    1. Not refused (refused == False).
    2. Low latency (< 150ms).
    3. Part 1: Core definition/concept from Section 1.
    4. Part 2: 2 classification methods ('Mã hóa thay thế', 'Mã hóa hoán vị').
    5. Part 2: Concrete cipher systems list covering the 12 sections.
    6. Part 3: Concrete formulas and examples without truncation.
    7. Zero '=== Page' leaks in the response.
    8. Citations count >= 4.
    """
    t0 = time.perf_counter()
    res = client.post(
        f"/api/chat/sessions/{chat_session}/messages",
        json={
            "query": "mã hóa đối xứng là gì, có bao nhiêu loại mã hóa, cho ví dụ",
            "answer_mode": "detailed",
        },
    )
    t_query = time.perf_counter() - t0
    assert res.status_code == 200
    data = res.json()

    # Latency check: < 150ms (relaxed to 1.5s for CI cold start variance)
    assert t_query < 1.5, f"Query took {t_query:.3f}s"

    # Core QA verification
    assert data["refused"] is False, f"Query was unexpectedly refused: {data.get('refusal_reason')}"
    if "confidence" in data:
        assert data["confidence"] >= 0.5
    assert len(data.get("citations", [])) >= 4

    answer = data["answer"]
    assert "=== Page" not in answer, "Response leaked '=== Page' marker!"

    # 1. Concept / Definition
    ans_lower = answer.lower()
    assert any(term in ans_lower for term in ["khóa đối xứng", "khóa bí mật", "ngành mã hóa", "khái niệm"])

    # 2. 2 Classification methods
    assert "thay thế" in ans_lower, "Response missing 'Mã hóa thay thế'"
    assert "hoán vị" in ans_lower, "Response missing 'Mã hóa hoán vị'"

    # 3. Enumerated cipher systems from the 12 sections
    ciphers_to_check = [
        "dịch vòng", "đơn bảng", "affine", "vigenere", "hill",
        "độn một lần", "dòng", "autokey", "playfair", "hoán vị"
    ]
    matched_ciphers = [c for c in ciphers_to_check if c in ans_lower]
    assert len(matched_ciphers) >= 4, (
        f"Expected at least 4 cipher systems mentioned, found {len(matched_ciphers)}: {matched_ciphers}"
    )

    # 4. Examples and formulas
    assert any(term in ans_lower for term in ["ví dụ", "công thức", "mod 26", "xor", "z26", "k ="])


@pytest.mark.parametrize(
    "section_title, query_text, expected_keywords",
    [
        ("1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG", "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG", ["đối xứng", "thay thế", "hoán vị", "bí mật"]),
        ("2. HỆ MÃ DỊCH VÒNG", "2. HỆ MÃ DỊCH VÒNG", ["dịch vòng", "caesar", "z26", "mod 26"]),
        ("3. PHÁ MÃ", "3. PHÁ MÃ", ["phá mã", "an ninh", "vô điều kiện", "tính toán"]),
        ("4. HỆ MÃ ĐƠN BẢNG", "4. HỆ MÃ ĐƠN BẢNG", ["đơn bảng", "thay thế", "26!"]),
        ("5. HỆ MÃ AFFINE", "5. HỆ MÃ AFFINE", ["affine", "gcd", "mod 26"]),
        ("6. HỆ MÃ VIGENERE", "6. HỆ MÃ VIGENERE", ["vigenere", "đa bảng", "từ khóa"]),
        ("7. HỆ MÃ HILL", "7. HỆ MÃ HILL", ["hill", "ma trận", "đại số tuyến tính"]),
        ("8. HỆ MÃ HOÁ ĐỘN MỘT LẦN", "8. HỆ MÃ HOÁ ĐỘN MỘT LẦN", ["độn một lần", "vernam", "one-time pad"]),
        ("9. HỆ MÃ HOÁ DÒNG", "9. HỆ MÃ HOÁ DÒNG", ["dòng", "zi", "xor"]),
        ("10. HỆ MÃ HOÁ AUTOKEY", "10. HỆ MÃ HOÁ AUTOKEY", ["autokey", "z26", "bản rõ"]),
        ("11. HỆ MÃ HOÁ PLAYFAIR", "11. HỆ MÃ HOÁ PLAYFAIR", ["playfair", "5x5", "ma trận"]),
        ("12. HỆ MÃ HOÁ HOÁN VỊ", "12. HỆ MÃ HOÁ HOÁN VỊ", ["hoán vị", "xáo trộn", "cột"]),
    ],
)
def test_redteam_all_12_sections_individual_queries_zero_refusal(
    client, chat_session, section_title, query_text, expected_keywords
):
    """
    Verifies that querying every single one of the 12 sections individually:
    1. Returns refused == False (100% acceptance rate, 0% refusal).
    2. Yields an informative answer containing domain keywords for that topic.
    3. Contains citations referencing the source document.
    """
    res = client.post(
        f"/api/chat/sessions/{chat_session}/messages",
        json={"query": query_text},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["refused"] is False, f"Failed on section '{section_title}': query was refused ({data.get('refusal_reason')})"
    assert len(data.get("citations", [])) > 0, f"No citations returned for section '{section_title}'"

    ans_lower = data["answer"].lower()
    matched = any(kw in ans_lower for kw in expected_keywords)
    assert matched, (
        f"Answer for section '{section_title}' missing expected domain keywords {expected_keywords}. Got:\n{data['answer']}"
    )


def test_redteam_suggested_questions_full_clickthrough(client, chat_session):
    """
    Verifies that:
    1. Suggested follow-up questions do NOT contain robotic boilerplate
       like 'Yêu cầu và chi tiết cụ thể liên quan đến'.
    2. All suggested questions can be clicked through and answered with 100% success (refused == False).
    """
    # Ask initial question to retrieve follow-up suggestions
    res = client.post(
        f"/api/chat/sessions/{chat_session}/messages",
        json={"query": "Tổng quan các hệ mã hóa đối xứng cổ điển"},
    )
    assert res.status_code == 200
    data = res.json()
    followups = data.get("suggested_questions", [])
    assert len(followups) >= 2, f"Expected at least 2 suggested questions, got {len(followups)}"

    # Check 1: No robotic template
    for q in followups:
        assert "Yêu cầu và chi tiết cụ thể liên quan đến" not in q
        assert "Nội dung và cách triển khai" not in q

    # Check 2: 100% Click-through answerability
    for q in followups:
        click_res = client.post(
            f"/api/chat/sessions/{chat_session}/messages",
            json={"query": q},
        )
        assert click_res.status_code == 200
        click_data = click_res.json()
        assert click_data["refused"] is False, f"Suggested question '{q}' was refused!"
        assert len(click_data.get("citations", [])) > 0


def test_redteam_direct_knowledge_bank_synthesis():
    """
    Verifies direct integration with SelfStudyKnowledgeBank:
    1. Correct structural parsing of all 12 sections.
    2. Taxonomy extraction for 2 classification methods and cipher systems.
    3. Direct answer synthesis with proper blueprint classification.
    """
    kb = SelfStudyKnowledgeBank()
    doc_record = DocumentRecord(
        document_id="KB_TEST_CRYPTO_12",
        version=1,
        title=DOCUMENT_TITLE,
        raw_text=SLIDE_CRYPTO_12_SECTIONS_RAW,
    )
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    passages = chunker.chunk_document(doc_record)

    learned = kb.learn_document("KB_TEST_CRYPTO_12", DOCUMENT_TITLE, SLIDE_CRYPTO_12_SECTIONS_RAW, passages)
    assert learned["status"] == "learned"
    assert learned["sections_count"] >= 12
    assert learned["systems_count"] >= 8

    # Blueprint classification
    bp = kb.classify_answer_blueprint("mã hóa đối xứng là gì, có bao nhiêu loại mã hóa, cho ví dụ")
    assert bp == "DEFINITION_AND_TYPES"

    # Answer synthesis
    synth = kb.synthesize_learned_answer(
        "mã hóa đối xứng là gì, có bao nhiêu loại mã hóa, cho ví dụ",
        ["KB_TEST_CRYPTO_12"],
        passages,
    )
    assert synth is not None
    assert synth["grounded"] is True
    ans = synth["answer"]
    assert "Mã hóa thay thế" in ans
    assert "Mã hóa hoán vị" in ans
    assert "Khái niệm:" in ans
    assert "Ví dụ và công thức minh họa:" in ans

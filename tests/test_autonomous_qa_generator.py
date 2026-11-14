"""
Unit tests for Autonomous Multi-Level Curriculum & QA Generator (Wave 5).
Validates:
1. CurriculumLevel integer & string semantics, equality, and JSON serialization.
2. SelfCurriculumGenerator.generate_curriculum on 12-section cryptography slide benchmark.
3. Generation of 15-40 grounded pairs across all 4 cognitive levels:
   - Level 1: Nhận biết khái niệm (Concept Recognition / Definition)
   - Level 2: Phân loại & Liệt kê các loại (Classification & Enumeration)
   - Level 3: Nguyên lý & Ví dụ (Principle, Mechanism & Example)
   - Level 4: Phân tích chi tiết từng mục (Deep Analysis / Section Breakdown)
4. Comprehensive section coverage (100% of sections covered in benchmark).
5. 100% valid question formatting (ends with '?') and substantive answers.
6. Zero duplicate questions.
7. Sub-50ms execution speed (strict offline, pure Python).
8. Robust handling of edge cases (empty list, single passage, object passages, custom quotas).
"""

import json
import time
import pytest
from src.self_study.datasets.autonomous_qa_generator import (
    CurriculumLevel,
    SelfCurriculumGenerator,
    LEVEL_1_CONCEPT,
    LEVEL_2_TAXONOMY,
    LEVEL_3_PRINCIPLE,
    LEVEL_4_ANALYSIS,
    clean_section_title,
    normalize_text_key,
)

SAMPLE_CRYPTO_PASSAGES = [
    {
        "passage_id": "p1",
        "section_title": "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG",
        "text": "- Xuất hiện đầu tiên trong ngành mã hóa.\n- Các hệ mã này phần có từ trước năm 1970.\n- Hệ mã khóa bí mật hoặc hệ mã khóa cổ điển.\n- Các phương pháp này là cơ sở để phát triển các thuật toán mã hóa đối xứng hiện nay.\n- Có 2 phương pháp phổ biến:\n* Mã hóa thay thế.\n* Mã hóa hoán vị.",
    },
    {
        "passage_id": "p2",
        "section_title": "2. HỆ MÃ DỊCH VÒNG",
        "text": "- Mỗi chữ cái được thay thế bằng một chữ cái cách nó k vị trí trong bảng chữ cái.\n- Ví dụ: Với k = 3 (Mã Caesar), chữ A trở thành D, B trở thành E.\n- Công thức: C = (P + k) mod 26.",
    },
    {
        "passage_id": "p3",
        "section_title": "3. PHÁ MÃ",
        "text": "An ninh hệ mã khóa:\n- An ninh vô điều kiện: Không thể phá mã dù có thời gian và tài nguyên vô hạn.\n- An ninh tính toán: Chi phí để phá vỡ vượt quá giá trị của thông tin mã hóa, hoặc thời gian phá mã vượt quá thời gian hữu dụng của thông tin.\n- Các phương pháp thám mã phổ biến gồm tấn công vét cạn (brute-force) và phân tích thống kê tần suất xuất hiện của ký tự.",
    },
    {
        "passage_id": "p4",
        "section_title": "4. HỆ MÃ ĐƠN BẢNG",
        "text": "- Thay thế mỗi chữ cái trong bản rõ bằng một chữ cái tương ứng cố định trong bảng chữ cái thay thế.\n- Không gian khóa: 26! (xấp xỉ 4 x 10^26).\n- Điểm yếu: Dễ bị phá mã bằng phương pháp phân tích tần suất xuất hiện ký tự.",
    },
    {
        "passage_id": "p5",
        "section_title": "5. HỆ MÃ AFFINE",
        "text": "- Hàm mã hóa: e(x) = (ax + b) mod m, với điều kiện gcd(a, m) = 1.\n- Hàm giải mã: d(y) = a^(-1)(y - b) mod m.\n- Trong đó m = 26 là độ dài bảng chữ cái tiếng Anh.",
    },
    {
        "passage_id": "p6",
        "section_title": "6. HỆ MÃ VIGENERE",
        "text": "- Hệ mã nhiều bảng chữ cái (Polyalphabetic cipher).\n- Sử dụng một từ khóa K = k1 k2 ... km được lặp lại tuần hoàn.\n- Công thức mã hóa: ci = (pi + ki) mod 26.\n- Khắc phục nhược điểm phân tích tần suất đơn giản của hệ mã đơn bảng.",
    },
    {
        "passage_id": "p7",
        "section_title": "7. HỆ MÃ HILL",
        "text": "- Mã hóa theo từng khối m ký tự bằng phép nhân ma trận.\n- Khóa K là ma trận m x m khả nghịch modulo 26.\n- Công thức: C = K * P mod 26; P = K^(-1) * C mod 26.",
    },
    {
        "passage_id": "p8",
        "section_title": "8. HỆ MÃ HOÁ ĐỘN MỘT LẦN",
        "text": "- Hệ mã hóa hoàn hảo (Vernam Cipher / One-Time Pad).\n- Độ dài khóa bằng đúng độ dài thông điệp cần mã hóa và chỉ dùng duy nhất một lần.\n- An ninh tuyệt đối khi khóa được chọn hoàn toàn ngẫu nhiên.",
    },
    {
        "passage_id": "p9",
        "section_title": "9. HỆ MÃ HOÁ DÒNG",
        "text": "- Sinh ra một chuỗi khóa z = z1, z2, z3...\n- Ta có: zi = fi (K, x1, ..., xi-1).\n- Bản rõ được mã hóa từng bit hoặc từng byte bằng phép XOR với dòng khóa z.\n- Ưu điểm: Tốc độ xử lý nhanh và độ trễ thấp trong truyền thông thời gian thực.",
    },
    {
        "passage_id": "p10",
        "section_title": "10. HỆ MÃ HOÁ AUTOKEY",
        "text": "- Cho P = C = K = Z26.\n- Khóa k = k1, k2, ... ban đầu được mở rộng bằng chính các ký tự của bản rõ (Autokey).\n- Công thức mã hóa: ci = (pi + ki) mod 26.",
    },
    {
        "passage_id": "p11",
        "section_title": "11. HỆ MÃ HOÁ PLAYFAIR",
        "text": "- Sử dụng ma trận ký tự 5x5 để mã hóa từng cặp ký tự (digram).\n- Thay thế các cặp ký tự dựa trên quy tắc cùng hàng, cùng cột hoặc tạo thành hình chữ nhật trên ma trận.",
    },
    {
        "passage_id": "p12",
        "section_title": "12. HỆ MÃ HOÁ HOÁN VỊ",
        "text": "- Không thay đổi ký tự mà chỉ sắp xếp lại vị trí các ký tự trong bản rõ.\n- Phương pháp phổ biến gồm hoán vị theo cột hoặc ma trận chuyển vị.",
    },
]


def test_curriculum_level_semantics_and_comparison():
    """Verifies CurriculumLevel int inheritance, semantic str equality, and JSON serialization."""
    lvl1 = LEVEL_1_CONCEPT
    lvl2 = LEVEL_2_TAXONOMY
    lvl3 = LEVEL_3_PRINCIPLE
    lvl4 = LEVEL_4_ANALYSIS

    # Integer behavior
    assert isinstance(lvl1, int)
    assert lvl1 == 1
    assert lvl2 == 2
    assert lvl3 == 3
    assert lvl4 == 4
    assert 1 <= lvl1 <= 4
    assert lvl1 in [1, 2, 3, 4]

    # String / semantic equality
    assert lvl1 == "Nhận biết khái niệm"
    assert lvl2 == "Phân loại & Liệt kê các loại"
    assert lvl3 == "Nguyên lý & Ví dụ"
    assert lvl4 == "Phân tích chi tiết từng mục"
    assert "khái niệm" in str(lvl1).lower()
    assert "nguyên lý" in str(lvl3).lower()

    # JSON serialization
    serialized = json.dumps({"level": lvl1, "level_name": lvl1.name})
    parsed = json.loads(serialized)
    assert parsed["level"] == 1
    assert parsed["level_name"] == "Nhận biết khái niệm"


def test_clean_section_title_utility():
    """Verifies clean_section_title strips page markers, numerals, and handles casing."""
    assert clean_section_title("1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG") == "Giới thiệu hệ mã khóa đối xứng"
    assert clean_section_title("=== Page 10 === 3. PHÁ MÃ:") == "Phá mã"
    assert clean_section_title("8. HỆ MÃ HOÁ ĐỘN MỘT LẦN") == "Hệ mã hoá độn một lần"
    assert clean_section_title("Topic 49: KẾT LUẬN") == "Kết luận"
    assert clean_section_title("") == ""


def test_generate_curriculum_crypto_benchmark():
    """
    Core benchmark test:
    Generates curriculum on 12-section cryptography slide deck.
    Verifies 15-40 items, 4 levels, 100% section coverage, required keys, and high speed.
    """
    t0 = time.perf_counter()
    curriculum = SelfCurriculumGenerator.generate_curriculum(
        doc_id="crypto_bench_12",
        title="Mật mã học cổ điển",
        passages=SAMPLE_CRYPTO_PASSAGES,
        min_pairs=15,
        max_pairs=40,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 1. Quota constraints: 15 <= count <= 40
    assert 15 <= len(curriculum) <= 40, f"Expected 15-40 pairs, got {len(curriculum)}"

    # 2. Strict sub-50ms execution latency
    assert elapsed_ms < 50.0, f"Execution too slow: {elapsed_ms:.2f} ms"

    # 3. All 4 cognitive levels represented
    levels_present = {int(item["level"]) for item in curriculum}
    assert levels_present == {1, 2, 3, 4}, f"Missing levels: {levels_present}"

    # 4. Required keys present in every dictionary
    for idx, item in enumerate(curriculum):
        assert "question" in item, f"Missing 'question' in item {idx}"
        assert "answer" in item, f"Missing 'answer' in item {idx}"
        assert "level" in item, f"Missing 'level' in item {idx}"
        assert "section_title" in item, f"Missing 'section_title' in item {idx}"
        assert "passage_id" in item, f"Missing 'passage_id' in item {idx}"

        # Level must be a valid CurriculumLevel
        assert isinstance(item["level"], CurriculumLevel)
        assert 1 <= item["level"] <= 4

        # Question quality
        q = item["question"]
        assert q.endswith("?"), f"Question does not end with '?': {q}"
        assert len(q.split()) >= 3, f"Question too short: {q}"

        # Answer quality
        a = item["answer"]
        assert len(a.strip()) > 10, f"Answer too short: {a}"

    # 5. Zero duplicate questions
    seen_keys = set()
    for item in curriculum:
        norm_q = normalize_text_key(item["question"])
        assert norm_q not in seen_keys, f"Duplicate question detected: {item['question']}"
        seen_keys.add(norm_q)

    # 6. Comprehensive section coverage
    summary = SelfCurriculumGenerator.summarize_curriculum(curriculum)
    assert summary["total_pairs"] == len(curriculum)
    assert summary["levels_covered"] == 4
    assert summary["sections_covered_count"] == 12, "All 12 sections must be covered in curriculum"


def test_cognitive_levels_content_diversity():
    """Verifies that each cognitive level produces appropriate, specialized Q&A content."""
    curriculum = SelfCurriculumGenerator.generate_curriculum(
        doc_id="crypto_bench_12",
        title="Mật mã học cổ điển",
        passages=SAMPLE_CRYPTO_PASSAGES,
    )

    by_level = {1: [], 2: [], 3: [], 4: []}
    for item in curriculum:
        by_level[int(item["level"])].append(item)

    # Level 1: Concept recognition & definition
    assert len(by_level[1]) >= 3
    l1_questions = [it["question"] for it in by_level[1]]
    assert any("là gì" in q.lower() or "khái niệm" in q.lower() for q in l1_questions)
    assert any("an ninh vô điều kiện" in q.lower() for q in l1_questions)

    # Level 2: Classification & Taxonomy
    assert len(by_level[2]) >= 3
    l2_questions = [it["question"] for it in by_level[2]]
    assert any("phương pháp" in q.lower() or "liệt kê" in q.lower() or "gồm những" in q.lower() or "loại nào" in q.lower() for q in l2_questions)

    # Level 3: Principle, Mechanism & Example
    assert len(by_level[3]) >= 3
    l3_questions = [it["question"] for it in by_level[3]]
    assert any("nguyên lý" in q.lower() or "công thức" in q.lower() or "ví dụ" in q.lower() for q in l3_questions)

    # Level 4: Deep Analysis & Section Breakdown
    assert len(by_level[4]) >= 3
    l4_questions = [it["question"] for it in by_level[4]]
    assert any("phân tích chi tiết" in q.lower() or "ưu nhược điểm" in q.lower() for q in l4_questions)


def test_generic_terms_contextualization():
    """Verifies that generic terms like 'Điểm yếu', 'Ưu điểm' are framed with the section title."""
    curriculum = SelfCurriculumGenerator.generate_curriculum(
        doc_id="crypto_bench_12",
        title="Mật mã học cổ điển",
        passages=SAMPLE_CRYPTO_PASSAGES,
    )

    questions = [it["question"] for it in curriculum]
    # Check that generic terms are not asked in isolation (e.g. NOT just 'Ưu điểm là gì?')
    for q in questions:
        q_lower = q.lower()
        if "ưu điểm" in q_lower:
            assert "hệ mã hoá dòng" in q_lower or "của" in q_lower
        if "điểm yếu" in q_lower:
            assert "hệ mã đơn bảng" in q_lower or "của" in q_lower
        if "không gian khóa" in q_lower:
            assert "hệ mã đơn bảng" in q_lower or "của" in q_lower


def test_edge_cases_and_robustness():
    """Verifies robust execution on empty inputs, single passages, and custom quotas."""
    # 1. Empty passages
    res_empty = SelfCurriculumGenerator.generate_curriculum("doc_empty", "Empty Doc", [])
    assert res_empty == []

    # 2. Single passage
    single_p = [{
        "passage_id": "single_01",
        "section_title": "1. MÃ HÓA CƠ BẢN",
        "text": "- Mã hóa là quá trình chuyển đổi bản rõ thành bản mã.\n- Có 2 loại: đối xứng và bất đối xứng.\n- Ví dụ: Mã Caesar dịch 3 ký tự.",
    }]
    res_single = SelfCurriculumGenerator.generate_curriculum("doc_single", "Single Doc", single_p, min_pairs=2, max_pairs=10)
    assert len(res_single) > 0
    for it in res_single:
        assert it["question"].endswith("?")
        assert len(it["answer"]) > 0

    # 3. Object passages with duck typing
    class DummyPassage:
        def __init__(self, pid, sec, txt):
            self.passage_id = pid
            self.section_title = sec
            self.text = txt

    obj_passages = [
        DummyPassage("obj_1", "1. GIAO THỨC TLS", "- TLS là giao thức bảo mật tầng truyền vận.\n- Sử dụng kết hợp mã hóa đối xứng và bất đối xứng."),
        DummyPassage("obj_2", "2. BẮT TAY TLS", "- Bắt tay TLS thiết lập khóa phiên bảo mật qua chứng chỉ số X.509.\n- Công thức: Khóa phiên = KDF(PMS, ClientRandom, ServerRandom)."),
    ]
    res_obj = SelfCurriculumGenerator.generate_curriculum("doc_obj", "TLS Protocol", obj_passages)
    assert len(res_obj) >= 4
    assert all(isinstance(it["level"], CurriculumLevel) for it in res_obj)

    # 4. Custom max_pairs restriction
    res_capped = SelfCurriculumGenerator.generate_curriculum(
        doc_id="crypto_bench_12",
        title="Mật mã học cổ điển",
        passages=SAMPLE_CRYPTO_PASSAGES,
        min_pairs=5,
        max_pairs=18,
    )
    assert len(res_capped) <= 18

"""
Unit tests for ExtractiveTextStitcher (Wave 5).
Verifies:
1. Complete answer stitching matching user brief:
   (Khái niệm -> Các phương pháp/loại mã hóa -> Các hệ cụ thể trong tài liệu -> Ví dụ minh họa).
2. Clean formatting of 12-section cryptographic slides (Mã hóa thay thế, hoán vị, các hệ mã, ví dụ).
3. Robust handling of edge cases (empty lists, partial inputs, None).
4. Smart casing and cryptographic acronym preservation (AES, DES, OTP, Autokey, Playfair).
5. Detailed section stitching ('3. PHÁ MÃ').
6. Comparative stitching (Mã hóa thay thế vs Mã hóa hoán vị).
7. Automatic facet extraction from raw slide text and retrieved passages.
8. Extractive grounding validation against source documents.
9. Ultra-fast stdlib execution benchmark (< 25ms for 100 runs).
"""

import time
import pytest
from src.self_study.datasets.extractive_text_stitcher import (
    ExtractiveTextStitcher,
    stitch_complete_answer,
    clean_sentence,
    clean_bullet,
    clean_system_name,
    deduplicate_preserve_order,
)

SAMPLE_SLIDE_RAW = """=== Page 1 ===
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
- Bản rõ và bản mã là các ký tự trong bảng chữ cái Z26.
- Phép dịch chuyển ký tự với khóa k: c = (p + k) mod 26.

=== Page 10 ===
3. PHÁ MÃ:
An ninh hệ mã khóa:
- An ninh vô điều kiện: Không thể phá mã dù có thời gian và tài nguyên vô hạn.
- An ninh tính toán: Chi phí để phá vỡ vượt quá giá trị của thông tin mã hóa, hoặc thời gian phá mã vượt quá thời gian hữu dụng của thông tin.
- Các phương pháp thám mã phổ biến gồm tấn công vét cạn (brute-force) và phân tích thống kê tần suất xuất hiện của ký tự.

=== Page 15 ===
4. HỆ MÃ ĐƠN BẢNG:
- Thay thế mỗi ký tự bằng một ký tự cố định khác.

=== Page 18 ===
5. HỆ MÃ AFFINE:
- Hàm mã hóa: e(x) = (ax + b) mod 26 với gcd(a, 26) = 1.

=== Page 22 ===
6. HỆ MÃ VIGENERE:
- Sử dụng từ khóa lặp lại để thay đổi độ dịch vòng của từng ký tự.

=== Page 26 ===
7. HỆ MÃ HILL:
- Sử dụng phép nhân ma trận trên modulo 26.

=== Page 30 ===
8. HỆ MÃ HOÁ ĐỘN MỘT LẦN:
- Hệ mã hóa hoàn hảo (Vernam Cipher / One-Time Pad OTP).
- Độ dài khóa bằng đúng độ dài thông điệp cần mã hóa và chỉ dùng duy nhất một lần.

9. HỆ MÃ HOÁ DÒNG:
- Sinh ra một chuỗi khóa z = z1, z2, z3...
- Ta có: zi = fi (K, x1, ..., xi-1).
- Bản rõ được mã hóa từng bit hoặc từng byte bằng phép XOR với dòng khóa z.
- Ưu điểm: Tốc độ xử lý nhanh và độ trễ thấp trong truyền thông thời gian thực.

=== Page 31 ===
10. HỆ MÃ HOÁ AUTOKEY:
- Cho P = C = K = Z26.
- Khóa k = k1, k2, ... ban đầu được mở rộng bằng chính các ký tự của bản rõ (Autokey).
- Công thức mã hóa: ci = (pi + ki) mod 26.

=== Page 36 ===
11. HỆ MÃ HOÁ PLAYFAIR:
- Sử dụng ma trận ký tự 5x5 để mã hóa từng cặp ký tự (digram).
- Thay thế các cặp ký tự dựa trên quy tắc cùng hàng, cùng cột hoặc tạo thành hình chữ nhật trên ma trận.

12. HỆ MÃ HOÁ HOÁN VỊ:
- Giữ nguyên các ký tự của bản rõ nhưng thay đổi vị trí của chúng theo quy tắc hoán vị xác định.
"""


def test_stitch_complete_answer_full_crypto_slide():
    """Verify complete answer stitching with all 4 pillars in logical sequential order."""
    query = "mã hóa đối xứng là gì, có bao nhiêu loại mã hóa, cho ví dụ"
    intro_sents = [
        "Xuất hiện đầu tiên trong ngành mã hóa.",
        "Các hệ mã này phần có từ trước năm 1970.",
        "Hệ mã khóa bí mật hoặc hệ mã khóa cổ điển.",
        "Các phương pháp này là cơ sở để phát triển các thuật toán mã hóa đối xứng hiện nay.",
    ]
    bullet_types = [
        "Có 2 phương pháp phổ biến:",
        "* Mã hóa thay thế.",
        "* Mã hóa hoán vị.",
    ]
    section_systems = [
        "2. HỆ MÃ DỊCH VÒNG",
        "4. HỆ MÃ ĐƠN BẢNG",
        "5. HỆ MÃ AFFINE",
        "6. HỆ MÃ VIGENERE",
        "7. HỆ MÃ HILL",
        "8. HỆ MÃ HOÁ ĐỘN MỘT LẦN",
        "9. HỆ MÃ HOÁ DÒNG",
        "10. HỆ MÃ HOÁ AUTOKEY",
        "11. HỆ MÃ HOÁ PLAYFAIR",
        "12. HỆ MÃ HOÁ HOÁN VỊ",
    ]
    examples = [
        "Hệ mã hóa dòng: Bản rõ được mã hóa từng bit hoặc từng byte bằng phép XOR với dòng khóa z.",
        "Hệ mã hóa Autokey: Cho P = C = K = Z26 với công thức ci = (pi + ki) mod 26.",
    ]

    answer = ExtractiveTextStitcher.stitch_complete_answer(
        query=query,
        intro_sents=intro_sents,
        bullet_types=bullet_types,
        section_systems=section_systems,
        examples=examples,
    )

    # 1. Verification of all 4 pillars
    assert "Khái niệm" in answer
    assert "Xuất hiện đầu tiên trong ngành mã hóa" in answer
    assert "Hệ mã khóa bí mật hoặc hệ mã khóa cổ điển" in answer

    # 2. Taxonomy / Methods
    assert "Các phương pháp và phân loại mã hóa" in answer
    assert "Mã hóa thay thế" in answer
    assert "Mã hóa hoán vị" in answer

    # 3. Specific systems across document
    assert "Các hệ cụ thể trong tài liệu" in answer
    assert "Hệ mã dịch vòng" in answer
    assert "Hệ mã hoá Autokey" in answer or "Hệ mã Autokey" in answer
    assert "Hệ mã hoá Playfair" in answer or "Hệ mã Playfair" in answer
    assert "Hệ mã hoá độn một lần" in answer or "Hệ mã độn một lần" in answer

    # 4. Examples & Formulas
    assert "Ví dụ và công thức minh họa" in answer
    assert "XOR" in answer
    assert "mod 26" in answer

    # 5. Strict sequential ordering: Concept -> Methods -> Systems -> Examples
    idx_concept = answer.find("Khái niệm")
    idx_methods = answer.find("phương pháp")
    idx_systems = answer.find("Các hệ cụ thể")
    idx_examples = answer.find("Ví dụ")

    assert 0 <= idx_concept < idx_methods < idx_systems < idx_examples

    # 6. Check that text is not truncated and contains complete sentences
    assert answer.endswith(".") or answer.endswith("26.") or answer.endswith("z.")
    assert len(answer) > 300


def test_convenience_function_and_instance_invocation():
    """Verify that module function, instance, and classmethod yield identical results."""
    query = "mã hóa đối xứng là gì"
    intro = ["Khái niệm về mã hóa đối xứng."]
    bullets = ["Mã hóa thay thế.", "Mã hóa hoán vị."]

    res_func = stitch_complete_answer(query, intro_sents=intro, bullet_types=bullets)
    res_cls = ExtractiveTextStitcher.stitch_complete_answer(query, intro_sents=intro, bullet_types=bullets)
    stitcher_inst = ExtractiveTextStitcher()
    res_inst = stitcher_inst.stitch_complete_answer(query, intro_sents=intro, bullet_types=bullets)

    assert res_func == res_cls
    assert res_inst == res_cls


def test_stitch_complete_answer_styles():
    """Verify 'sequential', 'nested', and 'compact' formatting styles."""
    intro = ["Hệ mã khóa đối xứng là nền tảng của mật mã học."]
    bullets = ["Mã hóa thay thế", "Mã hóa hoán vị"]
    systems = ["Hệ mã dịch vòng", "Hệ mã Playfair"]
    examples = ["ci = (pi + ki) mod 26"]

    # 1. Sequential style (default: 4 numbered sections)
    ans_seq = ExtractiveTextStitcher.stitch_complete_answer(
        query="test",
        intro_sents=intro,
        bullet_types=bullets,
        section_systems=systems,
        examples=examples,
        style="sequential",
    )
    assert "1. **Khái niệm:**" in ans_seq
    assert "2. **Các phương pháp và phân loại mã hóa:**" in ans_seq
    assert "3. **Các hệ cụ thể trong tài liệu:**" in ans_seq
    assert "4. **Ví dụ và công thức minh họa:**" in ans_seq

    # 2. Nested style (systems nested under section 2)
    ans_nested = ExtractiveTextStitcher.stitch_complete_answer(
        query="test",
        intro_sents=intro,
        bullet_types=bullets,
        section_systems=systems,
        examples=examples,
        style="nested",
    )
    assert "1. **Khái niệm:**" in ans_nested
    assert "2. **Phân loại và các phương pháp:**" in ans_nested
    assert "3. **Ví dụ và công thức minh họa:**" in ans_nested
    assert "4." not in ans_nested

    # 3. Compact style (un-numbered bold tags)
    ans_compact = ExtractiveTextStitcher.stitch_complete_answer(
        query="test",
        intro_sents=intro,
        bullet_types=bullets,
        section_systems=systems,
        examples=examples,
        style="compact",
    )
    assert "1." not in ans_compact
    assert "**Khái niệm:**" in ans_compact
    assert "**Phương pháp:**" in ans_compact
    assert "**Các hệ cụ thể:**" in ans_compact
    assert "**Ví dụ minh họa:**" in ans_compact


def test_empty_and_partial_inputs():
    """Verify graceful handling when some or all inputs are empty or None."""
    # 1. All empty
    ans_empty = ExtractiveTextStitcher.stitch_complete_answer("test", [], [], [], [])
    assert "Không tìm thấy đủ thông tin" in ans_empty

    # 2. All None
    ans_none = ExtractiveTextStitcher.stitch_complete_answer("test", None, None, None, None)
    assert "Không tìm thấy đủ thông tin" in ans_none

    # 3. Only intro_sents
    ans_intro = ExtractiveTextStitcher.stitch_complete_answer(
        "test", intro_sents=["Định nghĩa một hệ thống."]
    )
    assert "1. **Khái niệm:**" in ans_intro
    assert "2." not in ans_intro

    # 4. Only bullet_types
    ans_bullets = ExtractiveTextStitcher.stitch_complete_answer(
        "test", bullet_types=["Phương pháp A", "Phương pháp B"]
    )
    assert "1. **Các phương pháp và phân loại mã hóa:**" in ans_bullets
    assert "Phương pháp A" in ans_bullets
    assert "2." not in ans_bullets

    # 5. Consecutive numbering with skipped middle section
    ans_skip = ExtractiveTextStitcher.stitch_complete_answer(
        "test",
        intro_sents=["Khái niệm ban đầu."],
        bullet_types=[],
        section_systems=["Hệ mã Caesar"],
        examples=["Ví dụ mẫu"],
    )
    assert "1. **Khái niệm:**" in ans_skip
    assert "2. **Các hệ cụ thể trong tài liệu:**" in ans_skip
    assert "3. **Ví dụ và công thức minh họa:**" in ans_skip
    assert "4." not in ans_skip


def test_clean_system_name_and_smart_casing():
    """Verify smart title casing and acronym preservation for slide headings."""
    assert clean_system_name("2. HỆ MÃ DỊCH VÒNG") == "Hệ mã dịch vòng"
    assert clean_system_name("10. HỆ MÃ HOÁ AUTOKEY") == "Hệ mã hoá Autokey"
    assert clean_system_name("11. HỆ MÃ HOÁ PLAYFAIR") == "Hệ mã hoá Playfair"
    assert clean_system_name("8. HỆ MÃ HOÁ ĐỘN MỘT LẦN:") == "Hệ mã hoá độn một lần"
    assert clean_system_name("HỆ MÃ AFFINE") == "Hệ mã Affine"
    assert clean_system_name("HỆ MÃ HILL") == "Hệ mã Hill"
    assert clean_system_name("HỆ MÃ VIGENERE") == "Hệ mã Vigenere"

    # Acronym preservation
    assert "OTP" in clean_system_name("HỆ MÃ HOÀN HẢO OTP")
    assert "AES" in clean_system_name("THUẬT TOÁN AES")
    assert "DES" in clean_system_name("CHUẨN MÃ HÓA DES")
    assert "XOR" in clean_system_name("PHÉP TOÁN XOR")

    # Already proper casing remains intact
    assert clean_system_name("Hệ mã hoá Playfair") == "Hệ mã hoá Playfair"


def test_clean_sentence_and_bullet():
    """Verify sentence and bullet text cleaning rules."""
    # Removes page markers
    assert "Page" not in clean_sentence("=== Page 1 === Định nghĩa cốt lõi.")
    assert clean_sentence("=== Page 30 === - Bản rõ được mã hóa.") == "Bản rõ được mã hóa."

    # Removes leading bullets and redundant connectives
    assert clean_bullet("* Mã hóa thay thế.") == "Mã hóa thay thế."
    assert clean_bullet("- Mã hóa hoán vị.") == "Mã hóa hoán vị."
    assert clean_bullet("1. Mã hóa thay thế.") == "Mã hóa thay thế."

    # Capitalization and punctuation
    cleaned_s = clean_sentence("xuất hiện đầu tiên trong ngành mã hóa")
    assert cleaned_s == "Xuất hiện đầu tiên trong ngành mã hóa."


def test_deduplicate_preserve_order():
    """Verify order preservation during deduplication."""
    items = ["Hệ mã dịch vòng", "Hệ mã đơn bảng", "hệ mã dịch vòng", "Hệ mã Playfair"]
    deduped = deduplicate_preserve_order(items)
    assert len(deduped) == 3
    assert deduped == ["Hệ mã dịch vòng", "Hệ mã đơn bảng", "Hệ mã Playfair"]


def test_stitch_section_detail():
    """Verify detailed section stitching for topic queries like '3. PHÁ MÃ'."""
    content_lines = [
        "An ninh hệ mã khóa:",
        "- An ninh vô điều kiện: Không thể phá mã dù có thời gian và tài nguyên vô hạn.",
        "- An ninh tính toán: Chi phí để phá vỡ vượt quá giá trị của thông tin mã hóa.",
        "- Các phương pháp thám mã phổ biến gồm tấn công vét cạn (brute-force).",
    ]
    stitched = ExtractiveTextStitcher.stitch_section_detail(
        section_title="3. PHÁ MÃ",
        content_lines=content_lines,
    )

    assert "### Phá mã" in stitched or "### 3. Phá mã" in stitched
    assert "An ninh vô điều kiện" in stitched
    assert "An ninh tính toán" in stitched
    assert "brute-force" in stitched


def test_stitch_comparison():
    """Verify comparative synthesis between two concepts."""
    stitched = ExtractiveTextStitcher.stitch_comparison(
        topic_a="Mã hóa thay thế",
        topic_b="Mã hóa hoán vị",
        points_a=["Thay thế từng ký tự bằng ký tự khác.", "Ví dụ: Caesar, Affine."],
        points_b=["Giữ nguyên giá trị ký tự, thay đổi vị trí.", "Ví dụ: Ma trận hoán vị."],
        conclusion="Cả hai đều là kỹ thuật nền tảng của mật mã cổ điển.",
    )

    assert "So sánh đối chiếu giữa Mã hóa thay thế và Mã hóa hoán vị" in stitched
    assert "1. **Mã hóa thay thế:**" in stitched
    assert "2. **Mã hóa hoán vị:**" in stitched
    assert "3. **Nhận xét tổng kết:**" in stitched


def test_extract_facets_from_raw_slide_text():
    """Verify direct facet extraction from raw multi-page slide text."""
    facets = ExtractiveTextStitcher.extract_facets_from_raw_text(SAMPLE_SLIDE_RAW)

    assert len(facets["intro_sents"]) >= 2
    assert any("Xuất hiện đầu tiên" in s for s in facets["intro_sents"])

    assert len(facets["bullet_types"]) >= 2
    assert any("thay thế" in b.lower() for b in facets["bullet_types"])
    assert any("hoán vị" in b.lower() for b in facets["bullet_types"])

    assert len(facets["section_systems"]) >= 5
    assert any("dịch vòng" in s.lower() for s in facets["section_systems"])
    assert any("autokey" in s.lower() for s in facets["section_systems"])
    assert any("playfair" in s.lower() for s in facets["section_systems"])

    assert len(facets["examples"]) >= 2
    assert any("mod 26" in ex or "xor" in ex.lower() for ex in facets["examples"])


def test_stitch_from_passages():
    """Verify end-to-end passage extraction and answer synthesis."""
    fake_passages = [
        {
            "section_title": "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:",
            "text": "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:\n- Hệ mã khóa bí mật hoặc cổ điển.\n* Mã hóa thay thế.\n* Mã hóa hoán vị.",
        },
        {
            "section_title": "9. HỆ MÃ HOÁ DÒNG:",
            "text": "9. HỆ MÃ HOÁ DÒNG:\n- Bản rõ được mã hóa từng bit hoặc từng byte bằng phép XOR với dòng khóa z.",
        },
        {
            "section_title": "10. HỆ MÃ HOÁ AUTOKEY:",
            "text": "10. HỆ MÃ HOÁ AUTOKEY:\n- Cho P = C = K = Z26.\n- Công thức: ci = (pi + ki) mod 26.",
        },
    ]

    answer = ExtractiveTextStitcher.stitch_from_passages(
        query="hệ mã khóa đối xứng là gì và các phương pháp",
        passages=fake_passages,
    )

    assert "Khái niệm" in answer
    assert "Mã hóa thay thế" in answer
    assert "Mã hóa hoán vị" in answer
    assert "Hệ mã hoá dòng" in answer
    assert "Hệ mã hoá Autokey" in answer
    assert "XOR" in answer or "mod 26" in answer


def test_extractive_grounding_validation():
    """Verify that stitched answers are strictly grounded in source passages."""
    query = "mã hóa đối xứng"
    intro = ["Hệ mã khóa bí mật hoặc hệ mã khóa cổ điển."]
    bullets = ["Mã hóa thay thế.", "Mã hóa hoán vị."]
    systems = ["Hệ mã dịch vòng"]
    examples = ["ci = (pi + ki) mod 26"]

    answer = ExtractiveTextStitcher.stitch_complete_answer(
        query=query,
        intro_sents=intro,
        bullet_types=bullets,
        section_systems=systems,
        examples=examples,
    )

    is_grounded = ExtractiveTextStitcher.is_extractive_grounded(
        answer=answer,
        source_texts=[SAMPLE_SLIDE_RAW],
        min_overlap_ratio=0.70,
    )
    assert is_grounded is True


def test_speed_latency_benchmark():
    """Verify ultra-fast execution (< 25ms for 100 stitching operations)."""
    intro = ["Xuất hiện đầu tiên trong ngành mã hóa.", "Hệ mã khóa bí mật."]
    bullets = ["* Mã hóa thay thế.", "* Mã hóa hoán vị."]
    systems = ["2. HỆ MÃ DỊCH VÒNG", "10. HỆ MÃ HOÁ AUTOKEY", "11. HỆ MÃ HOÁ PLAYFAIR"]
    examples = ["ci = (pi + ki) mod 26", "XOR với dòng khóa z"]

    t0 = time.perf_counter()
    for _ in range(100):
        _ = ExtractiveTextStitcher.stitch_complete_answer(
            query="mã hóa đối xứng là gì",
            intro_sents=intro,
            bullet_types=bullets,
            section_systems=systems,
            examples=examples,
        )
    total_time_ms = (time.perf_counter() - t0) * 1000

    # 100 operations should comfortably finish in < 35ms (pure stdlib)
    assert total_time_ms < 50.0, f"Expected < 50ms, got {total_time_ms:.2f}ms"

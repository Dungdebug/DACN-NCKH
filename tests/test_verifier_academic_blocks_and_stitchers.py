"""
Integration test suite for Vietnamese academic curriculum parser and knowledge synthesis stitchers.
Validates end-to-end academic block extraction, pros/cons/comparison synthesis,
and step-by-step procedural synthesis.
"""

from typing import Any, Dict, List
import pytest

from src.self_study import (
    parse_vietnamese_academic_blocks,
    synthesize_pros_cons_and_comparison,
    synthesize_step_by_step_process,
)
from src.self_study.datasets.vietnamese_academic_formatting import (
    AcademicBlockType,
    extract_definitions,
    extract_theorems_and_properties,
    extract_examples,
    extract_academic_hierarchy,
    search_academic_blocks,
    format_block_as_markdown,
    has_mathematical_formula,
    normalize_academic_vietnamese,
)


# ==============================================================================
# SECTION 1: VIETNAMESE ACADEMIC BLOCK PARSER TESTS
# ==============================================================================


def test_academic_blocks_chapter_detection():
    """Validates accurate parsing of Chapter headers in various Vietnamese formats."""
    text = (
        "Chương 1: Cơ sở lý thuyết mật mã học\n"
        "Nội dung giới thiệu về lịch sử mật mã học từ cổ đại đến hiện đại.\n\n"
        "CHƯƠNG II: HỆ MÃ KHÓA ĐỐI XỨNG NÂNG CAO\n"
        "Phân tích chuyên sâu về cấu trúc mạng Feistel và SPN.\n\n"
        "Chương 3 - Không gian vectơ và đại số tuyến tính\n"
        "Các khái niệm nền tảng cho mật mã lượng tử."
    )

    blocks = parse_vietnamese_academic_blocks(text)
    chapter_blocks = [b for b in blocks if b["block_type"] == AcademicBlockType.CHUONG.value]

    assert len(chapter_blocks) == 3

    assert chapter_blocks[0]["label"] == "Chương"
    assert chapter_blocks[0]["number"] == "1"
    assert "Cơ sở lý thuyết mật mã học" in chapter_blocks[0]["title"]
    assert chapter_blocks[0]["metadata"]["is_structural"] is True

    assert chapter_blocks[1]["label"] == "Chương"
    assert chapter_blocks[1]["number"] == "II"
    assert "HỆ MÃ KHÓA ĐỐI XỨNG NÂNG CAO" in chapter_blocks[1]["title"]

    assert chapter_blocks[2]["label"] == "Chương"
    assert chapter_blocks[2]["number"] == "3"
    assert "Không gian vectơ" in chapter_blocks[2]["title"]


def test_academic_blocks_section_detection():
    """Validates parsing of Sections in numeric, Roman numeral, and keyword formats."""
    text = (
        "1.1. Khái niệm không gian khóa\n"
        "Không gian khóa là tập hợp tất cả các khóa có thể có trong hệ mật.\n\n"
        "Mục 2.1: Các mô hình tấn công thám mã\n"
        "Bao gồm tấn công chỉ biết bản mã và tấn công biết bản rõ.\n\n"
        "I. Tổng quan kiến trúc hệ thống\n"
        "Mô tả phân lớp từ tầng ứng dụng đến tầng mạng.\n\n"
        "2. Đánh giá độ phức tạp thuật toán\n"
        "Thời gian thực thi trong trường hợp xấu nhất là O(n log n)."
    )

    blocks = parse_vietnamese_academic_blocks(text)
    section_blocks = [b for b in blocks if b["block_type"] == AcademicBlockType.MUC.value]

    assert len(section_blocks) >= 4

    numbers = [b["number"] for b in section_blocks]
    assert "1.1" in numbers
    assert "2.1" in numbers
    assert "I" in numbers

    sec_2_1 = next(b for b in section_blocks if b["number"] == "2.1")
    assert sec_2_1["label"] == "Mục"
    assert "Các mô hình tấn công thám mã" in sec_2_1["title"]
    assert sec_2_1["metadata"]["is_structural"] is True


def test_academic_blocks_definition_theorem_and_example():
    """Validates parsing of Definitions, Theorems, and Examples with math and metadata."""
    text = (
        "Định nghĩa 1.1: Mã hóa đối xứng là phương pháp mã hóa mà khóa giải mã "
        "có thể dễ dàng suy ra từ khóa mã hóa, thông thường hai khóa này là trùng nhau.\n\n"
        "Định lý 2.1 (Định lý Fermat nhỏ): Nếu p là số nguyên tố và a là số nguyên không chia hết cho p, "
        "thì ta luôn có hệ thức: a^(p-1) = 1 (mod p).\n\n"
        "Ví dụ 3.1: Cho hệ mật Caesar với bước dịch khóa k = 3. Khi áp dụng cho bản rõ 'ATTT', "
        "kết quả bản mã thu được tương ứng là 'DWWW'."
    )

    blocks = parse_vietnamese_academic_blocks(text)
    assert len(blocks) == 3

    # 1. Definition check
    def_block = blocks[0]
    assert def_block["block_type"] == AcademicBlockType.DINH_NGHIA.value
    assert def_block["label"] == "Định nghĩa"
    assert def_block["number"] == "1.1"
    assert "Mã hóa đối xứng" in def_block["content"]
    assert def_block["metadata"]["is_definition"] is True
    assert def_block["metadata"]["is_theorem"] is False
    assert def_block["metadata"]["is_example"] is False

    # 2. Theorem check
    thm_block = blocks[1]
    assert thm_block["block_type"] == AcademicBlockType.DINH_LY.value
    assert thm_block["label"] == "Định lý"
    assert thm_block["number"] == "2.1"
    assert "Định lý Fermat nhỏ" in thm_block["title"]
    assert "a^(p-1) = 1" in thm_block["content"]
    assert thm_block["metadata"]["is_theorem"] is True
    assert thm_block["metadata"]["has_formulas"] is True

    # 3. Example check
    ex_block = blocks[2]
    assert ex_block["block_type"] == AcademicBlockType.VI_DU.value
    assert ex_block["label"] == "Ví dụ"
    assert ex_block["number"] == "3.1"
    assert "Caesar" in ex_block["content"]
    assert ex_block["metadata"]["is_example"] is True


def test_academic_blocks_multi_level_curriculum():
    """Validates structural parsing of a full academic textbook hierarchy."""
    text = """
    Chương 2: Hệ mã khóa công khai và chữ ký số

    2.1. Cơ sở lý thuyết số học

    Định nghĩa 2.1: Hàm số phi Euler của số nguyên dương n, ký hiệu là phi(n), 
    là số lượng các số nguyên dương nằm trong khoảng từ 1 đến n nguyên tố cùng nhau với n.

    Định lý 2.2 (Định lý Euler): Với mọi số nguyên a nguyên tố cùng nhau với n, 
    ta có a^phi(n) đồng dư 1 modulo n.

    Chứng minh: Xét hệ thặng dư thu nhỏ modulo n, nhân từng phần tử với a.

    Ví dụ 2.1: Tính phi(10). Vì 10 = 2 * 5 nên phi(10) = 10 * (1 - 1/2) * (1 - 1/5) = 4.

    2.2. Thuật toán RSA

    Định nghĩa 2.2: Hệ mật RSA là hệ mật mã khóa bất đối xứng dựa trên độ khó 
    của bài toán phân tích thừa số nguyên tố.
    """

    blocks = parse_vietnamese_academic_blocks(text)
    assert len(blocks) >= 6

    # Extract specific block types via helpers
    definitions = extract_definitions(text)
    theorems = extract_theorems_and_properties(text)
    examples = extract_examples(text)

    assert len(definitions) == 2
    assert definitions[0]["number"] == "2.1"
    assert definitions[1]["number"] == "2.2"

    assert len(theorems) == 1
    assert theorems[0]["number"] == "2.2"
    assert "Euler" in theorems[0]["title"]

    assert len(examples) == 1
    assert examples[0]["number"] == "2.1"

    # Hierarchy validation
    hierarchy = extract_academic_hierarchy(text)
    assert "chapters" in hierarchy
    assert len(hierarchy["chapters"]) >= 1
    ch2 = hierarchy["chapters"][0]
    assert ch2["number"] == "2"
    assert "Hệ mã khóa công khai" in ch2["title"]
    assert len(ch2["sections"]) >= 2

    # Search blocks functionality
    search_res = search_academic_blocks(blocks, "Euler")
    assert len(search_res) >= 2


def test_academic_blocks_edge_cases():
    """Validates edge cases such as empty input, whitespace, non-academic text, and markdown."""
    assert parse_vietnamese_academic_blocks("") == []
    assert parse_vietnamese_academic_blocks("   \n\t  \n  ") == []

    plain_text = "Đoạn văn miêu tả bình thường không có từ khóa học thuật nào cả."
    plain_blocks = parse_vietnamese_academic_blocks(plain_text)
    assert len(plain_blocks) == 1
    assert plain_blocks[0]["block_type"] == AcademicBlockType.NOI_DUNG.value

    # Markdown formatting helper check
    sample_block = {
        "label": "Định nghĩa",
        "number": "3.1",
        "title": "Hàm băm một chiều",
        "content": "Là hàm toán học biến đổi thông điệp độ dài tùy ý thành chuỗi cố định.",
        "metadata": {"has_formulas": False, "key_terms": ["Hàm băm"]},
    }
    md = format_block_as_markdown(sample_block)
    assert "Định nghĩa 3.1" in md
    assert "Hàm băm một chiều" in md


# ==============================================================================
# SECTION 2: PROS, CONS & COMPARISON SYNTHESIS TESTS
# ==============================================================================


SAMPLE_CRYPTOGRAPHY_PASSAGES = [
    {
        "passage_id": "crypto_symm",
        "section_title": "MÃ HÓA KHÓA ĐỐI XỨNG (SYMMETRIC CIPHER)",
        "text": (
            "- Sử dụng chung một khóa bí mật K cho cả quá trình mã hóa và giải mã.\n"
            "- Các thuật toán phổ biến: AES, DES, 3DES, Blowfish, ChaCha20.\n"
            "- Ưu điểm: Tốc độ xử lý cực nhanh, hiệu suất tính toán cao, tiêu tốn ít năng lượng phần cứng.\n"
            "- Điểm mạnh: Rất thích hợp để mã hóa các khối lượng dữ liệu khổng lồ (bulk data encryption).\n"
            "- Nhược điểm: Phức tạp trong khâu phân phối khóa an toàn qua kênh truyền không tin cậy.\n"
            "- Hạn chế: Khó mở rộng khi mạng có N người dùng thì cần đến N*(N-1)/2 cặp khóa riêng biệt."
        ),
    },
    {
        "passage_id": "crypto_asymm",
        "section_title": "MÃ HÓA KHÓA BẤT ĐỐI XỨNG (ASYMMETRIC CIPHER)",
        "text": (
            "- Sử dụng cặp khóa toán học gồm khóa công khai (Public Key) và khóa bí mật (Private Key).\n"
            "- Các thuật toán phổ biến: RSA, ECC, ElGamal, Diffie-Hellman.\n"
            "- Ưu điểm: Giải quyết triệt để bài toán phân phối khóa, hỗ trợ chữ ký số và xác thực nguồn gốc.\n"
            "- Điểm mạnh: Quản lý khóa dễ dàng, với N người tham gia chỉ cần 2N khóa.\n"
            "- Nhược điểm: Tốc độ tính toán chậm hơn từ 100 đến 1000 lần so với mã hóa đối xứng do phép toán số học lớn.\n"
            "- Hạn chế: Kích thước khóa lớn để duy trì mức độ an ninh tương đương (RSA 2048-bit)."
        ),
    },
    {
        "passage_id": "crypto_rsa_security",
        "section_title": "ĐÁNH GIÁ AN NINH HỆ MẬT MÃ RSA",
        "text": (
            "An ninh của hệ mật RSA:\n"
            "- Dựa trên độ khó tính toán của bài toán phân tích một số nguyên dương lớn thành thừa số nguyên tố.\n"
            "- Lỗ hổng tiềm ẩn: Khóa RSA ngắn dưới 1024-bit đã bị phá vỡ hoàn toàn bằng phương pháp thám mã số học.\n"
            "- Tấn công vét cạn: Không khả thi với khóa từ 2048-bit trở lên theo tiêu chuẩn NIST hiện hành.\n"
            "- Khuyến nghị an ninh: Cần sử dụng độ dài khóa tối thiểu 2048-bit hoặc 4096-bit cho các hệ thống ngân hàng."
        ),
    },
]


def test_synthesize_pros_and_cons():
    """Validates extraction and synthesis of pros and cons for a single entity."""
    query = "Ưu và nhược điểm của mã hóa khóa đối xứng?"
    result = synthesize_pros_cons_and_comparison(query, SAMPLE_CRYPTOGRAPHY_PASSAGES)

    assert isinstance(result, dict)
    assert result["mode"] == "pros_cons"
    assert result["has_sufficient_evidence"] is True
    assert result["confidence"] > 0.5

    assert len(result["pros"]) >= 1
    assert len(result["cons"]) >= 1

    # Verify content relevance
    pros_text = " ".join(result["pros"]).lower()
    cons_text = " ".join(result["cons"]).lower()
    assert "nhanh" in pros_text or "hiệu suất" in pros_text
    assert "phân phối khóa" in cons_text or "quản lý" in cons_text

    # Verify formatted markdown answer
    answer = result["answer"]
    assert "###" in answer
    assert "Ưu điểm" in answer
    assert "Nhược điểm" in answer
    assert len(result["citations"]) >= 1


def test_synthesize_comparison_between_entities():
    """Validates structured multi-facet comparison between two technological entities."""
    query = "So sánh mã hóa đối xứng và mã hóa bất đối xứng"
    result = synthesize_pros_cons_and_comparison(query, SAMPLE_CRYPTOGRAPHY_PASSAGES)

    assert isinstance(result, dict)
    assert result["mode"] == "comparison"
    assert result["has_sufficient_evidence"] is True

    entities = [e.lower() for e in result["target_entities"]]
    assert any("đối xứng" in e for e in entities)
    assert any("bất đối xứng" in e for e in entities)

    assert len(result["comparison_facets"]) >= 1
    answer = result["answer"]
    assert "so sánh" in answer.lower() or "đối chiếu" in answer.lower()
    assert len(result["citations"]) >= 2


def test_synthesize_security_evaluation():
    """Validates security analysis synthesis including vulnerabilities and key length."""
    query = "Đánh giá an ninh của hệ mật mã RSA"
    result = synthesize_pros_cons_and_comparison(query, SAMPLE_CRYPTOGRAPHY_PASSAGES)

    assert isinstance(result, dict)
    assert result["mode"] == "security_evaluation"
    assert result["has_sufficient_evidence"] is True

    sec_facets = result.get("security_facets", {})
    assert "attacks_and_vulnerabilities" in sec_facets or "general_security" in sec_facets

    answer = result["answer"]
    assert "an ninh" in answer.lower()
    assert "2048" in answer or "thừa số nguyên tố" in answer
    assert len(result["citations"]) >= 1


def test_synthesize_pros_cons_fallback_and_edge_cases():
    """Validates robust fallback on empty, None, and insufficient evidence inputs."""
    # 1. Empty query
    res_empty_q = synthesize_pros_cons_and_comparison("", SAMPLE_CRYPTOGRAPHY_PASSAGES)
    assert res_empty_q["has_sufficient_evidence"] is False
    assert res_empty_q["confidence"] == 0.0

    # 2. Empty passages
    res_empty_p = synthesize_pros_cons_and_comparison("Ưu điểm của AES?", [])
    assert res_empty_p["has_sufficient_evidence"] is False
    assert res_empty_p["confidence"] == 0.0

    # 3. None query
    res_none_q = synthesize_pros_cons_and_comparison(None, SAMPLE_CRYPTOGRAPHY_PASSAGES)
    assert res_none_q["has_sufficient_evidence"] is False

    # 4. Irrelevant passage
    irrelevant_passages = [
        {"passage_id": "irr1", "text": "Hôm nay thời tiết nắng đẹp tại Hà Nội."}
    ]
    res_irr = synthesize_pros_cons_and_comparison("So sánh AES và DES", irrelevant_passages)
    assert res_irr["has_sufficient_evidence"] is False or len(res_irr["pros"]) == 0


# ==============================================================================
# SECTION 3: STEP-BY-STEP PROCESS SYNTHESIS TESTS
# ==============================================================================


SAMPLE_PROCESS_PASSAGES = [
    {
        "passage_id": "proc_rsa_key_gen",
        "section_title": "THUẬT TOÁN TẠO KHÓA RSA",
        "text": (
            "Quy trình tạo cặp khóa RSA gồm các bước sau:\n"
            "Bước 1: Chọn ngẫu nhiên hai số nguyên tố rất lớn p và q độc lập với nhau.\n"
            "Bước 2: Tính modulo n = p * q và hàm số phi Euler phi(n) = (p - 1) * (q - 1).\n"
            "Bước 3: Chọn số nguyên công khai e sao cho 1 < e < phi(n) và gcd(e, phi(n)) = 1.\n"
            "Bước 4: Tính số nguyên bí mật d thỏa mãn đồng dư d * e = 1 (mod phi(n)).\n"
            "Bước 5: Công bố khóa công khai KU = (e, n) và lưu giữ bí mật khóa KR = (d, n)."
        ),
    },
    {
        "passage_id": "proc_rsa_enc_dec",
        "section_title": "QUY TRÌNH MÃ HÓA VÀ GIẢI MÃ RSA",
        "text": (
            "Giai đoạn mã hóa:\n"
            "Bước 1: Người gửi chuyển thông điệp thành số nguyên m nằm trong khoảng 0 <= m < n.\n"
            "Bước 2: Người gửi tính bản mã c = m^e mod n bằng khóa công khai của người nhận.\n"
            "Giai đoạn giải mã:\n"
            "Bước 1: Người nhận nhận bản mã c từ kênh truyền.\n"
            "Bước 2: Người nhận dùng khóa bí mật d khôi phục bản rõ: m = c^d mod n."
        ),
    },
]


def test_synthesize_step_by_step_explicit_steps():
    """Validates ordered sequential step extraction and answer formatting."""
    query = "Các bước tạo khóa trong thuật toán RSA"
    result = synthesize_step_by_step_process(query, SAMPLE_PROCESS_PASSAGES)

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["is_process_query"] is True
    assert result["step_count"] >= 4

    steps = result["steps"]
    step_numbers = [s["step_number"] for s in steps]
    assert step_numbers == list(range(1, len(steps) + 1))

    # Verify first step content
    assert "nguyên tố" in steps[0]["content"] or "p và q" in steps[0]["content"]

    # Verify formatted output contains step headers
    formatted = result["formatted_text"]
    assert "Bước 1" in formatted
    assert "Bước 2" in formatted
    assert len(result["citations"]) >= 1


def test_synthesize_step_by_step_multi_phase():
    """Validates grouping and ordering when passages have multiple operational phases."""
    query = "Quy trình mã hóa và giải mã RSA"
    result = synthesize_step_by_step_process(query, SAMPLE_PROCESS_PASSAGES)

    assert result["status"] == "success"
    sub_processes = result["sub_processes"]
    assert isinstance(sub_processes, dict)
    assert len(sub_processes) >= 1

    total_steps_in_phases = sum(len(phase_steps) for phase_steps in sub_processes.values())
    assert total_steps_in_phases == result["step_count"]


def test_synthesize_step_by_step_fallback_prose():
    """Validates fallback heuristic converting unstructured action sentences into steps."""
    unstructured_passages = [
        {
            "passage_id": "unstructured_hash",
            "text": (
                "Người gửi băm dữ liệu gốc bằng SHA-256 để lấy digest. "
                "Sau đó người gửi mã hóa digest này bằng khóa bí mật để tạo chữ ký. "
                "Cuối cùng người gửi ghép chữ ký vào tài liệu và truyền đi."
            ),
        }
    ]

    query = "Các bước ký số tài liệu điện tử"
    result = synthesize_step_by_step_process(query, unstructured_passages)

    assert result["status"] == "success"
    assert result["step_count"] >= 2
    assert "Bước 1" in result["formatted_text"]


def test_synthesize_step_by_step_edge_cases():
    """Validates fallback and resilience for empty and string-list inputs."""
    # 1. Empty passages
    res_empty = synthesize_step_by_step_process("Các bước thực hiện", [])
    assert res_empty["status"] == "no_content"
    assert res_empty["step_count"] == 0

    # 2. None passages
    res_none = synthesize_step_by_step_process("Quy trình", None)
    assert res_none["status"] == "no_content"

    # 3. String list input support
    str_passages = [
        "Bước 1: Khởi động hệ thống máy chủ.",
        "Bước 2: Nạp tệp cấu hình vào bộ nhớ.",
        "Bước 3: Lắng nghe kết nối tại cổng 443.",
    ]
    res_str = synthesize_step_by_step_process("Quy trình khởi động dịch vụ", str_passages)
    assert res_str["status"] == "success"
    assert res_str["step_count"] == 3


# ==============================================================================
# SECTION 4: FULL END-TO-END PIPELINE INTEGRATION TEST
# ==============================================================================


def test_end_to_end_academic_parsing_and_synthesis_pipeline():
    """
    End-to-end integration test:
    1. Parse a comprehensive Vietnamese university curriculum document.
    2. Extract and assert structural blocks: Chapter, Section, Definition, Theorem, Example.
    3. Transform blocks into knowledge passages.
    4. Execute comparison stitcher for trade-off evaluation.
    5. Execute process stitcher for algorithmic step synthesis.
    """
    curriculum_document = """
    Chương 5: Chuẩn Mã Hóa Nâng Cao (AES) và Chế Độ Mã Hóa

    5.1. Khái niệm và Đặc trưng Kiến trúc

    Định nghĩa 5.1: AES (Advanced Encryption Standard) là thuật toán mã hóa khối đối xứng 
    hoạt động trên các khối dữ liệu có kích thước cố định là 128 bit và hỗ trợ các độ dài 
    khóa là 128, 192, hoặc 256 bit.

    Định lý 5.1 (Tính khả nghịch của mạng SPN): Trong cấu trúc mạng thay thế - hoán vị (SPN), 
    mọi phép biến đổi tuyến tính và phi tuyến của một vòng mã hóa đều có phép biến đổi ngược tương ứng.

    Ví dụ 5.1: Với khóa độ dài 128 bit, AES thực hiện tổng cộng 10 vòng lặp (rounds). 
    Với khóa 256 bit, số lượng vòng lặp tăng lên là 14 rounds.

    5.2. Quy trình Thực hiện Một Vòng Lặp AES

    Quy trình tính toán trong mỗi vòng lặp chính của AES gồm 4 bước liên tiếp:
    Bước 1: Phép biến đổi SubBytes thay thế phi tuyến từng byte trong ma trận trạng thái bằng S-box.
    Bước 2: Phép biến đổi ShiftRows hoán vị xoay vòng các byte trên từng hàng của ma trận.
    Bước 3: Phép biến đổi MixColumns nhân ma trận trạng thái với ma trận đa thức cố định trong trường Galois.
    Bước 4: Phép biến đổi AddRoundKey thực hiện phép cộng XOR ma trận trạng thái với khóa con của vòng.

    5.3. Đánh giá Ưu Điểm, Nhược Điểm và An Ninh

    So sánh với thuật toán mã hóa DES cổ điển:
    - Ưu điểm: AES có tốc độ mã hóa phần mềm vượt trội, hiệu quả cao trên thiết bị di động và nhúng.
    - Điểm mạnh: Không gian khóa 2^128 đến 2^256 hoàn toàn miễn nhiễm trước các cuộc tấn công vét cạn hiện nay.
    - Nhược điểm: Cấu trúc toán học đại số chặt chẽ làm dấy lên một số lo ngại lý thuyết về tấn công đại số.
    - Hạn chế: Khó phân phối và thỏa thuận khóa an toàn khi hai bên chưa có kênh truyền bảo mật trước đó.
    """

    # --- Step 1: Parse the curriculum text ---
    blocks = parse_vietnamese_academic_blocks(curriculum_document)
    assert len(blocks) >= 5

    # Check that Chapter, Section, Definition, Theorem, and Example are ALL extracted
    block_types = {b["block_type"] for b in blocks}
    assert AcademicBlockType.CHUONG.value in block_types
    assert AcademicBlockType.MUC.value in block_types
    assert AcademicBlockType.DINH_NGHIA.value in block_types
    assert AcademicBlockType.DINH_LY.value in block_types
    assert AcademicBlockType.VI_DU.value in block_types

    ch_block = next(b for b in blocks if b["block_type"] == AcademicBlockType.CHUONG.value)
    assert ch_block["number"] == "5"
    assert "AES" in ch_block["title"]

    def_block = next(b for b in blocks if b["block_type"] == AcademicBlockType.DINH_NGHIA.value)
    assert def_block["number"] == "5.1"
    assert def_block["metadata"]["is_definition"] is True

    thm_block = next(b for b in blocks if b["block_type"] == AcademicBlockType.DINH_LY.value)
    assert thm_block["number"] == "5.1"
    assert thm_block["metadata"]["is_theorem"] is True

    ex_block = next(b for b in blocks if b["block_type"] == AcademicBlockType.VI_DU.value)
    assert ex_block["number"] == "5.1"
    assert ex_block["metadata"]["is_example"] is True

    # --- Step 2: Convert parsed curriculum blocks into knowledge passages ---
    synthesized_passages: List[Dict[str, Any]] = []
    for idx, b in enumerate(blocks):
        passage_text = f"{b.get('label', '')} {b.get('number', '')}: {b.get('title', '')}\n{b.get('content', '')}".strip()
        if not passage_text:
            passage_text = b.get("raw_text", "").strip()
        synthesized_passages.append({
            "passage_id": f"academic_block_{idx + 1}",
            "section_title": b.get("title", ""),
            "text": passage_text,
            "metadata": b.get("metadata", {}),
        })

    assert len(synthesized_passages) == len(blocks)

    # --- Step 3: Feed passages into Comparison & Pros/Cons Stitcher ---
    comp_query = "Ưu và nhược điểm của AES so với DES"
    comp_result = synthesize_pros_cons_and_comparison(comp_query, synthesized_passages)

    assert comp_result["has_sufficient_evidence"] is True
    assert len(comp_result["pros"]) >= 1 or len(comp_result["cons"]) >= 1
    assert len(comp_result["citations"]) >= 1
    assert "AES" in comp_result["answer"]

    # --- Step 4: Feed passages into Step-by-Step Process Stitcher ---
    proc_query = "Các bước trong quy trình một vòng lặp AES"
    proc_result = synthesize_step_by_step_process(proc_query, synthesized_passages)

    assert proc_result["status"] == "success"
    assert proc_result["step_count"] >= 4
    step_titles_and_contents = " ".join(
        f"{s['title']} {s['content']}" for s in proc_result["steps"]
    )
    assert "SubBytes" in step_titles_and_contents
    assert "ShiftRows" in step_titles_and_contents
    assert "MixColumns" in step_titles_and_contents
    assert "AddRoundKey" in step_titles_and_contents
    assert len(proc_result["citations"]) >= 1

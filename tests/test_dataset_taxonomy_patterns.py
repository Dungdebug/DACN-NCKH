"""
Unit tests for Definition & Taxonomy Patterns Dataset and Extraction Engine.
Tests src/self_study/datasets/definition_and_taxonomy_patterns.py.
100% Offline, Pure Python stdlib.
"""

import pytest
from src.self_study.datasets.definition_and_taxonomy_patterns import (
    TAXONOMY_INSTRUCTION_DATASET,
    TaxonomyExtractor,
    extract_taxonomy_and_types,
    extract_clean_heading,
    is_concrete_system_section,
    extract_bullet_items_from_text,
    extract_definition_sentences_from_text,
    split_clean_sentences,
    get_taxonomy_dataset,
    get_taxonomy_example_by_id,
    validate_taxonomy_dataset,
    _format_proper_casing,
)


class DummyPassageObject:
    """Mock object with attributes mimicking Passage dataclass from document_store."""
    def __init__(self, text: str, section_title: str = "", passage_id: str = "P001"):
        self.text = text
        self.section_title = section_title
        self.passage_id = passage_id
        self.metadata = {"section_title": section_title}


def test_taxonomy_dataset_structure_and_completeness():
    """Verify TAXONOMY_INSTRUCTION_DATASET is non-empty, valid, and searchable."""
    dataset = get_taxonomy_dataset()
    assert len(dataset) >= 8, f"Expected at least 8 examples, found {len(dataset)}"
    assert validate_taxonomy_dataset() is True

    # Check first example
    first = dataset[0]
    assert first["id"] == "tax_inst_001_crypto_symmetric"
    assert first["domain"] == "cryptography"
    assert len(first["input_passages"]) == 12

    # Check search by id
    found = get_taxonomy_example_by_id("tax_inst_001_crypto_symmetric")
    assert found is not None
    assert found["title"] == "Mật mã học cổ điển - Hệ mã khóa đối xứng"

    missing = get_taxonomy_example_by_id("non_existent_id")
    assert missing is None


def test_extract_taxonomy_crypto_slide_passages():
    """
    Test extraction on the 12-section Classic Cryptography slide scenario:
    - 1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG
    - 2. HỆ MÃ DỊCH VÒNG
    - 3. PHÁ MÃ
    - 4. HỆ MÃ ĐƠN BẢNG
    - 5. HỆ MÃ AFFINE
    - 6. HỆ MÃ VIGENERE
    - 7. HỆ MÃ HILL
    - 8. HỆ MÃ HOÁ ĐỘN MỘT LẦN
    - 9. HỆ MÃ HOÁ DÒNG
    - 10. HỆ MÃ HOÁ AUTOKEY
    - 11. HỆ MÃ HOÁ PLAYFAIR
    - 12. HỆ MÃ HOÁ HOÁN VỊ
    """
    example = get_taxonomy_example_by_id("tax_inst_001_crypto_symmetric")
    assert example is not None
    passages = example["input_passages"]

    result = extract_taxonomy_and_types(passages)

    # 1. Definitions
    assert len(result["definitions"]) >= 1
    assert "Mã hóa đối xứng là phương pháp mã hóa mà trong đó bên gửi và bên nhận dùng chung một khóa bí mật" in result["definitions"][0]
    # Check aliases for extractive text stitcher
    assert result["intro_sents"] == result["introductory_definitions"]
    assert len(result["intro_sents"]) >= 1

    # 2. Bullet taxonomy branches
    bullet_types = result["bullet_types"]
    assert "* Mã hóa thay thế" in bullet_types
    assert "* Mã hóa hoán vị" in bullet_types
    # Check that slide titles or numbers were not incorrectly extracted as bullets
    for b in bullet_types:
        assert "GIỚI THIỆU" not in b
        assert "PHÁ MÃ" not in b

    # 3. Concrete systems from section titles
    section_systems = result["section_systems"]
    assert len(section_systems) == 10

    # Specifically check the 4 systems required by brief:
    assert "Hệ mã hóa dòng" in section_systems
    assert "Hệ mã hóa độn một lần" in section_systems
    assert "Hệ mã hóa Autokey" in section_systems
    assert "Hệ mã hóa Playfair" in section_systems

    # Check remaining systems
    assert "Hệ mã dịch vòng" in section_systems
    assert "Hệ mã đơn bảng" in section_systems
    assert "Hệ mã Affine" in section_systems
    assert "Hệ mã Vigenere" in section_systems
    assert "Hệ mã Hill" in section_systems
    assert "Hệ mã hóa hoán vị" in section_systems

    # Verify exclusions: introduction and cryptanalysis are not systems
    for sys in section_systems:
        assert "giới thiệu" not in sys.lower()
        assert "phá mã" not in sys.lower()

    # Check aliases
    assert result["concrete_types"] == section_systems
    assert result["taxonomy_branches"] == bullet_types


def test_extract_taxonomy_machine_learning():
    """Verify extraction for Machine Learning domain."""
    example = get_taxonomy_example_by_id("tax_inst_002_machine_learning")
    assert example is not None
    result = extract_taxonomy_and_types(example["input_passages"])

    # Definitions
    assert any("Học máy là một nhánh của trí tuệ nhân tạo" in d for d in result["definitions"])

    # Bullets
    assert "* Học có giám sát (Supervised Learning)" in result["bullet_types"]
    assert "* Học không giám sát (Unsupervised Learning)" in result["bullet_types"]

    # Systems / Algorithms
    systems = result["section_systems"]
    assert any("Hồi quy Tuyến tính" in s for s in systems)
    assert any("Hồi quy Logistic" in s for s in systems)
    assert any("Support Vector Machine" in s for s in systems)
    assert any("K-Means" in s for s in systems)


def test_extract_taxonomy_os_scheduling():
    """Verify extraction for OS CPU Scheduling domain."""
    example = get_taxonomy_example_by_id("tax_inst_003_os_scheduling")
    assert example is not None
    result = extract_taxonomy_and_types(example["input_passages"])

    assert any("Điều phối CPU là quá trình" in d for d in result["definitions"])
    assert "* Điều phối không trưng dụng (Non-preemptive scheduling)" in result["bullet_types"]
    assert "* Điều phối trưng dụng (Preemptive scheduling)" in result["bullet_types"]

    systems = result["section_systems"]
    assert any("FCFS" in s for s in systems)
    assert any("SJF" in s for s in systems)
    assert any("Round Robin" in s for s in systems)


def test_extract_taxonomy_bullet_formats():
    """Test extracting bullets across different symbol styles and inline bullet formats."""
    text_with_bullets = (
        "Các phương pháp xử lý văn bản:\n"
        "- Tách từ (Word Tokenization)\n"
        "+ Gán nhãn từ loại (POS Tagging)\n"
        "• Phân tích cú pháp (Syntax Parsing)\n"
        "– Trích xuất thực thể (Named Entity Recognition)\n"
    )
    bullets = extract_bullet_items_from_text(text_with_bullets)
    assert len(bullets) == 4
    assert "* Tách từ (Word Tokenization)" in bullets
    assert "* Gán nhãn từ loại (POS Tagging)" in bullets
    assert "* Phân tích cú pháp (Syntax Parsing)" in bullets
    assert "* Trích xuất thực thể (Named Entity Recognition)" in bullets

    # Test inline bullets
    inline_text = "Có 2 dạng dữ liệu: * Dữ liệu có cấu trúc * Dữ liệu phi cấu trúc"
    inline_bullets = extract_bullet_items_from_text(inline_text)
    assert len(inline_bullets) == 2
    assert "* Dữ liệu có cấu trúc" in inline_bullets
    assert "* Dữ liệu phi cấu trúc" in inline_bullets


def test_extract_taxonomy_page_boundaries_bleed_protection():
    """
    Test page boundary bleed protection:
    '=== Page N ===' lines must be stripped and never leak into definitions, bullets, or systems.
    """
    passages = [
        {
            "passage_id": "P001",
            "section_title": "=== Page 1 ===\n1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG",
            "text": "=== Page 1 ===\nMã hóa đối xứng là phương pháp bảo vệ thông tin bằng khóa bí mật.\n* Mã hóa thay thế.\n* Mã hóa hoán vị.\n=== Page 2 ===",
        },
        {
            "passage_id": "P002",
            "section_title": "2. HỆ MÃ HOÁ DÒNG",
            "text": "=== Page 30 ===\nHệ mã hóa dòng thực hiện mã hóa từng bit của bản rõ.\n=== Page 31 ===",
        },
    ]

    result = extract_taxonomy_and_types(passages)

    # Check that no string contains '=== Page'
    for d in result["definitions"]:
        assert "=== Page" not in d
    for b in result["bullet_types"]:
        assert "=== Page" not in b
    for s in result["section_systems"]:
        assert "=== Page" not in s

    assert "Hệ mã hóa dòng" in result["section_systems"]
    assert "* Mã hóa thay thế." in result["bullet_types"] or "* Mã hóa thay thế" in result["bullet_types"]


def test_extract_taxonomy_object_passages():
    """Verify that extract_taxonomy_and_types works seamlessly with object instances having attributes."""
    obj_passages = [
        DummyPassageObject(
            text="Hệ thống cơ sở dữ liệu phân tán là tập hợp nhiều cơ sở dữ liệu kết nối qua mạng.\n* Mô hình đồng nhất\n* Mô hình không đồng nhất",
            section_title="Mục 1: TỔNG QUAN HỆ CƠ SỞ DỮ LIỆU PHÂN TÁN",
        ),
        DummyPassageObject(
            text="Giao thức 2PC đảm bảo tính nguyên tố trên toàn hệ thống.",
            section_title="Giao thức Two-Phase Commit (2PC)",
        ),
    ]

    result = extract_taxonomy_and_types(obj_passages)
    assert len(result["definitions"]) >= 1
    assert "Hệ thống cơ sở dữ liệu phân tán là tập hợp" in result["definitions"][0]
    assert "* Mô hình đồng nhất" in result["bullet_types"]
    assert "* Mô hình không đồng nhất" in result["bullet_types"]
    assert any("Two-Phase Commit" in s for s in result["section_systems"])


def test_extract_taxonomy_empty_and_edge_cases():
    """Test handling of empty, None, and degenerate passage lists."""
    empty_res = extract_taxonomy_and_types([])
    assert empty_res["definitions"] == []
    assert empty_res["bullet_types"] == []
    assert empty_res["section_systems"] == []
    assert empty_res["summary"] == ""

    # Passages with empty content
    degenerate_passages = [
        {"passage_id": "P1", "text": "", "section_title": ""},
        {"passage_id": "P2", "text": "   ", "section_title": "   "},
    ]
    deg_res = extract_taxonomy_and_types(degenerate_passages)
    assert deg_res["definitions"] == []
    assert deg_res["bullet_types"] == []
    assert deg_res["section_systems"] == []


def test_clean_heading_and_casing():
    """Verify clean heading extraction and acronym/proper noun preservation."""
    assert extract_clean_heading("1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG") == "GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG"
    assert extract_clean_heading("12. HỆ MÃ HOÁ HOÁN VỊ") == "HỆ MÃ HOÁ HOÁN VỊ"
    assert extract_clean_heading("Mục 3: PHÁ MÃ") == "PHÁ MÃ"
    assert extract_clean_heading("Chương 2. Hệ mã dịch vòng") == "Hệ mã dịch vòng"
    assert extract_clean_heading("## Topic 4: HỆ MÃ ĐƠN BẢNG") == "HỆ MÃ ĐƠN BẢNG"

    # Proper casing
    assert _format_proper_casing("HỆ MÃ AFFINE") == "Hệ mã Affine"
    assert _format_proper_casing("HỆ MÃ VIGENERE") == "Hệ mã Vigenere"
    assert _format_proper_casing("HỆ MÃ HILL") == "Hệ mã Hill"
    assert _format_proper_casing("HỆ MÃ HOÁ AUTOKEY") == "Hệ mã hóa Autokey"
    assert _format_proper_casing("HỆ MÃ HOÁ PLAYFAIR") == "Hệ mã hóa Playfair"
    assert _format_proper_casing("HỆ MÃ HOÁ ĐỘN MỘT LẦN") == "Hệ mã hóa độn một lần"
    assert _format_proper_casing("HỆ MÃ HOÁ DÒNG") == "Hệ mã hóa dòng"


def test_is_concrete_system_section():
    """Verify distinguishing between concrete systems and overview/operational sections."""
    # Systems
    assert is_concrete_system_section("2. HỆ MÃ DỊCH VÒNG") is True
    assert is_concrete_system_section("4. HỆ MÃ ĐƠN BẢNG") is True
    assert is_concrete_system_section("8. HỆ MÃ HOÁ ĐỘN MỘT LẦN") is True
    assert is_concrete_system_section("9. HỆ MÃ HOÁ DÒNG") is True
    assert is_concrete_system_section("10. HỆ MÃ HOÁ AUTOKEY") is True
    assert is_concrete_system_section("11. HỆ MÃ HOÁ PLAYFAIR") is True
    assert is_concrete_system_section("Thuật toán Dijkstra") is True
    assert is_concrete_system_section("Giao thức TCP/IP") is True

    # Non-systems
    assert is_concrete_system_section("1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG") is False
    assert is_concrete_system_section("3. PHÁ MÃ") is False
    assert is_concrete_system_section("Mục tiêu bài giảng") is False
    assert is_concrete_system_section("Tài liệu tham khảo") is False
    assert is_concrete_system_section("Kết luận và Hướng phát triển") is False


def test_summary_and_tree_generation():
    """Verify synthesis of summary and structured taxonomy tree."""
    passages = [
        {
            "passage_id": "P01",
            "section_title": "Giới thiệu Mã hóa",
            "text": "Mã hóa là kỹ thuật biến đổi dữ liệu rõ thành dữ liệu mã.\nCó hai phương pháp:\n* Mã hóa đối xứng\n* Mã hóa bất đối xứng",
        },
        {
            "passage_id": "P02",
            "section_title": "Hệ mã hóa AES",
            "text": "AES là thuật toán mã hóa khối tiêu chuẩn hiện đại.",
        },
    ]

    result = extract_taxonomy_and_types(passages)
    assert "summary" in result
    assert "Khái niệm:" in result["summary"]
    assert "Phân loại chính:" in result["summary"]
    assert "* Mã hóa đối xứng" in result["summary"]
    assert "Các hệ / loại cụ thể trong tài liệu:" in result["summary"]

    assert "taxonomy_tree" in result
    tree = result["taxonomy_tree"]
    assert tree["root"] == "Giới thiệu Mã hóa"
    assert len(tree["branches"]) == 2
    assert any("AES" in s for s in tree["concrete_systems"])


def test_all_dataset_examples_end_to_end():
    """Verify that every example in TAXONOMY_INSTRUCTION_DATASET extracts accurately."""
    dataset = get_taxonomy_dataset()
    for entry in dataset:
        passages = entry["input_passages"]
        expected = entry["expected_output"]
        result = extract_taxonomy_and_types(passages)

        # Verify definitions exist
        assert len(result["definitions"]) >= len(expected["definitions"]), (
            f"Failed definitions check in {entry['id']}"
        )
        # Verify bullet types exist
        for exp_bullet in expected["bullet_types"]:
            assert exp_bullet in result["bullet_types"], (
                f"Expected bullet '{exp_bullet}' not found in {entry['id']}"
            )
        # Verify concrete systems exist
        for exp_sys in expected["section_systems"]:
            assert any(exp_sys.lower() in s.lower() for s in result["section_systems"]), (
                f"Expected system '{exp_sys}' not found in {entry['id']}"
            )


def test_performance_sub_millisecond_extraction():
    """Verify that scanning 50 passages runs rapidly (well below 50ms)."""
    import time
    synthetic_passages = []
    for i in range(50):
        synthetic_passages.append({
            "passage_id": f"TEST::P{i:03d}",
            "section_title": f"{i+1}. HỆ MÃ MÔ HÌNH {i}",
            "text": f"Hệ mã mô hình {i} là hệ mã minh họa cho hiệu năng truy xuất.\n* Nhánh phân loại A{i}\n* Nhánh phân loại B{i}",
        })

    t0 = time.perf_counter()
    res = extract_taxonomy_and_types(synthetic_passages)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(res["section_systems"]) == 50
    assert len(res["bullet_types"]) == 100
    assert elapsed_ms < 50.0, f"Extraction took {elapsed_ms:.2f}ms, expected < 50ms"


"""
Unit tests for Formula and Example Patterns Recognition & Dataset (Wave 5).
Verifies:
1. FORMULA_EXAMPLE_DATASET integrity, coverage (35+ entries), and field schemas.
2. Accurate discrimination between mathematical formula sentences ('Ta có: zi = fi...',
   'Cho P = C = K = Z26', 'mod 26') and theoretical definitions.
3. Accurate detection of illustrated examples and walkthrough blocks.
4. extract_examples_and_formulas:
   - Extracts mathematical formulas while excluding pure definitions and classifications.
   - Extracts complete multi-line illustrated examples when requested ('cho ví dụ').
   - Handles topic-specific filtering and graceful fallbacks.
   - Deduplication, input robustness, and high performance (< 15ms).
5. Query helper functions: find_formulas_by_topic, find_examples_by_topic.
"""

import time
import pytest
from src.self_study.datasets.formula_and_example_patterns import (
    FORMULA_EXAMPLE_DATASET,
    extract_examples_and_formulas,
    extract_examples,
    extract_formulas,
    is_formula_sentence,
    is_example_sentence,
    is_definition_sentence,
    classify_sentence_type,
    find_entries_by_topic,
    find_formulas_by_topic,
    find_examples_by_topic,
    get_formula_example_dataset,
)

SAMPLE_SLIDE_PASSAGES = [
    {
        "passage_id": "SLIDE::P001",
        "section_title": "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG",
        "text": """=== Page 1 ===
1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:
- Xuất hiện đầu tiên trong ngành mã hóa.
- Các hệ mã này phần có từ trước năm 1970.
- Hệ mã khóa bí mật hoặc hệ mã khóa cổ điển là phương pháp sử dụng chung một khóa bí mật.
- Có 2 phương pháp phổ biến:
* Mã hóa thay thế.
* Mã hóa hoán vị.
""",
    },
    {
        "passage_id": "SLIDE::P010",
        "section_title": "3. PHÁ MÃ",
        "text": """=== Page 10 ===
3. PHÁ MÃ:
- An ninh vô điều kiện: Không thể phá mã dù có thời gian và tài nguyên vô hạn.
- An ninh tính toán: Chi phí để phá vỡ vượt quá giá trị của thông tin mã hóa.
- Các phương pháp thám mã phổ biến gồm tấn công vét cạn và phân tích thống kê tần suất.
""",
    },
    {
        "passage_id": "SLIDE::P030",
        "section_title": "9. HỆ MÃ HOÁ DÒNG",
        "text": """=== Page 30 ===
9. HỆ MÃ HOÁ DÒNG:
- Sinh ra một chuỗi khóa z = z1, z2, z3...
- Ta có: zi = fi (K, x1, ..., xi-1).
- Bản rõ được mã hóa từng bit hoặc từng byte bằng phép XOR với dòng khóa z.
- Ưu điểm: Tốc độ xử lý nhanh và độ trễ thấp trong truyền thông thời gian thực.
""",
    },
    {
        "passage_id": "SLIDE::P031",
        "section_title": "10. HỆ MÃ HOÁ AUTOKEY",
        "text": """=== Page 31 ===
10. HỆ MÃ HOÁ AUTOKEY:
- Cho P = C = K = Z26.
- Khóa k = k1, k2, ... ban đầu được mở rộng bằng chính các ký tự của bản rõ (Autokey).
- Công thức mã hóa: ci = (pi + ki) mod 26.
""",
    },
    {
        "passage_id": "SLIDE::P032",
        "section_title": "10. HỆ MÃ HOÁ AUTOKEY (VÍ DỤ)",
        "text": """=== Page 32 ===
Ví dụ:
Cho bản rõ P = 'WEBDESIGN', khóa khởi tạo k1 = 7.
Dòng khóa k mở rộng từ bản rõ: k = (7, 22, 4, 1, 3, 4, 18, 8, 6).
Quá trình mã hóa:
- Ký tự W (22) + 7 mod 26 = 3 ('D').
- Ký tự E (4) + 22 mod 26 = 0 ('A').
Bản mã thu được C = 'DLFGHVMOW'.
""",
    },
]


def test_dataset_size_and_schema():
    """Verifies dataset has at least 35 entries with valid schema."""
    dataset = get_formula_example_dataset()
    assert len(dataset) >= 35
    assert dataset is FORMULA_EXAMPLE_DATASET

    required_keys = {"id", "topic", "category", "text", "is_formula", "is_example", "is_definition"}
    for entry in dataset:
        assert required_keys.issubset(entry.keys()), f"Missing keys in {entry.get('id')}"
        assert entry["category"] in {"formula", "example", "definition"}
        assert isinstance(entry["text"], str) and len(entry["text"]) > 10
        assert isinstance(entry["is_formula"], bool)
        assert isinstance(entry["is_example"], bool)
        assert isinstance(entry["is_definition"], bool)


def test_dataset_coverage_and_contrast():
    """Verifies coverage of classical and modern crypto, plus theoretical contrast definitions."""
    topics = {e["topic"] for e in FORMULA_EXAMPLE_DATASET}
    expected_topics = {
        "caesar_cipher",
        "affine_cipher",
        "vigenere_cipher",
        "hill_cipher",
        "one_time_pad",
        "stream_cipher",
        "autokey_cipher",
        "playfair_cipher",
        "permutation_cipher",
        "substitution_cipher",
        "rsa_cipher",
        "diffie_hellman",
        "extended_euclidean",
        "fermat_little_theorem",
    }
    for et in expected_topics:
        assert et in topics, f"Missing topic in dataset: {et}"

    # Verify contrast definitions are strictly marked
    def_entries = [e for e in FORMULA_EXAMPLE_DATASET if e["category"] == "definition"]
    assert len(def_entries) >= 5
    for d in def_entries:
        assert d["is_definition"] is True
        assert d["is_formula"] is False
        assert d["is_example"] is False


def test_is_formula_sentence_discrimination():
    """Verifies that mathematical formula sentences are correctly identified while theoretical definitions are rejected."""
    # Positive formula test cases
    assert is_formula_sentence("Ta có: zi = fi (K, x1, ..., xi-1).") is True
    assert is_formula_sentence("Cho P = C = K = Z26.") is True
    assert is_formula_sentence("Công thức mã hóa: ci = (pi + ki) mod 26.") is True
    assert is_formula_sentence("Giải mã: pi = (ci - ki) mod 26.") is True
    assert is_formula_sentence("Hàm mã hóa: e(x) = (7x + 3) mod 26.") is True
    assert is_formula_sentence("c = m^e mod n và m = c^d mod n.") is True
    assert is_formula_sentence("ci = pi ⊕ zi") is True
    assert is_formula_sentence("Bản rõ được mã hóa từng bit bằng phép XOR với dòng khóa z.") is True
    assert is_formula_sentence("C = P * K mod 26 với gcd(det(K), 26) = 1.") is True

    # Negative non-formula test cases (pure definitions, classifications, background)
    assert is_formula_sentence("Mã hóa đối xứng là phương pháp sử dụng chung một khóa bí mật.") is False
    assert is_formula_sentence("Xuất hiện đầu tiên trong ngành mã hóa.") is False
    assert is_formula_sentence("Các hệ mã này phần có từ trước năm 1970.") is False
    assert is_formula_sentence("An ninh vô điều kiện: Không thể phá mã dù có thời gian và tài nguyên vô hạn.") is False
    assert is_formula_sentence("An ninh tính toán: Chi phí để phá vỡ vượt quá giá trị của thông tin mã hóa.") is False
    assert is_formula_sentence("* Mã hóa thay thế.") is False
    assert is_formula_sentence("* Mã hóa hoán vị.") is False
    assert is_formula_sentence("1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:") is False
    assert is_formula_sentence("=== Page 1 ===") is False
    assert is_formula_sentence("") is False
    assert is_formula_sentence(None) is False


def test_is_example_sentence_detection():
    """Verifies detection of illustrated examples."""
    assert is_example_sentence("Ví dụ: Cho bản rõ P = 'HELLO', khóa k = 3.") is True
    assert is_example_sentence("Ví dụ minh họa hệ mã Autokey:") is True
    assert is_example_sentence("Chẳng hạn, xét hệ mã Affine với a = 7, b = 3.") is True
    assert is_example_sentence("Giả sử Alice muốn gửi thông điệp M cho Bob.") is True
    assert is_example_sentence("Xét ví dụ với ma trận khóa cấp 2x2:") is True
    assert is_example_sentence("Bước 1: Chuyển ký tự thành số theo bảng Z26.") is True
    assert is_example_sentence("Bản rõ P = 'WEBDESIGN'") is True

    # Non-examples
    assert is_example_sentence("Mã hóa đối xứng là phương pháp mã hóa phổ biến.") is False
    assert is_example_sentence("Ta có: zi = fi (K, x1, ..., xi-1).") is False
    assert is_example_sentence("=== Page 32 ===") is False


def test_is_definition_sentence_detection():
    """Verifies detection of theoretical definitions."""
    assert is_definition_sentence("Mã hóa đối xứng là phương pháp mật mã dùng chung một khóa.") is True
    assert is_definition_sentence("An ninh vô điều kiện: Không thể phá mã dù có tài nguyên vô hạn.") is True
    assert is_definition_sentence("An ninh tính toán: Chi phí phá mã vượt quá giá trị thông tin.") is True
    assert is_definition_sentence("Phá mã vét cạn là kỹ thuật thử tất cả các khóa khả dĩ.") is True

    # Pure formulas should not be pure definitions
    assert is_definition_sentence("Cho P = C = K = Z26.") is False
    assert is_definition_sentence("ci = (pi + ki) mod 26.") is False


def test_classify_sentence_type():
    """Verifies multi-class sentence classification."""
    assert classify_sentence_type("Ví dụ: Cho P = 'HELLO'") == "example"
    assert classify_sentence_type("Ta có: zi = fi (K, x1, ..., xi-1).") == "formula"
    assert classify_sentence_type("Mã hóa đối xứng là phương pháp mật mã.") == "definition"
    assert classify_sentence_type("Xuất hiện đầu tiên trong ngành mã hóa.") == "general"


def test_extract_examples_and_formulas_full_passages():
    """Verifies extraction across full multi-section slide passages."""
    results = extract_examples_and_formulas(SAMPLE_SLIDE_PASSAGES)

    assert len(results) >= 4

    # Must contain the key formulas specified in the prompt
    assert any("Ta có: zi = fi" in r for r in results)
    assert any("Cho P = C = K = Z26" in r for r in results)
    assert any("mod 26" in r for r in results)
    assert any("ci = (pi + ki) mod 26" in r for r in results)

    # Must NOT contain pure theoretical definition or classification lines
    assert not any("Xuất hiện đầu tiên trong ngành mã hóa" in r for r in results)
    assert not any("* Mã hóa thay thế" in r for r in results)
    assert not any("* Mã hóa hoán vị" in r for r in results)
    assert not any("=== Page" in r for r in results)


def test_extract_examples_and_formulas_with_topic_filter():
    """Verifies topic-specific extraction for Autokey and Stream Cipher."""
    # Query Autokey
    autokey_results = extract_examples_and_formulas(SAMPLE_SLIDE_PASSAGES, topic="autokey")
    assert len(autokey_results) >= 2
    assert any("Cho P = C = K = Z26" in r for r in autokey_results)
    assert any("ci = (pi + ki) mod 26" in r for r in autokey_results)
    # Stream cipher formula should not be in autokey filtered results
    assert not any("zi = fi" in r for r in autokey_results)

    # Query Stream cipher ("mã hóa dòng")
    stream_results = extract_examples_and_formulas(SAMPLE_SLIDE_PASSAGES, topic="hệ mã hóa dòng")
    assert any("zi = fi" in r for r in stream_results)
    assert not any("Cho P = C = K = Z26" in r for r in stream_results)


def test_extract_complete_illustrated_example_block():
    """Verifies that multi-line illustrated examples are extracted completely ('cho ví dụ')."""
    # User asks for example
    example_results = extract_examples_and_formulas(SAMPLE_SLIDE_PASSAGES, topic="cho ví dụ")

    # Find the Autokey example block
    autokey_ex = next((r for r in example_results if "WEBDESIGN" in r), None)
    assert autokey_ex is not None

    # Must contain full steps and result, NOT truncated
    assert "WEBDESIGN" in autokey_ex
    assert "k1 = 7" in autokey_ex or "7" in autokey_ex
    assert "DLFGHVMOW" in autokey_ex
    assert "mod 26" in autokey_ex


def test_extract_examples_and_extract_formulas_helpers():
    """Verifies dedicated extract_formulas and extract_examples helpers."""
    formulas = extract_formulas(SAMPLE_SLIDE_PASSAGES)
    assert any("zi = fi" in f for f in formulas)
    assert any("Z26" in f for f in formulas)

    examples = extract_examples(SAMPLE_SLIDE_PASSAGES)
    assert any("WEBDESIGN" in ex for ex in examples)
    # Pure formula line should not be in examples
    assert not any(ex == "Cho P = C = K = Z26." for ex in examples)


def test_extract_examples_and_formulas_empty_and_robustness():
    """Verifies edge cases: empty input, string input, dict input, unknown topic."""
    assert extract_examples_and_formulas([]) == []
    assert extract_examples_and_formulas(None) == []

    # Single dict input
    single_dict = {"text": "Công thức: y = (11x + 7) mod 26.", "section_title": "Affine"}
    res_single = extract_examples_and_formulas(single_dict)
    assert len(res_single) == 1
    assert "mod 26" in res_single[0]

    # Unknown topic fallback
    fallback_res = extract_examples_and_formulas(SAMPLE_SLIDE_PASSAGES, topic="chu de khong ton tai xyz")
    assert len(fallback_res) > 0  # Graceful fallback to all passages


def test_dataset_topic_search_helpers():
    """Verifies find_entries_by_topic, find_formulas_by_topic, find_examples_by_topic."""
    affine_entries = find_entries_by_topic("affine")
    assert len(affine_entries) >= 2

    affine_formulas = find_formulas_by_topic("affine")
    assert len(affine_formulas) >= 1
    assert any(f["is_formula"] for f in affine_formulas)

    affine_examples = find_examples_by_topic("affine")
    assert len(affine_examples) >= 1
    assert any(e["is_example"] for e in affine_examples)

    rsa_formulas = find_formulas_by_topic("rsa")
    assert any("m^e mod n" in f["text"] for f in rsa_formulas)


def test_extraction_latency():
    """Verifies that extraction is ultra-fast (< 15ms) for strict offline performance."""
    t0 = time.perf_counter()
    for _ in range(50):
        _ = extract_examples_and_formulas(SAMPLE_SLIDE_PASSAGES)
    t_total_ms = (time.perf_counter() - t0) * 1000
    avg_ms = t_total_ms / 50

    assert avg_ms < 15.0, f"Average extraction took {avg_ms:.2f}ms, expected < 15ms"


def test_object_passage_compatibility():
    """Verifies compatibility with object instances (e.g. Passage dataclass) having .text and .section_title."""
    class PassageMock:
        def __init__(self, text, section_title):
            self.text = text
            self.section_title = section_title

    passages = [
        PassageMock(
            text="Công thức mã hóa Affine: e(x) = (7x + 3) mod 26.",
            section_title="5. HỆ MÃ AFFINE",
        ),
        PassageMock(
            text="Mã hóa đối xứng là phương pháp dùng chung một khóa.",
            section_title="1. GIỚI THIỆU",
        ),
    ]

    results = extract_examples_and_formulas(passages)
    assert len(results) == 1
    assert "e(x) = (7x + 3) mod 26" in results[0]


def test_include_section_prefix():
    """Verifies that include_section=True prepends the section title."""
    results = extract_examples_and_formulas(
        SAMPLE_SLIDE_PASSAGES,
        topic="autokey",
        include_section=True,
    )
    assert len(results) >= 2
    assert any(r.startswith("10. HỆ MÃ HOÁ AUTOKEY:") for r in results)


def test_accent_insensitive_topic_variants():
    """Verifies that topic queries work regardless of accents or casing."""
    res1 = extract_examples_and_formulas(SAMPLE_SLIDE_PASSAGES, topic="HỆ MÃ AUTOKEY")
    res2 = extract_examples_and_formulas(SAMPLE_SLIDE_PASSAGES, topic="he ma autokey")
    res3 = extract_examples_and_formulas(SAMPLE_SLIDE_PASSAGES, topic="cho ví dụ về hệ mã autokey")

    assert len(res1) >= 2
    assert len(res2) >= 2
    assert len(res3) >= 1
    assert any("WEBDESIGN" in r for r in res3)


def test_strict_offline_compliance():
    """Verifies that formula_and_example_patterns.py uses only pure Python standard library and zero external dependencies."""
    import inspect
    import src.self_study.datasets.formula_and_example_patterns as mod

    source = inspect.getsource(mod)
    # Must not import requests, openai, google, anthropic, torch, or external network packages
    prohibited = ["requests", "openai", "google.generativeai", "anthropic", "urllib3", "aiohttp", "boto3"]
    for p in prohibited:
        assert f"import {p}" not in source
        assert f"from {p}" not in source


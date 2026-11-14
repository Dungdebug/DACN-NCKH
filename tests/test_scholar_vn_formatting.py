"""
Unit tests for Vietnamese Academic Structuring & Block Recognition Engine.
Tests src/self_study/datasets/vietnamese_academic_formatting.py.
100% Offline, Pure Python stdlib, Pytest Suite.
"""

import pytest
from src.self_study.datasets.vietnamese_academic_formatting import (
    AcademicBlockType,
    CANONICAL_LABELS,
    parse_vietnamese_academic_blocks,
    extract_definitions,
    extract_theorems_and_properties,
    extract_examples,
    extract_exercises,
    extract_academic_hierarchy,
    search_academic_blocks,
    format_block_as_markdown,
    generate_qa_seeds_from_blocks,
    strip_accents,
    normalize_academic_vietnamese,
    has_mathematical_formula,
    validate_academic_dataset,
    get_academic_benchmark_dataset,
    get_academic_benchmark_by_id,
    ACADEMIC_BLOCK_BENCHMARK,
)


def test_academic_block_enum_and_canonical_labels():
    """Verify all 12 required academic block types and canonical labels."""
    required_types = [
        ("chuong", "Chương"),
        ("phan", "Phần"),
        ("muc", "Mục"),
        ("bai", "Bài"),
        ("dinh_nghia", "Định nghĩa"),
        ("dinh_ly", "Định lý"),
        ("tinh_chat", "Tính chất"),
        ("he_qua", "Hệ quả"),
        ("chung_minh", "Chứng minh"),
        ("nhan_xet", "Nhận xét"),
        ("chu_y", "Chú ý"),
        ("vi_du", "Ví dụ"),
    ]

    for enum_val, expected_label in required_types:
        # Check Enum attribute exists
        assert hasattr(AcademicBlockType, enum_val.upper())
        assert getattr(AcademicBlockType, enum_val.upper()).value == enum_val
        # Check canonical label mapping
        assert CANONICAL_LABELS[enum_val] == expected_label


def test_parse_vietnamese_academic_blocks_all_12_types():
    """
    Test parsing a single comprehensive university lecture excerpt
    containing all 12 required block types.
    """
    text = """
# CHƯƠNG 1: ĐẠI SỐ TUYẾN TÍNH VÀ CƠ SỞ DỮ LIỆU

Phần 1: Không gian vectơ và Ánh xạ tuyến tính
Trong phần này, sinh viên được trang bị các kiến thức nền tảng.

1.1. Khái niệm không gian vectơ
Định nghĩa 1.1 (Không gian vectơ):
Cho tập hợp V khác rỗng và một trường K (thường là trường số thực R hoặc số phức C).
V cùng với hai phép toán cộng (+) và nhân vô hướng (.) được gọi là không gian vectơ trên trường K nếu thỏa mãn 8 tiên đề sau:
1. u + v = v + u với mọi u, v thuộc V.
2. (u + v) + w = u + (v + w).
3. Tồn tại phần tử không 0 trong V sao cho u + 0 = u.

Ví dụ 1.1 (Không gian Euclide R^n):
Tập hợp các bộ n số thực R^n với phép cộng tọa độ và nhân tọa độ là một không gian vectơ trên R.

Định lý 1.2 (Sự tồn tại của cơ sở):
Mọi không gian vectơ hữu hạn chiều V != {0} đều có ít nhất một cơ sở.
Chứng minh:
Vì V hữu hạn chiều nên tồn tại hệ sinh hữu hạn S = {v1, ..., vn}.
Nếu S độc lập tuyến tính thì S là cơ sở.
Sau hữu hạn bước, ta thu được hệ độc lập tuyến tính và sinh ra V. ∎

Tính chất 1.3 (Duy nhất của tọa độ):
Cho B = {e1, e2, ..., en} là một cơ sở của không gian vectơ V.
Khi đó với mọi x thuộc V, tồn tại duy nhất bộ số (c1, c2, ..., cn) sao cho x = c1*e1 + ... + cn*en.

Hệ quả 1.4:
Số chiều của không gian vectơ dim(V) được xác định duy nhất bằng số phần tử của bất kỳ cơ sở nào của V.

Nhận xét 1:
Khái niệm số chiều đặc trưng cho bậc tự do độc lập của không gian.

Chú ý:
Không được nhầm lẫn giữa hệ sinh và cơ sở. Hệ sinh có thể phụ thuộc tuyến tính.

Bài tập 1:
Cho tập hợp W gồm các vectơ trong R^3 có tọa độ dạng (x, y, 0).
Chứng minh rằng W là không gian con của R^3.
"""

    blocks = parse_vietnamese_academic_blocks(text)
    assert len(blocks) >= 12, f"Expected at least 12 blocks, got {len(blocks)}"

    found_types = {b["block_type"] for b in blocks}
    expected_types = {
        "chuong", "phan", "muc", "dinh_nghia", "vi_du",
        "dinh_ly", "chung_minh", "tinh_chat", "he_qua",
        "nhan_xet", "chu_y", "bai"
    }
    assert expected_types.issubset(found_types), f"Missing types: {expected_types - found_types}"

    # Verify specific block attributes
    def_blocks = [b for b in blocks if b["block_type"] == "dinh_nghia"]
    assert len(def_blocks) >= 1
    d = def_blocks[0]
    assert d["number"] == "1.1"
    assert d["title"] == "Không gian vectơ"
    assert "tiên đề" in d["content"]
    assert d["metadata"]["is_definition"] is True
    assert d["metadata"]["has_formulas"] is True

    thm_blocks = [b for b in blocks if b["block_type"] == "dinh_ly"]
    assert len(thm_blocks) >= 1
    t = thm_blocks[0]
    assert t["number"] == "1.2"
    assert t["title"] == "Sự tồn tại của cơ sở"
    assert t["metadata"]["is_theorem"] is True

    prop_blocks = [b for b in blocks if b["block_type"] == "tinh_chat"]
    assert len(prop_blocks) >= 1
    p = prop_blocks[0]
    assert p["number"] == "1.3"
    assert p["metadata"]["is_property"] is True

    proof_blocks = [b for b in blocks if b["block_type"] == "chung_minh"]
    assert len(proof_blocks) >= 1
    pr = proof_blocks[0]
    assert pr["metadata"]["is_proof"] is True
    assert "∎" in pr["content"]


def test_spelling_variants_and_synonyms():
    """Verify alternative Vietnamese spellings and synonymous academic terms."""
    text = """
Định lí 2.1 (Ma trận vuông): Mọi ma trận vuông cấp n đều có định thức.
Mệnh đề 2.2: Ánh xạ tuyến tính bảo toàn phép cộng.
Bổ đề 2.3: Bổ đề Gauss về số nguyên tố.
Thí dụ 1: Xét hàm số f(x) = x^2.
Lưu ý: Luôn kiểm tra điều kiện mẫu số khác không.
Lời giải: Ta phân tích vế trái thành nhân tử.
"""

    blocks = parse_vietnamese_academic_blocks(text)
    types = [b["block_type"] for b in blocks]

    # 'Định lí' -> dinh_ly
    assert blocks[0]["block_type"] == "dinh_ly"
    assert blocks[0]["label"] == "Định lý"

    # 'Mệnh đề' -> dinh_ly
    assert blocks[1]["block_type"] == "dinh_ly"

    # 'Bổ đề' -> dinh_ly
    assert blocks[2]["block_type"] == "dinh_ly"

    # 'Thí dụ' -> vi_du
    assert blocks[3]["block_type"] == "vi_du"
    assert blocks[3]["label"] == "Ví dụ"

    # 'Lưu ý' -> chu_y
    assert blocks[4]["block_type"] == "chu_y"
    assert blocks[4]["label"] == "Chú ý"

    # 'Lời giải' -> chung_minh
    assert blocks[5]["block_type"] == "chung_minh"
    assert blocks[5]["label"] == "Chứng minh"


def test_markdown_and_slide_formatting_variations():
    """Test bold headers, bullet items, and slide titles."""
    text = """
* **Định nghĩa 1.1 (Nhóm Abelian):** Một nhóm (G, *) được gọi là nhóm giao hoán (Abelian) nếu a * b = b * a với mọi a, b thuộc G.
* **Tính chất 1:** Phần tử trung hòa e là duy nhất.
- **Hệ quả 2:** Phần tử nghịch đảo của mỗi a thuộc G là duy nhất.
> **Chú ý:** Không phải mọi vị nhóm (monoid) đều là nhóm.
"""

    blocks = parse_vietnamese_academic_blocks(text)
    assert len(blocks) == 4

    assert blocks[0]["block_type"] == "dinh_nghia"
    assert blocks[0]["title"] == "Nhóm Abelian"
    assert "nhóm giao hoán" in blocks[0]["content"]

    assert blocks[1]["block_type"] == "tinh_chat"
    assert "duy nhất" in blocks[1]["content"]

    assert blocks[2]["block_type"] == "he_qua"
    assert "nghịch đảo" in blocks[2]["content"]

    assert blocks[3]["block_type"] == "chu_y"
    assert "monoid" in blocks[3]["content"]


def test_academic_benchmark_validation_and_retrieval():
    """Verify validation and retrieval of multi-domain academic dataset."""
    assert validate_academic_dataset() is True

    dataset = get_academic_benchmark_dataset()
    assert len(dataset) >= 8

    domains = {item["domain"] for item in dataset}
    assert "mathematics" in domains
    assert "information_security" in domains
    assert "operating_systems" in domains
    assert "data_structures" in domains
    assert "calculus" in domains
    assert "computer_networks" in domains
    assert "artificial_intelligence" in domains
    assert "database_systems" in domains

    rsa = get_academic_benchmark_by_id("acad_crypto_rsa_002")
    assert rsa is not None
    assert "RSA" in rsa["title"]
    assert "mathematics" not in rsa["domain"]

    missing = get_academic_benchmark_by_id("non_existent_id")
    assert missing is None


def test_specialized_extractors():
    """Test extract_definitions, extract_theorems_and_properties, extract_examples, extract_exercises."""
    sample = get_academic_benchmark_by_id("acad_linear_algebra_001")
    assert sample is not None
    raw_text = sample["text"]

    defs = extract_definitions(raw_text)
    assert len(defs) >= 1
    assert all(b["block_type"] == "dinh_nghia" for b in defs)

    theorems_and_props = extract_theorems_and_properties(raw_text)
    assert len(theorems_and_props) >= 3
    valid_tp = {"dinh_ly", "tinh_chat", "he_qua"}
    assert all(b["block_type"] in valid_tp for b in theorems_and_props)

    examples = extract_examples(raw_text)
    assert len(examples) >= 1
    assert all(b["block_type"] == "vi_du" for b in examples)

    exercises = extract_exercises(raw_text)
    assert len(exercises) >= 1
    assert all(b["block_type"] == "bai" for b in exercises)


def test_hierarchy_extraction():
    """Test tree outline generation from flat blocks."""
    sample = get_academic_benchmark_by_id("acad_linear_algebra_001")
    tree = extract_academic_hierarchy(sample["text"])

    assert "chapters" in tree
    assert len(tree["chapters"]) == 1
    ch = tree["chapters"][0]
    assert ch["number"] == "1"
    assert "KHÔNG GIAN VECTƠ" in ch["title"]

    assert len(ch["sections"]) >= 2
    sec0 = ch["sections"][0]
    assert sec0["type"] == "phan"
    assert "không gian vectơ" in sec0["title"].lower()

    sec1 = ch["sections"][1]
    assert sec1["type"] == "muc"
    assert sec1["number"] == "1.1"
    assert len(sec1["blocks"]) >= 5


def test_search_academic_blocks():
    """Test diacritic-insensitive and type-filtered searching."""
    sample = get_academic_benchmark_by_id("acad_crypto_rsa_002")
    blocks = parse_vietnamese_academic_blocks(sample["text"])

    # Search with accents
    results_accents = search_academic_blocks(blocks, "Định lý Euler")
    assert len(results_accents) >= 1
    assert results_accents[0]["block_type"] == "dinh_ly"

    # Search without accents (fuzzy match)
    results_no_accents = search_academic_blocks(blocks, "dinh ly euler")
    assert len(results_no_accents) >= 1
    assert results_no_accents[0]["block_type"] == "dinh_ly"

    # Search filtered by block type
    def_results = search_academic_blocks(blocks, "RSA", block_types=["dinh_nghia"])
    assert len(def_results) == 1
    assert def_results[0]["block_type"] == "dinh_nghia"


def test_qa_seed_generation():
    """Test converting academic blocks into grounded self-study QA seeds."""
    sample = get_academic_benchmark_by_id("acad_os_deadlock_003")
    blocks = parse_vietnamese_academic_blocks(sample["text"])
    qa_seeds = generate_qa_seeds_from_blocks(blocks)

    assert len(qa_seeds) >= 3

    rule_types = {q["rule_type"] for q in qa_seeds}
    assert "definition" in rule_types
    assert "theorem" in rule_types
    assert "property" in rule_types

    def_qa = next(q for q in qa_seeds if q["rule_type"] == "definition")
    assert "Định nghĩa" in def_qa["question"] or "Bế tắc" in def_qa["question"]
    assert len(def_qa["answer"]) > 10


def test_format_block_as_markdown():
    """Test converting block dictionary to clean Markdown."""
    block = {
        "block_type": "dinh_nghia",
        "label": "Định nghĩa",
        "number": "3.1",
        "title": "Bế tắc (Deadlock)",
        "content": "Bế tắc là trạng thái trong đó hai hay nhiều tiến trình bị chặn vĩnh viễn.",
    }
    md = format_block_as_markdown(block)
    assert "**Định nghĩa 3.1 (Bế tắc (Deadlock)):**" in md
    assert "bị chặn vĩnh viễn" in md


def test_edge_cases_and_robustness():
    """Test blank inputs, Roman numerals vs math variables, and OCR artifacts."""
    # Empty inputs
    assert parse_vietnamese_academic_blocks("") == []
    assert parse_vietnamese_academic_blocks("   \n\n  \t ") == []

    # Plain text without headers
    plain = "Hệ thống máy tính bao gồm phần cứng và phần mềm."
    blocks_plain = parse_vietnamese_academic_blocks(plain)
    assert len(blocks_plain) == 1
    assert blocks_plain[0]["block_type"] == "noi_dung"
    assert "Hệ thống máy tính" in blocks_plain[0]["content"]

    # Roman numeral vs vector space variable V
    text_roman = """
I. Cơ sở lý thuyết
V là một không gian vectơ trên trường K.
II. Phương pháp giải quyết
C là trường số phức.
"""
    b_roman = parse_vietnamese_academic_blocks(text_roman)
    assert len(b_roman) == 2
    assert b_roman[0]["block_type"] == "muc"
    assert b_roman[0]["number"] == "I"
    assert "V là một không gian" in b_roman[0]["content"]

    assert b_roman[1]["block_type"] == "muc"
    assert b_roman[1]["number"] == "II"
    assert "C là trường số phức" in b_roman[1]["content"]

    # OCR and Unicode cleaning
    dirty_text = "Định\xad nghĩa 1.1:\u200b Cho\u2010hàm số liên tục."
    clean = normalize_academic_vietnamese(dirty_text)
    assert "\xad" not in clean
    assert "\u200b" not in clean
    assert "-" in clean

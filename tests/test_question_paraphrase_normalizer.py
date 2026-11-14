"""
Unit tests for QuestionParaphraseNormalizer (Wave 5 / Self-Study Datasets).
100% Offline, Pure Python stdlib.
"""

import time
import pytest
from src.self_study.datasets.question_paraphrase_normalizer import (
    normalize_student_question,
    QuestionParaphraseNormalizer,
    PARAPHRASE_DATASET,
    strip_accents,
)


def test_normalize_student_question_contract():
    """Verifies the exact return schema and types of normalize_student_question."""
    query = "mã hóa đối xứng là gì?"
    result = normalize_student_question(query)

    assert isinstance(result, dict)
    required_keys = [
        "cleaned_query",
        "core_keywords",
        "quoted_targets",
        "asks_definition",
        "asks_classification",
        "asks_example",
        "asks_section_detail",
    ]
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"

    assert isinstance(result["cleaned_query"], str)
    assert isinstance(result["core_keywords"], list)
    assert isinstance(result["quoted_targets"], list)
    assert isinstance(result["asks_definition"], bool)
    assert isinstance(result["asks_classification"], bool)
    assert isinstance(result["asks_example"], bool)
    assert isinstance(result["asks_section_detail"], bool)


def test_multi_facet_question():
    """
    Verifies multi-facet query:
    'mã hóa là gì có bao nhiêu loại mã hóa và cho ví dụ'
    Must activate definition, classification, and example flags simultaneously.
    """
    query = "mã hóa là gì có bao nhiêu loại mã hóa và cho ví dụ"
    result = normalize_student_question(query)

    assert result["asks_definition"] is True, "Must detect definition facet ('là gì')"
    assert result["asks_classification"] is True, "Must detect classification facet ('bao nhiêu loại')"
    assert result["asks_example"] is True, "Must detect example facet ('cho ví dụ')"
    assert result["asks_section_detail"] is False, "Should not be a section detail query"
    assert any("mã hóa" in kw.lower() or "mã hoá" in kw.lower() for kw in result["core_keywords"])
    assert result["quoted_targets"] == []
    assert result["primary_intent"] == "multi_facet"
    assert result["is_multi_facet"] is True


def test_section_detail_unquoted():
    """
    Verifies unquoted natural section detail question:
    'nói rõ mục phá mã'
    Must activate asks_section_detail and extract 'phá mã' as core keyword.
    """
    query = "nói rõ mục phá mã"
    result = normalize_student_question(query)

    assert result["asks_section_detail"] is True, "Must detect section detail ('mục phá mã')"
    assert result["asks_definition"] is False
    assert result["asks_classification"] is False
    assert result["asks_example"] is False
    assert any("phá mã" in kw.lower() for kw in result["core_keywords"])
    assert result["primary_intent"] == "section_detail"


def test_suggested_question_quoted():
    """
    Verifies suggested question click-through with quotes:
    'PHÁ MÃ'
    Must extract quoted target and identify section detail request.
    """
    query = "'PHÁ MÃ'"
    result = normalize_student_question(query)

    assert result["quoted_targets"] == ["PHÁ MÃ"]
    assert result["cleaned_query"] == "PHÁ MÃ"
    assert result["asks_section_detail"] is True
    assert result["asks_definition"] is False
    assert result["asks_classification"] is False
    assert result["asks_example"] is False
    assert "PHÁ MÃ" in result["core_keywords"]


def test_various_quote_types_and_numbered_headings():
    """Verifies double quotes, smart curly quotes, and numbered heading sections."""
    # Double quotes
    res_dq = normalize_student_question('chi tiết về "HỆ MÃ DỊCH VÒNG"')
    assert "HỆ MÃ DỊCH VÒNG" in res_dq["quoted_targets"]
    assert res_dq["asks_section_detail"] is True

    # Smart curly quotes
    res_curly = normalize_student_question("nội dung của “PHÁ MÃ”")
    assert "PHÁ MÃ" in res_curly["quoted_targets"]
    assert res_curly["asks_section_detail"] is True

    # Quoted numbered heading
    res_num_q = normalize_student_question("'3. PHÁ MÃ'")
    assert "3. PHÁ MÃ" in res_num_q["quoted_targets"]
    assert res_num_q["asks_section_detail"] is True

    # Plain numbered heading without quotes
    res_num_plain = normalize_student_question("3. PHÁ MÃ")
    assert res_num_plain["asks_section_detail"] is True
    assert any("PHÁ MÃ" in kw for kw in res_num_plain["core_keywords"])

    # Multiple quotes
    res_multi_q = normalize_student_question("so sánh 'MÃ HÓA THAY THẾ' và 'MÃ HÓA HOÁN VỊ'")
    assert len(res_multi_q["quoted_targets"]) == 2
    assert "MÃ HÓA THAY THẾ" in res_multi_q["quoted_targets"]
    assert "MÃ HÓA HOÁN VỊ" in res_multi_q["quoted_targets"]


def test_definition_queries():
    """Verifies pure definition questions."""
    queries = [
        "thế nào là hệ mã khóa đối xứng?",
        "khái niệm về mật mã cổ điển",
        "mã hóa đối xứng nghĩa là gì",
        "định nghĩa bản rõ và bản mã",
        "cho biết khái niệm hệ mã affine",
    ]
    for q in queries:
        res = normalize_student_question(q)
        assert res["asks_definition"] is True, f"Failed definition detection on: {q}"
        assert res["asks_example"] is False
        assert res["asks_classification"] is False


def test_classification_queries():
    """Verifies pure classification questions."""
    queries = [
        "có bao nhiêu phương pháp mã hóa",
        "các loại mã hóa gồm những gì",
        "có mấy loại mật mã",
        "liệt kê các hệ mã đối xứng",
        "kể tên các phương pháp phân loại",
    ]
    for q in queries:
        res = normalize_student_question(q)
        assert res["asks_classification"] is True, f"Failed classification detection on: {q}"
        assert res["asks_definition"] is False
        assert res["asks_example"] is False


def test_example_queries():
    """Verifies pure example questions."""
    queries = [
        "cho ví dụ về hệ mã caesar",
        "minh họa cách giải mã với k=3",
        "lấy ví dụ minh họa cụ thể",
        "cho một ví dụ về mã affine",
    ]
    for q in queries:
        res = normalize_student_question(q)
        assert res["asks_example"] is True, f"Failed example detection on: {q}"
        assert res["asks_definition"] is False
        assert res["asks_classification"] is False


def test_paraphrase_benchmark_dataset():
    """Validates the built-in golden benchmark dataset."""
    assert len(PARAPHRASE_DATASET) >= 10
    normalizer = QuestionParaphraseNormalizer()

    for item in PARAPHRASE_DATASET:
        query = item["query"]
        expected_facets = item["expected_facets"]
        expected_keywords = item["expected_keywords_subset"]
        expected_quoted = item["expected_quoted"]

        res = normalizer.normalize(query)

        for facet_key, expected_val in expected_facets.items():
            actual_val = res[facet_key]
            assert actual_val == expected_val, (
                f"Query: '{query}' -> {facet_key} expected {expected_val}, got {actual_val}"
            )

        for kw in expected_keywords:
            assert any(kw.lower() in actual_kw.lower() for actual_kw in res["core_keywords"]), (
                f"Query: '{query}' -> Expected keyword '{kw}' not found in {res['core_keywords']}"
            )

        assert res["quoted_targets"] == expected_quoted, (
            f"Query: '{query}' -> Quoted targets expected {expected_quoted}, got {res['quoted_targets']}"
        )


def test_edge_cases_and_robustness():
    """Verifies behavior on empty, whitespace, and special characters."""
    # Empty string
    res_empty = normalize_student_question("")
    assert res_empty["cleaned_query"] == ""
    assert res_empty["core_keywords"] == []
    assert res_empty["quoted_targets"] == []
    assert res_empty["asks_definition"] is False
    assert res_empty["asks_classification"] is False
    assert res_empty["asks_example"] is False
    assert res_empty["asks_section_detail"] is False

    # Whitespace and punctuation
    res_punct = normalize_student_question("   ???   ")
    assert res_punct["cleaned_query"] == ""
    assert res_punct["core_keywords"] == []

    # Strip accents utility
    assert strip_accents("Mã hóa đối xứng") == "Ma hoa doi xung"
    assert strip_accents("Định nghĩa") == "Dinh nghia"


def test_colloquial_student_phrasings():
    """Verifies typical student conversational noise, politeness particles, and greetings."""
    # "cho em hỏi... là gì thế ạ"
    res1 = normalize_student_question("cho em hỏi mã hóa đối xứng là cái gì thế ạ?")
    assert res1["asks_definition"] is True
    assert any("mã hóa đối xứng" in kw.lower() for kw in res1["core_keywords"])

    # "thầy ơi cho em xin một ví dụ..."
    res2 = normalize_student_question("thầy ơi cho em xin một ví dụ về mã Affine với ạ")
    assert res2["asks_example"] is True
    assert any("affine" in kw.lower() for kw in res2["core_keywords"])

    # "vậy mục phá mã có những nội dung gì..."
    res3 = normalize_student_question("vậy mục phá mã có những nội dung gì ạ")
    assert res3["asks_section_detail"] is True
    assert any("phá mã" in kw.lower() for kw in res3["core_keywords"])


def test_module_level_import():
    """Verifies clean imports from question_paraphrase_normalizer module."""
    from src.self_study.datasets.question_paraphrase_normalizer import (
        normalize_student_question as norm_fn,
        QuestionParaphraseNormalizer as NormClass,
    )
    assert callable(norm_fn)
    assert isinstance(NormClass(), QuestionParaphraseNormalizer)


def test_accent_insensitive_and_mixed_cases():
    """Verifies that queries typed without accents or in mixed cases are properly understood."""
    res_no_accent = normalize_student_question("ma hoa la gi co bao nhieu loai va cho vi du")
    assert res_no_accent["asks_definition"] is True
    assert res_no_accent["asks_classification"] is True
    assert res_no_accent["asks_example"] is True

    res_sec_no_accent = normalize_student_question("noi ro muc pha ma")
    assert res_sec_no_accent["asks_section_detail"] is True


def test_speed_and_zero_latency():
    """Ensures lightning-fast performance (< 0.2ms per normalization)."""
    normalizer = QuestionParaphraseNormalizer()
    sample_queries = [
        "mã hóa là gì có bao nhiêu loại mã hóa và cho ví dụ",
        "nói rõ mục phá mã",
        "'PHÁ MÃ'",
        "thế nào là hệ mã khóa đối xứng?",
        "cho ví dụ minh họa về mã dịch vòng với k=3",
    ]

    start = time.perf_counter()
    iterations = 1000
    for i in range(iterations):
        q = sample_queries[i % len(sample_queries)]
        normalizer.normalize(q)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    assert avg_ms < 0.5, f"Normalization too slow: {avg_ms:.3f}ms per query (expected < 0.5ms)"

"""
Unit Tests for Cryptography and Information Security Semantic Patterns.
Tests bidirectional synonym mapping, domain query expansion, and few-shot QA datasets.
100% Offline, Pure Python stdlib, Zero External Dependencies.
"""

import pytest
from src.self_study.datasets.cryptography_and_security_patterns import (
    strip_accents,
    normalize_term,
    get_crypto_synonyms,
    expand_domain_query_terms,
    CryptoThesaurusRegistry,
    CRYPTO_SYNONYM_CLUSTERS,
    CRYPTO_FEW_SHOT_QA_EXAMPLES,
)


class TestCryptoTextNormalization:
    """Tests for Vietnamese text normalization and tone mark removal."""

    def test_strip_accents_vietnamese(self):
        assert strip_accents("Mã hóa đối xứng") == "Ma hoa doi xung"
        assert strip_accents("độn một lần") == "don mot lan"
        assert strip_accents("ĐƠN BẢNG") == "DON BANG"
        assert strip_accents("thay thế và hoán vị") == "thay the va hoan vi"

    def test_strip_accents_empty_and_ascii(self):
        assert strip_accents("") == ""
        assert strip_accents("stream cipher") == "stream cipher"
        assert strip_accents("12345!@#") == "12345!@#"

    def test_normalize_term(self):
        assert normalize_term("   Mã  Hóa   ĐỐI XỨNG   ") == "mã hóa đối xứng"
        assert normalize_term("\n\tstream   cipher\t") == "stream cipher"


class TestBidirectionalSynonymMapping:
    """Verifies bidirectional synonym mapping across core cryptographic concepts."""

    def test_symmetric_encryption_cluster(self):
        # Forward and backward mapping between symmetric encryption terms
        terms = ["mã hóa đối xứng", "hệ mã khóa đối xứng", "khóa bí mật", "mã cổ điển"]
        for term in terms:
            synonyms = get_crypto_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            # Must contain the others in the cluster
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_cryptanalysis_mapping(self):
        # Mapping between 'phá mã' <-> 'thám mã' <-> 'cryptanalysis'
        syns_pha_ma = get_crypto_synonyms("phá mã")
        syns_tham_ma = get_crypto_synonyms("thám mã")
        syns_cryptanalysis = get_crypto_synonyms("cryptanalysis")

        assert "thám mã" in syns_pha_ma
        assert "phá mã" in syns_tham_ma
        assert "cryptanalysis" in syns_pha_ma
        assert "phá mã" in syns_cryptanalysis

    def test_stream_cipher_mapping(self):
        # 'mã hóa dòng' <-> 'stream cipher'
        syns_dong = get_crypto_synonyms("mã hóa dòng")
        syns_stream = get_crypto_synonyms("stream cipher")

        assert "stream cipher" in syns_dong
        assert "mã hóa dòng" in syns_stream

    def test_one_time_pad_mapping(self):
        # 'độn một lần' <-> 'one-time pad' <-> 'vernam'
        syns_don = get_crypto_synonyms("độn một lần")
        syns_otp = get_crypto_synonyms("one-time pad")

        assert "one-time pad" in syns_don
        assert "độn một lần" in syns_otp
        assert "vernam" in syns_don

    def test_substitution_and_transposition_mapping(self):
        # 'thay thế' <-> 'substitution'
        syns_sub = get_crypto_synonyms("thay thế")
        assert "substitution" in syns_sub
        assert "mã hóa thay thế" in syns_sub

        # 'hoán vị' <-> 'transposition' <-> 'permutation'
        syns_trans = get_crypto_synonyms("hoán vị")
        assert "transposition" in syns_trans
        assert "permutation" in syns_trans
        assert "mã hóa hoán vị" in syns_trans

    def test_unaccented_lookup(self):
        # Searching with unaccented term returns accented canonical synonyms
        syns = get_crypto_synonyms("ma hoa doi xung")
        assert "mã hóa đối xứng" in syns
        assert "khóa bí mật" in syns

        syns_pha = get_crypto_synonyms("pha ma")
        assert "thám mã" in syns_pha

    def test_unknown_term_returns_empty(self):
        assert get_crypto_synonyms("từ_khóa_không_tồn_tại_123") == []


class TestExpandDomainQueryTerms:
    """Verifies query expansion for student questions and conversational inputs."""

    def test_single_term_expansion(self):
        expanded = expand_domain_query_terms("mã hóa đối xứng")
        assert "mã hóa đối xứng" in expanded
        assert "hệ mã khóa đối xứng" in expanded
        assert "khóa bí mật" in expanded
        assert "mã cổ điển" in expanded

    def test_multi_intent_query_expansion(self):
        query = "Cho tôi biết phá mã trong hệ mã khóa đối xứng là gì?"
        expanded = expand_domain_query_terms(query)

        # Must capture both cryptanalysis and symmetric clusters
        assert any("phá mã" in term or "thám mã" in term for term in expanded)
        assert any("đối xứng" in term or "khóa bí mật" in term for term in expanded)

    def test_stream_and_otp_joint_query(self):
        query = "Phân biệt hệ mã hóa độn một lần và stream cipher"
        expanded = expand_domain_query_terms(query)

        assert "độn một lần" in expanded or "one-time pad" in expanded
        assert "stream cipher" in expanded or "mã hóa dòng" in expanded

    def test_substitution_and_transposition_query(self):
        query = "Mã hóa thay thế và hoán vị khác nhau chỗ nào?"
        expanded = expand_domain_query_terms(query)

        assert any("thay thế" in t or "substitution" in t for t in expanded)
        assert any("hoán vị" in t or "transposition" in t for t in expanded)

    def test_unaccented_natural_queries(self):
        queries = [
            ("ma hoa doi xung la gi", "mã hóa đối xứng"),
            ("pha ma gom nhung gi", "thám mã"),
            ("don mot lan hoat dong nhu the nao", "độn một lần"),
            ("cac phep thay the va hoan vi", "hoán vị"),
        ]
        for q, expected_canonical in queries:
            res = expand_domain_query_terms(q)
            assert any(expected_canonical in item for item in res), (
                f"Query '{q}' failed to expand into '{expected_canonical}'"
            )

    def test_quoted_and_punctuated_queries(self):
        query = "Yêu cầu chi tiết liên quan đến 'PHÁ MÃ' gồm những gì?"
        expanded = expand_domain_query_terms(query)
        assert "phá mã" in expanded or "thám mã" in expanded

    def test_no_duplicate_terms_in_expansion(self):
        expanded = expand_domain_query_terms(
            "mã hóa đối xứng hệ mã khóa đối xứng khóa bí mật mã cổ điển"
        )
        assert len(expanded) == len(set(normalize_term(t) for t in expanded))

    def test_empty_and_irrelevant_queries(self):
        assert expand_domain_query_terms("") == []
        assert expand_domain_query_terms("     ") == []
        assert expand_domain_query_terms("hôm nay thời tiết đẹp quá không mưa") == []


class TestCryptoFewShotQaDataset:
    """Verifies the structure, contents, and pedagogical quality of few-shot QA examples."""

    def test_dataset_size_and_diversity(self):
        assert len(CRYPTO_FEW_SHOT_QA_EXAMPLES) >= 10

    def test_required_fields_presence(self):
        required_fields = {
            "id",
            "topic",
            "category",
            "canonical_terms",
            "question",
            "intent",
            "expected_action",
            "grounded_answer",
        }
        for item in CRYPTO_FEW_SHOT_QA_EXAMPLES:
            missing = required_fields - set(item.keys())
            assert not missing, f"Item {item.get('id')} is missing fields: {missing}"
            assert len(item["canonical_terms"]) > 0
            assert len(item["grounded_answer"]) > 100, (
                f"Grounded answer in {item['id']} is too brief or incomplete"
            )

    def test_coverage_of_slide_key_topics(self):
        topics_covered = {item["topic"] for item in CRYPTO_FEW_SHOT_QA_EXAMPLES}
        assert any("đối xứng" in t for t in topics_covered)
        assert any("Phá mã" in t for t in topics_covered)
        assert any("độn một lần" in t for t in topics_covered)
        assert any("dòng" in t for t in topics_covered)
        assert any("Playfair" in t for t in topics_covered)
        assert any("Affine" in t for t in topics_covered)
        assert any("hoán vị" in t or "thay thế" in t for t in topics_covered)


class TestCustomRegistryExtensibility:
    """Tests custom isolated registry instantiation according to Clean Code standards."""

    def test_custom_registry_override(self):
        custom_clusters = {
            "custom_crypto": ["hệ mã quantum", "mã hóa lượng tử", "post-quantum"],
        }
        registry = CryptoThesaurusRegistry(clusters=custom_clusters)
        syns = registry.get_synonyms("mã hóa lượng tử")
        assert "hệ mã quantum" in syns
        assert "post-quantum" in syns
        assert registry.get_synonyms("mã hóa đối xứng") == []

"""
Unit Tests for Software Engineering Semantic Patterns and Datasets.
Tests bidirectional synonym mapping, domain query expansion, and few-shot QA datasets
for SDLC, Agile/Scrum, Waterfall, SOLID, Design Patterns, Testing, and CI/CD.
100% Offline, Pure Python stdlib, Zero External Dependencies.
"""

import pytest
from src.self_study.datasets.software_engineering_patterns import (
    strip_accents,
    normalize_term,
    get_se_synonyms,
    expand_se_query,
    find_matching_se_topics,
    SEThesaurusRegistry,
    SE_SYNONYM_CLUSTERS,
    SE_FEW_SHOT_EXAMPLES,
)


class TestSETextNormalization:
    """Verifies Vietnamese tone-mark normalization and term cleaning."""

    def test_strip_accents_vietnamese(self):
        assert strip_accents("Vòng đời phát triển phần mềm") == "Vong doi phat trien phan mem"
        assert strip_accents("Mô hình thác nước") == "Mo hinh thac nuoc"
        assert strip_accents("Nguyên lý đơn trách nhiệm") == "Nguyen ly don trach nhiem"
        assert strip_accents("ĐƠN THỂ") == "DON THE"

    def test_strip_accents_empty_and_ascii(self):
        assert strip_accents("") == ""
        assert strip_accents("design patterns") == "design patterns"
        assert strip_accents("12345!@#$") == "12345!@#$"

    def test_normalize_term(self):
        assert normalize_term("   VÒNG  ĐỜI   PHÁT TRIỂN   ") == "vòng đời phát triển"
        assert normalize_term("\n\tagile   scrum\t") == "agile scrum"


class TestSEBidirectionalSynonyms:
    """Tests bidirectional synonym mapping across core SE concepts."""

    def test_sdlc_synonyms(self):
        syns_sdlc = get_se_synonyms("sdlc")
        syns_vn = get_se_synonyms("vòng đời phát triển phần mềm")

        assert len(syns_sdlc) >= 5
        assert "vòng đời phát triển phần mềm" in syns_sdlc
        assert "software development life cycle" in syns_sdlc
        assert "sdlc" in syns_vn

    def test_waterfall_synonyms(self):
        syns_waterfall = get_se_synonyms("waterfall")
        syns_thac_nuoc = get_se_synonyms("mô hình thác nước")

        assert "mô hình thác nước" in syns_waterfall
        assert "waterfall model" in syns_thac_nuoc
        assert "tuần tự tuyến tính" in syns_waterfall

    def test_agile_and_scrum_synonyms(self):
        syns_agile = get_se_synonyms("agile")
        syns_scrum = get_se_synonyms("scrum")

        assert "phương pháp agile" in syns_agile
        assert "agile manifesto" in syns_agile
        assert "sprint" in syns_scrum
        assert "product backlog" in syns_scrum
        assert "scrum master" in syns_scrum

    def test_solid_principles_synonyms(self):
        # General SOLID
        syns_solid = get_se_synonyms("solid")
        assert "nguyên lý solid" in syns_solid
        assert "solid principles" in syns_solid

        # S - SRP
        syns_srp = get_se_synonyms("srp")
        assert "single responsibility" in syns_srp
        assert "nguyên lý đơn trách nhiệm" in syns_srp

        # O - OCP
        syns_ocp = get_se_synonyms("ocp")
        assert "open closed principle" in syns_ocp
        assert "nguyên lý đóng mở" in syns_ocp

        # L - LSP
        syns_lsp = get_se_synonyms("lsp")
        assert "liskov substitution" in syns_lsp
        assert "nguyên lý thay thế liskov" in syns_lsp

        # I - ISP
        syns_isp = get_se_synonyms("isp")
        assert "interface segregation" in syns_isp
        assert "nguyên lý phân tách giao diện" in syns_isp

        # D - DIP
        syns_dip = get_se_synonyms("dip")
        assert "dependency inversion" in syns_dip
        assert "nguyên lý đảo ngược phụ thuộc" in syns_dip
        assert "dependency injection" in syns_dip

    def test_design_patterns_synonyms(self):
        # Singleton
        syns_singleton = get_se_synonyms("singleton")
        assert "mẫu singleton" in syns_singleton
        assert "thể hiện duy nhất" in syns_singleton
        assert "private constructor" in syns_singleton

        # Factory
        syns_factory = get_se_synonyms("factory")
        assert "factory method" in syns_factory
        assert "abstract factory" in syns_factory
        assert "mẫu nhà máy" in syns_factory

        # Observer
        syns_observer = get_se_synonyms("observer")
        assert "mẫu quan sát" in syns_observer
        assert "publish subscribe" in syns_observer
        assert "pub sub" in syns_observer

    def test_testing_and_ci_cd_synonyms(self):
        # Unit test
        syns_unit = get_se_synonyms("unit test")
        assert "kiểm thử đơn vị" in syns_unit
        assert "mock" in syns_unit

        # Integration test
        syns_integration = get_se_synonyms("integration test")
        assert "kiểm thử tích hợp" in syns_integration

        # TDD
        syns_tdd = get_se_synonyms("tdd")
        assert "test-driven development" in syns_tdd
        assert "red green refactor" in syns_tdd
        assert "viết test trước khi viết code" in syns_tdd

        # CI/CD
        syns_cicd = get_se_synonyms("ci/cd")
        assert "continuous integration continuous deployment" in syns_cicd
        assert "tích hợp liên tục và triển khai liên tục" in syns_cicd

        # CI
        syns_ci = get_se_synonyms("ci")
        assert "continuous integration" in syns_ci
        assert "tích hợp liên tục" in syns_ci

    def test_unaccented_tone_insensitive_lookup(self):
        # Tone-insensitive lookup matches Vietnamese canonical terms
        syns_vong_doi = get_se_synonyms("vong doi phat trien phan mem")
        assert "sdlc" in syns_vong_doi

        syns_thac = get_se_synonyms("mo hinh thac nuoc")
        assert "waterfall" in syns_thac

        syns_don_trach = get_se_synonyms("nguyen ly don trach nhiem")
        assert "single responsibility" in syns_don_trach

        syns_kiem_thu = get_se_synonyms("kiem thu don vi")
        assert "unit test" in syns_kiem_thu

    def test_unknown_term_returns_empty(self):
        assert get_se_synonyms("unknown_term_xyz_123") == []
        assert get_se_synonyms("") == []


class TestSEQueryExpansion:
    """Tests query expansion for conversational search and retrieval."""

    def test_expand_query_single_concept(self):
        expanded = expand_se_query("Tìm hiểu về mô hình thác nước")
        assert "mô hình thác nước" in expanded
        assert "waterfall" in expanded
        assert "waterfall model" in expanded

    def test_expand_query_multi_concept(self):
        expanded = expand_se_query("So sánh Scrum và Waterfall trong SDLC")
        # All clusters should be represented
        assert any("scrum" in term for term in expanded)
        assert any("waterfall" in term for term in expanded)
        assert any("sdlc" in term or "vòng đời" in term for term in expanded)

    def test_expand_query_solid_design_pattern(self):
        expanded = expand_se_query("Áp dụng Singleton và Factory theo nguyên lý SOLID")
        assert any("singleton" in term for term in expanded)
        assert any("factory" in term for term in expanded)
        assert any("solid" in term for term in expanded)

    def test_expand_query_empty_or_whitespace(self):
        assert expand_se_query("") == []
        assert expand_se_query("   \n\t  ") == []

    def test_identify_se_concepts(self):
        text = "Chúng tôi triển khai quy trình CI/CD kết hợp kiểm thử đơn vị Unit Test và TDD."
        topics = find_matching_se_topics(text)
        assert "ci_cd" in topics or "continuous_integration" in topics
        assert "unit_testing" in topics
        assert "tdd" in topics


class TestSEThesaurusRegistryCustom:
    """Verifies SEThesaurusRegistry with custom or injected synonym clusters."""

    def test_custom_registry(self):
        custom_clusters = {
            "custom_arch": ["clean architecture", "kiến trúc sạch", "onion architecture"],
            "custom_test": ["mutation testing", "kiểm thử đột biến"],
        }
        reg = SEThesaurusRegistry(clusters=custom_clusters)
        assert "kiến trúc sạch" in reg.get_synonyms("clean architecture")
        assert "mutation testing" in reg.get_synonyms("kiem thu dot bien")
        assert reg.get_cluster_name("onion architecture") == "custom_arch"
        assert reg.get_cluster_name("non_existent") is None


class TestSEFewShotExamples:
    """Verifies comprehensive coverage and schema integrity of SE_FEW_SHOT_EXAMPLES."""

    def test_dataset_size_and_schema(self):
        assert len(SE_FEW_SHOT_EXAMPLES) >= 12, "Expected at least 12 rich QA examples"

        seen_ids = set()
        required_keys = {
            "id",
            "topic",
            "category",
            "canonical_terms",
            "question",
            "intent",
            "expected_action",
            "grounded_answer",
            "bullet_types",
            "example",
        }

        for qa in SE_FEW_SHOT_EXAMPLES:
            assert required_keys.issubset(qa.keys()), f"Missing keys in {qa.get('id')}"
            assert qa["id"] not in seen_ids, f"Duplicate QA id: {qa['id']}"
            seen_ids.add(qa["id"])

            # Verify quality and completeness of fields
            assert len(qa["topic"]) > 0
            assert len(qa["canonical_terms"]) >= 2
            assert qa["question"].endswith("?"), f"Question should end with '?' in {qa['id']}"
            assert len(qa["grounded_answer"]) > 100, f"Grounded answer too short in {qa['id']}"
            assert len(qa["bullet_types"]) >= 2, f"Bullet types too sparse in {qa['id']}"
            assert len(qa["example"]) > 0

    def test_coverage_of_required_domains(self):
        categories = {qa["category"] for qa in SE_FEW_SHOT_EXAMPLES}
        assert "sdlc" in categories
        assert "agile_scrum" in categories
        assert "solid_principles" in categories
        assert "design_patterns" in categories
        assert "software_testing" in categories
        assert "ci_cd_devops" in categories

        # Verify specific required topics
        all_text = " ".join(
            qa["topic"] + " " + qa["question"] + " " + qa["grounded_answer"]
            for qa in SE_FEW_SHOT_EXAMPLES
        ).lower()

        assert "sdlc" in all_text or "vòng đời phát triển phần mềm" in all_text
        assert "waterfall" in all_text or "thác nước" in all_text
        assert "scrum" in all_text or "agile" in all_text
        assert "solid" in all_text
        assert "singleton" in all_text
        assert "factory" in all_text
        assert "observer" in all_text
        assert "unit test" in all_text
        assert "integration test" in all_text
        assert "tdd" in all_text or "red-green-refactor" in all_text
        assert "ci/cd" in all_text or "continuous integration" in all_text

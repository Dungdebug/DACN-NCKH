"""
Unit Tests for Artificial Intelligence, Machine Learning, RAG and Nested Learning Semantic Patterns.
Tests bidirectional synonym mapping, domain query expansion, and few-shot QA datasets.
100% Offline, Pure Python stdlib, Zero External Dependencies.
"""

import pytest
from src.self_study.datasets.ai_and_machine_learning_patterns import (
    strip_accents,
    normalize_term,
    get_ai_ml_synonyms,
    expand_ai_ml_query_terms,
    AIMLThesaurusRegistry,
    AI_ML_SYNONYM_CLUSTERS,
    AI_ML_FEW_SHOT_EXAMPLES,
)


class TestAIMLTextNormalization:
    """Tests for Vietnamese text normalization and tone mark removal."""

    def test_strip_accents_vietnamese(self):
        assert strip_accents("Học có giám sát") == "Hoc co giam sat"
        assert strip_accents("quá khớp và hạ độ dốc") == "qua khop va ha do doc"
        assert strip_accents("BỘ NHỚ LIÊN TỤC") == "BO NHO LIEN TUC"
        assert strip_accents("học tăng cường và học củng cố") == "hoc tang cuong va hoc cung co"
        assert strip_accents("độ chính xác và độ chệch") == "do chinh xac va do chech"

    def test_strip_accents_empty_and_ascii(self):
        assert strip_accents("") == ""
        assert strip_accents("transformer attention") == "transformer attention"
        assert strip_accents("12345!@#") == "12345!@#"

    def test_normalize_term(self):
        assert normalize_term("   Học  CÓ   GIÁM SÁT   ") == "học có giám sát"
        assert normalize_term("\n\tgradient   descent\t") == "gradient descent"
        assert normalize_term("  sa-cms  \n") == "sa-cms"


class TestBidirectionalAIMLSynonymMapping:
    """Verifies bidirectional synonym mapping across core AI/ML/RAG concepts."""

    def test_supervised_learning_cluster(self):
        terms = ["học có giám sát", "supervised learning", "phân lớp và hồi quy", "học giám sát"]
        for term in terms:
            synonyms = get_ai_ml_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_unsupervised_learning_cluster(self):
        terms = ["học không giám sát", "unsupervised learning", "gom cụm", "clustering"]
        for term in terms:
            synonyms = get_ai_ml_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_reinforcement_learning_cluster(self):
        terms = ["học tăng cường", "reinforcement learning", "q-learning", "policy gradient"]
        for term in terms:
            synonyms = get_ai_ml_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_overfitting_and_regularization_cluster(self):
        terms = ["quá khớp", "overfitting", "regularization", "chính quy hóa"]
        for term in terms:
            synonyms = get_ai_ml_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_underfitting_cluster(self):
        terms = ["dưới khớp", "underfitting", "high bias", "độ chệch cao"]
        for term in terms:
            synonyms = get_ai_ml_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_gradient_descent_cluster(self):
        terms = ["hạ độ dốc", "gradient descent", "sgd", "adam", "learning rate"]
        for term in terms:
            synonyms = get_ai_ml_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_attention_and_transformer_cluster(self):
        terms = ["cơ chế chú ý", "self-attention", "transformer", "multi-head attention"]
        for term in terms:
            synonyms = get_ai_ml_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_rag_cluster(self):
        terms = ["rag", "retrieval-augmented generation", "cơ sở dữ liệu vector", "embedding"]
        for term in terms:
            synonyms = get_ai_ml_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_bm25_cluster(self):
        terms = ["bm25", "okapi bm25", "lexical retrieval", "tf-idf"]
        for term in terms:
            synonyms = get_ai_ml_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_sa_cms_cluster(self):
        terms = ["sa-cms", "sparse-aware chunk management system", "sparse chunking", "semantic chunking"]
        for term in terms:
            synonyms = get_ai_ml_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_continuum_memory_and_nested_learning_cluster(self):
        terms = ["continuum memory", "bộ nhớ liên tục", "nested learning", "catastrophic forgetting"]
        for term in terms:
            synonyms = get_ai_ml_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_unaccented_lookup(self):
        syns = get_ai_ml_synonyms("hoc co giam sat")
        assert "học có giám sát" in syns
        assert "supervised learning" in syns

        syns_qk = get_ai_ml_synonyms("qua khop")
        assert "overfitting" in syns_qk
        assert "chính quy hóa" in syns_qk

        syns_hdd = get_ai_ml_synonyms("ha do doc")
        assert "gradient descent" in syns_hdd
        assert "sgd" in syns_hdd

        syns_bn = get_ai_ml_synonyms("bo nho lien tuc")
        assert "continuum memory" in syns_bn
        assert "nested learning" in syns_bn

    def test_unknown_term_returns_empty(self):
        assert get_ai_ml_synonyms("chế tạo máy dệt sợi cổ điển 1800") == []
        assert get_ai_ml_synonyms("") == []


class TestExpandAIMLQueryTerms:
    """Verifies domain query expansion for natural conversational queries."""

    def test_single_term_expansion(self):
        expanded = expand_ai_ml_query_terms("Giải thích về học tăng cường cho người mới bắt đầu?")
        assert len(expanded) > 0
        assert "học tăng cường" in expanded
        assert "reinforcement learning" in expanded
        assert "q-learning" in expanded

    def test_multi_intent_query_expansion(self):
        expanded = expand_ai_ml_query_terms(
            "So sánh hiện tượng quá khớp và thuật toán hạ độ dốc trong huấn luyện"
        )
        assert "quá khớp" in expanded
        assert "overfitting" in expanded
        assert "hạ độ dốc" in expanded
        assert "gradient descent" in expanded

    def test_rag_and_bm25_joint_query(self):
        expanded = expand_ai_ml_query_terms("Làm thế nào để tích hợp bm25 vào hệ thống rag?")
        assert "bm25" in expanded
        assert "okapi bm25" in expanded
        assert "rag" in expanded
        assert "retrieval-augmented generation" in expanded

    def test_sa_cms_and_continuum_memory_query(self):
        expanded = expand_ai_ml_query_terms(
            "Cơ chế SA-CMS quản lý chunk thế nào trong Continuum Memory và nested learning?"
        )
        assert "sa-cms" in expanded
        assert "sparse chunking" in expanded
        assert "continuum memory" in expanded
        assert "nested learning" in expanded

    def test_attention_and_transformer_query(self):
        expanded = expand_ai_ml_query_terms("Cơ chế tự chú ý self-attention trong mô hình transformer")
        assert "self-attention" in expanded
        assert "transformer" in expanded
        assert "cơ chế chú ý" in expanded

    def test_unaccented_natural_queries(self):
        expanded = expand_ai_ml_query_terms("hoc khong giam sat va clustering")
        assert "học không giám sát" in expanded
        assert "unsupervised learning" in expanded
        assert "gom cụm" in expanded

    def test_quoted_and_punctuated_queries(self):
        expanded = expand_ai_ml_query_terms(
            "Tại sao 'gradient descent' kết hợp với 'regularization' tránh được 'overfitting'?"
        )
        assert "gradient descent" in expanded
        assert "regularization" in expanded
        assert "overfitting" in expanded

    def test_no_duplicate_terms_in_expansion(self):
        expanded = expand_ai_ml_query_terms(
            "học có giám sát, supervised learning và dữ liệu có gán nhãn"
        )
        normalized = [normalize_term(t) for t in expanded]
        assert len(normalized) == len(set(normalized)), "Expanded terms must not contain duplicates"

    def test_empty_and_irrelevant_queries(self):
        assert expand_ai_ml_query_terms("") == []
        assert expand_ai_ml_query_terms("   ") == []
        assert expand_ai_ml_query_terms("Hôm nay thời tiết đẹp quá không mưa") == []


class TestAIMLторинFewShotQaDataset:
    """Verifies coverage, schema validity, and pedagogical depth of AI/ML few-shot blueprints."""

    def test_dataset_size_and_diversity(self):
        assert len(AI_ML_FEW_SHOT_EXAMPLES) >= 12, "Dataset should have at least 12 comprehensive examples"

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
            "bullet_types",
            "example",
        }
        for item in AI_ML_FEW_SHOT_EXAMPLES:
            missing = required_fields - set(item.keys())
            assert not missing, f"Item {item.get('id')} missing fields: {missing}"
            assert len(item["canonical_terms"]) >= 2, f"Item {item['id']} must have multiple canonical terms"
            assert len(item["grounded_answer"]) >= 150, f"Item {item['id']} grounded_answer too brief"
            assert len(item["bullet_types"]) >= 2, f"Item {item['id']} bullet_types must contain key takeaways"
            assert len(item["example"]) >= 10, f"Item {item['id']} example must be substantial"

    def test_coverage_of_assigned_key_topics(self):
        all_topics_text = " ".join(
            (item["topic"] + " " + item["question"] + " " + " ".join(item["canonical_terms"])).lower()
            for item in AI_ML_FEW_SHOT_EXAMPLES
        )

        key_topics = [
            "supervised",
            "unsupervised",
            "reinforcement",
            "overfitting",
            "gradient descent",
            "attention",
            "transformer",
            "rag",
            "bm25",
            "sa-cms",
            "continuum memory",
        ]

        for topic in key_topics:
            assert topic in all_topics_text, f"Topic '{topic}' must be covered in AI_ML_FEW_SHOT_EXAMPLES"


class TestCustomRegistryExtensibility:
    """Verifies that AIMLThesaurusRegistry can be customized or extended."""

    def test_custom_registry_override(self):
        custom_clusters = {
            "diffusion_models": ["diffusion model", "mô hình khuếch tán", "ddpm", "stable diffusion"],
            "lora": ["lora", "low-rank adaptation", "tinh chỉnh lora"],
        }
        custom_registry = AIMLThesaurusRegistry(clusters=custom_clusters)

        assert custom_registry.get_cluster_name("mô hình khuếch tán") == "diffusion_models"
        assert custom_registry.get_cluster_name("mo hinh khuech tan") == "diffusion_models"
        assert "stable diffusion" in custom_registry.get_synonyms("diffusion model")

        # Original cluster should not be present in customized instance
        assert custom_registry.get_cluster_name("học có giám sát") is None

        # Query expansion with custom registry
        expanded = custom_registry.expand_query("Làm thế nào để huấn luyện low-rank adaptation?")
        assert "lora" in expanded
        assert "tinh chỉnh lora" in expanded

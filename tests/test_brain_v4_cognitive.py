"""
Tests for Sovereign Brain 4.0 Cognitive Capabilities:
- Causal & Multi-Hop Bridge Decomposition
- 3+ Entity Comparison Decomposition
- Technical Focus Entity Extraction
- Reasoning Types Classification
"""

import pytest
from src.reasoning.query_decomposer import QueryDecomposer, SubQuery, DecompositionResult


def test_decomposition_result_default_fields():
    """Verify DecompositionResult preserves default backward-compatible fields with reasoning_type."""
    res = DecompositionResult(is_compound=False)
    assert res.is_compound is False
    assert res.sub_queries == []
    assert res.original_query == ""
    assert res.reasoning_type == "atomic"


def test_reasoning_type_atomic():
    """Single queries must have reasoning_type='atomic'."""
    decomposer = QueryDecomposer()
    res = decomposer.decompose("SmolLM2-135M có bao nhiêu tham số?")
    assert res.is_compound is False
    assert res.reasoning_type == "atomic"
    assert len(res.sub_queries) == 1
    assert res.sub_queries[0].intent == "atomic"


def test_reasoning_type_compound():
    """Conjunction queries must have reasoning_type='compound'."""
    decomposer = QueryDecomposer()
    query = "Quy định về thời gian công tác là gì đồng thời các biện pháp thực hiện ra sao?"
    res = decomposer.decompose(query)
    assert res.is_compound is True
    assert res.reasoning_type == "compound"
    assert len(res.sub_queries) == 2


def test_reasoning_type_sequential():
    """Sequential question marks must have reasoning_type='sequential'."""
    decomposer = QueryDecomposer()
    query = "Mô hình này gồm những tầng nào? Cách thức huấn luyện ra sao?"
    res = decomposer.decompose(query)
    assert res.is_compound is True
    assert res.reasoning_type == "sequential"
    assert len(res.sub_queries) == 2


def test_reasoning_type_comparison_two_entities():
    """Two-entity comparison queries must retain comparison_a / comparison_b and reasoning_type='comparison'."""
    decomposer = QueryDecomposer()
    query = "So sánh sự khác biệt giữa mô hình SA-CMS và baseline CMS"
    res = decomposer.decompose(query)
    assert res.is_compound is True
    assert res.reasoning_type == "comparison"
    assert len(res.sub_queries) == 2
    assert res.sub_queries[0].intent == "comparison_a"
    assert res.sub_queries[1].intent == "comparison_b"
    assert "sa-cms" in res.sub_queries[0].text.lower()
    assert "baseline cms" in res.sub_queries[1].text.lower()


def test_multi_entity_comparison_three_entities():
    """Three-entity comparison queries decompose into 3 subqueries with reasoning_type='multi_entity_comparison'."""
    decomposer = QueryDecomposer()
    query = "So sánh SA-CMS, baseline CMS và BM25"
    res = decomposer.decompose(query)
    assert res.is_compound is True
    assert res.reasoning_type == "multi_entity_comparison"
    assert len(res.sub_queries) == 3
    assert res.sub_queries[0].intent == "comparison_1"
    assert res.sub_queries[1].intent == "comparison_2"
    assert res.sub_queries[2].intent == "comparison_3"
    assert "sa-cms" in res.sub_queries[0].text.lower()
    assert "baseline cms" in res.sub_queries[1].text.lower()
    assert "bm25" in res.sub_queries[2].text.lower()


def test_multi_entity_comparison_four_entities():
    """Four-entity comparison queries decompose properly into 4 subqueries."""
    decomposer = QueryDecomposer()
    query = "Hãy so sánh giữa PyTorch, TensorFlow, JAX và ONNX"
    res = decomposer.decompose(query)
    assert res.is_compound is True
    assert res.reasoning_type == "multi_entity_comparison"
    assert len(res.sub_queries) == 4
    for i in range(4):
        assert res.sub_queries[i].intent == f"comparison_{i+1}"


def test_causal_chain_why_leads_to():
    """Causal pattern 'tại sao ... lại dẫn đến ...' must produce cause_source and effect_target."""
    decomposer = QueryDecomposer()
    query = "Tại sao việc giảm learning rate lại dẫn đến hội tụ chậm hơn?"
    res = decomposer.decompose(query)
    assert res.is_compound is True
    assert res.reasoning_type == "causal_chain"
    assert len(res.sub_queries) == 2
    # Cause source
    assert res.sub_queries[0].intent == "cause_source"
    assert res.sub_queries[0].dependency_index is None
    assert "giảm learning rate" in res.sub_queries[0].text.lower()
    # Effect target
    assert res.sub_queries[1].intent == "effect_target"
    assert res.sub_queries[1].dependency_index == 0
    assert "hội tụ chậm hơn" in res.sub_queries[1].text.lower()


def test_causal_chain_relation_between():
    """Relational pattern 'mối liên hệ giữa A và B' must decompose into causal chain."""
    decomposer = QueryDecomposer()
    query = "Mối liên hệ giữa kiến trúc Transformer và cơ chế Attention là gì?"
    res = decomposer.decompose(query)
    assert res.is_compound is True
    assert res.reasoning_type == "causal_chain"
    assert len(res.sub_queries) == 2
    assert res.sub_queries[0].intent == "cause_source"
    assert res.sub_queries[0].dependency_index is None
    assert "transformer" in res.sub_queries[0].text.lower()
    assert res.sub_queries[1].intent == "effect_target"
    assert res.sub_queries[1].dependency_index == 0
    assert "attention" in res.sub_queries[1].text.lower()


def test_causal_chain_impact_doi_voi():
    """Causal pattern 'ảnh hưởng của A đối với B' must decompose into causal chain."""
    decomposer = QueryDecomposer()
    query = "Ảnh hưởng của nhiệt độ cao đối với hiệu năng phần cứng?"
    res = decomposer.decompose(query)
    assert res.is_compound is True
    assert res.reasoning_type == "causal_chain"
    assert len(res.sub_queries) == 2
    assert res.sub_queries[0].intent == "cause_source"
    assert res.sub_queries[0].dependency_index is None
    assert "nhiệt độ cao" in res.sub_queries[0].text.lower()
    assert res.sub_queries[1].intent == "effect_target"
    assert res.sub_queries[1].dependency_index == 0
    assert "hiệu năng phần cứng" in res.sub_queries[1].text.lower()


def test_causal_chain_impact_len():
    """Causal pattern 'tác động của A lên B' must decompose into causal chain."""
    decomposer = QueryDecomposer()
    query = "Tác động của ô nhiễm không khí lên sức khỏe con người?"
    res = decomposer.decompose(query)
    assert res.is_compound is True
    assert res.reasoning_type == "causal_chain"
    assert len(res.sub_queries) == 2
    assert res.sub_queries[0].intent == "cause_source"
    assert res.sub_queries[0].dependency_index is None
    assert "ô nhiễm không khí" in res.sub_queries[0].text.lower()
    assert res.sub_queries[1].intent == "effect_target"
    assert res.sub_queries[1].dependency_index == 0
    assert "sức khỏe con người" in res.sub_queries[1].text.lower()


def test_extract_focus_entities_all_types():
    """Extract technical acronyms, models, hardware, topics, and TitleCase entities."""
    decomposer = QueryDecomposer()
    query = "So sánh SA-CMS, SmolLM2-135M, BM25, RAG trên GPU RTX 3050 theo Topic 1 và Chủ đề 2 của Sovereign Brain"
    entities = decomposer.extract_focus_entities(query)
    expected = [
        "SA-CMS",
        "SmolLM2-135M",
        "BM25",
        "RAG",
        "RTX 3050",
        "Topic 1",
        "Chủ đề 2",
        "Sovereign Brain",
    ]
    for exp in expected:
        assert exp in entities, f"Expected entity '{exp}' not found in {entities}"


def test_extract_focus_entities_empty_and_noise():
    """Empty query or query without entities returns empty list without crashing."""
    decomposer = QueryDecomposer()
    assert decomposer.extract_focus_entities("") == []
    assert decomposer.extract_focus_entities("   ") == []
    assert decomposer.extract_focus_entities("hôm nay trời đẹp không?") == []


def test_extract_focus_entities_quoted_terms():
    """Terms enclosed in quotes are extracted as focus entities."""
    decomposer = QueryDecomposer()
    query = 'Tìm kiếm tài liệu về "quy định an toàn lao động" và "bảo hiểm y tế"'
    entities = decomposer.extract_focus_entities(query)
    assert "quy định an toàn lao động" in entities
    assert "bảo hiểm y tế" in entities


def test_extract_focus_entities_suppresses_redundant_substrings():
    """If 'RTX 3050' is extracted, isolated 'RTX' must be suppressed."""
    decomposer = QueryDecomposer()
    query = "Đánh giá card đồ họa RTX 3050 với mô hình SmolLM2-135M"
    entities = decomposer.extract_focus_entities(query)
    assert "RTX 3050" in entities
    assert "RTX" not in entities
    assert "SmolLM2-135M" in entities
    assert "135M" not in entities

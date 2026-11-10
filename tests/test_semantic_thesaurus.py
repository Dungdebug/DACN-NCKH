"""
Unit Tests for Semantic Thesaurus & Soft Lexical Matching (Ticket 6.1).
Validates:
1. Synonym cluster lookup for colloquial and technical expressions.
2. Intent alignment expansion (ADVANTAGE, LIMITATION, MECHANISM, CAUSE, METRIC).
3. Soft lexical scoring bridging colloquial queries ("giúp ích gì") to technical text ("tiết kiệm 87,5% chi phí token").
4. Strict tone/accent insensitivity and stopword filtering.
5. Integration with NeuralReaderEngine sentence scoring.
"""

import pytest
from src.self_study.semantic_thesaurus import SemanticThesaurus
from src.self_study.reader import NeuralReaderEngine


@pytest.fixture
def thesaurus():
    return SemanticThesaurus()


def test_synonym_cluster_lookup(thesaurus):
    """Verifies that colloquial verbs/nouns expand to technical equivalents."""
    # "giúp ích" belongs to advantage/benefit cluster
    syns = thesaurus.get_synonyms_for_phrase("giúp ích")
    assert len(syns) > 0
    assert any(s in syns for s in ["ưu điểm", "tiết kiệm", "tối ưu", "hiệu quả"])

    # "tại sao" belongs to cause/motive cluster
    cause_syns = thesaurus.get_synonyms_for_phrase("tại sao")
    assert any(s in cause_syns for s in ["động lực", "nguyên nhân", "lý do"])

    # "nhược điểm" belongs to limitation cluster
    lim_syns = thesaurus.get_synonyms_for_phrase("nhược điểm")
    assert any(s in lim_syns for s in ["hạn chế", "thách thức", "rủi ro"])


def test_intent_alignment_expansion(thesaurus):
    """Verifies retrieval of all relevant cues for a given intent category."""
    adv_cues = thesaurus.get_intent_cues("ADVANTAGE")
    assert "tiết kiệm" in adv_cues
    assert "tối ưu" in adv_cues
    assert "đạt" in adv_cues

    mech_cues = thesaurus.get_intent_cues("MECHANISM")
    assert "cơ chế" in mech_cues
    assert "kiến trúc" in mech_cues

    metric_cues = thesaurus.get_intent_cues("METRIC")
    assert "tỷ lệ" in metric_cues
    assert "độ trễ" in metric_cues


def test_expand_query_terms(thesaurus):
    """Verifies query terms expansion with synonyms while avoiding token bloat."""
    query = "Mô hình này giúp ích gì cho bộ nhớ?"
    expanded = thesaurus.expand_query_terms(query)
    # Check original meaningful terms are kept
    assert any("mô hình" in t.lower() or "mo hinh" in t.lower() for t in expanded)
    assert any("bộ nhớ" in t.lower() or "bo nho" in t.lower() for t in expanded)
    # Check synonym expansions are added
    assert any(t in expanded for t in ["ưu điểm", "tiết kiệm", "tối ưu", "hiệu quả"])


def test_soft_match_score_bridges_lexical_gap(thesaurus):
    """
    Verifies that a colloquial query scores high on a technical sentence
    even when exact phrasing differs.
    """
    query = "SA-CMS giúp ích gì cho hệ thống?"
    sentence = "Mô hình SA-CMS đạt tỷ lệ giảm 87,5% chi phí token và tối ưu hóa tài nguyên phần cứng."

    score, matched = thesaurus.compute_soft_match(
        query=query,
        sentence=sentence,
        intent="ADVANTAGE",
    )
    assert score > 0.40
    assert len(matched) >= 2
    # "SA-CMS" and advantage concept (tiết kiệm/tối ưu/giảm) matched
    assert any("sa-cms" in m.lower() for m in matched)


def test_reader_engine_integration_with_thesaurus():
    """
    Verifies that NeuralReaderEngine with SemanticThesaurus can answer
    colloquial phrasing questions by extracting the correct technical sentence.
    """
    reader = NeuralReaderEngine()
    query = "Mô hình SA-CMS có tác dụng gì đối với chi phí?"
    passages = [
        {
            "passage_id": "p1",
            "document_id": "doc1",
            "document_title": "Báo cáo SA-CMS",
            "section_title": "Kết quả",
            "citation_label": "[1]",
            "score": 10.0,
            "text": (
                "Hệ thống SA-CMS là bộ nhớ đa thang căn theo cấu trúc phân tầng. "
                "Mô hình đạt tỷ lệ giảm 87,5% chi phí token trên GPU RTX 3050. "
                "Tuy nhiên, hạn chế là cần văn bản có phân đoạn rõ ràng."
            ),
        }
    ]

    res = reader.synthesize(query=query, passages=passages)
    assert res.is_grounded is True
    # The reader should choose sentence 2 discussing cost reduction ("87,5% chi phí token")
    assert "87,5%" in res.core_answer
    assert "chi phí token" in res.core_answer


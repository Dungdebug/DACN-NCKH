"""
Unit Tests for EntropySurprisalGate & Predictive Coding (Ticket 3.1).
Validates:
1. Information-theoretic surprisal metrics for stopwords vs technical terms.
2. High-information vs low-information sentence classification.
3. Conversational filler pruning with token compression ratio.
4. Fallback safety when all sentences are low-information.
5. Empty and edge-case inputs.
"""

import pytest
from src.reasoning.entropy_gate import (
    EntropySurprisalGate,
    SurprisalResult,
)


def test_word_surprisal_calculation():
    gate = EntropySurprisalGate()

    # Common function words must have low surprisal
    s_va = gate.calculate_word_surprisal("và")
    s_la = gate.calculate_word_surprisal("là")
    assert s_va < 2.5
    assert s_la < 2.5

    # Numbers and acronyms must have high surprisal
    s_metric = gate.calculate_word_surprisal("87,5%")
    s_acronym = gate.calculate_word_surprisal("SA-CMS")
    assert s_metric >= 8.0
    assert s_acronym >= 8.0


def test_sentence_surprisal_evaluation():
    gate = EntropySurprisalGate()

    factual = "Mô hình SA-CMS đạt tỷ lệ giảm 87,5% chi phí token so với RAG."
    res_factual = gate.evaluate_sentence_surprisal(factual)
    assert res_factual["is_high_info"] is True
    assert res_factual["has_metric"] is True
    assert res_factual["mean_surprisal"] >= 3.5

    filler = "Dạ vâng xin chào bạn rất vui được hỗ trợ bạn ngày hôm nay."
    res_filler = gate.evaluate_sentence_surprisal(filler)
    assert res_filler["is_high_info"] is False
    assert res_filler["has_metric"] is False


def test_filter_text_prunes_filler():
    gate = EntropySurprisalGate(surprisal_threshold=3.5)
    text = (
        "Xin chào bạn, rất vui được hỗ trợ bạn hôm nay. "
        "Mô hình SA-CMS là hệ thống bộ nhớ liên tục cấu trúc đa tầng. "
        "Như chúng ta đã biết thì việc này rất hữu ích. "
        "Độ trễ trung bình đạt 18.2 ms trên card RTX 3050."
    )

    res = gate.filter_text(text)
    assert isinstance(res, SurprisalResult)
    assert res.tokens_saved > 0
    assert res.compression_ratio_pct > 25.0
    assert len(res.high_info_spans) == 2
    assert "SA-CMS" in res.filtered_text
    assert "18.2 ms" in res.filtered_text
    assert "Xin chào bạn" not in res.filtered_text


def test_all_filler_safety_fallback():
    gate = EntropySurprisalGate(surprisal_threshold=5.0)
    all_filler = "Xin chào bạn. Chúc bạn một ngày tốt lành. Cảm ơn bạn đã hỏi."
    res = gate.filter_text(all_filler)
    # Must never return empty; safety fallback preserves the best sentence
    assert len(res.filtered_text) > 0
    assert len(res.high_info_spans) >= 1


def test_empty_input_handling():
    gate = EntropySurprisalGate()
    res = gate.filter_text("")
    assert res.original_tokens == 0
    assert res.filtered_tokens == 0
    assert res.filtered_text == ""


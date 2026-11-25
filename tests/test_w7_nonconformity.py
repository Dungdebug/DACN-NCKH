"""Tests for Ticket T5a: WaveCMS-R² nonconformity signals & Mondrian grouping."""

import math
import pytest

from src.wave_cms.nonconformity import (
    MONDRIAN_GROUPS,
    R3_SIGNAL_NAMES,
    SIGNAL_NAMES,
    assign_group,
    extract_signals,
)


def test_signal_names_and_keys_present():
    assert SIGNAL_NAMES == (
        "margin",
        "entropy",
        "lolo_drop",
        "salient_cov",
        "negation_mismatch",
        "answer_type_ok",
        "pmi",
        "bm25_max",
    )

    evidence = [
        {"text": "Albert Einstein published general relativity in 1915.", "bm25": 5.0},
        {"text": "Relativity revolutionized modern physics.", "bm25": 2.0},
    ]
    routing = {"alphas": [0.8, 0.2]}

    sig_no_pmi = extract_signals(
        query="When did Albert Einstein publish general relativity?",
        candidate_answer="1915",
        evidence=evidence,
        routing=routing,
        pmi=None,
    )
    for name in SIGNAL_NAMES:
        assert name in sig_no_pmi
        assert isinstance(sig_no_pmi[name], float)
        assert math.isfinite(sig_no_pmi[name])
    assert sig_no_pmi["pmi"] == 0.0
    assert sig_no_pmi["pmi_missing"] == 1.0
    assert math.isclose(sig_no_pmi["margin"], 0.6, rel_tol=1e-6)
    assert 0.0 <= sig_no_pmi["entropy"] <= 1.0
    assert math.isclose(sig_no_pmi["bm25_max"], 1.0 / (1.0 + math.exp(-1.0)), rel_tol=1e-6)

    sig_with_pmi = extract_signals(
        query="When did Albert Einstein publish general relativity?",
        candidate_answer="1915",
        evidence=evidence,
        routing=routing,
        pmi=1.75,
    )
    assert sig_with_pmi["pmi"] == 1.75
    assert sig_with_pmi["pmi_missing"] == 0.0


def test_lolo_drop_reflects_top1_uniqueness():
    query = "Who discovered penicillin in 1928?"
    ans = "Alexander Fleming"

    unique_ev = [
        {"text": "Alexander Fleming discovered penicillin in 1928.", "score": 0.9},
        {"text": "Antibiotics treat bacterial infections.", "score": 0.1},
    ]
    redundant_ev = [
        {"text": "Alexander Fleming discovered penicillin in 1928.", "score": 0.9},
        {"text": "In 1928, Alexander Fleming discovered penicillin.", "score": 0.8},
    ]

    drop_unique = extract_signals(query, ans, unique_ev)["lolo_drop"]
    drop_redundant = extract_signals(query, ans, redundant_ev)["lolo_drop"]
    assert drop_unique > drop_redundant
    assert drop_unique > 0.5
    assert drop_redundant == 0.0


def test_squad2_adversarial_negation_and_antonyms():
    # Consistent pair
    ok = extract_signals(
        query="Did the temperature increase after 2010?",
        candidate_answer="Yes, it increased.",
        evidence=[{"text": "Global temperature continued to increase after 2010."}],
    )
    assert ok["negation_mismatch"] == 0.0

    # Negation contradiction (EN)
    neg_en = extract_signals(
        query="Did the treaty prevent the war in 1914?",
        candidate_answer="It prevented the war.",
        evidence=[{"text": "The treaty did not prevent the war in 1914."}],
    )
    assert neg_en["negation_mismatch"] >= 0.7

    # Negation contradiction (VI)
    neg_vi = extract_signals(
        query="Thuật toán có hội tụ trên tập dữ liệu này không?",
        candidate_answer="Có hội tụ.",
        evidence=[{"text": "Thuật toán chưa hội tụ trên tập dữ liệu này."}],
    )
    assert neg_vi["negation_mismatch"] >= 0.7

    # Antonym pairs: increase vs decrease, before vs after, largest vs smallest, most vs least
    antonym_cases = [
        ("Did sales increase in Q4?", "Sales decreased in Q4."),
        ("Was the treaty signed before 1950?", "The treaty was signed after 1950."),
        ("Which city has the largest population?", "This city has the smallest population."),
        ("Which team won the most medals?", "The team won the least medals."),
    ]
    for q, ev_text in antonym_cases:
        sig = extract_signals(q, "answer", [{"text": ev_text}])
        assert sig["negation_mismatch"] >= 0.8, f"Failed antonym detection for: {q} vs {ev_text}"


def test_squad2_adversarial_entity_and_number_swap():
    # Entity swap: query asks about Marie Curie, evidence/answer talks about Isaac Newton
    match_entity = extract_signals(
        query="What element did Marie Curie isolate in 1902?",
        candidate_answer="radium",
        evidence=[{"text": "Marie Curie isolated pure radium in 1902."}],
    )
    swap_entity = extract_signals(
        query="What element did Marie Curie isolate in 1902?",
        candidate_answer="radium",
        evidence=[{"text": "Isaac Newton studied optics and gravity in 1902."}],
    )
    assert match_entity["salient_cov"] > swap_entity["salient_cov"]
    assert match_entity["salient_cov"] >= 0.8
    assert swap_entity["salient_cov"] < 0.6

    # Number/year swap: 1998 vs 2005
    match_num = extract_signals(
        query="What happened in 1998 at CERN?",
        candidate_answer="experiment started",
        evidence=[{"text": "In 1998, the CERN experiment started."}],
    )
    swap_num = extract_signals(
        query="What happened in 1998 at CERN?",
        candidate_answer="experiment started",
        evidence=[{"text": "In 2005, the CERN experiment started."}],
    )
    assert match_num["salient_cov"] > swap_num["salient_cov"]


def test_wh_answer_type_compatibility():
    # Good pairs
    assert extract_signals("When was Python released?", "1991", [])["answer_type_ok"] == 1.0
    assert extract_signals("Năm nào Việt Nam gia nhập WTO?", "năm 2007", [])["answer_type_ok"] == 1.0
    assert extract_signals("How many planets are in the solar system?", "8 planets", [])["answer_type_ok"] == 1.0
    assert extract_signals("Có bao nhiêu lớp trong mô hình?", "12 lớp", [])["answer_type_ok"] == 1.0
    assert extract_signals("Who invented the World Wide Web?", "Tim Berners-Lee", [])["answer_type_ok"] == 1.0
    assert extract_signals("Where is the Louvre located?", "Paris, France", [])["answer_type_ok"] == 1.0
    assert extract_signals("Why does ice float on water?", "Because ice is less dense than liquid water", [])["answer_type_ok"] == 1.0

    # Mismatched pairs
    assert extract_signals("When was Python released?", "Guido van Rossum", [])["answer_type_ok"] < 0.5
    assert extract_signals("How many planets are in the solar system?", "Mercury and Venus", [])["answer_type_ok"] < 0.5
    assert extract_signals("Who invented the World Wide Web?", "1989", [])["answer_type_ok"] < 0.5
    assert extract_signals("Where is the Louvre located?", "42", [])["answer_type_ok"] < 0.5
    assert extract_signals("Why does ice float on water?", "ice", [])["answer_type_ok"] < 0.5


def test_assign_group_invariant_to_hidden_labels():
    cases = [
        ({"source": "uit_vi_wiki", "title": "Lịch sử Việt Nam"}, "Chiến thắng Bạch Đằng năm nào?", 4000, "VI-encyclopedia"),
        ({"source": "wikipedia", "language": "en"}, "What is photosynthesis?", 3500, "EN-encyclopedia"),
        ({"source": "squad_v2", "language": "en"}, "Who wrote Hamlet?", 1200, "EN-reading"),
        ({"source": "qasper", "language": "en"}, "What F1 score did the model achieve?", 18000, "EN-longpaper"),
    ]

    for meta, query, length, expected_group in cases:
        assert expected_group in MONDRIAN_GROUPS
        g_clean = assign_group(doc_meta=meta, query=query, doc_length_chars=length)
        assert g_clean == expected_group

        # Inject hidden ground-truth labels; group assignment MUST be identical
        poisoned_a = {**meta, "answerable": True, "gold_answer": "1999", "is_impossible": False, "label": 1}
        poisoned_b = {**meta, "answerable": False, "gold_answer": "", "is_impossible": True, "label": 0}
        assert assign_group(doc_meta=poisoned_a, query=query, doc_length_chars=length) == g_clean
        assert assign_group(doc_meta=poisoned_b, query=query, doc_length_chars=length) == g_clean


def test_r3_signal_names_and_new_signals():
    assert R3_SIGNAL_NAMES == SIGNAL_NAMES + (
        "cross_retriever_agree",
        "sent_margin",
        "idf_query_cov",
        "analyzer_conf",
    )

    evidence = [
        {
            "text": (
                "Albert Einstein published general relativity in 1915. "
                "He later worked at Princeton University."
            ),
            "bm25": 5.0,
        }
    ]

    sig_agree = extract_signals(
        query="When did Albert Einstein publish general relativity?",
        candidate_answer="1915",
        evidence=evidence,
        bm25_best_sentence="Albert Einstein published general relativity in 1915.",
        idf_map={"einstein": 4.0, "relativity": 3.5, "publish": 2.0, "general": 1.0, "albert": 2.5},
        analyzer_conf=0.92,
        sentence_scores=[0.95, 0.25, 0.1],
    )
    for name in R3_SIGNAL_NAMES:
        assert name in sig_agree
        assert isinstance(sig_agree[name], float)
        assert math.isfinite(sig_agree[name])

    assert sig_agree["cross_retriever_agree"] > 0.8
    assert math.isclose(sig_agree["sent_margin"], 0.7, rel_tol=1e-6)
    assert math.isclose(sig_agree["idf_query_cov"], 1.0, rel_tol=1e-6)
    assert math.isclose(sig_agree["analyzer_conf"], 0.92, rel_tol=1e-6)

    # Disagreement with BM25 best sentence and fallback analyzer_conf / sent_margin
    sig_disagree = extract_signals(
        query="When did Albert Einstein publish general relativity?",
        candidate_answer="1915",
        evidence=evidence,
        bm25_best_sentence="Isaac Newton formulated classical mechanics in 1687.",
        analyzer_conf=None,
        sentence_scores=None,
    )
    assert sig_disagree["cross_retriever_agree"] == 0.0
    assert 0.0 <= sig_disagree["sent_margin"] <= 1.0
    assert sig_disagree["sent_margin"] > 0.5
    assert 0.0 < sig_disagree["analyzer_conf"] <= 1.0

    # Default bm25_best_sentence=None compares candidate_answer with best sentence of evidence[0]
    sig_default_bm25 = extract_signals(
        query="When did Albert Einstein publish general relativity?",
        candidate_answer="1915",
        evidence=evidence,
    )
    assert sig_default_bm25["cross_retriever_agree"] > 0.3


def test_r3_idf_query_cov_weighting():
    evidence = [{"text": "The photon energy depends on frequency."}]
    # Rare keyword "hamiltonian" missing vs common keyword "energy" present
    idf_map = {"hamiltonian": 6.0, "energy": 1.0}
    sig_missing_rare = extract_signals(
        query="How does the hamiltonian energy behave?",
        candidate_answer="depends on frequency",
        evidence=evidence,
        idf_map=idf_map,
    )
    assert sig_missing_rare["idf_query_cov"] < 0.3

    # Without idf_map, length-based approximation still bounds in [0, 1]
    sig_approx = extract_signals(
        query="How does the hamiltonian energy behave?",
        candidate_answer="depends on frequency",
        evidence=evidence,
    )
    assert 0.0 < sig_approx["idf_query_cov"] < 1.0


def test_r3_long_passage_anti_dilution():
    noise_sentences = [
        f"Noise sentence {i} discusses unrelated weather and did not mention physics."
        for i in range(15)
    ]
    target_sentence = "Marie Curie isolated pure radium in 1902 in Paris."
    long_passage = " ".join(noise_sentences[:7] + [target_sentence] + noise_sentences[7:])

    sig = extract_signals(
        query="What element did Marie Curie isolate in 1902?",
        candidate_answer="radium",
        evidence=[{"text": long_passage, "score": 0.9}],
    )
    # Sentence-level concentration avoids dilution from 15 noise sentences (even with "did not")
    assert sig["salient_cov"] >= 0.8
    assert sig["negation_mismatch"] == 0.0
    assert sig["idf_query_cov"] >= 0.7
    assert sig["sent_margin"] > 0.4
    assert sig["analyzer_conf"] > 0.75

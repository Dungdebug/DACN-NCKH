"""
Tests for Sovereign Brain 4.0 Neural Reader Engine upgrades:
1. APPLICATION query intent detection ('ung dung', 'ap dung', 'trien khai thuc te', 'use case', 'practical').
2. Acronym & Technical Entity Pinning Bonus (+2.0 per matched entity: SA-CMS, SmolLM2, BM25, RTX, B2, P2).
3. Ordered Bigram Phrase Bonus (+1.5 when 2+ consecutive non-stopword query tokens appear adjacent in sentence).
4. Backward compatibility with existing intent categories and grounded synthesis pipeline.
"""

import pytest
from src.self_study.reader import (
    NeuralReaderEngine,
    NeuralReaderResult,
    detect_query_intent,
    extract_technical_entities,
    split_clean_sentences,
    strip_accents,
)


# ============================================================================
# 1. APPLICATION Intent Detection & Backward Compatibility Tests
# ============================================================================

def test_detect_query_intent_application_variants():
    # 'ung dung'
    assert detect_query_intent("Ứng dụng của hệ thống SA-CMS trong thực tế") == "APPLICATION"
    assert detect_query_intent("Cac ung dung pho bien trong doanh nghiep") == "APPLICATION"

    # 'ap dung'
    assert detect_query_intent("Khả năng áp dụng phương pháp này vào edge device") == "APPLICATION"
    assert detect_query_intent("Co the ap dung mo hinh vao tim kiem van ban") == "APPLICATION"

    # 'trien khai thuc te'
    assert detect_query_intent("Kinh nghiệm triển khai thực tế trên GPU nhỏ") == "APPLICATION"
    assert detect_query_intent("Luu y khi trien khai thuc te mo hinh") == "APPLICATION"

    # 'use case'
    assert detect_query_intent("Các use case tiêu biểu của hệ thống bộ nhớ") == "APPLICATION"
    assert detect_query_intent("Nhung use case phu hop nhat") == "APPLICATION"

    # 'practical'
    assert detect_query_intent("Practical guide for offline document inference") == "APPLICATION"


def test_detect_query_intent_backward_compatibility():
    # Existing 10 intent labels must remain 100% functional
    assert detect_query_intent("Tóm tắt nội dung chính của đề cương?") == "SUMMARY"
    assert detect_query_intent("Hạn chế hoặc thách thức lớn nhất là gì?") == "LIMITATION"
    assert detect_query_intent("Ưu điểm vượt trội của phương pháp này") == "ADVANTAGE"
    assert detect_query_intent("Tại sao cần đồng bộ theo nhịp cập nhật?") == "CAUSE"
    assert detect_query_intent("Tỷ lệ giảm token đạt bao nhiêu %?") == "METRIC"
    assert detect_query_intent("Hệ thống bao gồm những thành phần nào?") == "ENUMERATION"
    assert detect_query_intent("So sánh SA-CMS với RAG truyền thống") == "COMPARISON"
    assert detect_query_intent("Mô hình hoạt động như thế nào?") == "MECHANISM"
    assert detect_query_intent("SA-CMS là gì?") == "DEFINITION"
    assert detect_query_intent("Thông tin về tác giả và đơn vị") == "GENERIC"


# ============================================================================
# 2. Acronym & Technical Entity Extraction & Pinning Bonus (+2.0) Tests
# ============================================================================

def test_extract_technical_entities():
    # Target entities: SA-CMS, SmolLM2, BM25, RTX, B2, P2
    query = "Mô hình SmolLM2 trên card RTX và thuật toán BM25 trong hệ thống SA-CMS mức B2 và P2 là gì?"
    entities = extract_technical_entities(query)

    assert "SmolLM2" in entities
    assert "RTX" in entities
    assert "BM25" in entities
    assert "SA-CMS" in entities
    assert "B2" in entities
    assert "P2" in entities

    # Non-technical words, title-case words, stopwords, plain numbers should not be included
    assert "Trong" not in entities
    assert "Hệ" not in entities
    assert "Thống" not in entities
    assert "và" not in entities
    assert "là" not in entities


def test_technical_entity_pinning_bonus_single_entity():
    engine = NeuralReaderEngine(min_confidence=0.10)
    query = "Mô hình SmolLM2 có cấu trúc ra sao?"
    query_keywords = engine._extract_keywords(query)
    intent = "MECHANISM"

    # Sentence A contains the exact technical entity 'SmolLM2'
    sent_with_entity = "Mô hình SmolLM2 sở hữu kiến trúc tham số tối ưu cho edge AI."
    # Sentence B has similar token overlap but lacks the pinned technical entity
    sent_without_entity = "Mô hình tham số sở hữu kiến trúc tối ưu cho edge AI."

    score_a = engine._score_sentence(sent_with_entity, query, query_keywords, intent, passage_rank=0)
    score_b = engine._score_sentence(sent_without_entity, query, query_keywords, intent, passage_rank=0)

    # Score A must be at least +2.0 higher due to the technical entity pinning bonus
    assert score_a > score_b
    assert score_a >= score_b + 2.0


def test_technical_entity_pinning_bonus_multiple_entities():
    engine = NeuralReaderEngine(min_confidence=0.10)
    query = "So sánh hiệu năng SmolLM2 và BM25"
    query_keywords = engine._extract_keywords(query)
    intent = "COMPARISON"

    # Candidate matching both technical entities (+4.0 entity boost)
    sent_both = "Hệ thống kết hợp mô hình SmolLM2 và thuật toán BM25 cho kết quả vượt trội."
    # Candidate matching only 1 entity (+2.0 entity boost)
    sent_one = "Hệ thống kết hợp mô hình SmolLM2 cho kết quả vượt trội."

    score_both = engine._score_sentence(sent_both, query, query_keywords, intent, passage_rank=0)
    score_one = engine._score_sentence(sent_one, query, query_keywords, intent, passage_rank=0)

    assert score_both > score_one
    # The presence of the second technical entity gives an additional boost
    assert score_both >= score_one + 2.0


def test_technical_entity_pinning_case_insensitive_matching():
    engine = NeuralReaderEngine(min_confidence=0.10)
    query = "Đánh giá card đồ họa RTX khi suy luận"
    query_keywords = engine._extract_keywords(query)
    intent = "GENERIC"

    # Exact token in lowercase 'rtx' in candidate sentence
    sent_lower = "Đánh giá card đồ họa rtx cho thấy độ trễ thấp."
    score = engine._score_sentence(sent_lower, query, query_keywords, intent, passage_rank=0)

    # Must receive the +2.0 bonus
    assert score >= 2.0


# ============================================================================
# 3. Ordered Bigram Phrase Bonus (+1.5) Tests
# ============================================================================

def test_ordered_bigram_phrase_bonus():
    engine = NeuralReaderEngine(min_confidence=0.10)
    # Query with consecutive non-stopword tokens: ('bo', 'nho'), ('nho', 'lien'), ('lien', 'tuc')
    query = "bộ nhớ liên tục"
    query_keywords = engine._extract_keywords(query)
    intent = "DEFINITION"

    # Sentence 1: tokens appear adjacent in exact order -> gets +1.5 phrase bonus
    sent_ordered = "Hệ thống là bộ nhớ liên tục được tối ưu hóa cho mô hình."

    # Sentence 2: words are dispersed across other words -> no ordered query bigram
    sent_dispersed = "Hệ thống là các bộ quy tắc giúp nhớ thông tin khi liên lạc qua thủ tục."

    score_ordered = engine._score_sentence(sent_ordered, query, query_keywords, intent, passage_rank=0)
    score_dispersed = engine._score_sentence(sent_dispersed, query, query_keywords, intent, passage_rank=0)

    assert score_ordered > score_dispersed
    assert score_ordered >= score_dispersed + 1.5


def test_ordered_bigram_phrase_bonus_directionality():
    engine = NeuralReaderEngine(min_confidence=0.10)
    # Consecutive non-stopwords: ('thoi', 'gian')
    query = "thời gian"
    query_keywords = engine._extract_keywords(query)
    intent = "METRIC"

    # Sentence with forward ordered bigram ('thoi', 'gian')
    sent_forward = "Kết quả đo được thời gian phản hồi là 15 ms."
    # Sentence with reversed tokens ('gian', 'thoi')
    sent_reverse = "Kết quả đo được gian thời phản hồi là 15 ms."

    score_forward = engine._score_sentence(sent_forward, query, query_keywords, intent, passage_rank=0)
    score_reverse = engine._score_sentence(sent_reverse, query, query_keywords, intent, passage_rank=0)

    # Forward ordered bigram gets +1.5 bonus over reverse
    assert score_forward > score_reverse
    assert score_forward >= score_reverse + 1.5


# ============================================================================
# 4. End-to-End Synthesize Integration Tests
# ============================================================================

@pytest.fixture
def sample_passages():
    return [
        {
            "passage_id": "doc1::p1",
            "document_id": "de_cuong_nckh",
            "document_title": "Đề cương NCKH",
            "section_title": "1. Giới thiệu & Khái niệm",
            "citation_label": "[1]",
            "text": (
                "SA-CMS là hệ thống bộ nhớ liên tục căn chỉnh theo cấu trúc tài liệu. "
                "Hệ thống vận hành với các tầng bộ nhớ B2 và P2 để tối ưu hóa truy xuất ngữ cảnh."
            ),
        },
        {
            "passage_id": "doc1::p2",
            "document_id": "de_cuong_nckh",
            "document_title": "Đề cương NCKH",
            "section_title": "4. Ứng dụng & Thực nghiệm",
            "citation_label": "[2]",
            "text": (
                "Ứng dụng thực tế của SA-CMS tập trung vào trợ lý nghiên cứu và tra cứu tài liệu học thuật. "
                "Khi triển khai thực tế trên GPU RTX 3050, hệ thống duy trì độ trễ dưới 20 ms."
            ),
        },
    ]


def test_synthesize_application_intent(sample_passages):
    engine = NeuralReaderEngine(min_confidence=0.20)
    query = "Ứng dụng của SA-CMS trong nghiên cứu học thuật?"

    res = engine.synthesize(query, sample_passages, answer_mode="balanced")

    assert res.is_grounded is True
    assert res.intent == "APPLICATION"
    assert "nghiên cứu" in res.core_answer.lower() or "học thuật" in res.core_answer.lower()
    assert "[2]" in res.formatted_answer
    assert len(res.followup_questions) > 0
    # Verifies application-tailored followup prompts
    assert any("triển khai thực tế" in fq.lower() for fq in res.followup_questions)


def test_synthesize_technical_pinning_and_bigram_selection(sample_passages):
    engine = NeuralReaderEngine(min_confidence=0.20)
    # Query contains technical entity B2 and bigram 'tầng bộ nhớ'
    query = "Tầng bộ nhớ B2 trong kiến trúc hoạt động ra sao?"

    res = engine.synthesize(query, sample_passages, answer_mode="concise")

    assert res.is_grounded is True
    assert "[1]" in res.formatted_answer
    assert "b2" in res.core_answer.lower()

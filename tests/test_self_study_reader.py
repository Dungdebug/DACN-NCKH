"""
Unit & Integration Tests for NeuralReaderEngine (Ticket 1.3).
Validates:
1. Intent Classification: Accurately identifies 7 question categories.
2. Decimal & Abbreviation Protection: Sentences split cleanly without truncating numbers or names.
3. Grounded Synthesis: Core answer and key points strictly originate from source text.
4. Refusal Integrity: Low confidence / out-of-domain queries are safely refused (zero hallucination).
5. Answer Modes: Respects concise vs balanced depth.
6. Follow-up Question Tail: Generates relevant trailing self-study prompts.
"""

import pytest
from src.self_study.reader import (
    NeuralReaderEngine,
    NeuralReaderResult,
    detect_query_intent,
    split_clean_sentences,
    strip_accents,
)


@pytest.fixture
def sample_passages():
    return [
        {
            "passage_id": "doc1::p1",
            "document_id": "de_cuong_nckh",
            "document_title": "Đề cương NCKH",
            "section_title": "1. Tổng quan",
            "citation_label": "[1]",
            "text": (
                "SA-CMS là hệ thống bộ nhớ liên tục căn chỉnh theo cấu trúc tài liệu. "
                "Hệ thống giải quyết bài toán suy luận trên văn bản dài với chi phí token tối thiểu. "
                "Thay vì dùng sliding window cố định, SA-CMS đồng bộ nhịp cập nhật theo đoạn văn, mục và tài liệu."
            ),
        },
        {
            "passage_id": "doc1::p2",
            "document_id": "de_cuong_nckh",
            "document_title": "Đề cương NCKH",
            "section_title": "3. Kết quả thực nghiệm",
            "citation_label": "[2]",
            "text": (
                "Mô hình đạt tỷ lệ giảm 87,5% chi phí token so với RAG truyền thống. "
                "Độ trễ phản hồi trung bình duy trì dưới 18,2 ms trên GPU RTX 3050. "
                "Tuy nhiên, hạn chế của hệ thống là cần tài liệu có cấu trúc phân đoạn rõ ràng."
            ),
        },
    ]


def test_detect_query_intent():
    assert detect_query_intent("SA-CMS là gì?") == "DEFINITION"
    assert detect_query_intent("Mô hình hoạt động như thế nào qua các cấp độ?") == "MECHANISM"
    assert detect_query_intent("Tỷ lệ giảm token đạt bao nhiêu %?") == "METRIC"
    assert detect_query_intent("Hệ thống bao gồm những thành phần nào?") == "ENUMERATION"
    assert detect_query_intent("So sánh SA-CMS với RAG truyền thống có gì khác?") == "COMPARISON"
    assert detect_query_intent("Hạn chế hoặc thách thức lớn nhất là gì?") == "LIMITATION"
    assert detect_query_intent("Tóm tắt nội dung chính của đề cương?") == "SUMMARY"
    assert detect_query_intent("Thông tin về tác giả") == "GENERIC"


def test_split_clean_sentences_protects_decimals_and_abbrevs():
    text = (
        "Mô hình đạt 87,5% độ chính xác và giảm 3.14 lần chi phí. "
        "Nghiên cứu được thực hiện tại TP.HCM dưới sự hướng dẫn của PGS.TS Nguyễn Văn A. "
        "Kết quả đo được là 15.2 ms mỗi lượt truy vấn."
    )
    sentences = split_clean_sentences(text)
    assert len(sentences) == 3
    assert "87,5%" in sentences[0]
    assert "3.14" in sentences[0]
    assert "TP.HCM" in sentences[1]
    assert "PGS.TS" in sentences[1]
    assert "15.2 ms" in sentences[2]


def test_neural_reader_grounded_synthesis_definition(sample_passages):
    reader = NeuralReaderEngine(min_confidence=0.20)
    query = "SA-CMS là gì?"

    res = reader.synthesize(query, sample_passages, answer_mode="balanced")

    assert res.is_grounded is True
    assert res.intent == "DEFINITION"
    assert "bộ nhớ liên tục" in res.core_answer.lower()
    assert "[1]" in res.formatted_answer
    assert res.tokens_used > 0
    assert res.latency_ms > 0
    assert len(res.followup_questions) > 0


def test_neural_reader_grounded_synthesis_metric(sample_passages):
    reader = NeuralReaderEngine(min_confidence=0.20)
    query = "Tỷ lệ giảm chi phí token là bao nhiêu?"

    res = reader.synthesize(query, sample_passages, answer_mode="concise")

    assert res.is_grounded is True
    assert res.intent == "METRIC"
    assert "87,5%" in res.core_answer
    assert "[2]" in res.formatted_answer


def test_neural_reader_grounded_synthesis_limitation(sample_passages):
    reader = NeuralReaderEngine(min_confidence=0.20)
    query = "Hạn chế của hệ thống là gì?"

    res = reader.synthesize(query, sample_passages, answer_mode="balanced")

    assert res.is_grounded is True
    assert res.intent == "LIMITATION"
    assert "hạn chế" in res.core_answer.lower()
    assert "cấu trúc phân đoạn" in res.core_answer.lower()


def test_neural_reader_refusal_on_irrelevant_query(sample_passages):
    reader = NeuralReaderEngine(min_confidence=0.25)
    # Query completely unrelated to computing or memory
    query = "Cách nấu phở bò gia truyền Hà Nội như thế nào?"

    res = reader.synthesize(query, sample_passages)

    assert res.is_grounded is False
    assert "Không tìm thấy đủ thông tin" in res.formatted_answer
    assert len(res.citations) == 0


def test_neural_reader_empty_inputs():
    reader = NeuralReaderEngine()

    res_empty_q = reader.synthesize("", [])
    assert res_empty_q.is_grounded is False
    assert "Không tìm thấy đủ thông tin" in res_empty_q.formatted_answer

    res_empty_p = reader.synthesize("SA-CMS là gì?", [])
    assert res_empty_p.is_grounded is False
    assert "Không tìm thấy đủ thông tin" in res_empty_p.formatted_answer


def test_neural_reader_serialization(sample_passages):
    reader = NeuralReaderEngine(min_confidence=0.20)
    res = reader.synthesize("SA-CMS là gì?", sample_passages)
    d = res.to_dict()

    assert isinstance(d, dict)
    assert "core_answer" in d
    assert "key_points" in d
    assert "formatted_answer" in d
    assert "citations" in d
    assert "confidence" in d
    assert "is_grounded" in d
    assert "tokens_used" in d
    assert "latency_ms" in d
    assert "followup_questions" in d


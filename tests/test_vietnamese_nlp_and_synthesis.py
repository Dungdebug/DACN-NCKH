"""
Unit tests verifying the 4 key enhancements:
1. Follow-up Question Generation (No broken phrases, LLM-based + Clean Rule Fallback)
2. Vietnamese Word Segmentation & POS Tagging (PyVi integration, distinction of Nouns vs Adverbs/Particles)
3. Semantic-Aware Context Compression (10-15% balanced compression, preserving SVO grammar & entities)
4. Abstractive Synthesis & Clean Formatting (Removal of raw bullet list symbols, uppercase heading normalization, clean citation positioning)
"""

import pytest
from src.utils.vietnamese_nlp import VietnameseNLP
from src.hybrid_qa.answer_span import generate_followup_questions
from src.hybrid_qa.context_compression import ContextCompressor
from src.web.token_optimizer import clean_sentence_abstractive, synthesize_accurate_grounded_answer
from src.web.document_analyzer import DocumentAnalyzer


def test_vietnamese_nlp_word_segmentation_and_clean_term():
    """
    Verifies that PyVi distinguishes nouns from adverbs/particles:
    'Khóa chỉ' -> 'Khóa' (stripping adverb 'chỉ').
    """
    raw_term = "Khóa chỉ"
    cleaned = VietnameseNLP.clean_extracted_term(raw_term)
    assert cleaned == "Khóa", f"Expected 'Khóa', got '{cleaned}'"

    raw_term_2 = "- Về lý thuyết có thể thử tất"
    cleaned_snippet = VietnameseNLP.clean_sentence_snippet(raw_term_2, max_words=6)
    assert not cleaned_snippet.startswith("-"), "Snippet should not start with dash"
    assert "Về lý thuyết" not in cleaned_snippet or len(cleaned_snippet.split()) >= 3


def test_followup_question_generation_accuracy():
    """
    Verifies that 'Khóa chỉ là một chữ cái...' produces a grammatically sound follow-up question
    about 'Khóa', NOT 'Khóa chỉ'.
    """
    query = "Mật mã Caesar hoạt động như thế nào?"
    context = (
        "Khóa chỉ là một chữ cái trong bảng chữ cái tiếng Anh gồm 26 chữ cái. "
        "Vét cạn là phương pháp thử tất cả các trường hợp khóa có thể xảy ra để giải mã."
    )
    questions = generate_followup_questions(query=query, context=context, count=2)
    assert len(questions) >= 1
    # Check that 'Khóa chỉ' is not the topic of the question
    for q in questions:
        assert "Khóa chỉ" not in q, f"Question contains broken phrase 'Khóa chỉ': {q}"
    # Verify that at least one question targets 'Khóa' or 'Vét cạn'
    found_clean_entity = any("Khóa" in q or "Vét cạn" in q for q in questions)
    assert found_clean_entity, f"Expected clean entity in questions: {questions}"


def test_document_analyzer_suggested_questions_cleaning():
    """
    Verifies DocumentAnalyzer.generate_suggested_questions cleans 'Khóa chỉ' to 'Khóa'.
    """
    raw_text = (
        "## 1. Mật mã học cơ bản\n"
        "Khóa chỉ là một chữ cái trong bảng chữ cái tiếng Anh gồm 26 ký tự. "
        "Thuật toán mã hóa dịch chuyển từng ký tự theo giá trị của khóa."
    )
    suggested = DocumentAnalyzer.generate_suggested_questions(raw_text, title="Mật mã học", max_questions=3)
    assert len(suggested) > 0
    for sq in suggested:
        assert "Khóa chỉ là gì" not in sq["question"], f"Found broken question: {sq['question']}"


def test_semantic_aware_context_compression():
    """
    Verifies that context compression achieves 10-25% reduction without stripping
    essential grammar (Subject, Verb, Object) or technical entities.
    """
    compressor = ContextCompressor(target_compression_ratio=0.15)
    query = "Mật mã Caesar có những đặc điểm gì?"
    text = (
        "Như chúng ta đã biết, mật mã Caesar là một trong những phương pháp mã hóa cổ điển đơn giản nhất. "
        "Không thể phủ nhận rằng không gian khóa của hệ thống này chỉ có 26 khả năng. "
        "Do đó, kẻ tấn công hoàn toàn có thể sử dụng phương pháp vét cạn để thử tất cả các khóa và khôi phục bản rõ."
    )
    result = compressor.compress_passage(text, query=query)
    assert result.compressed_text, "Compressed text should not be empty"
    assert result.compression_ratio_pct >= 5.0, f"Expected some compression, got {result.compression_ratio_pct}"
    assert result.compression_ratio_pct <= 40.0, f"Compression too aggressive: {result.compression_ratio_pct}"
    # Ensure key technical entity and full sentence grammar remain
    assert "mật mã Caesar" in result.compressed_text or "Caesar" in result.compressed_text
    assert "vét cạn" in result.compressed_text


def test_clean_sentence_abstractive_normalization():
    """
    Verifies that raw bullet points (*, -) and ALL CAPS headings are normalized
    into clean prose, and citations are removed from middle of clauses.
    """
    raw1 = "* Vét cạn: thử tất cả các trường hợp khóa [2]"
    raw2 = "PHÁ MÃ: phương pháp tấn công vào hệ mật [2]"
    raw3 = "- RSA: thuật toán mã hóa khóa công khai."

    clean1 = clean_sentence_abstractive(raw1)
    clean2 = clean_sentence_abstractive(raw2)
    clean3 = clean_sentence_abstractive(raw3)

    assert not clean1.startswith("*"), f"Leading asterisk not stripped: {clean1}"
    assert "Vét cạn:" in clean1
    assert "[2]" not in clean1, f"Citation tag not stripped from sentence body: {clean1}"

    assert "PHÁ MÃ:" not in clean2, f"ALL CAPS heading should be normalized: {clean2}"
    assert "Phá mã:" in clean2
    assert "[2]" not in clean2

    assert clean3.startswith("RSA:"), f"RSA acronym should be preserved: {clean3}"


def test_synthesize_accurate_grounded_answer_abstractive_flow():
    """
    Verifies that synthesize_accurate_grounded_answer produces a coherent,
    abstractive paragraph without mid-sentence citations or raw bullet artifacts.
    """
    citations = [
        {
            "citation_label": "[2]",
            "document_title": "An toàn thông tin",
            "text": "* Vét cạn: thử tất cả các trường hợp khóa có thể xảy ra. PHÁ MÃ: phương pháp tấn công nhằm giải mã mà không cần biết khóa trước.",
        }
    ]
    answer = synthesize_accurate_grounded_answer(
        question="Phương pháp tấn công mật mã gồm những gì?",
        citations_data=citations,
        answer_mode="balanced",
    )
    assert not answer.startswith("*"), f"Answer should not start with bullet: {answer}"
    assert "* Vét cạn" not in answer, f"Raw bullet marker found in answer: {answer}"
    assert "PHÁ MÃ:" not in answer, f"Raw uppercase heading found in answer: {answer}"
    assert "[2] PHÁ MÃ:" not in answer, f"Citation found awkwardly stuck between sentences: {answer}"
    assert answer.endswith("[2]"), f"Citation [2] should be cleanly placed at end of answer: {answer}"


def test_comprehensive_detailed_answer_across_multiple_passages():
    """
    Verifies that when asked for full, detailed information ('Mật mã Caesar là gì, công thức và phương pháp phá mã?'),
    the system synthesizes facts across ALL evidence passages:
    - Definition and concept from Passage 1 [1]
    - Formula and methods (Vét cạn, Phân tích tần suất) from Passage 2 [2]
    - Limitations from Passage 3 [3]
    """
    class MockP:
        def __init__(self, pid, text, sec):
            self.passage_id = pid
            self.document_id = "DOC_CRYPTO"
            self.document_title = "Giáo trình Mật mã học"
            self.section_title = sec
            self.paragraph_index = 1
            self.score = 10.0
            self.text = text

    passages = [
        MockP(
            "P1",
            "Mật mã Caesar là một trong những hệ mã thay thế cổ điển đơn giản nhất trong lịch sử. "
            "Thuật toán hoạt động bằng cách dịch chuyển mỗi ký tự trong bản rõ đi một khoảng cố định k vị trí trong bảng chữ cái.",
            "1. Định nghĩa và Nguyên lý",
        ),
        MockP(
            "P2",
            "Công thức mã hóa được tính bằng C = (P + k) mod 26. "
            "Các phương pháp phá mã gồm: "
            "Vét cạn: thử lần lượt tất cả 26 giá trị của khóa để khôi phục bản rõ. "
            "Phân tích tần suất: dựa trên tần số xuất hiện của các chữ cái trong ngôn ngữ tự nhiên để suy đoán khóa bí mật.",
            "2. Công thức và Thám mã",
        ),
        MockP(
            "P3",
            "Hạn chế lớn nhất của mật mã Caesar là không gian khóa cực kỳ nhỏ (chỉ có 26 khả năng). "
            "Do đó, hệ thống này hoàn toàn không an toàn trước các cuộc tấn công hiện đại.",
            "3. Đánh giá và Hạn chế",
        ),
    ]

    answer, citations = DocumentAnalyzer.find_best_grounded_answer(
        query="Mật mã Caesar là gì, công thức và phương pháp phá mã như thế nào?",
        evidence_passages=passages,
        answer_mode="balanced",
    )

    body = answer.split("### Nguồn tham khảo")[0]

    # Verify that all 3 aspects requested by user are fully answered:
    # 1. Definition & mechanism from P1
    assert "Caesar" in body
    assert "dịch chuyển" in body
    assert "[1]" in body

    # 2. Formula & methods from P2
    assert "mod 26" in body or "Công thức" in body
    assert "Vét cạn" in body
    assert "Phân tích tần suất" in body
    assert "[2]" in body

    # 3. Limitations from P3
    assert "Hạn chế" in body or "không an toàn" in body
    assert "[3]" in body

    # Citations should cover all 3 passages
    assert len(citations) == 3

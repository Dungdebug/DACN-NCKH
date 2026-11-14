"""
Unit tests for DocumentChunker Topic & Heading Boundary Enforcement.
Verifies that:
1. Topic 48 and Topic 49 are never conjoined in the same chunk.
2. Inline topic headers following a sentence terminator are cleanly split with ZERO overlap.
3. Overlap never steps backward across topic/section boundaries.
4. Multi-language aliases (Topic, Chủ đề, Chu de, Bài, Mục, Chương, Phần, Section, Chapter, Lesson, Unit)
   are correctly recognized and populate section_title and section_id.
5. Short trailing chunks across different topics are never merged.
"""

import pytest
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.document_store import DocumentRecord


def test_chunker_topic_48_and_49_strict_isolation():
    """
    Verifies that Topic 48 and Topic 49 are completely isolated:
    - No passage contains both Topic 48 and Topic 49.
    - Topic 48 body sentences never leak into Topic 49 chunk as overlap.
    - Topic 49 passages have section_title starting with 'Topic 49'.
    """
    raw_text = (
        "Topic 48: Describe a bad restaurant experience\n"
        "Although the problem was solved, I did not have a good experience at that restaurant. "
        "I think good customer service is very important for every business. "
        "Topic 49: Describe an outdoor sport you would like to try\n"
        "I would like to talk about an outdoor sport I want to try. "
        "It is hiking. I first knew about this sport through videos on social media. "
        "In the future, I hope I can go hiking with my friends."
    )

    doc = DocumentRecord(
        document_id="DOC_TOPIC_TEST",
        version=1,
        title="Topic Speak",
        raw_text=raw_text,
    )

    chunker = DocumentChunker(chunk_size=200, chunk_overlap=32, min_chunk_size=30, adaptive_overlap=True)
    passages = chunker.chunk_document(doc)

    assert len(passages) >= 2, f"Expected at least 2 passages, got {len(passages)}"

    # 1. No passage contains both Topic 48 and Topic 49
    for p in passages:
        has_48 = "Topic 48" in p.text
        has_49 = "Topic 49" in p.text
        assert not (has_48 and has_49), f"Passage {p.passage_id} contains both Topic 48 and 49: {p.text}"

    # 2. Check Topic 48 passage
    p48_list = [p for p in passages if "Topic 48" in p.text]
    assert len(p48_list) >= 1
    assert "Topic 49" not in p48_list[0].text
    assert p48_list[0].section_title.startswith("Topic 48")

    # 3. Check Topic 49 passage
    p49_list = [p for p in passages if "Topic 49" in p.text]
    assert len(p49_list) >= 1
    # Zero overlap check: restaurant content must NOT appear in Topic 49 passages
    for p in p49_list:
        assert "restaurant" not in p.text.lower()
        assert "customer service" not in p.text.lower()
        assert p.section_title.startswith("Topic 49")
        assert p.metadata.get("section_id") == "SEC002"


def test_chunker_inline_heading_after_period():
    """
    Verifies that when a topic heading appears inline right after a period
    (common artifact of DOCX text extraction where newlines are stripped or converted to spaces),
    the chunker detects the heading boundary and splits with zero overlap.
    """
    raw_text = (
        "Customer service is crucial for every business. Topic 49: Describe an outdoor sport you would like to try. "
        "Hiking is an amazing activity."
    )

    doc = DocumentRecord(
        document_id="DOC_INLINE_HEADING",
        version=1,
        title="Inline Test",
        raw_text=raw_text,
    )

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=20)
    passages = chunker.chunk_document(doc)

    assert len(passages) == 2
    assert "Topic 49" not in passages[0].text
    assert "Customer service" not in passages[1].text
    assert passages[1].section_title.startswith("Topic 49")


def test_chunker_vietnamese_and_multilingual_topic_aliases():
    """
    Verifies that DocumentChunker recognizes all required topic heading aliases:
    - Chủ đề N:
    - Chu de N:
    - Bài N:
    - Mục N:
    - Chương N:
    - Phần N:
    - Section N:
    - Lesson N:
    - Unit N:
    - Markdown headings: # / ##
    - Numbered headings: 1. / 1.1.
    """
    headings_sample = [
        "Chủ đề 47: Mô tả một thành công bạn tự hào",
        "Chu de 12: Thoi quen hang ngay",
        "Bài 3: Kỹ năng đọc hiểu",
        "Mục 1.2: Mục tiêu nghiên cứu",
        "Chương 1: Tổng quan lý thuyết",
        "Phần 2: Phương pháp thực nghiệm",
        "Section 4: Evaluation and Results",
        "Lesson 5: Vocabulary in Context",
        "Unit 2: Environmental Protection",
        "# 1. Giới thiệu tổng quan",
        "2.1. Kiến trúc mô hình",
    ]

    for h in headings_sample:
        assert DocumentChunker.is_heading(h), f"Failed to recognize heading: '{h}'"

    # End-to-end document chunking with mixed headings
    composite_text = (
        "Chủ đề 47: Mô tả một thành công bạn tự hào\n"
        "Tôi đã hoàn thành đồ án tốt nghiệp xuất sắc. "
        "Bài 3: Luyện tập phát âm\n"
        "Cần phát âm rõ các phụ âm cuối."
    )

    doc = DocumentRecord(
        document_id="DOC_COMPOSITE",
        version=1,
        title="Composite Test",
        raw_text=composite_text,
    )

    chunker = DocumentChunker(chunk_size=120, chunk_overlap=20, min_chunk_size=15)
    passages = chunker.chunk_document(doc)

    assert len(passages) == 2
    assert passages[0].section_title.startswith("Chủ đề 47")
    assert passages[0].metadata["section_id"] == "SEC001"
    assert passages[1].section_title.startswith("Bài 3")
    assert passages[1].metadata["section_id"] == "SEC002"
    assert "Bài 3" not in passages[0].text
    assert "đồ án tốt nghiệp" not in passages[1].text


def test_chunker_short_trailing_topic_not_merged_into_prior_topic():
    """
    Verifies that if a trailing topic is short (< min_chunk_size),
    it is NOT merged into the previous topic's chunk.
    """
    raw_text = (
        "Topic 49: Describe an outdoor sport you would like to try\n"
        "I would like to talk about an outdoor sport I want to try. It is hiking. "
        "I think it will be an interesting and unforgettable experience.\n"
        "Topic 50: Describe a video\n"
        "A short video."
    )

    doc = DocumentRecord(
        document_id="DOC_TRAILING_TOPIC",
        version=1,
        title="Trailing Test",
        raw_text=raw_text,
    )

    chunker = DocumentChunker(chunk_size=200, chunk_overlap=30, min_chunk_size=60)
    passages = chunker.chunk_document(doc)

    # Topic 50 must have its own passage despite being short (< 60 chars)
    assert len(passages) == 2
    assert "Topic 50" not in passages[0].text
    assert passages[1].section_title.startswith("Topic 50")
    assert "Topic 50" in passages[1].text


def test_chunker_no_passage_contains_two_different_topic_numbers():
    """
    Simulates Topic_speak.docx with 4 consecutive topics and validates that
    no passage contains more than 1 distinct Topic number.
    """
    import re

    raw_text = (
        "Topic 1: Describe a time when you were ill\n"
        "I would like to talk about a time when I was ill. Last year, I had the flu. "
        "Topic 2: Describe a famous person that you would like to meet\n"
        "His name is Ludwig van Beethoven. He was a famous German composer and pianist. "
        "Topic 49: Describe an outdoor sport you would like to try\n"
        "It is hiking. I first knew about this sport through videos on social media. "
        "Topic 50: Describe an interesting video that you watched on social media\n"
        "The video was about Beethoven, a famous German composer."
    )

    doc = DocumentRecord(
        document_id="DOC_TOPIC_SPEAK_SIM",
        version=1,
        title="Topic Speak Simulation",
        raw_text=raw_text,
    )

    chunker = DocumentChunker(chunk_size=180, chunk_overlap=32, min_chunk_size=30)
    passages = chunker.chunk_document(doc)

    for p in passages:
        topics_in_passage = re.findall(r"\bTopic\s*(\d+)", p.text)
        unique_topics = set(topics_in_passage)
        assert len(unique_topics) <= 1, (
            f"Passage {p.passage_id} contains multiple distinct topic numbers {unique_topics}: {p.text}"
        )


def test_chunker_page_break_and_slide_headings_isolation():
    """
    Verifies that === Page N === is treated as a hard boundary with zero overlap,
    page markers are stripped from passage text, and slide headings are cleanly isolated.
    """
    slide_text = (
        "=== Page 1 ===\n"
        "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:\n"
        "- Xuất hiện đầu tiên trong ngành mã hóa.\n"
        "- Các hệ mã này phần có từ trước năm 1970.\n"
        "- Hệ mã khóa bí mật hoặc hệ mã khóa cổ điển.\n"
        "- Các phương pháp này là cơ sở để phát triển các thuật toán mã hóa đối xứng hiện nay.\n"
        "- Có 2 phương pháp phổ biến:\n"
        "* Mã hóa thay thế.\n"
        "* Mã hóa hoán vị.\n\n"
        "=== Page 10 ===\n"
        "3. PHÁ MÃ:\n"
        "An ninh hệ mã khóa:\n"
        "- An ninh vô điều kiện: Không thể phá mã dù có thời gian và tài nguyên vô hạn.\n"
        "- An ninh tính toán: Chi phí để phá vỡ vượt quá giá trị của thông tin mã hóa.\n\n"
        "=== Page 30 ===\n"
        "8. HỆ MÃ HOÁ ĐỘN MỘT LẦN:\n"
        "- Hệ mã hóa hoàn hảo (Vernam Cipher / One-Time Pad).\n\n"
        "9. HỆ MÃ HOÁ DÒNG:\n"
        "- Sinh ra một chuỗi khóa z = z1, z2, z3...\n"
        "- Ta có: zi = fi (K, x1, ..., xi-1).\n"
        "- Bản rõ được mã hóa từng bit hoặc từng byte bằng phép XOR với dòng khóa z.\n\n"
        "=== Page 31 ===\n"
        "10. HỆ MÃ HOÁ AUTOKEY:\n"
        "- Cho P = C = K = Z26.\n"
        "- Khóa k = k1, k2, ... ban đầu được mở rộng bằng chính các ký tự của bản rõ.\n\n"
        "=== Page 36 ===\n"
        "11. HỆ MÃ HOÁ PLAYFAIR:\n"
        "- Sử dụng ma trận ký tự 5x5 để mã hóa từng cặp ký tự."
    )

    doc = DocumentRecord(
        document_id="SLIDE_TEST_DOC",
        version=1,
        title="Slide 2: Các Hệ Mã Hóa Đối Xứng",
        raw_text=slide_text,
    )

    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    passages = chunker.chunk_document(doc)

    assert len(passages) >= 4

    # 1. Page artifacts must not pollute passage texts
    for p in passages:
        assert "=== Page" not in p.text, f"Passage {p.passage_id} contains page artifact: {p.text}"

    # 2. Topic/Section 9 (HỆ MÃ HOÁ DÒNG) must NOT contain Topic/Section 10 (HỆ MÃ HOÁ AUTOKEY)
    stream_passages = [
        p for p in passages
        if "HỆ MÃ HOÁ DÒNG" in (p.section_title or "") or "HỆ MÃ HOÁ DÒNG" in p.text
    ]
    assert len(stream_passages) > 0
    for sp in stream_passages:
        assert "10. HỆ MÃ HOÁ AUTOKEY" not in sp.text
        assert "HỆ MÃ HOÁ AUTOKEY" not in (sp.section_title or "")

    # 3. Section 1 must preserve short bullet points (* Mã hóa thay thế, * Mã hóa hoán vị)
    sec1_passages = [
        p for p in passages
        if "GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG" in (p.section_title or "") or "GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG" in p.text
    ]
    assert len(sec1_passages) > 0
    sec1_combined = " ".join(p.text for p in sec1_passages)
    assert "Mã hóa thay thế" in sec1_combined
    assert "Mã hóa hoán vị" in sec1_combined


def test_chunker_roman_numerals_and_slide_caps_headings():
    """
    Verifies that Roman numerals (I., II., III.) and slide uppercase headings
    are recognized as headings without splitting the number from the title.
    """
    text = (
        "I. GIỚI THIỆU TỔNG QUAN\n"
        "Hệ thống thông tin bảo mật đóng vai trò thiết yếu.\n"
        "II. CÁC PHƯƠNG PHÁP MÃ HÓA CỔ ĐIỂN\n"
        "Bao gồm mã hóa thay thế và mã hóa hoán vị."
    )

    doc = DocumentRecord(
        document_id="ROMAN_TEST_DOC",
        version=1,
        title="Roman Heading Test",
        raw_text=text,
    )

    chunker = DocumentChunker(chunk_size=150, chunk_overlap=30)
    passages = chunker.chunk_document(doc)

    assert len(passages) == 2
    assert passages[0].section_title.startswith("I. GIỚI THIỆU TỔNG QUAN")
    assert passages[0].metadata["section_id"] == "SEC001"
    assert passages[1].section_title.startswith("II. CÁC PHƯƠNG PHÁP MÃ HÓA CỔ ĐIỂN")
    assert passages[1].metadata["section_id"] == "SEC002"
    assert "II. CÁC PHƯƠNG PHÁP" not in passages[0].text
    assert "I. GIỚI THIỆU" not in passages[1].text


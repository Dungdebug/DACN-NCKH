"""
Unit & Integration Tests for Token-Reduction Scanning & Clean Grounded QA Mechanisms:
1. Multi-Stage Document Noise & Boilerplate Sanitizer (clean_document_noise)
2. Adaptive Structural Boundary Overlap Chunker (Zero-Overlap on Paragraph Boundaries)
3. Dynamic Top-K Evidence Pruning (Dominance & Marginal Coverage Early-Stopping)
4. Sub-Passage Query-Guided Context Pruner (compress_evidence_context)
5. Salient Grounded Answer Synthesizer without garbage/decimal-splitting (synthesize_accurate_grounded_answer)
"""

import pytest
from src.web.file_parser import clean_document_noise, extract_text_from_file
from src.hybrid_qa.document_store import DocumentRecord
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.retriever import RetrievalResult
from src.hybrid_qa.evidence import EvidenceSelector
from src.web.token_optimizer import (
    split_clean_sentences,
    strip_passage_artifacts,
    compress_evidence_context,
    synthesize_accurate_grounded_answer,
)


def test_clean_document_noise_reduces_tokens_and_preserves_facts():
    noisy_doc = (
        "BÁO CÁO NỘI BỘ MẬT - CÔNG TY ABC\n"
        "1. Giới thiệu chung ............................ 1\n"
        "2. Kiến trúc SA-CMS ............................ 5\n"
        "Trang 1/10\n"
        "**************************************************\n\n\n\n"
        "# 1. Giới thiệu Hệ thống SA-CMS\n\n"
        "Mô hình SmolLM2-135M có 134.5M tham số gốc được đóng băng hoàn toàn.\n"
        "BÁO CÁO NỘI BỘ MẬT - CÔNG TY ABC\n"
        "Page 2 of 10\n"
        "--------------------------------------------------\n\n\n"
        "# 2. Hiệu năng Thực nghiệm\n\n"
        "Kiến trúc Lai P2 đạt 100% độ trung thực trích dẫn và 743.86 MB VRAM đỉnh.\n"
        "BÁO CÁO NỘI BỘ MẬT - CÔNG TY ABC\n"
        "- 3 -\n"
    )

    cleaned, stats = clean_document_noise(noisy_doc)

    # Boilerplate, TOC dots, repeated headers, and page numbers must be stripped
    assert "...." not in cleaned
    assert "Trang 1/10" not in cleaned
    assert "Page 2 of 10" not in cleaned
    assert "BÁO CÁO NỘI BỘ MẬT - CÔNG TY ABC" not in cleaned
    assert "****" not in cleaned

    # Core facts and decimal numbers must be 100% preserved
    assert "SmolLM2-135M" in cleaned
    assert "134.5M" in cleaned
    assert "743.86 MB" in cleaned

    # Token reduction stats must show positive savings
    assert stats["tokens_saved"] > 0
    assert stats["reduction_pct"] > 15.0
    assert stats["removed_lines_count"] >= 6


def test_adaptive_boundary_chunker_zero_overlap_across_paragraphs():
    para_1 = "Đoạn văn thứ nhất trình bày về kiến trúc bộ nhớ tham số tương tục SA-CMS. Hệ thống cập nhật theo ranh giới cấu trúc tự nhiên của văn bản."
    para_2 = "Đoạn văn thứ hai phân tích về bộ truy xuất BM25 và cổng kiểm soát từ chối. Cơ chế này ngăn chặn triệt để hiện tượng ảo giác khi thiếu bằng chứng."
    full_text = f"{para_1}\n\n{para_2}"

    doc = DocumentRecord(
        document_id="DOC_ADAPTIVE",
        version=1,
        title="Adaptive Overlap Test",
        raw_text=full_text,
    )

    # Chunk size fits 1 paragraph per chunk
    chunker = DocumentChunker(chunk_size=150, chunk_overlap=40, min_chunk_size=30, adaptive_overlap=True)
    passages = chunker.chunk_document(doc)

    assert len(passages) == 2
    # Because there is a natural \n\n boundary between para_1 and para_2,
    # para_2 must NOT repeat the ending sentence of para_1
    assert "ranh giới cấu trúc tự nhiên" not in passages[1].text
    assert "Đoạn văn thứ hai" in passages[1].text


def test_dynamic_top_k_prunes_low_value_tail_passages():
    selector = EvidenceSelector(score_threshold=1.0, max_evidence=5, dynamic_top_k=True)
    query = "Kiến trúc SA-CMS có bao nhiêu tham số huấn luyện?"

    results = [
        RetrievalResult(
            "DOC001::P000",
            "DOC001",
            "Kiến trúc SA-CMS tích hợp bộ nhớ tham số 3 cấp độ với đúng 5,314,752 tham số huấn luyện.",
            18.5,
        ),
        RetrievalResult(
            "DOC001::P001",
            "DOC001",
            "Ngoài ra tài liệu còn đề cập đến lịch sử phát triển các hệ thống cơ sở dữ liệu.",
            3.2,
        ),
        RetrievalResult(
            "DOC001::P002",
            "DOC001",
            "Các ứng dụng giao diện người dùng được thiết kế theo chuẩn hiện đại.",
            3.0,
        ),
    ]

    evidence = selector.select_evidence(query, results, dynamic_top_k=True)
    assert evidence.has_sufficient_evidence is True
    # Only the dominant passage covering the query should be kept
    assert len(evidence.evidence_passages) == 1
    assert evidence.evidence_passages[0].passage_id == "DOC001::P000"


def test_split_clean_sentences_preserves_decimals_and_strips_artifacts():
    raw_passage = (
        "=== Page 1 ===\n"
        "# 1.2. Thông số kỹ thuật\n"
        "[DOC_001::P002] Mô hình nền SmolLM2-135M có 134.5M tham số gốc và tiêu thụ 743.86 MB VRAM. "
        "PGS. TS. Nguyễn Văn A đã kiểm chứng với p = 3.03e-7 trên tập dữ liệu."
    )

    sents = split_clean_sentences(raw_passage)
    assert len(sents) == 2
    assert "=== Page 1 ===" not in sents[0]
    assert "DOC_001::P002" not in sents[0]
    assert "134.5M" in sents[0]
    assert "743.86 MB" in sents[0]
    assert "PGS. TS. Nguyễn Văn A" in sents[1]


def test_compress_evidence_context_reduces_prompt_tokens():
    question = "Mức tiêu thụ VRAM đỉnh của hệ thống trên GPU RTX 3050 là bao nhiêu?"
    passages = [
        {
            "passage_id": "DOC001::P001",
            "text": (
                "=== Page 4 ===\n"
                "Trong bối cảnh các mô hình ngôn ngữ lớn ngày càng đòi hỏi phần cứng đắt đỏ, việc tối ưu hóa bộ nhớ là rất cấp thiết. "
                "Nhiều nghiên cứu trước đây phải sử dụng các cụm máy chủ A100 dung lượng lớn. "
                "Hệ thống SA-CMS huấn luyện thành công trên GPU thương mại RTX 3050 6GB với mức tiêu thụ VRAM đỉnh điểm chỉ 743.86 MB. "
                "Điều này mở ra cơ hội triển khai rộng rãi tại các phòng thí nghiệm sinh viên."
            ),
        }
    ]

    compressed_ctx, stats = compress_evidence_context(question, passages, max_sentences_per_passage=1)
    assert "743.86 MB" in compressed_ctx
    assert "=== Page 4 ===" not in compressed_ctx
    assert stats["tokens_saved"] > 0
    assert stats["compression_ratio_pct"] > 30.0


def test_synthesize_accurate_grounded_answer_selects_exact_salient_sentence_not_first_sentence():
    question = "Tỷ lệ từ chối chính xác (Refusal Recall) của kiến trúc lai P2 đạt bao nhiêu phần trăm?"
    citations_data = [
        {
            "citation_index": 1,
            "citation_label": "[1]",
            "passage_id": "DOC001::P003",
            "document_id": "DOC001",
            "document_title": "Báo cáo Khoa học SA-CMS",
            "section_title": "Kết quả RQ3",
            "paragraph_index": 3,
            "score": 14.2,
            "text": (
                "=== Page 2 ===\n"
                "# 3. Kết quả thực nghiệm RQ3\n"
                "Chương này trình bày tổng quan các phương pháp đánh giá trên tập dữ liệu tiếng Việt. "
                "Các mô hình cơ sở thường gặp hiện tượng ảo giác nghiêm trọng khi câu hỏi nằm ngoài tài liệu. "
                "Kiến trúc Lai P2 giải quyết triệt để bài toán trích dẫn và ảo giác, đạt 100% độ trung thực trích dẫn và 76.0% tỷ lệ từ chối chính xác (Refusal Recall)."
            ),
        }
    ]

    answer = synthesize_accurate_grounded_answer(
        question=question,
        citations_data=citations_data,
        route="DIRECT_LOOKUP",
        answer_mode="balanced",
    )

    # Must NOT pick the irrelevant opening sentence ("Chương này trình bày tổng quan...")
    # and must NOT leak '=== Page 2 ===' or broken decimals '76' / '0%'
    assert "=== Page 2 ===" not in answer
    assert "76.0%" in answer
    assert "Refusal Recall" in answer
    assert "[1]" in answer


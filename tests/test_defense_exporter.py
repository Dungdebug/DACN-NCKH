"""
Unit tests for Multi-Scale Progressive Digest & Thesis Defense Exporter (Ticket 7.6).
Verifies 3-level summaries (L1 TL;DR, L2 Executive, L3 Section Digest)
and LaTeX benchmark table generation for thesis defense.
"""

import pytest
from src.brain.progressive_summarizer import ProgressiveSummarizer, ProgressiveDigest
from src.evaluation.thesis_benchmark_exporter import ThesisBenchmarkExporter, BenchmarkScorecard


@pytest.fixture
def sample_academic_text():
    return (
        "## 1. Đặt vấn đề\n"
        "Các mô hình ngôn ngữ lớn gặp thách thức nghiêm trọng về chi phí tính toán khi xử lý tài liệu dài. "
        "Chi phí chú ý bậc hai O(N^2) khiến việc nhồi toàn bộ văn bản vào cửa sổ ngữ cảnh trở nên đắt đỏ.\n\n"
        "## 2. Kiến trúc SA-CMS\n"
        "Chúng tôi đề xuất SA-CMS kết hợp cấu trúc phân tầng bộ nhớ liên tục căn theo ranh giới cấu trúc văn bản. "
        "Hệ thống phân rã thành Level 1 cho đoạn văn, Level 2 cho phần và Level 3 cho toàn văn bản.\n\n"
        "## 3. Thực nghiệm\n"
        "Kết quả đo đạc trên SmolLM2-135M cho thấy SA-CMS giảm 87,5% chi phí token so với RAG tiêu chuẩn. "
        "Hệ thống duy trì độ chính xác nội dung đạt 94,2% và không bị rò rỉ ngữ cảnh.\n\n"
        "## 4. Kết luận\n"
        "Nghiên cứu khẳng định SA-CMS là giải pháp khả thi để xây dựng bộ não hỏi đáp tài liệu tự chủ trên thiết bị cá nhân."
    )


def test_progressive_summarizer_l1_l2_l3(sample_academic_text):
    summarizer = ProgressiveSummarizer()
    digest: ProgressiveDigest = summarizer.generate_digest(sample_academic_text, doc_title="SA-CMS Research")
    
    # L1: Micro TL;DR <= 35 words
    assert digest.l1_tldr is not None
    assert len(digest.l1_tldr.split()) <= 40
    assert "SA-CMS" in digest.l1_tldr
    
    # L2: Executive Summary (3-5 structured sentences)
    assert digest.l2_executive is not None
    assert "87,5%" in digest.l2_executive or "chi phí token" in digest.l2_executive
    assert len(digest.l2_executive.split(".")) >= 3
    
    # L3: Structural Section Digest
    assert len(digest.l3_section_digest) >= 3
    sec_titles = [s["section_title"] for s in digest.l3_section_digest]
    assert any("Đặt vấn đề" in t or "Kiến trúc" in t for t in sec_titles)


def test_thesis_benchmark_exporter_latex_generation():
    exporter = ThesisBenchmarkExporter()
    scorecards = [
        BenchmarkScorecard(method="B1 (ICL)", token_savings_pct=0.0, grounding_acc_pct=88.5, latency_ms=450.0, vram_mb=3400),
        BenchmarkScorecard(method="B2 (RAG)", token_savings_pct=45.0, grounding_acc_pct=91.0, latency_ms=180.0, vram_mb=2100),
        BenchmarkScorecard(method="B5 (Hope-Attention)", token_savings_pct=60.0, grounding_acc_pct=89.5, latency_ms=120.0, vram_mb=1850),
        BenchmarkScorecard(method="P2 (SA-CMS Hybrid)", token_savings_pct=87.5, grounding_acc_pct=94.2, latency_ms=85.0, vram_mb=1650),
    ]
    
    latex_table = exporter.export_rq_latex_table(scorecards)
    
    assert "\\begin{table}" in latex_table
    assert "\\end{table}" in latex_table
    assert "P2 (SA-CMS Hybrid)" in latex_table
    assert "87.5" in latex_table or "87,5" in latex_table
    assert "\\caption" in latex_table


def test_ablation_latex_export():
    exporter = ThesisBenchmarkExporter()
    ablation_data = [
        {"config": "Full SA-CMS (Levels 1+2+3)", "f1_score": 94.2, "tokens_saved": "87.5%"},
        {"config": "No Level 2 (Section Memory)", "f1_score": 86.1, "tokens_saved": "72.0%"},
        {"config": "Fixed-Token Schedule (C=1024)", "f1_score": 79.4, "tokens_saved": "64.5%"},
    ]
    
    latex_table = exporter.export_ablation_latex_table(ablation_data)
    assert "\\begin{tabular}" in latex_table
    assert "Fixed-Token Schedule" in latex_table
    assert "\\end{table}" in latex_table


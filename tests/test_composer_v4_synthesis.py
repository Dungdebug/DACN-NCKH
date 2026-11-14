"""
Unit and Integration Tests for Sovereign Brain 4.0 Coherent Answer Composer (Ticket 6.5 Upgrade).
Validates:
1. Discourse dimension classification ('metric', 'limitation', 'mechanism', 'elaboration', 'causal', 'comparison').
2. 100% backward compatibility of classify_sentence_role() and compose().
3. Detailed answer mode support for structural elaboration candidates.
4. Structured comparative block synthesis (synthesize_comparative_block):
   - Header starting with '⚖️ **Đối chiếu đa chiều từ chứng cứ:**'
   - Bullets formatted with ONLY **bold** and `code`
   - ZERO raw '###' markdown headings and ZERO single-asterisk '*italic*' artifacts
   - Verifiable substring grounding and citation tracking ([1], [2])
5. Edge cases: empty candidates, deduplication, missing section titles.
"""

import pytest
from src.self_study.answer_composer import CoherentAnswerComposer


@pytest.fixture
def composer():
    return CoherentAnswerComposer()


def test_classify_discourse_dimension_all_categories(composer):
    """Verifies that all 6 discourse dimensions are accurately classified."""
    # 1. Limitation / Constraint
    assert composer.classify_discourse_dimension(
        "Tuy nhiên, hạn chế của hệ thống là cần GPU rời với dung lượng tối thiểu 8GB."
    ) == "limitation"
    assert composer.classify_discourse_dimension(
        "Thách thức lớn nhất trong việc mở rộng quy mô là độ phức tạp tính toán."
    ) == "limitation"

    # 2. Comparison / Contrast
    assert composer.classify_discourse_dimension(
        "So với RAG truyền thống, SA-CMS giúp tối ưu hóa bộ nhớ hiệu quả hơn."
    ) == "comparison"
    assert composer.classify_discourse_dimension(
        "Phương pháp A vượt trội hơn phương pháp B trên mọi chỉ số benchmark."
    ) == "comparison"
    assert composer.classify_discourse_dimension(
        "Trong khi mô hình baseline yêu cầu 16GB VRAM, hệ thống mới chỉ cần 4GB."
    ) == "comparison"
    assert composer.classify_discourse_dimension(
        "Đánh giá hiệu năng SA-CMS vs Baseline RAG."
    ) == "comparison"

    # 3. Causal / Effect
    assert composer.classify_discourse_dimension(
        "Việc tối ưu hóa cấu trúc dẫn đến giảm thiểu độ trễ truy xuất."
    ) == "causal"
    assert composer.classify_discourse_dimension(
        "Do đó, hệ thống duy trì được tính ổn định và chống trôi dạt thông tin."
    ) == "causal"
    assert composer.classify_discourse_dimension(
        "Nhờ đó, tốc độ hội tụ của mô hình được cải thiện rõ rệt."
    ) == "causal"
    assert composer.classify_discourse_dimension(
        "Sự suy hao tín hiệu gây ra hiện tượng mất mát dữ liệu."
    ) == "causal"

    # 4. Metric / Quantitative
    assert composer.classify_discourse_dimension(
        "Mô hình đạt tỷ lệ giảm 87,5% chi phí token."
    ) == "metric"
    assert composer.classify_discourse_dimension(
        "Độ trễ trung bình đo được là 18.2 ms."
    ) == "metric"

    # 5. Mechanism / Architecture
    assert composer.classify_discourse_dimension(
        "Cơ chế cập nhật diễn ra thông qua cổng thích ứng ba tầng."
    ) == "mechanism"
    assert composer.classify_discourse_dimension(
        "Kiến trúc tích hợp các bộ điều hợp nơ-ron cục bộ."
    ) == "mechanism"

    # 6. Elaboration / Background
    assert composer.classify_discourse_dimension(
        "Tài liệu nghiên cứu các vấn đề về biểu diễn tri thức trong tài liệu dài."
    ) == "elaboration"


def test_classify_sentence_role_backward_compatibility(composer):
    """Ensures existing classify_sentence_role() continues returning exactly the 4 legacy roles."""
    assert composer.classify_sentence_role("Mô hình đạt tỷ lệ giảm 87,5% chi phí token.") == "metric"
    assert composer.classify_sentence_role("Độ trễ phản hồi trung bình là 18.2 ms.") == "metric"
    assert composer.classify_sentence_role("Hạn chế của hệ thống là cần tài liệu chuẩn hóa.") == "limitation"
    assert composer.classify_sentence_role("Tuy nhiên, thách thức đặt ra là bộ nhớ GPU.") == "limitation"
    assert composer.classify_sentence_role("Cơ chế cập nhật diễn ra thông qua cổng thích ứng.") == "mechanism"
    assert composer.classify_sentence_role("Tài liệu nghiên cứu các vấn đề về trí tuệ nhân tạo.") == "elaboration"


def test_compose_detailed_mode_includes_elaboration(composer):
    """Verifies that compose() in detailed mode includes elaboration candidate when limitation is absent."""
    candidates = [
        {
            "sentence": "Hệ thống SA-CMS là bộ nhớ liên tục căn chỉnh cấu trúc tài liệu.",
            "citation_label": "[1]",
        },
        {
            "sentence": "Mô hình đạt tỷ lệ giảm 87,5% chi phí token trên GPU RTX 3050.",
            "citation_label": "[2]",
        },
        {
            "sentence": "Cơ chế cập nhật đồng bộ qua 3 cấp độ đoạn văn, phần mục và toàn văn tài liệu.",
            "citation_label": "[3]",
        },
    ]

    prose, key_points, active_citations = composer.compose(
        query="Kiến trúc SA-CMS là gì và hiệu năng ra sao?",
        candidates=candidates,
        answer_mode="detailed",
    )

    # Core sentence [1] must be present
    assert "bộ nhớ liên tục" in prose
    assert "[1]" in prose

    # Metric sentence [2] must be present with fluent discourse marker
    assert "87,5%" in prose
    assert "[2]" in prose

    # Elaboration sentence [3] must be included in detailed mode
    assert "Cụ thể hơn," in prose
    assert "3 cấp độ đoạn văn" in prose
    assert "[3]" in prose

    # All 3 sentences in key points and active citations
    assert len(key_points) == 3
    assert len(active_citations) == 3


def test_synthesize_comparative_block_structure_and_formatting(composer):
    """
    Verifies:
    1. Header matches '⚖️ **Đối chiếu đa chiều từ chứng cứ:**'.
    2. Uses ONLY **bold** and `code`.
    3. ZERO markdown headings (no '###').
    4. ZERO single-asterisk '*italic*' syntax.
    5. Correct bullet markers ('- ...').
    6. Grounded citations [1], [2].
    """
    candidates = [
        {
            "sentence": "So với RAG truyền thống, SA-CMS giúp giảm 87,5% chi phí token.",
            "citation_label": "[1]",
            "section_title": "3. Kết quả Thực nghiệm",
        },
        {
            "sentence": "Độ trễ phản hồi trung bình của hệ thống là 18.2 ms trên RTX 3050 4GB.",
            "citation_label": "[2]",
            "section_title": "3. Kết quả Thực nghiệm",
        },
        {
            "sentence": "Trong khi baseline RAG cần 16GB VRAM, hệ thống mới chỉ chiếm 743 MB.",
            "citation_label": "[3]",
            "section_title": "2. Đối chuẩn Tài nguyên",
        },
    ]

    block = composer.synthesize_comparative_block(candidates)

    # Header check
    assert block.startswith("⚖️ **Đối chiếu đa chiều từ chứng cứ:**")

    # Bullet markers check
    assert "- **" in block
    assert "(`3. Kết quả Thực nghiệm`)" in block
    assert "(`2. Đối chuẩn Tài nguyên`)" in block

    # Content checks
    assert "87,5% chi phí token" in block
    assert "18.2 ms" in block
    assert "[1]" in block
    assert "[2]" in block
    assert "[3]" in block

    # Strict formatting checks
    assert "###" not in block, "Comparative block must not contain raw ### headings"
    # Ensure no single-asterisk italic
    non_bold_block = block.replace("**", "")
    assert "*" not in non_bold_block, "Comparative block must not contain single-asterisk *italic*"


def test_synthesize_comparative_block_edge_cases(composer):
    """Verifies edge cases: empty list, deduplication, missing section titles."""
    # Empty candidate list
    assert composer.synthesize_comparative_block([]) == ""

    # Candidates without section title
    cands_no_sec = [
        {
            "sentence": "Phương pháp A vượt trội hơn phương pháp B về độ chính xác F1.",
            "citation_label": "[1]",
        },
        {
            "sentence": "Do đó, năng suất xử lý dữ liệu được cải thiện đáng kể.",
            "citation_label": "[2]",
        },
    ]
    block = composer.synthesize_comparative_block(cands_no_sec)
    assert block.startswith("⚖️ **Đối chiếu đa chiều từ chứng cứ:**")
    assert "- **So sánh & Đối chiếu**:" in block
    assert "- **Tác động & Hệ quả**:" in block
    assert "###" not in block
    assert "*" not in block.replace("**", "")

    # Duplicate candidate sentences
    duplicate_cands = [
        {"sentence": "Phương pháp A vượt trội hơn B.", "citation_label": "[1]"},
        {"sentence": "phương pháp a vượt trội hơn b.", "citation_label": "[1]"},
    ]
    block_dedup = composer.synthesize_comparative_block(duplicate_cands)
    assert block_dedup.count("- **") == 1

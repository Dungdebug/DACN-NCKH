from types import SimpleNamespace

from src.web.document_analyzer import DocumentAnalyzer
from src.web.formula_example_scanner import (
    extract_example_blocks,
    extract_formula_blocks,
    format_example_blocks_markdown,
    format_formula_blocks_markdown,
)


def test_extract_inline_multi_formula_blocks_with_conditions():
    text = (
        "Tổng số nucleotide của gen: N = A + T + G + X = 2A + 2G "
        "Tỉ lệ phần trăm nucleotide: %A + %G = 50% (với %A = %T, %G = %X) "
        "Chiều dài phân tử ADN (Å): L = (N / 2) × 3,4 (1 μm = 10⁴ Å = 10³ nm) "
        "Tổng số liên kết hydro: H = 2A + 3G = N + G"
    )
    blocks = extract_formula_blocks(text, citation_label="[1]")
    assert len(blocks) == 4
    assert blocks[0].label == "Tổng số nucleotide của gen"
    assert blocks[0].expression == "N = A + T + G + X = 2A + 2G"
    assert blocks[1].condition == "với %A = %T, %G = %X"
    assert blocks[2].expression == "L = (N / 2) × 3,4"
    assert "10⁴ Å" in blocks[2].condition

    md = format_formula_blocks_markdown(blocks)
    assert "📐 **Công thức & Phạm vi áp dụng trong tài liệu:**" in md
    assert "`L = (N / 2) × 3,4`" in md
    assert "Phạm vi/Quy đổi: 1 μm = 10⁴ Å = 10³ nm" in md


def test_extract_full_example_scope_stops_at_next_section():
    text = (
        "2.2. Ví dụ minh họa đa giai đoạn\n"
        "Ví dụ 2.1 (Hòa tan kim loại trong axit): Đề bài: Hòa tan 11,2 gam Fe vào HNO3 dư thu được V lít khí NO.\n"
        "Bước 1 (Tính số mol): n(Fe) = 11,2 / 56 = 0,2 mol.\n"
        "Bước 2 (Bảo toàn e): n(NO) = 0,6 / 3 = 0,2 mol.\n"
        "Bước 3 (Tính V): V(NO) = 0,2 × 24,79 = 4,958 lít.\n"
        "2.3. Phần lý thuyết tiếp theo\n"
        "Đây là đoạn không thuộc ví dụ."
    )
    examples = extract_example_blocks(text, citation_label="[2]")
    assert len(examples) == 1
    ex = examples[0]
    assert "Ví dụ 2.1" in ex.title
    assert "11,2 gam Fe" in ex.problem
    assert len(ex.steps) == 3
    assert "4,958 lít" in ex.steps[-1]

    md = format_example_blocks_markdown(examples)
    assert "📝 **Ví dụ minh họa chi tiết trong tài liệu:**" in md
    assert "*Đề bài:*" in md
    assert "Bước 3" in md
    assert "không thuộc ví dụ" not in md


def test_document_analyzer_renders_visual_formula_and_example_blocks():
    p1 = SimpleNamespace(
        passage_id="P1",
        document_id="DOC_BIO",
        document_title="Bài 3",
        section_title="3.1. Công thức ADN",
        paragraph_index=1,
        score=12.0,
        text=(
            "Chiều dài phân tử ADN (Å): L = (N / 2) × 3,4 (1 μm = 10⁴ Å = 10³ nm) "
            "Tổng số liên kết hydro: H = 2A + 3G = N + G"
        ),
    )
    p2 = SimpleNamespace(
        passage_id="P2",
        document_id="DOC_BIO",
        document_title="Bài 3",
        section_title="3.3. Ví dụ minh họa",
        paragraph_index=2,
        score=10.5,
        text=(
            "Ví dụ 3.1 (Tính thông số gen): Đề bài: Một gen có chiều dài 5100 Å.\n"
            "Bước 1 (Tổng Nu): N = 2 × 5100 / 3,4 = 3000 Nu.\n"
            "Bước 2 (Chu kỳ xoắn): C = 3000 / 20 = 150 chu kỳ."
        ),
    )
    ans, cits = DocumentAnalyzer.find_best_grounded_answer(
        query="Công thức tính chiều dài ADN và ví dụ minh họa",
        evidence_passages=[p1, p2],
        answer_mode="detailed",
    )
    assert "📐 **Công thức & Phạm vi áp dụng trong tài liệu:**" in ans
    assert "L = (N / 2) × 3,4" in ans
    assert "📝 **Ví dụ minh họa chi tiết trong tài liệu:**" in ans
    assert "Bước 1 (Tổng Nu)" in ans
    assert "Bước 2 (Chu kỳ xoắn)" in ans
    assert len(cits) >= 1


def test_oxidizing_vs_reducing_agent_subject_disambiguation():
    p_redox = SimpleNamespace(
        passage_id="P_REDOX",
        document_id="DOC_CHEM",
        document_title="Bài 2",
        section_title="2.1. Bản chất điện tử và Nguyên tắc cân bằng",
        paragraph_index=1,
        score=17.0,
        text=(
            "Chất khử (bị oxi hóa): Là chất nhường electron, làm số oxi hóa của nguyên tố tăng lên sau phản ứng. "
            "Chất oxi hóa (bị khử): Là chất nhận electron, làm số oxi hóa của nguyên tố giảm xuống sau phản ứng."
        ),
    )

    ans_ox, _ = DocumentAnalyzer.find_best_grounded_answer(
        query="chất oxi hóa là gì",
        evidence_passages=[p_redox],
        answer_mode="balanced",
    )
    assert ans_ox.startswith("Chất oxi hóa (bị khử):")

    ans_red, _ = DocumentAnalyzer.find_best_grounded_answer(
        query="chất khử là gì",
        evidence_passages=[p_redox],
        answer_mode="balanced",
    )
    assert ans_red.startswith("Chất khử (bị oxi hóa):")

    ans_both, _ = DocumentAnalyzer.find_best_grounded_answer(
        query="chất khử và chất oxi hóa là hai chất khác nhau",
        evidence_passages=[p_redox],
        answer_mode="balanced",
    )
    assert "Chất khử (bị oxi hóa):" in ans_both
    assert "Chất oxi hóa (bị khử):" in ans_both


def test_chunker_keeps_worked_example_steps_atomic():
    from src.hybrid_qa.chunker import DocumentChunker
    from src.hybrid_qa.document_store import DocumentRecord

    raw_text = (
        "2.2. Ví dụ minh họa đa giai đoạn\n"
        "Ví dụ 2.1 (Hòa tan kim loại trong axit): Đề bài: Hòa tan hoàn toàn 11,2 gam bột Sắt (Fe) vào dung dịch Axit nitric (HNO3) loãng dư thu được V lít khí NO ở điều kiện chuẩn.\n"
        "Bước 1 (Tính số mol): M(Fe) = 56 g/mol => n(Fe) = 11,2 / 56 = 0,2 mol.\n"
        "Bước 2 (Bán phản ứng): Quá trình oxi hóa Fe -> Fe+3 + 3e và quá trình khử N+5 + 3e -> N+2.\n"
        "Bước 3 (Bảo toàn electron): n(NO) = 0,6 / 3 = 0,2 mol.\n"
        "Bước 4 (Kết quả): V(NO) = 0,2 x 24,79 = 4,958 lít."
    )
    doc = DocumentRecord(document_id="DOC_EX_ATOMIC", title="Bài 2", version=1, raw_text=raw_text)
    chunker = DocumentChunker(chunk_size=256, chunk_overlap=32, min_chunk_size=64)
    passages = chunker.chunk_document(doc)
    ex_passages = [p for p in passages if "Ví dụ 2.1" in p.text]
    assert len(ex_passages) == 1
    assert "Bước 1" in ex_passages[0].text
    assert "Bước 4" in ex_passages[0].text



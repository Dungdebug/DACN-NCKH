"""
Tests for DefenseStrategist & DefensePlan (src/brain/defense_strategist.py).
Verifies:
- Classification across all 8 defense archetypes + GENERIC (with and without accents)
- Softmax top-1 minus top-2 margin confidence
- DefensePlan target drawers, slots, and expansion terms
- Argument assembly format, bold/code adherence, omission of empty slots, and non-fabrication
"""

import pytest
import re

from src.brain.defense_strategist import (
    DefenseStrategist,
    DefensePlan,
    classify_defense_archetype,
    ARCHETYPE_NOVELTY,
    ARCHETYPE_JUSTIFICATION,
    ARCHETYPE_VALIDITY,
    ARCHETYPE_LIMITATION,
    ARCHETYPE_SCALABILITY,
    ARCHETYPE_REPRODUCIBILITY,
    ARCHETYPE_FUTURE_WORK,
    ARCHETYPE_METRIC_DEFENSE,
    ARCHETYPE_GENERIC,
)
from src.brain.section_drawer import (
    DRAWER_PROBLEM,
    DRAWER_SOLUTION,
    DRAWER_EXPERIMENT,
    DRAWER_CONCLUSION,
)


def test_classify_all_archetypes_with_and_without_accents():
    """Verify each archetype is accurately classified from Vietnamese both accented and unaccented."""
    cases = [
        (ARCHETYPE_NOVELTY, "Điểm mới và đóng góp khoa học chính của đề tài là gì?"),
        (ARCHETYPE_NOVELTY, "diem moi va dong gop cua nghien cuu nay la gi?"),
        (ARCHETYPE_JUSTIFICATION, "Tại sao lại chọn SA-CMS mà không phải Transformer thuần túy?"),
        (ARCHETYPE_JUSTIFICATION, "vi sao dung phuong phap nay thay vi baseline truyen thong?"),
        (ARCHETYPE_VALIDITY, "Tập dữ liệu benchmark có đảm bảo tính khách quan và hợp lệ không?"),
        (ARCHETYPE_VALIDITY, "tap du lieu baseline co dam bao tinh khach quan khong?"),
        (ARCHETYPE_LIMITATION, "Hạn chế và nhược điểm lớn nhất của mô hình là gì?"),
        (ARCHETYPE_LIMITATION, "han che va nhuoc diem cua de tai nay la gi?"),
        (ARCHETYPE_SCALABILITY, "Khả năng mở rộng và triển khai thực tế trên GPU ra sao?"),
        (ARCHETYPE_SCALABILITY, "kha nang mo rong va trien khai thuc te production ra sao?"),
        (ARCHETYPE_REPRODUCIBILITY, "Làm thế nào để tái lập thực nghiệm và có công khai mã nguồn không?"),
        (ARCHETYPE_REPRODUCIBILITY, "lam the nao de tai lap ket qua thuc nghiem va source code?"),
        (ARCHETYPE_FUTURE_WORK, "Hướng phát triển trong tương lai của đề tài là gì?"),
        (ARCHETYPE_FUTURE_WORK, "huong phat trien tiep theo trong tuong lai la gi?"),
        (ARCHETYPE_METRIC_DEFENSE, "Tại sao chỉ số F1 và độ trễ latency lại đánh đổi với nhau?"),
        (ARCHETYPE_METRIC_DEFENSE, "chi so do tre latency va token efficiency duoc danh gia ra sao?"),
        (ARCHETYPE_GENERIC, "Thời tiết Hà Nội hôm nay như thế nào?"),
        (ARCHETYPE_GENERIC, "thoi tiet hom nay the nao?"),
    ]

    for expected_label, query in cases:
        label, conf = classify_defense_archetype(query)
        assert label == expected_label, f"Query '{query}' expected {expected_label} but got {label}"
        assert 0.0 <= conf <= 1.0


def test_defense_plan_justification():
    """Verify DefensePlan for JUSTIFICATION has exact slots and target drawers."""
    strategist = DefenseStrategist()
    query = "Tại sao nhóm nghiên cứu lại chọn cơ chế SA-CMS mà không phải Transformer?"
    plan = strategist.plan(query)

    assert plan.archetype == ARCHETYPE_JUSTIFICATION
    assert plan.confidence > 0.5
    assert plan.argument_slots == ["claim", "alternative", "evidence", "trade_off"]
    assert DRAWER_SOLUTION in plan.target_drawers
    assert DRAWER_EXPERIMENT in plan.target_drawers
    assert DRAWER_CONCLUSION in plan.target_drawers
    assert len(plan.expansion_terms) > 0


def test_defense_plan_limitation():
    """Verify DefensePlan for LIMITATION has exact slots and target drawers."""
    strategist = DefenseStrategist()
    query = "Những hạn chế và điểm yếu còn tồn tại trong nghiên cứu này là gì?"
    plan = strategist.plan(query)

    assert plan.archetype == ARCHETYPE_LIMITATION
    assert plan.confidence > 0.5
    assert plan.argument_slots == ["acknowledge", "evidence", "mitigation"]
    assert plan.target_drawers == [DRAWER_CONCLUSION, DRAWER_EXPERIMENT]
    assert "hạn chế" in plan.expansion_terms


def test_assemble_argument_format_and_slots():
    """Verify assemble_argument produces strict markdown format with bold and code only."""
    strategist = DefenseStrategist()
    plan = DefensePlan(
        archetype=ARCHETYPE_JUSTIFICATION,
        confidence=0.85,
        target_drawers=[DRAWER_SOLUTION, DRAWER_EXPERIMENT, DRAWER_CONCLUSION],
        argument_slots=["claim", "alternative", "evidence", "trade_off"],
        expansion_terms=["SA-CMS", "Transformer"],
    )

    candidates = [
        {
            "sentence": "SA-CMS đề xuất cơ chế phân tầng bộ nhớ liên cấp nhằm giải quyết nghẽn bộ nhớ.",
            "citation_label": "1",
            "section_title": "Phương pháp và Kiến trúc đề xuất",
        },
        {
            "sentence": "So với Transformer truyền thống, mô hình cũ tiêu tốn O(N^2) tài nguyên tính toán.",
            "citation_label": "[2]",
            "section_title": "Nghiên cứu liên quan và baseline",
        },
        {
            "sentence": "Kết quả thực nghiệm cho thấy phương pháp đạt mức giảm 42% VRAM và tăng tốc 1.8x.",
            "citation_label": "3",
            "section_title": "Thực nghiệm và Kết quả đo đạc",
        },
        {
            "sentence": "Tuy nhiên, sự đánh đổi là chi phí thời gian khởi tạo cấu trúc chỉ mục ban đầu.",
            "citation_label": "[4]",
            "section_title": "Thảo luận và Đánh đổi",
        },
    ]

    output = strategist.assemble_argument(plan, candidates)

    # 1. Starts with exact header
    assert output.startswith("🎓 **Lập luận bảo vệ:**")

    # 2. Contains all 4 slot bullets
    assert "- **Claim:**" in output
    assert "- **Alternative:**" in output
    assert "- **Evidence:**" in output
    assert "- **Trade-off:**" in output

    # 3. Citation labels properly formatted
    assert "[1]" in output
    assert "[2]" in output
    assert "[3]" in output
    assert "[4]" in output

    # 4. Strict markdown: No ### headings
    assert "###" not in output
    assert "##" not in output

    # 5. No single asterisk italic syntax (*italic*)
    # Match any isolated * that is not part of **
    non_bold_asterisks = re.findall(r"(?<!\*)\*(?!\*)", output)
    assert len(non_bold_asterisks) == 0, f"Found italic or stray asterisks: {non_bold_asterisks}"


def test_assemble_argument_omits_empty_slots():
    """Verify that slots without supporting evidence are omitted (never fabricated)."""
    strategist = DefenseStrategist()
    plan = DefensePlan(
        archetype=ARCHETYPE_JUSTIFICATION,
        confidence=0.9,
        target_drawers=[DRAWER_SOLUTION, DRAWER_EXPERIMENT, DRAWER_CONCLUSION],
        argument_slots=["claim", "alternative", "evidence", "trade_off"],
    )

    # Only provide evidence for claim and evidence (no alternative, no trade_off)
    candidates = [
        {
            "sentence": "Đề xuất kiến trúc SA-CMS tối ưu hóa chi phí token.",
            "citation_label": "[1]",
            "section_title": "Kiến trúc mô hình",
        },
        {
            "sentence": "Kết quả thực nghiệm đo đạc đạt F1-score 91.2% trên benchmark.",
            "citation_label": "[2]",
            "section_title": "Thực nghiệm",
        },
    ]

    output = strategist.assemble_argument(plan, candidates)

    assert "- **Claim:**" in output
    assert "- **Evidence:**" in output
    # Omitted slots must not appear
    assert "- **Alternative:**" not in output
    assert "- **Trade-off:**" not in output


def test_assemble_argument_empty_candidates():
    """Verify empty candidates returns header without crashing."""
    strategist = DefenseStrategist()
    plan = DefensePlan(
        archetype=ARCHETYPE_GENERIC,
        confidence=0.0,
        argument_slots=["claim", "evidence"],
    )
    output = strategist.assemble_argument(plan, [])
    assert output == "🎓 **Lập luận bảo vệ:**"

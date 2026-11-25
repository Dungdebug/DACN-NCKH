"""
Tests for T8 (w7-reason-loop): src/reasoning/reasoning_loop.py
"""

from itertools import combinations
import math
from typing import Any, Dict, List, Optional, Set

import pytest

from src.hybrid_qa.grounding_critic import CriticReport
from src.reasoning.reasoning_loop import (
    DROPPED_FACET_NOTE,
    FacetEvidence,
    ReasoningStep,
    ReasoningTrace,
    UnifiedReasoner,
    greedy_submodular_select,
)


# ---------------------------------------------------------------------------
# 1. Dataclass & API Contract §5 Tests
# ---------------------------------------------------------------------------


def test_dataclasses_and_to_dict_contract() -> None:
    step = ReasoningStep(kind="premise", text="Tiền đề [1]", citations=["[1]"], verified=True)
    facet = FacetEvidence(
        facet="mechanism",
        resolution="SECTION",
        text="Cơ chế hoạt động [1]",
        citations=["[1]"],
        status="ok",
    )
    trace = ReasoningTrace(
        mode="deep_broad",
        hops=2,
        rho_hat=0.85,
        rounds=1,
        conformal={"alpha": 0.1, "accepted": True},
        steps=[step],
        facets=[facet],
        removed_claims=["Số liệu bịa 999%."],
        answer="Cơ chế hoạt động [1]",
        solver_traces=["solve_quadratic(1, -5, 6): x1=3, x2=2"],
    )

    d = trace.to_dict()
    assert d["mode"] == "deep_broad"
    assert d["hops"] == 2
    assert d["rho_hat"] == pytest.approx(0.85)
    assert d["rounds"] == 1
    assert d["conformal"] == {"alpha": 0.1, "accepted": True}
    assert d["steps"][0]["kind"] == "premise"
    assert d["facets"][0]["facet"] == "mechanism"
    assert d["facets"][0]["resolution"] == "SECTION"
    assert d["facets"][0]["status"] == "ok"
    assert d["removed_claims"] == ["Số liệu bịa 999%."]
    assert len(d["solver_traces"]) == 1

    with pytest.raises(ValueError):
        ReasoningStep(kind="invalid_kind", text="x")
    with pytest.raises(ValueError):
        FacetEvidence(facet="invalid_facet", resolution="SECTION", text="x")
    with pytest.raises(ValueError):
        FacetEvidence(facet="definition", resolution="BAD_RES", text="x")
    with pytest.raises(ValueError):
        FacetEvidence(facet="definition", resolution="SECTION", text="x", status="bad")


# ---------------------------------------------------------------------------
# 2. Proposition 7: Deep Mode Finite Termination <= 2 Rounds
# ---------------------------------------------------------------------------


def test_proposition_7_deep_mode_stops_within_2_rounds_even_if_critic_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def always_new_retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        nonlocal call_count
        call_count += 1
        return [
            {
                "passage_id": f"p_{call_count}",
                "document_id": f"doc_{call_count}",
                "section_title": "Phương pháp",
                "citation_label": f"[{call_count}]",
                "text": (
                    f"Hệ thống WaveCMS sử dụng cơ chế nén thứ {call_count} "
                    f"để xử lý ngữ cảnh dài hiệu quả."
                ),
            }
        ]

    reasoner = UnifiedReasoner(retrieve_fn=always_new_retrieve, max_deep_rounds=2)

    # Force GroundingCritic to ALWAYS report unsupported claims on every round
    monkeypatch.setattr(
        reasoner._critic,
        "verify_answer",
        lambda ans, cits: CriticReport(
            is_faithful=False,
            faithfulness_score=0.2,
            claims=[],
            unsupported_claims=["Thiếu bằng chứng bổ sung cho nhận định."],
        ),
    )

    trace = reasoner.run("Cơ chế nén của WaveCMS hoạt động như thế nào?", mode="deep")
    assert trace.mode == "deep"
    assert trace.rounds <= 2
    assert trace.rounds == 2
    assert any(s.kind == "critique" for s in trace.steps)
    assert any(s.kind == "conclusion" for s in trace.steps)


def test_proposition_7_early_stop_when_evidence_stabilizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def static_retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return [
            {
                "passage_id": "p_static",
                "document_id": "doc_1",
                "section_title": "Kiến trúc",
                "citation_label": "[1]",
                "text": "Kiến trúc WaveCMS sử dụng biến đổi Haar đa phân giải.",
            }
        ]

    reasoner = UnifiedReasoner(retrieve_fn=static_retrieve, max_deep_rounds=2)
    monkeypatch.setattr(
        reasoner._critic,
        "verify_answer",
        lambda ans, cits: CriticReport(
            is_faithful=False,
            faithfulness_score=0.1,
            claims=[],
            unsupported_claims=["Vẫn thiếu chi tiết."],
        ),
    )

    trace = reasoner.run("Kiến trúc WaveCMS là gì?", mode="deep")
    # Round 1 retrieves p_static; round 2 gets identical p_static (E_{r+1} == E_r) and halts
    assert trace.rounds <= 2


# ---------------------------------------------------------------------------
# 3. Proposition 6: Submodular Greedy >= (1 - 1/e) * OPT vs Brute Force
# ---------------------------------------------------------------------------


def _eval_cover(selected: List[Dict[str, Any]], weights: Optional[Dict[Any, float]] = None) -> float:
    covered: Set[Any] = set()
    for item in selected:
        covered |= set(item["cover"])
    if weights is None:
        return float(len(covered))
    return sum(weights.get(e, 1.0) for e in covered)


@pytest.mark.parametrize(
    "candidates,budget,weights",
    [
        # Classic tight-ish greedy vs OPT set-cover instance
        (
            [
                {"id": "c1", "cover": {1, 2, 3, 4}},
                {"id": "c2", "cover": {1, 2, 5, 6}},
                {"id": "c3", "cover": {3, 4, 7, 8}},
                {"id": "c4", "cover": {5, 6, 9}},
                {"id": "c5", "cover": {7, 8, 10}},
            ],
            2,
            None,
        ),
        # Overlapping 5x4 facet-resolution grid simulation
        (
            [
                {"id": "a", "cover": {("def", "LEAF"), ("mech", "LEAF"), "k1"}},
                {"id": "b", "cover": {("def", "LEAF"), ("rel", "SEC"), "k2", "k3"}},
                {"id": "c", "cover": {("mech", "LEAF"), ("app", "DOC"), "k4", "k5"}},
                {"id": "d", "cover": {("lim", "CORPUS"), "k1", "k6"}},
                {"id": "e", "cover": {("rel", "SEC"), ("lim", "CORPUS"), "k7"}},
                {"id": "f", "cover": {("app", "DOC"), "k8"}},
            ],
            3,
            None,
        ),
        # Weighted coverage instance
        (
            [
                {"id": "w1", "cover": {"u1", "u2"}},
                {"id": "w2", "cover": {"u1", "u3"}},
                {"id": "w3", "cover": {"u2", "u4"}},
                {"id": "w4", "cover": {"u3", "u5"}},
            ],
            2,
            {"u1": 5.0, "u2": 2.0, "u3": 2.0, "u4": 3.0, "u5": 3.0},
        ),
    ],
)
def test_proposition_6_greedy_submodular_beats_1_minus_1_over_e_opt(
    candidates: List[Dict[str, Any]],
    budget: int,
    weights: Optional[Dict[Any, float]],
) -> None:
    greedy_sel = greedy_submodular_select(candidates, budget=budget, item_weights=weights)
    assert len(greedy_sel) <= budget

    greedy_val = _eval_cover(greedy_sel, weights)
    opt_val = max(
        _eval_cover(list(comb), weights)
        for comb in combinations(candidates, min(budget, len(candidates)))
    )
    bound = (1.0 - 1.0 / math.e) * opt_val
    assert greedy_val >= bound - 1e-9


# ---------------------------------------------------------------------------
# 4. Broad Mode: Gate Rejection -> status="dropped", No Leakage
# ---------------------------------------------------------------------------


def test_broad_mode_gate_rejection_drops_facet_without_leakage() -> None:
    docs = [
        {
            "passage_id": "p1",
            "document_id": "doc_a",
            "section_title": "Định nghĩa",
            "citation_label": "[1]",
            "resolution": "LEAF_PARAGRAPH",
            "facet": "definition",
            "text": "WaveCMS là hệ thống bộ nhớ phân cấp dùng biến đổi Haar.",
        },
        {
            "passage_id": "p2",
            "document_id": "doc_b",
            "section_title": "Hạn chế",
            "citation_label": "[2]",
            "resolution": "SECTION",
            "facet": "limitation",
            "text": "SECRET_UNGROUNDED_LIMITATION chi phí nạp ban đầu rất cao 999GB.",
        },
    ]

    def strict_gate(facet: str, candidates: List[Dict[str, Any]]) -> bool:
        return facet != "limitation"

    reasoner = UnifiedReasoner(
        retrieve_fn=lambda q, top_k=5, resolution=None: docs,
        gate_fn=strict_gate,
        broad_budget=5,
    )

    trace = reasoner.run("Tổng quan về WaveCMS", mode="broad")
    assert trace.mode == "broad"

    lim_facets = [f for f in trace.facets if f.facet == "limitation"]
    assert len(lim_facets) == 1
    assert lim_facets[0].status == "dropped"
    assert lim_facets[0].text == DROPPED_FACET_NOTE
    assert lim_facets[0].citations == []
    assert "SECRET_UNGROUNDED_LIMITATION" not in trace.answer

    def_facets = [f for f in trace.facets if f.facet == "definition"]
    assert len(def_facets) == 1
    assert def_facets[0].status == "ok"
    assert "WaveCMS" in trace.answer


# ---------------------------------------------------------------------------
# 5. Numeric Provenance Invariant (§3.5): Fabricated Numbers -> removed_claims
# ---------------------------------------------------------------------------


def test_numeric_provenance_removes_fabricated_numbers_into_removed_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = [
        {
            "passage_id": "p1",
            "document_id": "doc_1",
            "section_title": "Kết quả",
            "citation_label": "[1]",
            "text": "Hệ thống đạt độ chính xác 92% trên tập kiểm thử chuẩn.",
        }
    ]

    reasoner = UnifiedReasoner(
        retrieve_fn=lambda q, top_k=5: docs,
        max_deep_rounds=1,
    )

    # Inject a fabricated number (888.5%) into the narrative generator
    monkeypatch.setattr(
        reasoner._dag_builder,
        "compose_chain_narrative",
        lambda chain: (
            "Hệ thống đạt độ chính xác 92% trên tập kiểm thử chuẩn [1]. "
            "Ngoài ra tốc độ tăng vọt 888.5% so với đối thủ [1]."
        ),
    )

    trace = reasoner.run("Độ chính xác của hệ thống là bao nhiêu?", mode="deep")
    assert "92%" in trace.answer
    assert "888.5" not in trace.answer
    assert any("888.5" in c for c in trace.removed_claims)


# ---------------------------------------------------------------------------
# 6. PremiseValidator Refutation + Symbolic Solver Verification
# ---------------------------------------------------------------------------


def test_premise_validator_refutes_false_presupposition() -> None:
    docs = [
        {
            "passage_id": "p1",
            "document_id": "doc_1",
            "section_title": "Hiệu quả",
            "citation_label": "[1]",
            "text": "SA-CMS giúp giảm chi phí vận hành xuống 40% nhờ nén Haar.",
        }
    ]
    reasoner = UnifiedReasoner(retrieve_fn=lambda q, top_k=5: docs)

    trace = reasoner.run("Tại sao SA-CMS lại làm tăng chi phí vận hành?", mode="deep")
    assert trace.steps[0].kind == "premise"
    assert trace.steps[0].verified is True
    assert "không phù hợp với tài liệu" in trace.answer.lower()
    assert "giảm chi phí" in trace.answer.lower()


def test_symbolic_solver_verifies_quadratic_and_chemical_equations() -> None:
    reasoner = UnifiedReasoner(retrieve_fn=lambda q, top_k=5: [])

    q_trace = reasoner.run("Giải phương trình 2x^2 - 7x + 3 = 0", mode="deep")
    assert any("solve_quadratic" in t for t in q_trace.solver_traces)
    assert any("x1=3" in t and "x2=1/2" in t for t in q_trace.solver_traces)
    assert any(s.kind == "formula" and s.verified for s in q_trace.steps)

    c_trace = reasoner.run("Cân bằng phương trình Fe + O2 -> Fe2O3", mode="deep")
    assert any("4Fe + 3O2 -> 2Fe2O3" in t for t in c_trace.solver_traces)
    assert any(s.kind == "formula" and s.verified for s in c_trace.steps)


# ---------------------------------------------------------------------------
# 7. Fast & Deep-Broad Modes
# ---------------------------------------------------------------------------


def test_fast_and_deep_broad_modes() -> None:
    docs = [
        {
            "passage_id": "p1",
            "document_id": "doc_1",
            "section_title": "Cơ chế",
            "citation_label": "[1]",
            "resolution": "LEAF_PARAGRAPH",
            "facet": "mechanism",
            "text": "WaveCMS sử dụng biến đổi Haar để nén ngữ cảnh xuống 4 lần.",
        }
    ]
    reasoner = UnifiedReasoner(retrieve_fn=lambda q, top_k=5, resolution=None: docs)

    fast_trace = reasoner.run(
        "WaveCMS nén ngữ cảnh như thế nào?",
        mode="fast",
        base_routing={"rho_hat": 0.91, "conformal": {"accepted": True}},
    )
    assert fast_trace.mode == "fast"
    assert fast_trace.rounds == 1
    assert fast_trace.rho_hat == pytest.approx(0.91)
    assert fast_trace.steps[-1].kind == "conclusion"

    db_trace = reasoner.run("WaveCMS nén ngữ cảnh như thế nào?", mode="deep_broad")
    assert db_trace.mode == "deep_broad"
    assert len(db_trace.facets) == 5
    assert "4" in db_trace.answer


def test_broad_mode_symbolic_solver_math_chem_crypto_with_facets() -> None:
    docs = [
        {
            "passage_id": "p1",
            "document_id": "doc_1",
            "section_title": "Lý thuyết",
            "citation_label": "[1]",
            "resolution": "LEAF_PARAGRAPH",
            "facet": "definition",
            "text": "Phương pháp ký hiệu đảm bảo nghiệm chính xác tuyệt đối.",
        },
        {
            "passage_id": "p2",
            "document_id": "doc_2",
            "section_title": "Ứng dụng",
            "citation_label": "[2]",
            "resolution": "SECTION",
            "facet": "application",
            "text": "Áp dụng cho toán học, hóa học và mật mã cổ điển.",
        },
    ]
    reasoner = UnifiedReasoner(retrieve_fn=lambda q, top_k=5, resolution=None: docs)

    # Math: solve_quadratic
    m_trace = reasoner.run("Giải phương trình 2x^2 - 7x + 3 = 0", mode="broad")
    assert m_trace.mode == "broad"
    assert len(m_trace.facets) == 5
    assert any("solve_quadratic" in t and "x1=3" in t and "x2=1/2" in t for t in m_trace.solver_traces)
    assert any(s.kind == "formula" and s.verified for s in m_trace.steps)
    assert m_trace.answer.startswith("solve_quadratic")
    assert "Phương pháp ký hiệu" in m_trace.answer

    # Chemistry: balance_equation
    c_trace = reasoner.run("Cân bằng phương trình Fe + O2 -> Fe2O3", mode="broad")
    assert any("4Fe + 3O2 -> 2Fe2O3" in t for t in c_trace.solver_traces)
    assert any(s.kind == "formula" and s.verified for s in c_trace.steps)
    assert c_trace.answer.startswith("balance_equation")
    assert len(c_trace.facets) == 5

    # Cryptography: caesar, vigenere, affine
    caesar_trace = reasoner.run("caesar('HELLO', 3)", mode="broad")
    assert any("KHOOR" in t for t in caesar_trace.solver_traces)
    assert caesar_trace.answer.startswith("caesar")
    assert len(caesar_trace.facets) == 5

    vig_trace = reasoner.run("vigenere('ATTACKATDAWN', 'LEMON')", mode="broad")
    assert any("LXFOPVEFRNHR" in t for t in vig_trace.solver_traces)
    assert len(vig_trace.facets) == 5

    aff_trace = reasoner.run("affine('HELLO', 5, 8)", mode="broad")
    assert any("RCLLA" in t for t in aff_trace.solver_traces)
    assert len(aff_trace.facets) == 5


def test_natural_qa_number_filter_preserves_normalized_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = [
        {
            "passage_id": "p1",
            "document_id": "doc_1",
            "section_title": "Thống kê",
            "citation_label": "[1]",
            "text": "Giai đoạn 2024-2025 ghi nhận 1.000 mẫu thử với tỷ lệ 95,5% đạt chuẩn.",
        }
    ]
    reasoner = UnifiedReasoner(retrieve_fn=lambda q, top_k=5: docs, max_deep_rounds=1)
    monkeypatch.setattr(
        reasoner._dag_builder,
        "compose_chain_narrative",
        lambda chain: (
            "Trong năm 2025, hệ thống xử lý 1000 mẫu với tỷ lệ 95.5% đạt chuẩn theo báo cáo 2023 [1]."
        ),
    )

    trace = reasoner.run("Báo cáo năm 2023 ghi nhận kết quả gì?", mode="deep")
    assert "2025" in trace.answer
    assert "1000" in trace.answer
    assert "95.5%" in trace.answer
    assert "2023" in trace.answer
    assert trace.removed_claims == []

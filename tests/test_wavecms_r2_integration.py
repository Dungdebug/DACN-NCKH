"""
End-to-End Integration Tests for Sovereign Brain 7.0+ (WaveCMS-R^2, Ticket T11):
- Ingests 5 multi-domain documents (Quadratic Math, Redox Chemistry, Cryptography,
  AI Architecture, Computer Networks) via both WaveCMSEngine(routing_mode="sparse_lse")
  and FastAPI TestClient(app).
- Exercises all 4 thinking modes (fast, deep, broad, deep_broad) and verifies
  reasoning_trace schema, steps, facets, conformal, and Numeric Grounding Invariant
  (0% numeric hallucination).
- Verifies Theorem 4 (tree_argmax == flat_argmax, Banach rho_hat < 1).
- Verifies Theorem 3* (bit-exact == on term_count, corpus_df, kappa_mass_fp) after
  deleting 1 document via DELETE /api/documents/{doc_id} and WaveTreeMemory.unlearn_document.
- Verifies GET /api/comparison/dashboard reads real data from results/wavecms_r2_ablation.json.
- Cleans up all temporary documents and chat sessions created during tests.
"""

import math
from typing import Dict, List, Tuple

import pytest
import torch
from fastapi.testclient import TestClient

from src.reasoning.reasoning_loop import UnifiedReasoner
from src.reasoning.symbolic_solver import verify_numeric_provenance
from src.wave_cms.causal_attribution import MondrianConformalRiskController
from src.wave_cms.nonconformity import R3_SIGNAL_NAMES, extract_signals
from src.wave_cms.tree_memory import WaveTreeMemory, extract_sparse_terms
from src.wave_cms.wave_engine import WaveCMSEngine
from src.web.app import app


DOMAIN_DOCS: List[Tuple[str, str, str]] = [
    (
        "doc_math",
        "Toan Bac Hai W7 E2E",
        """# PHẦN 1: PHƯƠNG TRÌNH BẬC HAI
## 1.1. Công thức nghiệm
Phương trình bậc hai có dạng ax^2 + bx + c = 0 (a != 0) với biệt thức Delta = b^2 - 4ac.
Ví dụ giải phương trình 2x^2 - 7x + 3 = 0: Delta = (-7)^2 - 4 * 2 * 3 = 25 > 0, nghiệm x1 = 3 và x2 = 0.5.
""",
    ),
    (
        "doc_chem",
        "Hoa Oxi Hoa Khu W7 E2E",
        """# PHẦN 2: PHẢN ỨNG OXI HÓA - KHỬ
## 2.1. Bảo toàn electron
Chất khử nhường electron (số oxi hóa tăng), chất oxi hóa nhận electron (số oxi hóa giảm).
Ví dụ cân bằng phản ứng: 4Al + 3O2 -> 2Al2O3. Tổng số electron nhường bằng tổng số electron nhận.
""",
    ),
    (
        "doc_crypto",
        "Mat Ma Co Dien W7 E2E",
        """# PHẦN 3: MẬT MÃ HỌC CỔ ĐIỂN
## 3.1. Mã Caesar và Vigenere
Mã Caesar dịch chuyển mỗi ký tự trong bảng chữ cái 26 ký tự theo khóa k (ví dụ k = 3).
Mã Vigenere sử dụng từ khóa lặp lại để dịch chuyển đa bảng, còn Autokey dùng bản rõ làm khóa tiếp theo.
""",
    ),
    (
        "doc_ai",
        "Kien Truc AI WaveCMS W7 E2E",
        """# PHẦN 4: KIẾN TRÚC AI VÀ WAVECMS-R2
## 4.1. Bộ nhớ cây Haar Wavelet
WaveCMS-R2 lưu trữ tri thức trên cây Haar Wavelet bất biến theo phép cộng, hỗ trợ truy xuất đa bước Banach với rho_hat < 1.
Cổng Conformal Risk Control kiểm soát rủi ro sai sót dựa trên nhóm Mondrian.
""",
    ),
    (
        "doc_net",
        "Mang May Tinh TCP IP W7 E2E",
        """# PHẦN 5: MẠNG MÁY TÍNH TCP/IP
## 5.1. Bắt tay 3 bước TCP
Giao thức TCP thiết lập kết nối qua quy trình bắt tay 3 bước: SYN, SYN-ACK, ACK trên cổng 443 cho HTTPS.
Giao thức DNS phân giải tên miền sang địa chỉ IP trước khi gửi gói tin qua tầng mạng.
""",
    ),
]

VALID_STEP_KINDS = {"premise", "mechanism", "formula", "critique", "conclusion"}
VALID_FACETS = {"definition", "mechanism", "relation_comparison", "application", "limitation"}
VALID_RESOLUTIONS = {"LEAF_PARAGRAPH", "SECTION", "DOCUMENT", "CORPUS"}


def _flat_sparse_lse_reference(engine: WaveCMSEngine, h: torch.Tensor, query_text: str) -> Dict[str, float]:
    tree = engine.tree_memory
    router = engine.readout_engine.router
    leaves = tree.get_leaves()
    n_leaves = len(leaves)
    query_terms = list(dict.fromkeys(extract_sparse_terms(query_text)))
    idf_map = {t: max(0.0, float(tree.corpus_df.idf(t, n_leaves))) for t in query_terms}
    avgdl = max(1.0, sum(lf.doc_len for lf in leaves) / max(1, n_leaves))
    k1, b = 1.2, 0.75
    tau = max(1e-4, router.temperature)
    tau_s = max(1e-4, router.tau_sparse)
    q_proj = router.w_q(h.squeeze().to(router.w_q.weight.device))

    logits: List[float] = []
    for lf in leaves:
        sparse_sum = 0.0
        denom_norm = k1 * (1.0 - b + b * (lf.doc_len / avgdl))
        for t in query_terms:
            tf = lf.term_tf.get(t, 0)
            if tf > 0:
                sparse_sum += idf_map[t] * (tf / (tf + denom_norm))
        key = router._get_node_key(lf).to(q_proj.device)
        dense_score = float(torch.dot(q_proj, key).item()) / tau
        logits.append((sparse_sum / tau_s) + dense_score)

    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    z_sum = sum(exps)
    return {lf.node_id: e / z_sum for lf, e in zip(leaves, exps)}


def _assert_trace_schema(trace: dict, expected_mode: str) -> None:
    assert isinstance(trace, dict)
    assert trace.get("mode") == expected_mode
    assert isinstance(trace.get("hops"), int) and trace["hops"] >= 1
    assert isinstance(trace.get("rho_hat"), (int, float))
    assert isinstance(trace.get("rounds"), int) and trace["rounds"] >= 1
    conformal = trace.get("conformal")
    assert isinstance(conformal, dict)
    assert {"score", "group", "lambda_g"}.issubset(conformal.keys())
    assert isinstance(trace.get("steps"), list)
    assert isinstance(trace.get("facets"), list)
    assert isinstance(trace.get("removed_claims"), list)


def test_engine_sparse_lse_theorem4_and_theorem3_star_bit_exact():
    """
    Tests WaveCMSEngine(routing_mode="sparse_lse") on 5 multi-domain documents:
    - Theorem 4: tree_argmax == flat_argmax and Banach rho_hat < 1.
    - Theorem 3*: unlearning 1 doc yields bit-exact (==) term_count, corpus_df, kappa_mass_fp
      compared to a clean tree built from scratch with only the remaining 4 docs.
    """
    engine = WaveCMSEngine(r=8, dim=64, tau=0.25, routing_mode="sparse_lse")
    for doc_id, title, text in DOMAIN_DOCS:
        engine.ingest_document(doc_id=doc_id, title=title, sections=text)

    domain_queries = [
        "Giải phương trình bậc hai 2x^2 - 7x + 3 = 0 và tính Delta",
        "Bảo toàn electron trong phản ứng oxi hóa khử 4Al + 3O2 -> 2Al2O3",
        "Mã Caesar dịch chuyển 26 ký tự và mã Vigenere",
        "WaveCMS-R2 cây Haar Wavelet và Conformal Risk Control",
        "Quy trình bắt tay 3 bước TCP SYN SYN-ACK ACK trên cổng 443",
    ]
    for q in domain_queries:
        eq_res = engine.readout_engine.readout(query=q, tree=engine.tree_memory, query_text=q)
        assert eq_res.is_contractive is True
        assert eq_res.contraction_factor < 1.0

        route_res = engine.readout_engine.router.route(
            eq_res.fixed_point, engine.tree_memory, query_text=q, memory_tree=engine.tree_memory
        )
        flat_alpha = _flat_sparse_lse_reference(engine, eq_res.fixed_point, q)
        tree_argmax = max(route_res.leaf_weights.items(), key=lambda kv: kv[1])[0]
        flat_argmax = max(flat_alpha.items(), key=lambda kv: kv[1])[0]
        assert tree_argmax == flat_argmax

        q_res = engine.query(question=q)
        assert q_res.is_contractive is True
        assert q_res.contraction_factor < 1.0

    # Theorem 3*: Unlearn doc_net on WaveTreeMemory and compare bit-for-bit against 4-doc clean tree
    target_delete = "doc_net"
    unlearn_res = engine.tree_memory.unlearn_document(target_delete)
    assert unlearn_res["certified_exact"] is True

    clean_engine = WaveCMSEngine(r=8, dim=64, tau=0.25, routing_mode="sparse_lse")
    for doc_id, title, text in DOMAIN_DOCS:
        if doc_id != target_delete:
            clean_engine.ingest_document(doc_id=doc_id, title=title, sections=text)

    assert engine.tree_memory.corpus_df == clean_engine.tree_memory.corpus_df
    assert engine.tree_memory.root.term_count == clean_engine.tree_memory.root.term_count
    assert engine.tree_memory.root.kappa_mass_fp == clean_engine.tree_memory.root.kappa_mass_fp

    for doc_id, _, _ in DOMAIN_DOCS:
        if doc_id == target_delete:
            continue
        nodes_u = engine.tree_memory.documents[doc_id].get_subtree_nodes()
        nodes_c = clean_engine.tree_memory.documents[doc_id].get_subtree_nodes()
        assert len(nodes_u) == len(nodes_c)
        for nu, nc in zip(nodes_u, nodes_c):
            assert nu.term_count == nc.term_count
            assert nu.kappa_mass_fp == nc.kappa_mass_fp


def test_e2e_api_5_domains_4_modes_unlearn_and_dashboard():
    """
    End-to-end FastAPI TestClient(app) verification:
    1. Ingest 5 multi-domain documents via POST /api/documents/raw.
    2. Query across all 4 thinking modes (fast, deep, broad, deep_broad), checking
       reasoning_trace, steps, facets, conformal, and 0% numeric hallucination.
    3. Delete 1 document via DELETE /api/documents/{doc_id} and verify exact unlearning.
    4. Verify GET /api/comparison/dashboard loads real data from results/wavecms_r2_ablation.json.
    5. Clean up all temporary documents and sessions.
    """
    client = TestClient(app)
    created_doc_ids: List[str] = []
    created_session_ids: List[str] = []

    try:
        for _, title, text in DOMAIN_DOCS:
            res = client.post("/api/documents/raw", json={"title": title, "text": text})
            assert res.status_code == 200
            created_doc_ids.append(res.json()["document_id"])

        sess_res = client.post(
            "/api/chat/sessions",
            json={"title": "W7 E2E 5-Domain Session", "document_ids": created_doc_ids},
        )
        assert sess_res.status_code == 200
        sid = sess_res.json()["session_id"]
        created_session_ids.append(sid)

        evidence_corpus = [text for _, _, text in DOMAIN_DOCS]
        mode_queries = [
            ("fast", "Quy trình bắt tay 3 bước TCP sử dụng các gói tin nào và cổng bao nhiêu?"),
            ("deep", "Giải phương trình bậc hai 2x^2 - 7x + 3 = 0 và tính Delta"),
            ("broad", "Chất khử, chất oxi hóa và nguyên tắc bảo toàn electron hoạt động ra sao?"),
            ("deep_broad", "Mã Caesar dịch chuyển 26 ký tự với khóa k = 3 và mã Vigenere khác nhau thế nào?"),
        ]

        for mode, query in mode_queries:
            resp = client.post(
                f"/api/chat/sessions/{sid}/messages",
                json={"query": query, "answer_mode": "balanced", "thinking_mode": mode},
            )
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["refused"] is False
            trace = payload["reasoning_trace"]
            _assert_trace_schema(trace, mode)

            if mode in {"deep", "deep_broad"}:
                assert len(trace["steps"]) >= 1
                assert all(s["kind"] in VALID_STEP_KINDS for s in trace["steps"])
                assert any(s.get("verified") is True for s in trace["steps"])

            if mode in {"broad", "deep_broad"}:
                assert len(trace["facets"]) == 5
                for f in trace["facets"]:
                    assert f["facet"] in VALID_FACETS
                    assert f["resolution"] in VALID_RESOLUTIONS
                    assert f["status"] in {"ok", "dropped"}

            # Numeric Grounding Invariant (0% numeric hallucination)
            # Strip trailing citation/section boilerplate lines when checking answer numbers
            solver_traces = trace.get("solver_traces") or []
            ok_num, ungrounded = verify_numeric_provenance(
                answer=payload["answer"],
                evidence_texts=evidence_corpus + [query],
                solver_trace=[str(t) for t in solver_traces],
            )
            assert ok_num is True, f"Mode {mode} produced ungrounded numbers: {ungrounded}"

        # Delete 1 document via DELETE /api/documents/{doc_id}
        doc_to_delete = created_doc_ids.pop()
        del_res = client.delete(f"/api/documents/{doc_to_delete}")
        assert del_res.status_code == 200
        del_data = del_res.json()
        assert del_data["success"] is True
        if del_data.get("wave_cms_unlearning") is not None:
            assert del_data["wave_cms_unlearning"].get("discrepancy", 0.0) < 1e-12

        # Verify GET /api/comparison/dashboard reads real data from results/wavecms_r2_ablation.json
        dash_res = client.get("/api/comparison/dashboard")
        assert dash_res.status_code == 200
        dash = dash_res.json()
        ablation_w7 = dash.get("ablation_w7")
        assert isinstance(ablation_w7, dict)
        assert ablation_w7.get("schema_version") == "wavecms_r2_ablation_w7_v1"
        pooled = ablation_w7.get("pooled_test_results", {})
        assert "W0_standard_bm25_rag" in pooled
        assert "W6_wavecms_r2_full" in pooled
        assert "LOO_W6_minus_r3_signals" in pooled
        w6_pooled = pooled["W6_wavecms_r2_full"]
        assert "aurc_pct" in w6_pooled
        assert "ans_acc_at_halluc_15_pct" in w6_pooled

        per_seed = ablation_w7.get("per_seed_summary", {})
        assert isinstance(per_seed, dict)
        assert "W6_wavecms_r2_full" in per_seed
        assert "LOO_W6_minus_r3_signals" in per_seed

        assert isinstance(ablation_w7.get("pmi_r3_kill_criterion"), dict)
        rs60 = ablation_w7.get("reasoning_suite_60_problems", {})
        assert rs60.get("broad", {}).get("accuracy_pct") == 100.0

        ablation_rows = ablation_w7.get("rows")
        assert isinstance(ablation_rows, list) and len(ablation_rows) >= 14
        variants_seen = {r["variant"]: r for r in ablation_rows}
        assert "W0_standard_bm25_rag" in variants_seen
        assert "W6_wavecms_r2_full" in variants_seen
        assert any(k.startswith("LOO_") for k in variants_seen)
        assert variants_seen["W6_wavecms_r2_full"]["overall_accuracy_pct"] == pooled["W6_wavecms_r2_full"]["overall_accuracy_pct"]
        assert isinstance(variants_seen["W6_wavecms_r2_full"]["per_seed_overall_str"], str)

        p1_w7 = dash.get("pillar1_sparse_lse_w7")
        assert isinstance(p1_w7, dict)
        p1_rows = p1_w7.get("rows")
        assert isinstance(p1_rows, list) and len(p1_rows) == 5
        assert [r["n"] for r in p1_rows] == [0, 5, 10, 20, 64]
        row_64 = next(r for r in p1_rows if r["n"] == 64)
        assert row_64["sparse_lse_hit_at_1_pct"] == 70.0
        assert row_64["mean_key_hit_at_1_pct"] == 0.0
    finally:
        for sid in created_session_ids:
            client.delete(f"/api/chat/sessions/{sid}")
        for did in created_doc_ids:
            client.delete(f"/api/documents/{did}")


def test_r3_nonconformity_12_signals_and_mondrian_crc():
    """
    Verifies WaveCMS-R3 nonconformity extraction (12 signals in R3_SIGNAL_NAMES)
    and MondrianConformalRiskController with forward_select_r3=True,
    calibrate_groups(..., fallback_to_global=True), and audit_resplit_risk(..., n_trials=100).
    """
    assert len(R3_SIGNAL_NAMES) == 12
    evidence = [
        {
            "passage_id": "p1",
            "text": "Phuong trinh bac hai 2x^2 - 7x + 3 = 0 co Delta = 25 va nghiem x1 = 3, x2 = 0.5.",
            "score": 0.92,
            "bm25_score": 4.2,
        },
        {
            "passage_id": "p2",
            "text": "Cong thuc nghiem tong quat cua phuong trinh bac hai la ax^2 + bx + c = 0.",
            "score": 0.61,
            "bm25_score": 2.1,
        },
    ]
    sigs = extract_signals(
        query="Tinh Delta va nghiem cua 2x^2 - 7x + 3 = 0",
        candidate_answer="Delta = 25, x1 = 3, x2 = 0.5",
        evidence=evidence,
        routing={"weights": [0.92, 0.61], "bm25_scores": [4.2, 2.1]},
        pmi=1.25,
        bm25_best_sentence="Phuong trinh bac hai 2x^2 - 7x + 3 = 0 co Delta = 25 va nghiem x1 = 3, x2 = 0.5.",
        idf_map={"delta": 2.5, "nghiem": 1.8, "phuong": 1.2, "trinh": 1.2},
        analyzer_conf=0.88,
        sentence_scores=[0.95, 0.42, 0.15],
    )
    for name in R3_SIGNAL_NAMES:
        assert name in sigs
        assert isinstance(sigs[name], (int, float))
        assert math.isfinite(float(sigs[name]))

    def _make_row(idx: int, group: str, y: int) -> dict:
        base = 0.85 if y == 1 else 0.25
        jitter = ((idx % 5) - 2) * 0.03
        row_sigs = {name: max(0.0, min(1.0, base + jitter)) for name in R3_SIGNAL_NAMES}
        row_sigs["negation_mismatch"] = 0.05 if y == 1 else 0.75
        row_sigs["pmi_missing"] = 0.0
        return {"signals": row_sigs, "y": y, "group": group}

    fit_rows = []
    for i in range(40):
        g = "VI-encyclopedia" if i % 2 == 0 else "EN-encyclopedia"
        fit_rows.append(_make_row(i, g, 1 if i % 3 != 0 else 0))

    cal_rows = (
        [_make_row(i, "VI-encyclopedia", 1 if i % 4 != 0 else 0) for i in range(16)]
        + [_make_row(i, "EN-encyclopedia", 1 if i % 4 != 0 else 0) for i in range(16)]
        + [_make_row(i, "EN-reading", 1 if i % 2 == 0 else 0) for i in range(3)]
    )
    pooled_rows = cal_rows + [
        _make_row(i + 100, "VI-encyclopedia" if i % 2 == 0 else "EN-encyclopedia", 1 if i % 3 != 0 else 0)
        for i in range(25)
    ]

    ctrl = MondrianConformalRiskController(epsilon=0.10, use_global_fallback=True)
    fit_res = ctrl.fit_weights(fit_rows, forward_select_r3=True)
    assert fit_res["status"] == "SUCCESS"
    assert "selected_signal_names" in fit_res
    assert "kept_r3_signals" in fit_res
    assert "killed_r3_signals" in fit_res
    assert len(fit_res["selected_signal_names"]) >= 8

    cal_stats = ctrl.calibrate_groups(cal_rows, eps_by_group=0.10, fallback_to_global=True)
    assert ctrl.is_calibrated is True
    assert {"VI-encyclopedia", "EN-encyclopedia", "EN-reading"}.issubset(cal_stats.keys())
    assert cal_stats["EN-reading"]["fallback_to_global"] is True
    for stats in cal_stats.values():
        assert isinstance(stats["lambda_g"], float)

    audit = ctrl.audit_resplit_risk(
        pooled_rows, eps=0.10, cal_fraction=0.25, n_trials=100, seed=42, fallback_to_global=True
    )
    assert {"VI-encyclopedia", "EN-encyclopedia", "EN-reading"}.issubset(audit.keys())
    for group_report in audit.values():
        assert {"mean_risk", "std_risk", "se_risk", "ci95_upper", "exceeds_bound"}.issubset(
            group_report.keys()
        )


def test_r3_unified_reasoner_broad_symbolic_and_facets():
    """
    Verifies UnifiedReasoner in 'broad' mode solves symbolic problems (Math, Chemistry,
    Cryptography) while still returning multi-perspective facets.
    """
    multi_facet_docs = [
        {
            "passage_id": "f_def",
            "document_id": "doc_sym",
            "section_title": "Definition",
            "citation_label": "[1]",
            "resolution": "LEAF_PARAGRAPH",
            "facet": "definition",
            "text": "Phuong trinh bac hai va phan ung oxi hoa khu la cac bai toan ky hieu co ban.",
        },
        {
            "passage_id": "f_mech",
            "document_id": "doc_sym",
            "section_title": "Mechanism",
            "citation_label": "[2]",
            "resolution": "SECTION",
            "facet": "mechanism",
            "text": "Co che giai gom tinh biet thuc Delta, can bang nguyen to va dich chuyen ky tu Caesar.",
        },
        {
            "passage_id": "f_app",
            "document_id": "doc_sym",
            "section_title": "Application",
            "citation_label": "[3]",
            "resolution": "DOCUMENT",
            "facet": "application",
            "text": "Ung dung trong toan hoc, hoa hoc vo co va mat ma co dien.",
        },
    ]

    reasoner = UnifiedReasoner(
        retrieve_fn=lambda q, top_k=5, resolution=None: multi_facet_docs[:top_k],
        broad_budget=5,
    )

    symbolic_cases = [
        ("Giai phuong trinh 2x^2 - 7x + 3 = 0", ["x1=3", "x2=1/2"]),
        ("Can bang phuong trinh hoa hoc Al + O2 -> Al2O3", ["4Al", "3O2", "2Al2O3"]),
        ("Giai ma caesar('KHOOR', 3, decrypt=True)", ["HELLO"]),
    ]

    for query, expected_tokens in symbolic_cases:
        trace = reasoner.run(query=query, mode="broad")
        assert trace.mode == "broad"
        assert len(trace.solver_traces) >= 1
        for tok in expected_tokens:
            assert tok in trace.answer, f"Expected '{tok}' in broad answer: {trace.answer}"
        assert len(trace.facets) == 5
        assert any(f.status == "ok" for f in trace.facets)
        for f in trace.facets:
            assert f.facet in VALID_FACETS
            assert f.resolution in VALID_RESOLUTIONS
            assert f.status in {"ok", "dropped"}

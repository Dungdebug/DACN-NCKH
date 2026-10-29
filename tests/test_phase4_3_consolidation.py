"""
Unit Tests for Phase 4.3 Benchmark Consolidation & Result Artifacts
Validates:
- Master experiment manifest
- All 8 master CSV tables
- All 10 figures
- All 5 LaTeX tables
- Benchmark matrix
- Anomaly, readiness, and contradiction audits
- Data provenance manifest
- Research status dashboard
"""

import json
import csv
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
RES_DIR = ROOT_DIR / "results" / "phase4_3"
DOCS_DIR = ROOT_DIR / "docs"
TABLES_DIR = RES_DIR / "tables"
FIG_DIR = RES_DIR / "figures"
LATEX_DIR = RES_DIR / "latex_tables"

def test_master_manifest_integrity():
    manifest_path = RES_DIR / "master_experiment_manifest.json"
    assert manifest_path.exists(), "Master manifest must exist"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "meta" in data
    assert "methods_catalog" in data
    assert "datasets_catalog" in data
    assert "experiments" in data
    assert len(data["experiments"]) >= 40
    # Ensure B3 is marked EXCLUDED
    b3_entries = [e for e in data["experiments"] if e["method"] == "B3"]
    assert len(b3_entries) == 1
    assert b3_entries[0]["status"] == "EXCLUDED"

def test_master_tables_existence():
    expected_tables = [
        "01_rq1_summary.csv",
        "02_rq2_summary.csv",
        "03_rq3_summary.csv",
        "04_rq4_status.csv",
        "05_rq5_cost.csv",
        "06_method_dataset_matrix.csv",
        "07_checkpoint_matrix.csv",
        "08_missing_experiments.csv"
    ]
    for tbl in expected_tables:
        tbl_path = TABLES_DIR / tbl
        assert tbl_path.exists(), f"Table {tbl} must exist"
        with open(tbl_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) > 0, f"Table {tbl} must not be empty"
            # Ensure source_artifact column is present
            assert "source_artifact" in reader[0], f"Table {tbl} must have source_artifact reference"

def test_rq1_table_content():
    path = TABLES_DIR / "01_rq1_summary.csv"
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    methods = set(r["method"] for r in rows)
    assert methods == {"B1", "B4", "B5", "P1"}
    datasets = set(r["dataset"] for r in rows)
    assert datasets == {"MK-NIAH", "QASPER", "LongHealth"}

def test_rq2_table_content():
    path = TABLES_DIR / "02_rq2_summary.csv"
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pooled = next(r for r in rows if "Pooled" in r["comparison_or_ablation"])
    assert int(pooled["n_observations"]) == 90
    assert float(pooled["mean_diff"]) == pytest.approx(0.0093, abs=1e-4)
    assert float(pooled["t_statistic"]) == pytest.approx(0.8203, abs=1e-4)
    # Verify A1 and A3 status
    a1 = next(r for r in rows if "A1" in r["comparison_or_ablation"])
    a3 = next(r for r in rows if "A3" in r["comparison_or_ablation"])
    assert a1["status"] == "NEED_EXTERNAL_GPU"
    assert a3["status"] == "NEED_EXTERNAL_GPU"

def test_rq3_table_content():
    path = TABLES_DIR / "03_rq3_summary.csv"
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    methods = [r["method"] for r in rows]
    assert set(methods) == {"B1", "B2", "B5", "P1", "P2"}
    b2 = next(r for r in rows if r["method"] == "B2")
    p2 = next(r for r in rows if r["method"] == "P2")
    assert float(b2["correct_refusal_pct"]) == 76.0
    assert float(p2["correct_refusal_pct"]) == 76.0
    assert float(b2["false_refusal_pct"]) == 0.0
    assert float(p2["false_refusal_pct"]) == 0.0

def test_benchmark_matrix():
    path = RES_DIR / "benchmark_matrix.csv"
    assert path.exists()
    allowed_statuses = {"COMPLETED", "PARTIAL", "NEED_EXTERNAL_GPU", "NOT_RUN", "EXCLUDED"}
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected_rows = ["B1", "B2", "B3", "B4", "B5", "P1", "P2", "A1", "A2", "A3"]
    actual_rows = [r["Method"] for r in rows]
    for exp in expected_rows:
        assert exp in actual_rows
    for r in rows:
        assert r["Status"] in allowed_statuses
        assert r["RQ1"] in allowed_statuses
        assert r["RQ2"] in allowed_statuses
        assert r["RQ3"] in allowed_statuses
        assert r["RQ4"] in allowed_statuses
        assert r["RQ5"] in allowed_statuses

def test_figures_existence():
    expected_figs = [
        "fig_rq1_qasper.png",
        "fig_rq1_longhealth.png",
        "fig_rq1_mkniah.png",
        "fig_rq2_p1_vs_b5.png",
        "fig_rq3_f1.png",
        "fig_rq3_faithfulness.png",
        "fig_rq3_refusal.png",
        "fig_rq5_latency.png",
        "fig_rq5_vram.png",
        "fig_checkpoint_size.png"
    ]
    for fig in expected_figs:
        fig_path = FIG_DIR / fig
        assert fig_path.exists(), f"Figure {fig} must exist"
        assert fig_path.stat().st_size > 1000, f"Figure {fig} must not be empty"

def test_latex_tables_existence():
    expected_tex = [
        "rq1_main.tex",
        "rq2_comparison.tex",
        "rq3_context_evicted.tex",
        "rq5_cost.tex",
        "benchmark_status.tex"
    ]
    for tex in expected_tex:
        tex_path = LATEX_DIR / tex
        assert tex_path.exists(), f"LaTeX file {tex} must exist"
        assert tex_path.stat().st_size > 100, f"LaTeX file {tex} must not be empty"

def test_provenance_manifest():
    prov_path = RES_DIR / "data_provenance.json"
    assert prov_path.exists()
    with open(prov_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["meta"]["unverified_metrics_count"] == 0
    assert data["meta"]["orphan_metrics_count"] == 0
    for rec in data["provenance_records"]:
        assert rec["verification_flag"] == "VERIFIED"

def test_anomaly_and_readiness_audits():
    assert (RES_DIR / "rq3_anomaly_audit.json").exists()
    assert (DOCS_DIR / "phase4_3_rq3_anomaly_audit.md").exists()
    assert (RES_DIR / "rq4_external_readiness.json").exists()
    assert (DOCS_DIR / "phase4_3_rq4_external_readiness.md").exists()
    assert (RES_DIR / "ablation_external_readiness.json").exists()
    assert (RES_DIR / "contradiction_report.json").exists()
    assert (DOCS_DIR / "phase4_3_contradiction_report.md").exists()
    assert (DOCS_DIR / "phase4_3_research_status.md").exists()

def test_research_status_dashboard_keywords():
    doc_path = DOCS_DIR / "phase4_3_research_status.md"
    content = doc_path.read_text(encoding="utf-8")
    assert "| RQ | Current Status | Evidence Available | Missing | External GPU |" in content
    assert "RQ1" in content and "COMPLETED" in content
    assert "RQ2" in content and "PARTIAL" in content
    assert "RQ3" in content and "COMPLETED" in content
    assert "RQ4" in content and "NEED_EXTERNAL_GPU" in content
    assert "RQ5" in content and "COMPLETED" in content
    # Ensure disallowed evaluation terms are not present in table statuses
    assert "| PASS |" not in content
    assert "| FAIL |" not in content
    assert "| CONFIRMED |" not in content
    assert "| REJECTED |" not in content

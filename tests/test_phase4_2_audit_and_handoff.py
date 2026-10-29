"""
Tests for Phase 4.2 Scientific Audits, External GPU Packages, and Hand-off Validator.
"""

import json
from pathlib import Path
import pytest
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent

def test_rq2_statistical_audit_integrity():
    audit_file = ROOT_DIR / "results" / "phase4_2" / "rq2_statistical_audit.json"
    assert audit_file.exists(), f"Missing {audit_file}"
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "ten_audit_questions" in data
    assert len(data["ten_audit_questions"]) == 10
    
    reconciled = data["reconciled_statistics"]["pooled_all_90_items"]
    assert reconciled["n_observations"] == 90
    assert reconciled["mean_difference"] == pytest.approx(0.0093, abs=1e-4)
    assert reconciled["t_statistic"] == pytest.approx(0.8203, abs=1e-3)
    assert reconciled["t_pvalue"] == pytest.approx(0.4143, abs=1e-3)
    assert reconciled["wilcoxon_stat"] == pytest.approx(31.0, abs=1e-1)

def test_rq1_data_audit_integrity():
    audit_file = ROOT_DIR / "results" / "phase4_2" / "rq1_data_audit.json"
    assert audit_file.exists(), f"Missing {audit_file}"
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    benchmarks = data["benchmarks"]
    assert benchmarks["mkniah"]["sample_count"] == 100
    assert benchmarks["mkniah"]["missing_items"] == 0
    assert benchmarks["qasper"]["document_count"] == 10
    assert benchmarks["qasper"]["missing_items"] == 0
    assert benchmarks["longhealth"]["questions_count"] == 20
    assert benchmarks["longhealth"]["missing_items"] == 0

def test_rq3_consistency_audit_integrity():
    audit_file = ROOT_DIR / "results" / "phase4_2" / "rq3_consistency_audit.json"
    assert audit_file.exists(), f"Missing {audit_file}"
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    alignment = data["b2_vs_p2_alignment_per_seed"]
    for seed_str in ["42", "43", "44"]:
        assert alignment[seed_str]["refusal_decision_match_pct"] == 100.0
        assert alignment[seed_str]["citation_set_match_pct"] == 100.0

def test_external_artifacts_validator_passes():
    validator_script = ROOT_DIR / "scripts" / "validate_phase4_2_external_artifacts.py"
    assert validator_script.exists()
    
    res = subprocess.run([sys.executable, str(validator_script)], capture_output=True, text=True)
    assert res.returncode == 0, f"Validator failed: {res.stdout}\n{res.stderr}"
    assert "REPRODUCIBILITY VALIDATION PASSED" in res.stdout

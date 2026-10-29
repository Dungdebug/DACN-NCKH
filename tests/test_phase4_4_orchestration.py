"""
Unit Tests for Phase 4.4 External GPU Execution Integration & Result Ingestion
Validates:
- external_package_audit.json
- One-command runner scripts in external_gpu/phase4_4/
- Preflight hardware validation logic
- A1, A3, RQ4 dry-run execution contracts
- Result ingestion structure
- Phase 4.4 progress documentation
"""

import sys
import json
import subprocess
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
PHASE_4_4_DIR = ROOT_DIR / "external_gpu" / "phase4_4"
RES_4_4_DIR = ROOT_DIR / "results" / "phase4_4"
DOCS_DIR = ROOT_DIR / "docs"

def test_external_package_audit_file():
    audit_file = RES_4_4_DIR / "external_package_audit.json"
    assert audit_file.exists(), "external_package_audit.json must exist"
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["meta"]["current_machine_training"] is False
    assert "A1" in data["packages"]
    assert "A3" in data["packages"]
    assert "RQ4" in data["packages"]
    assert "training_handoff_phase4_2" in data["packages"]
    assert data["packages"]["A1"]["parameter_conformance"] is True
    assert data["packages"]["A3"]["parameter_conformance"] is True
    assert data["packages"]["RQ4"]["total_expected_snapshots"] == 36

def test_runner_scripts_existence():
    required_scripts = [
        "preflight.py",
        "run_a1.py",
        "run_a3.py",
        "run_rq4.py",
        "verify_results.py",
        "collect_results.py",
        "run_all.py"
    ]
    for sc in required_scripts:
        p = PHASE_4_4_DIR / sc
        assert p.exists(), f"Script {sc} must exist in external_gpu/phase4_4/"
        assert p.stat().st_size > 100, f"Script {sc} must not be empty"

def test_preflight_dry_run():
    script = PHASE_4_4_DIR / "preflight.py"
    res = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert res.returncode == 0, f"Preflight dry run failed: {res.stderr}\n{res.stdout}"
    assert "All preflight checks passed" in res.stdout

def test_run_a1_dry_run():
    script = PHASE_4_4_DIR / "run_a1.py"
    res = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert res.returncode == 0, f"run_a1 dry run failed: {res.stderr}\n{res.stdout}"
    assert "A1 Contract & Model Dry-Run Complete" in res.stdout

def test_run_a3_dry_run():
    script = PHASE_4_4_DIR / "run_a3.py"
    res = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert res.returncode == 0, f"run_a3 dry run failed: {res.stderr}\n{res.stdout}"
    assert "A3 Contract & Model Dry-Run Complete" in res.stdout

def test_run_rq4_dry_run():
    script = PHASE_4_4_DIR / "run_rq4.py"
    res = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert res.returncode == 0, f"run_rq4 dry run failed: {res.stderr}\n{res.stdout}"
    assert "RQ4 Contract & Corpus Dry-Run Complete" in res.stdout

def test_verify_results_dry_run():
    script = PHASE_4_4_DIR / "verify_results.py"
    res = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert res.returncode == 0, f"verify_results dry run failed: {res.stderr}\n{res.stdout}"

def test_collect_results_dry_run():
    script = PHASE_4_4_DIR / "collect_results.py"
    res = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert res.returncode == 0, f"collect_results dry run failed: {res.stderr}\n{res.stdout}"
    assert (RES_4_4_DIR / "manifests" / "ingestion_manifest.json").exists()

def test_ingestion_directories_structure():
    expected_subdirs = ["A1", "A3", "RQ4", "checkpoints", "validation", "manifests"]
    for sub in expected_subdirs:
        p = RES_4_4_DIR / sub
        assert p.exists() and p.is_dir(), f"Ingestion directory {sub} must exist"

def test_progress_doc_status():
    doc_path = DOCS_DIR / "phase4_4_progress.md"
    assert doc_path.exists()
    content = doc_path.read_text(encoding="utf-8")
    assert "NEED_EXTERNAL_GPU" in content
    assert "PARTIAL" in content
    assert "COMPLETED" in content
    assert "RTX 3050" in content

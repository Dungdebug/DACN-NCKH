#!/usr/bin/env python3
"""
Verification Script for RQ4 Sequential Document Ingestion Results.
Validates presence and mathematical consistency of all 4 milestones:
- D0, D0+5, D0+10, D0+20.
"""

import sys
import json
import hashlib
from pathlib import Path

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_rq4_results():
    print("=" * 70)
    print("RQ4 SEQUENTIAL RESULTS INTEGRITY VERIFICATION")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent
    res_file = base_dir / "sequential_results.json"
    
    report = {
        "results_file_present": False,
        "milestones_verified": {},
        "all_milestones_present": False,
        "forgetting_formula_valid": True,
        "overall_status": "PASS"
    }

    if not res_file.exists():
        print(f"[FAIL] Missing results file: {res_file.name}")
        report["overall_status"] = "PENDING_RUNS"
        out_file = base_dir / "verification_report.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return False

    report["results_file_present"] = True
    report["sha256"] = compute_sha256(res_file)

    with open(res_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    milestones = ["D0", "D0+5", "D0+10", "D0+20"]
    evals = data.get("milestone_evaluations", {})
    all_present = True

    for m in milestones:
        if m in evals:
            rec = evals[m]
            acc_before = rec.get("accuracy_before", 0.0)
            acc_after = rec.get("accuracy_after", 0.0)
            fk = rec.get("forgetting_fk", 0.0)
            expected_fk = round(acc_before - acc_after, 4)
            
            formula_ok = abs(fk - expected_fk) < 1e-4
            if not formula_ok:
                report["forgetting_formula_valid"] = False

            report["milestones_verified"][m] = {
                "present": True,
                "accuracy_before": acc_before,
                "accuracy_after": acc_after,
                "forgetting_fk": fk,
                "formula_consistent": formula_ok
            }
            print(f"[OK] Milestone {m} verified: Before={acc_before}, After={acc_after}, F_k={fk}")
        else:
            all_present = False
            report["milestones_verified"][m] = {"present": False}
            print(f"[FAIL] Missing milestone: {m}")

    report["all_milestones_present"] = all_present
    if not all_present or not report["forgetting_formula_valid"]:
        report["overall_status"] = "FAIL_VALIDATION"

    out_file = base_dir / "verification_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nVerification Report saved to: {out_file}")
    print(f"Status: {report['overall_status']}")
    print("=" * 70)
    return report["overall_status"] == "PASS"

if __name__ == "__main__":
    success = verify_rq4_results()
    sys.exit(0 if success else 1)

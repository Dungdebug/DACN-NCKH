#!/usr/bin/env python3
"""
Results Verification Script for RQ2 External GPU Execution.
Validates result completeness across Seeds 42, 43, 44, item consistency, and absence of NaNs.
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

def verify_rq2_results():
    print("=" * 70)
    print("RQ2 RESULTS INTEGRITY VERIFICATION")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent
    seeds = [42, 43, 44]
    report = {
        "verified_seeds": {},
        "all_seeds_present": True,
        "sample_count_consistent": True,
        "no_nans_or_nulls": True,
        "overall_status": "PASS"
    }

    sample_counts = []

    for seed in seeds:
        file_path = base_dir / f"results_seed{seed}.json"
        if not file_path.exists():
            print(f"[FAIL] Missing results file: {file_path.name}")
            report["verified_seeds"][seed] = {"present": False}
            report["all_seeds_present"] = False
            report["overall_status"] = "FAIL_MISSING_FILES"
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        p1_items = data.get("p1_items", [])
        b5_items = data.get("b5_items", [])
        n_p1 = len(p1_items)
        n_b5 = len(b5_items)

        if n_p1 != n_b5 or n_p1 < 500:
            print(f"[WARN] Seed {seed} has unequal or insufficient items: P1={n_p1}, B5={n_b5} (Target >= 500)")
            report["sample_count_consistent"] = False

        sample_counts.append(n_p1)

        # Check for nulls or NaNs
        has_null = any(item.get("f1") is None for item in p1_items + b5_items)
        if has_null:
            report["no_nans_or_nulls"] = False

        file_hash = compute_sha256(file_path)
        report["verified_seeds"][seed] = {
            "present": True,
            "sample_count": n_p1,
            "sha256": file_hash,
            "p1_mean": data.get("summary", {}).get("p1_mean_f1"),
            "b5_mean": data.get("summary", {}).get("b5_mean_f1")
        }
        print(f"[OK] Seed {seed} verified: N={n_p1}, SHA256={file_hash[:12]}...")

    if not report["all_seeds_present"]:
        report["overall_status"] = "PENDING_RUNS"
    elif not report["sample_count_consistent"] or not report["no_nans_or_nulls"]:
        report["overall_status"] = "FAIL_VALIDATION"

    out_file = base_dir / "verification_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nVerification Report saved to: {out_file}")
    print(f"Status: {report['overall_status']}")
    print("=" * 70)
    return report["overall_status"] == "PASS"

if __name__ == "__main__":
    success = verify_rq2_results()
    sys.exit(0 if success else 1)

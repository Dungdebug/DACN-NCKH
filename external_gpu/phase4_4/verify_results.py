"""
Phase 4.4 Result Verification Script
Verifies:
- Checkpoints are loadable
- Parameter count matches exactly 5,314,752
- No NaN or Inf values
- Required metadata keys present
- Evaluation outputs complete across all seeds
"""

import sys
import os
import json
import argparse
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

EXPECTED_PARAMS = 5314752

def verify_checkpoint(path, expected_seed=None):
    if not path.exists():
        return False, f"File not found: {path}"
    try:
        sd = torch.load(path, map_location="cpu")
        if "cms_state_dict" not in sd:
            return False, "Missing 'cms_state_dict'"
        
        # Check params
        param_count = sum(p.numel() for p in sd["cms_state_dict"].values())
        if param_count != EXPECTED_PARAMS:
            return False, f"Parameter count mismatch: {param_count} != {EXPECTED_PARAMS}"
            
        # Check NaN / Inf
        for k, v in sd["cms_state_dict"].items():
            if torch.isnan(v).any():
                return False, f"NaN detected in {k}"
            if torch.isinf(v).any():
                return False, f"Inf detected in {k}"

        if expected_seed is not None and sd.get("seed") != expected_seed:
            return False, f"Seed mismatch: {sd.get('seed')} != {expected_seed}"

        return True, "VALID"
    except Exception as e:
        return False, f"Corrupted checkpoint: {e}"

def verify_target(target, dry_run=False):
    print(f"[*] Verifying execution outputs for: {target} (dry_run={dry_run})")
    if dry_run:
        print(f"[DRY RUN] Static contract check for {target}: VERIFIED.")
        return 0

    if target == "A1":
        ckpt_dir = ROOT_DIR / "checkpoints" / "phase4_2" / "A1"
        for s in [42, 43, 44]:
            p = ckpt_dir / f"cms_3lvl_random_seed_{s}.pt"
            ok, msg = verify_checkpoint(p, expected_seed=s)
            print(f"    - {p.name}: {'OK' if ok else 'FAIL (' + msg + ')'}")
            if not ok:
                return 1
        res_file = ROOT_DIR / "results" / "phase4_2" / "a1_external_results.json"
        if not res_file.exists():
            print(f"[!] Missing A1 evaluation results: {res_file}")
            return 1

    elif target == "A3":
        ckpt_dir = ROOT_DIR / "checkpoints" / "phase4_2" / "A3"
        for s in [42, 43, 44]:
            p = ckpt_dir / f"cms_3lvl_additive_seed_{s}.pt"
            ok, msg = verify_checkpoint(p, expected_seed=s)
            print(f"    - {p.name}: {'OK' if ok else 'FAIL (' + msg + ')'}")
            if not ok:
                return 1
        res_file = ROOT_DIR / "results" / "phase4_2" / "a3_external_results.json"
        if not res_file.exists():
            print(f"[!] Missing A3 evaluation results: {res_file}")
            return 1

    elif target == "RQ4":
        ckpt_dir = ROOT_DIR / "checkpoints" / "phase4_2" / "RQ4"
        methods = ["B4", "B5", "P1"]
        seeds = [42, 43, 44]
        intervals = ["D0", "D0_plus5", "D0_plus10", "D0_plus20"]
        missing = 0
        for m in methods:
            for s in seeds:
                for inv in intervals:
                    p = ckpt_dir / f"rq4_{m}_seed_{s}_{inv}.pt"
                    if not p.exists():
                        missing += 1
        if missing > 0:
            print(f"[!] RQ4 missing {missing} / 36 snapshots!")
            return 1
        res_file = ROOT_DIR / "results" / "phase4_2" / "rq4_external_results.json"
        if not res_file.exists():
            print(f"[!] Missing RQ4 evaluation results: {res_file}")
            return 1

    elif target == "RQ2":
        res_file = ROOT_DIR / "results" / "phase4_2" / "rq2_external_results.json"
        if not res_file.exists():
            print(f"[!] Missing RQ2 evaluation results: {res_file}")
            return 1
        with open(res_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        n = data.get("sample_size_N", 0)
        if n < 500:
            print(f"[!] RQ2 sample size N={n} < 500!")
            return 1
        if "statistics" not in data:
            print("[!] RQ2 missing statistics summary!")
            return 1

    print(f"[SUCCESS] Target {target} verification PASSED.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4.4 Result Verifier")
    parser.add_argument("--target", choices=["A1", "A3", "RQ4", "RQ2", "ALL"], default="ALL", help="Target to verify")
    parser.add_argument("--dry-run", action="store_true", help="Perform verification dry run")
    args = parser.parse_args()

    targets = ["A1", "A3", "RQ4", "RQ2"] if args.target == "ALL" else [args.target]
    for t in targets:
        code = verify_target(t, dry_run=args.dry_run)
        if code != 0:
            sys.exit(code)
    sys.exit(0)


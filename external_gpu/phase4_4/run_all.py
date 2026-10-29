"""
Phase 4.4 / 4.5 Master One-Command External GPU Runner
Executes sequential pipeline:
Preflight -> A1 -> Validate A1 -> A3 -> Validate A3 -> RQ4 -> Validate RQ4 -> RQ2 -> Validate RQ2 -> Ingest Results & Build Package

If any stage fails, execution halts immediately with error code.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PHASE_4_4_DIR = ROOT_DIR / "external_gpu" / "phase4_4"

def run_step(step_name, cmd):
    print("\n" + "=" * 75)
    print(f"PIPELINE STEP: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 75)
    res = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if res.returncode != 0:
        print(f"\n[PIPELINE ABORTED] Step '{step_name}' failed with returncode {res.returncode}.")
        sys.exit(res.returncode)
    print(f"[PIPELINE STEP PASSED] '{step_name}' completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Phase 4.4 / 4.5 One-Command External Runner")
    parser.add_argument("--dry-run", action="store_true", help="Run entire pipeline in static verification mode without training")
    parser.add_argument("--allow-non-3050", action="store_true", help="Allow running on GPUs other than RTX 3050")
    args = parser.parse_args()

    python_bin = sys.executable
    dry_flag = ["--dry-run"] if args.dry_run else []
    allow_flag = ["--allow-non-3050"] if args.allow_non_3050 else []

    print("=" * 75)
    print("PHASE 4.5 MASTER EXTERNAL GPU PIPELINE")
    print(f"Execution Mode: {'DRY RUN (STATIC VALIDATION)' if args.dry_run else 'PRODUCTION GPU RUN'}")
    print("=" * 75)

    # Step 1: Preflight
    run_step("1. Hardware Preflight Check", [python_bin, str(PHASE_4_4_DIR / "preflight.py")] + allow_flag + dry_flag)

    # Step 2: A1 Execution
    run_step("2. A1 Random Boundary Execution", [python_bin, str(PHASE_4_4_DIR / "run_a1.py")] + dry_flag)

    # Step 3: Verify A1
    run_step("3. A1 Result Verification", [python_bin, str(PHASE_4_4_DIR / "verify_results.py"), "--target", "A1"] + dry_flag)

    # Step 4: A3 Execution
    run_step("4. A3 Additive Ungated Execution", [python_bin, str(PHASE_4_4_DIR / "run_a3.py")] + dry_flag)

    # Step 5: Verify A3
    run_step("5. A3 Result Verification", [python_bin, str(PHASE_4_4_DIR / "verify_results.py"), "--target", "A3"] + dry_flag)

    # Step 6: RQ4 Execution
    run_step("6. RQ4 Continual Ingestion Execution", [python_bin, str(PHASE_4_4_DIR / "run_rq4.py")] + dry_flag)

    # Step 7: Verify RQ4
    run_step("7. RQ4 Result Verification", [python_bin, str(PHASE_4_4_DIR / "verify_results.py"), "--target", "RQ4"] + dry_flag)

    # Step 8: RQ2 Execution
    run_step("8. RQ2 Structure vs Fixed-Token Execution", [python_bin, str(PHASE_4_4_DIR / "run_rq2.py")] + dry_flag)

    # Step 9: Verify RQ2
    run_step("9. RQ2 Result Verification", [python_bin, str(PHASE_4_4_DIR / "verify_results.py"), "--target", "RQ2"] + dry_flag)

    # Step 10: Result Ingestion & Package Creation
    run_step("10. Result Collection & Package Creation", [python_bin, str(PHASE_4_4_DIR / "collect_results.py")] + dry_flag)

    print("\n" + "=" * 75)
    print("[MASTER PIPELINE COMPLETED] All 10 stages executed and verified successfully.")
    print("=" * 75)

if __name__ == "__main__":
    main()

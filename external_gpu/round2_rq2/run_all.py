#!/usr/bin/env python3
"""
Master Orchestration Script for RQ2 External GPU Execution.
Sequentially runs:
1. preflight.py
2. run_seed42.py
3. run_seed43.py
4. run_seed44.py
5. verify_results.py
6. collect_results.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 80)
    print("STARTING RQ2 COMPLETE EXTERNAL GPU PIPELINE")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent
    scripts = [
        ("Preflight Check", "preflight.py"),
        ("Execution Seed 42", "run_seed42.py"),
        ("Execution Seed 43", "run_seed43.py"),
        ("Execution Seed 44", "run_seed44.py"),
        ("Results Verification", "verify_results.py"),
        ("Statistical Collection & Verdict", "collect_results.py")
    ]

    for label, script_name in scripts:
        script_path = base_dir / script_name
        print(f"\n---> [STEP] Running {label} ({script_name})...")
        cmd = [sys.executable, str(script_path)]
        if len(sys.argv) > 1 and "run_seed" in script_name:
            cmd.append(sys.argv[1])

        res = subprocess.run(cmd, cwd=base_dir)
        if res.returncode != 0:
            print(f"\n[ERROR] Pipeline aborted at step: {label} (exit code {res.returncode})")
            sys.exit(res.returncode)

    print("\n" + "=" * 80)
    print("ALL RQ2 STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Master Orchestration Script for RQ4 External GPU Execution.
Sequentially runs:
1. preflight.py
2. run_sequential.py
3. verify_results.py
4. collect_results.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 80)
    print("STARTING RQ4 COMPLETE EXTERNAL GPU PIPELINE")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent
    scripts = [
        ("Preflight Check", "preflight.py"),
        ("Sequential Ingestion", "run_sequential.py"),
        ("Results Verification", "verify_results.py"),
        ("Retention Synthesis & Report", "collect_results.py")
    ]

    for label, script_name in scripts:
        script_path = base_dir / script_name
        print(f"\n---> [STEP] Running {label} ({script_name})...")
        cmd = [sys.executable, str(script_path)]

        res = subprocess.run(cmd, cwd=base_dir)
        if res.returncode != 0:
            print(f"\n[ERROR] Pipeline aborted at step: {label} (exit code {res.returncode})")
            sys.exit(res.returncode)

    print("\n" + "=" * 80)
    print("ALL RQ4 STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()

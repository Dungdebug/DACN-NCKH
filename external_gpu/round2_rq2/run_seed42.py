#!/usr/bin/env python3
"""Run RQ2 evaluation under Seed 42."""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq2_runner_core import run_rq2_experiment

if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    run_rq2_experiment(seed=42, num_samples=num_samples)

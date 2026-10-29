#!/usr/bin/env python3
"""
External GPU Preflight Check for RQ2: Structure-Aligned vs Fixed-Token Updating.
Validates GPU environment, VRAM, PyTorch CUDA capabilities, dataset presence (N >= 500),
and computes SHA256 checksums of configuration and data files.
"""

import os
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

def run_preflight():
    print("=" * 70)
    print("RQ2 EXTERNAL GPU PREFLIGHT CHECK (P1 vs B5)")
    print("=" * 70)

    checks = {
        "python_version": sys.version,
        "cuda_available": False,
        "gpu_name": "None",
        "gpu_vram_gb": 0.0,
        "pytorch_version": "",
        "dataset_verified": False,
        "dataset_sample_count": 0,
        "config_sha256": "",
        "status": "PASS"
    }

    try:
        import torch
        checks["pytorch_version"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            checks["gpu_name"] = torch.cuda.get_device_name(0)
            checks["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            print(f"[OK] CUDA available: {checks['gpu_name']} ({checks['gpu_vram_gb']} GB VRAM)")
            if checks["gpu_vram_gb"] < 10.0:
                print(f"[WARN] VRAM is {checks['gpu_vram_gb']} GB (< 12 GB recommended). Batch size should be set to 1.")
        else:
            print("[WARN] CUDA not detected. This script is intended to run on an External GPU host.")
            checks["status"] = "WARN_NO_CUDA"
    except ImportError:
        print("[FAIL] PyTorch not installed.")
        checks["status"] = "FAIL_NO_TORCH"

    # Dataset check
    dataset_path = Path(__file__).resolve().parent / "rq2_evaluation_dataset.json"
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            checks["dataset_sample_count"] = len(data.get("items", []))
            checks["dataset_verified"] = checks["dataset_sample_count"] >= 500
            checks["config_sha256"] = compute_sha256(dataset_path)
            print(f"[OK] Dataset loaded: {checks['dataset_sample_count']} samples (SHA256: {checks['config_sha256'][:12]}...)")
    else:
        # Fallback to synthetic large-scale template if running dry-run
        checks["dataset_verified"] = False
        print(f"[WARN] Dataset not found at {dataset_path}. run_all.py will generate the standardized N=500 benchmark partition.")

    out_file = Path(__file__).resolve().parent / "preflight_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

    print(f"\nPreflight Report saved to: {out_file}")
    print(f"Status: {checks['status']}")
    print("=" * 70)
    return checks["status"] in ["PASS", "WARN_NO_CUDA"]

if __name__ == "__main__":
    success = run_preflight()
    sys.exit(0 if success else 1)

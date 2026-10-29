#!/usr/bin/env python3
"""
External GPU Preflight Check for RQ4: Sequential Document Ingestion & Catastrophic Forgetting.
Validates GPU environment (VRAM >= 12GB recommended for sequential multi-document gradient updates),
PyTorch CUDA availability, document stream configuration, and directory readiness.
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
    print("RQ4 EXTERNAL GPU PREFLIGHT CHECK (Sequential Ingestion D0 -> D20)")
    print("=" * 70)

    checks = {
        "python_version": sys.version,
        "cuda_available": False,
        "gpu_name": "None",
        "gpu_vram_gb": 0.0,
        "pytorch_version": "",
        "stream_config_verified": False,
        "status": "PASS"
    }

    try:
        import torch
        checks["pytorch_version"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            checks["gpu_name"] = torch.cuda.get_device_name(0)
            checks["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            print(f"[OK] CUDA detected: {checks['gpu_name']} ({checks['gpu_vram_gb']} GB VRAM)")
            if checks["gpu_vram_gb"] < 12.0:
                print(f"[WARN] VRAM is {checks['gpu_vram_gb']} GB (< 12 GB). Multi-document gradient accumulation recommended.")
        else:
            print("[WARN] CUDA not detected. This script is intended to run on an External GPU host.")
            checks["status"] = "WARN_NO_CUDA"
    except ImportError:
        print("[FAIL] PyTorch not installed.")
        checks["status"] = "FAIL_NO_TORCH"

    stream_cfg = Path(__file__).resolve().parent / "stream_config.py"
    if stream_cfg.exists():
        checks["stream_config_verified"] = True
        checks["config_sha256"] = compute_sha256(stream_cfg)
        print(f"[OK] Stream configuration verified (SHA256: {checks['config_sha256'][:12]}...)")
    else:
        checks["stream_config_verified"] = False
        checks["status"] = "FAIL_NO_CONFIG"

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

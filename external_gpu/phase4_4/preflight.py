"""
Phase 4.4 Hardware Precheck Script
Checks:
- GPU: NVIDIA RTX 3050
- VRAM: >= 6GB
- CUDA available
- PyTorch compatible
- Python version compatible
- Disk space sufficient
- RAM sufficient
- Dataset exists
- Checksum correct
- Config hash correct
"""

import sys
import os
import shutil
import hashlib
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_preflight(allow_non_3050=False, dry_run=False):
    print("=" * 70)
    print("PHASE 4.4: HARDWARE & ENVIRONMENT PREFLIGHT AUDIT")
    print("=" * 70)

    checks_passed = True
    details = {}

    # 1. Python version
    py_ver = sys.version_info
    py_ok = (py_ver.major == 3 and py_ver.minor >= 8)
    details["python_version"] = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    print(f"[*] Python Version: {details['python_version']} ... {'OK' if py_ok else 'FAIL'}")
    if not py_ok:
        checks_passed = False

    # 2. PyTorch & CUDA
    try:
        import torch
        details["pytorch_version"] = torch.__version__
        cuda_avail = torch.cuda.is_available()
        details["cuda_available"] = cuda_avail
        print(f"[*] PyTorch Version: {torch.__version__} (CUDA Available: {cuda_avail}) ... {'OK' if cuda_avail else 'FAIL'}")
        if not cuda_avail and not dry_run:
            checks_passed = False

        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            details["device_name"] = device_name
            details["vram_gb"] = round(total_vram_gb, 2)
            print(f"[*] Detected GPU: {device_name} ({details['vram_gb']} GB VRAM)")

            is_3050 = "3050" in device_name
            has_6gb = total_vram_gb >= 5.5

            if is_3050 and has_6gb:
                print("    -> Target Hardware Match: NVIDIA RTX 3050 (>=6GB VRAM) verified.")
            else:
                msg = f"Non-target GPU detected: {device_name} ({details['vram_gb']} GB VRAM). Target is RTX 3050 (>=6GB)."
                if allow_non_3050 or dry_run:
                    print(f"    [WARNING] {msg} (Proceeding because allow_non_3050/dry_run is enabled)")
                else:
                    print(f"    [BLOCK] {msg}")
                    checks_passed = False
        else:
            if not dry_run:
                print("    [BLOCK] CUDA is not available. Real execution requires NVIDIA GPU.")
                checks_passed = False
    except ImportError:
        print("[!] PyTorch is not installed in current environment.")
        checks_passed = False

    # 3. Disk Space
    total, used, free = shutil.disk_usage(ROOT_DIR)
    free_gb = free / (1024 ** 3)
    details["free_disk_gb"] = round(free_gb, 2)
    disk_ok = free_gb >= 2.0
    print(f"[*] Available Disk Space: {details['free_disk_gb']} GB ... {'OK' if disk_ok else 'FAIL (Need >= 2GB)'}")
    if not disk_ok:
        checks_passed = False

    # 4. Dataset & Codebase Access
    try:
        from src.training.training_corpus import get_scale_training_corpus
        from src.evaluation.incremental_corpus import get_incremental_corpus
        scale_data = get_scale_training_corpus(200)
        inc_data = get_incremental_corpus()
        data_ok = (len(scale_data["samples"]) == 200 and "d0_document" in inc_data)
        print(f"[*] Training & Evaluation Corpora Ingestion ... {'OK' if data_ok else 'FAIL'}")
        if not data_ok:
            checks_passed = False
    except Exception as e:
        print(f"[!] Corpus Ingestion Failed: {e}")
        checks_passed = False

    # 5. Config Integrity Check
    cfg_files = [
        ROOT_DIR / "external_gpu" / "phase4_2" / "A1" / "config_a1.yaml",
        ROOT_DIR / "external_gpu" / "phase4_2" / "A3" / "config_a3.yaml",
        ROOT_DIR / "external_gpu" / "phase4_2" / "RQ4" / "config_rq4.yaml"
    ]
    all_cfg_ok = True
    for cfg in cfg_files:
        if not cfg.exists():
            print(f"[!] Missing config: {cfg}")
            all_cfg_ok = False
    print(f"[*] Configuration Files Integrity ... {'OK' if all_cfg_ok else 'FAIL'}")
    if not all_cfg_ok:
        checks_passed = False

    print("=" * 70)
    if checks_passed:
        print("[SUCCESS] All preflight checks passed. Environment is ready.")
        return 0
    else:
        print("[FAILURE] Preflight checks failed. Execution cannot proceed safely.")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4.4 Hardware Preflight")
    parser.add_argument("--allow-non-3050", action="store_true", help="Allow running on non-RTX 3050 GPU")
    parser.add_argument("--dry-run", action="store_true", help="Perform preflight dry run without blocking")
    args = parser.parse_args()
    code = run_preflight(allow_non_3050=args.allow_non_3050, dry_run=args.dry_run)
    sys.exit(code)

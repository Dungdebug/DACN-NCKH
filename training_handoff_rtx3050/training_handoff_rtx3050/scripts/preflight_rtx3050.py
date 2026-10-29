"""
PREFLIGHT VALIDATION FOR RTX 3050 EXTERNAL TRAINING
Enforces strict hardware, environment, data integrity, and leakage protection checks.
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_preflight(allow_non_target_gpu: bool = False) -> bool:
    print("=" * 80)
    print("PHASE 4.0.4 — PREFLIGHT VALIDATION (NVIDIA RTX 3050 6GB / 8GB)")
    print("=" * 80)

    checks_passed = []
    checks_failed = []

    def check(name: str, passed: bool, message: str):
        if passed:
            print(f"  [PASS] {name}: {message}")
            checks_passed.append((name, message))
        else:
            print(f"  [FAIL] {name}: {message}")
            checks_failed.append((name, message))

    # --------------------------------------------------------------------------
    # Check 1: PyTorch Version
    # --------------------------------------------------------------------------
    try:
        import torch
        pt_ver = torch.__version__
        major, minor = int(pt_ver.split(".")[0]), int(pt_ver.split(".")[1].split("+")[0])
        check("PyTorch Version", major >= 2, f"Installed {pt_ver} (>= 2.0.0 required)")
    except Exception as e:
        check("PyTorch Version", False, f"Error: {e}")
        print("\nPREFLIGHT STATUS: FAIL")
        return False

    # --------------------------------------------------------------------------
    # Check 2: CUDA Availability
    # --------------------------------------------------------------------------
    cuda_avail = torch.cuda.is_available()
    check("CUDA Availability", cuda_avail, f"CUDA Available: {cuda_avail}")
    if not cuda_avail:
        print("  CUDA is not available on this machine. Training requires an NVIDIA GPU.")
        print("\nPREFLIGHT STATUS: FAIL")
        return False

    # --------------------------------------------------------------------------
    # Check 3: CUDA Version
    # --------------------------------------------------------------------------
    cuda_ver = torch.version.cuda
    check("CUDA Version", cuda_ver is not None, f"PyTorch CUDA Version: {cuda_ver}")

    # --------------------------------------------------------------------------
    # Check 4: GPU Name & VRAM Validation
    # --------------------------------------------------------------------------
    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)

    is_rtx_3050 = "3050" in gpu_name
    is_valid_vram = (5.0 <= vram_gb <= 9.0)

    if allow_non_target_gpu:
        print(f"  [NOTE] --allow-non-target-gpu specified. Bypassing strict RTX 3050 name filter.")
        check("GPU Hardware Identification", True, f"Found '{gpu_name}' with {vram_gb:.2f} GB VRAM (Test Mode Allowed)")
    else:
        valid_hw = is_rtx_3050 and is_valid_vram
        check(
            "GPU Hardware Identification",
            valid_hw,
            f"Device: '{gpu_name}' | Total VRAM: {vram_gb:.2f} GB (Requires NVIDIA RTX 3050 with 6GB or 8GB VRAM)"
        )

    # --------------------------------------------------------------------------
    # Check 5: Training Dataset Exists & Exactly 200 Samples
    # --------------------------------------------------------------------------
    train_path = PACKAGE_ROOT / "data" / "train" / "train_200_samples.json"
    train_ok = False
    if train_path.exists():
        try:
            with open(train_path, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            samples = train_data.get("samples", [])
            docs = train_data.get("documents", [])
            train_ok = (len(samples) == 200 and len(docs) == 20)
            msg = f"Found {len(samples)} samples across {len(docs)} documents (Exactly 200 required)"
        except Exception as e:
            msg = f"Corrupt file: {e}"
    else:
        msg = f"Missing file: {train_path}"
    check("Training Dataset Lock", train_ok, msg)

    # --------------------------------------------------------------------------
    # Check 6: Validation Dataset Exists & Exactly 50 Samples
    # --------------------------------------------------------------------------
    val_path = PACKAGE_ROOT / "data" / "validation" / "val_50_samples.json"
    val_ok = False
    if val_path.exists():
        try:
            with open(val_path, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            samples = val_data.get("samples", [])
            docs = val_data.get("documents", [])
            val_ok = (len(samples) == 50 and len(docs) == 5)
            msg = f"Found {len(samples)} samples across {len(docs)} documents (Exactly 50 required)"
        except Exception as e:
            msg = f"Corrupt file: {e}"
    else:
        msg = f"Missing file: {val_path}"
    check("Validation Dataset Lock", val_ok, msg)

    # --------------------------------------------------------------------------
    # Check 7: Data Leakage Protection (No TEST Datasets in Package)
    # --------------------------------------------------------------------------
    leakage_detected = []
    forbidden_markers = [
        "vietnamese_final_corpus.py",
        "vietnamese_final_questions",
        "QASPERDocumentBenchmark",
        "NaturalMKNIAHBenchmark",
        "LongHealthDocumentBenchmark",
    ]
    for root, _, files in os.walk(PACKAGE_ROOT / "src"):
        for file in files:
            fp = Path(root) / file
            if fp.suffix == ".py":
                try:
                    content = fp.read_text(encoding="utf-8", errors="ignore")
                    for marker in forbidden_markers:
                        if marker in content and "types.py" not in file:
                            leakage_detected.append(f"{file} contains '{marker}'")
                except Exception:
                    pass

    leak_ok = (len(leakage_detected) == 0)
    check(
        "Data Leakage Protection",
        leak_ok,
        "Isolated package contains ZERO test benchmarks or test evaluation sets" if leak_ok else f"Leakage detected: {leakage_detected}"
    )

    # --------------------------------------------------------------------------
    # Check 8: Checksum Verification against SHA256SUMS.txt
    # --------------------------------------------------------------------------
    checksum_file = PACKAGE_ROOT / "checksums" / "SHA256SUMS.txt"
    checksum_ok = False
    if checksum_file.exists():
        try:
            mismatches = []
            verified_count = 0
            with open(checksum_file, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip().lstrip('\ufeff')
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        exp_hash, rel_path = parts[0].strip().lower(), parts[1].strip().replace("\\", "/")
                        target_file = PACKAGE_ROOT / rel_path
                        if not target_file.exists():
                            mismatches.append(f"Missing file: {rel_path}")
                        else:
                            curr_hash = sha256_file(target_file).lower()
                            if curr_hash != exp_hash:
                                mismatches.append(f"Hash mismatch on {rel_path}")
                            else:
                                verified_count += 1
            checksum_ok = (len(mismatches) == 0 and verified_count > 0)
            msg = f"Verified {verified_count} tracked files against SHA256SUMS.txt" if checksum_ok else f"Mismatches: {mismatches}"
        except Exception as e:
            msg = f"Checksum check failed: {e}"
    else:
        msg = f"Checksum file missing: {checksum_file}"
    check("Integrity Checksums", checksum_ok, msg)

    # --------------------------------------------------------------------------
    # Check 9: Model Loading & Frozen Backbone Test
    # --------------------------------------------------------------------------
    model_ok = False
    try:
        from src.hope_attention.pretrained_hope import PretrainedHopeLM
        model = PretrainedHopeLM(
            model_name_or_path="HuggingFaceTB/SmolLM2-135M",
            num_levels=1,
            device="cuda:0",
            enable_cms=True,
        )
        total_p = sum(p.numel() for p in model.backbone.parameters())
        trainable_p = sum(p.numel() for p in model.get_cms_parameters())
        backbone_frozen = all(not p.requires_grad for p in model.backbone.parameters())
        model_ok = (total_p == 134515008 and backbone_frozen and trainable_p > 0)
        msg = f"SmolLM2-135M loaded on cuda:0 ({total_p:,} params, 100% frozen, {trainable_p:,} adapter params trainable)"
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        msg = f"Model load error: {e}"
    check("Backbone & Adapter Initialization", model_ok, msg)

    # --------------------------------------------------------------------------
    # Check 10: Tokenizer Loading
    # --------------------------------------------------------------------------
    tok_ok = False
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        enc = tok.encode("Structure-Aligned Continual Memory Systems", add_special_tokens=False)
        tok_ok = len(enc) > 0
        msg = f"SmolLM2 Tokenizer loaded successfully (vocab: {tok.vocab_size:,})"
    except Exception as e:
        msg = f"Tokenizer error: {e}"
    check("Tokenizer Loading", tok_ok, msg)

    # --------------------------------------------------------------------------
    # Check 11: Output Directories Writable
    # --------------------------------------------------------------------------
    output_ok = False
    try:
        test_probe_1 = PACKAGE_ROOT / "output" / "checkpoints" / ".write_probe"
        test_probe_2 = PACKAGE_ROOT / "output" / "logs" / ".write_probe"
        test_probe_1.write_text("probe", encoding="utf-8")
        test_probe_2.write_text("probe", encoding="utf-8")
        test_probe_1.unlink()
        test_probe_2.unlink()
        output_ok = True
        msg = "output/checkpoints/ and output/logs/ are writable"
    except Exception as e:
        msg = f"Write permission error: {e}"
    check("Output Directory Permissions", output_ok, msg)

    # --------------------------------------------------------------------------
    # Final Verdict
    # --------------------------------------------------------------------------
    print("=" * 80)
    all_passed = (len(checks_failed) == 0)
    if all_passed:
        print("PREFLIGHT STATUS: PASS")
        print("System is 100% validated and ready for external training on RTX 3050.")
    else:
        print("PREFLIGHT STATUS: FAIL")
        print(f"Failed {len(checks_failed)} checks:")
        for name, message in checks_failed:
            print(f"  - {name}: {message}")
    print("=" * 80)

    return all_passed

if __name__ == "__main__":
    allow_any = "--allow-non-target-gpu" in sys.argv
    success = run_preflight(allow_non_target_gpu=allow_any)
    sys.exit(0 if success else 1)

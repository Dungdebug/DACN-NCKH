"""
Phase 4.4 Runner for A3: SA-CMS Additive / Ungated
Enforces A3 Execution Contract:
- 200 training samples
- 5,314,752 trainable parameters
- effective batch = 4 (batch 2, grad_accum 2)
- LR = 1e-4
- optimizer = AdamW
- 150 update steps
- seeds = [42, 43, 44]
- Checkpoint naming: cms_3lvl_additive_seed_{seed}.pt
- Dry-run capability for static contract verification
"""

import sys
import os
import json
import yaml
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

A3_PKG_DIR = ROOT_DIR / "external_gpu" / "phase4_2" / "A3"
CKPT_OUT_DIR = ROOT_DIR / "checkpoints" / "phase4_2" / "A3"
CKPT_OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_PARAMS = 5314752
EXPECTED_SEEDS = [42, 43, 44]
EXPECTED_SAMPLES = 200
EXPECTED_STEPS = 150

def run_a3(dry_run=False):
    print("=" * 70)
    print("PHASE 4.4: EXECUTING A3 (SA-CMS ADDITIVE / UNGATED)")
    print(f"Mode: {'DRY RUN (No Training)' if dry_run else 'REAL GPU EXECUTION'}")
    print("=" * 70)

    # 1. Load and verify config
    cfg_path = A3_PKG_DIR / "config_a3.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Verify contract
    assert cfg["model"]["expected_trainable_parameters"] == EXPECTED_PARAMS
    assert cfg["model"]["aggregation"] == "additive"
    assert cfg["training"]["samples_count"] == EXPECTED_SAMPLES
    assert cfg["training"]["seeds"] == EXPECTED_SEEDS
    assert cfg["training"]["optimizer"] == "AdamW"
    assert float(cfg["training"]["learning_rate"]) == 0.0001
    eff_bs = cfg["training"]["batch_size"] * cfg["training"]["grad_accum_steps"]
    assert eff_bs == 4
    calc_steps = (cfg["training"]["samples_count"] // eff_bs) * cfg["training"]["epochs"]
    assert calc_steps == EXPECTED_STEPS

    print("[*] A3 Execution Contract Verified:")
    print(f"    - Parameters: {EXPECTED_PARAMS}")
    print(f"    - Samples: {EXPECTED_SAMPLES}")
    print(f"    - Steps: {EXPECTED_STEPS}")
    print(f"    - Seeds: {EXPECTED_SEEDS}")
    print(f"    - Aggregation: additive (ungated)")

    if dry_run:
        print("[DRY RUN] Initializing A3 additive architecture model to verify parameter count...")
        import torch
        from src.hope_attention.sa_cms import StructureAlignedHopeLM
        model = StructureAlignedHopeLM(num_levels=3, device="cpu", torch_dtype=torch.float32, enable_cms=True)
        cms_p = sum(p.numel() for p in model.cms.parameters())
        assert cms_p == EXPECTED_PARAMS, f"Parameter count mismatch: {cms_p} != {EXPECTED_PARAMS}"
        print(f"[DRY RUN] Architecture verification PASSED: {cms_p} parameters.")
        
        # Test dummy forward pass
        dummy_ids = torch.randint(0, 1000, (1, 16))
        with torch.no_grad():
            out, _ = model.forward(dummy_ids)
        assert out.shape[-1] == model.vocab_size
        print(f"[DRY RUN] Dummy forward pass PASSED without training.")
        print("[SUCCESS] A3 Contract & Model Dry-Run Complete.")
        return 0

    # Real execution
    import subprocess
    train_script = A3_PKG_DIR / "train_a3.py"
    eval_script = A3_PKG_DIR / "eval_a3.py"
    val_script = A3_PKG_DIR / "validate_a3.py"

    print(f"[*] Launching A3 training: {train_script} ...")
    res = subprocess.run([sys.executable, str(train_script)], cwd=str(ROOT_DIR))
    if res.returncode != 0:
        print("[!] Training failed!")
        return 1

    print(f"[*] Validating A3 checkpoints: {val_script} ...")
    res = subprocess.run([sys.executable, str(val_script)], cwd=str(ROOT_DIR))
    if res.returncode != 0:
        print("[!] Checkpoint validation failed!")
        return 1

    print(f"[*] Running A3 evaluation: {eval_script} ...")
    res = subprocess.run([sys.executable, str(eval_script)], cwd=str(ROOT_DIR))
    if res.returncode != 0:
        print("[!] Evaluation failed!")
        return 1

    print("[SUCCESS] A3 Execution & Evaluation Complete.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4.4 A3 Runner")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without training")
    args = parser.parse_args()
    code = run_a3(dry_run=args.dry_run)
    sys.exit(code)

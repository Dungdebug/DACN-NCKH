# Validation Script for A3 Checkpoints
import os
import sys
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CKPT_DIR = ROOT_DIR / "checkpoints" / "phase4_2" / "A3"
SEEDS = [42, 43, 44]
EXPECTED_PARAMS = 5314752

def validate():
    print("Validating A3 Checkpoints...")
    all_ok = True
    for s in SEEDS:
        p = CKPT_DIR / f"cms_3lvl_additive_seed_{s}.pt"
        if not p.exists():
            print(f"FAIL: Missing {p}")
            all_ok = False
            continue
        state = torch.load(p, map_location="cpu")
        cms_sd = state.get("cms_state_dict", {})
        n_params = sum(t.numel() for t in cms_sd.values())
        has_nan = any(torch.isnan(t).any().item() or torch.isinf(t).any().item() for t in cms_sd.values())
        print(f"Seed {s}: params={n_params} (expected {EXPECTED_PARAMS}), has_nan={has_nan}")
        if n_params != EXPECTED_PARAMS or has_nan or state.get("seed") != s or state.get("num_samples") != 200:
            all_ok = False
    if all_ok:
        print("A3 VALIDATION: ALL PASSED")
    else:
        print("A3 VALIDATION: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    validate()

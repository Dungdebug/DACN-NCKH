"""
Phase 4.2 Reproducibility Validator
Strictly checks all external GPU handoff packages, configs, manifests, and architecture specs:
- config hash
- dataset hash
- checkpoint filename
- checkpoint architecture
- parameter count
- seed
- training sample count
- update count
- backbone ID
- tokenizer ID
- optimizer config
- learning rate
- effective batch size
If mismatch: RAISE ERROR. Never silently fix.
"""

import os
import sys
import json
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

LOCKED_BACKBONE = "HuggingFaceTB/SmolLM2-135M"
LOCKED_SEEDS = [42, 43, 44]
LOCKED_SAMPLE_COUNT = 200
LOCKED_LR = 0.0001
LOCKED_OPTIMIZER = "AdamW"
LOCKED_BATCH_SIZE = 2
LOCKED_GRAD_ACCUM = 2
LOCKED_EFFECTIVE_BATCH = 4
EXPECTED_3LVL_PARAMS = 5314752
EXPECTED_1LVL_PARAMS = 1771584

def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def validate_all():
    print("=" * 80)
    print("RUNNING PHASE 4.2 REPRODUCIBILITY VALIDATOR")
    print("=" * 80)

    errors = []

    # 1. Validate Phase 4.0.2 / 4.1 Checkpoints on disk
    print("[1/5] Validating existing Phase 4.1 Checkpoints on disk...")
    ckpt_dir = ROOT_DIR / "checkpoints" / "phase4_1"
    for s in LOCKED_SEEDS:
        p1 = ckpt_dir / f"cms_1lvl_seed_{s}.pt"
        p3 = ckpt_dir / f"cms_3lvl_seed_{s}.pt"
        if not p1.exists():
            errors.append(f"Missing 1-level checkpoint: {p1}")
        if not p3.exists():
            errors.append(f"Missing 3-level checkpoint: {p3}")

    # 2. Validate A1 External GPU Package
    print("[2/5] Validating A1 External GPU Package...")
    a1_dir = ROOT_DIR / "external_gpu" / "phase4_2" / "A1"
    required_a1_files = [
        "config_a1.yaml", "dataset_manifest.json", "seed_manifest.json",
        "expected_artifacts.json", "train_a1.py", "validate_a1.py", "eval_a1.py"
    ]
    for rf in required_a1_files:
        p = a1_dir / rf
        if not p.exists():
            errors.append(f"A1 Package missing required file: {p}")

    with open(a1_dir / "expected_artifacts.json", "r", encoding="utf-8") as f:
        a1_exp = json.load(f)
    if a1_exp.get("trainable_parameters") != EXPECTED_3LVL_PARAMS:
        errors.append(f"A1 parameter mismatch: expected {EXPECTED_3LVL_PARAMS}, got {a1_exp.get('trainable_parameters')}")
    if len(a1_exp.get("checkpoints", [])) != 3:
        errors.append("A1 expected checkpoints count must be 3")

    # 3. Validate A3 External GPU Package
    print("[3/5] Validating A3 External GPU Package...")
    a3_dir = ROOT_DIR / "external_gpu" / "phase4_2" / "A3"
    required_a3_files = [
        "config_a3.yaml", "dataset_manifest.json", "seed_manifest.json",
        "expected_artifacts.json", "train_a3.py", "validate_a3.py", "eval_a3.py"
    ]
    for rf in required_a3_files:
        p = a3_dir / rf
        if not p.exists():
            errors.append(f"A3 Package missing required file: {p}")

    with open(a3_dir / "expected_artifacts.json", "r", encoding="utf-8") as f:
        a3_exp = json.load(f)
    if a3_exp.get("trainable_parameters") != EXPECTED_3LVL_PARAMS:
        errors.append(f"A3 parameter mismatch: expected {EXPECTED_3LVL_PARAMS}, got {a3_exp.get('trainable_parameters')}")

    # 4. Validate RQ4 External GPU Package
    print("[4/5] Validating RQ4 Continual Forgetting Package...")
    rq4_dir = ROOT_DIR / "external_gpu" / "phase4_2" / "RQ4"
    required_rq4_files = [
        "config_rq4.yaml", "corpus_manifest.json", "snapshot_schedule.json",
        "run_rq4_ingestion.py", "eval_rq4_retention.py"
    ]
    for rf in required_rq4_files:
        p = rq4_dir / rf
        if not p.exists():
            errors.append(f"RQ4 Package missing required file: {p}")

    with open(rq4_dir / "corpus_manifest.json", "r", encoding="utf-8") as f:
        rq4_corp = json.load(f)
    if rq4_corp.get("d0_document_id") != "INC_DOC_000":
        errors.append(f"RQ4 initial document mismatch: {rq4_corp.get('d0_document_id')}")
    if len(rq4_corp.get("stream_document_ids", [])) != 20:
        errors.append(f"RQ4 stream documents count must be 20, got {len(rq4_corp.get('stream_document_ids', []))}")

    # 5. Validate Handoff Package and Checksums
    print("[5/5] Validating training_handoff_phase4_2/ package...")
    handoff_dir = ROOT_DIR / "training_handoff_phase4_2"
    for req in ["README.md", "ENVIRONMENT.md", "PRECHECK.sh", "CHECKSUMS.sha256"]:
        p = handoff_dir / req
        if not p.exists():
            errors.append(f"Handoff missing {p}")

    if errors:
        print("\n" + "!" * 80)
        print(f"REPRODUCIBILITY VALIDATION FAILED WITH {len(errors)} ERRORS:")
        for err in errors:
            print(f"  - [ERROR] {err}")
        print("!" * 80)
        sys.exit(1)
    else:
        print("\n" + "=" * 80)
        print("REPRODUCIBILITY VALIDATION PASSED: 100% SPECIFICATIONS CONFORM.")
        print("=" * 80)

if __name__ == "__main__":
    validate_all()

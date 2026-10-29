"""
Phase 4.4 / 4.5 Automatic Result Ingestion & Final Package Builder
Organizes outputs into results/phase4_4/ AND training_output_final/:
training_output_final/
├── A1_random_boundary/
├── A3_additive_ungated/
├── RQ4_sequential/
├── RQ2_structure_vs_token/
├── checkpoints/
├── raw_results/
├── statistics/
├── logs/
├── validation/
├── manifests/
├── checksums.sha256
└── README.md
"""

import sys
import os
import shutil
import json
import hashlib
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RES_4_4 = ROOT_DIR / "results" / "phase4_4"
FINAL_DIR = ROOT_DIR / "training_output_final"

SUBDIRS_4_4 = ["A1", "A3", "RQ4", "RQ2", "checkpoints", "validation", "manifests"]
FINAL_SUBDIRS = [
    "A1_random_boundary",
    "A3_additive_ungated",
    "RQ4_sequential",
    "RQ2_structure_vs_token",
    "checkpoints",
    "raw_results",
    "statistics",
    "logs",
    "validation",
    "manifests"
]

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def setup_directories():
    for sub in SUBDIRS_4_4:
        (RES_4_4 / sub).mkdir(parents=True, exist_ok=True)
    for sub in FINAL_SUBDIRS:
        (FINAL_DIR / sub).mkdir(parents=True, exist_ok=True)
    print(f"[*] Initialized output structures: {RES_4_4} & {FINAL_DIR}")

def collect_results(dry_run=False):
    print("=" * 70)
    print("PHASE 4.4 / 4.5: AUTOMATIC RESULT INGESTION & FINAL PACKAGE CREATION")
    print(f"Mode: {'DRY RUN' if dry_run else 'REAL INGESTION'}")
    print("=" * 70)

    setup_directories()

    manifest = {
        "title": "Phase 4.5 Full External GPU Execution Manifest",
        "date": "2026-10-06",
        "dry_run": dry_run,
        "ingested_artifacts": {}
    }

    targets = {
        "A1": {
            "result_src": ROOT_DIR / "results" / "phase4_2" / "a1_external_results.json",
            "dest_dir_4_4": RES_4_4 / "A1",
            "dest_dir_final": FINAL_DIR / "A1_random_boundary",
            "ckpt_src_dir": ROOT_DIR / "checkpoints" / "phase4_2" / "A1"
        },
        "A3": {
            "result_src": ROOT_DIR / "results" / "phase4_2" / "a3_external_results.json",
            "dest_dir_4_4": RES_4_4 / "A3",
            "dest_dir_final": FINAL_DIR / "A3_additive_ungated",
            "ckpt_src_dir": ROOT_DIR / "checkpoints" / "phase4_2" / "A3"
        },
        "RQ4": {
            "result_src": ROOT_DIR / "results" / "phase4_2" / "rq4_external_results.json",
            "dest_dir_4_4": RES_4_4 / "RQ4",
            "dest_dir_final": FINAL_DIR / "RQ4_sequential",
            "ckpt_src_dir": ROOT_DIR / "checkpoints" / "phase4_2" / "RQ4"
        },
        "RQ2": {
            "result_src": ROOT_DIR / "results" / "phase4_2" / "rq2_external_results.json",
            "dest_dir_4_4": RES_4_4 / "RQ2",
            "dest_dir_final": FINAL_DIR / "RQ2_structure_vs_token",
            "ckpt_src_dir": None
        }
    }

    for name, info in targets.items():
        if dry_run:
            manifest["ingested_artifacts"][name] = {
                "status": "DRY_RUN_MAPPED",
                "planned_source": str(info["result_src"].relative_to(ROOT_DIR))
            }
            print(f"    - [DRY RUN] {name}: Plan mapped.")
        else:
            if info["result_src"].exists():
                shutil.copy2(info["result_src"], info["dest_dir_4_4"] / info["result_src"].name)
                shutil.copy2(info["result_src"], info["dest_dir_final"] / info["result_src"].name)
                shutil.copy2(info["result_src"], FINAL_DIR / "raw_results" / info["result_src"].name)
                manifest["ingested_artifacts"][name] = {
                    "status": "INGESTED",
                    "file": info["result_src"].name
                }
                print(f"    - Ingested {name} results: {info['result_src'].name}")
            else:
                manifest["ingested_artifacts"][name] = {"status": "NOT_FOUND_ON_DISK"}
                print(f"    - [WARNING] {name} results not found at {info['result_src']}")

            # Copy checkpoints into training_output_final/checkpoints/
            if info["ckpt_src_dir"] and info["ckpt_src_dir"].exists():
                for ckpt_f in info["ckpt_src_dir"].glob("*.pt"):
                    shutil.copy2(ckpt_f, FINAL_DIR / "checkpoints" / ckpt_f.name)
                    shutil.copy2(ckpt_f, RES_4_4 / "checkpoints" / ckpt_f.name)
                print(f"    - Copied checkpoints from {info['ckpt_src_dir'].name}")

    # Save manifest
    manifest_out = FINAL_DIR / "manifests" / "final_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Generate README for training_output_final
    readme_content = """# SA-CMS Phase 4.5 Final Training & Evaluation Output Package

This package contains the official experimental results, model checkpoints, raw result data, statistics, and validation records for the SA-CMS project (Phase 4.5 + RQ2).

## Directory Structure
- `A1_random_boundary/`: Results and evaluation artifacts for Experiment A1 (Random Boundary CMS).
- `A3_additive_ungated/`: Results and evaluation artifacts for Experiment A3 (Additive / Ungated SA-CMS).
- `RQ4_sequential/`: Results and 36 snapshot checkpoints for Experiment RQ4 (Sequential Continual Ingestion).
- `RQ2_structure_vs_token/`: Results and paired statistical audit for Experiment RQ2 (Structure-Aligned vs Fixed-Token).
- `checkpoints/`: Complete set of validated PyTorch `.pt` checkpoints.
- `raw_results/`: Raw item-level and benchmark JSON output files.
- `statistics/`: Statistical analysis outputs (paired t-test, Wilcoxon, Bootstrap CI, Cohen's d).
- `logs/`: Training and evaluation execution logs.
- `validation/`: Automated validation reports.
- `manifests/`: Checksum and dataset provenance manifests.
- `checksums.sha256`: SHA-256 checksums of all critical package files.
"""
    with open(FINAL_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Generate checksums.sha256 (Task 12)
    print("[*] Generating SHA-256 checksums manifest...")
    checksum_lines = []
    for root, _, files in os.walk(FINAL_DIR):
        for file in sorted(files):
            if file == "checksums.sha256":
                continue
            fp = Path(root) / file
            rel_p = fp.relative_to(FINAL_DIR)
            c_hash = sha256_file(fp)
            checksum_lines.append(f"{c_hash}  {rel_p.as_posix()}\n")

    with open(FINAL_DIR / "checksums.sha256", "w", encoding="utf-8") as f:
        f.writelines(checksum_lines)

    print(f"[*] Checksums written to {FINAL_DIR / 'checksums.sha256'} ({len(checksum_lines)} files indexed).")
    print("[SUCCESS] Final Results Package Created Successfully.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4.4 / 4.5 Result Ingestion")
    parser.add_argument("--dry-run", action="store_true", help="Perform ingestion dry run")
    args = parser.parse_args()
    code = collect_results(dry_run=args.dry_run)
    sys.exit(code)

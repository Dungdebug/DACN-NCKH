"""
Phase 4.4 Runner for RQ4: Continual Document Ingestion & Catastrophic Forgetting
Enforces RQ4 Execution Contract:
- 21 sequential documents (D0 + 20 stream docs)
- Strict document ingestion order (no shuffling)
- 4 snapshot intervals: D0, D0_plus5, D0_plus10, D0_plus20
- Checkpoints: rq4_{method}_seed_{seed}_{interval}.pt for methods [B4, B5, P1]
- Forgetting metric: F_k = Accuracy(D0) - Accuracy(D0_plus_k) for k in {5, 10, 20}
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

RQ4_PKG_DIR = ROOT_DIR / "external_gpu" / "phase4_2" / "RQ4"
CKPT_OUT_DIR = ROOT_DIR / "checkpoints" / "phase4_2" / "RQ4"
CKPT_OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_METHODS = ["B4", "B5", "P1"]
EXPECTED_SEEDS = [42, 43, 44]
EXPECTED_INTERVALS = ["D0", "D0_plus5", "D0_plus10", "D0_plus20"]

def run_rq4(dry_run=False):
    print("=" * 70)
    print("PHASE 4.4: EXECUTING RQ4 (CONTINUAL INGESTION & FORGETTING)")
    print(f"Mode: {'DRY RUN (No Training)' if dry_run else 'REAL GPU EXECUTION'}")
    print("=" * 70)

    # 1. Load configs & manifests
    cfg_path = RQ4_PKG_DIR / "config_rq4.yaml"
    corp_path = RQ4_PKG_DIR / "corpus_manifest.json"
    snap_path = RQ4_PKG_DIR / "snapshot_schedule.json"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(corp_path, "r", encoding="utf-8") as f:
        corp = json.load(f)
    with open(snap_path, "r", encoding="utf-8") as f:
        snap = json.load(f)

    # Verify contracts
    assert cfg["experiment"]["methods"] == EXPECTED_METHODS
    assert cfg["experiment"]["seeds"] == EXPECTED_SEEDS
    intervals = [s["checkpoint_id"] for s in snap["schedule"]]
    assert intervals == EXPECTED_INTERVALS
    assert len(corp["stream_document_ids"]) == 20
    assert corp["d0_document_id"] == "INC_DOC_000"

    print("[*] RQ4 Execution Contract Verified:")
    print(f"    - Initial Document: {corp['d0_document_id']}")
    print(f"    - Stream Documents: {len(corp['stream_document_ids'])} sequential docs")
    print(f"    - Methods: {EXPECTED_METHODS}")
    print(f"    - Seeds: {EXPECTED_SEEDS}")
    print(f"    - Snapshot Intervals: {EXPECTED_INTERVALS}")
    print(f"    - Forgetting Formula: F_k = Accuracy(D0) - Accuracy(D0_plus_k)")

    if dry_run:
        print("[DRY RUN] Verifying incremental corpus ordering and diagnostic QA items...")
        from src.evaluation.incremental_corpus import get_incremental_corpus
        corpus_data = get_incremental_corpus()
        d0_doc = corpus_data["d0_document"]
        stream_docs = corpus_data["stream_documents"]

        assert d0_doc["doc_id"] == "INC_DOC_000"
        assert len(d0_doc["questions"]) == 10
        assert len(stream_docs) == 20
        # Check ordering
        for i, sdoc in enumerate(stream_docs, start=1):
            expected_id = f"INC_DOC_{i:03d}"
            assert sdoc["doc_id"] == expected_id, f"Ordering mismatch: {sdoc['doc_id']} != {expected_id}"
        print(f"[DRY RUN] Document ordering strictly sequential (INC_DOC_001 to INC_DOC_020).")
        print(f"[DRY RUN] Diagnostic QA set verified (10 items on D0).")
        print("[SUCCESS] RQ4 Contract & Corpus Dry-Run Complete.")
        return 0

    # Real execution
    import subprocess
    run_script = RQ4_PKG_DIR / "run_rq4_ingestion.py"
    eval_script = RQ4_PKG_DIR / "eval_rq4_retention.py"

    print(f"[*] Launching RQ4 sequential ingestion: {run_script} ...")
    res = subprocess.run([sys.executable, str(run_script)], cwd=str(ROOT_DIR))
    if res.returncode != 0:
        print("[!] Sequential ingestion failed!")
        return 1

    print(f"[*] Running RQ4 retention & forgetting evaluation: {eval_script} ...")
    res = subprocess.run([sys.executable, str(eval_script)], cwd=str(ROOT_DIR))
    if res.returncode != 0:
        print("[!] Evaluation failed!")
        return 1

    print("[SUCCESS] RQ4 Execution & Evaluation Complete.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4.4 RQ4 Runner")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without training")
    args = parser.parse_args()
    code = run_rq4(dry_run=args.dry_run)
    sys.exit(code)

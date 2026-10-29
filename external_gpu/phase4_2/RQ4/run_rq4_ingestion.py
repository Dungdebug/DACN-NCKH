# Standalone Sequential Ingestion and Snapshot Runner for RQ4 (RTX 3050)
import os
import sys
import json
import time
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hope_attention.sa_cms import StructureAlignedHopeLM
from src.evaluation.incremental_corpus import get_incremental_corpus

SEEDS = [42, 43, 44]
METHODS = ["B4", "B5", "P1"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SNAP_DIR = ROOT_DIR / "checkpoints" / "phase4_2" / "RQ4"
SNAP_DIR.mkdir(parents=True, exist_ok=True)

def run_rq4_seed(method: str, seed: int):
    print(f"=== RQ4 Sequential Ingestion: Method {method} | Seed {seed} ===")
    corpus = get_incremental_corpus()
    d0_doc = corpus["d0_document"]
    stream_docs = corpus["stream_documents"]

    n_lvl = 1 if method == "B4" else 3
    sched = "structure" if method == "P1" else "fixed_token"

    model = StructureAlignedHopeLM(num_levels=n_lvl, device=DEVICE, torch_dtype=DTYPE, enable_cms=True)
    # Load base adapter
    base_ckpt = ROOT_DIR / "checkpoints" / "phase4_1" / f"cms_{n_lvl}lvl_seed_{seed}.pt"
    if base_ckpt.exists():
        state = torch.load(base_ckpt, map_location=DEVICE)
        model.cms.load_state_dict(state["cms_state_dict"], strict=False)

    model.reset_memory()

    # 1. Ingest D0 and save snapshot D0
    model.ingest_structured_document(d0_doc["text"], schedule_mode=sched, seed=seed)
    torch.save({
        "cms_state_dict": model.cms.state_dict(),
        "ingested_docs": ["INC_DOC_000"],
        "checkpoint_id": "D0",
        "method": method,
        "seed": seed
    }, SNAP_DIR / f"rq4_{method}_seed_{seed}_D0.pt")

    # 2. Ingest +5 docs
    for doc in stream_docs[:5]:
        model.ingest_structured_document(doc["text"], schedule_mode=sched, seed=seed)
    torch.save({
        "cms_state_dict": model.cms.state_dict(),
        "ingested_docs": ["INC_DOC_000"] + [d["doc_id"] for d in stream_docs[:5]],
        "checkpoint_id": "D0_plus5",
        "method": method,
        "seed": seed
    }, SNAP_DIR / f"rq4_{method}_seed_{seed}_D0_plus5.pt")

    # 3. Ingest +10 docs
    for doc in stream_docs[5:10]:
        model.ingest_structured_document(doc["text"], schedule_mode=sched, seed=seed)
    torch.save({
        "cms_state_dict": model.cms.state_dict(),
        "ingested_docs": ["INC_DOC_000"] + [d["doc_id"] for d in stream_docs[:10]],
        "checkpoint_id": "D0_plus10",
        "method": method,
        "seed": seed
    }, SNAP_DIR / f"rq4_{method}_seed_{seed}_D0_plus10.pt")

    # 4. Ingest +20 docs
    for doc in stream_docs[10:20]:
        model.ingest_structured_document(doc["text"], schedule_mode=sched, seed=seed)
    torch.save({
        "cms_state_dict": model.cms.state_dict(),
        "ingested_docs": ["INC_DOC_000"] + [d["doc_id"] for d in stream_docs[:20]],
        "checkpoint_id": "D0_plus20",
        "method": method,
        "seed": seed
    }, SNAP_DIR / f"rq4_{method}_seed_{seed}_D0_plus20.pt")

    print(f"Completed and saved snapshots for {method} seed {seed}")

def main():
    for m in METHODS:
        for s in SEEDS:
            run_rq4_seed(m, s)

if __name__ == "__main__":
    main()

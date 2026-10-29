# Evaluation Script for A3 on QASPER, LongHealth, MK-NIAH
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
from src.evaluation.pretrained_benchmarks import (
    QASPERDocumentBenchmark,
    LongHealthDocumentBenchmark,
    NaturalMKNIAHBenchmark,
)

CKPT_DIR = ROOT_DIR / "checkpoints" / "phase4_2" / "A3"
OUT_FILE = ROOT_DIR / "results" / "phase4_2" / "a3_external_results.json"
SEEDS = [42, 43, 44]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def evaluate():
    print("Evaluating A3 (Additive/Ungated) Checkpoints on QASPER, LongHealth & MK-NIAH...")
    qasper = QASPERDocumentBenchmark(num_documents=10)
    lh = LongHealthDocumentBenchmark(num_documents=5)
    mkniah = NaturalMKNIAHBenchmark(num_samples=10)
    
    results = {"benchmark": "A3_Evaluation", "runs": []}

    for s in SEEDS:
        ckpt_p = CKPT_DIR / f"cms_3lvl_additive_seed_{s}.pt"
        if not ckpt_p.exists():
            print(f"[WARNING] Checkpoint {ckpt_p} does not exist!")
            continue

        print(f"--- Evaluating Seed {s} ---")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        model = StructureAlignedHopeLM(
            num_levels=3,
            device=DEVICE,
            torch_dtype=DTYPE,
            enable_cms=True
        )
        sd = torch.load(ckpt_p, map_location=DEVICE)
        model.cms.load_state_dict(sd["cms_state_dict"], strict=False)
        model.eval()

        # 1. QASPER Eval
        q_res = qasper.evaluate_model(model, enable_online_cms=True)

        # 2. LongHealth Eval
        lh_res = lh.evaluate_model(model, enable_online_cms=True)

        # 3. MK-NIAH Eval
        mk_res = mkniah.evaluate_model(model, enable_online_cms=True)

        elapsed = time.time() - t0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        run_info = {
            "seed": s,
            "qasper": {
                "f1": q_res["f1"],
                "perplexity": q_res["perplexity"],
                "avg_loss": q_res["avg_loss"],
                "exact_match": q_res["exact_match"]
            },
            "longhealth": {
                "accuracy": lh_res["accuracy_pct"],
                "perplexity": lh_res["perplexity"],
                "avg_loss": lh_res["avg_loss"],
                "avg_target_prob": lh_res["avg_target_prob"]
            },
            "mkniah": {
                "accuracy": mk_res["accuracy_pct"],
                "avg_target_prob": mk_res["avg_target_prob"]
            },
            "performance": {
                "latency_seconds": round(elapsed, 2),
                "peak_vram_mb": round(peak_vram_mb, 2)
            }
        }
        results["runs"].append(run_info)
        print(f"Seed {s} complete: QASPER F1={q_res['f1']:.4f}, LH Acc={lh_res['accuracy_pct']:.2f}%, MK-NIAH Acc={mk_res['accuracy_pct']:.2f}%")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved A3 external eval results to {OUT_FILE}")

if __name__ == "__main__":
    evaluate()


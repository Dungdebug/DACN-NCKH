# Evaluation Script for RQ4 Retention and Forgetting Calculation
# Computes F_k = Accuracy_before - Accuracy_after_k on initial document D0
import os
import sys
import json
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hope_attention.sa_cms import StructureAlignedHopeLM
from src.evaluation.incremental_corpus import get_incremental_corpus

SNAP_DIR = ROOT_DIR / "checkpoints" / "phase4_2" / "RQ4"
OUT_FILE = ROOT_DIR / "results" / "phase4_2" / "rq4_external_results.json"
SEEDS = [42, 43, 44]
METHODS = ["B4", "B5", "P1"]
CHECKPOINTS = ["D0", "D0_plus5", "D0_plus10", "D0_plus20"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_retention():
    print("Evaluating D0 Retention on Ingested Snapshots...")
    corpus = get_incremental_corpus()
    d0_doc = corpus["d0_document"]
    d0_questions = d0_doc["questions"]

    results = {"benchmark": "RQ4_Continual_Forgetting", "runs": []}

    for m in METHODS:
        n_lvl = 1 if m == "B4" else 3
        for s in SEEDS:
            acc_by_ckpt = {}
            for ckpt_id in CHECKPOINTS:
                snap_path = SNAP_DIR / f"rq4_{m}_seed_{s}_{ckpt_id}.pt"
                if not snap_path.exists():
                    continue
                model = StructureAlignedHopeLM(num_levels=n_lvl, device=DEVICE, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32, enable_cms=True)
                sd = torch.load(snap_path, map_location=DEVICE)
                model.cms.load_state_dict(sd["cms_state_dict"], strict=False)
                model.eval()

                correct = 0
                for q in d0_questions:
                    prompt = f"Question: {q['question']}\nAnswer:"
                    enc = model.tokenizer(prompt, return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        curr_ids = enc.input_ids.clone()
                        for _ in range(16):
                            logits, _ = model.forward(curr_ids)
                            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                            curr_ids = torch.cat([curr_ids, nxt], dim=1)
                            if nxt.item() == model.tokenizer.eos_token_id:
                                break
                    ans = model.tokenizer.decode(curr_ids[0, enc.input_ids.size(1):], skip_special_tokens=True).strip()
                    if any(kw.lower() in ans.lower() for kw in q["keywords"]):
                        correct += 1
                acc = (correct / len(d0_questions)) * 100.0
                acc_by_ckpt[ckpt_id] = acc

            if "D0" in acc_by_ckpt:
                acc_0 = acc_by_ckpt["D0"]
                delta_5 = acc_0 - acc_by_ckpt.get("D0_plus5", acc_0)
                delta_10 = acc_0 - acc_by_ckpt.get("D0_plus10", acc_0)
                delta_20 = acc_0 - acc_by_ckpt.get("D0_plus20", acc_0)

                results["runs"].append({
                    "method": m,
                    "seed": s,
                    "acc_initial_d0": acc_0,
                    "acc_plus_5": acc_by_ckpt.get("D0_plus5", 0.0),
                    "acc_plus_10": acc_by_ckpt.get("D0_plus10", 0.0),
                    "acc_plus_20": acc_by_ckpt.get("D0_plus20", 0.0),
                    "forgetting_f5": round(delta_5, 2),
                    "forgetting_f10": round(delta_10, 2),
                    "forgetting_f20": round(delta_20, 2),
                })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved RQ4 results to {OUT_FILE}")

if __name__ == "__main__":
    evaluate_retention()

"""
One-off audit: re-derive A1 / A3 / P1 ablation metrics directly from the checkpoints on disk
using the official ablation harness (scripts/evaluate_ablation_a1_a3.py), to determine which of
the conflicting artifact sets is reproducible:

  (a) results/raw_predictions/ablation_raw_predictions.jsonl   (item level, 360 rows)
  (b) results/phase4_2/a1_external_results.json                (run aggregates)
  (c) results/phase4_2/a3_external_results.json                (run aggregates)

Notes on the harness:
- QASPER prompts include the document context ("Context: ... Question: ... Answer:").
- LongHealth is scored by argmax over the A/B/C/D option logits.
- MK-NIAH uses enable_online_cms=False (no per-sample ingestion).

Outputs results/phase4_2/ablation_reproduction_audit.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import statistics

import torch
import torch.nn as nn

from src.cms.mlp_chain import MLPBlock
from src.evaluation.pretrained_benchmarks import (
    LongHealthDocumentBenchmark,
    NaturalMKNIAHBenchmark,
    QASPERDocumentBenchmark,
)
from src.hope_attention.sa_cms import StructureAlignedHopeLM

SEEDS = [42, 43, 44]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Pull the official helper functions (greedy decode with KV cache, SQuAD F1) from the
# evaluation script itself so this audit uses exactly the published measurement code.
_SRC = (ROOT / "scripts" / "evaluate_ablation_a1_a3.py").read_text(encoding="utf-8").split("def main()")[0]
_NS = {"__file__": str(ROOT / "scripts" / "evaluate_ablation_a1_a3.py"), "__name__": "ablation_harness"}
exec(compile(_SRC, "ablation_harness", "exec"), _NS)
generate_greedy = _NS["generate_greedy"]
compute_token_f1 = _NS["compute_token_f1"]
UngatedAdditiveMLPChain = _NS["UngatedAdditiveMLPChain"]

CHECKPOINTS = {
    "A1": ROOT / "training_output_final" / "checkpoints" / "cms_3lvl_random_seed_{seed}.pt",
    "A3": ROOT / "training_output_final" / "checkpoints" / "cms_3lvl_additive_seed_{seed}.pt",
    "P1": ROOT / "checkpoints" / "phase4_1" / "cms_3lvl_seed_{seed}.pt",
}


def build_model(variant):
    model = StructureAlignedHopeLM(num_levels=3, device=DEVICE, torch_dtype=DTYPE, enable_cms=True)
    if variant == "A3":
        model.cms.chain = UngatedAdditiveMLPChain(num_levels=3, d_model=576, d_ff=1536).to(
            device=DEVICE, dtype=DTYPE
        )
    return model


def evaluate_variant(variant):
    qasper = QASPERDocumentBenchmark(num_documents=10)
    longhealth = LongHealthDocumentBenchmark(num_documents=5)
    mkniah = NaturalMKNIAHBenchmark(num_samples=10)

    runs = []
    missing = []
    for seed in SEEDS:
        ckpt = CHECKPOINTS[variant]
        ckpt_path = Path(str(ckpt).format(seed=seed))
        if not ckpt_path.exists():
            # A1/A3 checkpoints live under training_output_final/checkpoints/, which is
            # git-ignored, so a fresh clone cannot reproduce them. Record the gap rather
            # than crashing or silently reporting a partial aggregate.
            missing.append(str(ckpt_path.relative_to(ROOT)).replace("\\", "/"))
            continue

        model = build_model(variant)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.cms.load_state_dict(state["cms_state_dict"], strict=False)
        model.eval()

        qasper_f1s = []
        for context, question, answer in qasper.documents:
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            enc = model.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            qasper_f1s.append(compute_token_f1(generate_greedy(model, enc, 32), answer))

        lh_correct = 0
        lh_total = 0
        for record in longhealth.clinical_records:
            for question_item in record["questions"]:
                options = "\n".join(f"{k}. {v}" for k, v in question_item["options"].items())
                prompt = (
                    f"{record['text']}\n\nQuestion: {question_item['question']}\n"
                    f"Options:\n{options}\nAnswer:"
                )
                ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
                with torch.no_grad():
                    logits, _ = model.forward(ids)
                    probs = torch.softmax(logits[0, -1, :], dim=-1)
                    option_probs = {
                        letter: probs[model.tokenizer.encode(f" {letter}", add_special_tokens=False)[0]].item()
                        for letter in "ABCD"
                    }
                lh_total += 1
                if max(option_probs, key=option_probs.get) == question_item["correct_letter"]:
                    lh_correct += 1

        mk_result = mkniah.evaluate_model(model, enable_online_cms=False)
        mk_correct = sum(1 for d in mk_result["detailed_samples"] if d["is_correct"])

        run = {
            "seed": seed,
            "checkpoint": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
            "qasper_f1": round(sum(qasper_f1s) / max(1, len(qasper_f1s)), 4),
            "longhealth_acc_pct": round(lh_correct / max(1, lh_total) * 100.0, 2),
            "mkniah_acc_pct": round(mk_correct / max(1, len(mk_result["detailed_samples"])) * 100.0, 2),
        }
        runs.append(run)
        print(f"  {variant} seed {seed}: {run}", flush=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def agg(key):
        values = [r[key] for r in runs]
        if not values:
            return {"mean": None, "sd": None, "per_seed": []}
        return {
            "mean": round(statistics.mean(values), 4),
            "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "per_seed": values,
        }

    return {
        "variant": variant,
        "runs": runs,
        "status": "ok" if not missing else ("partial" if runs else "checkpoints_missing"),
        "missing_checkpoints": missing,
        "aggregate": {
            "qasper_f1": agg("qasper_f1"),
            "longhealth_acc_pct": agg("longhealth_acc_pct"),
            "mkniah_acc_pct": agg("mkniah_acc_pct"),
        },
    }


if __name__ == "__main__":
    audit = {
        "purpose": (
            "Determine which ablation artifact set is reproducible from the checkpoints on disk, "
            "using the official harness in scripts/evaluate_ablation_a1_a3.py."
        ),
        "harness": {
            "qasper_prompt": "Context: {...} Question: {...} Answer:  (context INCLUDED)",
            "qasper_decode": "greedy, max_new_tokens=32, KV-cache",
            "longhealth": "argmax over A/B/C/D option logits",
            "mkniah": "10 samples, enable_online_cms=False, greedy 6 tokens, exact code match",
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "variants": {},
    }
    for variant in ["A1", "A3", "P1"]:
        print(f"=== {variant} ===", flush=True)
        audit["variants"][variant] = evaluate_variant(variant)

    dest = ROOT / "results" / "phase4_2" / "ablation_reproduction_audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    missing_variants = [
        name for name, payload in audit["variants"].items() if payload["status"] != "ok"
    ]
    if missing_variants:
        print(f"INCOMPLETE: {', '.join(missing_variants)} could not be fully reproduced", flush=True)
    print("saved", dest, flush=True)
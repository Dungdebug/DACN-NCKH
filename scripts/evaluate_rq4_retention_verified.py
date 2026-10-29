"""
Verified Retention and Forgetting Evaluation for RQ4 (Continual Document Ingestion).
Uses all 36 snapshot checkpoints across B4, B5, P1 x seeds 42, 43, 44 x intervals D0, D0+5, D0+10, D0+20.
Evaluates using continuous Token F1, Exact Match, Target Logprob, and legacy keyword accuracy.
Exports every prediction to results/raw_predictions/rq4_raw_predictions.jsonl.
Updates results in training_output_final/raw_results/rq4_external_results.json.
"""

import sys
import os
import re
import json
import time
import math
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hope_attention.sa_cms import StructureAlignedHopeLM
from src.evaluation.incremental_corpus import get_incremental_corpus

SNAP_DIR = ROOT_DIR / "training_output_final" / "checkpoints"
OUT_DIR_RAW = ROOT_DIR / "results" / "raw_predictions"
OUT_DIR_RAW.mkdir(parents=True, exist_ok=True)
RAW_PREDICTIONS_FILE = OUT_DIR_RAW / "rq4_raw_predictions.jsonl"

SEEDS = [42, 43, 44]
METHODS = ["B4", "B5", "P1"]
CHECKPOINTS = ["D0", "D0_plus5", "D0_plus10", "D0_plus20"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_target_logprob(model, prompt: str, target_str: str) -> float:
    enc_prompt = model.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    enc_full = model.tokenizer(prompt + " " + target_str, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = enc_prompt.size(1)
    if enc_full.size(1) <= prompt_len:
        return 0.0
    with torch.no_grad():
        logits, _ = model.forward(enc_full)
        shift_logits = logits[0, prompt_len - 1 : -1, :]
        shift_labels = enc_full[0, prompt_len:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        return float(target_log_probs.sum().item())


def get_level_norms(model) -> Tuple[float, float, float]:
    norms = [0.0, 0.0, 0.0]
    if hasattr(model, "cms") and model.cms is not None:
        chain = getattr(model.cms, "chain", None)
        if chain is not None and hasattr(chain, "blocks"):
            for i, b in enumerate(chain.blocks):
                if i < 3:
                    n_sq = sum(torch.norm(p.float()).item()**2 for p in b.parameters())
                    norms[i] = round(float(n_sq**0.5), 4)
    return norms[0], norms[1], norms[2]


def generate_greedy(model, prompt_ids: torch.Tensor, max_new_tokens: int = 24) -> str:
    """Generates autoregressively with greedy decoding using KV-cache for fast execution."""
    with torch.no_grad():
        out = model.backbone.model(input_ids=prompt_ids, use_cache=True, return_dict=True)
        kv = out.past_key_values
        h = out.last_hidden_state
        if hasattr(model, "cms") and model.cms is not None:
            h = h + model.cms(model.cms_norm(h))
        logits = model.backbone.lm_head(h)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids = [nxt.item()]

        for _ in range(max_new_tokens - 1):
            if nxt.item() == model.tokenizer.eos_token_id:
                break
            out = model.backbone.model(input_ids=nxt, past_key_values=kv, use_cache=True, return_dict=True)
            kv = out.past_key_values
            h = out.last_hidden_state
            if hasattr(model, "cms") and model.cms is not None:
                h = h + model.cms(model.cms_norm(h))
            logits = model.backbone.lm_head(h)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(nxt.item())

    return model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def run_rq4_evaluation():
    print("=" * 80)
    print("RE-EVALUATING RQ4 CONTINUAL FORGETTING ON 36 SNAPSHOTS")
    print(f"Device: {DEVICE} | DType: {DTYPE}")
    print(f"Output Raw Predictions JSONL: {RAW_PREDICTIONS_FILE}")
    print("=" * 80)

    corpus = get_incremental_corpus()
    d0_doc = corpus["d0_document"]
    d0_questions = d0_doc["questions"]

    results = {"benchmark": "RQ4_Continual_Forgetting", "runs": []}
    jsonl_f = open(RAW_PREDICTIONS_FILE, "w", encoding="utf-8")

    for m in METHODS:
        n_lvl = 1 if m == "B4" else 3
        print(f"\n--- Method {m} (num_levels={n_lvl}) ---")

        for s in SEEDS:
            f1_by_ckpt = {}
            em_by_ckpt = {}
            acc_by_ckpt = {}
            logprob_by_ckpt = {}

            for ckpt_id in CHECKPOINTS:
                snap_path = SNAP_DIR / f"rq4_{m}_seed_{s}_{ckpt_id}.pt"
                if not snap_path.exists():
                    print(f"[WARNING] Missing {snap_path}")
                    continue

                model = StructureAlignedHopeLM(num_levels=n_lvl, device=DEVICE, torch_dtype=DTYPE, enable_cms=True)
                sd = torch.load(snap_path, map_location=DEVICE, weights_only=False)
                model.cms.load_state_dict(sd["cms_state_dict"], strict=False)
                model.eval()

                l1, l2, l3 = get_level_norms(model)

                f1_list, em_list, logprob_list = [], [], []
                legacy_correct = 0

                for q in d0_questions:
                    q_id = q.get("q_id", "Q")
                    prompt = f"Question: {q['question']}\nAnswer:"
                    gt = q["ground_truth"]

                    enc = model.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
                    raw_ans = generate_greedy(model, enc, max_new_tokens=24)
                    norm_ans = normalize_answer(raw_ans)

                    f1 = compute_token_f1(raw_ans, gt)
                    em = compute_exact_match(raw_ans, gt)
                    logprob = compute_target_logprob(model, prompt, gt)

                    f1_list.append(f1)
                    em_list.append(em)
                    logprob_list.append(logprob)

                    # Legacy keyword matching
                    if any(kw.lower() in raw_ans.lower() for kw in q.get("keywords", [])):
                        legacy_correct += 1

                    jsonl_f.write(json.dumps({
                        "exp_id": "RQ4", "method": m, "seed": s, "interval": ckpt_id,
                        "question_id": q_id, "question_text": q["question"], "ground_truth": gt,
                        "raw_prediction": raw_ans, "normalized_prediction": norm_ans,
                        "token_f1": round(f1, 4), "exact_match": round(em, 4), "target_logprob": round(logprob, 4),
                        "legacy_correct": 1.0 if any(kw.lower() in raw_ans.lower() for kw in q.get("keywords", [])) else 0.0,
                        "l1_norm": l1, "l2_norm": l2, "l3_norm": l3
                    }, ensure_ascii=False) + "\n")
                    jsonl_f.flush()

                mean_f1 = float(sum(f1_list) / max(1, len(f1_list)))
                mean_em = float(sum(em_list) / max(1, len(em_list)))
                mean_logprob = float(sum(logprob_list) / max(1, len(logprob_list)))
                legacy_acc = float((legacy_correct / max(1, len(d0_questions))) * 100.0)

                f1_by_ckpt[ckpt_id] = round(mean_f1, 4)
                em_by_ckpt[ckpt_id] = round(mean_em, 4)
                logprob_by_ckpt[ckpt_id] = round(mean_logprob, 4)
                acc_by_ckpt[ckpt_id] = round(legacy_acc, 2)

            # Compute forgetting deltas
            if "D0" in f1_by_ckpt:
                f1_0 = f1_by_ckpt["D0"]
                acc_0 = acc_by_ckpt["D0"]

                # Continuous Token F1 forgetting
                f1_delta_5 = f1_0 - f1_by_ckpt.get("D0_plus5", f1_0)
                f1_delta_10 = f1_0 - f1_by_ckpt.get("D0_plus10", f1_0)
                f1_delta_20 = f1_0 - f1_by_ckpt.get("D0_plus20", f1_0)

                # Legacy Accuracy forgetting
                acc_delta_5 = acc_0 - acc_by_ckpt.get("D0_plus5", acc_0)
                acc_delta_10 = acc_0 - acc_by_ckpt.get("D0_plus10", acc_0)
                acc_delta_20 = acc_0 - acc_by_ckpt.get("D0_plus20", acc_0)

                run_entry = {
                    "method": m,
                    "seed": s,
                    "primary_metric": "token_f1",
                    "f1_initial_d0": f1_0,
                    "f1_plus_5": f1_by_ckpt.get("D0_plus5", 0.0),
                    "f1_plus_10": f1_by_ckpt.get("D0_plus10", 0.0),
                    "f1_plus_20": f1_by_ckpt.get("D0_plus20", 0.0),
                    "forgetting_f1_f5": round(f1_delta_5, 4),
                    "forgetting_f1_f10": round(f1_delta_10, 4),
                    "forgetting_f1_f20": round(f1_delta_20, 4),
                    "legacy_acc_initial_d0": acc_0,
                    "legacy_acc_plus_5": acc_by_ckpt.get("D0_plus5", 0.0),
                    "legacy_acc_plus_10": acc_by_ckpt.get("D0_plus10", 0.0),
                    "legacy_acc_plus_20": acc_by_ckpt.get("D0_plus20", 0.0),
                    "legacy_forgetting_f5": round(acc_delta_5, 2),
                    "legacy_forgetting_f10": round(acc_delta_10, 2),
                    "legacy_forgetting_f20": round(acc_delta_20, 2),
                }
                results["runs"].append(run_entry)
                print(f"  Seed {s} -> F1 D0: {f1_0:.4f} | F1 +20: {f1_by_ckpt.get('D0_plus20', 0.0):.4f} | Forgetting Delta F20: {f1_delta_20:+.4f}")

    jsonl_f.close()
    print(f"\n[SUCCESS] RQ4 Raw Predictions exported to {RAW_PREDICTIONS_FILE}")

    # Save to targets
    out_targets = [
        ROOT_DIR / "training_output_final" / "raw_results" / "rq4_external_results.json",
        ROOT_DIR / "training_output_final" / "RQ4_sequential" / "rq4_external_results.json",
        ROOT_DIR / "results" / "phase4_4" / "RQ4" / "rq4_external_results.json",
        ROOT_DIR / "results" / "phase4_2" / "rq4_external_results.json"
    ]
    for p in out_targets:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    print(f"[SUCCESS] RQ4 JSON results saved to {out_targets[0]}")


if __name__ == "__main__":
    run_rq4_evaluation()

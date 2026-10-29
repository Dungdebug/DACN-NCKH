"""
Phase 4.4 / 4.5 Runner for RQ2: Structure-Aligned (P1) vs Fixed-Token (B5) CMS Update Schedule.
Strictly conforms to Stage 2 & 3 remediation requirements:
- Multi-token generation: max_new_tokens=32, do_sample=False (Greedy decoding).
- Academic evaluation: SQuAD/QASPER Standard Token-level F1, Exact Match (EM), Target Sequence Logprob.
- Output raw predictions JSONL: results/raw_predictions/rq2_raw_predictions.jsonl.
- Statistical testing: Paired t-test, Wilcoxon signed-rank test, 95% Bootstrap CI, Cohen's d.
"""

import sys
import os
import re
import json
import copy
import time
import string
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Tuple

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hope_attention.sa_cms import StructureAlignedHopeLM
from src.evaluation.pretrained_benchmarks import QASPERDocumentBenchmark, LongHealthDocumentBenchmark
from src.hybrid_qa.vietnamese_final_corpus import (
    get_vietnamese_final_documents,
    get_vietnamese_final_questions,
)

CKPT_DIR_P1 = ROOT_DIR / "checkpoints" / "phase4_1"
CKPT_DIR_B5 = ROOT_DIR / "checkpoints" / "phase4_1"

OUT_DIR_RAW = ROOT_DIR / "results" / "raw_predictions"
OUT_DIR_RAW.mkdir(parents=True, exist_ok=True)
RAW_PREDICTIONS_FILE = OUT_DIR_RAW / "rq2_raw_predictions.jsonl"

SEEDS = [42, 43, 44]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def normalize_answer(s: str) -> str:
    """Standard academic answer normalization (SQuAD / QASPER)."""
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
    """Computes Bag-of-Words Token-level F1."""
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
    """Computes binary exact match on normalized answers."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_target_logprob(model, prompt: str, target_str: str) -> float:
    """Computes total log-probability of target sequence given prompt."""
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
    """Returns L2 norms of Level 1, Level 2, and Level 3 CMS parameters."""
    norms = [0.0, 0.0, 0.0]
    if hasattr(model, "cms") and model.cms is not None:
        chain = getattr(model.cms, "chain", None)
        if chain is not None and hasattr(chain, "blocks"):
            for i, b in enumerate(chain.blocks):
                if i < 3:
                    n_sq = sum(torch.norm(p.float()).item()**2 for p in b.parameters())
                    norms[i] = round(float(n_sq**0.5), 4)
    return norms[0], norms[1], norms[2]


def generate_greedy(model, prompt_ids: torch.Tensor, max_new_tokens: int = 32) -> str:
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


def generate_greedy_with_prefix(model, query_ids: torch.Tensor, prefix_kv, max_new_tokens: int = 32) -> str:
    """Generates autoregressively with greedy decoding using cached document prefix KV."""
    with torch.no_grad():
        kv = copy.deepcopy(prefix_kv)
        out = model.backbone.model(input_ids=query_ids, past_key_values=kv, use_cache=True, return_dict=True)
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


def bootstrap_ci(diffs: np.ndarray, n_bootstrap: int = 1000, ci: float = 95.0, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    means = []
    n = len(diffs)
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (100.0 - ci) / 2.0)
    upper = np.percentile(means, 100.0 - (100.0 - ci) / 2.0)
    return float(lower), float(upper)


def run_rq2(dry_run: bool = False):
    print("=" * 80)
    print("RE-EVALUATING RQ2: STRUCTURE-ALIGNED (P1) VS FIXED-TOKEN (B5)")
    print(f"Device: {DEVICE} | DType: {DTYPE} | Max New Tokens: 32 (Greedy)")
    print(f"Output Raw Predictions JSONL: {RAW_PREDICTIONS_FILE}")
    print("=" * 80)

    # 1. Load Datasets
    vn_docs_raw = get_vietnamese_final_documents()
    vn_docs = {d["document_id"]: d for d in vn_docs_raw}
    vn_questions = get_vietnamese_final_questions()
    lh_bench = LongHealthDocumentBenchmark(num_documents=5)
    q_bench = QASPERDocumentBenchmark(num_documents=10)

    total_items_per_seed = len(vn_questions) + sum(len(d["questions"]) for d in lh_bench.clinical_records) + len(q_bench.documents)
    total_eval_items = total_items_per_seed * len(SEEDS)
    print(f"[*] Total Evaluation Samples: N = {total_eval_items} items across {len(SEEDS)} seeds.")

    if dry_run:
        print("[DRY RUN] Verification successful. Exiting.")
        return 0

    # Open raw predictions JSONL file
    raw_jsonl_f = open(RAW_PREDICTIONS_FILE, "w", encoding="utf-8")

    all_item_pairs = []

    for s in SEEDS:
        print(f"\n--- Evaluating Seed {s} ---")

        # Load P1
        model_p1 = StructureAlignedHopeLM(num_levels=3, device=DEVICE, torch_dtype=DTYPE, enable_cms=True)
        ckpt_p1 = CKPT_DIR_P1 / f"cms_3lvl_seed_{s}.pt"
        if ckpt_p1.exists():
            sd_p1 = torch.load(ckpt_p1, map_location=DEVICE, weights_only=False)
            model_p1.cms.load_state_dict(sd_p1["cms_state_dict"], strict=False)
        model_p1.eval()

        # Load B5
        model_b5 = StructureAlignedHopeLM(num_levels=3, device=DEVICE, torch_dtype=DTYPE, enable_cms=True)
        ckpt_b5 = CKPT_DIR_B5 / f"cms_3lvl_seed_{s}.pt"
        if ckpt_b5.exists():
            sd_b5 = torch.load(ckpt_b5, map_location=DEVICE, weights_only=False)
            model_b5.cms.load_state_dict(sd_b5["cms_state_dict"], strict=False)
        model_b5.eval()

        # Get level norms
        p1_l1, p1_l2, p1_l3 = get_level_norms(model_p1)
        b5_l1, b5_l2, b5_l3 = get_level_norms(model_b5)

        # -------------------------------------------------------------
        # 1. Evaluate Vietnamese QA (350 questions from 20 documents)
        # -------------------------------------------------------------
        # Group questions by document
        doc_to_questions: Dict[str, List[Dict[str, Any]]] = {}
        for q in vn_questions:
            d_id = q.get("document_id") or q.get("doc_id")
            doc_to_questions.setdefault(d_id, []).append(q)

        for d_id, q_list in doc_to_questions.items():
            doc = vn_docs.get(d_id)
            if not doc:
                continue
            doc_text = doc.get("raw_text") or doc.get("text", "")

            # Ingest doc into P1
            model_p1.reset_memory()
            model_p1.ingest_structured_document(doc_text, schedule_mode="structure", seed=s)

            # Ingest doc into B5
            model_b5.reset_memory()
            model_b5.ingest_structured_document(doc_text, schedule_mode="fixed_token", seed=s)

            # Precompute document context KV cache once per document
            ctx_prompt = f"Context: {doc_text}\n"
            enc_ctx_p1 = model_p1.tokenizer(ctx_prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                out_ctx_p1 = model_p1.backbone.model(input_ids=enc_ctx_p1, use_cache=True, return_dict=True)
                p1_prefix_kv = out_ctx_p1.past_key_values

            enc_ctx_b5 = model_b5.tokenizer(ctx_prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                out_ctx_b5 = model_b5.backbone.model(input_ids=enc_ctx_b5, use_cache=True, return_dict=True)
                b5_prefix_kv = out_ctx_b5.past_key_values

            for q in q_list:
                q_id = q.get("question_id") or q.get("q_id")
                q_text = q["question"]
                gt = q.get("ground_truth_answer") or q.get("ground_truth") or ""
                q_prompt = f"Question: {q_text}\nAnswer:"
                full_prompt = f"{ctx_prompt}{q_prompt}"

                enc_q_p1 = model_p1.tokenizer(q_prompt, return_tensors="pt").input_ids.to(DEVICE)
                raw_pred_p1 = generate_greedy_with_prefix(model_p1, enc_q_p1, p1_prefix_kv, max_new_tokens=32)
                f1_p1 = compute_token_f1(raw_pred_p1, gt)
                em_p1 = compute_exact_match(raw_pred_p1, gt)
                logprob_p1 = compute_target_logprob(model_p1, full_prompt, gt)
                legacy_hit_p1 = 1.0 if (len(raw_pred_p1) > 0 and (raw_pred_p1[:1].lower() in gt.lower() or gt.lower() in raw_pred_p1[:1].lower())) else 0.0

                enc_q_b5 = model_b5.tokenizer(q_prompt, return_tensors="pt").input_ids.to(DEVICE)
                raw_pred_b5 = generate_greedy_with_prefix(model_b5, enc_q_b5, b5_prefix_kv, max_new_tokens=32)
                f1_b5 = compute_token_f1(raw_pred_b5, gt)
                em_b5 = compute_exact_match(raw_pred_b5, gt)
                logprob_b5 = compute_target_logprob(model_b5, full_prompt, gt)
                legacy_hit_b5 = 1.0 if (len(raw_pred_b5) > 0 and (raw_pred_b5[:1].lower() in gt.lower() or gt.lower() in raw_pred_b5[:1].lower())) else 0.0

                # Write JSONL for P1
                raw_jsonl_f.write(json.dumps({
                    "exp_id": "RQ2", "method": "P1", "seed": s, "dataset": "VietnameseQA",
                    "question_id": q_id, "question_text": q_text, "ground_truth": gt,
                    "raw_prediction": raw_pred_p1, "normalized_prediction": normalize_answer(raw_pred_p1),
                    "token_f1": round(f1_p1, 4), "exact_match": round(em_p1, 4), "target_logprob": round(logprob_p1, 4),
                    "legacy_single_token_hit": legacy_hit_p1,
                    "l1_norm": p1_l1, "l2_norm": p1_l2, "l3_norm": p1_l3
                }, ensure_ascii=False) + "\n")

                # Write JSONL for B5
                raw_jsonl_f.write(json.dumps({
                    "exp_id": "RQ2", "method": "B5", "seed": s, "dataset": "VietnameseQA",
                    "question_id": q_id, "question_text": q_text, "ground_truth": gt,
                    "raw_prediction": raw_pred_b5, "normalized_prediction": normalize_answer(raw_pred_b5),
                    "token_f1": round(f1_b5, 4), "exact_match": round(em_b5, 4), "target_logprob": round(logprob_b5, 4),
                    "legacy_single_token_hit": legacy_hit_b5,
                    "l1_norm": b5_l1, "l2_norm": b5_l2, "l3_norm": b5_l3
                }, ensure_ascii=False) + "\n")
                raw_jsonl_f.flush()

                if len(all_item_pairs) % 25 == 0:
                    print(f"  [Progress] Seed {s} | Item {len(all_item_pairs)}/{total_items_per_seed} | P1 F1: {f1_p1:.2f} | B5 F1: {f1_b5:.2f}")

                all_item_pairs.append({
                    "dataset": "VietnameseQA",
                    "item_id": q_id,
                    "seed": s,
                    "question_text": q_text,
                    "ground_truth": gt,
                    "raw_pred_p1": raw_pred_p1,
                    "raw_pred_b5": raw_pred_b5,
                    "p1_token_f1": round(f1_p1, 4),
                    "b5_token_f1": round(f1_b5, 4),
                    "diff_token_f1": round(f1_p1 - f1_b5, 4),
                    "p1_em": em_p1,
                    "b5_em": em_b5,
                    "legacy_score_p1": legacy_hit_p1,
                    "legacy_score_b5": legacy_hit_b5,
                    "legacy_diff": legacy_hit_p1 - legacy_hit_b5
                })

        # -------------------------------------------------------------
        # 2. Evaluate LongHealth (20 MCQs per seed)
        # -------------------------------------------------------------
        for rec in lh_bench.clinical_records:
            model_p1.reset_memory()
            model_p1.ingest_structured_document(rec["text"], schedule_mode="structure", seed=s)

            model_b5.reset_memory()
            model_b5.ingest_structured_document(rec["text"], schedule_mode="fixed_token", seed=s)

            for q_idx, q in enumerate(rec["questions"]):
                options_str = "\n".join([f"{k}. {v}" for k, v in q["options"].items()])
                prompt = f"{rec['text']}\n\nQuestion: {q['question']}\nOptions:\n{options_str}\nAnswer:"
                gt_letter = q["correct_letter"]

                # P1 MCQ
                enc_p1 = model_p1.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
                with torch.no_grad():
                    logits_p1, _ = model_p1.forward(enc_p1)
                    probs_p1 = torch.softmax(logits_p1[0, -1, :], dim=-1)
                    opt_p1 = {k: probs_p1[model_p1.tokenizer.encode(f" {k}", add_special_tokens=False)[0]].item() for k in ["A", "B", "C", "D"]}
                    pred_p1 = max(opt_p1.keys(), key=lambda k: opt_p1[k])
                f1_p1 = 1.0 if pred_p1 == gt_letter else 0.0
                em_p1 = f1_p1

                # B5 MCQ
                enc_b5 = model_b5.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
                with torch.no_grad():
                    logits_b5, _ = model_b5.forward(enc_b5)
                    probs_b5 = torch.softmax(logits_b5[0, -1, :], dim=-1)
                    opt_b5 = {k: probs_b5[model_b5.tokenizer.encode(f" {k}", add_special_tokens=False)[0]].item() for k in ["A", "B", "C", "D"]}
                    pred_b5 = max(opt_b5.keys(), key=lambda k: opt_b5[k])
                f1_b5 = 1.0 if pred_b5 == gt_letter else 0.0
                em_b5 = f1_b5

                q_id = f"{rec['doc_id']}_Q{q_idx}"
                raw_jsonl_f.write(json.dumps({
                    "exp_id": "RQ2", "method": "P1", "seed": s, "dataset": "LongHealth",
                    "question_id": q_id, "question_text": q["question"], "ground_truth": gt_letter,
                    "raw_prediction": pred_p1, "normalized_prediction": pred_p1,
                    "token_f1": f1_p1, "exact_match": em_p1, "target_logprob": 0.0,
                    "legacy_single_token_hit": f1_p1,
                    "l1_norm": p1_l1, "l2_norm": p1_l2, "l3_norm": p1_l3
                }) + "\n")

                raw_jsonl_f.write(json.dumps({
                    "exp_id": "RQ2", "method": "B5", "seed": s, "dataset": "LongHealth",
                    "question_id": q_id, "question_text": q["question"], "ground_truth": gt_letter,
                    "raw_prediction": pred_b5, "normalized_prediction": pred_b5,
                    "token_f1": f1_b5, "exact_match": em_b5, "target_logprob": 0.0,
                    "legacy_single_token_hit": f1_b5,
                    "l1_norm": b5_l1, "l2_norm": b5_l2, "l3_norm": b5_l3
                }) + "\n")

                all_item_pairs.append({
                    "dataset": "LongHealth",
                    "item_id": q_id,
                    "seed": s,
                    "question_text": q["question"],
                    "ground_truth": gt_letter,
                    "raw_pred_p1": pred_p1,
                    "raw_pred_b5": pred_b5,
                    "p1_token_f1": f1_p1,
                    "b5_token_f1": f1_b5,
                    "diff_token_f1": f1_p1 - f1_b5,
                    "p1_em": em_p1,
                    "b5_em": em_b5,
                    "legacy_score_p1": f1_p1,
                    "legacy_score_b5": f1_b5,
                    "legacy_diff": f1_p1 - f1_b5
                })

        # -------------------------------------------------------------
        # 3. Evaluate QASPER (10 docs per seed)
        # -------------------------------------------------------------
        for d_idx, doc_tuple in enumerate(q_bench.documents):
            context, question, answer = doc_tuple
            qa_prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

            # P1
            model_p1.reset_memory()
            model_p1.ingest_structured_document(context, schedule_mode="structure", seed=s)
            enc_p1 = model_p1.tokenizer(qa_prompt, return_tensors="pt").input_ids.to(DEVICE)
            raw_p1 = generate_greedy(model_p1, enc_p1, max_new_tokens=32)
            f1_p1 = compute_token_f1(raw_p1, answer)
            em_p1 = compute_exact_match(raw_p1, answer)

            # B5
            model_b5.reset_memory()
            model_b5.ingest_structured_document(context, schedule_mode="fixed_token", seed=s)
            enc_b5 = model_b5.tokenizer(qa_prompt, return_tensors="pt").input_ids.to(DEVICE)
            raw_b5 = generate_greedy(model_b5, enc_b5, max_new_tokens=32)
            f1_b5 = compute_token_f1(raw_b5, answer)
            em_b5 = compute_exact_match(raw_b5, answer)

            q_id = f"QASPER_DOC_{d_idx}"
            raw_jsonl_f.write(json.dumps({
                "exp_id": "RQ2", "method": "P1", "seed": s, "dataset": "QASPER",
                "question_id": q_id, "question_text": question, "ground_truth": answer,
                "raw_prediction": raw_p1, "normalized_prediction": normalize_answer(raw_p1),
                "token_f1": round(f1_p1, 4), "exact_match": round(em_p1, 4), "target_logprob": 0.0,
                "legacy_single_token_hit": round(f1_p1, 4),
                "l1_norm": p1_l1, "l2_norm": p1_l2, "l3_norm": p1_l3
            }) + "\n")

            raw_jsonl_f.write(json.dumps({
                "exp_id": "RQ2", "method": "B5", "seed": s, "dataset": "QASPER",
                "question_id": q_id, "question_text": question, "ground_truth": answer,
                "raw_prediction": raw_b5, "normalized_prediction": normalize_answer(raw_b5),
                "token_f1": round(f1_b5, 4), "exact_match": round(em_b5, 4), "target_logprob": 0.0,
                "legacy_single_token_hit": round(f1_b5, 4),
                "l1_norm": b5_l1, "l2_norm": b5_l2, "l3_norm": b5_l3
            }) + "\n")

            all_item_pairs.append({
                "dataset": "QASPER",
                "item_id": q_id,
                "seed": s,
                "question_text": question,
                "ground_truth": answer,
                "raw_pred_p1": raw_p1,
                "raw_pred_b5": raw_b5,
                "p1_token_f1": round(f1_p1, 4),
                "b5_token_f1": round(f1_b5, 4),
                "diff_token_f1": round(f1_p1 - f1_b5, 4),
                "p1_em": em_p1,
                "b5_em": em_b5,
                "legacy_score_p1": round(f1_p1, 4),
                "legacy_score_b5": round(f1_b5, 4),
                "legacy_diff": round(f1_p1 - f1_b5, 4)
            })

    raw_jsonl_f.close()
    print(f"\n[SUCCESS] Exported raw predictions to {RAW_PREDICTIONS_FILE}")

    # -------------------------------------------------------------
    # 4. Statistical Analysis
    # -------------------------------------------------------------
    # Primary metric: Token F1
    p1_f1 = np.array([p["p1_token_f1"] for p in all_item_pairs])
    b5_f1 = np.array([p["b5_token_f1"] for p in all_item_pairs])
    diff_f1 = p1_f1 - b5_f1

    mean_p1_f1 = float(np.mean(p1_f1))
    mean_b5_f1 = float(np.mean(b5_f1))
    mean_diff_f1 = float(np.mean(diff_f1))
    std_diff_f1 = float(np.std(diff_f1, ddof=1)) if len(diff_f1) > 1 else 0.0

    t_stat_f1, p_val_f1 = scipy.stats.ttest_rel(p1_f1, b5_f1)
    w_stat_f1, w_pval_f1 = scipy.stats.wilcoxon(p1_f1, b5_f1, zero_method="wilcox")
    ci_low_f1, ci_high_f1 = bootstrap_ci(diff_f1, n_bootstrap=1000, ci=95.0)
    cohens_d_f1 = float(mean_diff_f1 / std_diff_f1) if std_diff_f1 > 1e-8 else 0.0

    # Secondary metric: Legacy single token hit
    p1_leg = np.array([p["legacy_score_p1"] for p in all_item_pairs])
    b5_leg = np.array([p["legacy_score_b5"] for p in all_item_pairs])
    diff_leg = p1_leg - b5_leg
    mean_p1_leg = float(np.mean(p1_leg))
    mean_b5_leg = float(np.mean(b5_leg))
    mean_diff_leg = float(np.mean(diff_leg))

    stats_summary = {
        "N_items": len(all_item_pairs),
        "primary_metric": "token_f1",
        "mean_p1_token_f1": round(mean_p1_f1, 4),
        "mean_b5_token_f1": round(mean_b5_f1, 4),
        "mean_difference_token_f1": round(mean_diff_f1, 4),
        "std_difference_token_f1": round(std_diff_f1, 4),
        "t_statistic": round(float(t_stat_f1), 4),
        "t_pvalue": float(p_val_f1),
        "wilcoxon_statistic": round(float(w_stat_f1), 4),
        "wilcoxon_pvalue": float(w_pval_f1),
        "bootstrap_95_ci": [round(ci_low_f1, 4), round(ci_high_f1, 4)],
        "cohens_d": round(cohens_d_f1, 4),
        "legacy_single_token_hit": {
            "mean_p1": round(mean_p1_leg, 4),
            "mean_b5": round(mean_b5_leg, 4),
            "mean_diff": round(mean_diff_leg, 4)
        }
    }

    final_results = {
        "experiment": "RQ2_Structure_vs_Fixed_Token",
        "sample_size_N": len(all_item_pairs),
        "seeds": SEEDS,
        "statistics": stats_summary,
        "item_level_pairs": all_item_pairs
    }

    # Save to all required output locations
    out_files = [
        ROOT_DIR / "results" / "phase4_2" / "rq2_external_results.json",
        ROOT_DIR / "training_output_final" / "raw_results" / "rq2_external_results.json",
        ROOT_DIR / "results" / "phase4_4" / "RQ2" / "rq2_external_results.json"
    ]
    for out_p in out_files:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RQ2 RE-EVALUATION STATISTICAL SUMMARY (OFFICIAL TOKEN F1):")
    print(f"  Sample Size N: {len(all_item_pairs)}")
    print(f"  P1 Mean Token F1: {mean_p1_f1:.4f} | B5 Mean Token F1: {mean_b5_f1:.4f}")
    print(f"  Mean Difference (P1 - B5): {mean_diff_f1:+.4f} (95% CI: [{ci_low_f1:.4f}, {ci_high_f1:.4f}])")
    print(f"  Paired t-test: t = {t_stat_f1:.4f}, p = {p_val_f1:.6e}")
    print(f"  Wilcoxon test: W = {w_stat_f1:.4f}, p = {w_pval_f1:.6e}")
    print(f"  Cohen's d: {cohens_d_f1:.4f}")
    print(f"  [Legacy single-token hit for reference: P1 = {mean_p1_leg:.4f}, B5 = {mean_b5_leg:.4f}, Diff = {mean_diff_leg:+.4f}]")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2 Verified Re-Evaluation Runner")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without inference")
    args = parser.parse_args()
    sys.exit(run_rq2(dry_run=args.dry_run))

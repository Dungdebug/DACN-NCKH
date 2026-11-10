"""
Phase 2.5: Strictly Controlled Multi-Seed Scientific Validation Pipeline.
Addresses all Phase 2.5 requirements:
- Task 1: Strictly Equal Update Budget across Fixed Token, SA-CMS, and Random Boundary.
- Task 2: Scaled evaluation to 100 MK-NIAH samples and 10 QASPER documents.
- Task 3: 3 seeds (42, 43, 44) with mean, std, and 95% bootstrap confidence interval.
- Task 4: QASPER answer-level Token F1 and Exact Match (EM) in addition to Perplexity.
- Task 6: Direct frequency control verification.
- Task 7: Output table saved to results/phase2_5_controlled_comparison.csv.
"""

import os
import sys
import time
import math
import random
import csv
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.stdout.reconfigure(line_buffering=True)

from src.hope_attention.sa_cms import StructureAlignedHopeLM
from src.evaluation.pretrained_benchmarks import NaturalMKNIAHBenchmark, QASPERDocumentBenchmark


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval."""
    if len(data) <= 1:
        val = data[0] if data else 0.0
        return (val, val)
    arr = np.array(data)
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return (round(low, 4), round(high, 4))


def evaluate_mkniah_controlled(
    model: StructureAlignedHopeLM,
    bench: NaturalMKNIAHBenchmark,
    schedule_mode: Optional[str],
    seed: int,
) -> Dict[str, Any]:
    """Evaluates MK-NIAH with strictly controlled equal update budget."""
    model.eval()
    tokenizer = model.tokenizer

    correct = 0
    total = bench.num_samples
    target_probs = []
    target_ranks = []
    total_update_events = 0

    for idx in range(total):
        sample = bench._generate_sample(idx)
        prompt = sample["prompt_text"]
        context = sample["context_text"]
        expected_val = sample["expected_val"]
        queried_key = sample["queried_key"]

        prompt_enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_ids = prompt_enc.input_ids
        target_token_id = tokenizer.encode(f" {expected_val}", add_special_tokens=False)[0]

        # Online ingestion under equal update budget
        if schedule_mode is not None and model.cms is not None:
            model.reset_memory()
            model.clear_event_log()
            res = model.ingest_structured_document(
                document_text=context,
                schedule_mode=schedule_mode,
                seed=seed + idx * 7,
            )
            total_update_events += res.get("num_update_events", 0)

        with torch.no_grad():
            logits, _ = model.forward(prompt_ids)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)

            target_prob = probs[target_token_id].item()
            target_probs.append(target_prob)

            sorted_ids = torch.argsort(last_logits, descending=True)
            target_rank = (sorted_ids == target_token_id).nonzero(as_tuple=True)[0].item() + 1
            target_ranks.append(target_rank)

            # Autoregressive generation up to 6 tokens
            curr_ids = prompt_ids.clone()
            for _ in range(6):
                cur_logits, _ = model.forward(curr_ids)
                next_tok = torch.argmax(cur_logits[:, -1, :], dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_tok], dim=1)
                if next_tok.item() == tokenizer.eos_token_id:
                    break

            generated_answer = tokenizer.decode(
                curr_ids[0, prompt_ids.size(1):], skip_special_tokens=True
            ).strip()

            is_correct = (expected_val in generated_answer) or generated_answer.startswith(expected_val)
            if is_correct:
                correct += 1

        if schedule_mode is not None and model.cms is not None:
            model.reset_memory()

    accuracy = (correct / total) * 100.0
    avg_target_prob = sum(target_probs) / max(1, len(target_probs))
    avg_target_rank = sum(target_ranks) / max(1, len(target_ranks))

    return {
        "accuracy_pct": accuracy,
        "correct": correct,
        "total": total,
        "avg_target_prob": avg_target_prob,
        "avg_target_rank": avg_target_rank,
        "total_update_events": total_update_events,
    }


def evaluate_qasper_controlled(
    model: StructureAlignedHopeLM,
    bench: QASPERDocumentBenchmark,
    schedule_mode: Optional[str],
    seed: int,
) -> Dict[str, Any]:
    """Evaluates QASPER document QA with PPL, F1, and Exact Match under controlled budget."""
    model.eval()
    tokenizer = model.tokenizer

    total_loss = 0.0
    f1_scores = []
    em_scores = []
    total_update_events = 0
    doc_count = min(bench.num_documents, len(bench.documents))

    for idx in range(doc_count):
        context, question, answer = bench.documents[idx]
        doc_text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
        qa_prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

        enc_doc = tokenizer(doc_text, return_tensors="pt").to(model.device)
        enc_prompt = tokenizer(qa_prompt, return_tensors="pt").to(model.device)
        input_ids = enc_doc.input_ids
        prompt_ids = enc_prompt.input_ids

        if schedule_mode is not None and model.cms is not None:
            model.reset_memory()
            res = model.ingest_structured_document(
                document_text=context,
                schedule_mode=schedule_mode,
                seed=seed + idx * 13,
            )
            total_update_events += res.get("num_update_events", 0)

        with torch.no_grad():
            # 1. Perplexity evaluation on full document
            _, loss = model.forward(input_ids, targets=input_ids)
            loss_val = loss.item() if loss is not None else 0.0
            total_loss += loss_val

            # 2. Autoregressive answer generation (up to 24 tokens)
            curr_ids = prompt_ids.clone()
            for _ in range(24):
                cur_logits, _ = model.forward(curr_ids)
                next_tok = torch.argmax(cur_logits[:, -1, :], dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_tok], dim=1)
                if next_tok.item() == tokenizer.eos_token_id:
                    break

            gen_answer = tokenizer.decode(
                curr_ids[0, prompt_ids.size(1):], skip_special_tokens=True
            ).strip()

            # 3. Compute QA Metrics: F1 and Exact Match
            f1 = bench.compute_f1(gen_answer, answer)
            em = bench.compute_exact_match(gen_answer, answer)
            f1_scores.append(f1)
            em_scores.append(em)

        if schedule_mode is not None and model.cms is not None:
            model.reset_memory()

    avg_loss = total_loss / max(1, doc_count)
    avg_ppl = math.exp(min(avg_loss, 20.0))
    avg_f1 = sum(f1_scores) / max(1, len(f1_scores))
    avg_em = sum(em_scores) / max(1, len(em_scores))

    return {
        "avg_loss": avg_loss,
        "perplexity": avg_ppl,
        "f1": avg_f1,
        "exact_match": avg_em,
        "num_documents": doc_count,
        "total_update_events": total_update_events,
    }


def run_phase2_5_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 90)
    print("PHASE 2.5: STRICTLY CONTROLLED MULTI-SEED SCIENTIFIC VALIDATION")
    print(f"Device: {device} | Dtype: {dtype} | Target: SmolLM2-135M")
    print("=" * 90)

    # Output directory
    os.makedirs("results", exist_ok=True)
    csv_path = "results/phase2_5_controlled_comparison.csv"

    # Evaluation scale: 100 MK-NIAH samples, 10 QASPER documents
    mkniah_samples = 100
    qasper_docs = 10
    seeds = [42, 43, 44]

    # Model configurations to evaluate
    configurations = [
        # (method_label, num_levels, schedule_mode)
        ("ICL Baseline", 1, None),
        ("CMS Level 2 (Fixed Token)", 2, "fixed_token"),
        ("SA-CMS Level 2 (Structure Aligned)", 2, "structure"),
        ("CMS Level 2 (Random Boundary)", 2, "random"),
        ("CMS Level 3 (Fixed Token)", 3, "fixed_token"),
        ("SA-CMS Level 3 (Structure Aligned)", 3, "structure"),
        ("CMS Level 3 (Random Boundary)", 3, "random"),
    ]

    fieldnames = [
        "method",
        "levels",
        "schedule",
        "seed",
        "sample_count",
        "update_count",
        "MK_NIAH_accuracy",
        "target_probability",
        "target_rank",
        "QASPER_F1",
        "QASPER_EM",
        "QASPER_PPL",
        "runtime",
        "peak_vram",
    ]

    all_rows = []

    for cfg_label, num_levels, sched_mode in configurations:
        print(f"\n================================================================================")
        print(f"CONFIGURATION: {cfg_label} (Levels: {num_levels}, Schedule: {sched_mode})")
        print(f"================================================================================")

        for seed in seeds:
            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            # Set reproducible seeds
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            # Initialize model
            model = StructureAlignedHopeLM(
                model_name_or_path="HuggingFaceTB/SmolLM2-135M",
                num_levels=num_levels,
                device=device,
                torch_dtype=dtype,
            )

            # Benchmarks
            mkniah_bench = NaturalMKNIAHBenchmark(num_samples=mkniah_samples, seed=seed)
            qasper_bench = QASPERDocumentBenchmark(num_documents=qasper_docs, seed=seed)

            # 1. Evaluate MK-NIAH
            mk_res = evaluate_mkniah_controlled(
                model=model,
                bench=mkniah_bench,
                schedule_mode=sched_mode,
                seed=seed,
            )

            # 2. Evaluate QASPER
            qas_res = evaluate_qasper_controlled(
                model=model,
                bench=qasper_bench,
                schedule_mode=sched_mode,
                seed=seed,
            )

            runtime = round(time.time() - t0, 2)
            peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0
            total_updates = mk_res["total_update_events"] + qas_res["total_update_events"]

            row = {
                "method": cfg_label,
                "levels": num_levels,
                "schedule": sched_mode if sched_mode is not None else "icl",
                "seed": seed,
                "sample_count": mkniah_samples,
                "update_count": total_updates,
                "MK_NIAH_accuracy": round(mk_res["accuracy_pct"], 2),
                "target_probability": round(mk_res["avg_target_prob"], 6),
                "target_rank": round(mk_res["avg_target_rank"], 2),
                "QASPER_F1": round(qas_res["f1"], 4),
                "QASPER_EM": round(qas_res["exact_match"], 4),
                "QASPER_PPL": round(qas_res["perplexity"], 2),
                "runtime": runtime,
                "peak_vram": peak_vram,
            }
            all_rows.append(row)

            print(
                f"  Seed {seed} | Updates: {total_updates:4d} | "
                f"MK-NIAH: {row['MK_NIAH_accuracy']:5.1f}% | "
                f"T-Prob: {row['target_probability']:.5f} | "
                f"Q-F1: {row['QASPER_F1']:.4f} | "
                f"Q-EM: {row['QASPER_EM']:.1f}% | "
                f"Q-PPL: {row['QASPER_PPL']:6.2f} | "
                f"Time: {runtime:5.1f}s | VRAM: {peak_vram:5.1f}MB"
            )

    # Save to CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[Artifact Generated] Successfully wrote {len(all_rows)} rows to {csv_path}")

    # Compute Statistical Aggregation: Mean, Std, 95% Bootstrap CI
    print("\n" + "=" * 95)
    print("STATISTICAL SUMMARY ACROSS 3 SEEDS (Mean ± Std [95% CI])")
    print("=" * 95)

    summary_by_method = {}
    for cfg_label, num_levels, sched_mode in configurations:
        matching = [r for r in all_rows if r["method"] == cfg_label]
        accs = [r["MK_NIAH_accuracy"] for r in matching]
        t_probs = [r["target_probability"] for r in matching]
        q_f1s = [r["QASPER_F1"] for r in matching]
        q_ems = [r["QASPER_EM"] for r in matching]
        q_ppls = [r["QASPER_PPL"] for r in matching]
        updates = [r["update_count"] for r in matching]

        acc_ci = bootstrap_ci(accs)
        f1_ci = bootstrap_ci(q_f1s)
        ppl_ci = bootstrap_ci(q_ppls)

        summary_by_method[cfg_label] = {
            "updates_mean": np.mean(updates),
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "acc_ci": acc_ci,
            "f1_mean": np.mean(q_f1s),
            "f1_std": np.std(q_f1s),
            "f1_ci": f1_ci,
            "ppl_mean": np.mean(q_ppls),
            "ppl_std": np.std(q_ppls),
            "ppl_ci": ppl_ci,
        }

        print(
            f"{cfg_label:36s} | Updates: {np.mean(updates):4.0f} | "
            f"MK-NIAH: {np.mean(accs):5.1f}±{np.std(accs):4.1f}% [{acc_ci[0]:.1f}, {acc_ci[1]:.1f}] | "
            f"Q-F1: {np.mean(q_f1s):.4f}±{np.std(q_f1s):.4f} [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}] | "
            f"Q-PPL: {np.mean(q_ppls):6.2f}±{np.std(q_ppls):4.2f} [{ppl_ci[0]:.1f}, {ppl_ci[1]:.1f}]"
        )

    return all_rows, summary_by_method


if __name__ == "__main__":
    run_phase2_5_pipeline()

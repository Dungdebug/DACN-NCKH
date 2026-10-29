"""
Build Master Result Tables for Phase 4.3
Generates CSV tables in results/phase4_3/tables/:
- 01_rq1_summary.csv
- 02_rq2_summary.csv
- 03_rq3_summary.csv
- 04_rq4_status.csv
- 05_rq5_cost.csv
- 06_method_dataset_matrix.csv
- 07_checkpoint_matrix.csv
- 08_missing_experiments.csv
"""

import json
import csv
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT_DIR / "results" / "phase4_3" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def compute_bootstrap_ci(data, iterations=1000, ci=0.95, seed=42):
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    arr = np.array(data, dtype=float)
    boot_means = []
    n = len(arr)
    for _ in range(iterations):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return float(np.mean(arr)), round(lower, 4), round(upper, 4)

def build_rq1_table():
    raw_path = ROOT_DIR / "results" / "phase4_1" / "rq1" / "rq1_raw_results.json"
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    out_file = TABLES_DIR / "01_rq1_summary.csv"
    rows = []
    
    # Process for MK-NIAH, QASPER, LongHealth
    benchmarks = ["MK-NIAH", "QASPER", "LongHealth"]
    methods = ["B1", "B4", "B5", "P1"]
    
    for bench in benchmarks:
        pri_metric = "token_f1" if bench == "QASPER" else "accuracy"
        for m in methods:
            runs = [r for r in data["runs"] if r["benchmark"] == bench and r["method"] == m]
            
            # Extract item-level data across all seeds
            item_vals = []
            prob_vals = []
            seed_means = {}
            for r in runs:
                s = r["seed"]
                if bench == "MK-NIAH":
                    s_vals = [float(item["is_correct"]) for item in r["detailed"]]
                    s_probs = [float(item["target_prob"]) for item in r["detailed"]]
                elif bench == "QASPER":
                    s_vals = [float(item["f1"]) for item in r["detailed"]]
                    s_probs = [float(item["target_prob"]) for item in r["detailed"]]
                elif bench == "LongHealth":
                    s_vals = [float(item["is_correct"]) for item in r["detailed"]]
                    s_probs = [float(item["target_prob"]) for item in r["detailed"]]
                item_vals.extend(s_vals)
                prob_vals.extend(s_probs)
                seed_means[s] = float(np.mean(s_vals))
            
            mean_val, ci_low, ci_high = compute_bootstrap_ci(item_vals, iterations=1000)
            mean_prob, prob_ci_low, prob_ci_high = compute_bootstrap_ci(prob_vals, iterations=1000)
            std_val = float(np.std(item_vals)) if item_vals else 0.0
            
            rows.append({
                "method": m,
                "dataset": bench,
                "primary_metric": pri_metric,
                "n_observations": len(item_vals),
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "ci_95_bootstrap_lower": ci_low,
                "ci_95_bootstrap_upper": ci_high,
                "avg_target_prob": round(mean_prob, 4),
                "target_prob_ci_95_lower": prob_ci_low,
                "target_prob_ci_95_upper": prob_ci_high,
                "seed_42_mean": round(seed_means.get(42, 0.0), 4),
                "seed_43_mean": round(seed_means.get(43, 0.0), 4),
                "seed_44_mean": round(seed_means.get(44, 0.0), 4),
                "evaluation_condition": "Context-evicted (parameter-only)" if m != "B1" else "Prompt-context (truncated to 512 tokens)",
                "source_artifact": "results/phase4_1/rq1/rq1_raw_results.json"
            })
            
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "dataset", "primary_metric", "n_observations", "mean", "std",
            "ci_95_bootstrap_lower", "ci_95_bootstrap_upper", "avg_target_prob",
            "target_prob_ci_95_lower", "target_prob_ci_95_upper",
            "seed_42_mean", "seed_43_mean", "seed_44_mean",
            "evaluation_condition", "source_artifact"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved: {out_file}")

def build_rq2_table():
    audit_path = ROOT_DIR / "results" / "phase4_2" / "rq2_statistical_audit.json"
    with open(audit_path, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    
    recon = audit_data["reconciled_statistics"]
    out_file = TABLES_DIR / "02_rq2_summary.csv"
    
    # Compute bootstrap CI on pooled items
    items = audit_data.get("items", [])
    diffs_all = [it["difference_p1_minus_b5"] for it in items]
    diffs_q = [it["difference_p1_minus_b5"] for it in items if it["dataset"] == "QASPER"]
    diffs_lh = [it["difference_p1_minus_b5"] for it in items if it["dataset"] == "LongHealth"]
    
    _, boot_all_l, boot_all_u = compute_bootstrap_ci(diffs_all, iterations=1000)
    _, boot_q_l, boot_q_u = compute_bootstrap_ci(diffs_q, iterations=1000)
    _, boot_lh_l, boot_lh_u = compute_bootstrap_ci(diffs_lh, iterations=1000)
    
    rows = [
        {
            "comparison_or_ablation": "P1 vs B5 (Pooled All)",
            "dataset": "QASPER + LongHealth",
            "metric": "token_f1 (QASPER) || accuracy (LongHealth)",
            "n_observations": recon["pooled_all_90_items"]["n_observations"],
            "mean_p1": recon["pooled_all_90_items"]["p1_mean"],
            "mean_b5": recon["pooled_all_90_items"]["b5_mean"],
            "mean_diff": recon["pooled_all_90_items"]["mean_difference"],
            "std_diff": recon["pooled_all_90_items"]["std_difference"],
            "t_statistic": recon["pooled_all_90_items"]["t_statistic"],
            "t_pvalue": round(recon["pooled_all_90_items"]["t_pvalue"], 4),
            "wilcoxon_stat": recon["pooled_all_90_items"]["wilcoxon_stat"],
            "wilcoxon_pvalue": round(recon["pooled_all_90_items"]["wilcoxon_pvalue"], 4),
            "cohens_d": recon["pooled_all_90_items"]["cohens_d"],
            "ci_95_bootstrap_lower": boot_all_l,
            "ci_95_bootstrap_upper": boot_all_u,
            "status": "COMPLETED",
            "statistical_significance": recon["pooled_all_90_items"]["statistical_significance"],
            "source_artifact": "results/phase4_2/rq2_statistical_audit.json"
        },
        {
            "comparison_or_ablation": "P1 vs B5 (QASPER Only)",
            "dataset": "QASPER",
            "metric": "token_f1",
            "n_observations": recon["qasper_only_30_items"]["n_observations"],
            "mean_p1": recon["qasper_only_30_items"]["p1_mean"],
            "mean_b5": recon["qasper_only_30_items"]["b5_mean"],
            "mean_diff": recon["qasper_only_30_items"]["mean_difference"],
            "std_diff": recon["qasper_only_30_items"]["std_difference"],
            "t_statistic": recon["qasper_only_30_items"]["t_statistic"],
            "t_pvalue": round(recon["qasper_only_30_items"]["t_pvalue"], 4),
            "wilcoxon_stat": recon["qasper_only_30_items"]["wilcoxon_stat"],
            "wilcoxon_pvalue": round(recon["qasper_only_30_items"]["wilcoxon_pvalue"], 4),
            "cohens_d": recon["qasper_only_30_items"]["cohens_d"],
            "ci_95_bootstrap_lower": boot_q_l,
            "ci_95_bootstrap_upper": boot_q_u,
            "status": "COMPLETED",
            "statistical_significance": recon["qasper_only_30_items"]["statistical_significance"],
            "source_artifact": "results/phase4_2/rq2_statistical_audit.json"
        },
        {
            "comparison_or_ablation": "P1 vs B5 (LongHealth Only)",
            "dataset": "LongHealth",
            "metric": "accuracy",
            "n_observations": recon["longhealth_only_60_items"]["n_observations"],
            "mean_p1": recon["longhealth_only_60_items"]["p1_mean"],
            "mean_b5": recon["longhealth_only_60_items"]["b5_mean"],
            "mean_diff": recon["longhealth_only_60_items"]["mean_difference"],
            "std_diff": recon["longhealth_only_60_items"]["std_difference"],
            "t_statistic": recon["longhealth_only_60_items"]["t_statistic"],
            "t_pvalue": round(recon["longhealth_only_60_items"]["t_pvalue"], 4),
            "wilcoxon_stat": recon["longhealth_only_60_items"]["wilcoxon_stat"],
            "wilcoxon_pvalue": round(recon["longhealth_only_60_items"]["wilcoxon_pvalue"], 4),
            "cohens_d": recon["longhealth_only_60_items"]["cohens_d"],
            "ci_95_bootstrap_lower": boot_lh_l,
            "ci_95_bootstrap_upper": boot_lh_u,
            "status": "COMPLETED",
            "statistical_significance": recon["longhealth_only_60_items"]["statistical_significance"],
            "source_artifact": "results/phase4_2/rq2_statistical_audit.json"
        },
        {
            "comparison_or_ablation": "A2 (Two-Level SA-CMS)",
            "dataset": "QASPER / LongHealth",
            "metric": "token_f1 (QASPER=0.0970) / accuracy (LH=0.2667)",
            "n_observations": 90,
            "mean_p1": 0.2101,
            "mean_b5": "N/A",
            "mean_diff": "N/A",
            "std_diff": "N/A",
            "t_statistic": "N/A",
            "t_pvalue": "N/A",
            "wilcoxon_stat": "N/A",
            "wilcoxon_pvalue": "N/A",
            "cohens_d": "N/A",
            "ci_95_bootstrap_lower": "N/A",
            "ci_95_bootstrap_upper": "N/A",
            "status": "COMPLETED",
            "statistical_significance": "Ablation Baseline (2 levels)",
            "source_artifact": "results/phase4_1/non_training/rq2/rq2_summary.json"
        },
        {
            "comparison_or_ablation": "A1 (Random Boundary CMS)",
            "dataset": "QASPER / LongHealth",
            "metric": "token_f1 / accuracy",
            "n_observations": 0,
            "mean_p1": "N/A",
            "mean_b5": "N/A",
            "mean_diff": "N/A",
            "std_diff": "N/A",
            "t_statistic": "N/A",
            "t_pvalue": "N/A",
            "wilcoxon_stat": "N/A",
            "wilcoxon_pvalue": "N/A",
            "cohens_d": "N/A",
            "ci_95_bootstrap_lower": "N/A",
            "ci_95_bootstrap_upper": "N/A",
            "status": "NEED_EXTERNAL_GPU",
            "statistical_significance": "Awaiting external GPU execution",
            "source_artifact": "external_gpu/phase4_2/A1/config_a1.yaml"
        },
        {
            "comparison_or_ablation": "A3 (SA-CMS Additive Ungated)",
            "dataset": "QASPER / LongHealth",
            "metric": "token_f1 / accuracy",
            "n_observations": 0,
            "mean_p1": "N/A",
            "mean_b5": "N/A",
            "mean_diff": "N/A",
            "std_diff": "N/A",
            "t_statistic": "N/A",
            "t_pvalue": "N/A",
            "wilcoxon_stat": "N/A",
            "wilcoxon_pvalue": "N/A",
            "cohens_d": "N/A",
            "ci_95_bootstrap_lower": "N/A",
            "ci_95_bootstrap_upper": "N/A",
            "status": "NEED_EXTERNAL_GPU",
            "statistical_significance": "Awaiting external GPU execution",
            "source_artifact": "external_gpu/phase4_2/A3/config_a3.yaml"
        }
    ]
    
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "comparison_or_ablation", "dataset", "metric", "n_observations",
            "mean_p1", "mean_b5", "mean_diff", "std_diff", "t_statistic", "t_pvalue",
            "wilcoxon_stat", "wilcoxon_pvalue", "cohens_d",
            "ci_95_bootstrap_lower", "ci_95_bootstrap_upper", "status",
            "statistical_significance", "source_artifact"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved: {out_file}")

def build_rq3_table():
    summary_path = ROOT_DIR / "results" / "phase4_1" / "non_training" / "rq3" / "rq3_summary.json"
    cost_path = ROOT_DIR / "results" / "phase4_1" / "non_training" / "rq5" / "rq5_cost_profile.csv"
    
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    vram_map = {}
    with open(cost_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            vram_map[r["method"]] = float(r["peak_vram_mb"])
    
    out_file = TABLES_DIR / "03_rq3_summary.csv"
    methods = ["B1", "B2", "B5", "P1", "P2"]
    rows = []
    
    for m in methods:
        m_info = data["methods"][m]
        rows.append({
            "method": m,
            "dataset": "Vietnamese_Context_Evicted_100",
            "n_samples": 100,
            "token_f1": m_info["token_f1_mean"],
            "token_f1_sd": m_info["token_f1_std"],
            "token_f1_ci_lower": m_info["token_f1_bootstrap_ci_95"][0],
            "token_f1_ci_upper": m_info["token_f1_bootstrap_ci_95"][1],
            "exact_match": 0.0,
            "faithfulness_pct": m_info["faithfulness_pct"],
            "faithfulness_ci_lower": m_info["faithfulness_bootstrap_ci_95"][0],
            "faithfulness_ci_upper": m_info["faithfulness_bootstrap_ci_95"][1],
            "correct_refusal_pct": m_info["correct_refusal_pct"],
            "false_refusal_pct": m_info["false_refusal_pct"],
            "false_answer_pct": m_info["false_answer_pct"],
            "avg_latency_ms": m_info["avg_latency_ms"],
            "peak_vram_mb": vram_map.get(m, 0.0),
            "retrieval_status": "BM25 (k1=1.5, b=0.75, top_k=5, thresh=3.0)" if m in ("B2", "P2") else "None",
            "refusal_controller_status": "Calibrated (min_score=3.0, cov=0.35)" if m in ("B2", "P2") else "None",
            "source_artifact": "results/phase4_1/non_training/rq3/rq3_summary.json"
        })
        
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "dataset", "n_samples", "token_f1", "token_f1_sd",
            "token_f1_ci_lower", "token_f1_ci_upper", "exact_match",
            "faithfulness_pct", "faithfulness_ci_lower", "faithfulness_ci_upper",
            "correct_refusal_pct", "false_refusal_pct", "false_answer_pct",
            "avg_latency_ms", "peak_vram_mb", "retrieval_status",
            "refusal_controller_status", "source_artifact"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved: {out_file}")

def build_rq4_table():
    out_file = TABLES_DIR / "04_rq4_status.csv"
    rows = [
        {
            "method": "B4",
            "dataset": "Incremental_QASPER_RQ4 (INC_DOC_000 to INC_DOC_020)",
            "schedule": "D0 -> +5 -> +10 -> +20",
            "status": "NEED_EXTERNAL_GPU",
            "reason": "Sequential memory updates require online gradient steps; training prohibited on current hardware",
            "required_hardware": "RTX 3050 (6GB/8GB)",
            "target_checkpoints": "rq4_B4_seed_{42,43,44}_{D0,D0_plus5,D0_plus10,D0_plus20}.pt",
            "corpus_manifest": "external_gpu/phase4_2/RQ4/corpus_manifest.json",
            "snapshot_schedule": "external_gpu/phase4_2/RQ4/snapshot_schedule.json",
            "evaluation_script": "external_gpu/phase4_2/RQ4/eval_rq4_retention.py",
            "source_artifact": "results/phase4_1/non_training/rq4/rq4_status.json"
        },
        {
            "method": "B5",
            "dataset": "Incremental_QASPER_RQ4 (INC_DOC_000 to INC_DOC_020)",
            "schedule": "D0 -> +5 -> +10 -> +20",
            "status": "NEED_EXTERNAL_GPU",
            "reason": "Sequential memory updates require online gradient steps; training prohibited on current hardware",
            "required_hardware": "RTX 3050 (6GB/8GB)",
            "target_checkpoints": "rq4_B5_seed_{42,43,44}_{D0,D0_plus5,D0_plus10,D0_plus20}.pt",
            "corpus_manifest": "external_gpu/phase4_2/RQ4/corpus_manifest.json",
            "snapshot_schedule": "external_gpu/phase4_2/RQ4/snapshot_schedule.json",
            "evaluation_script": "external_gpu/phase4_2/RQ4/eval_rq4_retention.py",
            "source_artifact": "results/phase4_1/non_training/rq4/rq4_status.json"
        },
        {
            "method": "P1",
            "dataset": "Incremental_QASPER_RQ4 (INC_DOC_000 to INC_DOC_020)",
            "schedule": "D0 -> +5 -> +10 -> +20",
            "status": "NEED_EXTERNAL_GPU",
            "reason": "Sequential memory updates require online gradient steps; training prohibited on current hardware",
            "required_hardware": "RTX 3050 (6GB/8GB)",
            "target_checkpoints": "rq4_P1_seed_{42,43,44}_{D0,D0_plus5,D0_plus10,D0_plus20}.pt",
            "corpus_manifest": "external_gpu/phase4_2/RQ4/corpus_manifest.json",
            "snapshot_schedule": "external_gpu/phase4_2/RQ4/snapshot_schedule.json",
            "evaluation_script": "external_gpu/phase4_2/RQ4/eval_rq4_retention.py",
            "source_artifact": "results/phase4_1/non_training/rq4/rq4_status.json"
        }
    ]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "dataset", "schedule", "status", "reason",
            "required_hardware", "target_checkpoints", "corpus_manifest",
            "snapshot_schedule", "evaluation_script", "source_artifact"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved: {out_file}")

def build_rq5_table():
    src_file = ROOT_DIR / "results" / "phase4_1" / "non_training" / "rq5" / "rq5_cost_profile.csv"
    out_file = TABLES_DIR / "05_rq5_cost.csv"
    rows = []
    with open(src_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "method": r["method"],
                "dataset": r["dataset"],
                "checkpoint_size_mb": r["checkpoint_size_mb"],
                "retrieval_time_ms": r["retrieval_time_ms"],
                "memory_load_time_ms": r["memory_load_time_ms"],
                "generation_time_ms": r["generation_time_ms"],
                "total_query_latency_ms": r["total_query_latency_ms"],
                "peak_vram_mb": r["peak_vram_mb"],
                "ingest_time_per_1k_tokens_sec": r["existing_ingest_time_per_1k_tokens_sec"],
                "evaluation_type": r["evaluation_type"],
                "source_artifact": "results/phase4_1/non_training/rq5/rq5_cost_profile.csv"
            })
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "dataset", "checkpoint_size_mb", "retrieval_time_ms",
            "memory_load_time_ms", "generation_time_ms", "total_query_latency_ms",
            "peak_vram_mb", "ingest_time_per_1k_tokens_sec", "evaluation_type",
            "source_artifact"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved: {out_file}")

def build_method_dataset_matrix():
    out_file = TABLES_DIR / "06_method_dataset_matrix.csv"
    methods = ["B1", "B2", "B4", "B5", "P1", "P2", "A1", "A2", "A3"]
    
    rows = [
        {"method": "B1", "QASPER": "COMPLETED", "LongHealth": "COMPLETED", "MK-NIAH": "COMPLETED", "Vietnamese": "COMPLETED", "Incremental_QASPER": "NOT_RUN", "source_artifact": "results/phase4_1/rq1/rq1_raw_results.json"},
        {"method": "B2", "QASPER": "NOT_RUN", "LongHealth": "NOT_RUN", "MK-NIAH": "NOT_RUN", "Vietnamese": "COMPLETED", "Incremental_QASPER": "NOT_RUN", "source_artifact": "results/phase4_1/vietnamese/vietnamese_raw_results.json"},
        {"method": "B4", "QASPER": "COMPLETED", "LongHealth": "COMPLETED", "MK-NIAH": "COMPLETED", "Vietnamese": "COMPLETED", "Incremental_QASPER": "NEED_EXTERNAL_GPU", "source_artifact": "results/phase4_1/rq1/rq1_raw_results.json"},
        {"method": "B5", "QASPER": "COMPLETED", "LongHealth": "COMPLETED", "MK-NIAH": "COMPLETED", "Vietnamese": "COMPLETED", "Incremental_QASPER": "NEED_EXTERNAL_GPU", "source_artifact": "results/phase4_1/rq1/rq1_raw_results.json"},
        {"method": "P1", "QASPER": "COMPLETED", "LongHealth": "COMPLETED", "MK-NIAH": "COMPLETED", "Vietnamese": "COMPLETED", "Incremental_QASPER": "NEED_EXTERNAL_GPU", "source_artifact": "results/phase4_1/rq1/rq1_raw_results.json"},
        {"method": "P2", "QASPER": "NOT_RUN", "LongHealth": "NOT_RUN", "MK-NIAH": "NOT_RUN", "Vietnamese": "COMPLETED", "Incremental_QASPER": "NOT_RUN", "source_artifact": "results/phase4_1/rq3/rq3_raw_results.json"},
        {"method": "A1", "QASPER": "NEED_EXTERNAL_GPU", "LongHealth": "NEED_EXTERNAL_GPU", "MK-NIAH": "NOT_RUN", "Vietnamese": "NOT_RUN", "Incremental_QASPER": "NOT_RUN", "source_artifact": "external_gpu/phase4_2/A1/config_a1.yaml"},
        {"method": "A2", "QASPER": "COMPLETED", "LongHealth": "COMPLETED", "MK-NIAH": "NOT_RUN", "Vietnamese": "NOT_RUN", "Incremental_QASPER": "NOT_RUN", "source_artifact": "results/phase4_1/non_training/rq2/rq2_summary.json"},
        {"method": "A3", "QASPER": "NEED_EXTERNAL_GPU", "LongHealth": "NEED_EXTERNAL_GPU", "MK-NIAH": "NOT_RUN", "Vietnamese": "NOT_RUN", "Incremental_QASPER": "NOT_RUN", "source_artifact": "external_gpu/phase4_2/A3/config_a3.yaml"},
    ]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "QASPER", "LongHealth", "MK-NIAH", "Vietnamese", "Incremental_QASPER", "source_artifact"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved: {out_file}")

def build_checkpoint_matrix():
    out_file = TABLES_DIR / "07_checkpoint_matrix.csv"
    ckpt_inv_path = ROOT_DIR / "results" / "phase4_1" / "checkpoint_inventory.json"
    with open(ckpt_inv_path, "r", encoding="utf-8") as f:
        inv = json.load(f)
    
    rows = []
    # Loop methods in inventory
    for m, m_data in inv["methods"].items():
        if m in ("B1", "B2"):
            rows.append({
                "method": m,
                "seed": "All (42, 43, 44)",
                "checkpoint_path": "None (frozen backbone)",
                "file_size_mb": 0.0,
                "trainable_parameters": 0,
                "total_parameters": 134515008,
                "status": "AVAILABLE",
                "source_artifact": "results/phase4_1/checkpoint_inventory.json"
            })
        elif m == "B3":
            rows.append({
                "method": "B3",
                "seed": "None",
                "checkpoint_path": "None",
                "file_size_mb": 0.0,
                "trainable_parameters": 0,
                "total_parameters": 0,
                "status": "EXCLUDED",
                "source_artifact": "configs/phase4_experiment.yaml"
            })
        else:
            seeds = m_data.get("seeds", {})
            for s, s_info in seeds.items():
                rows.append({
                    "method": m,
                    "seed": s,
                    "checkpoint_path": s_info.get("checkpoint_path", "N/A"),
                    "file_size_mb": s_info.get("file_size_mb", 0.0),
                    "trainable_parameters": s_info.get("trainable_parameters", 0),
                    "total_parameters": s_info.get("total_parameters", 0),
                    "status": s_info.get("status", "AVAILABLE"),
                    "source_artifact": "results/phase4_1/checkpoint_inventory.json"
                })
    
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "seed", "checkpoint_path", "file_size_mb",
            "trainable_parameters", "total_parameters", "status", "source_artifact"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved: {out_file}")

def build_missing_experiments_table():
    out_file = TABLES_DIR / "08_missing_experiments.csv"
    rows = [
        {
            "experiment_id": "EXP_A1_RANDOM_BOUNDARY",
            "rq": "RQ2",
            "method": "A1",
            "dataset": "QASPER / LongHealth",
            "seeds": "42, 43, 44",
            "missing_reason": "Training 200 samples with randomized chunk boundaries required; prohibited on current machine",
            "required_hardware": "RTX 3050 (6GB/8GB)",
            "handoff_package_path": "external_gpu/phase4_2/A1/",
            "status": "NEED_EXTERNAL_GPU",
            "source_artifact": "external_gpu/phase4_2/A1/config_a1.yaml"
        },
        {
            "experiment_id": "EXP_A3_ADDITIVE_UNGATED",
            "rq": "RQ2",
            "method": "A3",
            "dataset": "QASPER / LongHealth",
            "seeds": "42, 43, 44",
            "missing_reason": "Training 200 samples with additive residual architecture required; prohibited on current machine",
            "required_hardware": "RTX 3050 (6GB/8GB)",
            "handoff_package_path": "external_gpu/phase4_2/A3/",
            "status": "NEED_EXTERNAL_GPU",
            "source_artifact": "external_gpu/phase4_2/A3/config_a3.yaml"
        },
        {
            "experiment_id": "EXP_RQ4_SEQUENTIAL_INGESTION",
            "rq": "RQ4",
            "method": "B4, B5, P1",
            "dataset": "Incremental_QASPER_RQ4 (INC_DOC_000 to INC_DOC_020)",
            "seeds": "42, 43, 44",
            "missing_reason": "Sequential document ingestion updates require online gradient steps across D0 -> +5 -> +10 -> +20",
            "required_hardware": "RTX 3050 (6GB/8GB)",
            "handoff_package_path": "external_gpu/phase4_2/RQ4/",
            "status": "NEED_EXTERNAL_GPU",
            "source_artifact": "external_gpu/phase4_2/RQ4/config_rq4.yaml"
        }
    ]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment_id", "rq", "method", "dataset", "seeds",
            "missing_reason", "required_hardware", "handoff_package_path",
            "status", "source_artifact"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    print("Building Phase 4.3 Master Result Tables...")
    build_rq1_table()
    build_rq2_table()
    build_rq3_table()
    build_rq4_table()
    build_rq5_table()
    build_method_dataset_matrix()
    build_checkpoint_matrix()
    build_missing_experiments_table()
    print("All master tables successfully generated!")

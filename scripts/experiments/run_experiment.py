"""
Micro-scale Experiment for Phase 1: Investigating the Effect of Memory Levels on Document QA & MK-NIAH.
Corresponds to Figure 7 in arXiv:2512.24695v1 (Left: MK-NIAH, Right: QASPER Perplexity).
"""

import os
import sys
import json
import time
import torch
import yaml
from typing import Dict, Any, List

from src.hope_attention.model import HopeAttentionLM
from src.evaluation.mk_niah import MKNIAHBenchmark
from src.evaluation.doc_qa import DocumentQABenchmark
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import setup_logger, save_metrics_json


def run_micro_experiment(config_path: str = "configs/micro_experiment.yaml") -> Dict[str, Any]:
    logger = setup_logger(name="micro_experiment", log_dir="logs")
    logger.info("=== STARTING PHASE 1 REPRODUCTION EXPERIMENT ===")

    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["experiment"].get("device") == "cuda" else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seed for reproducibility
    seed = cfg["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]
    cms_configs = cfg["cms_configurations"]

    # Setup benchmarks
    mk_niah_bench = MKNIAHBenchmark(
        vocab_size=model_cfg["vocab_size"],
        seed=seed,
    )
    doc_qa_bench = DocumentQABenchmark(
        vocab_size=model_cfg["vocab_size"],
        seed=seed,
    )

    results = {
        "experiment_name": cfg["experiment"]["name"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "seed": seed,
        "configurations": [],
    }

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for config_item in cms_configs:
        name = config_item["name"]
        num_levels = config_item["num_levels"]
        lowest_chunk = config_item["lowest_chunk_size"]
        base_lr = config_item["base_lr"]

        logger.info(f"--- Running evaluation for: {name} (num_levels={num_levels}) ---")

        # Initialize model with this level configuration
        model = HopeAttentionLM(
            vocab_size=model_cfg["vocab_size"],
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            d_ff=model_cfg["d_ff"],
            num_levels=num_levels,
            lowest_chunk_size=lowest_chunk,
            base_lr=base_lr,
            dropout=model_cfg["dropout"],
            max_seq_len=model_cfg["max_seq_len"],
        ).to(device)

        # 1. Evaluate on MK-NIAH
        logger.info(f"[{name}] Running MK-NIAH benchmark...")
        mk_res = mk_niah_bench.evaluate_model(
            model=model,
            num_samples=eval_cfg["mk_niah"]["num_samples"],
            context_length=eval_cfg["mk_niah"]["context_length"],
            num_needles=eval_cfg["mk_niah"]["num_needles"],
            enable_online_cms=(num_levels > 1),
            device=device,
        )
        logger.info(f"[{name}] MK-NIAH Accuracy: {mk_res['accuracy_pct']:.2f}% ({mk_res['correct']}/{mk_res['num_samples']}), Avg Target Prob: {mk_res.get('avg_target_prob', 0.0):.6f}")

        # 2. Evaluate on Document QA (Perplexity and Answer Loss)
        logger.info(f"[{name}] Running Document QA benchmark...")
        doc_res = doc_qa_bench.evaluate_model(
            model=model,
            num_docs=eval_cfg["doc_qa"]["num_docs"],
            doc_len=eval_cfg["doc_qa"]["doc_len"],
            qa_len=eval_cfg["doc_qa"]["qa_len"],
            enable_online_cms=(num_levels > 1),
            device=device,
        )
        logger.info(f"[{name}] Doc QA Avg Loss: {doc_res['avg_loss']:.4f}, Perplexity: {doc_res['perplexity']:.4f}")

        # 3. Save Checkpoint
        ckpt_path = os.path.join("checkpoints", f"{name}.pt")
        save_checkpoint(
            model=model,
            save_path=ckpt_path,
            config=config_item,
            metrics={"mk_niah": mk_res, "doc_qa": doc_res},
        )
        logger.info(f"[{name}] Saved checkpoint to {ckpt_path}")

        # Record metrics
        entry = {
            "name": name,
            "num_levels": num_levels,
            "chunk_sizes": model.blocks[0].cms.schedule.chunk_sizes,
            "frequencies": model.blocks[0].cms.schedule.frequencies,
            "mk_niah_accuracy_pct": mk_res["accuracy_pct"],
            "mk_niah_correct": mk_res["correct"],
            "mk_niah_total": mk_res["num_samples"],
            "mk_niah_avg_target_prob": mk_res.get("avg_target_prob", 0.0),
            "doc_qa_avg_loss": doc_res["avg_loss"],
            "doc_qa_perplexity": doc_res["perplexity"],
            "checkpoint_path": ckpt_path,
        }
        results["configurations"].append(entry)

    # Save final results JSON
    results_path = os.path.join("results", "experiment_results.json")
    save_metrics_json(results, results_path)
    logger.info(f"Saved complete experiment results to: {results_path}")

    # Print summary table
    logger.info("=== EXPERIMENT RESULTS SUMMARY TABLE ===")
    logger.info(f"{'Config Name':<25} | {'Levels':<6} | {'Chunk Sizes':<20} | {'MK-NIAH Acc (%)':<15} | {'Doc QA PPL':<12}")
    logger.info("-" * 88)
    for c in results["configurations"]:
        chunks_str = str(c["chunk_sizes"])
        logger.info(f"{c['name']:<25} | {c['num_levels']:<6} | {chunks_str:<20} | {c['mk_niah_accuracy_pct']:<15.2f} | {c['doc_qa_perplexity']:<12.4f}")

    return results


if __name__ == "__main__":
    run_micro_experiment()

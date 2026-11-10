"""
Track 2 Reproduction Experiment Runner on Pretrained SmolLM2-135M.
Evaluates baseline configurations (Level 1 ICL, Level 2 CMS, Level 3 CMS, Level 4 CMS)
on MK-NIAH (RULER format) and QASPER (Document QA Perplexity).
Paper reference: arXiv:2512.24695v1 (Section 7.1, 7.3, 8.3 & Section 9.1, Figure 7).
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import json
import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hope_attention.pretrained_hope import PretrainedHopeLM
from src.evaluation.pretrained_benchmarks import NaturalMKNIAHBenchmark, QASPERDocumentBenchmark
from src.utils.logger import setup_logger


def run_track2_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = str(ROOT_DIR / "logs")
    filename = f"pretrained_experiment_{timestamp}.log"
    logger = setup_logger("PretrainedReproduction", log_dir=log_dir, filename=filename)

    logger.info("=" * 80)
    logger.info("STARTING TRACK 2 PRETRAINED BASELINE REPRODUCTION (SMOLLM2-135M)")
    logger.info("Paper Reference: arXiv:2512.24695v1 (Section 9.1 & Figure 7)")
    logger.info("Hardware: NVIDIA GeForce GTX 1650 Ti (4GB VRAM)")
    logger.info("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using compute device: {device}")

    # Set seed
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Define Configurations
    configurations = [
        {
            "name": "level_1_icl_baseline",
            "num_levels": 1,
            "chunk_sizes": [64],
            "description": "Pure In-Context Learning (ICL) baseline — No CMS adaptation (paper Section 9.1)",
        },
        {
            "name": "level_2_cms",
            "num_levels": 2,
            "chunk_sizes": [64, 32],
            "description": "2-level Continuum Memory System (lowest chunk size 64, higher 32)",
        },
        {
            "name": "level_3_cms",
            "num_levels": 3,
            "chunk_sizes": [64, 32, 16],
            "description": "3-level Continuum Memory System (multi-timescale [64, 32, 16])",
        },
        {
            "name": "level_4_cms",
            "num_levels": 4,
            "chunk_sizes": [64, 32, 16, 8],
            "description": "4-level Continuum Memory System (multi-timescale [64, 32, 16, 8])",
        },
    ]

    # Initialize Benchmarks
    mkniah_bench = NaturalMKNIAHBenchmark(num_samples=10, num_keys=3, seed=seed)
    qasper_bench = QASPERDocumentBenchmark(num_documents=5, seed=seed)

    experiment_results = []
    checkpoint_dir = ROOT_DIR / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for cfg in configurations:
        cfg_name = cfg["name"]
        num_levels = cfg["num_levels"]
        chunk_sizes = cfg["chunk_sizes"]

        logger.info("\n" + "=" * 60)
        logger.info(f"RUNNING CONFIGURATION: {cfg_name} (num_levels={num_levels}, chunks={chunk_sizes})")
        logger.info(f"Description: {cfg['description']}")
        logger.info("=" * 60)

        start_time = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Initialize model
        model = PretrainedHopeLM(
            model_name_or_path="HuggingFaceTB/SmolLM2-135M",
            num_levels=num_levels,
            chunk_sizes=chunk_sizes,
            device=device,
        )

        total_params = sum(p.numel() for p in model.parameters())
        cms_params = sum(p.numel() for p in model.get_cms_parameters())
        initial_vram = torch.cuda.memory_allocated() / (1024 ** 2)
        logger.info(f"Model loaded: Total Params: {total_params:,} | Trainable CMS Params: {cms_params:,}")
        logger.info(f"Initial VRAM Allocated: {initial_vram:.2f} MB")

        # 1. Evaluate MK-NIAH (RULER format)
        enable_cms = (num_levels > 1)
        logger.info(f"[{cfg_name}] Running MK-NIAH Benchmark (10 samples, online_cms={enable_cms})...")
        mkniah_res = mkniah_bench.evaluate_model(model, enable_online_cms=enable_cms, logger=logger)
        logger.info(
            f"[{cfg_name}] MK-NIAH Accuracy: {mkniah_res['accuracy_pct']:.2f}% "
            f"({mkniah_res['correct']}/{mkniah_res['total']}), "
            f"Avg Target Prob: {mkniah_res['avg_target_prob']:.6f}"
        )

        # 2. Evaluate QASPER (Document QA Perplexity)
        logger.info(f"[{cfg_name}] Running QASPER Benchmark (5 documents, online_cms={enable_cms})...")
        qasper_res = qasper_bench.evaluate_model(model, enable_online_cms=enable_cms, logger=logger)
        logger.info(
            f"[{cfg_name}] QASPER Avg Loss: {qasper_res['avg_loss']:.4f}, "
            f"Perplexity: {qasper_res['perplexity']:.4f}"
        )

        # 3. Save Checkpoint
        ckpt_path = str(checkpoint_dir / f"pretrained_{cfg_name}.pt")
        model.save_cms_checkpoint(ckpt_path)
        logger.info(f"[{cfg_name}] Saved CMS checkpoint to: {ckpt_path}")

        elapsed_time = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"[{cfg_name}] Completed in {elapsed_time:.2f}s | Peak VRAM: {peak_vram:.2f} MB")

        res_record = {
            "name": cfg_name,
            "num_levels": num_levels,
            "chunk_sizes": chunk_sizes,
            "total_params": total_params,
            "cms_params": cms_params,
            "mkniah_accuracy_pct": mkniah_res["accuracy_pct"],
            "mkniah_correct": mkniah_res["correct"],
            "mkniah_total": mkniah_res["total"],
            "mkniah_avg_target_prob": mkniah_res["avg_target_prob"],
            "qasper_avg_loss": qasper_res["avg_loss"],
            "qasper_perplexity": qasper_res["perplexity"],
            "runtime_seconds": round(elapsed_time, 2),
            "peak_vram_mb": round(peak_vram, 2),
            "checkpoint_path": ckpt_path,
        }
        experiment_results.append(res_record)

        # Free model from GPU memory
        del model
        torch.cuda.empty_cache()

    # Save summary results
    results_dir = ROOT_DIR / "results"
    os.makedirs(results_dir, exist_ok=True)
    summary_path = results_dir / "pretrained_experiment_results.json"
    summary_payload = {
        "experiment_name": "track2_pretrained_smollm2_reproduction",
        "timestamp": timestamp,
        "device": device,
        "seed": seed,
        "backbone_model": "HuggingFaceTB/SmolLM2-135M",
        "configurations": experiment_results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved complete results to: {summary_path}")

    # Print summary table
    logger.info("\n" + "=" * 95)
    logger.info("TRACK 2 REPRODUCTION RESULTS SUMMARY TABLE (SmolLM2-135M)")
    logger.info("=" * 95)
    header = (
        f"{'Config Name':<24} | {'Levels':<6} | {'Chunks':<18} | "
        f"{'MK-NIAH Acc (%)':<16} | {'Avg Target Prob':<16} | {'QASPER PPL':<12} | {'Peak VRAM'}"
    )
    logger.info(header)
    logger.info("-" * 110)
    for r in experiment_results:
        row = (
            f"{r['name']:<24} | {r['num_levels']:<6} | {str(r['chunk_sizes']):<18} | "
            f"{r['mkniah_accuracy_pct']:<16.2f} | {r['mkniah_avg_target_prob']:<16.6f} | "
            f"{r['qasper_perplexity']:<12.4f} | {r['peak_vram_mb']:.1f} MB"
        )
        logger.info(row)
    logger.info("=" * 95)

    return summary_payload


if __name__ == "__main__":
    run_track2_experiment()

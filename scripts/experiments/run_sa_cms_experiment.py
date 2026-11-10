"""
Phase 2 Experiment Runner: Controlled Comparison & Ablations for SA-CMS vs. Baseline.
Evaluates:
- A1: Fixed token schedule (Baseline CMS)
- A2: Real paragraph/section boundaries (SA-CMS)
- A3: Random boundaries matching update count (Ablation)
Across memory levels: 1 level (ICL), 2 levels, 3 levels.
Generates: results/sa_cms_comparison.csv and results/sa_cms_comparison.md
"""

import os
import sys
import time
import math
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hope_attention.sa_cms import StructureAlignedHopeLM
from src.evaluation.pretrained_benchmarks import NaturalMKNIAHBenchmark, QASPERDocumentBenchmark
from src.utils.logger import setup_logger


def evaluate_mkniah_with_schedule(
    model: StructureAlignedHopeLM,
    bench: NaturalMKNIAHBenchmark,
    schedule_mode: str,
    fixed_chunk_sizes: Optional[List[int]] = None,
    seed: int = 42,
    logger=None,
) -> Dict[str, Any]:
    """Evaluates MK-NIAH using specified schedule mode (structure, fixed_token, or random)."""
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

        # Ingestion step if CMS is present (num_levels > 1)
        if model.cms is not None:
            model.reset_memory()
            model.clear_event_log()
            res = model.ingest_structured_document(
                document_text=context,
                schedule_mode=schedule_mode,
                fixed_chunk_sizes=fixed_chunk_sizes,
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

        if model.cms is not None:
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


def evaluate_qasper_with_schedule(
    model: StructureAlignedHopeLM,
    bench: QASPERDocumentBenchmark,
    schedule_mode: str,
    fixed_chunk_sizes: Optional[List[int]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluates QASPER Cross-Entropy Loss and Perplexity using specified schedule mode."""
    model.eval()
    tokenizer = model.tokenizer

    total_loss = 0.0
    doc_count = min(bench.num_documents, len(bench.documents))

    for idx in range(doc_count):
        context, question, answer = bench.documents[idx]
        doc_text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"

        enc = tokenizer(doc_text, return_tensors="pt").to(model.device)
        input_ids = enc.input_ids

        if model.cms is not None:
            model.reset_memory()
            model.ingest_structured_document(
                document_text=context,
                schedule_mode=schedule_mode,
                fixed_chunk_sizes=fixed_chunk_sizes,
                seed=seed + idx * 13,
            )

        with torch.no_grad():
            _, loss = model.forward(input_ids, targets=input_ids)
            loss_val = loss.item() if loss is not None else 0.0
            total_loss += loss_val

        if model.cms is not None:
            model.reset_memory()

    avg_loss = total_loss / max(1, doc_count)
    avg_ppl = math.exp(min(avg_loss, 20.0))

    return {
        "avg_loss": avg_loss,
        "perplexity": avg_ppl,
        "num_documents": doc_count,
    }


def run_sa_cms_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = str(ROOT_DIR / "logs")
    filename = f"sa_cms_experiment_{timestamp}.log"
    logger = setup_logger("SACMSExperiment", log_dir=log_dir, filename=filename)

    logger.info("=" * 90)
    logger.info("STARTING PHASE 2: SA-CMS CONTROLLED COMPARISON & ABLATIONS")
    logger.info("Comparing: Fixed Token (A1) vs. SA-CMS Structure (A2) vs. Random (A3)")
    logger.info("Backbone: HuggingFaceTB/SmolLM2-135M | GPU: GTX 1650 Ti (4GB VRAM)")
    logger.info("=" * 90)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Compute device: {device}")

    seed = 42
    sample_count = 50  # Scaled sample count as per Task 7
    qasper_doc_count = 5

    # Define experimental matrix
    experimental_matrix = [
        # Track 2 Baseline: 1 level (ICL)
        {
            "name": "level_1_icl",
            "levels": 1,
            "schedule": "none",
            "fixed_chunks": [64],
            "desc": "1-level Pure ICL Baseline (no CMS)",
        },
        # 2 Levels: Fixed vs SA-CMS vs Random
        {
            "name": "level_2_fixed_token",
            "levels": 2,
            "schedule": "fixed_token",
            "fixed_chunks": [64, 32],
            "desc": "A1: 2-level CMS with Fixed Token Schedule [64, 32]",
        },
        {
            "name": "level_2_sa_cms",
            "levels": 2,
            "schedule": "structure",
            "fixed_chunks": None,
            "desc": "A2: 2-level SA-CMS with Paragraph + Section Boundaries",
        },
        {
            "name": "level_2_random",
            "levels": 2,
            "schedule": "random",
            "fixed_chunks": None,
            "desc": "A3: 2-level Ablation with Random Boundaries (Matched Event Count)",
        },
        # 3 Levels: Fixed vs SA-CMS vs Random
        {
            "name": "level_3_fixed_token",
            "levels": 3,
            "schedule": "fixed_token",
            "fixed_chunks": [64, 32, 16],
            "desc": "A1: 3-level CMS with Fixed Token Schedule [64, 32, 16]",
        },
        {
            "name": "level_3_sa_cms",
            "levels": 3,
            "schedule": "structure",
            "fixed_chunks": None,
            "desc": "A2: 3-level SA-CMS with Paragraph + Section + Document Boundaries",
        },
        {
            "name": "level_3_random",
            "levels": 3,
            "schedule": "random",
            "fixed_chunks": None,
            "desc": "A3: 3-level Ablation with Random Boundaries (Matched Event Count)",
        },
    ]

    mkniah_bench = NaturalMKNIAHBenchmark(num_samples=sample_count, num_keys=3, seed=seed)
    qasper_bench = QASPERDocumentBenchmark(num_documents=qasper_doc_count, seed=seed)

    checkpoint_dir = ROOT_DIR / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    all_results = []

    for cfg in experimental_matrix:
        cfg_name = cfg["name"]
        levels = cfg["levels"]
        schedule = cfg["schedule"]
        fixed_chunks = cfg["fixed_chunks"]

        logger.info("\n" + "=" * 70)
        logger.info(f"RUNNING CONFIGURATION: {cfg_name} (Levels={levels}, Schedule={schedule})")
        logger.info(f"Details: {cfg['desc']}")
        logger.info("=" * 70)

        start_time = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Load model
        model = StructureAlignedHopeLM(
            model_name_or_path="HuggingFaceTB/SmolLM2-135M",
            num_levels=levels,
            chunk_sizes=fixed_chunks or [64, 32, 16][:levels],
            device=device,
        )

        total_params = sum(p.numel() for p in model.parameters())
        cms_params = sum(p.numel() for p in model.get_cms_parameters())

        # Evaluate MK-NIAH
        logger.info(f"[{cfg_name}] Running MK-NIAH Benchmark ({sample_count} samples)...")
        mkniah_res = evaluate_mkniah_with_schedule(
            model=model,
            bench=mkniah_bench,
            schedule_mode=schedule if levels > 1 else "fixed_token",
            fixed_chunk_sizes=fixed_chunks,
            seed=seed,
            logger=logger,
        )
        logger.info(
            f"[{cfg_name}] MK-NIAH Acc: {mkniah_res['accuracy_pct']:.2f}% ({mkniah_res['correct']}/{mkniah_res['total']}) | "
            f"Avg Target Prob: {mkniah_res['avg_target_prob']:.6f} | Avg Rank: {mkniah_res['avg_target_rank']:.1f} | "
            f"Total Updates: {mkniah_res['total_update_events']}"
        )

        # Evaluate QASPER
        logger.info(f"[{cfg_name}] Running QASPER Benchmark ({qasper_doc_count} documents)...")
        qasper_res = evaluate_qasper_with_schedule(
            model=model,
            bench=qasper_bench,
            schedule_mode=schedule if levels > 1 else "fixed_token",
            fixed_chunk_sizes=fixed_chunks,
            seed=seed,
        )
        logger.info(
            f"[{cfg_name}] QASPER Avg Loss: {qasper_res['avg_loss']:.4f} | Perplexity: {qasper_res['perplexity']:.4f}"
        )

        # Save Checkpoint
        ckpt_path = str(checkpoint_dir / f"sa_cms_{cfg_name}.pt")
        model.save_cms_checkpoint(ckpt_path)

        elapsed = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"[{cfg_name}] Finished in {elapsed:.2f}s | Peak VRAM: {peak_vram:.2f} MB")

        record = {
            "model": "SmolLM2-135M",
            "levels": levels,
            "schedule": schedule,
            "sample_count": sample_count,
            "MK_NIAH_accuracy": round(mkniah_res["accuracy_pct"], 2),
            "avg_target_probability": round(mkniah_res["avg_target_prob"], 6),
            "avg_target_rank": round(mkniah_res["avg_target_rank"], 2),
            "QASPER_PPL": round(qasper_res["perplexity"], 4),
            "runtime_seconds": round(elapsed, 2),
            "peak_vram_mb": round(peak_vram, 2),
            "update_count": mkniah_res["total_update_events"],
            "config_name": cfg_name,
            "description": cfg["desc"],
            "checkpoint_path": ckpt_path,
        }
        all_results.append(record)

        del model
        torch.cuda.empty_cache()

    # STEP 10: GENERATE CSV AND MARKDOWN COMPARISON TABLES
    results_dir = ROOT_DIR / "results"
    os.makedirs(results_dir, exist_ok=True)

    # 1. CSV Output
    csv_path = results_dir / "sa_cms_comparison.csv"
    csv_columns = [
        "model", "levels", "schedule", "sample_count",
        "MK_NIAH_accuracy", "avg_target_probability", "avg_target_rank",
        "QASPER_PPL", "runtime_seconds", "peak_vram_mb", "update_count"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for r in all_results:
            row_data = {col: r[col] for col in csv_columns}
            writer.writerow(row_data)
    logger.info(f"\nGenerated CSV comparison at: {csv_path}")

    # 2. Markdown Output
    md_path = results_dir / "sa_cms_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BẢNG SO SÁNH NGHIÊN CỨU: SA-CMS VS. BASELINE CMS & ABLATIONS\n\n")
        f.write(f"> **Mô hình nền:** `HuggingFaceTB/SmolLM2-135M` (134.5M tham số)\n")
        f.write(f"> **Thời điểm thực nghiệm:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"> **Tập kiểm thử:** MK-NIAH ({sample_count} mẫu) & QASPER ({qasper_doc_count} tài liệu)\n")
        f.write(f"> **Mục tiêu khoa học:** Kiểm chứng giả thuyết: *Lịch trình cập nhật căn chỉnh cấu trúc (SA-CMS) có giúp cải thiện độ chính xác truy xuất và perplexity so với lịch trình token cố định dưới cùng ngân sách cập nhật?*\n\n")
        f.write("---\n\n")
        f.write("## 1. BẢNG TỔNG HỢP KẾT QUẢ ĐỐI CHỨNG (RESEARCH COMPARISON TABLE)\n\n")
        f.write("| Model | Levels | Schedule (Lịch trình) | Samples | MK-NIAH Acc (%) | Avg Target Prob | Avg Target Rank | QASPER PPL | Updates | Runtime (s) | Peak VRAM |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in all_results:
            sched_label = r["schedule"]
            if sched_label == "none":
                sched_label = "None (ICL)"
            elif sched_label == "fixed_token":
                sched_label = "Fixed Token (A1)"
            elif sched_label == "structure":
                sched_label = "**SA-CMS Structure (A2)**"
            elif sched_label == "random":
                sched_label = "Random Boundary (A3)"

            f.write(
                f"| {r['model']} | {r['levels']} | {sched_label} | {r['sample_count']} | "
                f"**{r['MK_NIAH_accuracy']:.2f}%** | {r['avg_target_probability']:.6f} | "
                f"{r['avg_target_rank']:.1f} | {r['QASPER_PPL']:.4f} | {r['update_count']} | "
                f"{r['runtime_seconds']:.1f}s | {r['peak_vram_mb']:.1f} MB |\n"
            )
        f.write("\n---\n\n")
        f.write("## 2. PHÂN TÍCH VÀ DIỄN GIẢI KHOA HỌC (SCIENTIFIC INTERPRETATION)\n\n")
        f.write("1. **So sánh SA-CMS (A2) vs. Fixed Token (A1):** Đánh giá hiệu quả của việc thay thế cửa sổ token nhân tạo bằng ranh giới đoạn văn / đề mục tự nhiên.\n")
        f.write("2. **So sánh SA-CMS (A2) vs. Random Boundary (A3):** Kiểm chứng xem sự cải thiện đến từ cấu trúc ngữ nghĩa thực sự hay chỉ là do số lượng sự kiện cập nhật ngẫu nhiên.\n")
        f.write("3. **Động học đa thang (1 vs 2 vs 3 levels):** Phân tích sự tương tác giữa các thang thời gian (đoạn văn vs. đề mục vs. toàn văn).\n")

    logger.info(f"Generated Markdown report at: {md_path}")

    # 3. JSON Output
    json_path = results_dir / "sa_cms_experiment_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Generated JSON records at: {json_path}")

    return all_results


if __name__ == "__main__":
    run_sa_cms_experiment()

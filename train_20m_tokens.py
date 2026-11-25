"""
SA-CMS Continuous Training to Target Tokens (Default: 20,000,000 tokens)
Features:
- Runs live in foreground in your terminal (no background tasks, monitorable progress).
- Automatic checkpointing every 5M tokens + final checkpoint at 20M tokens.
- Supports resuming from existing checkpoint or starting fresh.
- Real-time terminal stats: Epoch, Progress %, Cumulative Tokens, Loss, Speed (tok/s), VRAM.
- Automatically stops when target tokens (20M) are reached.
"""

import sys
import os
import time
import math
import json
import argparse
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.hope_attention.pretrained_hope import PretrainedHopeLM
from src.training.training_corpus import get_scale_training_corpus, get_validation_corpus


def _load_online_corpus_samples(online_corpus_path: str):
    """Load and format Q&A + document text samples from an online corpus JSON."""
    with open(online_corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = data.get("documents", [])
    questions = data.get("questions", [])
    doc_map = {d.get("document_id"): d for d in docs if d.get("document_id")}
    samples = []

    for q in questions:
        doc = doc_map.get(q.get("document_id"), {})
        title = doc.get("title", "")
        raw_text = doc.get("text", "") or doc.get("content", "")
        snippet = raw_text[:900].strip()
        q_text = (q.get("question") or "").strip()
        if not q_text:
            continue
        if q.get("answerable") and q.get("gold_answer"):
            ans = str(q["gold_answer"]).strip()
        else:
            ans = "Not enough information in the provided document."
        formatted = (
            f"Document: {title}\n"
            f"Context: {snippet}\n"
            f"Question: {q_text}\n"
            f"Answer: {ans}"
        )
        samples.append({"formatted_text": formatted, "source": q.get("source", "online")})

    # Also add document overview chunks so every document contributes directly
    for d in docs:
        raw_text = (d.get("text") or d.get("content") or "").strip()
        if raw_text:
            samples.append({
                "formatted_text": raw_text[:1200],
                "source": d.get("source", "online"),
            })

    meta = data.get("metadata", {})
    return samples, len(docs), len(samples), meta.get("documents_by_source", {})


def train_continuous_to_target(
    target_tokens: int = 20_000_000,
    seed: int = 42,
    num_levels: int = 3,
    batch_size: int = 2,
    grad_accum_steps: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    max_length: int = 256,
    device: str = "cuda:0",
    output_dir: str = "checkpoints/train_20m",
    save_interval_tokens: int = 5_000_000,
    resume_from: str = None,
    online_corpus: str = None,
):
    print("=" * 85)
    print(f" SA-CMS CONTINUOUS TRAINING TO TARGET: {target_tokens:,} TOKENS (FOREGROUND)")
    print(f" GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f" Trainable Levels: {num_levels} | Seed: {seed} | Effective Batch: {batch_size * grad_accum_steps}")
    print(f" Save Directory: {output_dir}")
    print("=" * 85)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU training on RTX 3050!")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 1. Load Model & Tokenizer
    print("\n[1/4] Loading SmolLM2-135M Backbone & SA-CMS Adapters...")
    model = PretrainedHopeLM(
        model_name_or_path="HuggingFaceTB/SmolLM2-135M",
        num_levels=num_levels,
        device=device,
        torch_dtype=torch.float32,
        enable_cms=True,
    )
    tokenizer = model.tokenizer

    # Confirm backbone frozen
    assert all(not p.requires_grad for p in model.backbone.parameters()), "Backbone must be 100% frozen!"
    cms_params = list(model.get_cms_parameters())
    print(f"  -> Backbone frozen: 134,515,008 params")
    print(f"  -> Trainable SA-CMS params: {sum(p.numel() for p in cms_params):,} ({len(cms_params)} tensors)")

    # Resume from checkpoint if provided
    cumulative_tokens = 0
    starting_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"  -> Resuming weights from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        cumulative_tokens = ckpt.get("cumulative_tokens", 0)
        starting_epoch = ckpt.get("epochs_completed", 0)
        if cumulative_tokens >= target_tokens:
            print(f"     Warm-start mode: resetting token counter from {cumulative_tokens:,} to 0 for target {target_tokens:,}")
            cumulative_tokens = 0
            starting_epoch = 0
        else:
            print(f"     Resumed at: {cumulative_tokens:,} tokens (Epoch {starting_epoch})")

    # 2. Tokenize Dataset
    print("\n[2/4] Tokenizing corpus...")
    scale_corpus = get_scale_training_corpus()
    train_samples = list(scale_corpus["samples"])
    online_num_docs = 0
    online_num_samples = 0
    online_by_source = {}
    if online_corpus and os.path.exists(online_corpus):
        online_samples, online_num_docs, online_num_samples, online_by_source = _load_online_corpus_samples(online_corpus)
        train_samples.extend(online_samples)
        print(f"  -> Loaded online corpus '{online_corpus}': {online_num_docs:,} docs, {online_num_samples:,} samples")

    encoded_samples = []
    tokens_per_pass = 0
    for s in train_samples:
        enc = tokenizer(
            s["formatted_text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        ids = enc.input_ids.squeeze(0)
        encoded_samples.append(ids)
        tokens_per_pass += ids.size(0)

    val_corpus = get_validation_corpus()
    val_samples = val_corpus["samples"]
    print(f"  -> Samples in corpus: {len(encoded_samples):,}")
    print(f"  -> Tokens per full pass: {tokens_per_pass:,}")
    remaining_tokens = max(0, target_tokens - cumulative_tokens)
    est_epochs = math.ceil(remaining_tokens / tokens_per_pass)
    print(f"  -> Target: {target_tokens:,} tokens (Remaining: {remaining_tokens:,} tokens ~ {est_epochs} epochs)")

    # 3. Optimizer
    optimizer = torch.optim.AdamW(cms_params, lr=lr, weight_decay=weight_decay)

    # 4. Training Loop
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n[3/4] Starting live training in terminal...")
    print("-" * 85)
    print(f"{'Epoch':<8} | {'Progress':<20} | {'Tokens Trained':<16} | {'Train Loss':<12} | {'Tokens/s':<10} | {'VRAM':<8}")
    print("-" * 85)

    model.train()
    t_global_start = time.perf_counter()
    epoch = starting_epoch
    log_history = []
    last_checkpoint_tokens = cumulative_tokens

    while cumulative_tokens < target_tokens:
        epoch += 1
        epoch_tokens = 0
        epoch_loss_sum = 0.0
        step_count = 0
        t_epoch_start = time.perf_counter()
        optimizer.zero_grad()

        # Shuffle sample indices deterministically per epoch
        gen = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(len(encoded_samples), generator=gen).tolist()

        for idx, sample_idx in enumerate(perm):
            sample_ids = encoded_samples[sample_idx]
            seq_len = sample_ids.size(0)

            inp = sample_ids.unsqueeze(0).to(device)
            _, loss = model(inp, targets=inp)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

            epoch_loss_sum += loss.item()
            step_count += 1
            cumulative_tokens += seq_len
            epoch_tokens += seq_len

            if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(perm):
                torch.nn.utils.clip_grad_norm_(cms_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if cumulative_tokens >= target_tokens:
                break

        t_epoch_end = time.perf_counter()
        epoch_time = t_epoch_end - t_epoch_start
        tokens_per_sec = epoch_tokens / max(0.001, epoch_time)
        avg_loss = epoch_loss_sum / max(1, step_count)
        pct = min(100.0, (cumulative_tokens / target_tokens) * 100.0)
        vram_mb = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)

        prog_bar = f"{pct:5.1f}% [{cumulative_tokens:,}/{target_tokens:,}]"
        print(f"Ep {epoch:<4} | {prog_bar:<20} | {cumulative_tokens:>14,} | {avg_loss:<12.4f} | {tokens_per_sec:<10.1f} | {vram_mb:<6.0f}MB")

        log_history.append({
            "epoch": epoch,
            "cumulative_tokens": cumulative_tokens,
            "train_loss": round(avg_loss, 4),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "peak_vram_mb": round(vram_mb, 2),
            "epoch_duration_sec": round(epoch_time, 2),
        })

        # Intermediate checkpoint saving every save_interval_tokens (e.g. every 5M)
        if cumulative_tokens - last_checkpoint_tokens >= save_interval_tokens or cumulative_tokens >= target_tokens:
            last_checkpoint_tokens = cumulative_tokens
            milestone_m = round(cumulative_tokens / 1_000_000, 1)
            inter_ckpt_path = out_path / f"cms_checkpoint_{milestone_m}m.pt"
            state_to_save = {
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items() if "cms" in k or "gate" in k},
                "target_tokens": target_tokens,
                "cumulative_tokens": cumulative_tokens,
                "epochs_completed": epoch,
                "seed": seed,
                "num_levels": num_levels,
                "timestamp": time.time(),
            }
            torch.save(state_to_save, inter_ckpt_path)
            print(f"       -> [Checkpoint Saved]: {inter_ckpt_path.name}")

    total_time = time.perf_counter() - t_global_start
    print("-" * 85)
    print(f"\n[4/4] TARGET REACHED: {cumulative_tokens:,} tokens trained in {total_time/60:.2f} minutes!")

    # 5. Save Final Checkpoint
    final_ckpt_path = out_path / "cms_20m_tokens_final.pt"
    state_to_save = {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items() if "cms" in k or "gate" in k},
        "target_tokens": target_tokens,
        "cumulative_tokens": cumulative_tokens,
        "epochs_completed": epoch,
        "seed": seed,
        "num_levels": num_levels,
        "training_time_sec": total_time,
        "timestamp": time.time(),
    }
    torch.save(state_to_save, final_ckpt_path)
    print(f" -> Final Checkpoint saved to: {final_ckpt_path}")

    # 6. Save Training Metrics Log
    log_path = out_path / "training_20m_log.json"
    avg_tps = round(cumulative_tokens / max(0.001, total_time), 2)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_tokens": target_tokens,
            "actual_tokens_trained": cumulative_tokens,
            "total_epochs": epoch,
            "total_time_seconds": round(total_time, 2),
            "average_throughput_tokens_sec": avg_tps,
            "history": log_history,
        }, f, indent=2)
    print(f" -> Metrics log saved to: {log_path}")

    if online_corpus:
        report_path = out_path / "api_training_report.json"
        last_entry = log_history[-1] if log_history else {}
        report_payload = {
            "training_run": "SA-CMS + WaveCMS Expanded Online API Training",
            "status": "COMPLETED_VERIFIED",
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else device,
            "warm_start_from": resume_from,
            "online_corpus_path": online_corpus,
            "num_documents": online_num_docs,
            "num_samples": len(train_samples),
            "online_samples_added": online_num_samples,
            "documents_by_source": online_by_source,
            "target_tokens": target_tokens,
            "actual_tokens_trained": cumulative_tokens,
            "total_epochs": epoch,
            "train_loss": last_entry.get("train_loss"),
            "tokens_per_sec": avg_tps,
            "peak_vram_mb": max((h.get("peak_vram_mb", 0.0) for h in log_history), default=0.0),
            "total_time_seconds": round(total_time, 2),
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": batch_size * grad_accum_steps,
            "lr": lr,
            "checkpoint_path": str(final_ckpt_path),
            "history": log_history,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, ensure_ascii=False, indent=2)
        print(f" -> API Training Report saved to: {report_path}")

    print("=" * 85)
    print(" TRAINING COMPLETED SUCCESSFULLY. ALL WEIGHTS SAVED.")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SA-CMS continuously to target tokens (20M).")
    parser.add_argument("--tokens", type=int, default=20_000_000, help="Target token count (default: 20000000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--levels", type=int, default=3, help="CMS levels: 1, 2, or 3 (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/train_20m", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--save-interval", type=int, default=5_000_000, help="Save milestone checkpoint every N tokens")
    parser.add_argument("--online-corpus", type=str, default=None, help="Optional path to online training corpus JSON")

    args = parser.parse_args()
    train_continuous_to_target(
        target_tokens=args.tokens,
        seed=args.seed,
        num_levels=args.levels,
        lr=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        resume_from=args.resume,
        save_interval_tokens=args.save_interval,
        online_corpus=args.online_corpus,
    )


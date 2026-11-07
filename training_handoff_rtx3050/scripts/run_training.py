"""
PHASE 4.0.4 — OFFICIAL TRAINING RUNNER FOR RTX 3050
Executes locked training on exactly 200 samples for seeds [42, 43, 44] and levels [1, 2, 3].
Produces strictly isolated checkpoints and metadata logs.
"""

import sys
import os
import time
import math
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.hope_attention.pretrained_hope import PretrainedHopeLM

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def evaluate_validation(model, tokenizer, val_samples: List[Dict[str, Any]], device: str = "cuda:0") -> Tuple[float, float]:
    """Computes validation loss and perplexity on 50 validation samples."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for s in val_samples:
            enc = tokenizer(s["formatted_text"], return_tensors="pt", truncation=True, max_length=256).to(device)
            _, loss = model(enc.input_ids, targets=enc.input_ids)
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                count += 1
    model.train()
    avg_loss = total_loss / max(1, count)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl

def train_single_run(
    num_levels: int,
    seed: int,
    config: Dict[str, Any],
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"STARTING TRAINING RUN: Seed {seed} | num_levels = {num_levels} ({'DRY RUN' if dry_run else 'FULL 200 SAMPLES'})")
    print("=" * 80)

    device = config["hardware"].get("device", "cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training requires an NVIDIA GPU.")

    # 1. Seed initialization (independent run guarantee)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2. Reset peak memory tracking
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 3. Model & Tokenizer loading
    backbone_name = config["model"]["backbone_name_or_path"]
    model = PretrainedHopeLM(
        model_name_or_path=backbone_name,
        num_levels=num_levels,
        device=device,
        torch_dtype=torch.float32,
        enable_cms=True,
    )
    tokenizer = model.tokenizer

    # Confirm backbone frozen
    backbone_frozen = all(not p.requires_grad for p in model.backbone.parameters())
    assert backbone_frozen, "Backbone must be 100% frozen."

    # 4. Hyper-parameters
    hp = config["training_hyperparameters"]
    lr = float(hp["learning_rate"])
    weight_decay = float(hp["weight_decay"])
    grad_clip = float(hp.get("gradient_clipping", 1.0))
    batch_size = int(hp["batch_size"])
    grad_accum_steps = int(hp["gradient_accumulation_steps"])
    num_epochs = 1 if dry_run else int(hp["epochs"])

    # Optimizer
    cms_params = list(model.get_cms_parameters())
    assert len(cms_params) > 0, "No trainable CMS parameters found."
    optimizer = torch.optim.AdamW(cms_params, lr=lr, weight_decay=weight_decay)

    # 5. Tokenize samples
    effective_samples = train_samples[:4] if dry_run else train_samples
    assert len(effective_samples) == 200 or dry_run, f"Expected 200 training samples, got {len(effective_samples)}"

    encoded_samples = []
    total_tokens = 0
    for s in effective_samples:
        enc = tokenizer(s["formatted_text"], return_tensors="pt", truncation=True, max_length=hp["max_sequence_length"])
        ids = enc.input_ids.squeeze(0)
        encoded_samples.append(ids)
        total_tokens += ids.size(0)

    # Initial validation
    val_loss_init, val_ppl_init = evaluate_validation(model, tokenizer, val_samples, device=device)
    print(f"Pre-training Validation Loss: {val_loss_init:.4f} | Validation PPL: {val_ppl_init:.2f}")

    # 6. Training Loop
    model.train()
    t_start = time.perf_counter()
    epoch_losses = []

    for epoch in range(num_epochs):
        t_epoch_start = time.perf_counter()
        optimizer.zero_grad()
        epoch_loss_sum = 0.0
        step_count = 0

        for idx, sample_ids in enumerate(encoded_samples):
            inp = sample_ids.unsqueeze(0).to(device)
            _, loss = model(inp, targets=inp)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()
            epoch_loss_sum += loss.item()
            step_count += 1

            if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(encoded_samples):
                torch.nn.utils.clip_grad_norm_(cms_params, grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        avg_epoch_loss = epoch_loss_sum / max(1, step_count)
        epoch_losses.append(avg_epoch_loss)
        val_loss, val_ppl = evaluate_validation(model, tokenizer, val_samples, device=device)
        t_epoch = time.perf_counter() - t_epoch_start
        print(f"  Epoch {epoch + 1}/{num_epochs} [{t_epoch:.1f}s] | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

    runtime_sec = time.perf_counter() - t_start
    throughput_samples_s = (len(effective_samples) * num_epochs) / max(0.001, runtime_sec)
    throughput_tokens_s = (total_tokens * num_epochs) / max(0.001, runtime_sec)
    peak_vram_mb = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)

    # 7. Checkpoint Save
    ckpt_dir = PACKAGE_ROOT / config["output"]["checkpoints_dir"] / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"cms_{num_levels}lvl_seed_{seed}.pt"

    state = {
        "cms_state_dict": model.cms.state_dict() if model.cms else {},
        "cms_norm_state_dict": model.cms_norm.state_dict() if model.cms_norm else {},
        "num_levels": num_levels,
        "seed": seed,
        "num_samples": len(effective_samples),
        "num_epochs": num_epochs,
        "final_train_loss": epoch_losses[-1],
        "final_val_loss": val_loss,
        "final_val_ppl": val_ppl,
        "backbone_name": backbone_name,
        "runtime_sec": runtime_sec,
    }
    torch.save(state, ckpt_path)
    ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"Saved Checkpoint: {ckpt_path} ({ckpt_size_mb:.2f} MB)")

    # 8. Run Metadata Log
    logs_dir = PACKAGE_ROOT / config["output"]["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    meta_path = logs_dir / f"training_meta_seed{seed}_lvl{num_levels}.json"

    method_tag = "B4" if num_levels == 1 else ("A2" if num_levels == 2 else "B5_P1_P2")
    metadata = {
        "method": method_tag,
        "num_levels": num_levels,
        "seed": seed,
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "python_version": sys.version.split()[0],
        },
        "config_sha256": sha256_file(PACKAGE_ROOT / "configs" / "locked_training.yaml"),
        "train_dataset_sha256": sha256_file(PACKAGE_ROOT / "data" / "train" / "train_200_samples.json"),
        "training_sample_count": len(effective_samples),
        "total_tokens_processed": total_tokens * num_epochs,
        "epochs": num_epochs,
        "train_losses_per_epoch": [round(l, 4) for l in epoch_losses],
        "final_train_loss": round(epoch_losses[-1], 4),
        "final_val_loss": round(val_loss, 4),
        "final_val_ppl": round(val_ppl, 2),
        "runtime_sec": round(runtime_sec, 2),
        "throughput_samples_per_sec": round(throughput_samples_s, 2),
        "throughput_tokens_per_sec": round(throughput_tokens_s, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "checkpoint_path": str(ckpt_path.relative_to(PACKAGE_ROOT)),
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved Metadata Log: {meta_path}")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Phase 4.0.4 Locked Training on RTX 3050")
    parser.add_argument("--seed", type=str, default="all", help="Seed to train: 42, 43, 44 or 'all'")
    parser.add_argument("--levels", type=str, default="all", help="Memory levels to train: 1, 2, 3 or 'all'")
    parser.add_argument("--config", type=str, default="configs/locked_training.yaml", help="Path to locked config")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 quick test epoch on 4 samples")
    args = parser.parse_args()

    config_path = PACKAGE_ROOT / args.config
    assert config_path.exists(), f"Configuration file not found: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load datasets
    train_path = PACKAGE_ROOT / config["data"]["train_file"]
    val_path = PACKAGE_ROOT / config["data"]["validation_file"]
    assert train_path.exists(), f"Training dataset not found: {train_path}"
    assert val_path.exists(), f"Validation dataset not found: {val_path}"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    train_samples = train_data["samples"]
    val_samples = val_data["samples"]

    # Parse seeds
    if args.seed.lower() == "all":
        seeds = config["training_hyperparameters"]["seeds"]
    else:
        seeds = [int(args.seed)]

    # Parse levels
    if args.levels.lower() == "all":
        levels = [1, 2, 3]
    else:
        levels = [int(args.levels)]

    print("=" * 80)
    print("PHASE 4.0.4: EXTERNAL TRAINING SUITE INITIALIZED")
    print(f"Target Seeds: {seeds} | Target Levels: {levels}")
    print(f"Training Sample Count: {len(train_samples)} (LOCKED) | Validation Count: {len(val_samples)}")
    print("=" * 80)

    all_runs = []
    t_global_start = time.perf_counter()

    for s in seeds:
        for lvl in levels:
            meta = train_single_run(
                num_levels=lvl,
                seed=s,
                config=config,
                train_samples=train_samples,
                val_samples=val_samples,
                dry_run=args.dry_run,
            )
            all_runs.append(meta)

    total_time = time.perf_counter() - t_global_start
    print("=" * 80)
    print(f"ALL TRAINING RUNS COMPLETED in {total_time:.2f}s ({total_time/60.0:.2f} mins)")
    print(f"Generated {len(all_runs)} checkpoints and metadata logs in output/")
    print("=" * 80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MASTER TRAINING ENTRYPOINT — SA-CMS 3-LEVEL NEURAL MEMORY (Unified Training Runner).

Unifies all training workflows into a single authoritative command:
1. Full Dataset Training (Default): 135 documents, ~2.97M tokens, 5 epochs, 2.0x Vietnamese oversampling.
2. Pilot Training: Fast 100-sample verification run.
3. Token-Budget Training: Trains until a specified token target is reached (--target_tokens).

Usage Examples:
    # 1. Full training on GPU (Recommended for production)
    python train.py --mode full --epochs 5 --seed 42

    # 2. Fast pilot check (100 samples)
    python train.py --mode pilot

    # 3. Custom token target
    python train.py --mode budget --target_tokens 3000000

    # 4. Force CPU execution
    python train.py --device cpu
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Enforce strict offline execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SA_CMS_OFFLINE_MODE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="SA-CMS Unified Master Training Runner"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "pilot", "budget"],
        default="full",
        help="Training mode: 'full' (all documents, ~2.97M tokens), 'pilot' (quick smoke test), 'budget' (token-capped)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for full/pilot training (default: 5)",
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=3_000_000,
        help="Token budget limit when running in 'budget' mode (default: 3,000,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.6e-4,
        help="AdamW peak learning rate (default: 1.6e-4)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device: 'auto' (prefers CUDA if available), 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/approved_model",
        help="Directory to save the trained model checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import torch

    # Device selection
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    print("=" * 75)
    print("SA-CMS UNIFIED MASTER TRAINING RUNNER")
    print(f"Mode         : {args.mode.upper()}")
    print(f"Device       : {device_str.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device_str == "cuda" else ""))
    print(f"Seed         : {args.seed}")
    print(f"Epochs       : {args.epochs}")
    print(f"Output Dir   : {args.output_dir}")
    if args.mode == "budget":
        print(f"Token Target : {args.target_tokens:,} tokens")
    print("=" * 75)

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    if args.mode == "pilot":
        from scripts.run_training_pilot import main as pilot_main
        print("[*] Launching Pilot Training verification (100 samples)...")
        pilot_main()
    else:
        # Full or Budget Training
        from scripts.run_training import run_training_for_seed
        print(f"[*] Starting SA-CMS 3-Level Training (Seed={args.seed}, Epochs={args.epochs})...")
        metrics = run_training_for_seed(
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            device_override=device_str,
            output_dir=out_dir,
        )
        print("\n[✓] Training completed successfully!")
        if metrics:
            print(f"[*] Training Summary: Loss={metrics.get('final_loss', 'N/A')}, Tokens={metrics.get('tokens_processed', 'N/A')}")

    elapsed_min = (time.perf_counter() - t_start) / 60
    print(f"[✓] Tổng thời gian thực hiện: {elapsed_min:.2f} phút.")


if __name__ == "__main__":
    main()


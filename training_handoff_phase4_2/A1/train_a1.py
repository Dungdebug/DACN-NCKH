# Standalone Training Script for A1 (CMS Level-3 Random Boundary)
# Strictly adheres to Phase 4.0.2 / 4.2 protocol: Exactly 200 samples, Seeds [42, 43, 44].
import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hope_attention.pretrained_hope import PretrainedHopeLM
from src.training.training_corpus import get_scale_training_corpus

SEEDS = [42, 43, 44]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = ROOT_DIR / "checkpoints" / "phase4_2" / "A1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def train_seed(seed: int):
    print(f"=== Training A1 (Random Boundary) on Seed {seed} ===")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    corpus = get_scale_training_corpus(200)
    samples = corpus["samples"]

    model = PretrainedHopeLM(
        model_name_or_path="HuggingFaceTB/SmolLM2-135M",
        num_levels=3,
        device=DEVICE,
        torch_dtype=torch.float32,
        enable_cms=True,
    )
    tokenizer = model.tokenizer
    optimizer = torch.optim.AdamW(model.get_cms_parameters(), lr=1e-4, weight_decay=0.01)

    encoded = []
    for s in samples:
        enc = tokenizer(s["formatted_text"], return_tensors="pt", truncation=True, max_length=256)
        encoded.append(enc.input_ids.squeeze(0))

    model.train()
    batch_size = 2
    grad_accum_steps = 2
    for epoch in range(3):
        optimizer.zero_grad()
        for idx, sample_ids in enumerate(encoded):
            inp = sample_ids.unsqueeze(0).to(DEVICE)
            _, loss = model(inp, targets=inp)
            loss = loss / grad_accum_steps
            loss.backward()

            if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(encoded):
                optimizer.step()
                optimizer.zero_grad()

    ckpt_path = OUT_DIR / f"cms_3lvl_random_seed_{seed}.pt"
    state = {
        "cms_state_dict": model.cms.state_dict() if model.cms else {},
        "cms_norm_state_dict": model.cms_norm.state_dict() if model.cms_norm else {},
        "num_levels": 3,
        "seed": seed,
        "num_samples": len(samples),
        "num_epochs": 3,
        "ablation_type": "A1_random_boundary"
    }
    torch.save(state, ckpt_path)
    print(f"Saved: {ckpt_path}")

def main():
    for s in SEEDS:
        train_seed(s)

if __name__ == "__main__":
    main()

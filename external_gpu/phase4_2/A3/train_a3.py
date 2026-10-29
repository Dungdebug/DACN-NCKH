# Standalone Training Script for A3 (SA-CMS Additive / Ungated)
import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hope_attention.sa_cms import StructureAlignedHopeLM
from src.training.training_corpus import get_scale_training_corpus
from src.cms.mlp_chain import MLPBlock

SEEDS = [42, 43, 44]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = ROOT_DIR / "checkpoints" / "phase4_2" / "A3"
FINAL_DIR = ROOT_DIR / "training_output_final" / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)


class UngatedAdditiveMLPChain(nn.Module):
    def __init__(self, num_levels: int = 3, d_model: int = 576, d_ff: int = 1536, dropout: float = 0.0):
        super().__init__()
        self.num_levels = num_levels
        self.d_model = d_model
        self.d_ff = d_ff
        self.blocks = nn.ModuleList([
            MLPBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
            for _ in range(num_levels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [block(x) for block in self.blocks]
        return sum(outputs) / float(self.num_levels)


def train_seed(seed: int):
    print(f"=== Training A3 (Additive Ungated) on Seed {seed} ===")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    corpus = get_scale_training_corpus(200)
    samples = corpus["samples"]

    model = StructureAlignedHopeLM(
        model_name_or_path="HuggingFaceTB/SmolLM2-135M",
        num_levels=3,
        device=DEVICE,
        torch_dtype=torch.float32,
        enable_cms=True,
    )
    model.cms.chain = UngatedAdditiveMLPChain(num_levels=3, d_model=576, d_ff=1536).to(DEVICE)
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
            logits, loss = model(inp, targets=inp)
            loss = loss / grad_accum_steps
            loss.backward()

            if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(encoded):
                torch.nn.utils.clip_grad_norm_(model.get_cms_parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

    ckpt_name = f"cms_3lvl_additive_seed_{seed}.pt"
    state = {
        "cms_state_dict": model.cms.state_dict(),
        "cms_norm_state_dict": model.cms_norm.state_dict(),
        "num_levels": 3,
        "seed": seed,
        "num_samples": len(samples),
        "num_epochs": 3,
        "ablation_type": "A3_additive"
    }
    for dest in [OUT_DIR / ckpt_name, FINAL_DIR / ckpt_name]:
        torch.save(state, dest)
    print(f"Saved: {OUT_DIR / ckpt_name}")


def main():
    for s in SEEDS:
        train_seed(s)


if __name__ == "__main__":
    main()

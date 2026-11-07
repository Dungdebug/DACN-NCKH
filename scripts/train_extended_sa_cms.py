"""
Extended Offline GPU Training Script for 3-Level SA-CMS (NL DocChat Enhancement).

Warm-starts from the approved Phase 4.1 checkpoint (`local_runtime/checkpoints/cms_3lvl_seed_42.pt`)
and continues training ONLY the 3-level Continuum Memory System (`cms` + `cms_norm`, 5,315,904 params)
while keeping 100% of the SmolLM2-135M backbone (`134,515,008` params) strictly frozen.

Key Properties:
- 100% Offline Execution (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `SA_CMS_OFFLINE_MODE=1`).
- Bilingual Scale Corpus:
  * 1,000 English Document QA samples (`TR_DOC_001`..`TR_DOC_100`)
  * 275 Vietnamese Structured Grounded + Memory-Recall QA samples (`VN_TR_DOC_001`..`VN_TR_DOC_025`, 2x oversampled/epoch)
  * Strictly disjoint validation sets: 50 English (`VAL_DOC_001`..`VAL_DOC_005`) + 20 Vietnamese (`VN_VAL_DOC_001`..`VN_VAL_DOC_004`)
  * 0% overlap with any Phase 3 / Phase 4 evaluation benchmark.
- Dual Causal + Answer-Span Cross-Entropy Objective (`0.25 * full_lm_loss + 0.75 * answer_loss`)
  with zero sequence truncation (`max_length=832`).
- Saves to NEW checkpoint files (`checkpoints/extended_training/cms_3lvl_extended.pt` and
  `local_runtime/checkpoints/cms_3lvl_extended.pt`), leaving all Phase 4 checkpoints 100% untouched.
"""

import os
import sys
import json
import math
import time
import random
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Enforce strict offline mode before importing transformers / torch
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SA_CMS_OFFLINE_MODE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from src.hope_attention.sa_cms import StructureAlignedHopeLM
from src.training.training_corpus import get_scale_training_corpus, get_validation_corpus
from src.training.vietnamese_training_corpus import get_vietnamese_training_corpus


def set_deterministic_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def encode_qa_sample(
    tokenizer: Any,
    prompt_text: str,
    full_text: str,
    lang: str,
    sample_id: str,
    max_length: int = 832,
) -> Dict[str, Any]:
    """
    Tokenizes a QA pair into `input_ids` and records `prompt_len` so we can compute
    both full causal LM loss and answer-only cross-entropy loss accurately.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    truncated = False
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        truncated = True

    prompt_len = min(len(prompt_ids), max(0, len(full_ids) - 1))
    return {
        "sample_id": sample_id,
        "lang": lang,
        "input_ids": full_ids,
        "prompt_len": prompt_len,
        "seq_len": len(full_ids),
        "truncated": truncated,
    }


def build_datasets(tokenizer: Any, max_length: int = 832) -> Dict[str, Any]:
    """Builds tokenized English + Vietnamese training and validation sets."""
    en_train_raw = get_scale_training_corpus(1000)["samples"]
    en_val_raw = get_validation_corpus()["samples"]
    vn_bundle = get_vietnamese_training_corpus()
    vn_train_raw = vn_bundle["train_samples"]
    vn_val_raw = vn_bundle["val_samples"]

    en_train: List[Dict[str, Any]] = []
    for s in en_train_raw:
        prompt_text = f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer:"
        full_text = f"{prompt_text} {s['answer']}\n"
        en_train.append(
            encode_qa_sample(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                full_text=full_text,
                lang="en",
                sample_id=s["sample_id"],
                max_length=max_length,
            )
        )

    en_val: List[Dict[str, Any]] = []
    for idx, s in enumerate(en_val_raw, start=1):
        prompt_text = f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer:"
        full_text = f"{prompt_text} {s['answer']}\n"
        en_val.append(
            encode_qa_sample(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                full_text=full_text,
                lang="en",
                sample_id=s.get("sample_id", f"EN_VAL_{idx:03d}"),
                max_length=max_length,
            )
        )

    vn_train: List[Dict[str, Any]] = []
    for s in vn_train_raw:
        vn_train.append(
            encode_qa_sample(
                tokenizer=tokenizer,
                prompt_text=s["prompt_text"],
                full_text=s["formatted_text"],
                lang="vi",
                sample_id=s["sample_id"],
                max_length=max_length,
            )
        )

    vn_val: List[Dict[str, Any]] = []
    for s in vn_val_raw:
        vn_val.append(
            encode_qa_sample(
                tokenizer=tokenizer,
                prompt_text=s["prompt_text"],
                full_text=s["formatted_text"],
                lang="vi",
                sample_id=s["sample_id"],
                max_length=max_length,
            )
        )

    return {
        "en_train": en_train,
        "vn_train": vn_train,
        "en_val": en_val,
        "vn_val": vn_val,
        "en_val_raw": en_val_raw,
        "vn_val_raw": vn_val_raw,
    }


def collate_batch(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Right-pads a mini-batch to the maximum length within that mini-batch.
    Returns:
      input_ids:     (B, T)
      attention_mask:(B, T)
      full_labels:   (B, T) with -100 on padding
      answer_labels: (B, T) with -100 on prompt tokens and padding
    """
    max_t = max(item["seq_len"] for item in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_t), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, max_t), dtype=torch.long, device=device)
    full_labels = torch.full((bsz, max_t), -100, dtype=torch.long, device=device)
    answer_labels = torch.full((bsz, max_t), -100, dtype=torch.long, device=device)

    for row, item in enumerate(batch):
        seq = item["input_ids"]
        slen = len(seq)
        plen = item["prompt_len"]
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)
        input_ids[row, :slen] = seq_tensor
        attention_mask[row, :slen] = 1
        full_labels[row, :slen] = seq_tensor
        answer_labels[row, plen:slen] = seq_tensor[plen:slen]

    return input_ids, attention_mask, full_labels, answer_labels


def compute_dual_losses(
    logits: torch.Tensor,
    full_labels: torch.Tensor,
    answer_labels: torch.Tensor,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes shifted causal LM cross-entropy on:
    1) full sequence (`full_loss`)
    2) answer tokens only (`answer_loss`)
    3) weighted training objective `0.25 * full_loss + 0.75 * answer_loss`
    """
    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_full = full_labels[..., 1:].contiguous().view(-1)
    shift_ans = answer_labels[..., 1:].contiguous().view(-1)

    full_loss = F.cross_entropy(shift_logits, shift_full, ignore_index=-100)
    if (shift_ans != -100).any():
        ans_loss = F.cross_entropy(shift_logits, shift_ans, ignore_index=-100)
    else:
        ans_loss = full_loss

    combined_loss = 0.25 * full_loss + 0.75 * ans_loss
    return combined_loss, full_loss, ans_loss


@torch.no_grad()
def evaluate_dataset(
    model: StructureAlignedHopeLM,
    samples: List[Dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates average per-sample full causal LM loss/PPL and answer-span loss/PPL."""
    model.eval()
    total_full_loss = 0.0
    total_ans_loss = 0.0
    count = 0

    # Evaluate in length-sorted mini-batches of 2 for accuracy and low VRAM
    ordered = sorted(samples, key=lambda x: x["seq_len"])
    for idx in range(0, len(ordered), 2):
        batch = ordered[idx : idx + 2]
        input_ids, attention_mask, full_labels, answer_labels = collate_batch(
            batch, pad_token_id=pad_token_id, device=device
        )
        logits, _ = model.forward(input_ids=input_ids, attention_mask=attention_mask, targets=None)
        _, full_loss, ans_loss = compute_dual_losses(
            logits=logits,
            full_labels=full_labels,
            answer_labels=answer_labels,
            vocab_size=model.vocab_size,
        )
        bsz = len(batch)
        total_full_loss += full_loss.item() * bsz
        total_ans_loss += ans_loss.item() * bsz
        count += bsz

    avg_full = total_full_loss / max(1, count)
    avg_ans = total_ans_loss / max(1, count)
    return {
        "full_loss": round(avg_full, 4),
        "full_ppl": round(math.exp(min(avg_full, 20.0)), 4),
        "answer_loss": round(avg_ans, 4),
        "answer_ppl": round(math.exp(min(avg_ans, 20.0)), 4),
    }


@torch.no_grad()
def generate_qualitative_preview(
    model: StructureAlignedHopeLM,
    tokenizer: Any,
    vn_val_raw: List[Dict[str, Any]],
    en_val_raw: List[Dict[str, Any]],
    device: torch.device,
) -> List[Dict[str, str]]:
    """Generates greedy single-line answers on a small held-out validation subset."""
    model.eval()
    previews: List[Dict[str, str]] = []

    test_cases = []
    for s in vn_val_raw[:3]:
        test_cases.append(("vi", s["question"], s["prompt_text"], s["answer"]))
    for s in en_val_raw[:2]:
        p_txt = f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer:"
        test_cases.append(("en", s["question"], p_txt, s["answer"]))

    for lang, q_text, prompt_text, ref_ans in test_cases:
        enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=768)
        cur_ids = enc.input_ids.to(device)
        generated: List[int] = []

        for _ in range(48):
            logits, _ = model.forward(input_ids=cur_ids)
            next_tok = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            if next_tok == tokenizer.eos_token_id:
                break
            generated.append(next_tok)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_tok]], device=device)], dim=1)
            decoded_so_far = tokenizer.decode(generated, skip_special_tokens=True)
            if "\n" in decoded_so_far:
                break

        pred_text = tokenizer.decode(generated, skip_special_tokens=True).split("\n")[0].strip()
        previews.append({
            "lang": lang,
            "question": q_text,
            "reference": ref_ans,
            "prediction": pred_text,
        })

    return previews


def make_epoch_batches(
    en_train: List[Dict[str, Any]],
    vn_train: List[Dict[str, Any]],
    rng: random.Random,
    vn_oversample: int = 2,
) -> List[List[Dict[str, Any]]]:
    """
    Combines 1,000 EN samples + 2x oversampled 275 VN samples (1,550 total),
    applies noisy length bucketing to minimize padding waste, and shuffles batch order.
    Uses batch_size=4 for short sequences (<=256 tokens) and batch_size=2 for long sequences (>256 tokens).
    """
    pool = list(en_train) + (list(vn_train) * vn_oversample)
    rng.shuffle(pool)
    # Sort with random jitter (+/- 24 tokens) so batches have similar lengths while varying every epoch
    pool.sort(key=lambda item: item["seq_len"] + rng.randint(-24, 24))

    batches: List[List[Dict[str, Any]]] = []
    idx = 0
    n = len(pool)
    while idx < n:
        bsz = 4 if pool[idx]["seq_len"] <= 240 else 2
        batch = pool[idx : idx + bsz]
        # Re-check max length in candidate batch
        if max(x["seq_len"] for x in batch) > 280 and len(batch) > 2:
            batch = pool[idx : idx + 2]
            bsz = len(batch)
        batches.append(batch)
        idx += bsz

    rng.shuffle(batches)
    return batches


def main() -> None:
    print("=" * 78)
    print("  NL DocChat — EXTENDED 3-LEVEL SA-CMS OFFLINE GPU TRAINING")
    print("=" * 78)

    seed = 42
    num_epochs = 6
    vn_oversample = 2
    base_lr = 1.5e-4
    min_lr = 1.5e-5
    weight_decay = 0.01
    target_effective_batch = 8
    max_length = 832

    set_deterministic_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for extended GPU training, but torch.cuda.is_available() is False.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
    torch.cuda.reset_peak_memory_stats(device)

    local_model_dir = ROOT_DIR / "local_runtime" / "model"
    local_tok_dir = ROOT_DIR / "local_runtime" / "tokenizer"
    seed42_ckpt_path = ROOT_DIR / "local_runtime" / "checkpoints" / "cms_3lvl_seed_42.pt"

    if not (local_model_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"Offline backbone missing at {local_model_dir}")
    if not seed42_ckpt_path.exists():
        raise FileNotFoundError(f"Warm-start parent checkpoint missing at {seed42_ckpt_path}")

    parent_sha256_before = compute_sha256(seed42_ckpt_path)
    print(f"[GPU] Device: {gpu_name} ({vram_total_mb} MB VRAM) | CUDA {torch.version.cuda}")
    print(f"[OFFLINE] Backbone dir: {local_model_dir}")
    print(f"[WARM-START] Parent checkpoint: {seed42_ckpt_path.name} (SHA-256: {parent_sha256_before[:16]}...)")

    # 1. Load offline StructureAlignedHopeLM in FP32 for high-precision gradient updates
    model = StructureAlignedHopeLM(
        model_name_or_path=str(local_model_dir),
        num_levels=3,
        device="cuda",
        torch_dtype=torch.float32,
        enable_cms=True,
    )
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    # Verify strict parameter freezing on backbone & trainability on CMS + cms_norm
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False
    cms_params = model.get_cms_parameters()
    for p in cms_params:
        p.requires_grad = True

    frozen_backbone_params = sum(p.numel() for p in model.backbone.parameters() if not p.requires_grad)
    leaked_backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    trainable_cms_params = sum(p.numel() for p in model.cms.parameters() if p.requires_grad)
    trainable_norm_params = sum(p.numel() for p in model.cms_norm.parameters() if p.requires_grad)
    total_trainable_params = sum(p.numel() for p in cms_params if p.requires_grad)

    assert leaked_backbone_params == 0, f"Backbone freeze violated: {leaked_backbone_params} trainable params!"
    print(
        f"[PARAMS] Frozen Backbone: {frozen_backbone_params:,} | "
        f"Trainable CMS: {trainable_cms_params:,} + cms_norm: {trainable_norm_params:,} = {total_trainable_params:,}"
    )

    # 2. Warm-start from cms_3lvl_seed_42.pt
    parent_state = torch.load(str(seed42_ckpt_path), map_location=device)
    model.cms.load_state_dict(parent_state["cms_state_dict"], strict=True)
    if "cms_norm_state_dict" in parent_state and parent_state["cms_norm_state_dict"] is not None:
        model.cms_norm.load_state_dict(parent_state["cms_norm_state_dict"], strict=True)
    model.cms.save_initial_state()

    # Snapshot parent weights for exact post-training delta verification
    initial_cms_snapshot = {k: v.detach().cpu().clone() for k, v in model.cms.state_dict().items()}
    initial_norm_snapshot = {k: v.detach().cpu().clone() for k, v in model.cms_norm.state_dict().items()}

    # 3. Build datasets
    datasets = build_datasets(tokenizer=tokenizer, max_length=max_length)
    en_train = datasets["en_train"]
    vn_train = datasets["vn_train"]
    en_val = datasets["en_val"]
    vn_val = datasets["vn_val"]

    trunc_count = sum(1 for s in (en_train + vn_train + en_val + vn_val) if s["truncated"])
    print(
        f"[DATA] Unique Train: {len(en_train) + len(vn_train):,} "
        f"(EN={len(en_train):,}, VN={len(vn_train):,}; VN {vn_oversample}x/epoch -> {len(en_train) + vn_oversample * len(vn_train):,}/epoch) | "
        f"Val: EN={len(en_val)}, VN={len(vn_val)} | Truncated samples: {trunc_count}"
    )

    # 4. Pre-training baseline evaluation (Epoch 0 = cms_3lvl_seed_42.pt)
    print("\n[EVAL] Evaluating warm-start parent checkpoint (Epoch 0 — cms_3lvl_seed_42.pt)...")
    baseline_en = evaluate_dataset(model, en_val, pad_token_id=pad_token_id, device=device)
    baseline_vn = evaluate_dataset(model, vn_val, pad_token_id=pad_token_id, device=device)
    baseline_combined_ans_loss = round(0.5 * baseline_en["answer_loss"] + 0.5 * baseline_vn["answer_loss"], 4)
    previews_before = generate_qualitative_preview(
        model, tokenizer, datasets["vn_val_raw"], datasets["en_val_raw"], device=device
    )
    print(
        f"  Epoch 0 (Seed42) | EN Ans Loss: {baseline_en['answer_loss']:.4f} (PPL {baseline_en['answer_ppl']:.2f}) | "
        f"VN Ans Loss: {baseline_vn['answer_loss']:.4f} (PPL {baseline_vn['answer_ppl']:.2f}) | "
        f"Combined Ans Loss: {baseline_combined_ans_loss:.4f}"
    )

    # 5. Optimizer & Cosine Schedule Setup
    rng = random.Random(seed)
    precomputed_epoch_batches = [
        make_epoch_batches(en_train, vn_train, rng=rng, vn_oversample=vn_oversample)
        for _ in range(num_epochs)
    ]

    # Group each epoch's mini-batches into accumulation windows (~target_effective_batch samples each)
    # so .backward() runs immediately per mini-batch and frees logits VRAM right away.
    epoch_windows: List[List[List[List[Dict[str, Any]]]]] = []
    for ep_batches in precomputed_epoch_batches:
        windows: List[List[List[Dict[str, Any]]]] = []
        cur_win: List[List[Dict[str, Any]]] = []
        cur_cnt = 0
        for b in ep_batches:
            cur_win.append(b)
            cur_cnt += len(b)
            if cur_cnt >= target_effective_batch:
                windows.append(cur_win)
                cur_win = []
                cur_cnt = 0
        if cur_win:
            windows.append(cur_win)
        epoch_windows.append(windows)

    total_opt_steps = sum(len(w) for w in epoch_windows)

    warmup_steps = max(5, int(0.08 * total_opt_steps))
    optimizer = torch.optim.AdamW(cms_params, lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    def get_lr_at_step(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return base_lr * float(step_idx + 1) / float(warmup_steps)
        progress = float(step_idx - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    best_combined_ans_loss = baseline_combined_ans_loss
    best_epoch = 0
    best_cms_state = {k: v.detach().cpu().clone() for k, v in model.cms.state_dict().items()}
    best_norm_state = {k: v.detach().cpu().clone() for k, v in model.cms_norm.state_dict().items()}
    best_metrics = {
        "en_val": baseline_en,
        "vn_val": baseline_vn,
        "combined_answer_loss": baseline_combined_ans_loss,
    }

    epoch_history: List[Dict[str, Any]] = []
    global_opt_step = 0
    train_start_time = time.time()

    print(
        f"\n[TRAIN] Starting {num_epochs} epochs ({total_opt_steps} optimizer steps, "
        f"warmup={warmup_steps} steps, effective_batch~{target_effective_batch})..."
    )

    for epoch in range(1, num_epochs + 1):
        ep_start = time.time()
        model.cms.train()
        model.cms_norm.train()
        model.backbone.eval()

        ep_windows = epoch_windows[epoch - 1]
        optimizer.zero_grad(set_to_none=True)

        ep_train_comb_sum = 0.0
        ep_train_ans_sum = 0.0
        ep_train_full_sum = 0.0
        ep_sample_count = 0
        ep_grad_norms: List[float] = []

        for w_idx, window in enumerate(ep_windows):
            total_window_samples = float(sum(len(b) for b in window))
            cur_lr = get_lr_at_step(global_opt_step)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            for b_idx, batch in enumerate(window):
                bsz = len(batch)
                input_ids, attention_mask, full_labels, answer_labels = collate_batch(
                    batch, pad_token_id=pad_token_id, device=device
                )

                # Extract frozen backbone hidden states under no_grad for speed & low VRAM,
                # then run trainable cms_norm + cms + frozen lm_head with autograd active.
                with torch.no_grad():
                    backbone_out = model.backbone.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    base_hidden = backbone_out.last_hidden_state

                normed_h = model.cms_norm(base_hidden)
                mem_residual = model.cms(normed_h)
                aug_hidden = base_hidden + mem_residual
                logits = model.backbone.lm_head(aug_hidden)

                comb_loss, full_loss, ans_loss = compute_dual_losses(
                    logits=logits,
                    full_labels=full_labels,
                    answer_labels=answer_labels,
                    vocab_size=model.vocab_size,
                )

                if torch.isnan(comb_loss) or torch.isinf(comb_loss):
                    raise RuntimeError(f"NaN/Inf loss encountered at epoch {epoch}, window {w_idx}, batch {b_idx}!")

                ep_train_comb_sum += comb_loss.item() * bsz
                ep_train_full_sum += full_loss.item() * bsz
                ep_train_ans_sum += ans_loss.item() * bsz
                ep_sample_count += bsz

                scaled_loss = comb_loss * (float(bsz) / total_window_samples)
                scaled_loss.backward()

            grad_norm = float(torch.nn.utils.clip_grad_norm_(cms_params, max_norm=1.0).item())
            ep_grad_norms.append(grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_opt_step += 1

        # End-of-epoch validation
        val_en = evaluate_dataset(model, en_val, pad_token_id=pad_token_id, device=device)
        val_vn = evaluate_dataset(model, vn_val, pad_token_id=pad_token_id, device=device)
        combined_ans_loss = round(0.5 * val_en["answer_loss"] + 0.5 * val_vn["answer_loss"], 4)
        ep_duration = round(time.time() - ep_start, 2)
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)

        avg_train_comb = round(ep_train_comb_sum / max(1, ep_sample_count), 4)
        avg_train_full = round(ep_train_full_sum / max(1, ep_sample_count), 4)
        avg_train_ans = round(ep_train_ans_sum / max(1, ep_sample_count), 4)
        avg_grad_norm = round(sum(ep_grad_norms) / max(1, len(ep_grad_norms)), 4)
        cur_lr_end = round(optimizer.param_groups[0]["lr"], 7)

        improved = combined_ans_loss < best_combined_ans_loss
        if improved:
            best_combined_ans_loss = combined_ans_loss
            best_epoch = epoch
            best_cms_state = {k: v.detach().cpu().clone() for k, v in model.cms.state_dict().items()}
            best_norm_state = {k: v.detach().cpu().clone() for k, v in model.cms_norm.state_dict().items()}
            best_metrics = {
                "en_val": val_en,
                "vn_val": val_vn,
                "combined_answer_loss": combined_ans_loss,
            }

        ep_record = {
            "epoch": epoch,
            "train_combined_loss": avg_train_comb,
            "train_full_loss": avg_train_full,
            "train_answer_loss": avg_train_ans,
            "en_val": val_en,
            "vn_val": val_vn,
            "combined_val_answer_loss": combined_ans_loss,
            "avg_grad_norm": avg_grad_norm,
            "lr_end": cur_lr_end,
            "duration_sec": ep_duration,
            "peak_vram_mb": peak_vram_mb,
            "is_best": improved,
        }
        epoch_history.append(ep_record)

        marker = "*BEST*" if improved else ""
        print(
            f"  Epoch {epoch}/{num_epochs} ({ep_duration:.1f}s) | "
            f"Train Ans: {avg_train_ans:.4f} | "
            f"EN Val Ans: {val_en['answer_loss']:.4f} (PPL {val_en['answer_ppl']:.2f}) | "
            f"VN Val Ans: {val_vn['answer_loss']:.4f} (PPL {val_vn['answer_ppl']:.2f}) | "
            f"Comb: {combined_ans_loss:.4f} | GradNorm: {avg_grad_norm:.3f} | VRAM: {peak_vram_mb:.0f}MB {marker}"
        )

    total_train_sec = round(time.time() - train_start_time, 2)

    # 6. Restore best checkpoint weights and compute parameter delta diagnostics
    model.cms.load_state_dict({k: v.to(device) for k, v in best_cms_state.items()}, strict=True)
    model.cms_norm.load_state_dict({k: v.to(device) for k, v in best_norm_state.items()}, strict=True)
    model.cms.save_initial_state()

    previews_after = generate_qualitative_preview(
        model, tokenizer, datasets["vn_val_raw"], datasets["en_val_raw"], device=device
    )

    max_abs_delta = 0.0
    level_l2_deltas: Dict[str, float] = {"level_0": 0.0, "level_1": 0.0, "level_2": 0.0, "cms_norm": 0.0}
    for k, v_after in best_cms_state.items():
        diff = (v_after - initial_cms_snapshot[k]).float()
        max_abs_delta = max(max_abs_delta, float(diff.abs().max().item()))
        sq_norm = float((diff * diff).sum().item())
        if ".0." in k:
            level_l2_deltas["level_0"] += sq_norm
        elif ".1." in k:
            level_l2_deltas["level_1"] += sq_norm
        elif ".2." in k:
            level_l2_deltas["level_2"] += sq_norm

    norm_sq = 0.0
    for k, v_after in best_norm_state.items():
        diff = (v_after - initial_norm_snapshot[k]).float()
        max_abs_delta = max(max_abs_delta, float(diff.abs().max().item()))
        norm_sq += float((diff * diff).sum().item())
    level_l2_deltas["cms_norm"] = norm_sq
    level_l2_deltas = {k: round(math.sqrt(val), 6) for k, val in level_l2_deltas.items()}

    # 7. Save to archive + active local_runtime checkpoint directory (leaving cms_3lvl_seed_42.pt untouched!)
    archive_dir = ROOT_DIR / "checkpoints" / "extended_training"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_ckpt_path = archive_dir / "cms_3lvl_extended.pt"
    runtime_ckpt_path = ROOT_DIR / "local_runtime" / "checkpoints" / "cms_3lvl_extended.pt"

    checkpoint_payload = {
        "cms_state_dict": best_cms_state,
        "cms_norm_state_dict": best_norm_state,
        "num_levels": 3,
        "seed": seed,
        "num_samples": len(en_train) + len(vn_train),
        "effective_samples_per_epoch": len(en_train) + vn_oversample * len(vn_train),
        "num_epochs": num_epochs,
        "best_epoch": best_epoch,
        "parent_checkpoint": "cms_3lvl_seed_42.pt",
        "parent_sha256": parent_sha256_before,
        "backbone_name": "HuggingFaceTB/SmolLM2-135M",
        "baseline_metrics": {
            "en_val": baseline_en,
            "vn_val": baseline_vn,
            "combined_answer_loss": baseline_combined_ans_loss,
        },
        "best_metrics": best_metrics,
    }

    torch.save(checkpoint_payload, str(archive_ckpt_path))
    # Copy exact bytes so archive and runtime checkpoint share identical SHA-256
    runtime_ckpt_path.write_bytes(archive_ckpt_path.read_bytes())

    extended_sha256 = compute_sha256(runtime_ckpt_path)
    parent_sha256_after = compute_sha256(seed42_ckpt_path)
    assert parent_sha256_before == parent_sha256_after, "Parent checkpoint cms_3lvl_seed_42.pt was modified!"

    # Build qualitative side-by-side comparison
    qualitative_comparison = []
    for before_item, after_item in zip(previews_before, previews_after):
        qualitative_comparison.append({
            "lang": before_item["lang"],
            "question": before_item["question"],
            "reference": before_item["reference"],
            "before_seed42": before_item["prediction"],
            "after_extended": after_item["prediction"],
        })

    report = {
        "training_run": "SA-CMS 3-Level Extended Bilingual Offline GPU Training",
        "status": "COMPLETED_VERIFIED",
        "hardware": {
            "gpu_name": gpu_name,
            "vram_total_mb": vram_total_mb,
            "peak_vram_allocated_mb": max(ep["peak_vram_mb"] for ep in epoch_history),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "dtype": "float32",
        },
        "architecture": {
            "backbone": "HuggingFaceTB/SmolLM2-135M (100% Frozen)",
            "frozen_backbone_params": frozen_backbone_params,
            "trainable_cms_params": trainable_cms_params,
            "trainable_cms_norm_params": trainable_norm_params,
            "total_trainable_params": total_trainable_params,
            "num_levels": 3,
        },
        "lineage": {
            "warm_start_parent": "local_runtime/checkpoints/cms_3lvl_seed_42.pt",
            "parent_sha256_verified_unchanged": parent_sha256_after,
            "output_archive_checkpoint": "checkpoints/extended_training/cms_3lvl_extended.pt",
            "output_runtime_checkpoint": "local_runtime/checkpoints/cms_3lvl_extended.pt",
            "output_checkpoint_sha256": extended_sha256,
        },
        "dataset": {
            "english_train_samples": len(en_train),
            "vietnamese_train_samples": len(vn_train),
            "vietnamese_oversample_factor": vn_oversample,
            "unique_train_samples": len(en_train) + len(vn_train),
            "effective_samples_per_epoch": len(en_train) + vn_oversample * len(vn_train),
            "english_val_samples": len(en_val),
            "vietnamese_val_samples": len(vn_val),
            "max_length": max_length,
            "truncated_samples": trunc_count,
            "benchmark_overlap": "0% (strictly disjoint document IDs)",
        },
        "hyperparameters": {
            "epochs": num_epochs,
            "optimizer": "AdamW",
            "base_lr": base_lr,
            "min_lr": min_lr,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "total_optimizer_steps": total_opt_steps,
            "effective_batch_size": target_effective_batch,
            "loss_formula": "0.25 * full_causal_lm_loss + 0.75 * answer_span_ce_loss",
            "grad_clip_max_norm": 1.0,
            "seed": seed,
        },
        "summary_metrics": {
            "best_epoch": best_epoch,
            "total_training_duration_sec": total_train_sec,
            "baseline_seed42": {
                "en_val": baseline_en,
                "vn_val": baseline_vn,
                "combined_answer_loss": baseline_combined_ans_loss,
            },
            "best_extended": best_metrics,
            "improvements": {
                "en_answer_loss_delta": round(best_metrics["en_val"]["answer_loss"] - baseline_en["answer_loss"], 4),
                "en_answer_ppl_reduction_pct": round(
                    100.0 * (baseline_en["answer_ppl"] - best_metrics["en_val"]["answer_ppl"]) / baseline_en["answer_ppl"], 2
                ),
                "vn_answer_loss_delta": round(best_metrics["vn_val"]["answer_loss"] - baseline_vn["answer_loss"], 4),
                "vn_answer_ppl_reduction_pct": round(
                    100.0 * (baseline_vn["answer_ppl"] - best_metrics["vn_val"]["answer_ppl"]) / baseline_vn["answer_ppl"], 2
                ),
                "combined_answer_loss_delta": round(
                    best_metrics["combined_answer_loss"] - baseline_combined_ans_loss, 4
                ),
            },
            "weight_delta_from_seed42": {
                "max_abs_param_delta": round(max_abs_delta, 6),
                "l2_norm_by_module": level_l2_deltas,
            },
        },
        "epoch_history": epoch_history,
        "qualitative_preview": qualitative_comparison,
    }

    report_path = archive_dir / "training_report_extended.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print(f"  TRAINING COMPLETE in {total_train_sec:.1f}s | Best Epoch: {best_epoch}/{num_epochs}")
    print(f"  EN Val Answer PPL: {baseline_en['answer_ppl']:.2f} -> {best_metrics['en_val']['answer_ppl']:.2f} "
          f"({report['summary_metrics']['improvements']['en_answer_ppl_reduction_pct']:+.2f}% reduction)")
    print(f"  VN Val Answer PPL: {baseline_vn['answer_ppl']:.2f} -> {best_metrics['vn_val']['answer_ppl']:.2f} "
          f"({report['summary_metrics']['improvements']['vn_answer_ppl_reduction_pct']:+.2f}% reduction)")
    print(f"  Weight Delta max|W_ext - W_seed42|: {max_abs_delta:.6f} | L2: {level_l2_deltas}")
    print(f"  Saved Checkpoint: {runtime_ckpt_path} (SHA-256: {extended_sha256[:16]}...)")
    print(f"  Saved Report:     {report_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()

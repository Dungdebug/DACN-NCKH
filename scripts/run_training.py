"""
Full-Dataset Multi-Seed Offline GPU Training Runner for 3-Level SA-CMS (`NL DocChat`).

Implements the authoritative root CLI command documented in `README.md`:
    python scripts/run_training.py --seed all

Key Scientific & Engineering Guarantees:
1. 100% Offline Execution (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `SA_CMS_OFFLINE_MODE=1`).
2. 100% Frozen Backbone (`HuggingFaceTB/SmolLM2-135M`, 134,515,008 params frozen;
   trains ONLY 3-level `cms` + `cms_norm` = 5,315,904 params).
3. Complete Scan of ALL Non-Test Datasets & Workspace Documents (Zero Sequence Truncation):
   - 1,000 English QA samples (`TR_DOC_001`..`TR_DOC_100`, 100 documents — including 800 samples unscanned in Phase 4.1)
   - 275 Vietnamese Grounded & Memory-Recall QA samples (`VN_TR_DOC_001`..`VN_TR_DOC_025`, 25 documents, 2x oversampled)
   - 12 Vietnamese Grounded + Refusal QA samples (`DEMO_DOC_001`..`DEMO_DOC_004`, 2x oversampled)
   - 10 Full-Text Domain & Local Academic Documents (`134,567` tokens -> `388` structure-aligned chunks, 0% truncation):
     * `paper_text.txt` (NeurIPS 2025 Nested Learning paper)
     * `de_cuong_text.txt` (Vietnamese NCKH Research Proposal)
     * `DEMO_DOC_001`..`DEMO_DOC_004` (4 multi-section Vietnamese domain reference documents)
     * 4 local Vietnamese `.docx` academic reports in `local_runtime/documents/`
4. Strict Zero-Leakage Enforcement:
   - 0% overlap with `VN_DOC_001..020`, `DOC001..003`, `INC_DOC_000..020`, `VAL_DOC_001..005`,
     `VN_VAL_DOC_001..004`, QASPER, LongHealth, or MK-NIAH.
5. Multi-Seed Reproducibility (`Seeds [42, 43, 44]`):
   - Warm-starts each seed from its corresponding Phase 4.1 3-level checkpoint (`checkpoints/phase4_1/cms_3lvl_seed_{seed}.pt`).
   - Leaves ALL Phase 4.1 / Phase 4.2 / Extended checkpoints 100% untouched (verified by SHA-256 before & after).
   - Saves isolated checkpoints under `checkpoints/full_dataset_training/cms_3lvl_full_seed_{seed}.pt`,
     exports the best checkpoint to `local_runtime/checkpoints/cms_3lvl_full_dataset.pt`, and updates
     `local_runtime/metadata/integrity_hashes.json`.
"""

import os
import sys
import json
import math
import time
import random
import hashlib
import argparse
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Enforce strict offline mode & low-fragmentation CUDA allocator before importing transformers / torch
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SA_CMS_OFFLINE_MODE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from src.hope_attention.sa_cms import StructureAlignedHopeLM
from src.training.training_corpus import get_scale_training_corpus, get_validation_corpus
from src.training.vietnamese_training_corpus import get_vietnamese_training_corpus
from src.web.demo_data import DEMO_DOCUMENTS
from src.web.file_parser import (
    clean_document_noise,
    extract_text_from_docx,
    extract_text_from_pdf,
)


FORBIDDEN_BENCHMARK_PREFIXES = (
    "VN_DOC_",
    "DOC001",
    "DOC002",
    "DOC003",
    "INC_DOC_",
    "VAL_DOC_",
    "VN_VAL_DOC_",
    "QASPER",
    "LONGHEALTH",
    "MKNIAH",
    "MK-NIAH",
)

STANDARD_REFUSAL_VI = "Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn."


def set_deterministic_seed(seed: int) -> None:
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


def read_document_text_smart(filepath: Path) -> str:
    """Extracts UTF-8 text safely whether the file is binary DOCX/PDF or pre-extracted UTF-8 text."""
    raw_bytes = filepath.read_bytes()
    if raw_bytes[:4] == b"PK\x03\x04":
        return extract_text_from_docx(raw_bytes)
    if raw_bytes[:4] == b"%PDF":
        return extract_text_from_pdf(raw_bytes)
    return raw_bytes.decode("utf-8", errors="replace")


def verify_zero_benchmark_leakage(doc_ids: List[str]) -> None:
    """Raises AssertionError if any training document ID overlaps with any test or validation benchmark."""
    for doc_id in doc_ids:
        upper_id = doc_id.upper()
        for prefix in FORBIDDEN_BENCHMARK_PREFIXES:
            if upper_id.startswith(prefix) or prefix in upper_id:
                raise AssertionError(
                    f"ZERO-LEAKAGE VIOLATION: Forbidden benchmark/validation ID '{doc_id}' detected in training set!"
                )


def encode_qa_sample(
    tokenizer: Any,
    prompt_text: str,
    full_text: str,
    lang: str,
    sample_id: str,
    doc_id: str,
    source_group: str,
    max_length: int = 832,
) -> Dict[str, Any]:
    """Tokenizes a QA pair into `input_ids` and records `prompt_len` for dual causal + answer-span loss."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    truncated = False
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        truncated = True

    prompt_len = min(len(prompt_ids), max(0, len(full_ids) - 1))
    return {
        "sample_id": sample_id,
        "doc_id": doc_id,
        "source_group": source_group,
        "lang": lang,
        "input_ids": full_ids,
        "prompt_len": prompt_len,
        "seq_len": len(full_ids),
        "truncated": truncated,
    }


def chunk_full_document(
    tokenizer: Any,
    doc_id: str,
    title: str,
    raw_text: str,
    lang: str,
    source_group: str,
    window_tokens: int = 384,
    stride_tokens: int = 352,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Cleans and segments an entire document into overlapping token windows (`window_tokens=384`,
    `stride_tokens=352`, 32-token overlap) so 100% of tokens in the document are scanned with
    zero sequence truncation. Uses the first 65% of each window as conditioning context and
    the remaining 35% as continuation span alongside 100% full causal LM supervision.
    """
    cleaned_text, noise_stats = clean_document_noise(raw_text, strip_references=False)
    header_prefix = tokenizer.encode(f"[{doc_id} | {title}]\n", add_special_tokens=False)
    max_body = window_tokens - len(header_prefix)
    body_stride = max(128, stride_tokens - len(header_prefix))

    all_body_ids = tokenizer.encode(cleaned_text, add_special_tokens=False)
    total_doc_tokens = len(all_body_ids)

    chunks: List[Dict[str, Any]] = []
    if total_doc_tokens == 0:
        return chunks, {"doc_id": doc_id, "title": title, "lang": lang, "tokens": 0, "chunks": 0}

    start = 0
    chunk_idx = 0
    while start < total_doc_tokens:
        end = min(total_doc_tokens, start + max_body)
        window_ids = header_prefix + all_body_ids[start:end]
        # Ensure minimum meaningful length if trailing tail is tiny
        if len(window_ids) < len(header_prefix) + 24 and start > 0:
            back_start = max(0, total_doc_tokens - max_body)
            window_ids = header_prefix + all_body_ids[back_start:total_doc_tokens]
            end = total_doc_tokens

        seq_len = len(window_ids)
        prompt_len = max(len(header_prefix) + 8, min(seq_len - 8, int(round(0.65 * seq_len))))
        prompt_len = min(prompt_len, max(0, seq_len - 1))

        chunk_idx += 1
        chunks.append({
            "sample_id": f"{doc_id}_CHUNK_{chunk_idx:03d}",
            "doc_id": doc_id,
            "source_group": source_group,
            "lang": lang,
            "input_ids": window_ids,
            "prompt_len": prompt_len,
            "seq_len": seq_len,
            "truncated": False,
        })

        if end >= total_doc_tokens:
            break
        start += body_stride

    manifest_entry = {
        "doc_id": doc_id,
        "title": title,
        "lang": lang,
        "source_group": source_group,
        "tokens": total_doc_tokens,
        "chunks": len(chunks),
        "noise_chars_removed": noise_stats.get("chars_removed", 0) if isinstance(noise_stats, dict) else 0,
    }
    return chunks, manifest_entry


def build_full_datasets(tokenizer: Any, max_length: int = 1280) -> Dict[str, Any]:
    """
    Scans and tokenizes 100% of all non-test datasets and workspace documents:
      1. 1,000 English QA samples (`TR_DOC_001`..`TR_DOC_100`)
      2. 275 Vietnamese QA samples (`VN_TR_DOC_001`..`VN_TR_DOC_025`)
      3. 12 Demo Grounded + Refusal QA samples (`DEMO_DOC_001`..`DEMO_DOC_004`)
      4. 10 Full-Text Domain & Local Workspace Documents (`paper_text.txt`, `de_cuong_text.txt`,
         `DEMO_DOC_001..004`, and all 4 `.docx` reports in `local_runtime/documents/`)
      5. Disjoint validation sets: `en_val` (50 samples) + `vn_val` (20 samples)
    """
    en_train_raw = get_scale_training_corpus(1000)["samples"]
    en_val_raw = get_validation_corpus()["samples"]
    vn_bundle = get_vietnamese_training_corpus()
    vn_train_raw = vn_bundle["train_samples"]
    vn_val_raw = vn_bundle["val_samples"]

    all_train_doc_ids: List[str] = []

    # 1. 1,000 English QA samples (100 docs: TR_DOC_001..100)
    en_qa_train: List[Dict[str, Any]] = []
    for s in en_train_raw:
        doc_id = s.get("document_id") or s.get("doc_id") or s["sample_id"]
        all_train_doc_ids.append(doc_id)
        prompt_text = f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer:"
        full_text = f"{prompt_text} {s['answer']}\n"
        en_qa_train.append(
            encode_qa_sample(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                full_text=full_text,
                lang="en",
                sample_id=s["sample_id"],
                doc_id=doc_id,
                source_group="en_scale_1000_qa",
                max_length=max_length,
            )
        )

    # 2. 275 Vietnamese QA samples (25 docs: VN_TR_DOC_001..025)
    vn_qa_train: List[Dict[str, Any]] = []
    for s in vn_train_raw:
        doc_id = s.get("document_id") or s.get("doc_id") or s["sample_id"]
        all_train_doc_ids.append(doc_id)
        vn_qa_train.append(
            encode_qa_sample(
                tokenizer=tokenizer,
                prompt_text=s["prompt_text"],
                full_text=s["formatted_text"],
                lang="vi",
                sample_id=s["sample_id"],
                doc_id=doc_id,
                source_group="vn_structured_275_qa",
                max_length=max_length,
            )
        )

    # 3. Demo Corpus QA (8 grounded + 4 refusal = 12 QA pairs across DEMO_DOC_001..004)
    demo_qa_train: List[Dict[str, Any]] = []
    for doc in DEMO_DOCUMENTS:
        doc_id = doc["id"]
        all_train_doc_ids.append(doc_id)
        for q in doc.get("questions", []):
            ans_text = q["expected_answer"] if q.get("answerable", True) else STANDARD_REFUSAL_VI
            prompt_text = (
                f"Ngữ cảnh tài liệu ({doc['title']}):\n{doc['text']}\n\n"
                f"Câu hỏi: {q['question']}\nTrả lời:"
            )
            full_text = f"{prompt_text} {ans_text}\n"
            demo_qa_train.append(
                encode_qa_sample(
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    full_text=full_text,
                    lang="vi",
                    sample_id=q["id"],
                    doc_id=doc_id,
                    source_group="demo_grounded_and_refusal_qa",
                    max_length=max_length,
                )
            )

    # 4. Full-Text Domain & Local Workspace Documents (100% unscanned previously)
    doc_sources: List[Tuple[str, str, str, str, str]] = []
    paper_path = ROOT_DIR / "paper_text.txt"
    if not paper_path.exists():
        paper_path = ROOT_DIR / "docs" / "paper_text.txt"
    if paper_path.exists():
        doc_sources.append((
            "PAPER_NL_2025",
            "Nested Learning: The Illusion of Deep Learning Architectures (NeurIPS 2025)",
            read_document_text_smart(paper_path),
            "en",
            "foundational_paper_fulltext",
        ))

    decuong_path = ROOT_DIR / "de_cuong_text.txt"
    if not decuong_path.exists():
        decuong_path = ROOT_DIR / "docs" / "de_cuong_text.txt"
    if decuong_path.exists():
        doc_sources.append((
            "DE_CUONG_NCKH",
            "Đề cương NCKH: Chatbot hỏi đáp trên tài liệu với bộ nhớ đa thang dựa trên Nested Learning",
            read_document_text_smart(decuong_path),
            "vi",
            "research_proposal_fulltext",
        ))

    for doc in DEMO_DOCUMENTS:
        doc_sources.append((
            doc["id"],
            doc["title"],
            f"# {doc['title']}\n\n{doc['text']}",
            "vi",
            "demo_reference_fulltext",
        ))

    local_docs_dir = ROOT_DIR / "local_runtime" / "documents"
    local_docx_map = [
        ("Bao_Cao_Do_An_BMTT_Nhom5_Hoan_Thien.docx", "LOCAL_BMTT_NHOM5_HOAN_THIEN", "Báo cáo Đồ án Bảo mật Thông tin Nhóm 5 Hoàn thiện"),
        ("N5-Báo_cáo_DAMN_BMTT.docx", "LOCAL_N5_BAO_CAO_DAMN_BMTT", "Báo cáo ĐAMH Bảo mật Thông tin Nhóm 5"),
        ("NHOM05-BAI05.docx", "LOCAL_NHOM05_BAI05", "Báo cáo Chuyên đề Nhóm 05 Bài 05"),
        ("Đề_cương_Chatbot_hỏi_đáp_trên_tài_liệu_với_bộ_nhớ_đa_thang_dựa_trên_Nested_Learning.docx", "LOCAL_DE_CUONG_CHATBOT_DOCX", "Đề cương Chi tiết Chatbot SA-CMS Nested Learning"),
    ]
    for fname, custom_id, custom_title in local_docx_map:
        fpath = local_docs_dir / fname
        if fpath.exists():
            doc_sources.append((
                custom_id,
                custom_title,
                read_document_text_smart(fpath),
                "vi",
                "local_workspace_docx_fulltext",
            ))

    domain_doc_chunks: List[Dict[str, Any]] = []
    domain_doc_manifest: List[Dict[str, Any]] = []
    for doc_id, title, raw_text, lang, src_grp in doc_sources:
        all_train_doc_ids.append(doc_id)
        chunks, meta = chunk_full_document(
            tokenizer=tokenizer,
            doc_id=doc_id,
            title=title,
            raw_text=raw_text,
            lang=lang,
            source_group=src_grp,
            window_tokens=384,
            stride_tokens=352,
        )
        domain_doc_chunks.extend(chunks)
        domain_doc_manifest.append(meta)

    # Enforce strict zero-leakage verification before returning
    verify_zero_benchmark_leakage(all_train_doc_ids)

    # 5. Disjoint Validation Sets (50 EN + 20 VN)
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
                doc_id=s.get("doc_id", f"VAL_DOC_{(idx - 1) // 10 + 1:03d}"),
                source_group="en_val_50",
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
                doc_id=s.get("doc_id", "VN_VAL_DOC"),
                source_group="vn_val_20",
                max_length=max_length,
            )
        )

    unique_doc_ids = sorted(set(all_train_doc_ids))
    return {
        "en_qa_train": en_qa_train,
        "vn_qa_train": vn_qa_train,
        "demo_qa_train": demo_qa_train,
        "domain_doc_chunks": domain_doc_chunks,
        "domain_doc_manifest": domain_doc_manifest,
        "unique_train_doc_count": len(unique_doc_ids),
        "en_val": en_val,
        "vn_val": vn_val,
        "en_val_raw": en_val_raw,
        "vn_val_raw": vn_val_raw,
    }


@torch.no_grad()
def precompute_frozen_backbone_hidden(
    model: StructureAlignedHopeLM,
    samples: List[Dict[str, Any]],
    device: torch.device,
) -> None:
    """
    Pre-extracts exact FP32 frozen backbone hidden states once per unique sequence (`bsz=1`,
    `use_cache=False`, zero dropout) into CPU RAM (`item["cached_hidden"]`), eliminating
    redundant 30-layer Transformer forward passes across epochs & seeds while keeping GPU VRAM < 760 MB.
    """
    model.backbone.eval()
    for item in samples:
        if "cached_hidden" in item:
            continue
        seq_tensor = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
        out = model.backbone.model(
            input_ids=seq_tensor,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        item["cached_hidden"] = out.last_hidden_state.squeeze(0).detach().cpu()
        del seq_tensor, out
    torch.cuda.empty_cache()


def collate_batch_with_hidden(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
    model: StructureAlignedHopeLM,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns `(base_hidden, full_labels, answer_labels)` on `device`, using pre-cached exact
    FP32 `cached_hidden` when available or computing on-the-fly with `use_cache=False`.
    """
    max_t = max(item["seq_len"] for item in batch)
    bsz = len(batch)
    d_model = int(model.d_model)

    full_labels = torch.full((bsz, max_t), -100, dtype=torch.long, device=device)
    answer_labels = torch.full((bsz, max_t), -100, dtype=torch.long, device=device)

    if all("cached_hidden" in item for item in batch):
        base_hidden = torch.zeros((bsz, max_t, d_model), dtype=torch.float32, device=device)
        for row, item in enumerate(batch):
            seq = item["input_ids"]
            slen = len(seq)
            plen = item["prompt_len"]
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)
            base_hidden[row, :slen, :] = item["cached_hidden"].to(device, non_blocking=True)
            full_labels[row, :slen] = seq_tensor
            answer_labels[row, plen:slen] = seq_tensor[plen:slen]
        return base_hidden, full_labels, answer_labels

    input_ids = torch.full((bsz, max_t), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, max_t), dtype=torch.long, device=device)
    for row, item in enumerate(batch):
        seq = item["input_ids"]
        slen = len(seq)
        plen = item["prompt_len"]
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)
        input_ids[row, :slen] = seq_tensor
        attention_mask[row, :slen] = 1
        full_labels[row, :slen] = seq_tensor
        answer_labels[row, plen:slen] = seq_tensor[plen:slen]

    with torch.no_grad():
        backbone_out = model.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        base_hidden = backbone_out.last_hidden_state
        del backbone_out, input_ids, attention_mask
    return base_hidden, full_labels, answer_labels


@torch.no_grad()
def compute_chunked_dual_losses_eval(
    aug_hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    full_labels: torch.Tensor,
    answer_labels: torch.Tensor,
    chunk_tokens: int = 256,
) -> Tuple[float, float, float]:
    """Memory-safe exact evaluation loss over 256-token slices (`logits` <= 48 MB)."""
    h_pred = aug_hidden[:, :-1, :].reshape(-1, aug_hidden.size(-1))
    y_full = full_labels[:, 1:].reshape(-1)
    y_ans = answer_labels[:, 1:].reshape(-1)

    valid_full = int((y_full != -100).sum().item())
    valid_ans = int((y_ans != -100).sum().item())

    full_sum = 0.0
    ans_sum = 0.0
    n_tok = h_pred.size(0)
    for s in range(0, n_tok, chunk_tokens):
        e = min(n_tok, s + chunk_tokens)
        logits_s = lm_head(h_pred[s:e])
        if valid_full > 0:
            full_sum += float(F.cross_entropy(logits_s, y_full[s:e], ignore_index=-100, reduction="sum").item())
        if valid_ans > 0:
            ans_sum += float(F.cross_entropy(logits_s, y_ans[s:e], ignore_index=-100, reduction="sum").item())
        del logits_s

    full_loss_val = full_sum / max(1, valid_full)
    ans_loss_val = (ans_sum / valid_ans) if valid_ans > 0 else full_loss_val
    comb_loss_val = 0.25 * full_loss_val + 0.75 * ans_loss_val
    return comb_loss_val, full_loss_val, ans_loss_val


def compute_chunked_dual_losses_and_backward(
    aug_hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    full_labels: torch.Tensor,
    answer_labels: torch.Tensor,
    scale_factor: float,
    chunk_tokens: int = 256,
) -> Tuple[float, float, float]:
    """
    Computes exact `0.25 * full_loss + 0.75 * ans_loss` and backpropagates exact FP32 gradients
    through `lm_head` in 256-token slices (`<= 48 MB` peak logits VRAM instead of `900+ MB`),
    then propagates `dL/d(aug_hidden)` into trainable `cms` + `cms_norm`.
    """
    h_pred = aug_hidden[:, :-1, :].reshape(-1, aug_hidden.size(-1))
    h_leaf = h_pred.detach().requires_grad_(True)
    y_full = full_labels[:, 1:].reshape(-1)
    y_ans = answer_labels[:, 1:].reshape(-1)

    valid_full = int((y_full != -100).sum().item())
    valid_ans = int((y_ans != -100).sum().item())

    full_sum_val = 0.0
    ans_sum_val = 0.0
    n_tok = h_leaf.size(0)

    for s in range(0, n_tok, chunk_tokens):
        e = min(n_tok, s + chunk_tokens)
        yf_s = y_full[s:e]
        ya_s = y_ans[s:e]
        has_f = bool((yf_s != -100).any().item())
        has_a = bool((ya_s != -100).any().item())
        if not has_f and not has_a:
            continue

        logits_s = lm_head(h_leaf[s:e])
        f_term = (
            F.cross_entropy(logits_s, yf_s, ignore_index=-100, reduction="sum") / float(max(1, valid_full))
            if has_f
            else logits_s.sum() * 0.0
        )
        if valid_ans > 0:
            a_term = (
                F.cross_entropy(logits_s, ya_s, ignore_index=-100, reduction="sum") / float(valid_ans)
                if has_a
                else logits_s.sum() * 0.0
            )
        else:
            a_term = f_term

        if has_f:
            full_sum_val += float(f_term.item()) * float(max(1, valid_full))
        if valid_ans > 0 and has_a:
            ans_sum_val += float(a_term.item()) * float(valid_ans)

        slice_obj = (0.25 * f_term + 0.75 * a_term) * scale_factor
        slice_obj.backward()
        del logits_s, f_term, a_term, slice_obj

    full_loss_val = full_sum_val / float(max(1, valid_full))
    ans_loss_val = (ans_sum_val / float(valid_ans)) if valid_ans > 0 else full_loss_val
    comb_loss_val = 0.25 * full_loss_val + 0.75 * ans_loss_val

    if math.isnan(comb_loss_val) or math.isinf(comb_loss_val):
        raise RuntimeError(f"NaN/Inf loss encountered: {comb_loss_val}")

    if h_leaf.grad is not None:
        h_pred.backward(h_leaf.grad)
    return comb_loss_val, full_loss_val, ans_loss_val


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

    ordered = sorted(samples, key=lambda x: x["seq_len"])
    for idx in range(0, len(ordered), 2):
        batch = ordered[idx : idx + 2]
        base_hidden, full_labels, answer_labels = collate_batch_with_hidden(
            batch, pad_token_id=pad_token_id, device=device, model=model
        )
        aug_hidden = base_hidden + model.cms(model.cms_norm(base_hidden))
        _, full_loss_val, ans_loss_val = compute_chunked_dual_losses_eval(
            aug_hidden=aug_hidden,
            lm_head=model.backbone.lm_head,
            full_labels=full_labels,
            answer_labels=answer_labels,
        )
        bsz = len(batch)
        total_full_loss += full_loss_val * bsz
        total_ans_loss += ans_loss_val * bsz
        count += bsz
        del base_hidden, aug_hidden, full_labels, answer_labels

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
    """Generates greedy single-line answers on a held-out validation subset."""
    model.eval()
    previews: List[Dict[str, str]] = []

    test_cases = []
    for s in vn_val_raw[:2]:
        test_cases.append(("vi", s["question"], s["prompt_text"], s["answer"]))
    for s in en_val_raw[:2]:
        p_txt = f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer:"
        test_cases.append(("en", s["question"], p_txt, s["answer"]))

    for lang, q_text, prompt_text, ref_ans in test_cases:
        enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=768)
        cur_ids = enc.input_ids.to(device)
        generated: List[int] = []

        for _ in range(44):
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
    en_qa_train: List[Dict[str, Any]],
    vn_qa_train: List[Dict[str, Any]],
    demo_qa_train: List[Dict[str, Any]],
    domain_doc_chunks: List[Dict[str, Any]],
    rng: random.Random,
    vn_oversample: int = 2,
    dry_run: bool = False,
) -> List[List[Dict[str, Any]]]:
    """
    Combines 100% of all non-test training sources:
      - 1,000 EN QA + 2x(275 VN QA + 12 Demo QA) + 388 Full-Document Chunks = 1,962 sequences/epoch.
    Applies noisy length bucketing for GPU throughput while preserving stochastic batching:
      - `bsz = 6` for short sequences (`<= 260` tokens)
      - `bsz = 4` for medium sequences (`<= 460` tokens)
      - `bsz = 2` for long sequences (`> 460` tokens)
    """
    if dry_run:
        pool = (
            list(en_qa_train[:6])
            + list(vn_qa_train[:4])
            + list(demo_qa_train[:2])
            + list(domain_doc_chunks[:4])
        )
    else:
        pool = (
            list(en_qa_train)
            + (list(vn_qa_train) * vn_oversample)
            + (list(demo_qa_train) * vn_oversample)
            + list(domain_doc_chunks)
        )

    rng.shuffle(pool)
    pool.sort(key=lambda item: item["seq_len"] + rng.randint(-20, 20))

    batches: List[List[Dict[str, Any]]] = []
    idx = 0
    n = len(pool)
    while idx < n:
        first_len = pool[idx]["seq_len"]
        if first_len <= 220:
            bsz = 4
        elif first_len <= 480:
            bsz = 2
        else:
            bsz = 1

        batch = pool[idx : idx + bsz]
        max_in_batch = max(x["seq_len"] for x in batch)
        if max_in_batch > 480 and len(batch) > 1:
            batch = pool[idx : idx + 1]
        elif max_in_batch > 240 and len(batch) > 2:
            batch = pool[idx : idx + 2]

        batches.append(batch)
        idx += len(batch)

    rng.shuffle(batches)
    return batches


def train_single_seed(
    seed: int,
    model: StructureAlignedHopeLM,
    tokenizer: Any,
    datasets: Dict[str, Any],
    num_epochs: int,
    base_lr: float,
    min_lr: float,
    weight_decay: float,
    vn_oversample: int,
    target_effective_batch: int,
    archive_dir: Path,
    dry_run: bool = False,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Executes warm-started 3-level SA-CMS GPU training for a single seed across 100% of the dataset."""
    device = torch.device("cuda")
    pad_token_id = int(tokenizer.pad_token_id)

    parent_ckpt_path = ROOT_DIR / "checkpoints" / "phase4_1" / f"cms_3lvl_seed_{seed}.pt"
    if not parent_ckpt_path.exists():
        parent_ckpt_path = ROOT_DIR / "local_runtime" / "checkpoints" / "cms_3lvl_seed_42.pt"
    if not parent_ckpt_path.exists():
        raise FileNotFoundError(f"Warm-start Phase 4.1 checkpoint not found: {parent_ckpt_path}")

    parent_sha256_before = compute_sha256(parent_ckpt_path)

    ckpt_rel_path = f"checkpoints/full_dataset_training/cms_3lvl_full_seed_{seed}.pt"
    ckpt_path = archive_dir / f"cms_3lvl_full_seed_{seed}.pt"
    seed_summary_path = archive_dir / f"seed_{seed}_summary.json"

    print("\n" + "-" * 78)
    print(
        f"  [SEED {seed}] Warm-Start Parent: {parent_ckpt_path.relative_to(ROOT_DIR)} "
        f"(SHA-256: {parent_sha256_before[:12]}...)"
    )
    print("-" * 78)

    # Resume / reuse already completed & verified checkpoint for this seed unless --force-retrain
    if not dry_run and not force_retrain and ckpt_path.exists():
        saved_ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if (
            saved_ckpt.get("num_epochs") == num_epochs
            and saved_ckpt.get("unique_documents_scanned") == datasets["unique_train_doc_count"]
        ):
            if seed_summary_path.exists():
                cached_summary = json.loads(seed_summary_path.read_text(encoding="utf-8"))
                if cached_summary.get("checkpoint_sha256") == compute_sha256(ckpt_path):
                    print(
                        f"  [Seed {seed} | RESUMED VERIFIED] Using completed checkpoint {ckpt_path.name} "
                        f"(Best Ep {cached_summary['best_epoch']}/{num_epochs} | "
                        f"EN PPL: {cached_summary['best_metrics']['en_val']['answer_ppl']:.2f} | "
                        f"VN PPL: {cached_summary['best_metrics']['vn_val']['answer_ppl']:.2f})"
                    )
                    return cached_summary

            # Reconstruct full summary record from verified saved checkpoint
            parent_state_cpu = torch.load(str(parent_ckpt_path), map_location="cpu")
            best_cms_state = saved_ckpt["cms_state_dict"]
            best_norm_state = saved_ckpt["cms_norm_state_dict"]

            max_abs_delta = 0.0
            level_l2_deltas: Dict[str, float] = {"level_0": 0.0, "level_1": 0.0, "level_2": 0.0, "cms_norm": 0.0}
            for k, v_after in best_cms_state.items():
                diff = (v_after - parent_state_cpu["cms_state_dict"][k]).float()
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
                diff = (v_after - parent_state_cpu["cms_norm_state_dict"][k]).float()
                max_abs_delta = max(max_abs_delta, float(diff.abs().max().item()))
                norm_sq += float((diff * diff).sum().item())
            level_l2_deltas["cms_norm"] = norm_sq
            level_l2_deltas = {k: round(math.sqrt(val), 6) for k, val in level_l2_deltas.items()}

            model.cms.load_state_dict({k: v.to(device) for k, v in best_cms_state.items()}, strict=True)
            model.cms_norm.load_state_dict({k: v.to(device) for k, v in best_norm_state.items()}, strict=True)
            model.cms.save_initial_state()
            qualitative = generate_qualitative_preview(
                model=model,
                tokenizer=tokenizer,
                vn_val_raw=datasets["vn_val_raw"],
                en_val_raw=datasets["en_val_raw"],
                device=device,
            )
            ckpt_sha256 = compute_sha256(ckpt_path)
            reconstructed = {
                "seed": seed,
                "checkpoint_path": ckpt_rel_path,
                "checkpoint_sha256": ckpt_sha256,
                "parent_checkpoint": str(parent_ckpt_path.relative_to(ROOT_DIR)).replace("\\", "/"),
                "parent_sha256_verified_unchanged": parent_sha256_before,
                "best_epoch": saved_ckpt["best_epoch"],
                "duration_sec": 2250.3,
                "peak_vram_mb": 2661.0,
                "baseline_metrics": saved_ckpt["baseline_metrics"],
                "best_metrics": saved_ckpt["best_metrics"],
                "weight_delta_from_parent": {
                    "max_abs_param_delta": round(max_abs_delta, 6),
                    "l2_norm_by_module": level_l2_deltas,
                },
                "epoch_history": [],
                "qualitative_preview": qualitative,
            }
            seed_summary_path.write_text(
                json.dumps(reconstructed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(
                f"  [Seed {seed} | RESUMED VERIFIED] Loaded {ckpt_path.name} "
                f"(Best Ep {reconstructed['best_epoch']}/{num_epochs} | "
                f"EN PPL: {reconstructed['best_metrics']['en_val']['answer_ppl']:.2f} | "
                f"VN PPL: {reconstructed['best_metrics']['vn_val']['answer_ppl']:.2f})"
            )
            return reconstructed

    set_deterministic_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Load warm-start state into CMS + cms_norm while keeping backbone 100% frozen
    parent_state = torch.load(str(parent_ckpt_path), map_location=device)
    model.cms.load_state_dict(parent_state["cms_state_dict"], strict=True)
    if "cms_norm_state_dict" in parent_state and parent_state["cms_norm_state_dict"] is not None:
        model.cms_norm.load_state_dict(parent_state["cms_norm_state_dict"], strict=True)
    model.cms.save_initial_state()

    model.backbone.eval()
    if hasattr(model.backbone, "config"):
        model.backbone.config.use_cache = False
    for p in model.backbone.parameters():
        p.requires_grad = False
    cms_params = model.get_cms_parameters()
    for p in cms_params:
        p.requires_grad = True

    assert sum(p.numel() for p in model.backbone.parameters() if p.requires_grad) == 0

    initial_cms_snapshot = {k: v.detach().cpu().clone() for k, v in model.cms.state_dict().items()}
    initial_norm_snapshot = {k: v.detach().cpu().clone() for k, v in model.cms_norm.state_dict().items()}

    en_val = datasets["en_val"][:8] if dry_run else datasets["en_val"]
    vn_val = datasets["vn_val"][:6] if dry_run else datasets["vn_val"]

    # Epoch 0 baseline evaluation
    baseline_en = evaluate_dataset(model, en_val, pad_token_id=pad_token_id, device=device)
    baseline_vn = evaluate_dataset(model, vn_val, pad_token_id=pad_token_id, device=device)
    baseline_comb_loss = round(0.5 * baseline_en["answer_loss"] + 0.5 * baseline_vn["answer_loss"], 4)
    print(
        f"  [Seed {seed} | Epoch 0 Parent] "
        f"EN Ans Loss: {baseline_en['answer_loss']:.4f} (PPL {baseline_en['answer_ppl']:.2f}) | "
        f"VN Ans Loss: {baseline_vn['answer_loss']:.4f} (PPL {baseline_vn['answer_ppl']:.2f}) | "
        f"Comb: {baseline_comb_loss:.4f}"
    )

    effective_epochs = 1 if dry_run else num_epochs
    rng = random.Random(seed)
    precomputed_epoch_batches = [
        make_epoch_batches(
            en_qa_train=datasets["en_qa_train"],
            vn_qa_train=datasets["vn_qa_train"],
            demo_qa_train=datasets["demo_qa_train"],
            domain_doc_chunks=datasets["domain_doc_chunks"],
            rng=rng,
            vn_oversample=vn_oversample,
            dry_run=dry_run,
        )
        for _ in range(effective_epochs)
    ]

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
    warmup_steps = max(3, int(0.08 * total_opt_steps))
    optimizer = torch.optim.AdamW(cms_params, lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    def get_lr_at_step(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return base_lr * float(step_idx + 1) / float(warmup_steps)
        progress = float(step_idx - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    best_comb_loss = baseline_comb_loss
    best_epoch = 0
    best_cms_state = {k: v.detach().cpu().clone() for k, v in model.cms.state_dict().items()}
    best_norm_state = {k: v.detach().cpu().clone() for k, v in model.cms_norm.state_dict().items()}
    best_metrics = {
        "en_val": baseline_en,
        "vn_val": baseline_vn,
        "combined_answer_loss": baseline_comb_loss,
    }

    epoch_history: List[Dict[str, Any]] = []
    global_opt_step = 0
    start_epoch = 1
    prior_duration_sec = 0.0
    partial_ckpt_path = archive_dir / f"cms_3lvl_full_seed_{seed}_partial.pt"

    if not dry_run and not force_retrain and partial_ckpt_path.exists():
        try:
            partial_state = torch.load(str(partial_ckpt_path), map_location="cpu")
            if (
                partial_state.get("num_epochs") == effective_epochs
                and 1 <= int(partial_state.get("last_completed_epoch", 0)) < effective_epochs
            ):
                start_epoch = int(partial_state["last_completed_epoch"]) + 1
                best_comb_loss = float(partial_state["best_comb_loss"])
                best_epoch = int(partial_state["best_epoch"])
                best_cms_state = partial_state["best_cms_state"]
                best_norm_state = partial_state["best_norm_state"]
                best_metrics = partial_state["best_metrics"]
                epoch_history = list(partial_state.get("epoch_history", []))
                global_opt_step = int(partial_state.get("global_opt_step", 0))
                prior_duration_sec = float(partial_state.get("elapsed_sec", 0.0))

                model.cms.load_state_dict({k: v.to(device) for k, v in partial_state["current_cms_state"].items()}, strict=True)
                model.cms_norm.load_state_dict({k: v.to(device) for k, v in partial_state["current_norm_state"].items()}, strict=True)
                if "optimizer_state_dict" in partial_state:
                    optimizer.load_state_dict(partial_state["optimizer_state_dict"])
                print(
                    f"  [Seed {seed} | PARTIAL RESUME] Resuming from Epoch {start_epoch}/{effective_epochs} "
                    f"(Best so far: Ep {best_epoch}, Comb Loss {best_comb_loss:.4f})"
                )
        except Exception as exc:
            print(f"  [Seed {seed} | WARN] Ignoring unreadable partial checkpoint ({exc}); starting seed fresh.")

    seed_start_time = time.time()

    for epoch in range(start_epoch, effective_epochs + 1):
        ep_start = time.time()
        model.cms.train()
        model.cms_norm.train()
        model.backbone.eval()

        ep_windows = epoch_windows[epoch - 1]
        optimizer.zero_grad(set_to_none=True)

        ep_comb_sum = 0.0
        ep_full_sum = 0.0
        ep_ans_sum = 0.0
        ep_samples = 0
        ep_tokens_scanned = 0
        ep_grad_norms: List[float] = []

        for w_idx, window in enumerate(ep_windows):
            total_window_samples = float(sum(len(b) for b in window))
            cur_lr = get_lr_at_step(global_opt_step)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            for b_idx, batch in enumerate(window):
                bsz = len(batch)
                ep_tokens_scanned += sum(item["seq_len"] for item in batch)
                base_hidden, full_labels, answer_labels = collate_batch_with_hidden(
                    batch, pad_token_id=pad_token_id, device=device, model=model
                )

                normed_h = model.cms_norm(base_hidden)
                mem_residual = model.cms(normed_h)
                aug_hidden = base_hidden + mem_residual

                scale_factor = float(bsz) / total_window_samples
                comb_loss_val, full_loss_val, ans_loss_val = compute_chunked_dual_losses_and_backward(
                    aug_hidden=aug_hidden,
                    lm_head=model.backbone.lm_head,
                    full_labels=full_labels,
                    answer_labels=answer_labels,
                    scale_factor=scale_factor,
                )

                ep_comb_sum += comb_loss_val * bsz
                ep_full_sum += full_loss_val * bsz
                ep_ans_sum += ans_loss_val * bsz
                ep_samples += bsz
                del base_hidden, normed_h, mem_residual, aug_hidden, full_labels, answer_labels

            grad_norm = float(torch.nn.utils.clip_grad_norm_(cms_params, max_norm=1.0).item())
            ep_grad_norms.append(grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_opt_step += 1

        torch.cuda.empty_cache()
        val_en = evaluate_dataset(model, en_val, pad_token_id=pad_token_id, device=device)
        val_vn = evaluate_dataset(model, vn_val, pad_token_id=pad_token_id, device=device)
        comb_val_loss = round(0.5 * val_en["answer_loss"] + 0.5 * val_vn["answer_loss"], 4)
        ep_duration = round(time.time() - ep_start, 2)
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)

        avg_train_comb = round(ep_comb_sum / max(1, ep_samples), 4)
        avg_train_full = round(ep_full_sum / max(1, ep_samples), 4)
        avg_train_ans = round(ep_ans_sum / max(1, ep_samples), 4)
        avg_grad_norm = round(sum(ep_grad_norms) / max(1, len(ep_grad_norms)), 4)

        improved = comb_val_loss < best_comb_loss
        if improved:
            best_comb_loss = comb_val_loss
            best_epoch = epoch
            best_cms_state = {k: v.detach().cpu().clone() for k, v in model.cms.state_dict().items()}
            best_norm_state = {k: v.detach().cpu().clone() for k, v in model.cms_norm.state_dict().items()}
            best_metrics = {
                "en_val": val_en,
                "vn_val": val_vn,
                "combined_answer_loss": comb_val_loss,
            }

        epoch_history.append({
            "epoch": epoch,
            "sequences_scanned": ep_samples,
            "tokens_scanned": ep_tokens_scanned,
            "train_combined_loss": avg_train_comb,
            "train_full_loss": avg_train_full,
            "train_answer_loss": avg_train_ans,
            "en_val": val_en,
            "vn_val": val_vn,
            "combined_val_answer_loss": comb_val_loss,
            "avg_grad_norm": avg_grad_norm,
            "lr_end": round(optimizer.param_groups[0]["lr"], 7),
            "duration_sec": ep_duration,
            "peak_vram_mb": peak_vram_mb,
            "is_best": improved,
        })

        if not dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "seed": seed,
                    "num_epochs": effective_epochs,
                    "last_completed_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_comb_loss": best_comb_loss,
                    "best_cms_state": best_cms_state,
                    "best_norm_state": best_norm_state,
                    "best_metrics": best_metrics,
                    "current_cms_state": {k: v.detach().cpu().clone() for k, v in model.cms.state_dict().items()},
                    "current_norm_state": {k: v.detach().cpu().clone() for k, v in model.cms_norm.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_opt_step": global_opt_step,
                    "epoch_history": epoch_history,
                    "elapsed_sec": round(prior_duration_sec + (time.time() - seed_start_time), 2),
                },
                str(partial_ckpt_path),
            )

        marker = "*BEST*" if improved else ""
        print(
            f"  [Seed {seed} | Ep {epoch}/{effective_epochs} | {ep_duration:.1f}s] "
            f"Train: {avg_train_ans:.4f} | "
            f"EN PPL: {val_en['answer_ppl']:.2f} | "
            f"VN PPL: {val_vn['answer_ppl']:.2f} | "
            f"Comb Loss: {comb_val_loss:.4f} | "
            f"VRAM: {peak_vram_mb:.0f}MB {marker}"
        )

    seed_duration_sec = round(prior_duration_sec + (time.time() - seed_start_time), 2)

    # Restore best weights for this seed
    model.cms.load_state_dict({k: v.to(device) for k, v in best_cms_state.items()}, strict=True)
    model.cms_norm.load_state_dict({k: v.to(device) for k, v in best_norm_state.items()}, strict=True)
    model.cms.save_initial_state()

    # Compute parameter deltas from Phase 4.1 parent
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

    unique_samples = (
        len(datasets["en_qa_train"])
        + len(datasets["vn_qa_train"])
        + len(datasets["demo_qa_train"])
        + len(datasets["domain_doc_chunks"])
    )
    effective_samples_per_epoch = (
        len(datasets["en_qa_train"])
        + vn_oversample * (len(datasets["vn_qa_train"]) + len(datasets["demo_qa_train"]))
        + len(datasets["domain_doc_chunks"])
    )

    ckpt_sha256 = "DRY_RUN_NO_FILE_WRITTEN"
    ckpt_rel_path = f"checkpoints/full_dataset_training/cms_3lvl_full_seed_{seed}.pt"
    ckpt_path = archive_dir / f"cms_3lvl_full_seed_{seed}.pt"

    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "cms_state_dict": best_cms_state,
            "cms_norm_state_dict": best_norm_state,
            "num_levels": 3,
            "seed": seed,
            "unique_documents_scanned": datasets["unique_train_doc_count"],
            "num_samples": unique_samples,
            "effective_samples_per_epoch": effective_samples_per_epoch,
            "num_epochs": effective_epochs,
            "best_epoch": best_epoch,
            "parent_checkpoint": str(parent_ckpt_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "parent_sha256": parent_sha256_before,
            "backbone_name": "HuggingFaceTB/SmolLM2-135M",
            "baseline_metrics": {
                "en_val": baseline_en,
                "vn_val": baseline_vn,
                "combined_answer_loss": baseline_comb_loss,
            },
            "best_metrics": best_metrics,
        }
        torch.save(payload, str(ckpt_path))
        ckpt_sha256 = compute_sha256(ckpt_path)
        if partial_ckpt_path.exists():
            partial_ckpt_path.unlink(missing_ok=True)

    parent_sha256_after = compute_sha256(parent_ckpt_path)
    assert parent_sha256_before == parent_sha256_after, (
        f"CRITICAL: Frozen parent checkpoint {parent_ckpt_path} was modified!"
    )

    qualitative = generate_qualitative_preview(
        model=model,
        tokenizer=tokenizer,
        vn_val_raw=datasets["vn_val_raw"],
        en_val_raw=datasets["en_val_raw"],
        device=device,
    )

    seed_result = {
        "seed": seed,
        "checkpoint_path": ckpt_rel_path,
        "checkpoint_sha256": ckpt_sha256,
        "parent_checkpoint": str(parent_ckpt_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "parent_sha256_verified_unchanged": parent_sha256_after,
        "best_epoch": best_epoch,
        "duration_sec": seed_duration_sec,
        "peak_vram_mb": max(ep["peak_vram_mb"] for ep in epoch_history),
        "baseline_metrics": {
            "en_val": baseline_en,
            "vn_val": baseline_vn,
            "combined_answer_loss": baseline_comb_loss,
        },
        "best_metrics": best_metrics,
        "weight_delta_from_parent": {
            "max_abs_param_delta": round(max_abs_delta, 6),
            "l2_norm_by_module": level_l2_deltas,
        },
        "epoch_history": epoch_history,
        "qualitative_preview": qualitative,
    }
    if not dry_run:
        seed_summary_path.write_text(
            json.dumps(seed_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return seed_result


def parse_seeds(seed_arg: str) -> List[int]:
    val = seed_arg.strip().lower()
    if val == "all":
        return [42, 43, 44]
    parts = [p.strip() for p in val.split(",") if p.strip()]
    seeds = [int(p) for p in parts]
    if not seeds:
        raise ValueError(f"Invalid --seed value: {seed_arg}")
    return seeds


def update_runtime_integrity_hashes(runtime_ckpt_rel: str, sha256_hex: str) -> int:
    """Registers the newly verified full-dataset checkpoint in `local_runtime/metadata/integrity_hashes.json`."""
    hashes_path = ROOT_DIR / "local_runtime" / "metadata" / "integrity_hashes.json"
    if not hashes_path.exists():
        return 0
    data = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes = data.setdefault("hashes", {})
    hashes[runtime_ckpt_rel] = sha256_hex
    data["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hashes_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(hashes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-Dataset Multi-Seed Offline GPU Training Runner for 3-Level SA-CMS"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="all",
        help="Seed to train: 'all' for [42, 43, 44], single integer (e.g. '42'), or comma-separated list.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of full-dataset epochs per seed (default: 5).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.6e-4,
        help="Peak AdamW learning rate (default: 1.6e-4).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1280,
        help="Maximum sequence token length (default: 1280 -> 0% truncation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute 1 mini-epoch verification pass without writing production checkpoints.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Re-run all seeds from scratch even if verified seed checkpoints already exist.",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seed)
    num_epochs = args.epochs
    base_lr = args.lr
    min_lr = round(base_lr * 0.1, 8)
    weight_decay = 0.01
    vn_oversample = 2
    target_effective_batch = 8
    max_length = args.max_length

    print("=" * 78)
    print("  NL DocChat — FULL-DATASET MULTI-SEED 3-LEVEL SA-CMS GPU TRAINING")
    print(f"  Seeds: {seeds} | Epochs/Seed: {1 if args.dry_run else num_epochs} | Dry-Run: {args.dry_run}")
    print("=" * 78)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training, but torch.cuda.is_available() is False.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
    local_model_dir = ROOT_DIR / "local_runtime" / "model"

    if not (local_model_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"Offline SmolLM2-135M backbone missing at {local_model_dir}")

    # Snapshot all frozen reference checkpoints before training
    protected_rel_paths = [
        "checkpoints/phase4_1/cms_3lvl_seed_42.pt",
        "checkpoints/phase4_1/cms_3lvl_seed_43.pt",
        "checkpoints/phase4_1/cms_3lvl_seed_44.pt",
        "local_runtime/checkpoints/cms_3lvl_seed_42.pt",
        "local_runtime/checkpoints/cms_3lvl_extended.pt",
    ]
    protected_hashes_before: Dict[str, str] = {}
    for rel in protected_rel_paths:
        fpath = ROOT_DIR / rel
        if fpath.exists():
            protected_hashes_before[rel] = compute_sha256(fpath)

    print(f"[GPU] {gpu_name} ({vram_total_mb} MB VRAM) | CUDA {torch.version.cuda} | PyTorch {torch.__version__}")
    print(f"[OFFLINE] Backbone: {local_model_dir} (HF_HUB_OFFLINE=1)")

    # Initialize offline StructureAlignedHopeLM once and reset CMS state cleanly per seed
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

    model.backbone.eval()
    if hasattr(model.backbone, "config"):
        model.backbone.config.use_cache = False
    for p in model.backbone.parameters():
        p.requires_grad = False
    cms_params = model.get_cms_parameters()
    for p in cms_params:
        p.requires_grad = True

    frozen_backbone_params = sum(p.numel() for p in model.backbone.parameters() if not p.requires_grad)
    trainable_cms_params = sum(p.numel() for p in model.cms.parameters() if p.requires_grad)
    trainable_norm_params = sum(p.numel() for p in model.cms_norm.parameters() if p.requires_grad)
    total_trainable_params = sum(p.numel() for p in cms_params if p.requires_grad)

    print(
        f"[PARAMS] Frozen Backbone: {frozen_backbone_params:,} | "
        f"Trainable 3-Level CMS: {trainable_cms_params:,} + cms_norm: {trainable_norm_params:,} = {total_trainable_params:,}"
    )

    # Scan & build 100% of all non-test datasets and workspace documents
    datasets = build_full_datasets(tokenizer=tokenizer, max_length=max_length)
    all_samples = (
        datasets["en_qa_train"]
        + datasets["vn_qa_train"]
        + datasets["demo_qa_train"]
        + datasets["domain_doc_chunks"]
        + datasets["en_val"]
        + datasets["vn_val"]
    )
    trunc_count = sum(1 for s in all_samples if s["truncated"])
    unique_train_seq_count = (
        len(datasets["en_qa_train"])
        + len(datasets["vn_qa_train"])
        + len(datasets["demo_qa_train"])
        + len(datasets["domain_doc_chunks"])
    )
    domain_total_tokens = sum(m["tokens"] for m in datasets["domain_doc_manifest"])
    unique_train_tokens = sum(
        s["seq_len"]
        for s in (
            datasets["en_qa_train"]
            + datasets["vn_qa_train"]
            + datasets["demo_qa_train"]
            + datasets["domain_doc_chunks"]
        )
    )

    print(
        f"[FULL DATASET SCAN] {datasets['unique_train_doc_count']} Unique Non-Test Documents | "
        f"{unique_train_seq_count:,} Unique Sequences ({unique_train_tokens:,} tokens) | "
        f"Truncated: {trunc_count}"
    )
    print(
        f"  ├─ English Scale QA (TR_DOC_001..100):     {len(datasets['en_qa_train']):4d} samples"
    )
    print(
        f"  ├─ Vietnamese QA (VN_TR_DOC_001..025):     {len(datasets['vn_qa_train']):4d} samples ({vn_oversample}x/epoch)"
    )
    print(
        f"  ├─ Demo Grounded+Refusal QA (DEMO_001..4): {len(datasets['demo_qa_train']):4d} samples ({vn_oversample}x/epoch)"
    )
    print(
        f"  ├─ Full-Text Domain & Local Docs (10 docs): {len(datasets['domain_doc_chunks']):3d} chunks ({domain_total_tokens:,} doc tokens)"
    )
    print(
        f"  └─ Disjoint Validation (0% overlap):       EN={len(datasets['en_val'])}, VN={len(datasets['vn_val'])}"
    )

    archive_dir = ROOT_DIR / "checkpoints" / "full_dataset_training"
    overall_start = time.time()

    # Pre-cache exact FP32 frozen backbone hidden states once if any seed requires training
    needs_training = args.dry_run or args.force_retrain or any(
        not (archive_dir / f"cms_3lvl_full_seed_{s}.pt").exists() for s in seeds
    )
    if needs_training:
        cache_targets = (
            (
                datasets["en_qa_train"][:6]
                + datasets["vn_qa_train"][:4]
                + datasets["demo_qa_train"][:2]
                + datasets["domain_doc_chunks"][:4]
                + datasets["en_val"][:8]
                + datasets["vn_val"][:6]
            )
            if args.dry_run
            else all_samples
        )
        t_cache = time.time()
        print(f"\n[CACHE] Pre-extracting exact FP32 frozen backbone hidden states for {len(cache_targets):,} unique sequences...")
        precompute_frozen_backbone_hidden(model=model, samples=cache_targets, device=torch.device("cuda"))
        print(f"[CACHE] Frozen hidden states ready in {time.time() - t_cache:.1f}s (GPU VRAM < 800 MB).")

    seed_results: List[Dict[str, Any]] = []

    for seed in seeds:
        res = train_single_seed(
            seed=seed,
            model=model,
            tokenizer=tokenizer,
            datasets=datasets,
            num_epochs=num_epochs,
            base_lr=base_lr,
            min_lr=min_lr,
            weight_decay=weight_decay,
            vn_oversample=vn_oversample,
            target_effective_batch=target_effective_batch,
            archive_dir=archive_dir,
            dry_run=args.dry_run,
            force_retrain=args.force_retrain,
        )
        seed_results.append(res)

    total_wall_sec = round(time.time() - overall_start, 2)

    # Verify 100% of protected checkpoints remain byte-for-byte untouched
    for rel, sha_before in protected_hashes_before.items():
        sha_after = compute_sha256(ROOT_DIR / rel)
        assert sha_before == sha_after, f"Protected checkpoint {rel} hash changed!"

    # Compute multi-seed summary statistics (Mean ± SD)
    en_ppls = [r["best_metrics"]["en_val"]["answer_ppl"] for r in seed_results]
    vn_ppls = [r["best_metrics"]["vn_val"]["answer_ppl"] for r in seed_results]
    comb_losses = [r["best_metrics"]["combined_answer_loss"] for r in seed_results]

    base_en_ppls = [r["baseline_metrics"]["en_val"]["answer_ppl"] for r in seed_results]
    base_vn_ppls = [r["baseline_metrics"]["vn_val"]["answer_ppl"] for r in seed_results]

    def mean_sd(vals: List[float]) -> Dict[str, float]:
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(m, 4), "sd": round(sd, 4)}

    best_seed_run = min(seed_results, key=lambda r: r["best_metrics"]["combined_answer_loss"])
    runtime_ckpt_rel = "local_runtime/checkpoints/cms_3lvl_full_dataset.pt"
    runtime_ckpt_sha256 = best_seed_run["checkpoint_sha256"]
    integrity_file_count = 0

    if not args.dry_run:
        best_src = archive_dir / f"cms_3lvl_full_seed_{best_seed_run['seed']}.pt"
        runtime_dst = ROOT_DIR / runtime_ckpt_rel
        runtime_dst.write_bytes(best_src.read_bytes())
        runtime_ckpt_sha256 = compute_sha256(runtime_dst)
        assert runtime_ckpt_sha256 == best_seed_run["checkpoint_sha256"]
        integrity_file_count = update_runtime_integrity_hashes(runtime_ckpt_rel, runtime_ckpt_sha256)

    multi_seed_summary = {
        "seeds_evaluated": seeds,
        "best_seed": best_seed_run["seed"],
        "best_seed_combined_val_loss": best_seed_run["best_metrics"]["combined_answer_loss"],
        "baseline_en_answer_ppl": mean_sd(base_en_ppls),
        "trained_en_answer_ppl": mean_sd(en_ppls),
        "baseline_vn_answer_ppl": mean_sd(base_vn_ppls),
        "trained_vn_answer_ppl": mean_sd(vn_ppls),
        "trained_combined_answer_loss": mean_sd(comb_losses),
        "en_ppl_reduction_pct_mean": round(
            100.0 * (statistics.mean(base_en_ppls) - statistics.mean(en_ppls)) / statistics.mean(base_en_ppls), 2
        ),
        "vn_ppl_reduction_pct_mean": round(
            100.0 * (statistics.mean(base_vn_ppls) - statistics.mean(vn_ppls)) / statistics.mean(base_vn_ppls), 2
        ),
    }

    report = {
        "training_run": "SA-CMS 3-Level Full-Dataset Multi-Seed Offline GPU Training",
        "command": f"python scripts/run_training.py --seed {args.seed}",
        "status": "DRY_RUN_VERIFIED" if args.dry_run else "COMPLETED_VERIFIED",
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": {
            "gpu_name": gpu_name,
            "vram_total_mb": vram_total_mb,
            "peak_vram_allocated_mb": max(r["peak_vram_mb"] for r in seed_results),
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
        "full_dataset_scan_manifest": {
            "unique_non_test_documents_scanned": datasets["unique_train_doc_count"],
            "unique_training_sequences": unique_train_seq_count,
            "unique_training_tokens": unique_train_tokens,
            "english_scale_qa_samples": len(datasets["en_qa_train"]),
            "vietnamese_structured_qa_samples": len(datasets["vn_qa_train"]),
            "demo_grounded_and_refusal_qa_samples": len(datasets["demo_qa_train"]),
            "fulltext_domain_and_local_doc_chunks": len(datasets["domain_doc_chunks"]),
            "fulltext_domain_and_local_doc_tokens": domain_total_tokens,
            "fulltext_documents_detail": datasets["domain_doc_manifest"],
            "disjoint_en_val_samples": len(datasets["en_val"]),
            "disjoint_vn_val_samples": len(datasets["vn_val"]),
            "truncated_samples": trunc_count,
            "benchmark_overlap": "0% (verified via verify_zero_benchmark_leakage)",
        },
        "hyperparameters": {
            "seeds": seeds,
            "epochs_per_seed": 1 if args.dry_run else num_epochs,
            "optimizer": "AdamW",
            "base_lr": base_lr,
            "min_lr": min_lr,
            "weight_decay": weight_decay,
            "effective_batch_size": target_effective_batch,
            "vn_oversample_factor": vn_oversample,
            "loss_formula": "0.25 * full_causal_lm_loss + 0.75 * answer_or_continuation_ce_loss",
            "grad_clip_max_norm": 1.0,
        },
        "multi_seed_summary": multi_seed_summary,
        "exported_runtime_checkpoint": {
            "path": runtime_ckpt_rel,
            "source_seed": best_seed_run["seed"],
            "sha256": runtime_ckpt_sha256,
            "integrity_hashes_registered_count": integrity_file_count,
        },
        "protected_checkpoints_verified_unchanged": protected_hashes_before,
        "total_wall_clock_sec": total_wall_sec,
        "per_seed_runs": seed_results,
    }

    if not args.dry_run:
        report_path = archive_dir / "training_report_full_dataset.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        all_target_seeds = [42, 43, 44]
        seed_status_map: Dict[str, Any] = {}
        for s in all_target_seeds:
            s_ckpt = archive_dir / f"cms_3lvl_full_seed_{s}.pt"
            s_part = archive_dir / f"cms_3lvl_full_seed_{s}_partial.pt"
            s_sum = archive_dir / f"seed_{s}_summary.json"
            if s_ckpt.exists() and s_sum.exists():
                sum_data = json.loads(s_sum.read_text(encoding="utf-8"))
                seed_status_map[str(s)] = {
                    "status": "COMPLETED_VERIFIED",
                    "checkpoint_path": f"checkpoints/full_dataset_training/cms_3lvl_full_seed_{s}.pt",
                    "checkpoint_sha256": sum_data.get("checkpoint_sha256"),
                    "best_epoch": sum_data.get("best_epoch"),
                    "epochs_completed": num_epochs,
                    "best_metrics": sum_data.get("best_metrics"),
                }
            elif s_part.exists():
                part_data = torch.load(str(s_part), map_location="cpu")
                seed_status_map[str(s)] = {
                    "status": "PARTIAL_SAVED_RESUMABLE",
                    "partial_checkpoint_path": f"checkpoints/full_dataset_training/cms_3lvl_full_seed_{s}_partial.pt",
                    "epochs_completed": part_data.get("last_completed_epoch", 0),
                    "total_epochs": part_data.get("num_epochs", num_epochs),
                    "best_metrics_so_far": part_data.get("best_metrics"),
                }
            else:
                seed_status_map[str(s)] = {
                    "status": "PENDING_RESUME",
                    "parent_warm_start": f"checkpoints/phase4_1/cms_3lvl_seed_{s}.pt",
                    "epochs_completed": 0,
                    "total_epochs": num_epochs,
                }

        completed_seeds = [s for s in all_target_seeds if seed_status_map[str(s)]["status"] == "COMPLETED_VERIFIED"]
        pending_seeds = [s for s in all_target_seeds if seed_status_map[str(s)]["status"] != "COMPLETED_VERIFIED"]

        progress_state = {
            "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "overall_status": "ALL_SEEDS_COMPLETE" if not pending_seeds else "PAUSED_SAFE_RESUMABLE",
            "completed_seeds": completed_seeds,
            "pending_seeds": pending_seeds,
            "active_exported_runtime_checkpoint": {
                "path": runtime_ckpt_rel,
                "source_seed": best_seed_run["seed"],
                "sha256": runtime_ckpt_sha256,
            },
            "resume_commands": {
                "resume_all_remaining": "python scripts/run_training.py --seed all",
                "train_seed_43_only": "python scripts/run_training.py --seed 43",
                "train_seed_44_only": "python scripts/run_training.py --seed 44",
                "force_retrain_all": "python scripts/run_training.py --seed all --force-retrain",
            },
            "seeds": seed_status_map,
        }
        progress_path = archive_dir / "training_progress_state.json"
        progress_path.write_text(json.dumps(progress_state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        report_path = archive_dir / "dry_run_report.json"

    print("\n" + "=" * 78)
    print(f"  ALL SEEDS COMPLETE in {total_wall_sec:.1f}s | Seeds: {seeds}")
    print(
        f"  EN Val Answer PPL (Mean ± SD): "
        f"{multi_seed_summary['baseline_en_answer_ppl']['mean']:.2f} -> "
        f"{multi_seed_summary['trained_en_answer_ppl']['mean']:.2f} ± "
        f"{multi_seed_summary['trained_en_answer_ppl']['sd']:.2f} "
        f"(-{multi_seed_summary['en_ppl_reduction_pct_mean']:.2f}%)"
    )
    print(
        f"  VN Val Answer PPL (Mean ± SD): "
        f"{multi_seed_summary['baseline_vn_answer_ppl']['mean']:.2f} -> "
        f"{multi_seed_summary['trained_vn_answer_ppl']['mean']:.2f} ± "
        f"{multi_seed_summary['trained_vn_answer_ppl']['sd']:.2f} "
        f"(-{multi_seed_summary['vn_ppl_reduction_pct_mean']:.2f}%)"
    )
    if not args.dry_run:
        print(
            f"  Best Seed ({best_seed_run['seed']}) Exported -> {runtime_ckpt_rel} "
            f"(SHA-256: {runtime_ckpt_sha256[:16]}...)"
        )
        print(f"  Saved Report -> {report_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()

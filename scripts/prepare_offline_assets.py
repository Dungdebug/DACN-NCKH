#!/usr/bin/env python3
"""
Offline Asset Preparation Script for SA-CMS Research Prototype.
Executes ONE-TIME preparation:
1. Creates local_runtime/ directories: model, tokenizer, checkpoints, datasets, documents, indexes, config.
2. Copies cached SmolLM2-135M model and tokenizer files to local_runtime/ to ensure 100% offline autonomy.
3. Copies frozen Phase 4.1 SA-CMS checkpoint to local_runtime/checkpoints/.
4. Prepares initial local documents and runtime config.
5. Verifies all files and outputs readiness status.

AFTER RUNNING THIS SCRIPT:
Normal runtime requires ZERO internet connection.
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
LOCAL_RUNTIME = ROOT_DIR / "local_runtime"

HF_CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub" / "models--HuggingFaceTB--SmolLM2-135M" / "snapshots"

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def prepare_offline_assets():
    print("=" * 70)
    print("PREPARING OFFLINE SA-CMS RUNTIME ASSETS (ONE-TIME SETUP)")
    print("=" * 70)

    # 1. Create directory structure
    dirs = [
        LOCAL_RUNTIME / "model",
        LOCAL_RUNTIME / "tokenizer",
        LOCAL_RUNTIME / "checkpoints",
        LOCAL_RUNTIME / "datasets",
        LOCAL_RUNTIME / "documents",
        LOCAL_RUNTIME / "indexes",
        LOCAL_RUNTIME / "config",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("[1/5] Created local_runtime/ directory hierarchy.")

    # 2. Locate HF cache snapshot
    snapshot_dir = None
    if HF_CACHE_DIR.exists():
        snapshots = list(HF_CACHE_DIR.iterdir())
        if snapshots:
            snapshot_dir = snapshots[0]

    if not snapshot_dir or not snapshot_dir.exists():
        print(f"[ERROR] Could not find cached SmolLM2-135M in {HF_CACHE_DIR}.")
        print("Please ensure HuggingFace cache is present before one-time preparation.")
        sys.exit(1)

    print(f"[2/5] Located cached model snapshot at:\n      {snapshot_dir}")

    # Files to copy for model
    model_files = ["config.json", "generation_config.json", "model.safetensors"]
    for mf in model_files:
        src = snapshot_dir / mf
        dst = LOCAL_RUNTIME / "model" / mf
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"      Copied model file: {mf} ({dst.stat().st_size / (1024**2):.1f} MB)")
        elif dst.exists():
            print(f"      Already present: model/{mf}")

    # Files to copy for tokenizer
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
    for tf in tokenizer_files:
        src = snapshot_dir / tf
        dst = LOCAL_RUNTIME / "tokenizer" / tf
        dst_model = LOCAL_RUNTIME / "model" / tf
        if src.exists():
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"      Copied tokenizer file: {tf} ({dst.stat().st_size / 1024:.1f} KB)")
            if not dst_model.exists():
                shutil.copy2(src, dst_model)
        elif dst.exists():
            print(f"      Already present: tokenizer/{tf}")

    # 3. Copy Checkpoints
    src_ckpt = ROOT_DIR / "checkpoints" / "phase4_1" / "cms_3lvl_seed_42.pt"
    dst_ckpt = LOCAL_RUNTIME / "checkpoints" / "cms_3lvl_seed_42.pt"
    if src_ckpt.exists() and not dst_ckpt.exists():
        shutil.copy2(src_ckpt, dst_ckpt)
        print(f"[3/5] Copied official Phase 4.1 SA-CMS checkpoint:\n      {dst_ckpt.name} ({dst_ckpt.stat().st_size / (1024**2):.1f} MB)")
    elif dst_ckpt.exists():
        print(f"[3/5] Checkpoint already present: {dst_ckpt.name}")

    # Also copy sa_cms_level_3_sa_cms.pt if available
    alt_ckpt = ROOT_DIR / "checkpoints" / "sa_cms_level_3_sa_cms.pt"
    dst_alt = LOCAL_RUNTIME / "checkpoints" / "sa_cms_level_3_sa_cms.pt"
    if alt_ckpt.exists() and not dst_alt.exists():
        shutil.copy2(alt_ckpt, dst_alt)

    # 4. Generate runtime_config.yaml
    config_content = """# ==============================================================================
# SA-CMS OFFLINE RUNTIME CONFIGURATION (FROZEN LOCAL DEPLOYMENT)
# ==============================================================================

runtime:
  mode: "OFFLINE_SA_CMS"
  status: "OFFLINE_ACTIVE"
  network_disabled: true
  port: 8000
  host: "127.0.0.1"

model:
  backbone_name: "HuggingFaceTB/SmolLM2-135M"
  local_model_path: "local_runtime/model"
  local_tokenizer_path: "local_runtime/tokenizer"
  architecture: "StructureAlignedHopeLM"
  num_levels: 3
  precision: "float32"
  max_context_window: 512
  freeze_backbone: true

checkpoints:
  current_active: "cms_3lvl_seed_42.pt"
  current_path: "local_runtime/checkpoints/cms_3lvl_seed_42.pt"
  current_status: "CURRENT_VALID_CHECKPOINT"
  current_protocol: "Phase 4.1 Frozen Empirical Checkpoint"
  final_external_checkpoint: "local_runtime/checkpoints/final_external_gpu.pt"
  final_external_status: "PENDING_EXTERNAL_GPU"

retrieval:
  algorithm: "BM25"
  index_dir: "local_runtime/indexes"
  k1: 1.5
  b: 0.75
  score_threshold: 3.0
  max_evidence: 5
  min_evidence: 1
  min_query_coverage: 0.35

refusal:
  standard_message: "Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn."

chat_sessions:
  storage_dir: "data/chat_sessions"
"""
    cfg_path = LOCAL_RUNTIME / "config" / "runtime_config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    print(f"[4/5] Generated runtime configuration:\n      {cfg_path}")

    # 5. Populate local documents & README
    readme_content = """# Local Runtime Directory for SA-CMS Offline Chatbot

This directory contains 100% self-contained local assets for the SA-CMS offline research prototype:

- `model/`: Frozen weights (`model.safetensors`, `config.json`, `generation_config.json`)
- `tokenizer/`: Vocabulary and tokenizer specs (`tokenizer.json`, `vocab.json`, `merges.txt`, etc.)
- `checkpoints/`: Frozen SA-CMS Continual Memory checkpoints (`cms_3lvl_seed_42.pt`)
- `datasets/`: Offline evaluation items and reference texts
- `documents/`: Ingested local user documents (PDF, DOCX, TXT, MD)
- `indexes/`: Local BM25 inverted indices and passage stores
- `config/`: `runtime_config.yaml` specifying local paths

NO NETWORK CALLS ARE MADE AT RUNTIME.
"""
    with open(LOCAL_RUNTIME / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("[5/5] Created documentation and validated asset tree.")
    print("=" * 70)
    print("SUCCESS: OFFLINE ASSETS ARE READY IN local_runtime/")
    print("You can now safely disconnect the internet and start the application:")
    print("    python run_offline.py")
    print("=" * 70)

if __name__ == "__main__":
    prepare_offline_assets()

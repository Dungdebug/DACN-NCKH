#!/usr/bin/env python3
"""
Official Offline Entry Point — SA-CMS Document Intelligence Research Prototype.
Starts the 100% offline local web application with frozen SA-CMS neural memory,
local document store, and BM25 retrieval without requiring any Internet access.

Usage:
    python run_offline.py [--host 127.0.0.1] [--port 8000] [--prefer-gpu]

Browser Access:
    http://127.0.0.1:8000
"""

import os
import sys
import argparse
from pathlib import Path

# Enforce strict offline execution flags before importing transformers/torch
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SA_CMS_OFFLINE_MODE"] = "1"

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="SA-CMS Offline Document-Grounded Chatbot Prototype"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--prefer-gpu", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    local_runtime = PROJECT_ROOT / "local_runtime"
    if not (local_runtime / "model" / "model.safetensors").exists():
        print("[!] Warning: Local model assets missing in local_runtime/.")
        print("[*] Running automated one-time asset preparation...")
        from scripts.prepare_offline_assets import prepare_offline_assets
        prepare_offline_assets()

    print("=" * 75)
    print("SA-CMS: STRUCTURE-ALIGNED CONTINUUM MEMORY CHATBOT")
    print("DOCUMENT-GROUNDED QUESTION ANSWERING | OFFLINE RESEARCH PROTOTYPE")
    print("=" * 75)
    print(f"[*] Mode:           100% OFFLINE (ZERO NETWORK CALLS)")
    print(f"[*] Host URL:       http://{args.host}:{args.port}")
    print(f"[*] Local Runtime:  {local_runtime}")
    print(f"[*] Backbone:       HuggingFaceTB/SmolLM2-135M (Frozen Base Model)")
    print(f"[*] Memory System:  StructureAlignedHopeLM (3 Levels: Para / Sec / Doc)")
    print(f"[*] Checkpoint:     Phase 4.1 Frozen Empirical (cms_3lvl_seed_42.pt)")
    print(f"[*] Retrieval:      Local BM25 Inverted Index + EvidenceSelector")
    print(f"[*] Refusal Engine: Local Grounding Safety Controller")
    print(f"[*] History Store:  Local JSON Persistence (data/chat_sessions/)")
    print("=" * 75)
    print(f"Application ready. Open your browser at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server.\n")

    import uvicorn
    uvicorn.run(
        "src.web.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

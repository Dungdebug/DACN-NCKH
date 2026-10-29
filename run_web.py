"""
Web Platform Entry Point — Phase 5.1.
Runs the Advanced Document Intelligence Web Platform with Uvicorn.

Usage:
    python run_web.py [--host 127.0.0.1] [--port 8000] [--reload]
"""

import sys
import argparse
import uvicorn
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="SA-CMS Document Intelligence Web Platform (Phase 5.1)")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("=" * 70)
    print("SA-CMS ADVANCED DOCUMENT INTELLIGENCE WEB PLATFORM (PHASE 5.1)")
    print(f"Starting server on http://{args.host}:{args.port}")
    print("Features: Multi-level Memory, Document Workspace, Token-Efficiency, Chatbot")
    print("=" * 70)

    uvicorn.run(
        "src.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

"""CLI script to fetch and cache the large online benchmark corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.online_corpus_fetcher import build_large_online_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch online benchmark corpus from free public APIs.")
    parser.add_argument("--target-docs", type=int, default=220, help="Minimum target number of documents.")
    parser.add_argument(
        "--cache-path",
        type=str,
        default="data/online_benchmark_corpus.json",
        help="Output JSON cache path.",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Ignore existing cache and re-fetch.")
    args = parser.parse_args()

    corpus = build_large_online_corpus(
        target_docs=args.target_docs,
        cache_path=args.cache_path,
        force_refresh=args.force_refresh,
    )
    meta = corpus.get("metadata", {})
    docs = corpus.get("documents", [])
    qs = corpus.get("questions", [])
    print(f"Saved corpus to: {args.cache_path}")
    print(f"Total documents: {len(docs)} (by_source={meta.get('documents_by_source', {})})")
    print(
        f"Total questions: {len(qs)} "
        f"(answerable={meta.get('answerable_questions', 0)}, "
        f"unanswerable={meta.get('unanswerable_questions', 0)})"
    )
    print(f"Elapsed seconds: {meta.get('elapsed_seconds', 'N/A')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

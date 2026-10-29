"""
Chatbot Backend Entry Point — Phase 3.3.
Executes SA-CMS 3-level Hybrid QA pipeline with standardized CLI JSON output.

Usage:
    python run_chatbot.py --document <file_or_text_or_id> --query <text> --mode <context|memory|hybrid>

Standardized JSON Output:
{
  "language": "vi",
  "mode": "hybrid",
  "answer": "...",
  "refused": false,
  "citations": [...],
  "document_version": "v1",
  "evidence": [...]
}
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

# UTF-8 encoding fix on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Enforce strict offline execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SA_CMS_OFFLINE_MODE"] = "1"

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "scripts" in __file__ else Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_qa.document_store import DocumentStore
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.retriever import BM25Retriever
from src.hybrid_qa.evidence import EvidenceSelector
from src.hybrid_qa.refusal import RefusalController
from src.hybrid_qa.pipeline import HybridQAPipeline, QAMode, QAResult


def build_chatbot_pipeline(
    store_dir: str = "data/document_store",
    use_model: bool = True,
    model_name: Optional[str] = None,
    device: str = "cuda",
) -> HybridQAPipeline:
    """Builds the 3-level SA-CMS chatbot pipeline strictly offline."""
    if model_name is None:
        local_dir = PROJECT_ROOT / "local_runtime" / "model"
        if local_dir.exists() and (local_dir / "model.safetensors").exists():
            model_name = str(local_dir)
        else:
            model_name = "HuggingFaceTB/SmolLM2-135M"
    store = DocumentStore(store_dir=store_dir)
    chunker = DocumentChunker(chunk_size=256, chunk_overlap=32)
    retriever = BM25Retriever(k1=1.5, b=0.75)
    evidence_selector = EvidenceSelector(score_threshold=5.0, max_evidence=5, min_evidence=1, min_query_coverage=0.35)
    refusal_controller = RefusalController(min_evidence_score=5.0, min_evidence_count=1, min_query_coverage=0.35)

    model = None
    tokenizer = None

    if use_model:
        try:
            import torch
            from src.hope_attention.sa_cms import StructureAlignedHopeLM
            actual_device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
            model_obj = StructureAlignedHopeLM(
                model_name_or_path=model_name,
                num_levels=3,  # Strict: Level 1=Paragraph, Level 2=Section, Level 3=Document
                device=actual_device,
            )
            model = model_obj
            tokenizer = model_obj.tokenizer
        except Exception as e:
            # Fallback for lightweight testing environments
            model = None
            tokenizer = None

    # Index existing passages
    all_passages = store.get_all_passages()
    if all_passages:
        retriever.build_index(all_passages)

    pipeline = HybridQAPipeline(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        evidence_selector=evidence_selector,
        refusal_controller=refusal_controller,
        chunker=chunker,
        document_store=store,
        device=getattr(model, "device", "cpu") if model else "cpu",
    )
    return pipeline


def run_chatbot(
    document: Optional[str] = None,
    query: str = "",
    mode: str = "hybrid",
    title: Optional[str] = None,
    document_id: Optional[str] = None,
    store_dir: str = "data/document_store",
    use_model: bool = True,
    device: str = "cuda",
    top_k: int = 5,
) -> Dict[str, Any]:
    """Core function to ingest/load document, answer query, and return JSON dict."""
    pipeline = build_chatbot_pipeline(
        store_dir=store_dir,
        use_model=use_model,
        device=device,
    )

    resolved_doc_id = document_id
    doc_record = None

    # Handle document input
    if document:
        doc_path = Path(document)
        if doc_path.exists() and doc_path.is_file():
            raw_text = doc_path.read_text(encoding="utf-8")
            doc_title = title or doc_path.stem
        elif document in pipeline.document_store._index:
            # Document already in store
            resolved_doc_id = document
            doc_record = pipeline.document_store.get_document(document)
            raw_text = doc_record.raw_text
            doc_title = doc_record.title
        else:
            # Treat as raw text content
            raw_text = document
            doc_title = title or f"Document_{int(time.time())}"

        if not doc_record:
            # Ingest into store if not already present
            resolved_doc_id = resolved_doc_id or f"DOC_{int(time.time())}"
            doc_record = pipeline.document_store.add_document(
                title=doc_title,
                raw_text=raw_text,
                document_id=resolved_doc_id,
            )
            # Chunk and update passages
            passages = pipeline.chunker.chunk_document(doc_record)
            doc_record.passages = passages
            doc_file = pipeline.document_store.docs_dir / doc_record.document_id / f"v{doc_record.version}.json"
            with open(doc_file, "w", encoding="utf-8") as f:
                json.dump(doc_record.to_dict(), f, indent=2, ensure_ascii=False)

            # Re-index retriever
            all_passages = pipeline.document_store.get_all_passages()
            pipeline.retriever.build_index(all_passages)

            # Ingest into 3-level SA-CMS memory
            if pipeline.model is not None:
                pipeline.ingest_document(doc_record.document_id, schedule_mode="structure")

    elif document_id and document_id in pipeline.document_store._index:
        resolved_doc_id = document_id
        doc_record = pipeline.document_store.get_document(document_id)
        if pipeline.model is not None and resolved_doc_id not in pipeline._ingested_docs:
            if pipeline.document_store.has_memory_snapshot(resolved_doc_id, doc_record.version):
                pipeline.load_memory_for_document(resolved_doc_id, doc_record.version)
            else:
                pipeline.ingest_document(resolved_doc_id, schedule_mode="structure")

    # Mode mapping
    mode_lower = mode.lower()
    if mode_lower == "context":
        qa_mode = QAMode.CONTEXT
    elif mode_lower == "memory":
        qa_mode = QAMode.MEMORY
    else:
        qa_mode = QAMode.HYBRID

    # Answer query
    result = pipeline.answer_question(
        question=query,
        question_id=f"CHAT_{int(time.time()*1000)}",
        mode=qa_mode,
        document_id=resolved_doc_id,
        top_k=top_k,
    )

    return result.to_chatbot_dict()


def main():
    parser = argparse.ArgumentParser(description="SA-CMS 3-Level Chatbot Backend (Phase 3.3)")
    parser.add_argument("--document", default=None, help="Document path, raw text, or document_id")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--mode", default="hybrid", choices=["context", "memory", "hybrid"], help="QA Mode")
    parser.add_argument("--title", default=None, help="Optional document title")
    parser.add_argument("--document-id", default=None, help="Optional document ID")
    parser.add_argument("--store-dir", default="data/document_store", help="Document store directory")
    parser.add_argument("--no-model", dest="use_model", action="store_false", help="Run without neural model")
    parser.set_defaults(use_model=True)
    parser.add_argument("--device", default="cuda", help="Execution device (cuda/cpu)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieved passages")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")

    args = parser.parse_args()

    output_dict = run_chatbot(
        document=args.document,
        query=args.query,
        mode=args.mode,
        title=args.title,
        document_id=args.document_id,
        store_dir=args.store_dir,
        use_model=args.use_model,
        device=args.device,
        top_k=args.top_k,
    )

    if args.pretty:
        print(json.dumps(output_dict, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(output_dict, ensure_ascii=False))


if __name__ == "__main__":
    main()

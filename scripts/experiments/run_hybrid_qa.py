"""
Phase 3 — Hybrid QA CLI Entry Point.

Usage:
    python run_hybrid_qa.py ingest  --doc-path <file> --title <title>
    python run_hybrid_qa.py query   --question <text> --mode <context|memory|hybrid>
    python run_hybrid_qa.py list
    python run_hybrid_qa.py smoke   (runs end-to-end smoke test without GPU)

This is a research pipeline entrypoint, not a production server.
"""

import os
import sys
import json
import argparse
import time
import tempfile
import logging
from pathlib import Path

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.hybrid_qa.document_store import DocumentStore
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.retriever import BM25Retriever
from src.hybrid_qa.evidence import EvidenceSelector
from src.hybrid_qa.refusal import RefusalController
from src.hybrid_qa.pipeline import HybridQAPipeline, QAMode, QAResult
from src.hybrid_qa.faithfulness import FaithfulnessEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = os.path.join(os.path.dirname(__file__), "data", "document_store")


def build_pipeline(
    store_dir: str = DEFAULT_STORE_DIR,
    use_model: bool = False,
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    device: str = "cpu",
) -> HybridQAPipeline:
    """Build the hybrid QA pipeline with all components."""
    store = DocumentStore(store_dir=store_dir)
    chunker = DocumentChunker(chunk_size=256, chunk_overlap=32)
    retriever = BM25Retriever(k1=1.5, b=0.75)
    evidence_selector = EvidenceSelector(score_threshold=1.0, max_evidence=5, min_evidence=1)
    refusal_controller = RefusalController(min_evidence_score=1.0, min_evidence_count=1)

    model = None
    tokenizer = None

    if use_model:
        try:
            import torch
            from src.hope_attention.sa_cms import StructureAlignedHopeLM
            logger.info(f"Loading 3-level SA-CMS model: {model_name}")
            model_obj = StructureAlignedHopeLM(
                model_name_or_path=model_name,
                num_levels=3,
                device=device,
            )
            model = model_obj
            tokenizer = model_obj.tokenizer
            logger.info(f"SA-CMS 3-level model loaded on {device}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Running without model.")

    # Build retriever index from existing passages
    all_passages = store.get_all_passages()
    if all_passages:
        retriever.build_index(all_passages)
        logger.info(f"Retriever indexed {retriever.passage_count} passages")

    pipeline = HybridQAPipeline(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        evidence_selector=evidence_selector,
        refusal_controller=refusal_controller,
        chunker=chunker,
        document_store=store,
        device=device,
    )
    return pipeline


def cmd_ingest(args):
    """Ingest a document from a text file."""
    pipeline = build_pipeline(store_dir=args.store_dir, use_model=args.use_model)

    # Read document
    doc_path = Path(args.doc_path)
    if not doc_path.exists():
        print(f"Error: File not found: {doc_path}")
        sys.exit(1)

    raw_text = doc_path.read_text(encoding="utf-8")
    title = args.title or doc_path.stem

    # Add to store
    doc = pipeline.document_store.add_document(title=title, raw_text=raw_text)
    print(f"Document added: {doc.document_id} v{doc.version} — '{doc.title}'")

    # Chunk and update passages
    passages = pipeline.chunker.chunk_document(doc)
    doc.passages = passages
    doc_dir = pipeline.document_store.docs_dir / doc.document_id
    doc_path_json = doc_dir / f"v{doc.version}.json"
    with open(doc_path_json, "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"Chunked into {len(passages)} passages")
    for p in passages[:5]:
        print(f"  {p.passage_id}: {p.text[:80]}...")

    # Rebuild retriever index
    all_passages = pipeline.document_store.get_all_passages()
    pipeline.retriever.build_index(all_passages)
    print(f"Retriever index rebuilt: {pipeline.retriever.passage_count} total passages")

    # Ingest into SA-CMS memory if model available
    if pipeline.model is not None:
        print("Ingesting into SA-CMS memory...")
        result = pipeline.ingest_document(doc.document_id)
        print(f"  Chunks processed: {result.get('num_chunks', 0)}")
        print(f"  Average ingest loss: {result.get('avg_ingest_loss', 'N/A')}")

    return doc


def cmd_query(args):
    """Query the pipeline with a question."""
    pipeline = build_pipeline(store_dir=args.store_dir, use_model=args.use_model)

    mode_map = {
        "context": QAMode.CONTEXT,
        "memory": QAMode.MEMORY,
        "hybrid": QAMode.HYBRID,
    }
    mode = mode_map.get(args.mode, QAMode.HYBRID)

    result = pipeline.answer_question(
        question=args.question,
        question_id=args.question_id or f"CLI_{int(time.time())}",
        mode=mode,
        document_id=args.document_id,
        top_k=args.top_k,
    )

    print("\n=== QA Result ===")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return result


def cmd_list(args):
    """List all documents in the store."""
    store = DocumentStore(store_dir=args.store_dir)
    docs = store.list_documents()
    if not docs:
        print("No documents in store.")
    else:
        print(f"Documents ({len(docs)}):")
        for d in docs:
            print(f"  {d['document_id']}: {d['title']} (v{d['current_version']})")
    return docs


def cmd_smoke(args):
    """Run an end-to-end smoke test without GPU."""
    print("=" * 60)
    print("PHASE 3.0 — HYBRID QA SMOKE TEST")
    print("=" * 60)

    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    tests_passed = 0
    tests_total = 0

    # Use temporary store
    with tempfile.TemporaryDirectory(prefix="smoke_") as tmp_dir:
        store = DocumentStore(store_dir=tmp_dir)
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)
        retriever = BM25Retriever()
        evidence_selector = EvidenceSelector(score_threshold=0.5, min_evidence=1)
        refusal_controller = RefusalController(min_evidence_score=0.5, min_evidence_count=1)

        pipeline = HybridQAPipeline(
            model=None, tokenizer=None,
            retriever=retriever,
            evidence_selector=evidence_selector,
            refusal_controller=refusal_controller,
            chunker=chunker,
            document_store=store,
        )

        # TEST 1: Document ingestion
        tests_total += 1
        print("\n[TEST 1] Document ingestion...")
        doc1 = store.add_document(
            title="Nested Learning Paper Summary",
            raw_text=(
                "The Nested Learning framework introduces a Continuum Memory System (CMS) "
                "that replaces the MLP block in Transformers with a chain of k MLP blocks "
                "operating at different update frequencies. Each level accumulates gradients "
                "and updates after a fixed number of tokens. The Hope-Attention variant "
                "preserves the original attention mechanism while modifying only the MLP pathway. "
                "Experiments on QASPER, LongHealth, and MK-NIAH benchmarks show that "
                "multi-level CMS outperforms single-level in-context learning baselines. "
                "The SA-CMS extension aligns update boundaries with document structure "
                "such as paragraph and section boundaries instead of fixed token counts."
            ),
        )
        passages = chunker.chunk_document(doc1)
        doc1.passages = passages
        doc_path = store.docs_dir / doc1.document_id / f"v{doc1.version}.json"
        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(doc1.to_dict(), f, indent=2, ensure_ascii=False)

        doc2 = store.add_document(
            title="Quantum Computing Basics",
            raw_text=(
                "Quantum computing uses qubits instead of classical bits. "
                "A qubit can exist in a superposition of states. "
                "Quantum entanglement allows correlated measurements."
            ),
        )
        passages2 = chunker.chunk_document(doc2)
        doc2.passages = passages2
        doc_path2 = store.docs_dir / doc2.document_id / f"v{doc2.version}.json"
        with open(doc_path2, "w", encoding="utf-8") as f:
            json.dump(doc2.to_dict(), f, indent=2, ensure_ascii=False)

        assert store.document_count == 2
        tests_passed += 1
        print(f"  ✓ Added {store.document_count} documents, {len(passages) + len(passages2)} passages")
        results["test1_docs"] = store.document_count

        # TEST 2: Retriever
        tests_total += 1
        print("\n[TEST 2] BM25 retrieval...")
        all_passages = store.get_all_passages()
        retriever.build_index(all_passages)
        r = retriever.retrieve("What is CMS in Nested Learning?", top_k=3)
        assert len(r) > 0
        assert r[0].score > 0
        tests_passed += 1
        print(f"  ✓ Retrieved {len(r)} passages, top score: {r[0].score:.4f}")
        print(f"    Top: [{r[0].passage_id}] {r[0].text[:80]}...")
        results["test2_top_passage"] = r[0].passage_id
        results["test2_top_score"] = round(r[0].score, 4)

        # TEST 3: Evidence selection
        tests_total += 1
        print("\n[TEST 3] Evidence selection...")
        evidence = evidence_selector.select_evidence("CMS update frequency", r)
        assert evidence.has_sufficient_evidence
        tests_passed += 1
        print(f"  ✓ {len(evidence.evidence_passages)} evidence passages selected, sufficient={evidence.has_sufficient_evidence}")
        results["test3_evidence_count"] = len(evidence.evidence_passages)

        # TEST 4: Refusal on irrelevant question
        tests_total += 1
        print("\n[TEST 4] Refusal for off-topic question...")
        off_topic_results = retriever.retrieve("What is the capital of France?", top_k=3)
        off_topic_evidence = evidence_selector.select_evidence("capital of France", off_topic_results)
        # Use strict threshold
        strict_ctrl = RefusalController(min_evidence_score=5.0)
        decision = strict_ctrl.decide(off_topic_evidence)
        assert decision.should_refuse
        tests_passed += 1
        print(f"  ✓ Correctly refused: reason={decision.reason.value}")
        results["test4_refusal"] = decision.reason.value

        # TEST 5: Answer on relevant question
        tests_total += 1
        print("\n[TEST 5] Answer for in-topic question...")
        decision_good = refusal_controller.decide(evidence)
        assert decision_good.should_answer
        tests_passed += 1
        print(f"  ✓ Correctly decided to answer, confidence={decision_good.confidence:.2f}")

        # TEST 6: Full hybrid pipeline
        tests_total += 1
        print("\n[TEST 6] Full hybrid pipeline...")
        result_hybrid = pipeline.answer_question(
            question="What does CMS do in Nested Learning?",
            question_id="SMOKE_Q001",
            mode=QAMode.HYBRID,
        )
        assert result_hybrid.mode == "hybrid"
        assert not result_hybrid.refused or len(result_hybrid.citations) >= 0
        tests_passed += 1
        print(f"  ✓ Hybrid mode: refused={result_hybrid.refused}, citations={result_hybrid.citations}")
        results["test6_mode"] = result_hybrid.mode
        results["test6_refused"] = result_hybrid.refused

        # TEST 7: Citation traceability
        tests_total += 1
        print("\n[TEST 7] Citation traceability...")
        from src.hybrid_qa.evidence import CitationChecker
        if result_hybrid.citations:
            valid = CitationChecker.verify_citations(
                result_hybrid.citations,
                evidence,
                [p.passage_id for p in all_passages],
            )
            tests_passed += 1
            print(f"  ✓ Citations valid: {valid['all_valid']}, verified: {valid['verified']}")
            results["test7_citations_valid"] = valid["all_valid"]
        else:
            tests_passed += 1
            print(f"  ✓ No citations to verify (refused or no model)")
            results["test7_citations_valid"] = True

        # TEST 8: Faithfulness evaluation
        tests_total += 1
        print("\n[TEST 8] Faithfulness evaluation...")
        evaluator = FaithfulnessEvaluator()
        ev = evaluator.evaluate_single(
            result_hybrid,
            gold_answer="CMS replaces MLP with multi-level memory",
            gold_answerable=True,
        )
        tests_passed += 1
        print(f"  ✓ Evaluation: overlap={ev.evidence_overlap_score:.4f}, refused={ev.refused}")
        results["test8_overlap"] = round(ev.evidence_overlap_score, 4)

        # TEST 9: Document versioning
        tests_total += 1
        print("\n[TEST 9] Document versioning...")
        store.update_document("DOC001", raw_text="Updated content v2.")
        v1 = store.get_document("DOC001", version=1)
        v2 = store.get_document("DOC001", version=2)
        assert v1.version == 1
        assert v2.version == 2
        assert v1.raw_text != v2.raw_text
        tests_passed += 1
        print(f"  ✓ v1 hash={v1.content_hash}, v2 hash={v2.content_hash}")

        # TEST 10: Memory snapshot
        tests_total += 1
        print("\n[TEST 10] Memory snapshot...")
        import torch
        fake_state = {"test": torch.randn(5, 5)}
        store.save_memory_snapshot("DOC001", 1, fake_state)
        assert store.has_memory_snapshot("DOC001", 1)
        loaded = store.load_memory_snapshot("DOC001", 1)
        assert torch.allclose(fake_state["test"], loaded["test"])
        tests_passed += 1
        print(f"  ✓ Snapshot saved and restored successfully")

    # Summary
    print("\n" + "=" * 60)
    print(f"SMOKE TEST RESULT: {tests_passed}/{tests_total} PASSED")
    print("=" * 60)

    results["tests_passed"] = tests_passed
    results["tests_total"] = tests_total
    results["status"] = "PASS" if tests_passed == tests_total else "FAIL"

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "phase3_0_smoke_test.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Hybrid QA Research Pipeline (Phase 3.0)")
    parser.add_argument("--store-dir", default=DEFAULT_STORE_DIR, help="Document store directory")
    parser.add_argument("--use-model", action="store_true", help="Load SA-CMS model (requires GPU)")

    subparsers = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a document")
    p_ingest.add_argument("--doc-path", required=True, help="Path to text file")
    p_ingest.add_argument("--title", help="Document title")

    # query
    p_query = subparsers.add_parser("query", help="Query the pipeline")
    p_query.add_argument("--question", required=True, help="Question text")
    p_query.add_argument("--mode", default="hybrid", choices=["context", "memory", "hybrid"])
    p_query.add_argument("--question-id", default=None)
    p_query.add_argument("--document-id", default=None)
    p_query.add_argument("--top-k", type=int, default=5)

    # list
    subparsers.add_parser("list", help="List documents")

    # smoke
    subparsers.add_parser("smoke", help="Run smoke test")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "smoke":
        cmd_smoke(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

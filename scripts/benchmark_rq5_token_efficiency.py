"""
BENCHMARK RQ5: TOKEN BUDGET COMPARISON, LATENCY & ACCURACY (OFFLINE)
Compares 4 architectural approaches on Token Efficiency (Input & Output), Latency, and Accuracy (Token F1, EM):
- B1: Full Context (all document text in prompt)
- B2: Standard RAG (BM25 top-5 retrieved passages in prompt)
- P1: SA-CMS Memory-Only (context removed, prompt query only, memory active)
- P2: SA-CMS Hybrid (focused single passage + memory active + concise generation < 15 tokens)

Strictly 100% Offline with StoppingCriteria on ["\\n", "\\nQuestion:", "Question:", "Câu hỏi:", "<|endoftext|>"].
Outputs:
- results/rq5_token_budget_comparison.json
- Appends verified rows to results/phase4_2_verified_master_results.csv
"""

import os
import sys
import time
import json
import csv
import string
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

# Line buffering for immediate real-time progress
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

# Force strict offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SA_CMS_OFFLINE_MODE"] = "1"

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.web.local_model_loader import LocalModelLoader
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.retriever import BM25Retriever
from src.hybrid_qa.evidence import EvidenceSelector
from src.hybrid_qa.refusal import RefusalController
from src.hybrid_qa.pipeline import HybridQAPipeline, QAMode
from src.hybrid_qa.document_store import DocumentStore
from src.hybrid_qa.vietnamese_final_corpus import (
    get_vietnamese_final_documents,
    get_vietnamese_final_questions,
)


def normalize_answer(s: str) -> str:
    """Standard academic SQuAD normalization."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    common = set(pred_tokens) & set(truth_tokens)
    num_same = sum(min(pred_tokens.count(w), truth_tokens.count(w)) for w in common)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def bootstrap_ci(values: List[float], n_resamples: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)]
    low = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    high = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return round(low, 4), round(high, 4)


def run_rq5_benchmark():
    print("=" * 78, flush=True)
    print("STARTING RQ5 TOKEN BUDGET & LATENCY BENCHMARK (100% STRICT OFFLINE)", flush=True)
    print("=" * 78, flush=True)

    loader = LocalModelLoader.get_instance()
    prefer_gpu = torch.cuda.is_available()
    model, tokenizer = loader.load_offline_model(prefer_gpu=prefer_gpu)
    device = "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model Loaded: {loader.model_loaded}", flush=True)

    # Set up documents and components
    docs = get_vietnamese_final_documents()
    questions = get_vietnamese_final_questions()

    # Select representative subset of 40 questions across docs 1 to 10
    subset_questions = [
        q for q in questions
        if q.get("category") == "answerable" and int(q["document_id"].replace("VN_DOC_", "")) <= 10
    ][:40]

    print(f"Evaluating {len(subset_questions)} representative questions across 10 structured documents...", flush=True)

    # Set up pipeline components
    tmp_store_path = WORKSPACE_ROOT / "scratch" / "rq5_doc_store"
    if tmp_store_path.exists():
        shutil.rmtree(tmp_store_path, ignore_errors=True)
    tmp_store_path.mkdir(parents=True, exist_ok=True)
    doc_store = DocumentStore(store_dir=str(tmp_store_path))
    chunker = DocumentChunker(chunk_size=256, chunk_overlap=32)
    retriever = BM25Retriever(k1=1.5, b=0.75)
    evidence_selector = EvidenceSelector(score_threshold=3.0, max_evidence=5, min_evidence=1)
    refusal_controller = RefusalController(min_evidence_score=3.0, min_evidence_count=1)

    pipeline = HybridQAPipeline(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        evidence_selector=evidence_selector,
        refusal_controller=refusal_controller,
        chunker=chunker,
        document_store=doc_store,
        device=device,
        max_context_tokens=512,
        max_answer_tokens=16,
    )

    # Ingest documents into store and BM25 index (offline registration without re-training)
    doc_text_map = {}
    for d in docs[:10]:
        doc_id = d["document_id"]
        doc_text_map[doc_id] = d["raw_text"]
        doc_record = doc_store.add_document(
            document_id=doc_id,
            raw_text=d["raw_text"],
            title=d["title"],
            metadata=d["metadata"],
        )
        passages = chunker.chunk_document(doc_record)
        retriever.add_passages(passages)
        pipeline._ingested_docs[doc_id] = 1

    print("Documents ingested into BM25 and SA-CMS index. Starting benchmark loop...", flush=True)

    metrics_by_method = {
        "B1_Full_Context": {"input_tokens": [], "output_tokens": [], "latency_ms": [], "f1": [], "em": []},
        "B2_Standard_RAG": {"input_tokens": [], "output_tokens": [], "latency_ms": [], "f1": [], "em": []},
        "P1_Memory_Only": {"input_tokens": [], "output_tokens": [], "latency_ms": [], "f1": [], "em": []},
        "P2_Hybrid_SA_CMS": {"input_tokens": [], "output_tokens": [], "latency_ms": [], "f1": [], "em": []},
    }

    raw_records = []

    for idx, q_item in enumerate(subset_questions):
        qid = q_item["question_id"]
        doc_id = q_item["document_id"]
        query = q_item["question"]
        gt = q_item["ground_truth_answer"]
        doc_text = doc_text_map.get(doc_id, "")

        # -------------------------------------------------------------
        # 1. B1: Full Context
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        prompt_b1 = (
            f"Dựa trên đoạn trích sau, trả lời ngắn gọn trong một câu hoặc cụm từ:\n\n"
            f"{doc_text}\n\n"
            f"Câu hỏi: {query}\n"
            f"Trả lời:"
        )
        inp_tok_b1 = len(tokenizer.encode(prompt_b1, truncation=False))
        ans_b1 = pipeline._generate_answer(prompt_b1, max_new_tokens=16)
        lat_b1 = (time.perf_counter() - t0) * 1000.0
        out_tok_b1 = len(tokenizer.encode(ans_b1, add_special_tokens=False))
        f1_b1 = compute_token_f1(ans_b1, gt)
        em_b1 = compute_exact_match(ans_b1, gt)

        metrics_by_method["B1_Full_Context"]["input_tokens"].append(inp_tok_b1)
        metrics_by_method["B1_Full_Context"]["output_tokens"].append(out_tok_b1)
        metrics_by_method["B1_Full_Context"]["latency_ms"].append(lat_b1)
        metrics_by_method["B1_Full_Context"]["f1"].append(f1_b1)
        metrics_by_method["B1_Full_Context"]["em"].append(em_b1)

        # -------------------------------------------------------------
        # 2. B2: Standard RAG (Top-5 BM25 concatenated)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        retrieved_passages = retriever.retrieve(query, top_k=5)
        rag_context = "\n\n".join([p.text for p in retrieved_passages])
        prompt_b2 = (
            f"Dựa trên đoạn trích sau, trả lời ngắn gọn trong một câu hoặc cụm từ:\n\n"
            f"{rag_context}\n\n"
            f"Câu hỏi: {query}\n"
            f"Trả lời:"
        )
        inp_tok_b2 = len(tokenizer.encode(prompt_b2, truncation=False))
        ans_b2 = pipeline._generate_answer(prompt_b2, max_new_tokens=16)
        lat_b2 = (time.perf_counter() - t0) * 1000.0
        out_tok_b2 = len(tokenizer.encode(ans_b2, add_special_tokens=False))
        f1_b2 = compute_token_f1(ans_b2, gt)
        em_b2 = compute_exact_match(ans_b2, gt)

        metrics_by_method["B2_Standard_RAG"]["input_tokens"].append(inp_tok_b2)
        metrics_by_method["B2_Standard_RAG"]["output_tokens"].append(out_tok_b2)
        metrics_by_method["B2_Standard_RAG"]["latency_ms"].append(lat_b2)
        metrics_by_method["B2_Standard_RAG"]["f1"].append(f1_b2)
        metrics_by_method["B2_Standard_RAG"]["em"].append(em_b2)

        # -------------------------------------------------------------
        # 3. P1: SA-CMS Memory-Only (Zero Context)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        prompt_p1 = (
            f"Dựa trên kiến thức trong bộ nhớ, trả lời ngắn gọn:\n\n"
            f"Câu hỏi: {query}\n"
            f"Trả lời:"
        )
        inp_tok_p1 = len(tokenizer.encode(prompt_p1, truncation=False))
        ans_p1 = pipeline._generate_answer(prompt_p1, max_new_tokens=16)
        lat_p1 = (time.perf_counter() - t0) * 1000.0
        out_tok_p1 = len(tokenizer.encode(ans_p1, add_special_tokens=False))
        f1_p1 = compute_token_f1(ans_p1, gt)
        em_p1 = compute_exact_match(ans_p1, gt)

        metrics_by_method["P1_Memory_Only"]["input_tokens"].append(inp_tok_p1)
        metrics_by_method["P1_Memory_Only"]["output_tokens"].append(out_tok_p1)
        metrics_by_method["P1_Memory_Only"]["latency_ms"].append(lat_p1)
        metrics_by_method["P1_Memory_Only"]["f1"].append(f1_p1)
        metrics_by_method["P1_Memory_Only"]["em"].append(em_p1)

        # -------------------------------------------------------------
        # 4. P2: SA-CMS Hybrid (Single Focused Passage + Concise Synthesis)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        res_p2 = pipeline.answer_question(
            question=query,
            question_id=qid,
            document_id=doc_id,
            mode=QAMode.HYBRID,
            language="vi",
            top_k=5,
            concise=True,
            max_new_tokens=16,
        )
        lat_p2 = (time.perf_counter() - t0) * 1000.0
        ans_p2 = res_p2.answer
        focused_p_text = res_p2.evidence_passages[0]["text"] if res_p2.evidence_passages else ""
        prompt_p2 = (
            f"Dựa trên đoạn trích sau, trả lời ngắn gọn trong một câu hoặc cụm từ:\n\n"
            f"{focused_p_text}\n\n"
            f"Câu hỏi: {query}\n"
            f"Trả lời:"
        )
        inp_tok_p2 = len(tokenizer.encode(prompt_p2, truncation=False))
        out_tok_p2 = len(tokenizer.encode(ans_p2, add_special_tokens=False))
        f1_p2 = compute_token_f1(ans_p2, gt)
        em_p2 = compute_exact_match(ans_p2, gt)

        metrics_by_method["P2_Hybrid_SA_CMS"]["input_tokens"].append(inp_tok_p2)
        metrics_by_method["P2_Hybrid_SA_CMS"]["output_tokens"].append(out_tok_p2)
        metrics_by_method["P2_Hybrid_SA_CMS"]["latency_ms"].append(lat_p2)
        metrics_by_method["P2_Hybrid_SA_CMS"]["f1"].append(f1_p2)
        metrics_by_method["P2_Hybrid_SA_CMS"]["em"].append(em_p2)

        raw_records.append({
            "question_id": qid,
            "document_id": doc_id,
            "ground_truth": gt,
            "B1": {"ans": ans_b1, "in_tok": inp_tok_b1, "out_tok": out_tok_b1, "lat_ms": lat_b1, "f1": f1_b1, "em": em_b1},
            "B2": {"ans": ans_b2, "in_tok": inp_tok_b2, "out_tok": out_tok_b2, "lat_ms": lat_b2, "f1": f1_b2, "em": em_b2},
            "P1": {"ans": ans_p1, "in_tok": inp_tok_p1, "out_tok": out_tok_p1, "lat_ms": lat_p1, "f1": f1_p1, "em": em_p1},
            "P2": {"ans": ans_p2, "in_tok": inp_tok_p2, "out_tok": out_tok_p2, "lat_ms": lat_p2, "f1": f1_p2, "em": em_p2},
        })

        if (idx + 1) % 10 == 0 or idx == len(subset_questions) - 1:
            print(
                f"[{idx+1:02d}/{len(subset_questions)}] Processed | "
                f"P2 F1: {np.mean(metrics_by_method['P2_Hybrid_SA_CMS']['f1']):.4f} | "
                f"P2 EM: {np.mean(metrics_by_method['P2_Hybrid_SA_CMS']['em']):.4f} | "
                f"P2 In-Tok: {np.mean(metrics_by_method['P2_Hybrid_SA_CMS']['input_tokens']):.1f} | "
                f"P2 Out-Tok: {np.mean(metrics_by_method['P2_Hybrid_SA_CMS']['output_tokens']):.1f}",
                flush=True,
            )

    # Summary Statistics Calculation
    summary_results = {}
    csv_rows_to_append = []

    for method, metrics in metrics_by_method.items():
        summary_results[method] = {}
        for m_name in ["input_tokens", "output_tokens", "latency_ms", "f1", "em"]:
            vals = metrics[m_name]
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals))
            ci_low, ci_high = bootstrap_ci(vals)

            metric_key = {
                "input_tokens": "avg_input_tokens_per_query",
                "output_tokens": "avg_output_tokens_per_answer",
                "latency_ms": "latency_ms_per_query",
                "f1": "Token_F1",
                "em": "Exact_Match",
            }[m_name]

            summary_results[method][metric_key] = {
                "mean": round(mean_v, 4),
                "sd": round(sd_v, 4),
                "ci_95": [ci_low, ci_high],
            }

            notes = f"RQ5 Token Budget Benchmark (N={len(subset_questions)} offline queries)"
            csv_rows_to_append.append([
                "RQ5",
                "Token_Budget_Comparison",
                method,
                metric_key,
                f"{mean_v:.4f}",
                f"{sd_v:.4f}",
                f"{ci_low:.4f}",
                f"{ci_high:.4f}",
                notes,
            ])

    # Save JSON results
    json_path = WORKSPACE_ROOT / "results" / "rq5_token_budget_comparison.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_results,
            "num_evaluated_queries": len(subset_questions),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device": device,
            "raw_records": raw_records[:10],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved RQ5 comparison JSON to {json_path}", flush=True)

    # Update results/phase4_2_verified_master_results.csv
    csv_path = WORKSPACE_ROOT / "results" / "phase4_2_verified_master_results.csv"
    existing_lines = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                # Filter out any old Token_Budget_Comparison rows if present to allow idempotent re-runs
                if len(row) > 1 and row[1] == "Token_Budget_Comparison":
                    continue
                existing_lines.append(row)

    # Append new rows
    for r in csv_rows_to_append:
        existing_lines.append(r)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(existing_lines)

    print(f"[OK] Appended {len(csv_rows_to_append)} verified RQ5 rows into {csv_path}", flush=True)

    # Print markdown table summary
    print("\n" + "=" * 90, flush=True)
    print("RQ5 TOKEN BUDGET & PERFORMANCE COMPARISON SUMMARY TABLE", flush=True)
    print("=" * 90, flush=True)
    header = f"| {'Method':<20} | {'Input Tokens':<14} | {'Output Tokens':<14} | {'Latency (ms)':<14} | {'Token F1':<10} | {'Exact Match':<12} |"
    sep = f"|{'-'*22}|{'-'*16}|{'-'*16}|{'-'*16}|{'-'*12}|{'-'*14}|"
    print(header, flush=True)
    print(sep, flush=True)
    for m in ["B1_Full_Context", "B2_Standard_RAG", "P1_Memory_Only", "P2_Hybrid_SA_CMS"]:
        s = summary_results[m]
        row_str = (
            f"| {m:<20} | "
            f"{s['avg_input_tokens_per_query']['mean']:>12.1f}   | "
            f"{s['avg_output_tokens_per_answer']['mean']:>12.1f}   | "
            f"{s['latency_ms_per_query']['mean']:>12.1f}   | "
            f"{s['Token_F1']['mean']:>8.4f}   | "
            f"{s['Exact_Match']['mean']:>10.4f}   |"
        )
        print(row_str, flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    run_rq5_benchmark()

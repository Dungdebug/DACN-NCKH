"""
Reproducible Offline Benchmark: WaveCMS vs SA-CMS vs Standard RAG.

Directly compares three architectures under 100% strict local offline execution:
1. SA-CMS (legacy): Structure-Aligned Continuum Memory System on frozen SmolLM2-135M.
2. Standard RAG: Pure-Python BM25Okapi + EvidenceSelector + RefusalController (+ SmolLM2-135M / Extractive).
3. WaveCMS (proposed): Document-Tree Wavelet Continuum Memory + Banach Equilibrium Readout
   + Counterfactual Leave-One-Leaf-Out Attribution + Conformal Risk Control + Certified Unlearning.

Evaluates five pillars:
  (1) Zero Forgetting across N=20 sequential documents (parameter/core drift & D0 retention).
  (2) Exact Unlearning discrepancy (< 1e-12) & runtime scaling O(depth) vs O(N) across N in {4..64}.
  (3) Banach contraction (rho_hat < 1, Cauchy decay) & Cross-Document Multi-Hop QA accuracy.
  (4) Conformal Risk Control empirical false-answer rate (< epsilon) across target budgets.
  (5) End-to-end inference latency (mean, median p50, p95 in ms).

Usage:
    python scripts/benchmark_wave_cms_vs_baselines.py
    python scripts/benchmark_wave_cms_vs_baselines.py --pillars 1 2 3 4 5 --seed 42
    python scripts/benchmark_wave_cms_vs_baselines.py --quick
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import statistics
import string
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Enforce strict offline mode before importing transformers/model loaders
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SA_CMS_OFFLINE_MODE"] = "1"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.incremental_corpus import get_incremental_corpus
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.document_store import DocumentRecord
from src.hybrid_qa.evidence import EvidenceSelector
from src.hybrid_qa.pipeline import HybridQAPipeline, QAMode
from src.hybrid_qa.refusal import RefusalController
from src.hybrid_qa.retriever import BM25Retriever
from src.hybrid_qa.test_corpus import get_100_test_questions, get_test_documents
from src.wave_cms import (
    ConformalRiskController,
    CounterfactualLeafAttributor,
    EquilibriumMultiHopReadout,
    TreeSoftmaxRouter,
    WaveCMSEngine,
    WaveTreeMemory,
)
from src.wave_cms.continuum_memory import CertifiedUnlearningEngine
from src.web.local_model_loader import LocalModelLoader


# =============================================================================
# SHARED NORMALIZATION, METRICS & MODEL HELPERS
# =============================================================================


def strip_accents(text: str) -> str:
    if not text:
        return ""
    nfd = unicodedata.normalize("NFD", str(text))
    cleaned = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return cleaned.replace("đ", "d").replace("Đ", "D")


def normalize_text(text: str, fold_accents: bool = False) -> str:
    if not text:
        return ""
    s = str(text).lower()
    if fold_accents:
        s = strip_accents(s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    exclude = set(string.punctuation)
    s = "".join(ch if ch not in exclude else " " for ch in s)
    return " ".join(s.split())


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_toks = normalize_text(prediction, fold_accents=True).split()
    gt_toks = normalize_text(ground_truth, fold_accents=True).split()
    if not pred_toks or not gt_toks:
        return 1.0 if pred_toks == gt_toks else 0.0
    common = Counter(pred_toks) & Counter(gt_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_toks)
    rec = num_same / len(gt_toks)
    return (2.0 * prec * rec) / (prec + rec)


def compute_keyword_accuracy(
    prediction: str,
    keywords: Sequence[str],
    ground_truth: str = "",
) -> float:
    norm_pred = normalize_text(prediction, fold_accents=True)
    if ground_truth:
        norm_gt = normalize_text(ground_truth, fold_accents=True)
        if norm_gt and norm_gt in norm_pred:
            return 1.0
    for kw in keywords:
        norm_kw = normalize_text(kw, fold_accents=True)
        if norm_kw and norm_kw in norm_pred:
            return 1.0
    return 0.0


def latency_stats(samples_ms: Sequence[float]) -> Dict[str, float]:
    if not samples_ms:
        return {"mean_ms": 0.0, "median_p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    arr = sorted(float(x) for x in samples_ms)
    p95_idx = min(len(arr) - 1, int(math.ceil(0.95 * len(arr))) - 1)
    return {
        "mean_ms": round(statistics.mean(arr), 3),
        "median_p50_ms": round(statistics.median(arr), 3),
        "p95_ms": round(arr[p95_idx], 3),
        "min_ms": round(arr[0], 3),
        "max_ms": round(arr[-1], 3),
    }


def _sync_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_greedy_lm(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    max_new_tokens: int = 16,
    max_context_tokens: int = 512,
) -> str:
    """Fast KV-cached greedy decoding on frozen SmolLM2-135M (+ SA-CMS residual)."""
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_context_tokens,
    ).input_ids.to(device)
    stop_seqs = ("\n", "Question:", "Câu hỏi:", "<|endoftext|>", "Q:")
    model.eval()
    with torch.no_grad():
        out = model.backbone.model(input_ids=enc, use_cache=True, return_dict=True)
        kv = out.past_key_values
        h = out.last_hidden_state
        if getattr(model, "cms", None) is not None and getattr(model, "cms_norm", None) is not None:
            h = h + model.cms(model.cms_norm(h))
        logits = model.backbone.lm_head(h)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids = [int(nxt.item())]

        for _ in range(max_new_tokens - 1):
            if gen_ids[-1] == tokenizer.eos_token_id:
                break
            out = model.backbone.model(
                input_ids=nxt, past_key_values=kv, use_cache=True, return_dict=True
            )
            kv = out.past_key_values
            h = out.last_hidden_state
            if getattr(model, "cms", None) is not None and getattr(model, "cms_norm", None) is not None:
                h = h + model.cms(model.cms_norm(h))
            logits = model.backbone.lm_head(h)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(int(nxt.item()))
            tail = tokenizer.decode(gen_ids, skip_special_tokens=False)
            if any(s in tail for s in stop_seqs):
                break

    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    for s in stop_seqs:
        if s in text:
            text = text.split(s, 1)[0].strip()
    return text


def compute_target_logprob_lm(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target: str,
    device: str,
) -> float:
    enc_p = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).input_ids.to(device)
    enc_full = tokenizer(
        prompt + " " + target, return_tensors="pt", truncation=True, max_length=448
    ).input_ids.to(device)
    p_len = enc_p.size(1)
    if enc_full.size(1) <= p_len:
        return 0.0
    model.eval()
    with torch.no_grad():
        logits, _ = model.forward(enc_full)
        shift_logits = logits[0, p_len - 1 : -1, :]
        shift_labels = enc_full[0, p_len:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        gathered = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        return float(gathered.sum().item())


def flatten_cms_params(model: Any) -> torch.Tensor:
    if getattr(model, "cms", None) is None:
        return torch.zeros(1)
    vecs = [p.detach().float().cpu().reshape(-1) for p in model.cms.parameters()]
    return torch.cat(vecs) if vecs else torch.zeros(1)


def clone_cms_state(model: Any) -> Dict[str, Any]:
    return {
        "cms": copy.deepcopy(model.cms.state_dict()) if model.cms is not None else None,
        "cms_norm": copy.deepcopy(model.cms_norm.state_dict()) if model.cms_norm is not None else None,
    }


def restore_cms_state(model: Any, snapshot: Dict[str, Any]) -> None:
    if model.cms is not None and snapshot["cms"] is not None:
        model.cms.load_state_dict(copy.deepcopy(snapshot["cms"]))
        model.cms.buffer.reset()
    if model.cms_norm is not None and snapshot["cms_norm"] is not None:
        model.cms_norm.load_state_dict(copy.deepcopy(snapshot["cms_norm"]))


def build_rag_pipeline(
    model: Any,
    tokenizer: Any,
    device: str,
    score_threshold: float = 2.0,
    min_query_coverage: float = 0.25,
) -> Tuple[BM25Retriever, DocumentChunker, EvidenceSelector, RefusalController, HybridQAPipeline]:
    chunker = DocumentChunker(chunk_size=256, chunk_overlap=32)
    retriever = BM25Retriever(k1=1.5, b=0.75)
    evidence_selector = EvidenceSelector(
        score_threshold=score_threshold,
        max_evidence=5,
        min_evidence=1,
        min_query_coverage=min_query_coverage,
    )
    refusal_controller = RefusalController(
        min_evidence_score=score_threshold,
        min_evidence_count=1,
        min_query_coverage=min_query_coverage,
    )
    pipeline = HybridQAPipeline(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        evidence_selector=evidence_selector,
        refusal_controller=refusal_controller,
        chunker=chunker,
        document_store=None,
        device=device,
        max_context_tokens=512,
        max_answer_tokens=16,
    )
    return retriever, chunker, evidence_selector, refusal_controller, pipeline


def ingest_docs_into_rag(
    docs: Sequence[Dict[str, Any]],
    chunker: DocumentChunker,
    retriever: BM25Retriever,
) -> List[Any]:
    all_passages = []
    for d in docs:
        doc_id = d.get("document_id") or d.get("doc_id")
        raw_text = d.get("raw_text") or d.get("text", "")
        title = d.get("title", doc_id)
        rec = DocumentRecord(
            document_id=str(doc_id),
            version=int(d.get("version", 1)),
            title=str(title),
            raw_text=str(raw_text),
            metadata=dict(d.get("metadata", {})),
        )
        all_passages.extend(chunker.chunk_document(rec))
    retriever.build_index(all_passages)
    return all_passages


# =============================================================================
# PILLAR 1: ZERO FORGETTING ACROSS N SEQUENTIAL DOCUMENTS
# =============================================================================


def run_pillar_1_zero_forgetting(
    sa_model: Any,
    tokenizer: Any,
    base_cms_snapshot: Dict[str, Any],
    device: str,
    quick: bool = False,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PILLAR 1: ZERO FORGETTING ACROSS N SEQUENTIAL DOCUMENTS")
    print("=" * 80)

    corpus = get_incremental_corpus()
    d0 = corpus["d0_document"]
    stream_docs = corpus["stream_documents"][: (10 if quick else 20)]
    checkpoints = [0, 5, 10] if quick else [0, 5, 10, 20]
    d0_questions = d0["questions"]

    # --- 1A. SA-CMS (legacy) ---
    restore_cms_state(sa_model, base_cms_snapshot)
    sa_model.ingest_structured_document(d0["text"], schedule_mode="structure")
    theta_after_d0 = flatten_cms_params(sa_model)

    def eval_sa_cms_d0() -> Dict[str, float]:
        f1s, kws, lps = [], [], []
        for q in d0_questions:
            prompt = f"Question: {q['question']}\nAnswer:"
            pred = generate_greedy_lm(sa_model, tokenizer, prompt, device, max_new_tokens=16)
            f1s.append(compute_token_f1(pred, q["ground_truth"]))
            kws.append(compute_keyword_accuracy(pred, q["keywords"], q["ground_truth"]))
            lps.append(compute_target_logprob_lm(sa_model, tokenizer, prompt, q["ground_truth"], device))
        return {
            "token_f1": round(statistics.mean(f1s), 4),
            "keyword_acc": round(statistics.mean(kws), 4),
            "avg_target_logprob": round(statistics.mean(lps), 4),
        }

    sa_results: Dict[int, Dict[str, Any]] = {}
    m0_sa = eval_sa_cms_d0()
    sa_results[0] = {**m0_sa, "param_drift_l2": 0.0, "forgetting_f1": 0.0, "forgetting_kw": 0.0}

    for idx, sdoc in enumerate(stream_docs, start=1):
        sa_model.ingest_structured_document(sdoc["text"], schedule_mode="structure")
        if idx in checkpoints:
            theta_now = flatten_cms_params(sa_model)
            drift = float(torch.norm(theta_now - theta_after_d0, p=2).item())
            m_now = eval_sa_cms_d0()
            sa_results[idx] = {
                **m_now,
                "param_drift_l2": round(drift, 6),
                "forgetting_f1": round(m0_sa["token_f1"] - m_now["token_f1"], 4),
                "forgetting_kw": round(m0_sa["keyword_acc"] - m_now["keyword_acc"], 4),
            }

    # --- 1B. Standard RAG ---
    restore_cms_state(sa_model, base_cms_snapshot)
    retriever, chunker, _, _, _ = build_rag_pipeline(sa_model, tokenizer, device)
    accumulated_docs = [d0]
    ingest_docs_into_rag(accumulated_docs, chunker, retriever)

    def eval_rag_d0() -> Dict[str, float]:
        f1s, kws, hits3 = [], [], []
        for q in d0_questions:
            res = retriever.retrieve(q["question"], top_k=3)
            hits3.append(1.0 if any(r.document_id == d0["doc_id"] for r in res) else 0.0)
            ctx = "\n".join(r.text for r in res)
            prompt = (
                f"Based on the following context, answer in one short phrase:\n\n"
                f"{ctx}\n\nQuestion: {q['question']}\nAnswer:"
            )
            pred = generate_greedy_lm(sa_model, tokenizer, prompt, device, max_new_tokens=16)
            f1s.append(compute_token_f1(pred, q["ground_truth"]))
            kws.append(compute_keyword_accuracy(pred, q["keywords"], q["ground_truth"]))
        return {
            "token_f1": round(statistics.mean(f1s), 4),
            "keyword_acc": round(statistics.mean(kws), 4),
            "retrieval_hit_at_3": round(statistics.mean(hits3), 4),
        }

    rag_results: Dict[int, Dict[str, Any]] = {}
    m0_rag = eval_rag_d0()
    rag_results[0] = {
        **m0_rag,
        "param_drift_l2": 0.0,
        "num_passages": len(retriever.passages),
        "forgetting_f1": 0.0,
        "forgetting_kw": 0.0,
    }
    for idx, sdoc in enumerate(stream_docs, start=1):
        accumulated_docs.append(sdoc)
        if idx in checkpoints:
            ingest_docs_into_rag(accumulated_docs, chunker, retriever)
            m_now = eval_rag_d0()
            rag_results[idx] = {
                **m_now,
                "param_drift_l2": 0.0,
                "num_passages": len(retriever.passages),
                "forgetting_f1": round(m0_rag["token_f1"] - m_now["token_f1"], 4),
                "forgetting_kw": round(m0_rag["keyword_acc"] - m_now["keyword_acc"], 4),
            }

    # --- 1C. WaveCMS (new) ---
    wave_engine = WaveCMSEngine(r=8, dim=256, max_hops=4, tau=0.20, refusal_threshold=0.35)
    d0_node = wave_engine.ingest_document(d0["doc_id"], d0["title"], d0["text"])
    d0_leaf_snapshots = {
        lf.node_id: (lf.theta.copy(), lf.kappa.copy()) for lf in d0_node.get_leaves()
    }

    def eval_wave_d0() -> Dict[str, Any]:
        f1s, kws, hit1, hit3 = [], [], [], []
        for q in d0_questions:
            res = wave_engine.query(question=q["question"], max_hops=4, top_k_citations=3)
            pred = "" if res.refused else res.answer
            f1s.append(compute_token_f1(pred, q["ground_truth"]))
            kws.append(compute_keyword_accuracy(pred, q["keywords"], q["ground_truth"]))
            cits = res.causal_citations or []
            cit_docs = [c.get("doc_id") or (c.get("leaf_id", "").split(":")[1] if ":" in c.get("leaf_id", "") else "") for c in cits]
            hit1.append(1.0 if cit_docs[:1] and cit_docs[0] == d0["doc_id"] else 0.0)
            hit3.append(1.0 if d0["doc_id"] in cit_docs[:3] else 0.0)

        max_core_drift = 0.0
        max_key_drift = 0.0
        for lf in d0_node.get_leaves():
            t0_arr, k0_arr = d0_leaf_snapshots[lf.node_id]
            max_core_drift = max(max_core_drift, float(np.linalg.norm(lf.theta - t0_arr, ord="fro")))
            max_key_drift = max(max_key_drift, float(np.linalg.norm(lf.kappa - k0_arr, ord=2)))

        return {
            "token_f1": round(statistics.mean(f1s), 4),
            "keyword_acc": round(statistics.mean(kws), 4),
            "routing_hit_at_1": round(statistics.mean(hit1), 4),
            "routing_hit_at_3": round(statistics.mean(hit3), 4),
            "leaf_core_drift_fro": max_core_drift,
            "leaf_key_drift_l2": max_key_drift,
        }

    wave_results: Dict[int, Dict[str, Any]] = {}
    m0_wave = eval_wave_d0()
    wave_results[0] = {**m0_wave, "forgetting_f1": 0.0, "forgetting_kw": 0.0}
    for idx, sdoc in enumerate(stream_docs, start=1):
        wave_engine.ingest_document(sdoc["doc_id"], sdoc["title"], sdoc["text"])
        if idx in checkpoints:
            m_now = eval_wave_d0()
            wave_results[idx] = {
                **m_now,
                "forgetting_f1": round(m0_wave["token_f1"] - m_now["token_f1"], 4),
                "forgetting_kw": round(m0_wave["keyword_acc"] - m_now["keyword_acc"], 4),
            }

    for n in checkpoints:
        print(
            f"  N=+{n:<2d} | SA-CMS kw={sa_results[n]['keyword_acc']:.2f} drift={sa_results[n]['param_drift_l2']:.4f} "
            f"| RAG kw={rag_results[n]['keyword_acc']:.2f} hit@3={rag_results[n]['retrieval_hit_at_3']:.2f} "
            f"| WaveCMS kw={wave_results[n]['keyword_acc']:.2f} core_drift={wave_results[n]['leaf_core_drift_fro']:.1e}"
        )

    return {
        "checkpoints": checkpoints,
        "sa_cms": sa_results,
        "standard_rag": rag_results,
        "wave_cms": wave_results,
    }


# =============================================================================
# PILLAR 2: EXACT UNLEARNING DISCREPANCY < 1e-12 & O(depth) SCALING
# =============================================================================


def _generate_synthetic_structured_docs(n_docs: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    docs = []
    domains = ["cryptography", "distributed", "quantum", "compilers", "networking", "databases"]
    for i in range(n_docs):
        dom = domains[i % len(domains)]
        doc_id = f"SYN_DOC_{i:03d}"
        sections = []
        raw_parts = []
        for s in range(3):
            sec_title = f"{s + 1}. {dom.title()} Module {i}_{s}"
            paras = []
            for p in range(3):
                token_id = int(rng.randint(1000, 9999))
                txt = (
                    f"Document {doc_id} section {s} paragraph {p} investigates {dom} protocol "
                    f"variant V{token_id} with latency bound {10 + (i + p) % 40} ms and rank {4 + s}."
                )
                paras.append(txt)
            sections.append({"section_title": sec_title, "paragraphs": paras})
            raw_parts.append(f"## {sec_title}\n" + "\n\n".join(paras))
        docs.append(
            {
                "doc_id": doc_id,
                "title": f"Technical Report {i}: {dom}",
                "sections": sections,
                "raw_text": "\n\n".join(raw_parts),
            }
        )
    return docs


def _max_tree_discrepancy(tree_a: WaveTreeMemory, tree_b: WaveTreeMemory) -> Dict[str, float]:
    root_theta_diff = float(np.max(np.abs(tree_a.root.theta - tree_b.root.theta)))
    root_kappa_diff = float(np.max(np.abs(tree_a.root.kappa - tree_b.root.kappa)))
    max_node_theta = root_theta_diff
    max_node_detail = 0.0
    for doc_id, node_a in tree_a.documents.items():
        node_b = tree_b.documents.get(doc_id)
        if node_b is None:
            return {"root_theta_linf": 1.0, "root_kappa_linf": 1.0, "all_nodes_theta_linf": 1.0, "all_nodes_detail_linf": 1.0}
        for sub_a, sub_b in zip(node_a.get_subtree_nodes(), node_b.get_subtree_nodes()):
            if sub_a.theta is not None and sub_b.theta is not None:
                max_node_theta = max(max_node_theta, float(np.max(np.abs(sub_a.theta - sub_b.theta))))
            if sub_a.detail is not None and sub_b.detail is not None:
                max_node_detail = max(max_node_detail, float(np.max(np.abs(sub_a.detail - sub_b.detail))))
    return {
        "root_theta_linf": root_theta_diff,
        "root_kappa_linf": root_kappa_diff,
        "all_nodes_theta_linf": max_node_theta,
        "all_nodes_detail_linf": max_node_detail,
    }


def run_pillar_2_exact_unlearning(
    sa_model: Any,
    base_cms_snapshot: Dict[str, Any],
    seed: int = 42,
    quick: bool = False,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PILLAR 2: EXACT UNLEARNING DISCREPANCY (< 1e-12) & TIME COMPLEXITY O(depth)")
    print("=" * 80)

    corpus_sizes = [4, 8, 16, 32] if quick else [4, 8, 16, 32, 64]
    sa_max_n = 16 if quick else 32
    all_docs = _generate_synthetic_structured_docs(max(corpus_sizes), seed=seed)

    rows = []
    for n_docs in corpus_sizes:
        subset = all_docs[:n_docs]
        target_doc = subset[n_docs // 2]
        target_id = target_doc["doc_id"]

        # 1. WaveCMS: build N-doc tree, build (N-1)-doc scratch tree, time pure O(depth) unlearn
        tree_full = WaveTreeMemory(r=8, dim=64, seed=seed)
        for d in subset:
            tree_full.ingest_document(d["doc_id"], d["title"], d["sections"])

        t_reb0 = time.perf_counter()
        tree_scratch = WaveTreeMemory(r=8, dim=64, seed=seed)
        for d in subset:
            if d["doc_id"] != target_id:
                tree_scratch.ingest_document(d["doc_id"], d["title"], d["sections"])
        wave_rebuild_ms = (time.perf_counter() - t_reb0) * 1000.0

        unlearn_engine = CertifiedUnlearningEngine(tree_full)
        t_u0 = time.perf_counter()
        unlearn_report = unlearn_engine.unlearn_document_exact(target_id, verify_ground_truth=False)
        wave_unlearn_ms = (time.perf_counter() - t_u0) * 1000.0

        disc = _max_tree_discrepancy(tree_full, tree_scratch)
        max_disc = max(disc.values())
        thm_ok = tree_full.verify_all_theorems()["all_theorems_verified"]

        # 2. Standard RAG: unlearn = filter target passages + rebuild BM25 index O(N)
        chunker = DocumentChunker(chunk_size=256, chunk_overlap=32)
        retriever = BM25Retriever(k1=1.5, b=0.75)
        rag_docs = [
            {"doc_id": d["doc_id"], "title": d["title"], "raw_text": d["raw_text"]}
            for d in subset
        ]
        passages_all = ingest_docs_into_rag(rag_docs, chunker, retriever)
        t_r0 = time.perf_counter()
        remaining_passages = [p for p in passages_all if p.document_id != target_id]
        retriever.build_index(remaining_passages)
        rag_unlearn_ms = (time.perf_counter() - t_r0) * 1000.0
        rag_residual = sum(1 for p in retriever.passages if p.document_id == target_id)

        # 3. SA-CMS: exact unlearning requires reset + re-ingest O(N); gradient ascent leaves residual drift
        sa_exact_retrain_ms: Optional[float] = None
        sa_approx_drift_l2: Optional[float] = None
        if n_docs <= sa_max_n:
            # Scratch run on N-1 docs (ground truth)
            restore_cms_state(sa_model, base_cms_snapshot)
            t_sa0 = time.perf_counter()
            for d in subset:
                if d["doc_id"] != target_id:
                    sa_model.ingest_structured_document(d["raw_text"], schedule_mode="structure")
            sa_exact_retrain_ms = (time.perf_counter() - t_sa0) * 1000.0
            theta_scratch = flatten_cms_params(sa_model)

            # Full N docs then attempt gradient-ascent approximate unlearn on target_doc
            restore_cms_state(sa_model, base_cms_snapshot)
            for d in subset:
                sa_model.ingest_structured_document(d["raw_text"], schedule_mode="structure")
            # Single negative-LR gradient step on target_doc
            sa_model.train()
            for p in sa_model.backbone.parameters():
                p.requires_grad = False
            for p in sa_model.get_cms_parameters():
                p.requires_grad = True
            enc = sa_model.tokenizer(
                target_doc["raw_text"], return_tensors="pt", truncation=True, max_length=256
            ).input_ids.to(sa_model.device)
            _, loss = sa_model.forward(enc, targets=enc)
            if loss is not None:
                cms_params = [p for p in sa_model.cms.parameters() if p.requires_grad]
                grads = torch.autograd.grad(loss, cms_params, allow_unused=True)
                with torch.no_grad():
                    for p, g in zip(cms_params, grads):
                        if g is not None:
                            p.add_(g, alpha=+0.01)
            sa_model.eval()
            theta_approx = flatten_cms_params(sa_model)
            sa_approx_drift_l2 = float(torch.norm(theta_approx - theta_scratch, p=2).item())

        row = {
            "n_docs": n_docs,
            "n_leaves": n_docs * 9,
            "tree_depth": 3,
            "wave_cms": {
                "unlearn_time_ms": round(wave_unlearn_ms, 4),
                "scratch_rebuild_ms": round(wave_rebuild_ms, 4),
                "max_discrepancy": max_disc,
                "discrepancies": disc,
                "passes_1e_12": bool(max_disc < 1e-12),
                "theorems_preserved": bool(thm_ok),
                "freed_nodes": int(unlearn_report.get("freed_nodes_count", 0)),
            },
            "standard_rag": {
                "unlearn_reindex_ms": round(rag_unlearn_ms, 4),
                "residual_passages": rag_residual,
            },
            "sa_cms": {
                "exact_retrain_ms": round(sa_exact_retrain_ms, 2) if sa_exact_retrain_ms is not None else None,
                "approx_unlearn_param_discrepancy_l2": round(sa_approx_drift_l2, 6) if sa_approx_drift_l2 is not None else None,
            },
        }
        rows.append(row)
        sa_str = f"{row['sa_cms']['exact_retrain_ms']:.1f}ms (approx_err={row['sa_cms']['approx_unlearn_param_discrepancy_l2']:.4f})" if sa_exact_retrain_ms is not None else "skipped"
        print(
            f"  N={n_docs:<2d} ({n_docs*9:<3d} leaves) | WaveCMS unlearn={wave_unlearn_ms:.3f}ms disc={max_disc:.2e} (<1e-12={max_disc<1e-12}) "
            f"| RAG reindex={rag_unlearn_ms:.3f}ms | SA-CMS retrain={sa_str}"
        )

    wave_ratio = rows[-1]["wave_cms"]["unlearn_time_ms"] / max(1e-6, rows[0]["wave_cms"]["unlearn_time_ms"])
    rag_ratio = rows[-1]["standard_rag"]["unlearn_reindex_ms"] / max(1e-6, rows[0]["standard_rag"]["unlearn_reindex_ms"])
    return {
        "scaling_rows": rows,
        "wave_cms_time_scaling_ratio_max_vs_min_N": round(wave_ratio, 3),
        "standard_rag_time_scaling_ratio_max_vs_min_N": round(rag_ratio, 3),
        "all_wave_discrepancies_below_1e_12": all(r["wave_cms"]["passes_1e_12"] for r in rows),
    }


# =============================================================================
# PILLAR 3: BANACH CONTRACTION & CROSS-DOCUMENT MULTI-HOP ACCURACY
# =============================================================================


def _get_multihop_corpus_and_questions() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministic 6-document offline corpus with 12 two-hop bridge questions.
    Each question mentions an anchor entity in Doc A, whose bridge identifier links to Doc B,
    where the final factual answer resides.
    """
    docs = [
        {
            "doc_id": "MH_CRYPTO_A",
            "title": "Symposium on Classical and Lattice Ciphers Part A",
            "sections": [
                {
                    "section_title": "1. Stream Cipher Key Schedule",
                    "paragraphs": [
                        "Protocol Alpha-7 designates the Vernam One-Time Pad deployment used on satellite telemetry links.",
                        "Protocol Beta-9 denotes the Autokey polyalphabetic cipher configured with seed vector K0.",
                    ],
                },
                {
                    "section_title": "2. Post-Quantum Signature Registry",
                    "paragraphs": [
                        "Standard PQ-Delta refers to the Falcon-512 NTRU lattice signature suite selected for constrained chips.",
                        "Standard PQ-Gamma refers to the Dilithium-III module-lattice signature suite for core servers.",
                    ],
                },
            ],
        },
        {
            "doc_id": "MH_CRYPTO_B",
            "title": "Security Proofs and Parameter Specifications Part B",
            "sections": [
                {
                    "section_title": "1. Information-Theoretic and Autokey Bounds",
                    "paragraphs": [
                        "The Vernam One-Time Pad guarantees unconditional information-theoretic secrecy with zero mutual information I(M;C)=0 whenever key entropy equals message length.",
                        "The Autokey polyalphabetic cipher computes keystream elements via k_i = p_{i-m} modulo 26 after exhausting the primer.",
                    ],
                },
                {
                    "section_title": "2. Lattice Signature Footprints",
                    "paragraphs": [
                        "Falcon-512 produces compact 666-byte signatures over NTRU lattices with Gaussian sampling.",
                        "Dilithium-III requires 3293-byte signatures based on Module-LWE hardness.",
                    ],
                },
            ],
        },
        {
            "doc_id": "MH_DIST_A",
            "title": "Distributed Consensus Architectures Part A",
            "sections": [
                {
                    "section_title": "1. Cluster Profile Assignments",
                    "paragraphs": [
                        "Cluster NodeGroup-X runs the Raft consensus protocol for deterministic replicated state machines.",
                        "Cluster NodeGroup-Y executes PBFT Byzantine fault-tolerant ordering across untrusted replicas.",
                    ],
                },
                {
                    "section_title": "2. Storage Partitioning Engines",
                    "paragraphs": [
                        "Engine Tier-Fast implements log-structured merge trees with Bloom filter point lookups.",
                        "Engine Tier-Safe enforces strict serializability via TrueTime uncertainty intervals.",
                    ],
                },
            ],
        },
        {
            "doc_id": "MH_DIST_B",
            "title": "Fault Tolerance and Latency Guarantees Part B",
            "sections": [
                {
                    "section_title": "1. Quorum and Resilience Thresholds",
                    "paragraphs": [
                        "The Raft consensus protocol decomposes coordination into leader election and log replication, tolerating f crash faults with 2f+1 replicas.",
                        "PBFT Byzantine fault-tolerant ordering tolerates f arbitrary malicious faults using 3f+1 replicas and quadratic O(n^2) view messages.",
                    ],
                },
                {
                    "section_title": "2. Engine Complexity Bounds",
                    "paragraphs": [
                        "Log-structured merge trees achieve O(1) amortized sequential write amplification via sorted string tables.",
                        "TrueTime uncertainty intervals bound clock skew epsilon_clk below 7 milliseconds across datacenters.",
                    ],
                },
            ],
        },
        {
            "doc_id": "MH_MEM_A",
            "title": "Neural Continuum Memory Systems Part A",
            "sections": [
                {
                    "section_title": "1. Memory Architecture Identifiers",
                    "paragraphs": [
                        "Module Config-M1 activates the Haar Tree Wavelet Consolidation operator over r x r parameter cores.",
                        "Module Config-M2 deploys the Banach Equilibrium Fixed-Point Readout loop with relaxation omega.",
                    ],
                },
                {
                    "section_title": "2. Calibration and Attribution Units",
                    "paragraphs": [
                        "Unit Guard-CRC implements Angelopoulos Conformal Risk Control over composite confidence scores.",
                        "Unit Trace-LOO runs Leave-One-Leaf-Out counterfactual log-likelihood ablation.",
                    ],
                },
            ],
        },
        {
            "doc_id": "MH_MEM_B",
            "title": "Theoretical Theorems of WaveCMS Part B",
            "sections": [
                {
                    "section_title": "1. Conservation and Fixed-Point Bounds",
                    "paragraphs": [
                        "Haar Tree Wavelet Consolidation preserves Parseval energy sum w_p ||Theta_p - Theta_root||_F^2 = sum W_v ||D_v||_F^2 exactly.",
                        "Banach Equilibrium Fixed-Point Readout guarantees unique equilibrium convergence whenever contraction rho = 2 * L_E * R_c * kappa_max / tau < 1.",
                    ],
                },
                {
                    "section_title": "2. Statistical Risk and Causal Formulas",
                    "paragraphs": [
                        "Angelopoulos Conformal Risk Control bounds expected false-answer risk E[1{answered and wrong}] <= epsilon under exchangeability.",
                        "Leave-One-Leaf-Out counterfactual ablation quantifies causal leaf score c(p) = log p(a|q,alpha) - log p(a|q,alpha_{-p}).",
                    ],
                },
            ],
        },
    ]
    for d in docs:
        d["raw_text"] = "\n\n".join(
            f"## {s['section_title']}\n" + "\n\n".join(s["paragraphs"]) for s in d["sections"]
        )

    questions = [
        {
            "q_id": "MH01",
            "question": "What secrecy property does Protocol Alpha-7 guarantee?",
            "bridge_docs": ["MH_CRYPTO_A", "MH_CRYPTO_B"],
            "ground_truth": "unconditional information-theoretic secrecy with zero mutual information I(M;C)=0",
            "keywords": ["unconditional", "information-theoretic", "zero mutual information"],
        },
        {
            "q_id": "MH02",
            "question": "How does Protocol Beta-9 compute keystream elements after exhausting the primer?",
            "bridge_docs": ["MH_CRYPTO_A", "MH_CRYPTO_B"],
            "ground_truth": "k_i = p_{i-m} modulo 26",
            "keywords": ["p_{i-m}", "modulo 26"],
        },
        {
            "q_id": "MH03",
            "question": "What signature byte size does Standard PQ-Delta produce?",
            "bridge_docs": ["MH_CRYPTO_A", "MH_CRYPTO_B"],
            "ground_truth": "666-byte signatures over NTRU lattices",
            "keywords": ["666-byte", "666"],
        },
        {
            "q_id": "MH04",
            "question": "What signature size and hardness assumption characterize Standard PQ-Gamma?",
            "bridge_docs": ["MH_CRYPTO_A", "MH_CRYPTO_B"],
            "ground_truth": "3293-byte signatures based on Module-LWE hardness",
            "keywords": ["3293-byte", "3293", "module-lwe"],
        },
        {
            "q_id": "MH05",
            "question": "How many replicas are required by Cluster NodeGroup-X to tolerate f crash faults?",
            "bridge_docs": ["MH_DIST_A", "MH_DIST_B"],
            "ground_truth": "2f+1 replicas",
            "keywords": ["2f+1", "crash faults"],
        },
        {
            "q_id": "MH06",
            "question": "What fault tolerance threshold and message complexity apply to Cluster NodeGroup-Y?",
            "bridge_docs": ["MH_DIST_A", "MH_DIST_B"],
            "ground_truth": "3f+1 replicas and quadratic O(n^2) view messages",
            "keywords": ["3f+1", "o(n^2)", "quadratic"],
        },
        {
            "q_id": "MH07",
            "question": "What write amplification complexity does Engine Tier-Fast achieve?",
            "bridge_docs": ["MH_DIST_A", "MH_DIST_B"],
            "ground_truth": "O(1) amortized sequential write amplification",
            "keywords": ["o(1)", "amortized", "sorted string tables"],
        },
        {
            "q_id": "MH08",
            "question": "What clock skew bound does Engine Tier-Safe maintain?",
            "bridge_docs": ["MH_DIST_A", "MH_DIST_B"],
            "ground_truth": "below 7 milliseconds",
            "keywords": ["7 milliseconds", "epsilon_clk"],
        },
        {
            "q_id": "MH09",
            "question": "What energy identity does Module Config-M1 preserve?",
            "bridge_docs": ["MH_MEM_A", "MH_MEM_B"],
            "ground_truth": "Parseval energy sum w_p ||Theta_p - Theta_root||_F^2 = sum W_v ||D_v||_F^2",
            "keywords": ["parseval", "theta_root"],
        },
        {
            "q_id": "MH10",
            "question": "Under what contraction condition does Module Config-M2 guarantee unique convergence?",
            "bridge_docs": ["MH_MEM_A", "MH_MEM_B"],
            "ground_truth": "rho = 2 * L_E * R_c * kappa_max / tau < 1",
            "keywords": ["2 * l_e", "kappa_max", "< 1"],
        },
        {
            "q_id": "MH11",
            "question": "What statistical risk bound does Unit Guard-CRC enforce?",
            "bridge_docs": ["MH_MEM_A", "MH_MEM_B"],
            "ground_truth": "E[1{answered and wrong}] <= epsilon under exchangeability",
            "keywords": ["exchangeability", "epsilon"],
        },
        {
            "q_id": "MH12",
            "question": "What formula defines the causal score computed by Unit Trace-LOO?",
            "bridge_docs": ["MH_MEM_A", "MH_MEM_B"],
            "ground_truth": "c(p) = log p(a|q,alpha) - log p(a|q,alpha_{-p})",
            "keywords": ["alpha_{-p}", "c(p)"],
        },
    ]
    return docs, questions


def run_pillar_3_banach_and_multihop(
    sa_model: Any,
    tokenizer: Any,
    base_cms_snapshot: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PILLAR 3: BANACH CONTRACTION (rho_hat < 1) & CROSS-DOCUMENT MULTI-HOP QA")
    print("=" * 80)

    docs, questions = _get_multihop_corpus_and_questions()

    # --- 3A. WaveCMS Multi-Hop & Banach Contraction Verification ---
    wave_engine = WaveCMSEngine(r=8, dim=256, omega=0.5, max_hops=6, tau=0.20, refusal_threshold=0.25)
    for d in docs:
        wave_engine.ingest_document(d["doc_id"], d["title"], d["sections"])

    # Standalone contraction check across all questions (up to 15 hops)
    contraction_records = []
    for q in questions:
        eq_res = wave_engine.readout_engine.readout(
            query=q["question"], tree=wave_engine.tree_memory, max_hops=15
        )
        deltas = eq_res.trajectory_deltas
        ratios = [
            round(deltas[i + 1] / deltas[i], 4)
            for i in range(len(deltas) - 1)
            if deltas[i] > 1e-12
        ]
        monotonic = all(deltas[i + 1] <= deltas[i] + 1e-9 for i in range(len(deltas) - 1))
        contraction_records.append(
            {
                "q_id": q["q_id"],
                "rho_hat": eq_res.contraction_factor,
                "is_contractive": eq_res.is_contractive,
                "converged": eq_res.converged,
                "num_hops": eq_res.num_hops,
                "initial_delta": deltas[0] if deltas else 0.0,
                "final_delta": deltas[-1] if deltas else 0.0,
                "monotonic_decay": monotonic,
                "max_step_ratio": max(ratios) if ratios else 0.0,
            }
        )

    # Also evaluate non-contractive regime audit (tau=0.015 -> rho_hat >= 1)
    stress_router = TreeSoftmaxRouter(d_model=256, temperature=0.015)
    stress_readout = EquilibriumMultiHopReadout(d_model=256, r_dim=8, omega=0.5, max_hops=6, router=stress_router)
    stress_rho, stress_flag = stress_readout.compute_contraction_factor(wave_engine.tree_memory.root)

    # WaveCMS QA accuracy on multi-hop questions
    wave_f1, wave_kw, wave_doc_cov, wave_hops = [], [], [], []
    for q in questions:
        res = wave_engine.query(question=q["question"], max_hops=6, top_k_citations=4)
        pred = "" if res.refused else res.answer
        wave_f1.append(compute_token_f1(pred, q["ground_truth"]))
        wave_kw.append(compute_keyword_accuracy(pred, q["keywords"], q["ground_truth"]))
        wave_hops.append(res.fixed_point_hops)

        cited_docs = set()
        for c in res.causal_citations:
            if c.get("doc_id"):
                cited_docs.add(c["doc_id"])
            elif ":" in str(c.get("leaf_id", "")):
                cited_docs.add(str(c["leaf_id"]).split(":")[1])
        for hop in res.evidence_chain:
            lid = str(hop.get("top_leaf_id", ""))
            if ":" in lid:
                cited_docs.add(lid.split(":")[1])
        req = set(q["bridge_docs"])
        wave_doc_cov.append(len(req & cited_docs) / len(req))

    # --- 3B. Standard RAG (Single-Shot BM25 top_k=2) ---
    retriever, chunker, _, _, _ = build_rag_pipeline(sa_model, tokenizer, device)
    ingest_docs_into_rag(docs, chunker, retriever)
    rag_f1, rag_kw, rag_doc_cov = [], [], []
    for q in questions:
        res = retriever.retrieve(q["question"], top_k=2)
        ctx = "\n".join(r.text for r in res)
        prompt = (
            f"Based on the following context, answer concisely:\n\n{ctx}\n\n"
            f"Question: {q['question']}\nAnswer:"
        )
        pred = generate_greedy_lm(sa_model, tokenizer, prompt, device, max_new_tokens=20)
        rag_f1.append(compute_token_f1(pred, q["ground_truth"]))
        rag_kw.append(compute_keyword_accuracy(pred, q["keywords"], q["ground_truth"]))
        ret_docs = {r.document_id for r in res}
        req = set(q["bridge_docs"])
        rag_doc_cov.append(len(req & ret_docs) / len(req))

    # --- 3C. SA-CMS (Memory-Only) ---
    restore_cms_state(sa_model, base_cms_snapshot)
    for d in docs:
        sa_model.ingest_structured_document(d["raw_text"], schedule_mode="structure")
    sa_f1, sa_kw = [], []
    for q in questions:
        prompt = f"Question: {q['question']}\nAnswer:"
        pred = generate_greedy_lm(sa_model, tokenizer, prompt, device, max_new_tokens=20)
        sa_f1.append(compute_token_f1(pred, q["ground_truth"]))
        sa_kw.append(compute_keyword_accuracy(pred, q["keywords"], q["ground_truth"]))

    banach_summary = {
        "mean_rho_hat": round(statistics.mean(r["rho_hat"] for r in contraction_records), 6),
        "contractive_rate": round(statistics.mean(1.0 if r["is_contractive"] else 0.0 for r in contraction_records), 4),
        "convergence_rate": round(statistics.mean(1.0 if r["converged"] else 0.0 for r in contraction_records), 4),
        "monotonic_decay_rate": round(statistics.mean(1.0 if r["monotonic_decay"] else 0.0 for r in contraction_records), 4),
        "mean_hops_to_converge": round(statistics.mean(r["num_hops"] for r in contraction_records), 2),
        "stress_non_contractive_audit": {"tau": 0.015, "rho_hat": stress_rho, "is_contractive": stress_flag},
    }
    multihop_summary = {
        "num_questions": len(questions),
        "wave_cms": {
            "keyword_acc": round(statistics.mean(wave_kw), 4),
            "token_f1": round(statistics.mean(wave_f1), 4),
            "bridge_doc_coverage": round(statistics.mean(wave_doc_cov), 4),
            "avg_hops": round(statistics.mean(wave_hops), 2),
        },
        "standard_rag_top2": {
            "keyword_acc": round(statistics.mean(rag_kw), 4),
            "token_f1": round(statistics.mean(rag_f1), 4),
            "bridge_doc_coverage": round(statistics.mean(rag_doc_cov), 4),
        },
        "sa_cms_memory": {
            "keyword_acc": round(statistics.mean(sa_kw), 4),
            "token_f1": round(statistics.mean(sa_f1), 4),
            "bridge_doc_coverage": 0.0,
        },
    }

    print(
        f"  Banach: mean_rho={banach_summary['mean_rho_hat']:.4f} | contractive_rate={banach_summary['contractive_rate']:.2%} "
        f"| converged={banach_summary['convergence_rate']:.2%} | stress(tau=0.015) rho={stress_rho:.2f} flag={stress_flag}"
    )
    print(
        f"  Multi-Hop QA ({len(questions)}Q): "
        f"SA-CMS kw={multihop_summary['sa_cms_memory']['keyword_acc']:.2f} "
        f"| RAG(top2) kw={multihop_summary['standard_rag_top2']['keyword_acc']:.2f} cov={multihop_summary['standard_rag_top2']['bridge_doc_coverage']:.2f} "
        f"| WaveCMS kw={multihop_summary['wave_cms']['keyword_acc']:.2f} cov={multihop_summary['wave_cms']['bridge_doc_coverage']:.2f}"
    )
    return {"banach_contraction": banach_summary, "multihop_qa": multihop_summary}


# =============================================================================
# PILLAR 4: CONFORMAL RISK CONTROL (FALSE-ANSWER RATE < EPSILON)
# =============================================================================


def _evaluate_crc_split(
    wave_records: List[Dict[str, Any]],
    rag_records: List[Dict[str, Any]],
    sa_records: List[Dict[str, Any]],
    epsilons: Sequence[float],
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    ans_idx = [i for i, r in enumerate(wave_records) if r["is_answerable"]]
    unans_idx = [i for i, r in enumerate(wave_records) if not r["is_answerable"]]
    rng.shuffle(ans_idx)
    rng.shuffle(unans_idx)

    cal_idx = ans_idx[: len(ans_idx) // 2] + unans_idx[: len(unans_idx) // 2]
    test_idx = ans_idx[len(ans_idx) // 2 :] + unans_idx[len(unans_idx) // 2 :]

    per_eps: Dict[str, Any] = {}
    for eps in epsilons:
        ctrl = ConformalRiskController(epsilon=eps)
        calib_data = [
            {
                "max_alpha": wave_records[i]["max_alpha"],
                "memory_reliance": wave_records[i]["memory_reliance"],
                "num_leaves": wave_records[i]["num_leaves"],
                "reliance_scale": wave_records[i]["reliance_scale"],
                "is_correct": wave_records[i]["is_correct"],
            }
            for i in cal_idx
        ]
        cal_res = ctrl.calibrate(calib_data, epsilon=eps)

        wave_false_ans = 0
        wave_unans_refused = 0
        wave_ans_accepted = 0
        for i in test_idx:
            rec = wave_records[i]
            dec = ctrl.evaluate(
                rec["max_alpha"],
                rec["memory_reliance"],
                rec["question"],
                num_leaves=rec["num_leaves"],
                reliance_scale=rec["reliance_scale"],
            )
            if not dec.refused and not rec["is_correct"]:
                wave_false_ans += 1
            if not rec["is_answerable"] and dec.refused:
                wave_unans_refused += 1
            if rec["is_answerable"] and not dec.refused:
                wave_ans_accepted += 1

        n_test = len(test_idx)
        n_test_unans = sum(1 for i in test_idx if not wave_records[i]["is_answerable"])
        n_test_ans = sum(1 for i in test_idx if wave_records[i]["is_answerable"])

        per_eps[f"eps_{eps:.2f}"] = {
            "epsilon": eps,
            "calibrated_threshold": round(float(cal_res["calibrated_threshold"]), 4),
            "calib_empirical_risk": round(float(cal_res.get("empirical_risk", 0.0)), 4),
            "wave_cms_false_answer_rate": round(wave_false_ans / max(1, n_test), 4),
            "wave_cms_meets_bound": bool((wave_false_ans / max(1, n_test)) <= eps),
            "wave_cms_refusal_acc_on_unans": round(wave_unans_refused / max(1, n_test_unans), 4),
            "wave_cms_answer_coverage": round(wave_ans_accepted / max(1, n_test_ans), 4),
        }

    # Standard RAG & SA-CMS on the exact same test_idx
    n_test = len(test_idx)
    n_test_unans = sum(1 for i in test_idx if not wave_records[i]["is_answerable"])
    n_test_ans = sum(1 for i in test_idx if wave_records[i]["is_answerable"])

    rag_false = sum(1 for i in test_idx if not rag_records[i]["refused"] and not rag_records[i]["is_correct"])
    rag_ref_acc = sum(1 for i in test_idx if not wave_records[i]["is_answerable"] and rag_records[i]["refused"])
    rag_cov = sum(1 for i in test_idx if wave_records[i]["is_answerable"] and not rag_records[i]["refused"])

    sa_false = sum(1 for i in test_idx if not sa_records[i]["is_correct"])
    sa_cov = sum(1 for i in test_idx if wave_records[i]["is_answerable"])

    return {
        "n_calib": len(cal_idx),
        "n_test": n_test,
        "wave_cms_by_epsilon": per_eps,
        "standard_rag": {
            "false_answer_rate": round(rag_false / max(1, n_test), 4),
            "refusal_acc_on_unans": round(rag_ref_acc / max(1, n_test_unans), 4),
            "answer_coverage": round(rag_cov / max(1, n_test_ans), 4),
        },
        "sa_cms_memory": {
            "false_answer_rate": round(sa_false / max(1, n_test), 4),
            "refusal_acc_on_unans": 0.0,
            "answer_coverage": round(sa_cov / max(1, n_test_ans), 4),
        },
    }


def run_pillar_4_conformal_risk_control(
    sa_model: Any,
    tokenizer: Any,
    base_cms_snapshot: Dict[str, Any],
    device: str,
    seed: int = 42,
    quick: bool = False,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PILLAR 4: CONFORMAL RISK CONTROL (FALSE-ANSWER RATE < EPSILON)")
    print("=" * 80)

    docs = get_test_documents()
    questions = get_100_test_questions()
    if quick:
        questions = questions[:25] + questions[50:63] + questions[75:87]  # 50 stratified items

    # 1. WaveCMS: ingest DOC001..DOC003, extract raw readout signals + correctness
    wave_engine = WaveCMSEngine(r=8, dim=256, max_hops=4, tau=0.20)
    for d in docs:
        wave_engine.ingest_document(d["document_id"], d["title"], d["raw_text"])

    wave_records: List[Dict[str, Any]] = []
    for q in questions:
        q_text = q["question"]
        is_ans = q["category"] == "answerable"
        eq_res = wave_engine.readout_engine.readout(query=q_text, tree=wave_engine.tree_memory, max_hops=4)
        alpha = eq_res.final_alpha or {}
        max_alpha = max(alpha.values()) if alpha else 0.0
        leaves = wave_engine.tree_memory.get_leaves()
        top_leaf = max(leaves, key=lambda lf: alpha.get(lf.node_id, 0.0)) if leaves else None
        cand_ans = top_leaf.text if top_leaf else ""
        theta_np = (
            eq_res.final_theta.detach().cpu().numpy().astype(np.float64)
            if eq_res.final_theta is not None
            else np.zeros((wave_engine.r, wave_engine.r))
        )
        logp_mem = wave_engine.attributor.compute_log_likelihood(theta_np, q_text, cand_ans)
        logp_w0 = wave_engine.attributor.compute_log_likelihood(
            np.zeros((wave_engine.r, wave_engine.r)), q_text, cand_ans
        )
        mem_rel = float(logp_mem - logp_w0)
        is_correct = False
        if is_ans and cand_ans:
            kw_ok = compute_keyword_accuracy(
                cand_ans, q.get("keywords", []), q.get("gold_answer") or ""
            )
            f1_ok = compute_token_f1(cand_ans, q.get("gold_answer") or "")
            is_correct = bool(kw_ok >= 1.0 or f1_ok >= 0.20)
        wave_records.append(
            {
                "question": q_text,
                "is_answerable": is_ans,
                "max_alpha": float(max_alpha),
                "memory_reliance": mem_rel,
                "num_leaves": len(leaves),
                "reliance_scale": (float(wave_engine.dim) / max(1, int(wave_engine.r))) ** 2,
                "is_correct": is_correct,
            }
        )

    # 2. Standard RAG: BM25 + EvidenceSelector + RefusalController
    retriever, chunker, ev_sel, ref_ctrl, _ = build_rag_pipeline(
        sa_model, tokenizer, device, score_threshold=3.0, min_query_coverage=0.35
    )
    ingest_docs_into_rag(docs, chunker, retriever)
    rag_records: List[Dict[str, Any]] = []
    for q in questions:
        q_text = q["question"]
        is_ans = q["category"] == "answerable"
        ret = retriever.retrieve(q_text, top_k=5)
        ev = ev_sel.select_evidence(q_text, ret)
        dec = ref_ctrl.decide(ev)
        refused = (not ev.has_sufficient_evidence) or dec.should_refuse
        is_correct = False
        if is_ans and not refused and ev.evidence_passages:
            top_txt = ev.evidence_passages[0].text
            kw_ok = compute_keyword_accuracy(top_txt, q.get("keywords", []), q.get("gold_answer") or "")
            is_correct = bool(kw_ok >= 1.0)
        rag_records.append({"refused": refused, "is_correct": is_correct})

    # 3. SA-CMS: memory-only unconditional answering
    restore_cms_state(sa_model, base_cms_snapshot)
    for d in docs:
        sa_model.ingest_structured_document(d["raw_text"], schedule_mode="structure")
    sa_records: List[Dict[str, Any]] = []
    for q in questions:
        q_text = q["question"]
        is_ans = q["category"] == "answerable"
        prompt = f"Câu hỏi: {q_text}\nTrả lời:"
        pred = generate_greedy_lm(sa_model, tokenizer, prompt, device, max_new_tokens=16)
        is_correct = False
        if is_ans:
            is_correct = bool(
                compute_keyword_accuracy(pred, q.get("keywords", []), q.get("gold_answer") or "") >= 1.0
                or compute_token_f1(pred, q.get("gold_answer") or "") >= 0.25
            )
        sa_records.append({"is_correct": is_correct})

    epsilons = [0.05, 0.10, 0.15]
    seeds = [seed] if quick else [seed, seed + 1, seed + 2]
    split_runs = [
        _evaluate_crc_split(wave_records, rag_records, sa_records, epsilons, s) for s in seeds
    ]

    agg_wave: Dict[str, Any] = {}
    for eps in epsilons:
        k = f"eps_{eps:.2f}"
        fars = [r["wave_cms_by_epsilon"][k]["wave_cms_false_answer_rate"] for r in split_runs]
        refs = [r["wave_cms_by_epsilon"][k]["wave_cms_refusal_acc_on_unans"] for r in split_runs]
        covs = [r["wave_cms_by_epsilon"][k]["wave_cms_answer_coverage"] for r in split_runs]
        mean_far = round(statistics.mean(fars), 4)
        agg_wave[k] = {
            "epsilon": eps,
            "mean_false_answer_rate": mean_far,
            "meets_epsilon_bound": bool(mean_far <= eps),
            "mean_refusal_acc_on_unans": round(statistics.mean(refs), 4),
            "mean_answer_coverage": round(statistics.mean(covs), 4),
        }

    rag_mean_far = round(statistics.mean(r["standard_rag"]["false_answer_rate"] for r in split_runs), 4)
    sa_mean_far = round(statistics.mean(r["sa_cms_memory"]["false_answer_rate"] for r in split_runs), 4)

    for eps in epsilons:
        w = agg_wave[f"eps_{eps:.2f}"]
        print(
            f"  eps={eps:.2f} | WaveCMS FAR={w['mean_false_answer_rate']:.4f} (<=eps: {w['meets_epsilon_bound']}) "
            f"refusal_acc={w['mean_refusal_acc_on_unans']:.2f} cov={w['mean_answer_coverage']:.2f} "
            f"| RAG FAR={rag_mean_far:.4f} | SA-CMS FAR={sa_mean_far:.4f}"
        )

    return {
        "total_questions": len(questions),
        "seeds": seeds,
        "wave_cms_aggregated": agg_wave,
        "standard_rag_mean": {
            "false_answer_rate": rag_mean_far,
            "refusal_acc_on_unans": round(
                statistics.mean(r["standard_rag"]["refusal_acc_on_unans"] for r in split_runs), 4
            ),
            "answer_coverage": round(
                statistics.mean(r["standard_rag"]["answer_coverage"] for r in split_runs), 4
            ),
        },
        "sa_cms_mean": {
            "false_answer_rate": sa_mean_far,
            "refusal_acc_on_unans": 0.0,
            "answer_coverage": 1.0,
        },
        "per_seed_runs": split_runs,
    }


# =============================================================================
# PILLAR 5: END-TO-END INFERENCE LATENCY (MS)
# =============================================================================


def run_pillar_5_inference_latency(
    sa_model: Any,
    tokenizer: Any,
    base_cms_snapshot: Dict[str, Any],
    device: str,
    quick: bool = False,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PILLAR 5: END-TO-END INFERENCE LATENCY (MS)")
    print("=" * 80)

    docs = get_test_documents()
    questions = get_100_test_questions()
    timed_questions = (questions[:10] + questions[50:55]) if quick else (questions[:15] + questions[50:60] + questions[75:80])
    warmup_questions = questions[15:18]

    # Setup WaveCMS
    wave_engine = WaveCMSEngine(r=8, dim=256, max_hops=4, tau=0.20)
    for d in docs:
        wave_engine.ingest_document(d["document_id"], d["title"], d["raw_text"])

    # Setup Standard RAG (both neural LLM generation and extractive pipeline)
    retriever, chunker, ev_sel, ref_ctrl, pipeline = build_rag_pipeline(
        sa_model, tokenizer, device, score_threshold=3.0, min_query_coverage=0.35
    )
    ingest_docs_into_rag(docs, chunker, retriever)

    # Setup SA-CMS
    restore_cms_state(sa_model, base_cms_snapshot)
    for d in docs:
        sa_model.ingest_structured_document(d["raw_text"], schedule_mode="structure")

    # Warmup
    for wq in warmup_questions:
        _ = wave_engine.query(question=wq["question"], max_hops=4)
        _ = generate_greedy_lm(sa_model, tokenizer, f"Câu hỏi: {wq['question']}\nTrả lời:", device, max_new_tokens=16)

    lat_wave, lat_wave_llm, lat_rag_llm, lat_rag_ext, lat_sa = [], [], [], [], []

    for q in timed_questions:
        qt = q["question"]

        # 1. WaveCMS Extractive (SovereignSLM extractive readout)
        t0 = time.perf_counter()
        _ = wave_engine.query(question=qt, max_hops=4)
        lat_wave.append((time.perf_counter() - t0) * 1000.0)

        # 2. WaveCMS + Neural LM (WaveCMS routing + causal_citations + SmolLM2-135M greedy)
        _sync_cuda(device)
        t0 = time.perf_counter()
        w_res = wave_engine.query(question=qt, max_hops=4)
        if not w_res.refused and w_res.causal_citations:
            ctx = "\n".join(c.get("text", "") for c in w_res.causal_citations[:3] if c.get("text"))
            if ctx:
                prompt = f"Dựa trên đoạn trích sau, trả lời ngắn gọn:\n\n{ctx}\n\nCâu hỏi: {qt}\nTrả lời:"
                _ = generate_greedy_lm(sa_model, tokenizer, prompt, device, max_new_tokens=16)
        _sync_cuda(device)
        lat_wave_llm.append((time.perf_counter() - t0) * 1000.0)

        # 3. Standard RAG (BM25 + EvidenceSelector + Refusal + SmolLM2-135M greedy)
        _sync_cuda(device)
        t0 = time.perf_counter()
        ret = retriever.retrieve(qt, top_k=3)
        ev = ev_sel.select_evidence(qt, ret)
        dec = ref_ctrl.decide(ev)
        if ev.has_sufficient_evidence and not dec.should_refuse:
            ctx = "\n".join(p.text for p in ev.evidence_passages[:3])
            prompt = f"Dựa trên đoạn trích sau, trả lời ngắn gọn:\n\n{ctx}\n\nCâu hỏi: {qt}\nTrả lời:"
            _ = generate_greedy_lm(sa_model, tokenizer, prompt, device, max_new_tokens=16)
        _sync_cuda(device)
        lat_rag_llm.append((time.perf_counter() - t0) * 1000.0)

        # 4. Standard RAG Extractive (HybridQAPipeline._answer_hybrid_mode style without neural gen)
        t0 = time.perf_counter()
        ret = retriever.retrieve(qt, top_k=3)
        ev = ev_sel.select_evidence(qt, ret)
        dec = ref_ctrl.decide(ev)
        if ev.has_sufficient_evidence and not dec.should_refuse:
            _ = pipeline._synthesize_clean_grounded_answer(qt, ev.evidence_passages[:1], language="vi", concise=True)
        lat_rag_ext.append((time.perf_counter() - t0) * 1000.0)

        # 5. SA-CMS (SmolLM2-135M + 3-level CMS greedy generation)
        _sync_cuda(device)
        t0 = time.perf_counter()
        _ = generate_greedy_lm(sa_model, tokenizer, f"Câu hỏi: {qt}\nTrả lời:", device, max_new_tokens=16)
        _sync_cuda(device)
        lat_sa.append((time.perf_counter() - t0) * 1000.0)

    wave_ext_stats = latency_stats(lat_wave)
    wave_llm_stats = latency_stats(lat_wave_llm)
    rag_llm_stats = latency_stats(lat_rag_llm)
    rag_ext_stats = latency_stats(lat_rag_ext)
    sa_llm_stats = latency_stats(lat_sa)

    stats = {
        "num_timed_queries": len(timed_questions),
        "device": device,
        "wave_cms_extractive": wave_ext_stats,
        "wave_cms_e2e": wave_ext_stats,  # backward-compatible alias for wave_cms_extractive
        "wave_cms_neural_lm": wave_llm_stats,
        "standard_rag_bm25_llm": rag_llm_stats,
        "standard_rag_bm25_extractive": rag_ext_stats,
        "sa_cms_llm": sa_llm_stats,
        "comparisons": {
            "extractive_group": {
                "wave_cms_extractive_mean_ms": wave_ext_stats["mean_ms"],
                "standard_rag_bm25_extractive_mean_ms": rag_ext_stats["mean_ms"],
            },
            "neural_lm_group": {
                "wave_cms_neural_lm_mean_ms": wave_llm_stats["mean_ms"],
                "standard_rag_bm25_llm_mean_ms": rag_llm_stats["mean_ms"],
                "sa_cms_llm_mean_ms": sa_llm_stats["mean_ms"],
            },
        },
    }
    print(
        f"  [Extractive] WaveCMS-Ext={wave_ext_stats['mean_ms']:.2f}ms (p95={wave_ext_stats['p95_ms']:.2f}ms) | "
        f"RAG-Ext={rag_ext_stats['mean_ms']:.2f}ms (p95={rag_ext_stats['p95_ms']:.2f}ms)"
    )
    print(
        f"  [Neural LM]  WaveCMS+LM={wave_llm_stats['mean_ms']:.2f}ms (p95={wave_llm_stats['p95_ms']:.2f}ms) | "
        f"RAG+LM={rag_llm_stats['mean_ms']:.2f}ms (p95={rag_llm_stats['p95_ms']:.2f}ms) | "
        f"SA-CMS+LM={sa_llm_stats['mean_ms']:.2f}ms (p95={sa_llm_stats['p95_ms']:.2f}ms)"
    )
    return stats


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark WaveCMS vs SA-CMS vs Standard RAG (Offline)")
    parser.add_argument(
        "--pillars",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        help="Benchmark pillars to run (default: 1 2 3 4 5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--quick", action="store_true", help="Run reduced-scale quick verification")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(ROOT_DIR / "results" / "wave_cms_vs_baselines_benchmark.json"),
        help="Path to write JSON benchmark report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("WAVECMS VS SA-CMS VS STANDARD RAG — REPRODUCIBLE OFFLINE BENCHMARK")
    print("=" * 80)

    t_start = time.perf_counter()
    loader = LocalModelLoader.get_instance()
    prefer_gpu = torch.cuda.is_available()
    sa_model, tokenizer = loader.load_offline_model(prefer_gpu=prefer_gpu)
    if sa_model is None or tokenizer is None:
        raise RuntimeError(
            "Offline SmolLM2-135M + SA-CMS model not available in local_runtime/. Cannot run fair baseline comparison."
        )
    device = loader.device
    base_cms_snapshot = clone_cms_state(sa_model)
    print(f"Loaded local model on device={device} (checkpoint={loader.active_checkpoint_name})")

    report: Dict[str, Any] = {
        "benchmark": "WaveCMS_vs_SA_CMS_vs_Standard_RAG",
        "seed": args.seed,
        "quick_mode": bool(args.quick),
        "device": device,
        "offline_mode": True,
        "sa_cms_checkpoint": loader.active_checkpoint_name,
        "pillars": {},
    }

    if 1 in args.pillars:
        report["pillars"]["pillar_1_zero_forgetting"] = run_pillar_1_zero_forgetting(
            sa_model, tokenizer, base_cms_snapshot, device, quick=args.quick
        )
    if 2 in args.pillars:
        report["pillars"]["pillar_2_exact_unlearning"] = run_pillar_2_exact_unlearning(
            sa_model, base_cms_snapshot, seed=args.seed, quick=args.quick
        )
    if 3 in args.pillars:
        report["pillars"]["pillar_3_banach_and_multihop"] = run_pillar_3_banach_and_multihop(
            sa_model, tokenizer, base_cms_snapshot, device
        )
    if 4 in args.pillars:
        report["pillars"]["pillar_4_conformal_risk_control"] = run_pillar_4_conformal_risk_control(
            sa_model, tokenizer, base_cms_snapshot, device, seed=args.seed, quick=args.quick
        )
    if 5 in args.pillars:
        report["pillars"]["pillar_5_inference_latency"] = run_pillar_5_inference_latency(
            sa_model, tokenizer, base_cms_snapshot, device, quick=args.quick
        )

    report["total_runtime_sec"] = round(time.perf_counter() - t_start, 2)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE in {report['total_runtime_sec']}s -> {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

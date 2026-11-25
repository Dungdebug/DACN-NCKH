"""
Ticket T10 (w7-ablation): WaveCMS-R^2 Publication-Grade Ablation Suite.

Implements:
1. Leakage-free 3-way stratified split (Fit 20% / Cal 20% / Test 60%) across
   seeds [42, 43, 44] on `data/online_benchmark_corpus.json` (272 docs, 987 queries),
   saved to `results/splits_w7.json` with verified Fit ∩ Cal ∩ Test = ∅.
2. Honest evaluation on Test 60% ONLY (no manual threshold clamping, real label
   y = 1 iff answerable and (F1 >= 0.25 or kw_acc == 1.0)) across:
   - W0: Standard Fixed-Chunk BM25 RAG
   - W1: Legacy SA-CMS Phase 4 (sentence chunking + RefusalController)
   - W2: WaveCMS 6.0 mean_key + 1-threshold Conformal (calibrated on Cal)
   - W3: T4 Sparse-LSE Router + T3* + 1-threshold Conformal (doc_scoped AND open_corpus)
   - W4a: W3 + PMIReranker (base SmolLM2, enable_cms=False, CUDA)
   - W4b: W3 + PMIReranker (SA-CMS adapter, enable_cms=True, CUDA)
   - W5: W4b + Mondrian Conformal Risk Control (8 signals, Fit weights, Cal lambda_g)
   - W6: Full WaveCMS-R^2 (W5 + UnifiedReasoner + SymbolicSolver + Ridge kill-criterion)
   Plus Leave-One-Out ablations of W6, McNemar chi^2, Bootstrap 10,000 (95% CI),
   Holm-Bonferroni, AUROC of s_w, and Empirical Risk P(ans & wrong | g) vs eps_g.
3. Reasoning Suite (60 multi-step Math / Chemistry / Crypto problems) evaluating
   UnifiedReasoner (deep, broad, deep_broad) + SymbolicSolver + 0% numeric hallucination.

Usage:
    python scripts/run_wavecms_r_ablation.py
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import statistics
import string
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

from src.hybrid_qa.answer_span import (
    relevance_score,
    select_best_sentence,
    split_sentences,
    strip_question_block,
)
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.document_store import DocumentRecord, Passage
from src.hybrid_qa.evidence import EvidenceSelector
from src.hybrid_qa.refusal import DecisionType, RefusalController
from src.hybrid_qa.retriever import BM25Retriever, RetrievalResult
from src.reasoning.reasoning_loop import UnifiedReasoner
from src.reasoning.symbolic_solver import (
    affine,
    balance_equation,
    caesar,
    solve_quadratic,
    verify_numeric_provenance,
    vigenere,
)
from src.slm.pmi_reranker import PMIReranker
from src.wave_cms.causal_attribution import (
    ConformalRiskController,
    MondrianConformalRiskController,
)
from src.wave_cms.equilibrium_readout import (
    TreeSoftmaxRouter,
    _deterministic_text_embedding,
)
from src.wave_cms.fast_weights import RidgeAssociativeMemory
from src.wave_cms.nonconformity import SIGNAL_NAMES, assign_group, extract_signals
from src.wave_cms.tree_memory import (
    DocumentTreeNode,
    WaveTreeMemory,
    compute_deterministic_routing_key,
    extract_sparse_terms,
)
from src.web.document_analyzer import DocumentAnalyzer


# =============================================================================
# TEXT NORMALIZATION & GROUND-TRUTH METRICS
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
    s = " ".join(tok for tok in s.split() if tok not in {"a", "an", "the"})
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
    if not norm_pred:
        return 0.0
    if ground_truth:
        norm_gt = normalize_text(ground_truth, fold_accents=True)
        if norm_gt and len(norm_gt) >= 3 and norm_gt in norm_pred:
            return 1.0
    valid_kws = [
        normalize_text(kw, fold_accents=True)
        for kw in keywords
        if normalize_text(kw, fold_accents=True)
    ]
    if not valid_kws:
        return 0.0
    matched = sum(1 for kw in valid_kws if kw in norm_pred)
    return 1.0 if matched >= max(1, math.ceil(len(valid_kws) * 0.5)) else 0.0


def _clean_markdown_and_citations(text: str) -> str:
    if not text:
        return ""
    s = re.sub(r"\[\d+\]", "", text)
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    lines = [
        ln.strip()
        for ln in s.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return " ".join(lines).strip()


def compute_true_y(candidate_answer: str, q_obj: Dict[str, Any]) -> Tuple[int, float, float]:
    """
    True label: y = 1 iff answerable and (F1 >= 0.25 or kw_acc == 1.0).
    Returns (y, f1, kw_acc).
    """
    answerable = bool(q_obj.get("answerable", True))
    gold = str(q_obj.get("gold_answer", ""))
    kws = q_obj.get("expected_keywords", [])
    clean_pred = _clean_markdown_and_citations(candidate_answer)
    f1 = compute_token_f1(clean_pred, gold)
    kw_acc = compute_keyword_accuracy(clean_pred, kws, gold)
    if not answerable:
        return 0, f1, kw_acc
    y = 1 if (f1 >= 0.25 or kw_acc >= 1.0) else 0
    return y, f1, kw_acc


def evaluate_decision(
    refused: bool,
    candidate_answer: str,
    q_obj: Dict[str, Any],
    latency_ms: float,
    score: Optional[float] = None,
    th_15: Optional[float] = None,
) -> Dict[str, Any]:
    answerable = bool(q_obj.get("answerable", True))
    y_cand, f1, kw_acc = compute_true_y(candidate_answer, q_obj)
    if answerable:
        correct = (not refused) and (y_cand == 1)
        hallucinated = False
    else:
        correct = bool(refused)
        hallucinated = not refused

    correct_at_15: Optional[int] = None
    hallucinated_at_15: Optional[int] = None
    if score is not None and th_15 is not None:
        ref_15 = bool(score < th_15)
        if answerable:
            correct_at_15 = int((not ref_15) and (y_cand == 1))
            hallucinated_at_15 = 0
        else:
            correct_at_15 = int(ref_15)
            hallucinated_at_15 = int(not ref_15)

    return {
        "question_id": q_obj["question_id"],
        "source": q_obj.get("source", "unknown"),
        "language": q_obj.get("language", "en"),
        "answerable": answerable,
        "refused": bool(refused),
        "y_cand": int(y_cand),
        "correct": int(1 if correct else 0),
        "hallucinated": int(1 if hallucinated else 0),
        "f1": float(f1 if not refused else 0.0),
        "kw_acc": float(kw_acc if not refused else 0.0),
        "latency_ms": float(latency_ms),
        "score": float(score) if score is not None else None,
        "correct_at_15": correct_at_15,
        "hallucinated_at_15": hallucinated_at_15,
    }


def compute_aurc_pct(records: Sequence[Dict[str, Any]]) -> Optional[float]:
    scored = [r for r in records if r.get("score") is not None]
    n = len(scored)
    if n == 0:
        return None
    ordered = sorted(scored, key=lambda r: float(r["score"]), reverse=True)
    y_sorted = np.array(
        [1.0 if (r["answerable"] and r["y_cand"] == 1) else 0.0 for r in ordered],
        dtype=np.float64,
    )
    ks = np.arange(1, n + 1, dtype=np.float64)
    coverages = ks / float(n)
    risks = np.cumsum(1.0 - y_sorted) / ks
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    aurc = float(
        trapz_fn(
            np.concatenate([[risks[0]], risks]),
            np.concatenate([[0.0], coverages]),
        )
    )
    return round(aurc * 100.0, 2)


def find_threshold_at_halluc_rate(
    cal_scores: Sequence[float],
    cal_answerable: Sequence[bool],
    target_halluc_rate: float = 0.15,
) -> float:
    unans_scores = [
        float(s) for s, ans in zip(cal_scores, cal_answerable) if not ans
    ]
    if not unans_scores:
        return 0.0
    unans_arr = np.asarray(unans_scores, dtype=np.float64)
    candidates = np.unique(
        np.concatenate([np.array([0.0]), np.asarray(cal_scores, dtype=np.float64), np.array([1.0 + 1e-9])])
    )
    candidates.sort()
    for lam in candidates:
        h_rate = float(np.mean(unans_arr >= lam))
        if h_rate <= target_halluc_rate + 1e-12:
            return float(lam)
    return float(1.0 + 1e-9)


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    ans_recs = [r for r in records if r["answerable"]]
    unans_recs = [r for r in records if not r["answerable"]]
    overall_acc = 100.0 * sum(r["correct"] for r in records) / n
    ans_acc = (
        100.0 * sum(r["correct"] for r in ans_recs) / len(ans_recs) if ans_recs else 0.0
    )
    unans_refusal = (
        100.0 * sum(1 for r in unans_recs if r["refused"]) / len(unans_recs)
        if unans_recs
        else 0.0
    )
    halluc_rate = (
        100.0 * sum(r["hallucinated"] for r in unans_recs) / len(unans_recs)
        if unans_recs
        else 0.0
    )
    mean_f1 = (
        100.0 * statistics.mean(r["f1"] for r in ans_recs) if ans_recs else 0.0
    )
    lats = sorted(r["latency_ms"] for r in records)
    p50 = statistics.median(lats)
    p95 = lats[min(len(lats) - 1, int(math.ceil(0.95 * len(lats))) - 1)]

    scored = [r for r in records if r.get("score") is not None]
    if scored:
        y_arr = np.array(
            [1 if (r["answerable"] and r["y_cand"] == 1) else 0 for r in scored],
            dtype=np.int32,
        )
        s_arr = np.array([float(r["score"]) for r in scored], dtype=np.float64)
        test_auroc = round(float(compute_auroc(y_arr, s_arr)), 4)
        aurc_pct = compute_aurc_pct(scored)
    else:
        test_auroc = None
        aurc_pct = None

    at15_recs = [r for r in records if r.get("correct_at_15") is not None]
    if at15_recs:
        overall_at_15 = round(100.0 * sum(int(r["correct_at_15"]) for r in at15_recs) / len(at15_recs), 2)
        ans_at15 = [r for r in at15_recs if r["answerable"]]
        ans_acc_at_15 = (
            round(100.0 * sum(int(r["correct_at_15"]) for r in ans_at15) / len(ans_at15), 2)
            if ans_at15
            else 0.0
        )
    else:
        overall_at_15 = None
        ans_acc_at_15 = None

    return {
        "n_total": n,
        "n_answerable": len(ans_recs),
        "n_unanswerable": len(unans_recs),
        "overall_accuracy_pct": round(overall_acc, 2),
        "answerable_accuracy_pct": round(ans_acc, 2),
        "unanswerable_refusal_pct": round(unans_refusal, 2),
        "hallucination_rate_pct": round(halluc_rate, 2),
        "mean_token_f1_pct": round(mean_f1, 2),
        "test_auroc": test_auroc,
        "aurc_pct": aurc_pct,
        "ans_acc_at_halluc_15_pct": ans_acc_at_15,
        "overall_acc_at_halluc_15_pct": overall_at_15,
        "mean_latency_ms": round(statistics.mean(lats), 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
    }


# =============================================================================
# 1. 3-WAY STRATIFIED SPLIT (FIT 20% / CAL 20% / TEST 60%)
# =============================================================================


def build_three_way_splits(
    questions: Sequence[Dict[str, Any]],
    seeds: Sequence[int] = (42, 43, 44),
) -> Dict[str, Any]:
    strata: Dict[str, List[str]] = defaultdict(list)
    for q in questions:
        key = f"{q.get('source', 'unknown')}__{bool(q.get('answerable', True))}"
        strata[key].append(str(q["question_id"]))

    splits_by_seed: Dict[str, Any] = {}
    for seed in seeds:
        rng = random.Random(seed)
        fit_ids: List[str] = []
        cal_ids: List[str] = []
        test_ids: List[str] = []
        stratum_counts: Dict[str, Dict[str, int]] = {}

        for key in sorted(strata.keys()):
            ids = list(strata[key])
            rng.shuffle(ids)
            n = len(ids)
            n_fit = max(1, int(round(0.20 * n))) if n >= 3 else 0
            n_cal = max(1, int(round(0.20 * n))) if n >= 3 else 0
            if n_fit + n_cal >= n:
                n_fit = max(0, n // 5)
                n_cal = max(0, n // 5)
            f_part = ids[:n_fit]
            c_part = ids[n_fit : n_fit + n_cal]
            t_part = ids[n_fit + n_cal :]
            fit_ids.extend(f_part)
            cal_ids.extend(c_part)
            test_ids.extend(t_part)
            stratum_counts[key] = {
                "total": n,
                "fit": len(f_part),
                "cal": len(c_part),
                "test": len(t_part),
            }

        fit_set, cal_set, test_set = set(fit_ids), set(cal_ids), set(test_ids)
        assert len(fit_set & cal_set) == 0, "Fit and Cal overlap!"
        assert len(fit_set & test_set) == 0, "Fit and Test overlap!"
        assert len(cal_set & test_set) == 0, "Cal and Test overlap!"
        assert len(fit_set) + len(cal_set) + len(test_set) == len(questions)

        splits_by_seed[str(seed)] = {
            "seed": seed,
            "fit_ids": fit_ids,
            "cal_ids": cal_ids,
            "test_ids": test_ids,
            "counts": {
                "fit": len(fit_ids),
                "cal": len(cal_ids),
                "test": len(test_ids),
                "total": len(questions),
            },
            "disjoint_verified": True,
            "strata": stratum_counts,
        }

    return {
        "schema_version": "wavecms_r2_splits_w7_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": "data/online_benchmark_corpus.json",
        "num_questions": len(questions),
        "ratios": {"fit": 0.20, "cal": 0.20, "test": 0.60},
        "stratified_by": ["source", "answerable"],
        "seeds": list(seeds),
        "splits": splits_by_seed,
    }


# =============================================================================
# FAST VECTORIZED CORPUS ROUTER (MATHEMATICALLY IDENTICAL TO TREESOFTMAXROUTER)
# =============================================================================


class FastCorpusRouter:
    """
    Pre-indexes WaveTreeMemory leaves for fast exact sparse_lse and mean_key
    routing across all 272 documents (open-corpus) or a single document.
    Matches TreeSoftmaxRouter._score_leaf_sparse_lse / mean_key to <= 1e-12.
    """

    def __init__(
        self,
        mem: WaveTreeMemory,
        router_sparse: TreeSoftmaxRouter,
        router_mean: TreeSoftmaxRouter,
    ) -> None:
        self.mem = mem
        self.router_sparse = router_sparse
        self.router_mean = router_mean

        self.leaves: List[DocumentTreeNode] = mem.get_leaves()
        self.n_leaves = len(self.leaves)
        self.leaf_idx_by_id: Dict[str, int] = {
            lf.node_id: idx for idx, lf in enumerate(self.leaves)
        }
        self.doc_leaf_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, lf in enumerate(self.leaves):
            did = str((lf.metadata or {}).get("document_id", ""))
            self.doc_leaf_indices[did].append(idx)

        # Precompute dense leaf key matrix (N, d) and sparse inverted index
        self.kappa_mat = np.vstack(
            [lf.kappa.astype(np.float64) for lf in self.leaves]
        )  # (N, 256)
        self.doc_lens = np.array(
            [max(0, int(lf.doc_len)) for lf in self.leaves], dtype=np.float64
        )
        self.avgdl = max(1.0, float(np.mean(self.doc_lens))) if self.n_leaves > 0 else 1.0
        self.inv_index: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        tmp_idx: Dict[str, List[int]] = defaultdict(list)
        tmp_tf: Dict[str, List[float]] = defaultdict(list)
        for idx, lf in enumerate(self.leaves):
            for term, tf in lf.term_tf.items():
                if tf > 0:
                    tmp_idx[term].append(idx)
                    tmp_tf[term].append(float(tf))
        for term, idxs in tmp_idx.items():
            self.inv_index[term] = (
                np.array(idxs, dtype=np.int32),
                np.array(tmp_tf[term], dtype=np.float64),
            )

        with torch.no_grad():
            self.wq_sparse = router_sparse.w_q.weight.detach().cpu().numpy().astype(np.float64)
            self.wq_mean = router_mean.w_q.weight.detach().cpu().numpy().astype(np.float64)

    def route_sparse_lse(
        self,
        query_text: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Tuple[List[Tuple[DocumentTreeNode, float, float]], float, float]:
        """
        Exact sparse_lse routing (Theorem 4).
        Returns ([(leaf_node, alpha, raw_logit), ...], margin, norm_entropy).
        """
        tau = max(1e-4, float(self.router_sparse.temperature))
        tau_s = max(1e-4, float(self.router_sparse.tau_sparse))
        h_np = compute_deterministic_routing_key(query_text, dim=self.mem.dim)
        norm = np.linalg.norm(h_np)
        if norm > 1e-8:
            h_np = h_np / norm
        q_proj = self.wq_sparse @ h_np

        if doc_id is not None and doc_id in self.doc_leaf_indices:
            cand_indices = np.array(self.doc_leaf_indices[doc_id], dtype=np.int32)
            # Vectorized exact sparse_lse over document leaves using corpus-wide IDF
            pass

        # Open-corpus (or very large doc): exact vectorized flat telescoping logits
        dense_logits = (self.kappa_mat @ q_proj) / tau
        sparse_scores = np.zeros(self.n_leaves, dtype=np.float64)
        query_terms = list(dict.fromkeys(extract_sparse_terms(query_text)))
        k1, b = 1.2, 0.75
        denom_arr = k1 * (1.0 - b + b * (self.doc_lens / self.avgdl))
        for t in query_terms:
            if t not in self.inv_index:
                continue
            idf_t = max(0.0, float(self.mem.corpus_df.idf(t, self.n_leaves)))
            idxs, tfs = self.inv_index[t]
            g = tfs / (tfs + denom_arr[idxs])
            sparse_scores[idxs] += idf_t * g

        logits = (sparse_scores / tau_s) + dense_logits
        if doc_id is not None and doc_id in self.doc_leaf_indices:
            sub_idx = np.array(self.doc_leaf_indices[doc_id], dtype=np.int32)
            sub_logits = logits[sub_idx]
            m = float(np.max(sub_logits))
            exps = np.exp(sub_logits - m)
            probs = exps / float(np.sum(exps))
            order = np.argsort(-probs)
            top_indices = sub_idx[order[:top_k]]
            top_probs = probs[order[:top_k]]
            top_logits = sub_logits[order[:top_k]]
            all_probs = probs
        else:
            m = float(np.max(logits))
            exps = np.exp(logits - m)
            probs = exps / float(np.sum(exps))
            order = np.argsort(-probs)
            top_indices = order[:top_k]
            top_probs = probs[order[:top_k]]
            top_logits = logits[order[:top_k]]
            all_probs = probs

        nz = all_probs[all_probs > 1e-12]
        ent = float(-np.sum(nz * np.log(nz))) if nz.size > 0 else 0.0
        norm_ent = ent / math.log(max(2, len(all_probs))) if len(all_probs) > 1 else 0.0
        margin = float(top_probs[0] - (top_probs[1] if len(top_probs) > 1 else 0.0))
        out = [
            (self.leaves[int(idx)], float(p), float(lg))
            for idx, p, lg in zip(top_indices, top_probs, top_logits)
        ]
        return out, margin, float(np.clip(norm_ent, 0.0, 1.0))

    def route_mean_key(
        self,
        query_text: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Tuple[List[Tuple[DocumentTreeNode, float, float]], float, float]:
        """WaveCMS 6.0 mean_key routing (uses TreeSoftmaxRouter for doc_scoped)."""
        h_np = compute_deterministic_routing_key(query_text, dim=self.mem.dim)
        norm = np.linalg.norm(h_np)
        if norm > 1e-8:
            h_np = h_np / norm

        if doc_id is not None and doc_id in self.mem.documents:
            doc_node = self.mem.documents[doc_id]
            if len(self.doc_leaf_indices.get(doc_id, [])) <= 250:
                h_t = torch.from_numpy(h_np).to(dtype=torch.float32)
                res = self.router_mean.route(h_t, doc_node)
                leaf_map = {lf.node_id: lf for lf in doc_node.get_leaves()}
                sorted_items = sorted(
                    res.leaf_weights.items(), key=lambda x: x[1], reverse=True
                )
                top_alpha = sorted_items[0][1] if sorted_items else 0.0
                sec_alpha = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
                out = [
                    (leaf_map[lid], float(w), float(w))
                    for lid, w in sorted_items[:top_k]
                    if lid in leaf_map
                ]
                return out, float(top_alpha - sec_alpha), float(res.normalized_entropy)

        tau = max(1e-4, float(self.router_mean.temperature))
        q_proj = self.wq_mean @ h_np
        logits = (self.kappa_mat @ q_proj) / tau
        if doc_id is not None and doc_id in self.doc_leaf_indices:
            sub_idx = np.array(self.doc_leaf_indices[doc_id], dtype=np.int32)
            sub_logits = logits[sub_idx]
            m = float(np.max(sub_logits))
            exps = np.exp(sub_logits - m)
            probs = exps / float(np.sum(exps))
            order = np.argsort(-probs)
            top_indices = sub_idx[order[:top_k]]
            top_probs = probs[order[:top_k]]
            top_logits = sub_logits[order[:top_k]]
            all_probs = probs
        else:
            m = float(np.max(logits))
            exps = np.exp(logits - m)
            probs = exps / float(np.sum(exps))
            order = np.argsort(-probs)
            top_indices = order[:top_k]
            top_probs = probs[order[:top_k]]
            top_logits = logits[order[:top_k]]
            all_probs = probs

        nz = all_probs[all_probs > 1e-12]
        ent = float(-np.sum(nz * np.log(nz))) if nz.size > 0 else 0.0
        norm_ent = ent / math.log(max(2, len(all_probs))) if len(all_probs) > 1 else 0.0
        margin = float(top_probs[0] - (top_probs[1] if len(top_probs) > 1 else 0.0))
        out = [
            (self.leaves[int(idx)], float(p), float(lg))
            for idx, p, lg in zip(top_indices, top_probs, top_logits)
        ]
        return out, margin, float(np.clip(norm_ent, 0.0, 1.0))


# =============================================================================
# ANSWER EXTRACTION & STATISTICAL TESTS
# =============================================================================


class _PassageAdapter:
    def __init__(self, pid: str, did: str, title: str, text: str, score: float, sec: str) -> None:
        self.passage_id = pid
        self.document_id = did
        self.document_title = title
        self.text = text
        self.score = score
        self.section_title = sec
        self.paragraph_index = 0


def extract_candidate_answer(
    query: str,
    passages: Sequence[Dict[str, Any]],
    doc_map: Dict[str, DocumentRecord],
) -> Tuple[str, bool, float, List[float]]:
    """
    Extracts concise grounded candidate answer from ranked passages.
    Returns (candidate_answer, analyzer_refused, analyzer_conf, sentence_scores).
    """
    if not passages:
        return "", True, 0.0, []
    adapters = []
    sent_pool: List[str] = []
    sec_pool: List[str] = []
    for p in passages:
        did = str(p.get("document_id", ""))
        drec = doc_map.get(did)
        title = drec.title if drec else did
        adapters.append(
            _PassageAdapter(
                pid=str(p.get("passage_id", "")),
                did=did,
                title=title,
                text=str(p.get("text", "")),
                score=float(p.get("score", 0.0)),
                sec=str(p.get("section_title", "")),
            )
        )
        for s in split_sentences(str(p.get("text", ""))):
            sent_pool.append(s)
            sec_pool.append(str(p.get("section_title", "")))

    raw_ans, cits = DocumentAnalyzer.find_best_grounded_answer(
        query=query, evidence_passages=adapters, answer_mode="balanced"
    )
    clean_ans = strip_question_block(raw_ans).strip()
    analyzer_refused = (
        not cits or "Không tìm thấy đủ thông tin bằng chứng" in clean_ans
    )
    analyzer_core = _clean_markdown_and_citations(clean_ans)
    best_sent = select_best_sentence(query, sent_pool, sec_pool) if sent_pool else ""
    if best_sent and (
        not analyzer_core
        or relevance_score(query, best_sent) >= relevance_score(query, analyzer_core)
        or len(analyzer_core) > 1.8 * len(best_sent)
    ):
        final_ans = best_sent.strip()
    else:
        final_ans = analyzer_core or clean_ans

    sentence_scores = sorted(
        (float(relevance_score(query, s, extra=sec)) for s, sec in zip(sent_pool, sec_pool)),
        reverse=True,
    )[:10]
    best_sent_rel = float(sentence_scores[0]) if sentence_scores else 0.0
    ans_rel = float(relevance_score(query, final_ans)) if final_ans else 0.0
    analyzer_conf = float(np.clip(max(best_sent_rel, ans_rel), 0.0, 1.0))
    return final_ans, analyzer_refused, analyzer_conf, sentence_scores


def mcnemar_test(a_correct: np.ndarray, b_correct: np.ndarray) -> Dict[str, Any]:
    b_only = int(np.sum((a_correct == 1) & (b_correct == 0)))
    c_only = int(np.sum((a_correct == 0) & (b_correct == 1)))
    if b_only + c_only == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        chi2 = ((abs(b_only - c_only) - 1.0) ** 2) / float(b_only + c_only)
        p_val = float(stats.chi2.sf(chi2, df=1))
    return {
        "mcnemar_chi2": round(chi2, 4),
        "p_value": p_val,
        "a_only_correct": b_only,
        "b_only_correct": c_only,
    }


def bootstrap_ci(
    a_correct: np.ndarray,
    b_correct: Optional[np.ndarray] = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(a_correct)
    if n == 0:
        return 0.0, 0.0
    vals = a_correct.astype(np.float64) - (b_correct.astype(np.float64) if b_correct is not None else 0.0)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.mean(vals[idx], axis=1) * 100.0
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, float(p) * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return MondrianConformalRiskController._compute_auroc(y_true, y_score)


# =============================================================================
# 3. REASONING SUITE (60 MULTI-STEP MATH / CHEM / CRYPTO PROBLEMS)
# =============================================================================


def build_reasoning_suite() -> List[Dict[str, Any]]:
    suite: List[Dict[str, Any]] = []
    # 20 Quadratic Math problems (ax^2 + bx + c = 0 with integer/clean roots)
    math_roots = [
        (1, 2), (2, 3), (1, 5), (-1, 4), (-2, 5),
        (3, 4), (2, 6), (-3, 3), (1, -6), (4, 5),
        (2, -7), (-4, -1), (3, 7), (5, 5), (1, 8),
        (2, 9), (-5, 2), (6, 7), (-3, -4), (4, 9),
    ]
    for i, (r1, r2) in enumerate(math_roots):
        a = 1 if i % 3 != 0 else 2
        b = -a * (r1 + r2)
        c = a * r1 * r2
        b_str = f"+ {b}x" if b >= 0 else f"- {-b}x"
        c_str = f"+ {c}" if c >= 0 else f"- {-c}"
        eq = f"{a}x^2 {b_str} {c_str} = 0"
        doc_text = (
            f"Algebra Problem {i+1}: Solve the quadratic equation {eq}. "
            f"The discriminant is delta = b^2 - 4ac. "
            f"Apply the quadratic formula x = (-b +- sqrt(delta)) / (2a)."
        )
        suite.append({
            "id": f"REASON_MATH_{i+1:02d}",
            "domain": "math",
            "query": f"Solve the quadratic equation {eq} and find its roots.",
            "context_docs": [{"passage_id": f"math_{i+1}", "document_id": f"MATH_{i+1}", "section_title": "Quadratic Equation", "text": doc_text}],
            "expected_tokens": [str(r1), str(r2)],
        })

    # 20 Chemistry Balancing problems
    chem_eqs = [
        "H2 + O2 -> H2O", "N2 + H2 -> NH3", "Fe + O2 -> Fe2O3", "CH4 + O2 -> CO2 + H2O",
        "C3H8 + O2 -> CO2 + H2O", "Al + HCl -> AlCl3 + H2", "Na + H2O -> NaOH + H2",
        "KClO3 -> KCl + O2", "Fe + H2O -> Fe3O4 + H2", "CaCO3 + HCl -> CaCl2 + CO2 + H2O",
        "Mg + O2 -> MgO", "P4 + O2 -> P4O10", "C2H6 + O2 -> CO2 + H2O",
        "NH3 + O2 -> NO + H2O", "Cu + HNO3 -> Cu(NO3)2 + NO + H2O",
        "Zn + HCl -> ZnCl2 + H2", "Al + O2 -> Al2O3", "SO2 + O2 -> SO3",
        "CO + O2 -> CO2", "H2O2 -> H2O + O2",
    ]
    for i, eq in enumerate(chem_eqs):
        bal = balance_equation(eq)
        doc_text = (
            f"Chemistry Reaction {i+1}: Consider the unbalanced chemical reaction {eq}. "
            f"Conservation of mass requires equal atoms of each element on both sides."
        )
        suite.append({
            "id": f"REASON_CHEM_{i+1:02d}",
            "domain": "chemistry",
            "query": f"Balance the chemical equation {eq}.",
            "context_docs": [{"passage_id": f"chem_{i+1}", "document_id": f"CHEM_{i+1}", "section_title": "Stoichiometry", "text": doc_text}],
            "expected_tokens": [bal["balanced"]],
        })

    # 20 Classical Cryptography problems (Caesar, Vigenere, Affine)
    words = ["WAVECMS", "CONFORMAL", "BANACH", "HAARTREE", "MONDRIAN", "CAUSAL", "GROUNDED"]
    for i in range(20):
        plain = words[i % len(words)]
        if i % 3 == 0:
            shift = (i % 11) + 1
            cipher = caesar(plain, shift, decrypt=False)
            q = f"Decrypt Caesar cipher caesar('{cipher}', {shift}, decrypt=True)."
            expected = plain
            doc_text = f"Cryptography Note {i+1}: Caesar cipher shifts letters modulo 26. Ciphertext is {cipher} with shift {shift}."
        elif i % 3 == 1:
            key = "KEY" if i % 2 == 0 else "WAVE"
            cipher = vigenere(plain, key, decrypt=False)
            q = f"Decrypt Vigenere cipher vigenere('{cipher}', '{key}', decrypt=True)."
            expected = plain
            doc_text = f"Cryptography Note {i+1}: Vigenere cipher uses repeating key '{key}'. Ciphertext is {cipher}."
        else:
            a_coef, b_coef = 5, (i % 9) + 1
            cipher = affine(plain, a_coef, b_coef, decrypt=False)
            q = f"Decrypt Affine cipher affine('{cipher}', {a_coef}, {b_coef}, decrypt=True)."
            expected = plain
            doc_text = f"Cryptography Note {i+1}: Affine cipher E(x) = ({a_coef}x + {b_coef}) mod 26. Ciphertext is {cipher}."

        suite.append({
            "id": f"REASON_CRYPTO_{i+1:02d}",
            "domain": "crypto",
            "query": q,
            "context_docs": [{"passage_id": f"crypto_{i+1}", "document_id": f"CRYPTO_{i+1}", "section_title": "Classical Ciphers", "text": doc_text}],
            "expected_tokens": [expected],
        })

    return suite


def evaluate_reasoning_suite(suite: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    results_by_mode: Dict[str, Any] = {}
    for mode in ("deep", "broad", "deep_broad"):
        correct = 0
        hallucinated_nums = 0
        latencies: List[float] = []
        by_domain: Dict[str, List[int]] = defaultdict(list)

        for item in suite:
            docs = item["context_docs"]
            reasoner = UnifiedReasoner(
                retrieve_fn=lambda q, top_k=3, _docs=docs: _docs[:top_k],
                max_deep_rounds=2,
            )
            t0 = time.perf_counter()
            trace = reasoner.run(
                query=item["query"],
                mode=mode,
                context_docs=docs,
                base_routing={"rho_hat": 0.45, "conformal": {"score": 0.9}},
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            combined_out = f"{trace.answer} {' '.join(trace.solver_traces)}"
            ok_tok = all(tok.lower() in combined_out.lower() for tok in item["expected_tokens"])
            num_ok, _ = verify_numeric_provenance(
                trace.answer,
                [d["text"] for d in docs],
                solver_trace=trace.solver_traces,
            )
            if not num_ok:
                hallucinated_nums += 1
            is_ok = 1 if ok_tok else 0
            correct += is_ok
            by_domain[item["domain"]].append(is_ok)

        n = len(suite)
        results_by_mode[mode] = {
            "n": n,
            "accuracy_pct": round(100.0 * correct / n, 2),
            "numeric_hallucination_rate_pct": round(100.0 * hallucinated_nums / n, 2),
            "mean_latency_ms": round(statistics.mean(latencies), 2),
            "by_domain_accuracy_pct": {
                dom: round(100.0 * sum(vals) / len(vals), 2)
                for dom, vals in by_domain.items()
            },
        }
    return results_by_mode


# =============================================================================
# MAIN ABLATION PIPELINE
# =============================================================================


def main() -> None:
    t_start = time.perf_counter()
    corpus_path = ROOT_DIR / "data" / "online_benchmark_corpus.json"
    splits_path = ROOT_DIR / "results" / "splits_w7.json"
    out_path = ROOT_DIR / "results" / "wavecms_r2_ablation.json"

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    raw_docs = corpus["documents"]
    questions = corpus["questions"]
    print(f"[INFO] Loaded {len(raw_docs)} docs, {len(questions)} questions.")

    # 0. Run Reasoning Suite (60 problems) first for fast verification
    reasoning_suite = build_reasoning_suite()
    reasoning_results = evaluate_reasoning_suite(reasoning_suite)
    print(f"[INFO] Reasoning Suite (60 problems): { {m: r['accuracy_pct'] for m, r in reasoning_results.items()} }")

    # 1. Build and save leakage-free 3-way splits
    splits_obj = build_three_way_splits(questions, seeds=(42, 43, 44))
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_obj, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved 3-way stratified splits to {splits_path}")

    q_by_id: Dict[str, Dict[str, Any]] = {str(q["question_id"]): q for q in questions}
    raw_doc_by_id: Dict[str, Dict[str, Any]] = {str(d["document_id"]): d for d in raw_docs}

    # Build DocumentRecords
    docs: List[DocumentRecord] = []
    for d in raw_docs:
        raw_text = str(d.get("content") or d.get("text") or "")
        docs.append(
            DocumentRecord(
                document_id=str(d["document_id"]),
                version=1,
                title=str(d.get("title", d["document_id"])),
                raw_text=raw_text,
                metadata={
                    "source": str(d.get("source", "online")),
                    "language": str(d.get("language", "en")),
                },
            )
        )
    doc_map: Dict[str, DocumentRecord] = {d.document_id: d for d in docs}

    # 2. Build BM25 indices (W0 fixed, W1 sentence, W2+ structure/sentence)
    t_idx = time.perf_counter()
    def _build_bm25(chunk_size: int, overlap: int, strategy: str, adaptive: bool, prox: float) -> BM25Retriever:
        chunker = DocumentChunker(
            chunk_size=chunk_size, chunk_overlap=overlap, min_chunk_size=48,
            strategy=strategy, adaptive_overlap=adaptive,
        )
        passages: List[Passage] = []
        for doc in docs:
            passages.extend(chunker.chunk_document(doc))
        ret = BM25Retriever(k1=1.5, b=0.75, proximity_weight=prox)
        ret.build_index(passages)
        return ret

    w0_ret = _build_bm25(320, 0, "fixed", False, 0.0)
    w1_ret = _build_bm25(380, 32, "sentence", False, 0.0)
    w2_ret = _build_bm25(480, 48, "sentence", True, 0.25)
    print(f"[INFO] Built 3 BM25 indices in {time.perf_counter() - t_idx:.2f}s")

    # 3. Bulk ingest 272 docs into WaveTreeMemory ONCE
    t_mem = time.perf_counter()
    w_chunker = DocumentChunker(
        chunk_size=480, chunk_overlap=48, min_chunk_size=48,
        strategy="sentence", adaptive_overlap=True,
    )
    mem = WaveTreeMemory(r=8, key_dim=256)
    orig_tc = mem.telescoping_consolidation
    mem.telescoping_consolidation = lambda: None  # defer single O(N) pass to end
    for doc in docs:
        passages = w_chunker.chunk_document(doc)
        secs = [
            {"section_title": p.section_title or "Overview", "paragraphs": [p.text]}
            for p in passages
        ]
        mem.add_document(doc.document_id, doc.title, secs)
    mem.telescoping_consolidation = orig_tc
    mem.telescoping_consolidation()

    router_sparse = TreeSoftmaxRouter(d_model=256, temperature=0.2, routing_mode="sparse_lse", tau_sparse=1.0)
    router_mean = TreeSoftmaxRouter(d_model=256, temperature=0.2, routing_mode="mean_key")
    fast_router = FastCorpusRouter(mem, router_sparse, router_mean)
    idf_map: Dict[str, float] = {
        term: max(0.0, float(mem.corpus_df.idf(term, fast_router.n_leaves)))
        for term in fast_router.inv_index.keys()
    }
    print(f"[INFO] Ingested {len(docs)} docs ({fast_router.n_leaves} leaves) into WaveTreeMemory in {time.perf_counter() - t_mem:.2f}s")

    # 4. Load PMI Rerankers (W4a base SmolLM2, W4b SA-CMS adapter)
    pmi_w4a = PMIReranker.from_checkpoint(enable_cms=False, window=32, k=3, budget_ms=1000.0)
    pmi_w4b = PMIReranker.from_checkpoint(enable_cms=True, window=32, k=3, budget_ms=1000.0)
    if pmi_w4a.available:
        pmi_w4a.score("warmup query", ["warmup passage one", "warmup passage two", "warmup passage three"], use_budget_guard=False)
    if pmi_w4b.available:
        pmi_w4b.score("warmup query", ["warmup passage one", "warmup passage two", "warmup passage three"], use_budget_guard=False)
    print(f"[INFO] PMIReranker W4a={pmi_w4a.mode_label} (avail={pmi_w4a.available}), W4b={pmi_w4b.mode_label} (avail={pmi_w4b.available})")

    # 5. Precompute deterministic per-query candidate records (NO labels/thresholds used!)
    w1_selector = EvidenceSelector(score_threshold=1.5, max_evidence=5, min_evidence=1, min_query_coverage=0.15, dynamic_top_k=False)
    w1_refusal = RefusalController(min_evidence_score=2.0, min_evidence_count=1, min_query_coverage=0.20)
    scalar_crc_helper = ConformalRiskController(epsilon=0.10, default_threshold=0.45)

    per_query_cache: Dict[str, Dict[str, Any]] = {}
    t_pre = time.perf_counter()
    for idx, q in enumerate(questions):
        qid = str(q["question_id"])
        q_text = str(q["question"])
        doc_id = q.get("document_id")
        lang = str(q.get("language", "en"))
        d_raw = raw_doc_by_id.get(str(doc_id), {})
        doc_meta = {"source": q.get("source", ""), "language": lang}
        doc_len_chars = len(str(d_raw.get("content") or d_raw.get("text") or ""))
        group = assign_group(doc_meta=doc_meta, query=q_text, doc_length_chars=doc_len_chars)

        # --- W0: Standard Fixed-Chunk BM25 RAG ---
        t0 = time.perf_counter()
        res_w0 = w0_ret.retrieve(q_text, top_k=3, document_id=doc_id)
        if res_w0:
            sents0 = split_sentences(res_w0[0].text)
            ans_w0 = " ".join(sents0[:2]) if sents0 else res_w0[0].text[:240]
        else:
            ans_w0 = ""
        lat_w0 = (time.perf_counter() - t0) * 1000.0

        # --- W1: Legacy SA-CMS Phase 4 ---
        t0 = time.perf_counter()
        res_w1 = w1_ret.retrieve(q_text, top_k=5, document_id=doc_id)
        ev_w1 = w1_selector.select_evidence(q_text, res_w1, dynamic_top_k=False)
        dec_w1 = w1_refusal.decide(ev_w1, language=lang)
        if dec_w1.decision == DecisionType.REFUSAL:
            ans_w1, ref_w1 = dec_w1.refusal_message, True
        else:
            sp1, sc1 = [], []
            for c in ev_w1.evidence_passages:
                for s in split_sentences(c.text):
                    sp1.append(s)
                    sc1.append(c.section_title or "")
            ans_w1 = select_best_sentence(q_text, sp1, sc1) if sp1 else ""
            if not ans_w1 and ev_w1.evidence_passages:
                ans_w1 = ev_w1.evidence_passages[0].text[:220]
            ref_w1 = False
        lat_w1 = (time.perf_counter() - t0) * 1000.0

        # --- W2: WaveCMS 6.0 mean_key (doc_scoped) ---
        t0 = time.perf_counter()
        routed_w2, margin_w2, ent_w2 = fast_router.route_mean_key(q_text, doc_id=doc_id, top_k=5)
        pass_w2 = [
            {"passage_id": lf.node_id, "document_id": (lf.metadata or {}).get("document_id", doc_id),
             "text": lf.text, "score": w, "alpha": w, "section_title": lf.title}
            for lf, w, _ in routed_w2
        ]
        ans_w2, _, an_conf_w2, sents_w2 = extract_candidate_answer(q_text, pass_w2, doc_map)
        rel_w2 = relevance_score(q_text, ans_w2)
        max_a2 = routed_w2[0][1] if routed_w2 else 0.0
        score_w2 = scalar_crc_helper.compute_composite_confidence(max_a2, rel_w2 * 3.0 - 1.5)
        lat_w2 = (time.perf_counter() - t0) * 1000.0

        # --- W3: T4 Sparse-LSE Router (doc_scoped) ---
        t0 = time.perf_counter()
        routed_w3, margin_w3, ent_w3 = fast_router.route_sparse_lse(q_text, doc_id=doc_id, top_k=5)
        bm25_w3 = w2_ret.retrieve(q_text, top_k=5, document_id=doc_id, diversify=True)
        bm25_max_w3 = float(bm25_w3[0].score) if bm25_w3 else 0.0
        pass_w3 = [
            {"passage_id": lf.node_id, "document_id": (lf.metadata or {}).get("document_id", doc_id),
             "text": lf.text, "score": lg, "alpha": w, "bm25": bm25_max_w3, "section_title": lf.title}
            for lf, w, lg in routed_w3
        ]
        ans_w3, _, an_conf_w3, sents_w3 = extract_candidate_answer(q_text, pass_w3, doc_map)
        rel_w3 = relevance_score(q_text, ans_w3)
        alpha_r3 = 1.0 - math.exp(-0.22 * max(0.0, bm25_max_w3))
        score_w3 = scalar_crc_helper.compute_composite_confidence(alpha_r3, rel_w3 * 3.0 + margin_w3 * 2.0 - 1.5)
        lat_w3 = (time.perf_counter() - t0) * 1000.0

        # --- W3 Open-Corpus (document_id=None) ---
        t0 = time.perf_counter()
        routed_w3o, margin_w3o, _ = fast_router.route_sparse_lse(q_text, doc_id=None, top_k=5)
        pass_w3o = [
            {"passage_id": lf.node_id, "document_id": (lf.metadata or {}).get("document_id", ""),
             "text": lf.text, "score": lg, "alpha": w, "section_title": lf.title}
            for lf, w, lg in routed_w3o
        ]
        ans_w3o, _, _, _ = extract_candidate_answer(q_text, pass_w3o, doc_map)
        rel_w3o = relevance_score(q_text, ans_w3o)
        max_a3o = routed_w3o[0][1] if routed_w3o else 0.0
        score_w3o = scalar_crc_helper.compute_composite_confidence(min(1.0, max_a3o * 10.0), rel_w3o * 3.0 - 1.5)
        lat_w3o = (time.perf_counter() - t0) * 1000.0

        # --- W4a: W3 + PMIReranker (base SmolLM2, enable_cms=False, Margin-Gated) ---
        # Interleave [sparse_leaf_0, bm25_top_unique_0, sparse_leaf_1, ...] so k=3 scores both retrievers
        sparse_cands: List[Dict[str, Any]] = []
        seen_sparse: Set[str] = set()
        for p in pass_w3[:3]:
            if p["text"] not in seen_sparse:
                seen_sparse.add(p["text"])
                sparse_cands.append(dict(p))
        bm25_cands: List[Dict[str, Any]] = []
        seen_bm25: Set[str] = set(seen_sparse)
        for r in bm25_w3[:3]:
            if r.text not in seen_bm25:
                seen_bm25.add(r.text)
                bm25_cands.append({
                    "passage_id": r.passage_id, "document_id": r.document_id,
                    "text": r.text, "score": r.score, "alpha": 0.1,
                    "bm25": r.score, "section_title": r.section_title or "",
                })
        cand_pool: List[Dict[str, Any]] = []
        for i_c in range(max(len(sparse_cands), len(bm25_cands))):
            if i_c < len(sparse_cands):
                cand_pool.append(sparse_cands[i_c])
            if i_c < len(bm25_cands):
                cand_pool.append(bm25_cands[i_c])
        cand_texts = [p["text"] for p in cand_pool]

        bm25_best_sent = ans_w1 if not ref_w1 else ""
        if bm25_best_sent and ans_w3:
            w3_toks = set(normalize_text(ans_w3, fold_accents=True).split())
            w1_toks = set(normalize_text(bm25_best_sent, fold_accents=True).split())
            cross_agree_w3_w1 = (len(w3_toks & w1_toks) / max(1, min(len(w3_toks), len(w1_toks)))) if (w3_toks and w1_toks) else 0.0
        else:
            cross_agree_w3_w1 = 0.0

        if pmi_w4a.available and cand_texts:
            pmi_scores_a, pmi_ran_a = pmi_w4a.score_passages_gated(
                q_text,
                cand_texts,
                lse_margin=margin_w3,
                cross_agree=cross_agree_w3_w1,
                use_budget_guard=False,
            )
        else:
            pmi_scores_a, pmi_ran_a = None, False

        if pmi_ran_a and pmi_scores_a is not None:
            lat_w4a = lat_w3 + pmi_w4a.last_latency_ms
            order_a = sorted(range(len(cand_pool)), key=lambda i: pmi_scores_a[i], reverse=True)
            pass_w4a = [cand_pool[i] for i in order_a[:3]]
            ans_w4a, _, an_conf_w4a, sents_w4a = extract_candidate_answer(q_text, pass_w4a, doc_map)
            pmi_top_a: Optional[float] = float(pmi_scores_a[order_a[0]])
        else:
            lat_w4a = lat_w3
            pass_w4a = pass_w3[:3]
            ans_w4a = ans_w3
            an_conf_w4a = an_conf_w3
            sents_w4a = sents_w3
            pmi_top_a = None

        score_w4a = scalar_crc_helper.compute_composite_confidence(
            alpha_r3, relevance_score(q_text, ans_w4a) * 3.0 + (pmi_top_a or 0.0) * 0.4 - 1.2
        )

        # --- W4b: W3 + PMIReranker (SA-CMS adapter, enable_cms=True; kept isolated for honest negative reporting) ---
        pmi_scores_b = pmi_w4b.score(q_text, cand_texts, use_budget_guard=False) if pmi_w4b.available else [0.0] * len(cand_texts)
        lat_w4b = lat_w3 + (pmi_w4b.last_latency_ms if pmi_w4b.available else 0.0)
        order_b = sorted(range(len(cand_pool)), key=lambda i: pmi_scores_b[i], reverse=True)
        pass_w4b = [cand_pool[i] for i in order_b[:3]]
        ans_w4b, _, _, _ = extract_candidate_answer(q_text, pass_w4b, doc_map)
        pmi_top_b = float(pmi_scores_b[order_b[0]]) if order_b and pmi_w4b.available else None
        score_w4b = scalar_crc_helper.compute_composite_confidence(
            alpha_r3, relevance_score(q_text, ans_w4b) * 3.0 + (pmi_top_b or 0.0) * 0.4 - 1.2
        )

        # --- R³ Signal Extraction for W5 / W6 / LOO ---
        routing_meta = {
            "margin": margin_w3,
            "entropy": ent_w3,
            "bm25_max": bm25_max_w3,
            "bm25_scores": [float(r.score) for r in bm25_w3],
        }
        routing_meta_w2 = {
            "margin": margin_w2,
            "entropy": ent_w2,
            "bm25_max": bm25_max_w3,
            "bm25_scores": [float(r.score) for r in bm25_w3],
        }
        sigs_w5_pmi = extract_signals(
            query=q_text,
            candidate_answer=ans_w4a,
            evidence=pass_w4a,
            routing=routing_meta,
            pmi=pmi_top_a,
            bm25_best_sentence=bm25_best_sent,
            idf_map=idf_map,
            analyzer_conf=an_conf_w4a,
            sentence_scores=sents_w4a,
        )
        sigs_no_pmi = extract_signals(
            query=q_text,
            candidate_answer=ans_w3,
            evidence=pass_w3,
            routing=routing_meta,
            pmi=None,
            bm25_best_sentence=bm25_best_sent,
            idf_map=idf_map,
            analyzer_conf=an_conf_w3,
            sentence_scores=sents_w3,
        )
        sigs_w2 = extract_signals(
            query=q_text,
            candidate_answer=ans_w2,
            evidence=pass_w2,
            routing=routing_meta_w2,
            pmi=None,
            bm25_best_sentence=bm25_best_sent,
            idf_map=idf_map,
            analyzer_conf=an_conf_w2,
            sentence_scores=sents_w2,
        )
        y_w4a, _, _ = compute_true_y(ans_w4a, q)
        y_w4b, _, _ = compute_true_y(ans_w4b, q)
        y_w3, _, _ = compute_true_y(ans_w3, q)
        y_w2, _, _ = compute_true_y(ans_w2, q)

        per_query_cache[qid] = {
            "q": q,
            "group": group,
            "w0": {"ans": ans_w0, "refused": False, "lat": lat_w0},
            "w1": {"ans": ans_w1, "refused": ref_w1, "lat": lat_w1},
            "w2": {"ans": ans_w2, "score": score_w2, "y": y_w2, "lat": lat_w2, "pass": pass_w2},
            "w3": {"ans": ans_w3, "score": score_w3, "y": y_w3, "lat": lat_w3, "pass": pass_w3},
            "w3_open": {"ans": ans_w3o, "score": score_w3o, "y": compute_true_y(ans_w3o, q)[0], "lat": lat_w3o},
            "w4a": {"ans": ans_w4a, "score": score_w4a, "y": y_w4a, "lat": lat_w4a, "pass": pass_w4a, "pmi": pmi_top_a},
            "w4b": {"ans": ans_w4b, "score": score_w4b, "y": y_w4b, "lat": lat_w4b, "pass": pass_w4b, "pmi": pmi_top_b},
            "w5_pmi_row": {"signals": sigs_w5_pmi, "group": group, "y": y_w4a, "ans": ans_w4a, "lat": lat_w4a, "pass": pass_w4a},
            "w5_no_pmi_row": {"signals": sigs_no_pmi, "group": group, "y": y_w3, "ans": ans_w3, "lat": lat_w3, "pass": pass_w3},
            "w2_row": {"signals": sigs_w2, "group": group, "y": y_w2, "ans": ans_w2, "lat": lat_w2, "pass": pass_w2},
        }
        if (idx + 1) % 200 == 0:
            print(f"  [PRECOMPUTE] {idx + 1}/{len(questions)} queries ({time.perf_counter() - t_pre:.1f}s)")

    print(f"[INFO] Precomputed all {len(questions)} queries in {time.perf_counter() - t_pre:.2f}s")

    # 6. Check RidgeAssociativeMemory (T4) Kill-Criterion on Fit-QASPER (seed 42)
    s42_fit_ids = set(splits_obj["splits"]["42"]["fit_ids"])
    qasper_fit = [
        q_by_id[qid] for qid in s42_fit_ids
        if q_by_id[qid].get("source") == "qasper" and bool(q_by_id[qid].get("answerable", True))
    ]
    ridge = RidgeAssociativeMemory(dim=256, lam=1.0)
    qasper_doc_ids = {str(q["document_id"]) for q in qasper_fit if q.get("document_id")}
    for did in qasper_doc_ids:
        d_node = mem.documents.get(did)
        if d_node is None:
            continue
        leaves_d = d_node.get_leaves()
        if leaves_d:
            K = np.vstack([lf.kappa for lf in leaves_d])
            # Self-associative / section-title value target
            V = np.vstack([
                compute_deterministic_routing_key(f"{lf.title} {lf.text[:160]}", dim=256)
                for lf in leaves_d
            ])
            ridge.add_document_pairs(did, K, V)

    base_qasper_ok = 0
    ridge_qasper_ok = 0
    for q in qasper_fit:
        qid = str(q["question_id"])
        base_y = per_query_cache[qid]["w3"]["y"]
        base_qasper_ok += base_y
        did = str(q.get("document_id", ""))
        d_node = mem.documents.get(did)
        if d_node is None:
            ridge_qasper_ok += base_y
            continue
        q_vec = compute_deterministic_routing_key(q["question"], dim=256)
        v_hat = ridge.predict(q_vec)
        v_norm = np.linalg.norm(v_hat)
        if v_norm > 1e-8:
            v_hat = v_hat / v_norm
        leaves_d = d_node.get_leaves()
        scored_r = sorted(
            leaves_d,
            key=lambda lf: float(np.dot(v_hat, lf.kappa)),
            reverse=True,
        )[:3]
        pass_r = [
            {"passage_id": lf.node_id, "document_id": did, "text": lf.text, "score": 1.0, "section_title": lf.title}
            for lf in scored_r
        ]
        ans_r, _, _, _ = extract_candidate_answer(q["question"], pass_r, doc_map)
        y_r, _, _ = compute_true_y(ans_r, q)
        ridge_qasper_ok += y_r

    n_qf = max(1, len(qasper_fit))
    base_qf_pct = 100.0 * base_qasper_ok / n_qf
    ridge_qf_pct = 100.0 * ridge_qasper_ok / n_qf
    ridge_delta_pct = ridge_qf_pct - base_qf_pct
    ridge_killed = ridge_delta_pct < 1.0
    ridge_kill_report = {
        "n_fit_qasper": len(qasper_fit),
        "base_answerable_pct": round(base_qf_pct, 2),
        "ridge_answerable_pct": round(ridge_qf_pct, 2),
        "delta_pct": round(ridge_delta_pct, 2),
        "kill_criterion_threshold_pct": 1.0,
        "kill_criterion_triggered": bool(ridge_killed),
        "enabled_by_default_in_w6": not bool(ridge_killed),
    }
    print(f"[INFO] RidgeAssociativeMemory Fit-QASPER check: base={base_qf_pct:.2f}%, ridge={ridge_qf_pct:.2f}%, delta={ridge_delta_pct:+.2f}% -> killed={ridge_killed}")

    # 6b. R3-T4 PMI Kill-Criterion strictly on Fit+Cal (ZERO Test leakage!)
    pmi_kc_per_seed: Dict[str, Any] = {}
    delta_aurocs_fc: List[float] = []
    delta_ans15_fc: List[float] = []
    for seed_str, sp in splits_obj["splits"].items():
        fit_ids_kc = sp["fit_ids"]
        cal_ids_kc = sp["cal_ids"]
        fit_pmi_kc = [per_query_cache[qid]["w5_pmi_row"] for qid in fit_ids_kc]
        cal_pmi_kc = [per_query_cache[qid]["w5_pmi_row"] for qid in cal_ids_kc]
        fit_nopmi_kc = [per_query_cache[qid]["w5_no_pmi_row"] for qid in fit_ids_kc]
        cal_nopmi_kc = [per_query_cache[qid]["w5_no_pmi_row"] for qid in cal_ids_kc]

        ctrl_pmi_kc = MondrianConformalRiskController(epsilon=0.10, l2_reg=1.0, seed=int(seed_str))
        ctrl_pmi_kc.fit_weights(fit_pmi_kc, forward_select_r3=True)
        ctrl_nopmi_kc = MondrianConformalRiskController(epsilon=0.10, l2_reg=1.0, seed=int(seed_str))
        ctrl_nopmi_kc.fit_weights(fit_nopmi_kc, forward_select_r3=True)

        cal_scores_pmi = [ctrl_pmi_kc.score_signals(r["signals"]) for r in cal_pmi_kc]
        cal_scores_nopmi = [ctrl_nopmi_kc.score_signals(r["signals"]) for r in cal_nopmi_kc]
        cal_y_pmi = np.array([int(r["y"]) for r in cal_pmi_kc], dtype=np.int32)
        cal_y_nopmi = np.array([int(r["y"]) for r in cal_nopmi_kc], dtype=np.int32)
        cal_ans_flags = [bool(q_by_id[qid].get("answerable", True)) for qid in cal_ids_kc]

        auroc_pmi_kc = compute_auroc(cal_y_pmi, np.array(cal_scores_pmi, dtype=np.float64))
        auroc_nopmi_kc = compute_auroc(cal_y_nopmi, np.array(cal_scores_nopmi, dtype=np.float64))

        th15_pmi_kc = find_threshold_at_halluc_rate(cal_scores_pmi, cal_ans_flags, 0.15)
        th15_nopmi_kc = find_threshold_at_halluc_rate(cal_scores_nopmi, cal_ans_flags, 0.15)
        ans15_pmi_kc = 100.0 * float(np.mean([
            1 if (s >= th15_pmi_kc and r["y"] == 1) else 0
            for s, r, is_a in zip(cal_scores_pmi, cal_pmi_kc, cal_ans_flags) if is_a
        ]))
        ans15_nopmi_kc = 100.0 * float(np.mean([
            1 if (s >= th15_nopmi_kc and r["y"] == 1) else 0
            for s, r, is_a in zip(cal_scores_nopmi, cal_nopmi_kc, cal_ans_flags) if is_a
        ]))

        d_auc = auroc_pmi_kc - auroc_nopmi_kc
        d_a15 = ans15_pmi_kc - ans15_nopmi_kc
        delta_aurocs_fc.append(d_auc)
        delta_ans15_fc.append(d_a15)
        pmi_kc_per_seed[seed_str] = {
            "w5_pmi_cal_auroc": round(auroc_pmi_kc, 4),
            "w5_nopmi_cal_auroc": round(auroc_nopmi_kc, 4),
            "delta_auroc": round(d_auc, 4),
            "w5_pmi_cal_ans_at_halluc_15_pct": round(ans15_pmi_kc, 2),
            "w5_nopmi_cal_ans_at_halluc_15_pct": round(ans15_nopmi_kc, 2),
            "delta_ans_at_halluc_15_pp": round(d_a15, 2),
        }

    mean_delta_auroc_kc = float(statistics.mean(delta_aurocs_fc))
    mean_delta_ans15_kc = float(statistics.mean(delta_ans15_fc))
    pmi_killed = (mean_delta_auroc_kc < 0.02) or (mean_delta_ans15_kc < 1.5)
    active_w5_key = "w5_no_pmi_row" if pmi_killed else "w5_pmi_row"
    pmi_r3_kill_report = {
        "evaluation_split": "Fit+Cal ONLY (leak-free)",
        "mean_delta_auroc": round(mean_delta_auroc_kc, 4),
        "mean_delta_ans_at_halluc_15_pp": round(mean_delta_ans15_kc, 2),
        "kill_criterion_thresholds": {
            "min_delta_auroc": 0.02,
            "min_delta_ans_at_halluc_15_pp": 1.5,
        },
        "kill_criterion_triggered": bool(pmi_killed),
        "w5_w6_mode": "fast_zero_pmi" if pmi_killed else "gated_w4a_pmi",
        "per_seed": pmi_kc_per_seed,
    }
    print(
        f"[INFO] R3-T4 PMI Kill-Criterion (Fit+Cal): d_AUROC={mean_delta_auroc_kc:+.4f}, "
        f"d_Ans@15%={mean_delta_ans15_kc:+.2f}pp -> killed={pmi_killed} (mode={pmi_r3_kill_report['w5_w6_mode']})"
    )

    def _run_w6_reasoner(
        ans_base: str,
        refused_in: bool,
        ev_passages: Sequence[Dict[str, Any]],
        q_obj: Dict[str, Any],
    ) -> Tuple[str, bool, float]:
        t_r = time.perf_counter()
        ans_out = ans_base
        ref_out = refused_in
        if not ref_out:
            ev_docs = [
                {
                    "passage_id": p["passage_id"],
                    "document_id": p["document_id"],
                    "text": p["text"],
                    "section_title": p.get("section_title", ""),
                }
                for p in ev_passages
            ]
            reasoner = UnifiedReasoner(
                retrieve_fn=lambda _q, top_k=3, _ev=ev_docs: _ev[:top_k],
                max_deep_rounds=1,
            )
            norm_docs = reasoner._normalize_docs(ev_docs)
            prem = reasoner._premise_validator.validate_premise(q_obj["question"], norm_docs)
            if prem.has_false_premise:
                ref_out = True
            else:
                s_traces = reasoner._run_symbolic_solver(q_obj["question"], norm_docs)
                if s_traces:
                    ans_out = f"{ans_out} {' '.join(s_traces)}".strip()
                clean_out, _ = reasoner._filter_ungrounded_numbers(ans_out, norm_docs, s_traces)
                if clean_out:
                    ans_out = clean_out
        return ans_out, ref_out, (time.perf_counter() - t_r) * 1000.0

    # 7. Evaluate W0–W6 + True Leave-One-Out across all 3 seeds (Test 60% ONLY!)
    methods = [
        "W0_standard_bm25_rag",
        "W1_legacy_sa_cms",
        "W2_wavecms6_mean_key",
        "W3_sparse_lse_doc",
        "W3_sparse_lse_open_corpus",
        "W4a_sparse_lse_pmi_base",
        "W4b_sparse_lse_pmi_sacms",
        "W5_mondrian_8sig",
        "W6_wavecms_r2_full",
        # True Leave-One-Out ablations of W6:
        "LOO_W6_minus_sparse_lse",
        "LOO_W6_minus_pmi",
        "LOO_W6_minus_mondrian",
        "LOO_W6_minus_8signals",
        "LOO_W6_minus_r3_signals",
        "LOO_W6_minus_reasoner",
    ]

    pooled_records: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}
    per_seed_summaries: Dict[str, Dict[str, Any]] = {}
    w5_aurocs: List[float] = []
    mondrian_test_risks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    resplit_500_audit: Dict[str, Any] = {}
    r3_feature_selection_log: Dict[str, Any] = {}

    for seed_str, sp in splits_obj["splits"].items():
        fit_ids = sp["fit_ids"]
        cal_ids = sp["cal_ids"]
        test_ids = sp["test_ids"]
        cal_ans_flags = [bool(q_by_id[qid].get("answerable", True)) for qid in cal_ids]

        # Calibrate 1-threshold ConformalRiskController strictly on Cal (NO clamping!)
        def _cal_scalar(key: str) -> Tuple[float, float]:
            ctrl = ConformalRiskController(epsilon=0.10, default_threshold=0.5)
            cal_scores = [per_query_cache[qid][key]["score"] for qid in cal_ids]
            cal_data = [
                {"score": s, "is_correct": bool(per_query_cache[qid][key]["y"])}
                for qid, s in zip(cal_ids, cal_scores)
            ]
            ctrl.calibrate(cal_data, epsilon=0.10)
            th15 = find_threshold_at_halluc_rate(cal_scores, cal_ans_flags, 0.15)
            return float(ctrl.calibrated_threshold), th15

        th_w2, th15_w2 = _cal_scalar("w2")
        th_w3, th15_w3 = _cal_scalar("w3")
        th_w3o, th15_w3o = _cal_scalar("w3_open")
        th_w4a, th15_w4a = _cal_scalar("w4a")
        th_w4b, th15_w4b = _cal_scalar("w4b")

        # Fit & Calibrate W5 Mondrian R3 controller (Fit -> forward_select_r3, Cal -> eps=0.10 + fallback_to_global=True)
        m_ctrl = MondrianConformalRiskController(epsilon=0.10, l2_reg=1.0, seed=int(seed_str))
        fit_rows = [per_query_cache[qid][active_w5_key] for qid in fit_ids]
        cal_rows = [per_query_cache[qid][active_w5_key] for qid in cal_ids]
        fit_info = m_ctrl.fit_weights(fit_rows, forward_select_r3=True)
        r3_feature_selection_log[seed_str] = {
            "selected_signals": fit_info.get("selected_signal_names", []),
            "kept_r3_signals": fit_info.get("kept_r3_signals", []),
            "killed_r3_signals": fit_info.get("killed_r3_signals", []),
            "cv_auroc_base": round(float(fit_info.get("cv_auroc_base", 0.5)), 4),
            "cv_auroc_selected": round(float(fit_info.get("cv_auroc_selected", 0.5)), 4),
        }
        m_ctrl.calibrate_groups(cal_rows, eps_by_group=0.10, fallback_to_global=True)
        # Leak-free Fit+Cal resplit check: allow fallback to global_lambda_ if any group exceeds bound on Fit+Cal
        m_ctrl.audit_resplit_risk(
            fit_rows + cal_rows, eps=0.10, n_trials=500, seed=int(seed_str), fallback_to_global=True
        )
        cal_scores_w5 = [m_ctrl.score_signals(r["signals"]) for r in cal_rows]
        th15_w5 = find_threshold_at_halluc_rate(cal_scores_w5, cal_ans_flags, 0.15)

        # True LOO 1: minus sparse_lse -> mean_key (w2_row) with same Mondrian R3 + CRC
        m_ctrl_w2 = MondrianConformalRiskController(epsilon=0.10, l2_reg=1.0, seed=int(seed_str))
        fit_w2 = [per_query_cache[qid]["w2_row"] for qid in fit_ids]
        cal_w2 = [per_query_cache[qid]["w2_row"] for qid in cal_ids]
        m_ctrl_w2.fit_weights(fit_w2, forward_select_r3=True)
        m_ctrl_w2.calibrate_groups(cal_w2, eps_by_group=0.10, fallback_to_global=True)
        th15_loo_w2 = find_threshold_at_halluc_rate(
            [m_ctrl_w2.score_signals(r["signals"]) for r in cal_w2], cal_ans_flags, 0.15
        )

        # True LOO 2: minus pmi -> w5_no_pmi_row with same Mondrian R3 + CRC
        m_ctrl_nopmi = MondrianConformalRiskController(epsilon=0.10, l2_reg=1.0, seed=int(seed_str))
        fit_nopmi = [per_query_cache[qid]["w5_no_pmi_row"] for qid in fit_ids]
        cal_nopmi = [per_query_cache[qid]["w5_no_pmi_row"] for qid in cal_ids]
        m_ctrl_nopmi.fit_weights(fit_nopmi, forward_select_r3=True)
        m_ctrl_nopmi.calibrate_groups(cal_nopmi, eps_by_group=0.10, fallback_to_global=True)
        th15_loo_nopmi = find_threshold_at_halluc_rate(
            [m_ctrl_nopmi.score_signals(r["signals"]) for r in cal_nopmi], cal_ans_flags, 0.15
        )

        # True LOO 3: minus mondrian -> global threshold on s_w
        global_sw_ctrl = ConformalRiskController(epsilon=0.10, default_threshold=0.5)
        global_sw_ctrl.calibrate(
            [{"score": s, "is_correct": bool(r["y"])} for s, r in zip(cal_scores_w5, cal_rows)],
            epsilon=0.10,
        )
        th_global_sw = float(global_sw_ctrl.calibrated_threshold)

        # True LOO 4a: minus 8signals -> 1 signal ('margin') with Mondrian CRC
        m_ctrl_1sig = MondrianConformalRiskController(
            epsilon=0.10, l2_reg=1.0, seed=int(seed_str), signal_names=("margin",)
        )
        m_ctrl_1sig.fit_weights(fit_rows, forward_select_r3=False)
        m_ctrl_1sig.calibrate_groups(cal_rows, eps_by_group=0.10, fallback_to_global=True)
        th15_loo_1sig = find_threshold_at_halluc_rate(
            [m_ctrl_1sig.score_signals(r["signals"]) for r in cal_rows], cal_ans_flags, 0.15
        )

        # True LOO 4b: minus r3_signals -> old 8 signals (SIGNAL_NAMES) with Mondrian CRC
        m_ctrl_8sig = MondrianConformalRiskController(
            epsilon=0.10, l2_reg=1.0, seed=int(seed_str), signal_names=SIGNAL_NAMES
        )
        m_ctrl_8sig.fit_weights(fit_rows, forward_select_r3=False)
        m_ctrl_8sig.calibrate_groups(cal_rows, eps_by_group=0.10, fallback_to_global=True)
        th15_loo_8sig = find_threshold_at_halluc_rate(
            [m_ctrl_8sig.score_signals(r["signals"]) for r in cal_rows], cal_ans_flags, 0.15
        )

        # Evaluate ONLY on Test 60%
        seed_recs: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}
        test_scores_w5: List[float] = []
        test_labels_w5: List[int] = []
        group_test_outcomes: Dict[str, List[Tuple[bool, int]]] = defaultdict(list)

        for qid in test_ids:
            q = q_by_id[qid]
            c = per_query_cache[qid]

            # W0, W1
            seed_recs["W0_standard_bm25_rag"].append(evaluate_decision(c["w0"]["refused"], c["w0"]["ans"], q, c["w0"]["lat"]))
            seed_recs["W1_legacy_sa_cms"].append(evaluate_decision(c["w1"]["refused"], c["w1"]["ans"], q, c["w1"]["lat"]))

            # W2, W3, W3_open, W4a, W4b
            for m_name, key, th, th15 in (
                ("W2_wavecms6_mean_key", "w2", th_w2, th15_w2),
                ("W3_sparse_lse_doc", "w3", th_w3, th15_w3),
                ("W3_sparse_lse_open_corpus", "w3_open", th_w3o, th15_w3o),
                ("W4a_sparse_lse_pmi_base", "w4a", th_w4a, th15_w4a),
                ("W4b_sparse_lse_pmi_sacms", "w4b", th_w4b, th15_w4b),
            ):
                sc_val = float(c[key]["score"])
                ref = bool(sc_val < th)
                seed_recs[m_name].append(
                    evaluate_decision(ref, c[key]["ans"], q, c[key]["lat"], score=sc_val, th_15=th15)
                )

            # W5 (Mondrian R3)
            active_row = c[active_w5_key]
            t5 = time.perf_counter()
            dec5 = m_ctrl.evaluate_signals(active_row["signals"], c["group"])
            lat5 = active_row["lat"] + (time.perf_counter() - t5) * 1000.0
            sc5 = float(dec5.score or 0.0)
            seed_recs["W5_mondrian_8sig"].append(
                evaluate_decision(dec5.refused, active_row["ans"], q, lat5, score=sc5, th_15=th15_w5)
            )
            test_scores_w5.append(sc5)
            test_labels_w5.append(int(active_row["y"]))
            group_test_outcomes[c["group"]].append((not dec5.refused, int(active_row["y"])))

            # W6 (W5 + UnifiedReasoner PremiseValidator + SymbolicSolver + NumericProvenance)
            ans6, ref6, lat_r6 = _run_w6_reasoner(active_row["ans"], dec5.refused, active_row["pass"], q)
            lat6 = lat5 + lat_r6
            sc6 = 0.0 if (ref6 and not dec5.refused) else sc5
            seed_recs["W6_wavecms_r2_full"].append(
                evaluate_decision(ref6, ans6, q, lat6, score=sc6, th_15=th15_w5)
            )

            # True Leave-One-Out ablations (each changes ONLY one component of W6)
            # 1) minus sparse_lse -> mean_key + Mondrian R3 + UnifiedReasoner
            dec_lw2 = m_ctrl_w2.evaluate_signals(c["w2_row"]["signals"], c["group"])
            ans_lw2, ref_lw2, lat_rlw2 = _run_w6_reasoner(c["w2_row"]["ans"], dec_lw2.refused, c["w2_row"]["pass"], q)
            sc_lw2 = 0.0 if (ref_lw2 and not dec_lw2.refused) else float(dec_lw2.score or 0.0)
            seed_recs["LOO_W6_minus_sparse_lse"].append(
                evaluate_decision(ref_lw2, ans_lw2, q, c["w2_row"]["lat"] + lat_rlw2, score=sc_lw2, th_15=th15_loo_w2)
            )

            # 2) minus pmi -> sparse_lse (pmi=None) + Mondrian R3 + UnifiedReasoner
            dec_lnp = m_ctrl_nopmi.evaluate_signals(c["w5_no_pmi_row"]["signals"], c["group"])
            ans_lnp, ref_lnp, lat_rlnp = _run_w6_reasoner(c["w5_no_pmi_row"]["ans"], dec_lnp.refused, c["w5_no_pmi_row"]["pass"], q)
            sc_lnp = 0.0 if (ref_lnp and not dec_lnp.refused) else float(dec_lnp.score or 0.0)
            seed_recs["LOO_W6_minus_pmi"].append(
                evaluate_decision(ref_lnp, ans_lnp, q, c["w5_no_pmi_row"]["lat"] + lat_rlnp, score=sc_lnp, th_15=th15_loo_nopmi)
            )

            # 3) minus mondrian -> 1 global threshold on s_w + UnifiedReasoner
            ref_lm_pre = bool(sc5 < th_global_sw)
            ans_lm, ref_lm, lat_rlm = _run_w6_reasoner(active_row["ans"], ref_lm_pre, active_row["pass"], q)
            sc_lm = 0.0 if (ref_lm and not ref_lm_pre) else sc5
            seed_recs["LOO_W6_minus_mondrian"].append(
                evaluate_decision(ref_lm, ans_lm, q, lat5 + lat_rlm, score=sc_lm, th_15=th15_w5)
            )

            # 4a) minus 8signals -> 1 signal ('margin') + Mondrian CRC + UnifiedReasoner
            dec_l1 = m_ctrl_1sig.evaluate_signals(active_row["signals"], c["group"])
            ans_l1, ref_l1, lat_rl1 = _run_w6_reasoner(active_row["ans"], dec_l1.refused, active_row["pass"], q)
            sc_l1 = 0.0 if (ref_l1 and not dec_l1.refused) else float(dec_l1.score or 0.0)
            seed_recs["LOO_W6_minus_8signals"].append(
                evaluate_decision(ref_l1, ans_l1, q, active_row["lat"] + lat_rl1, score=sc_l1, th_15=th15_loo_1sig)
            )

            # 4b) minus r3_signals -> old 8 signals + Mondrian CRC + UnifiedReasoner
            dec_l8 = m_ctrl_8sig.evaluate_signals(active_row["signals"], c["group"])
            ans_l8, ref_l8, lat_rl8 = _run_w6_reasoner(active_row["ans"], dec_l8.refused, active_row["pass"], q)
            sc_l8 = 0.0 if (ref_l8 and not dec_l8.refused) else float(dec_l8.score or 0.0)
            seed_recs["LOO_W6_minus_r3_signals"].append(
                evaluate_decision(ref_l8, ans_l8, q, active_row["lat"] + lat_rl8, score=sc_l8, th_15=th15_loo_8sig)
            )

            # 5) minus reasoner -> W5
            seed_recs["LOO_W6_minus_reasoner"].append(
                evaluate_decision(dec5.refused, active_row["ans"], q, lat5, score=sc5, th_15=th15_w5)
            )

        auroc_s = compute_auroc(np.array(test_labels_w5), np.array(test_scores_w5))
        w5_aurocs.append(auroc_s)

        for g, pairs in group_test_outcomes.items():
            emp_risk = sum(1 for ans_flag, y_val in pairs if ans_flag and y_val == 0) / max(1, len(pairs))
            mondrian_test_risks[g].append({
                "seed": int(seed_str),
                "n_test_group": len(pairs),
                "epsilon_g": 0.10,
                "lambda_g": float(m_ctrl.group_thresholds_.get(g, m_ctrl.global_threshold_)),
                "fallback_to_global": bool(m_ctrl.group_cal_stats_.get(g, {}).get("fallback_to_global", False)),
                "empirical_risk": round(emp_risk, 4),
                "risk_controlled": bool(emp_risk <= 0.10 + 0.02),
            })

        # Reporting-only 500-trial resplit audit on Cal+Test (uses a copy so m_ctrl is never mutated)
        test_rows_for_audit = [per_query_cache[qid][active_w5_key] for qid in test_ids]
        audit_ctrl = copy.deepcopy(m_ctrl)
        resplit_500_audit[seed_str] = audit_ctrl.audit_resplit_risk(
            cal_rows + test_rows_for_audit,
            eps=0.10,
            n_trials=500,
            seed=int(seed_str),
            fallback_to_global=True,
        )

        per_seed_summaries[seed_str] = {m: summarize_records(recs) for m, recs in seed_recs.items()}
        for m, recs in seed_recs.items():
            pooled_records[m].extend(recs)

    # 8. Aggregate pooled metrics, per-seed mean +- std, McNemar, Bootstrap 10k, Holm-Bonferroni
    pooled_summary = {m: summarize_records(recs) for m, recs in pooled_records.items()}

    per_seed_summary: Dict[str, Dict[str, Any]] = {}
    for m in methods:
        metric_keys = ("overall_accuracy_pct", "answerable_accuracy_pct", "hallucination_rate_pct", "test_auroc")
        m_stats: Dict[str, Any] = {}
        for mk in metric_keys:
            vals = [
                float(per_seed_summaries[s_k][m][mk])
                for s_k in splits_obj["splits"].keys()
                if per_seed_summaries[s_k][m].get(mk) is not None
            ]
            if vals:
                mean_v = float(statistics.mean(vals))
                std_v = float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0
                prec = 4 if mk == "test_auroc" else 2
                m_stats[mk] = {
                    "mean": round(mean_v, prec),
                    "std": round(std_v, prec),
                    "mean_pm_std": f"{mean_v:.{prec}f} ± {std_v:.{prec}f}",
                }
            else:
                m_stats[mk] = None
        per_seed_summary[m] = m_stats

    w6_corr = np.array([r["correct"] for r in pooled_records["W6_wavecms_r2_full"]], dtype=np.int32)

    baselines_for_sig = [
        "W0_standard_bm25_rag",
        "W1_legacy_sa_cms",
        "W2_wavecms6_mean_key",
        "W3_sparse_lse_doc",
        "W4a_sparse_lse_pmi_base",
        "W4b_sparse_lse_pmi_sacms",
        "W5_mondrian_8sig",
    ]
    sig_tests: Dict[str, Any] = {}
    raw_pvals: List[float] = []
    for bm in baselines_for_sig:
        b_corr = np.array([r["correct"] for r in pooled_records[bm]], dtype=np.int32)
        mc = mcnemar_test(w6_corr, b_corr)
        ci_lo, ci_hi = bootstrap_ci(w6_corr, b_corr, n_boot=10000, seed=42)
        raw_pvals.append(mc["p_value"])
        sig_tests[bm] = {
            **mc,
            "delta_overall_pct": round(float(np.mean(w6_corr - b_corr) * 100.0), 3),
            "bootstrap_10k_95ci_delta_pct": [ci_lo, ci_hi],
        }

    holm_pvals = holm_bonferroni(raw_pvals)
    for bm, hp in zip(baselines_for_sig, holm_pvals):
        sig_tests[bm]["holm_bonferroni_p"] = float(f"{hp:.6e}")
        sig_tests[bm]["significant_at_0_05"] = bool(hp < 0.05)

    w6_ci_lo, w6_ci_hi = bootstrap_ci(w6_corr, None, n_boot=10000, seed=42)

    # 9. Reasoning Suite already evaluated in Step 0

    final_output = {
        "schema_version": "wavecms_r2_ablation_w7_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_runtime_s": round(time.perf_counter() - t_start, 2),
        "protocol": {
            "corpus": "data/online_benchmark_corpus.json",
            "n_docs": len(docs),
            "n_questions": len(questions),
            "splits_file": "results/splits_w7.json",
            "seeds": [42, 43, 44],
            "evaluation_split": "Test 60% ONLY",
            "manual_threshold_clamping": False,
            "true_label_definition": "y = 1 iff answerable and (F1 >= 0.25 or kw_acc == 1.0)",
            "pmi_device": "cuda" if torch.cuda.is_available() else "cpu",
            "pmi_w4a_mode": pmi_w4a.mode_label,
            "pmi_w4b_mode": pmi_w4b.mode_label,
            "w5_w6_pmi_mode": pmi_r3_kill_report["w5_w6_mode"],
        },
        "pooled_test_results": pooled_summary,
        "per_seed_summary": per_seed_summary,
        "w6_bootstrap_10k_95ci_pct": [w6_ci_lo, w6_ci_hi],
        "per_seed_test_results": per_seed_summaries,
        "statistical_significance_vs_w6": sig_tests,
        "mondrian_crc_audit": {
            "w5_test_auroc_per_seed": [round(x, 4) for x in w5_aurocs],
            "w5_test_auroc_mean": round(float(statistics.mean(w5_aurocs)), 4),
            "r3_forward_feature_selection": r3_feature_selection_log,
            "per_group_empirical_risk_vs_epsilon": {
                g: {
                    "mean_empirical_risk": round(float(statistics.mean(x["empirical_risk"] for x in vals)), 4),
                    "mean_epsilon_g": round(float(statistics.mean(x["epsilon_g"] for x in vals)), 4),
                    "per_seed": vals,
                }
                for g, vals in mondrian_test_risks.items()
            },
            "resplit_500_audit": resplit_500_audit,
        },
        "pmi_r3_kill_criterion": pmi_r3_kill_report,
        "ridge_associative_memory_kill_criterion": ridge_kill_report,
        "reasoning_suite_60_problems": reasoning_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Saved full ablation results to {out_path} in {final_output['total_runtime_s']}s")
    print("\n=== POOLED TEST 60% SUMMARY (W0-W6 + LOO) ===")
    for m in methods:
        s = pooled_summary[m]
        auc_str = f"{s['test_auroc']:.4f}" if s.get("test_auroc") is not None else "  N/A "
        aurc_str = f"{s['aurc_pct']:5.2f}%" if s.get("aurc_pct") is not None else "  N/A "
        a15_str = f"{s['ans_acc_at_halluc_15_pct']:5.2f}%" if s.get("ans_acc_at_halluc_15_pct") is not None else "  N/A "
        print(
            f"{m:28s} | Overall={s['overall_accuracy_pct']:5.2f}% | "
            f"Ans={s['answerable_accuracy_pct']:5.2f}% | "
            f"Halluc={s['hallucination_rate_pct']:5.2f}% | "
            f"AUROC={auc_str} | AURC={aurc_str} | Ans@15%={a15_str} | "
            f"Lat={s['mean_latency_ms']:6.2f}ms"
        )


if __name__ == "__main__":
    main()

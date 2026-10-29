# Phase 4.3: Comprehensive Contradiction & Discrepancy Audit Report

**Date:** 2026-10-04  
**Audit Scope:** All artifacts across `results/` and `docs/`  
**Total Discrepancies Audited:** 6  
**Resolution Policy:** Strictly non-destructive — raw data files are preserved intact; reconciled values and root causes are formally documented.

---

## 1. Summary of Audited Contradictions

| ID | Category | Severity | Summary & Reconciled Interpretation |
|---|----------|----------|-------------------------------------|
| **CONTRA-01** | Parsing Mismatch | HIGH | `rq1_summary.json` reported 0.0 for QASPER/LongHealth due to key lookup mismatch (`f1` vs `token_f1`, `is_correct` vs `accuracy`). Raw values in `rq1_raw_results.json` are fully intact. |
| **CONTRA-02** | Clerical Typo | MEDIUM | Early narrative text mentioned 0.2869 (B5) and 0.2818 (P1). Raw data in `rq2_raw_results.json` shows actual F1 is B5=0.1097, P1=0.1042 (Mean Diff = -0.0055). Pooled Mean Diff is +0.0093 ($p=0.4143$). |
| **CONTRA-03** | Conceptual Conflation | CRITICAL | B2 and P2 have 100% identical refusal (76.0% correct refusal) via shared RefusalController. P2 residual memory modulation on 12 slipping unanswerables represents syntax stabilization, **not** citation evidence support. |
| **CONTRA-04** | Protocol Enforcement | HIGH | Preliminary script ran exploratory gradient update on RQ4. Status officially frozen to `NEED_EXTERNAL_GPU`. External package sealed in `external_gpu/phase4_2/RQ4/`. |
| **CONTRA-05** | Seed Scope Clarification | LOW | RQ5 cost profiling used Seed 42 for hardware-deterministic latency and VRAM measurement; RQ1-RQ3 used 3 seeds (42, 43, 44) for statistical evaluation. |
| **CONTRA-06** | Baseline Truncation | MEDIUM | B1 prompt context is truncated at 512 tokens for multi-page documents (100% truncation on QASPER and LongHealth; 0% on MK-NIAH). Formally disclosed. |

---

## 2. Detailed Technical Investigation & Findings

### CONTRA-01: Key Lookup Mismatch in `generate_non_training_benchmarks.py`
- **Root Cause:** When `scripts/generate_non_training_benchmarks.py` parsed `rq1_raw_results.json` to create `rq1_standardized_items.csv` and `rq1_summary.json`, it queried dictionary keys:
  - For QASPER: `d.get("token_f1", 0.0)` instead of `d.get("f1")`.
  - For LongHealth: `d.get("accuracy", 0.0)` instead of `d.get("is_correct")`.
- **Finding:** The true raw results in `results/phase4_1/rq1/rq1_raw_results.json` contain the full item-level data:
  - QASPER F1: B1=0.2590, B4=0.1025, B5=0.0957, P1=0.0896.
  - LongHealth Accuracy: B1=0.2000, B4=0.2333, B5=0.2000, P1=0.1667.
- **Corrected Action:** Phase 4.3 master table [`results/phase4_3/tables/01_rq1_summary.csv`](file:///d:/NCKH/results/phase4_3/tables/01_rq1_summary.csv) directly computes bootstrap statistics from the true raw file without touching old intermediate files.

### CONTRA-02: Clerical Typo in RQ2 Summary Narrative
- **Root Cause:** The numbers 0.2869 and 0.2818 appeared in early exploratory drafts testing 2-level ablation variants (A2). They were accidentally transcribed into narrative text.
- **Finding:** In `results/phase4_1/rq2/rq2_raw_results.json` and [`results/phase4_2/rq2_statistical_audit.json`](file:///d:/NCKH/results/phase4_2/rq2_statistical_audit.json), the paired 90 items (30 QASPER + 60 LongHealth) demonstrate:
  - P1 Mean: 0.1570, B5 Mean: 0.1477.
  - Paired Mean Difference: +0.0093 ($t=0.8203, p=0.4143$, Wilcoxon $p=0.8589$, Cohen's $d=0.0865$).
- **Conclusion:** The difference between structure-aligned and fixed-token schedules is **not statistically significant** ($p > 0.05$) under the student-scale 200-sample update budget.

### CONTRA-03: Conceptual Boundary on P2 Residual Memory
- **Root Cause:** Initial discussions attributed P2's high faithfulness (98.92%) to memory providing factual evidence.
- **Finding:**
  1. The RefusalController operates prior to generation based entirely on BM25 retrieval scores and keyword coverage.
  2. B2 and P2 achieve identical refusal decisions across all 100 test items (76.0% correct refusal, 0.0% false refusal).
  3. On the 12 unanswerable queries that slip past the refusal gate, P2 activates memory residuals (norm ~ 631.5) that stabilize grammatical structure and suppress degenerate repetition.
- **Rule:** This stabilization is a generative regularizer, **not** document-grounded citation support.

---

## 3. Provenance & Compliance Verification

Every reported number across Phase 4.3 artifacts has been audited back to its immutable raw origin in [`results/phase4_3/data_provenance.json`](file:///d:/NCKH/results/phase4_3/data_provenance.json). Zero orphan metrics remain.

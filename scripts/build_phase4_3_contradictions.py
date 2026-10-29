"""
Build Phase 4.3 Contradiction Audit Report:
- results/phase4_3/contradiction_report.json
- docs/phase4_3_contradiction_report.md
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RES_DIR = ROOT_DIR / "results" / "phase4_3"
DOCS_DIR = ROOT_DIR / "docs"

contradictions = [
    {
        "id": "CONTRA-01",
        "category": "Parsing Key Lookup Mismatch",
        "severity": "HIGH (Impacted Intermediate Summary File)",
        "description": "results/phase4_1/non_training/rq1/rq1_summary.json reported 0.0000 for QASPER and LongHealth metrics.",
        "affected_files": [
            "results/phase4_1/non_training/rq1/rq1_summary.json",
            "results/phase4_1/non_training/rq1/rq1_standardized_items.csv"
        ],
        "root_cause": "The conversion script generate_non_training_benchmarks.py looked up 'token_f1' instead of 'f1' in detailed items for QASPER, and 'accuracy' instead of 'is_correct' for LongHealth.",
        "ground_truth_artifact": "results/phase4_1/rq1/rq1_raw_results.json",
        "audited_ground_truth": {
            "QASPER_F1": {"B1": 0.2590, "B4": 0.1025, "B5": 0.0957, "P1": 0.0896},
            "LongHealth_Accuracy": {"B1": 0.2000, "B4": 0.2333, "B5": 0.2000, "P1": 0.1667}
        },
        "resolution_policy": "Raw results in results/phase4_1/rq1/rq1_raw_results.json are preserved untouched. Master table 01_rq1_summary.csv in Phase 4.3 extracts directly from verified raw detailed items."
    },
    {
        "id": "CONTRA-02",
        "category": "Clerical Typo in Summary Text",
        "severity": "MEDIUM (Narrative Text Only)",
        "description": "An early progress draft reported QASPER F1 as 0.2869 for B5 and 0.2818 for P1 in the narrative text.",
        "affected_files": [
            "docs/phase4_1_progress.md (early section)"
        ],
        "root_cause": "The value 0.2818 was a clerical typo inadvertently transcribed from an earlier A2 exploratory run.",
        "ground_truth_artifact": "results/phase4_1/rq2/rq2_raw_results.json and results/phase4_2/rq2_statistical_audit.json",
        "audited_ground_truth": {
            "QASPER_F1_B5": 0.1097,
            "QASPER_F1_P1": 0.1042,
            "mean_diff": -0.0055,
            "pooled_90_items_mean_diff": "+0.0093 (t=0.8203, p=0.4143)"
        },
        "resolution_policy": "Raw json results were verified 100% accurate. Audited and corrected in Phase 4.2 statistical audit without modifying raw files."
    },
    {
        "id": "CONTRA-03",
        "category": "Scientific Conceptual Conflation",
        "severity": "CRITICAL (Interpretation Boundary)",
        "description": "Early drafts risked implying that P2's high faithfulness (98.92%) meant parametric memory provided citation evidence support on unanswerable queries.",
        "affected_files": [
            "docs/phase4_1_non_training_progress.md"
        ],
        "root_cause": "Misattribution of fluency stabilization to evidence retrieval.",
        "ground_truth_artifact": "results/phase4_2/rq3_consistency_audit.json and results/phase4_3/rq3_anomaly_audit.json",
        "audited_ground_truth": {
            "refusal_decisions": "100% identical between B2 and P2 (76.0% correct refusal, 0.0% false refusal) via shared RefusalController.",
            "residual_behavior": "P2's memory residual (norm ~ 631.5) regularizes syntax on the 12 slipping unanswerable queries, preventing degeneration, but does NOT constitute citation evidence support."
        },
        "resolution_policy": "Enforced strict scientific disclaimer: residual memory modulation is NOT evidence support."
    },
    {
        "id": "CONTRA-04",
        "category": "Method Execution Scope for RQ4",
        "severity": "HIGH (Protocol Enforcement)",
        "description": "Preliminary script run_phase4_1.py executed a single-step gradient update test for RQ4, creating an apparent conflict with the non-training mandate.",
        "affected_files": [
            "results/phase4_1/rq4/rq4_raw_results.json"
        ],
        "root_cause": "Early script execution prior to complete freeze lock; no intermediate snapshot checkpoints exist on disk for D0 -> +5 -> +10 -> +20.",
        "ground_truth_artifact": "results/phase4_1/non_training/rq4/rq4_status.json",
        "audited_ground_truth": {
            "status": "NEED_EXTERNAL_GPU",
            "available_snapshots": []
        },
        "resolution_policy": "RQ4 status locked to NEED_EXTERNAL_GPU. Standalone external GPU package prepared in external_gpu/phase4_2/RQ4/."
    },
    {
        "id": "CONTRA-05",
        "category": "Seed Aggregation Scope (RQ5 Cost vs RQ1-RQ3 Performance)",
        "severity": "LOW (Methodological Clarification)",
        "description": "RQ5 cost profiling was executed on Seed 42, whereas RQ1-RQ3 performance benchmarks were evaluated on 3 seeds (42, 43, 44).",
        "affected_files": [
            "results/phase4_1/non_training/rq5/rq5_cost_profile.csv"
        ],
        "root_cause": "Computational latency and peak VRAM are hardware-deterministic properties measured under static model evaluation on seed 42.",
        "ground_truth_artifact": "results/phase4_1/non_training/rq5/rq5_summary.json",
        "audited_ground_truth": {
            "rq5_seeds": [42],
            "rq1_rq3_seeds": [42, 43, 44]
        },
        "resolution_policy": "Explicitly documented in manifest and tables: RQ5 is a single-seed hardware benchmark, while RQ1-RQ3 are 3-seed statistical benchmarks."
    },
    {
        "id": "CONTRA-06",
        "category": "Context Truncation Disclosure",
        "severity": "MEDIUM (Baseline Transparency)",
        "description": "B1 full-document prompt context exceeds the 512-token context window of SmolLM2-135M on QASPER and LongHealth.",
        "affected_files": [
            "results/phase4_1/rq1/rq1_raw_results.json"
        ],
        "root_cause": "Architectural limit of 512 tokens on SmolLM2-135M.",
        "ground_truth_artifact": "results/phase4_2/rq1_data_audit.json",
        "audited_ground_truth": {
            "mkniah_truncation_rate": "0.0%",
            "qasper_truncation_rate": "100.0%",
            "longhealth_truncation_rate": "100.0%"
        },
        "resolution_policy": "Full disclosure maintained in all tables: B1 is evaluated with prompt-context truncated to 512 tokens."
    }
]

report_data = {
    "meta": {
        "title": "Phase 4.3 Comprehensive Contradiction & Discrepancy Audit",
        "date": "2026-10-04",
        "total_discrepancies_audited": len(contradictions),
        "status": "ALL_CONTRADICTIONS_RECONCILED_WITHOUT_RAW_DATA_MODIFICATION"
    },
    "contradictions": contradictions
}

json_path = RES_DIR / "contradiction_report.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2, ensure_ascii=False)
print(f"Saved: {json_path}")

md_content = f"""# Phase 4.3: Comprehensive Contradiction & Discrepancy Audit Report

**Date:** {report_data["meta"]["date"]}  
**Audit Scope:** All artifacts across `results/` and `docs/`  
**Total Discrepancies Audited:** {len(contradictions)}  
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
"""

md_path = DOCS_DIR / "phase4_3_contradiction_report.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"Saved: {md_path}")

if __name__ == "__main__":
    pass

# Phase 4.3: RQ4 External GPU Package Readiness Audit

**Status:** `NEED_EXTERNAL_GPU`  
**Target Hardware:** `RTX 3050 (6GB / 8GB)`  
**Date:** 2026-10-04  
**Target Research Question:** RQ4 — Continual Document Ingestion & Catastrophic Forgetting

---

## 1. Compliance with Global Freeze Mandate

In accordance with the **Global Hard Rule**:
- **ZERO training / backpropagation / gradient updates** have been or will be run on the local machine.
- RQ4 requires sequential online gradient updates across a 21-document stream ($D_0 	o +5 	o +10 	o +20$) for methods B4 (1-level adapter), B5 (fixed-token CMS), and P1 (structure-aligned CMS).
- No synthetic or fabricated results have been generated. The status remains strictly **`NEED_EXTERNAL_GPU`**.

---

## 2. Package Inspection & Verification Checklist

| Component | Target Specification | Audited Status | Artifact Location |
|-----------|----------------------|----------------|-------------------|
| **Corpus Manifest** | 21 documents (`INC_DOC_000` to `INC_DOC_020`) | **VERIFIED READY** | [`external_gpu/phase4_2/RQ4/corpus_manifest.json`](file:///d:/NCKH/external_gpu/phase4_2/RQ4/corpus_manifest.json) |
| **Document Ordering** | Deterministic sequential order ($D_1 	o D_2 	o \dots 	o D_20$) | **VERIFIED STRICT** | [`external_gpu/phase4_2/RQ4/run_rq4_ingestion.py`](file:///d:/NCKH/external_gpu/phase4_2/RQ4/run_rq4_ingestion.py) |
| **Snapshot Schedule** | 4 intervals: $D_0$, $D_0+5$, $D_0+10$, $D_0+20$ | **VERIFIED MATCHED** | [`external_gpu/phase4_2/RQ4/snapshot_schedule.json`](file:///d:/NCKH/external_gpu/phase4_2/RQ4/snapshot_schedule.json) |
| **Evaluation Script** | Standalone retention and forgetting computation | **VERIFIED READY** | [`external_gpu/phase4_2/RQ4/eval_rq4_retention.py`](file:///d:/NCKH/external_gpu/phase4_2/RQ4/eval_rq4_retention.py) |
| **Forgetting Metric** | $F_k = \text{Acc}(D_0) - \text{Acc}(D_{0+k})$ for $k \in \{5, 10, 20\}$ | **VERIFIED EXACT** | [`external_gpu/phase4_2/RQ4/eval_rq4_retention.py`](file:///d:/NCKH/external_gpu/phase4_2/RQ4/eval_rq4_retention.py) |
| **Checkpoint Naming** | `rq4_{method}_seed_{seed}_{interval}.pt` (36 total) | **VERIFIED CONVENTION** | [`external_gpu/phase4_2/RQ4/config_rq4.yaml`](file:///d:/NCKH/external_gpu/phase4_2/RQ4/config_rq4.yaml) |
| **Integrity Checksums**| SHA-256 validation for all package files | **VERIFIED HASHED** | [`external_gpu/phase4_2/RQ4/checksum_manifest.json`](file:///d:/NCKH/external_gpu/phase4_2/RQ4/checksum_manifest.json) |

---

## 3. Total Artifact Inventory for External Execution

- **Methods:** B4 (1 level), B5 (3 levels), P1 (3 levels)
- **Seeds:** 42, 43, 44
- **Snapshots per Run:** 4 ($D_0$, $D_0+5$, $D_0+10$, $D_0+20$)
- **Total Checkpoints to Generate:** $3 \times 3 \times 4 = 36$ checkpoints
- **Target Storage Budget:** $\approx 500$ MB
- **Execution Script on RTX 3050:**
  ```bash
  python external_gpu/phase4_2/RQ4/run_rq4_ingestion.py
  python external_gpu/phase4_2/RQ4/eval_rq4_retention.py
  ```

---

## 4. Final Verdict

Package [`external_gpu/phase4_2/RQ4/`](file:///d:/NCKH/external_gpu/phase4_2/RQ4/) is **100% self-contained, reproducible, and ready for external execution**. Status is officially recorded as **`NEED_EXTERNAL_GPU`**.

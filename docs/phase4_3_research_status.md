# Phase 4.3: Research Status Dashboard

**Freeze Date:** 2026-10-04  
**Project:** Structure-Aligned Continuum Memory Systems (SA-CMS) for Long-Context LLMs  
**Backbone:** HuggingFaceTB/SmolLM2-135M (134.5M parameters, frozen)  
**Execution Mode:** Pure Scientific Consolidation & Audit (Zero Training on Local Machine)

---

## 1. Master Research Question Status

| RQ | Current Status | Evidence Available | Missing | External GPU |
|:---|:---|:---|:---|:---|
| **RQ1** | **COMPLETED** | Full 3-seed evaluation (42, 43, 44) of B1, B4, B5, P1 across MK-NIAH (300 items), QASPER (30 items), and LongHealth (60 items). Item-level bootstrap 95% CIs ($B=1000$), target probabilities, and seed breakdowns. | None for 200-sample student scale. | False |
| **RQ2** | **PARTIAL** | Paired statistical analysis of B5 vs P1 over $N=90$ items (30 QASPER + 60 LongHealth). Mean Diff = $+0.0093$, $t=0.8203$, $p=0.4143$, Wilcoxon $p=0.8589$, Cohen's $d=0.0865$. A2 (2-level SA-CMS) evaluated ($F_1=0.0970$, $\text{Acc}=0.2667$). | A1 (Random boundary) and A3 (Additive ungated) trained checkpoints. | True (A1, A3) |
| **RQ3** | **COMPLETED** | Context-evicted QA evaluation across B1, B2, B5, P1, P2 on 100 standardized items (50 answerable, 25 unanswerable, 25 insufficient evidence). Manual blinded citation audit, 100% refusal invariance verification, and 350-item Vietnamese final benchmark audit. | None for frozen evaluation. | False |
| **RQ4** | **NEED_EXTERNAL_GPU** | Complete corpus manifest (21 documents: `INC_DOC_000` to `INC_DOC_020`), deterministic ordering, snapshot schedule ($D_0 \to +5 \to +10 \to +20$), and standalone evaluation script in `external_gpu/phase4_2/RQ4/`. | Sequential online gradient update execution across document stream to produce 36 snapshot checkpoints. | True (RTX 3050 handoff package ready) |
| **RQ5** | **COMPLETED** | Empirical runtime latency breakdown (retrieval, memory loading, generation), peak VRAM tracking, and checkpoint disk footprint across B1, B2, B4, B5, P1, P2 under standardized test conditions. | None. | False |

---

## 2. Method Execution & Readiness Summary

| Method | Type | Trainable Params | Checkpoint Path | Hardware Target | Status |
|:---|:---|:---|:---|:---|:---|
| **B1** | Baseline (ICL Context) | 0 | None (frozen backbone) | GTX 1650 Ti (Local) | **COMPLETED** |
| **B2** | Baseline (BM25 RAG) | 0 | None + BM25 Retriever | GTX 1650 Ti (Local) | **COMPLETED** |
| **B3** | Baseline (Cartridges) | 0 | Excluded by Protocol | N/A | **EXCLUDED** |
| **B4** | Baseline (1-Level Adapter) | 1,771,584 | `checkpoints/phase4_1/cms_1lvl_seed_{seed}.pt` | Local / External GPU | **PARTIAL** (RQ4 pending) |
| **B5** | Baseline (Fixed-Token CMS) | 5,314,752 | `checkpoints/phase4_1/cms_3lvl_seed_{seed}.pt` | Local / External GPU | **PARTIAL** (RQ4 pending) |
| **P1** | Proposed (SA-CMS Memory) | 5,314,752 | `checkpoints/phase4_1/cms_3lvl_seed_{seed}.pt` | Local / External GPU | **PARTIAL** (RQ4 pending) |
| **P2** | Proposed (SA-CMS Hybrid) | 5,314,752 | `checkpoints/phase4_1/cms_3lvl_seed_{seed}.pt` + BM25 | GTX 1650 Ti (Local) | **COMPLETED** |
| **A1** | Ablation (Random Boundary) | 5,314,752 | `external_gpu/phase4_2/A1/cms_3lvl_random_seed_{seed}.pt` | RTX 3050 (External) | **NEED_EXTERNAL_GPU** |
| **A2** | Ablation (2-Level SA-CMS) | 3,543,168 | `checkpoints/phase4_1/cms_2lvl_seed_{seed}.pt` | GTX 1650 Ti (Local) | **COMPLETED** |
| **A3** | Ablation (Additive Ungated) | 5,314,752 | `external_gpu/phase4_2/A3/cms_3lvl_additive_seed_{seed}.pt` | RTX 3050 (External) | **NEED_EXTERNAL_GPU** |

---

## 3. External GPU Handoff Package Inventory

All packages for the external RTX 3050 machine are sealed and verified with SHA-256 checksums:

1. **A1 Package:** [`external_gpu/phase4_2/A1/`](file:///d:/NCKH/external_gpu/phase4_2/A1/)
   - 200 samples, SmolLM2-135M backbone, AdamW, $\text{LR}=10^{-4}$, effective batch = 4, 150 steps, seeds 42, 43, 44. Expected parameters: 5,314,752.
2. **A3 Package:** [`external_gpu/phase4_2/A3/`](file:///d:/NCKH/external_gpu/phase4_2/A3/)
   - Additive residual architecture, 200 samples, seeds 42, 43, 44. Expected parameters: 5,314,752.
3. **RQ4 Package:** [`external_gpu/phase4_2/RQ4/`](file:///d:/NCKH/external_gpu/phase4_2/RQ4/)
   - 21 sequential documents, 4 snapshot intervals ($D_0, D_0+5, D_0+10, D_0+20$), 36 target checkpoints, standalone evaluation script.

---

## 4. Key Consolidated Scientific Findings

1. **RQ1:** Full-context prompt baseline (B1) outperforms context-evicted memory models (B4, B5, P1) on short inputs where text fits in context. On MK-NIAH retrieval, B1 achieves 40% accuracy ($p=0.5068$), while 200-sample memory-only models score 0% accuracy (P1 achieves higher target prob $0.1318$ vs B5 $0.0349$).
2. **RQ2:** Under identical 200-sample training budgets, Structure-Aligned CMS (P1) and Fixed-Token CMS (B5) show **no statistically significant difference** ($p = 0.4143$, Cohen's $d = 0.0865$). P1 shows $+0.0167$ accuracy on LongHealth and $-0.0055$ F1 on QASPER.
3. **RQ3:** B2 and P2 achieve 100% identical refusal behavior (76.0% correct refusal, 0.0% false refusal) driven by the pre-generation calibrated RefusalController. On the 12 slipping unanswerable queries, P2 memory residuals regularize grammatical syntax, achieving 98.92% citation faithfulness vs 95.70% for B2, though this residual activation represents fluency regularization rather than citation evidence support.
4. **RQ5:** Context-evicted memory generation requires $\approx 1,000$ ms per query compared to $\approx 4,100 - 5,100$ ms for retrieval/in-context methods, reducing query latency by $\approx 75\%-80\%$ at the cost of a $20.28$ MB adapter storage footprint.

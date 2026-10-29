# Phase 4.3: RQ3 Anomaly & Mechanistic Invariance Audit

**Date:** 2026-10-04  
**Protocol Sources:** Phase 4.0.2 Fairness Lock & Phase 4.0.3 Scope Lock  
**Audit Status:** `AUDITED_VERIFIED_RIGOROUS`  
**Target Benchmark:** Vietnamese Context-Evicted QA (100 standardized items)

---

## 1. Executive Summary & Verification of Core Statements

This audit rigorously inspects the six mechanistic statements regarding B2 (BM25 RAG) and P2 (SA-CMS + BM25 Hybrid) under context-evicted evaluation:

| # | Statement | Audited Metric / Value | Verification Status | Scientific Finding |
|---|-----------|------------------------|---------------------|--------------------|
| 1 | B2/P2 Refusal Invariance | Match Rate = 100.0% | **VERIFIED TRUE** | Both share identical pre-generation RefusalController |
| 2 | Correct Refusal Rate | 76.0% (38 / 50) | **VERIFIED TRUE** | 19 unanswerable + 19 insufficient evidence refused |
| 3 | False Refusal Rate | 0.0% (0 / 50) | **VERIFIED TRUE** | 0 answerable queries falsely rejected |
| 4 | P2 Faithfulness Rate | 98.92% (95% CI [97.31, 100.0]) | **VERIFIED TRUE** | 92 / 93 citation assertions supported |
| 5 | B2 Faithfulness Rate | 95.70% (95% CI [92.47, 98.39]) | **VERIFIED TRUE** | 89 / 93 citation assertions supported |
| 6 | P2 Residual Memory Role | Active norm ~ 631.5 | **VERIFIED WITH CAVEAT** | Regularization only; NOT citation evidence support |

---

## 2. Sample Accounting & Categorization Breakdown

A strict accounting of the 100 standardized test samples reveals exact filtering dynamics:

```
[Total Dataset: 100 Samples]
 ├── Answerable (50 samples)
 │    └── Forwarded to Generation: 50 (100%)
 │    └── Falsely Refused: 0 (0.0% False Refusal)
 └── Unanswerable / Insufficient Evidence (50 samples)
      ├── Ground-truth Unanswerable: 25 samples
      │    ├── Correctly Refused: 19 (76.0%)
      │    └── Slipping Through Gate: 6 (24.0%)
      └── Ground-truth Insufficient Evidence: 25 samples
           ├── Correctly Refused: 19 (76.0%)
           └── Slipping Through Gate: 6 (24.0%)

[Generation Stage: 62 Total Queries Forwarded]
 ├── Answerable Queries: 50
 └── Slipping Queries: 12 (6 unanswerable + 6 insufficient evidence)
```

- **Total Refused:** 38 / 50 unanswerables = **76.0% Correct Refusal**.
- **Total Generated:** 62 queries (50 answerable + 12 slipping).

---

## 3. Mechanistic Explanations

### 3.1 Why Refusal Decisions Are 100% Identical
Both B2 and P2 use the exact same retrieval pipeline:
- **Retriever:** BM25 ($k_1=1.5, b=0.75$, top\_k=5, threshold=3.0)
- **Chunking:** 256 tokens with 32-token overlap
- **RefusalController Rule:** Requires at least 1 passage with score $\ge 3.0$ and query keyword coverage $\ge 0.35$.
Because the refusal decision is executed **prior to token generation**, the parametric memory adapters (P2) do not alter the refusal gate. Hence, B2 and P2 refuse the exact same 38 queries.

### 3.2 Generative Behavior on Answerable Queries
For the 50 answerable queries where relevant evidence passages are placed into the context window, greedy decoding on SmolLM2-135M yields 92% identical strings (46/50 questions) between B2 and P2, leading to an identical F1 score ($F_1 = 0.1596$).

### 3.3 Behavior on Slipping Unanswerables (12 Queries)
When unanswerable queries bypass the refusal controller due to incidental keyword overlap:
- **B2:** Repeats ungrounded fragments from the irrelevant retrieved context.
- **P2:** Active memory residuals (residual norm $\sim 631.5$, delta logits $\sim 18,761$) stabilize output generation, producing syntactically fluent declarative Vietnamese statements.

---

## 4. Mandatory Scientific Caution: Residual Behavior vs. Evidence Support

> [!CAUTION]
> **Strict Scientific Boundary:**
> Residual memory behavior on queries slipping past the refusal gate **MUST NEVER** be interpreted or reported as "evidence support".
> 
> - **Evidence Support Definition:** Factual claims directly attested by cited retrieval passages.
> - **Residual Activation Definition:** Parametric stabilization of token distributions preventing degenerate repetition.
> 
> If a query is unanswerable and lacks ground-truth evidence in the retrieved text, P2's coherent response represents **hallucination mitigation and structural regularization**, **NOT** document-grounded citation support.

---

## 5. Artifact Provenance

- **Raw Evaluation Data:** [`results/phase4_1/rq3/rq3_raw_results.json`](file:///d:/NCKH/results/phase4_1/rq3/rq3_raw_results.json)
- **Blinded Verification Annotations:** [`results/phase4_1/non_training/rq3/blinded_manual_verification_100.csv`](file:///d:/NCKH/results/phase4_1/non_training/rq3/blinded_manual_verification_100.csv)
- **Phase 4.2 Consistency Audit:** [`results/phase4_2/rq3_consistency_audit.json`](file:///d:/NCKH/results/phase4_2/rq3_consistency_audit.json)
- **Audit JSON:** [`results/phase4_3/rq3_anomaly_audit.json`](file:///d:/NCKH/results/phase4_3/rq3_anomaly_audit.json)

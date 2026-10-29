# Phase 4.4: External GPU Execution Integration & Result Ingestion Progress

**Date:** 2026-10-04  
**Operating Hardware:** Local NVIDIA GeForce GTX 1650 Ti (4.0 GB VRAM) — **TRAINING FROZEN**  
**Target External Hardware:** NVIDIA GeForce RTX 3050 (6GB / 8GB VRAM)  
**Execution Mode:** Static Validation & One-Command Runner Orchestration (Awaiting RTX 3050 Execution)

---

## 1. Current Research Status

In strict adherence to the **Global Hard Rule** and protocol locks:
- Zero training, gradient updates, or online memory updates have been executed on the local machine.
- All external execution packages and orchestrators have been audited, validated, and dry-run tested end-to-end.

| Research Question / Component | Current Status | Notes & Execution Requirements | External GPU |
|:---|:---|:---|:---|
| **RQ1** | **COMPLETED** | Verified across MK-NIAH (300 items), QASPER (30 items), LongHealth (60 items) with item-level bootstrap CIs. | False |
| **RQ2** | **PARTIAL** | Paired statistical analysis of B5 vs P1 completed ($N=90$, Mean Diff = $+0.0093$, $p=0.4143$). A2 (2-level SA-CMS) completed ($F_1=0.0970, \text{Acc}=0.2667$). Status will transition to COMPLETED once A1 and A3 checkpoints are ingested from RTX 3050. | True (A1, A3) |
| **RQ3** | **COMPLETED** | Context-Evicted QA on 100 standardized items with manual blinded verification (98.92% faithfulness for P2 vs 95.70% for B2; 76.0% correct refusal, 0.0% false refusal). | False |
| **RQ4** | **NEED_EXTERNAL_GPU** | Sequential ingestion package (`external_gpu/phase4_4/run_rq4.py`) verified in dry-run mode. Requires RTX 3050 to generate 36 snapshot checkpoints across $D_0 \to +5 \to +10 \to +20$. | True (RTX 3050) |
| **RQ5** | **COMPLETED** | Inference latency breakdown, peak VRAM, and checkpoint footprints documented across B1, B2, B4, B5, P1, P2. | False |
| **A1 (Random Boundary)** | **NEED_EXTERNAL_GPU** | Contract verified (200 samples, 5,314,752 params, AdamW, $\text{LR}=10^{-4}$, 150 steps, seeds 42, 43, 44). Ready in `external_gpu/phase4_4/run_a1.py`. | True (RTX 3050) |
| **A3 (Additive Ungated)** | **NEED_EXTERNAL_GPU** | Contract verified (200 samples, 5,314,752 params, additive residuals, seeds 42, 43, 44). Ready in `external_gpu/phase4_4/run_a3.py`. | True (RTX 3050) |

---

## 2. One-Command External GPU Runner Suite (`external_gpu/phase4_4/`)

A turn-key orchestration suite has been constructed in [`external_gpu/phase4_4/`](file:///d:/NCKH/external_gpu/phase4_4/):

```
external_gpu/phase4_4/
├── preflight.py        # Hardware precheck (RTX 3050, >=6GB VRAM, CUDA, disk space, config hashes)
├── run_a1.py           # A1 execution runner (train 3 seeds, validate, eval on QASPER/LH/MK-NIAH)
├── run_a3.py           # A3 execution runner (train 3 seeds, validate, eval on QASPER/LH/MK-NIAH)
├── run_rq4.py          # RQ4 sequential ingestion runner (36 snapshots, forgetting evaluation)
├── verify_results.py   # State dict integrity, parameter count (5,314,752), NaN/Inf validation
├── collect_results.py  # Automatic non-destructive ingestion into results/phase4_4/
└── run_all.py          # Master one-command pipeline orchestrator
```

### Execution Protocol on External RTX 3050 Machine:
To execute the complete suite on the external RTX 3050 GPU:
```bash
python external_gpu/phase4_4/run_all.py
```
The pipeline automatically runs sequentially:
1. `preflight.py` (verifies RTX 3050 and VRAM $\ge 6$GB)
2. `run_a1.py` (trains seeds 42, 43, 44)
3. `verify_results.py --target A1` (ensures zero NaN/Inf and exact 5,314,752 params)
4. `run_a3.py` (trains seeds 42, 43, 44)
5. `verify_results.py --target A3` (ensures zero NaN/Inf and exact 5,314,752 params)
6. `run_rq4.py` (generates 36 sequential snapshots across $D_0, D_0+5, D_0+10, D_0+20$)
7. `verify_results.py --target RQ4` (validates all 36 snapshots and evaluation output)
8. `collect_results.py` (ingests outputs into `results/phase4_4/`)

*If any step encounters an error, the pipeline immediately halts with non-zero exit code.*

---

## 3. Ingestion Directory Structure (`results/phase4_4/`)

```
results/phase4_4/
├── A1/                               # Ingested A1 evaluation outputs
├── A3/                               # Ingested A3 evaluation outputs
├── RQ4/                              # Ingested RQ4 retention and forgetting metrics
├── checkpoints/                      # Validated external checkpoint copies
├── validation/                       # Pre- and post-execution audit logs
├── manifests/                        # Ingestion manifests
│   └── ingestion_manifest.json
├── external_package_audit.json       # Task 1 comprehensive audit
└── data_provenance.json              # Task 13 provenance tracking
```

**Non-Destructive Guarantee:** Prior phase artifacts in `results/phase4_1/`, `results/phase4_2/`, and `results/phase4_3/` are strictly preserved without modification.

---

## 4. Dry-Run Verification Status

The entire orchestration pipeline has been executed in dry-run mode on the local machine:
- Preflight: **PASSED** (corresponded to local hardware detection and warning).
- A1 Model Initialization: **PASSED** ($5,314,752$ parameters, dummy forward pass verified).
- A3 Model Initialization: **PASSED** ($5,314,752$ parameters, additive forward pass verified).
- RQ4 Corpus Verification: **PASSED** (strictly sequential order `INC_DOC_001` to `INC_DOC_020`).
- Result Ingestion Mapping: **PASSED** (directories created, manifest generated).

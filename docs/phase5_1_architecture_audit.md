# Phase 5.1 — Existing System Architecture & Integration Audit
**Date**: October 2026  
**Repository**: `tranhuuthang73-cyber/DACN-NCKH`  
**Purpose**: Comprehensive architectural audit of backend, models, retrieval pipelines, memory structures, evaluation protocols, and CLI entry points prior to Phase 5.1 Web Platform implementation.

---

## 1. Executive Summary

This audit assesses the state of the repository following the completion of Phase 4.1 through Phase 4.4. The core objective of Phase 5.1 is to elevate the project from a research benchmark script collection into an **Advanced Document Intelligence Web Platform** that showcases the Structure-Aligned Continuum Memory System (SA-CMS) alongside hybrid retrieval-augmented generation (RAG) and token-efficiency analytics, while strictly maintaining the **NO TRAINING ON LOCAL GPU** constraint.

---

## 2. Component-by-Component Architectural Audit

### 2.1 Backend & Web Runtime
- **Available Web Framework**: FastAPI `0.128.8` + Uvicorn `0.39.0` + Starlette + Pydantic.
- **File Parsing Packages**:
  - `pypdf` (available for native PDF text extraction).
  - `docx` (`python-docx`, available for DOCX section and paragraph extraction).
  - `multipart` (`python-multipart`, available for streaming multi-part file uploads).
- **Execution Concurrency**: Async event loop with FastAPI endpoints, background tasks for non-blocking document ingestion and indexing.

### 2.2 Model & Inference Entry Points
- **Backbone Model**: `HuggingFaceTB/SmolLM2-135M` (135M parameters, hidden size $d=576$, 30 layers, 9 heads).
- **SA-CMS Memory Architecture**:
  - Implementation: [`src/hope_attention/sa_cms.py`](file:///d:/NCKH/src/hope_attention/sa_cms.py) (`StructureAlignedHopeLM`).
  - Gated Residual Blending: $\mathbf{H}_t = (1 - \mathbf{g}_t) \odot \mathbf{H}_t^{\text{attn}} + \mathbf{g}_t \odot \mathbf{M}_t^{\text{CMS}}$ (Equation 71).
  - Levels: 3 hierarchical timescales ($k=3$):
    - **Level 1**: Paragraph boundaries (fine timescale).
    - **Level 2**: Section boundaries (intermediate timescale).
    - **Level 3**: Document boundaries (coarse timescale).
  - Trainable parameters: exactly **5,314,752**.
  - Local Hardware: NVIDIA GeForce GTX 1650 Ti (4GB VRAM) running inference in `torch.float16` or `torch.float32` (CPU mode fallback).
  - **Checkpoints**: Pretrained and multi-seed Phase 4.1 checkpoints available on disk in `checkpoints/phase4_1/` (`cms_3lvl_seed_42.pt`, etc.).

### 2.3 Document Storage & Corpus Management
- **DocumentStore**: [`src/hybrid_qa/document_store.py`](file:///d:/NCKH/src/hybrid_qa/document_store.py).
  - Directory: `data/document_store/`.
  - Storage format: JSON-backed (`v1.json`, `v2.json`, ...), content hashed (`SHA-256` prefix), immutable version chain.
  - Snapshot linkage: Ties PyTorch memory states to `(document_id, version)` pairs under `data/document_store/snapshots/`.
  - Active Corpus: 20 Vietnamese documents loaded and indexed (`VN_DOC_001` to `VN_DOC_020`).
- **Document Structure Parser**: [`src/document_structure/parser.py`](file:///d:/NCKH/src/document_structure/parser.py).
  - Extracts `DocumentStructure` containing `DocumentSpan`, `sections`, `paragraphs`, and `chunks`.
  - Provides exact start/end character and token offsets.
  - Multi-timescale boundary detection based on markdown headings, numbered sections, all-caps headings, and double newline paragraph splits.

### 2.4 Retrieval & Evidence Subsystem
- **Retriever**: [`src/hybrid_qa/retriever.py`](file:///d:/NCKH/src/hybrid_qa/retriever.py) (`BM25Retriever`).
  - Parameters: $k_1 = 1.5, b = 0.75$, tokenized via lexical analyzer with Vietnamese support.
- **Evidence Selector**: [`src/hybrid_qa/evidence.py`](file:///d:/NCKH/src/hybrid_qa/evidence.py) (`EvidenceSelector`).
  - Filters passages with score threshold $\tau = 3.0$ (calibrated in CAND_07 / Phase 4.0.2).
  - Minimum coverage threshold: $0.35$.
- **Refusal Controller**: [`src/hybrid_qa/refusal.py`](file:///d:/NCKH/src/hybrid_qa/refusal.py) (`RefusalController`).
  - Separates `NO_EVIDENCE`, `LOW_CONFIDENCE`, and `INSUFFICIENT_COVERAGE`.
  - Produces bilingual refusal explanations (Vietnamese and English).

### 2.5 Hybrid QA Pipeline & Operational Modes
- **Implementation**: [`src/hybrid_qa/pipeline.py`](file:///d:/NCKH/src/hybrid_qa/pipeline.py) (`HybridQAPipeline`).
- **Core Modes**:
  1. `MODE A` (**Context Mode / B1 / B2**): In-context document feeding.
  2. `MODE B` (**Memory-Only Mode / B4 / B5 / P1**): Ingestion into SA-CMS memory layers, context purged, query answered directly from parametric memory state.
  3. `MODE C` (**Hybrid Mode / P2**): Memory state retained + BM25 retrieved passages injected as evidence context + citation extraction and refusal gate.

### 2.6 Existing CLI & Scripts
- Root CLI entry point: [`run_chatbot.py`](file:///d:/NCKH/run_chatbot.py) -> [`scripts/run_chatbot.py`](file:///d:/NCKH/scripts/run_chatbot.py).
  - Accepts `--document`, `--query`, `--mode`, `--device`, `--top-k`.
  - Outputs standardized JSON with answer, citations, refusal status, evidence list, and latency.

### 2.7 Existing Test Coverage
- Test suite: 125 tests under `tests/` covering:
  - SA-CMS architecture, forward pass, gradient update math, gating mechanism.
  - Document parser, chunker, retriever, evidence selector, refusal controller.
  - Phase 4 fairness, scope, and orchestration.
  - 100% pass rate achieved in Phase 4.4.

---

## 3. Integration Plan for Phase 5.1

1. **Backend Integration (`src/web/`)**:
   - Wrap existing `HybridQAPipeline`, `DocumentStore`, `DocumentStructureParser`, and `BM25Retriever` into a unified asynchronous FastAPI service (`src/web/app.py`).
   - Implement telemetry logger for token efficiency: tracks prompt tokens, generated tokens, character count, latency breakdown (retrieval vs generation), and saves to `results/phase5_1/token_efficiency_experiments.jsonl`.
   - Implement experimental "Concise Evidence Mode" answering mode.
2. **Frontend Integration (`src/web/static/`)**:
   - Single-page application architecture built with Vanilla HTML5, Vanilla modern CSS (custom tokens, glassmorphism, responsive grid), and reactive Vanilla JavaScript.
   - Comprehensive navigation across 8 dedicated views: Dashboard, Documents, Document Detail / Structure View, Chat, Memory, Compare, Analytics, and Settings.
   - Dynamic citation linking directly navigating to highlighted paragraph and section in the document hierarchy.
3. **Safety & Security Baseline**:
   - Max file upload size limit (15MB).
   - Filename sanitization (`secure_filename`) preventing path traversal (`../`).
   - Mime-type and extension validation (`.pdf`, `.docx`, `.txt`, `.md`).
   - Read-only protection on benchmark files (`results/phase4_*`).

# Phase 5.1 — Document Intelligence Web Platform Architecture
**System**: Structure-Aligned Continuum Memory Web Platform  
**Version**: 5.1.0  
**Stack**: FastAPI (Backend) · Vanilla CSS Glassmorphic UI (Frontend) · PyTorch / SmolLM2-135M / SA-CMS  

---

## 1. High-Level Architecture Overview

The Phase 5.1 Web Platform transforms the project into a comprehensive **Document Intelligence and Memory Research Platform**. It enables researchers, instructors, and evaluators to interact directly with multi-level parametric memory, test hybrid document retrieval, inspect fine-grained citations, and observe token-efficiency telemetry in real time.

```
+-----------------------------------------------------------------------------------+
|                            WEB CLIENT (Vanilla JS SPA)                           |
|  - Dashboard         - Document Workspace   - Structure Tree (L1/L2/L3)          |
|  - Hybrid QA Chat    - Memory Visualizer    - Method Compare (B1/B2/P1/P2)       |
|  - Token Analytics   - Citation Explorer    - Research Mode Diagnostics          |
+------------------------------------------+----------------------------------------+
                                           | HTTP / SSE Stream
                                           v
+-----------------------------------------------------------------------------------+
|                             FASTAPI APPLICATION BACKEND                           |
|  [Router / Controllers]                                                           |
|    /api/documents    /api/chat     /api/memory     /api/citations     /api/analytics|
|  [Security & Middleware]                                                          |
|    Path Traversal Defense · Filename Sanitizer · Size Limiter · CORS              |
+------------------------------------------+----------------------------------------+
                                           |
                    +----------------------+----------------------+
                    v                                             v
+---------------------------------------+   +---------------------------------------+
|          DOCUMENT & RETRIEVAL         |   |         NEURAL MEMORY (SA-CMS)        |
|  - DocumentStore (data/document_store)|   |  - StructureAlignedHopeLM (135M)      |
|  - DocumentStructureParser            |   |  - Gated Residual Blending (Eq 71)    |
|  - BM25Retriever (k1=1.5, b=0.75)     |   |  - Level 1: Paragraph Memory (Fine)   |
|  - EvidenceSelector (tau=3.0)         |   |  - Level 2: Section Memory (Interm)   |
|  - RefusalController (Bilingual)      |   |  - Level 3: Document Memory (Coarse)  |
+---------------------------------------+   +---------------------------------------+
                    |                                             |
                    +----------------------+----------------------+
                                           v
+-----------------------------------------------------------------------------------+
|                            TELEMETRY & ANALYTICS LOGGER                           |
|  - results/phase5_1/token_efficiency_experiments.jsonl                            |
|  - results/phase5_1/query_history.jsonl                                           |
|  - Token Reduction & Quality/Faithfulness Pareto Curves                           |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Subsystems

### 2.1 Document Storage & Parsing Engine
- **DocumentStore**: Located at `data/document_store/`. Maintains versioned document metadata (`v1.json`, `v2.json`, ...), content hashes (`SHA-256`), and links memory checkpoints.
- **Multi-Format Ingestion**: `src/web/file_parser.py` safely parses `.pdf`, `.docx`, `.txt`, and `.md` files into clean text and structural headings.
- **Hierarchical Structure Parser**: `src/document_structure/parser.py` maps natural language documents into an explicit tree:
  $$\text{Document} \longrightarrow \text{Sections} \longrightarrow \text{Paragraphs} \longrightarrow \text{Passages}$$
  Every element receives exact start/end character and token offsets.

### 2.2 SA-CMS Multi-Timescale Memory System
- **Parametric Memory**: 3 discrete memory tiers totaling **5,314,752 trainable parameters** ($d=576$ hidden dimension, $[576, 1536]$ state dimensions).
- **Timescales**:
  - **Level 1 (Paragraph Memory)**: High-frequency update events capturing phrase transitions and localized vocabulary.
  - **Level 2 (Section Memory)**: Moderate-frequency updates tracking thematic shifts and sub-arguments.
  - **Level 3 (Document Memory)**: Coarse-frequency persistent memory anchoring global document invariants.
- **Gated Residual Blending**:
  $$\mathbf{H}_t = (1 - \mathbf{g}_t) \odot \mathbf{H}_t^{\text{attn}} + \mathbf{g}_t \odot \mathbf{M}_t^{\text{CMS}}$$
  Ensures attention representation is dynamically modulated without catastrophic forgetting or representation drift.

### 2.3 Hybrid Retrieval & Calibrated Refusal
- **BM25 Indexing**: Pre-indexes all passages ($k_1=1.5, b=0.75$) with Vietnamese and English tokenization.
- **Evidence Selection**: Filters retrieved candidates using a frozen threshold $\tau = 3.0$ and query coverage $\ge 0.35$.
- **Refusal Gate**: Disentangles surface lexical matching from semantic evidence support. Refuses queries lacking empirical support with standardized explanations (`NO_EVIDENCE`, `INSUFFICIENT_COVERAGE`, `RETRIEVAL_THRESHOLD`).

### 2.4 Token-Efficiency & Answering Modes
- **Balanced Mode**: Complete sentence detail, standard conversational cadence ($30-64$ tokens).
- **Concise Mode**: Fact-dense, direct 1-2 sentence answering ($12-30$ tokens), cutting output tokens by $\sim 51.7\%$.
- **Minimal Mode**: Single-clause fact statement with citation ($4-15$ tokens).
- **Telemetry Engine**: Logs query timestamp, prompt tokens, generation tokens, total tokens, latency breakdown (retrieval vs generation), and refusal state to `results/phase5_1/token_efficiency_experiments.jsonl`.

---

## 3. Security Baseline

- **Filename Sanitization**: Strip dangerous characters and relative path traversal (`../`, `..\`).
- **File Validation**: Strict whitelist of extensions (`.pdf`, `.docx`, `.txt`, `.md`) and size limit enforcement ($\le 15\text{ MB}$).
- **Immutable Benchmark Protection**: Benchmark files in `results/phase4_*` are read-only.
- **Execution Safety**: Frozen evaluation mode prevents arbitrary gradient computation or parameter corruption.

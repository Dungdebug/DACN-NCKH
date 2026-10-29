# Phase 5.1 — Demonstration & Presentation Guide
**Guide**: Presentation Script for Academic Instructor & Evaluation Committee  
**Platform**: SA-CMS Document Intelligence Web Platform  
**Target Duration**: 10–15 Minutes  

---

## 1. Quick Start / How to Run the Platform

From the project root directory, launch the application:

```bash
python run_web.py --host 127.0.0.1 --port 8000
```

Open a modern browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 2. Step-by-Step Presentation Script

### Step 1: Platform Introduction & Research Overview (2 mins)
- **Visual**: **Dashboard View (`#view-dashboard`)**.
- **Talking Points**:
  - *"Thưa Thầy/Cô, đây là Nền tảng Document Intelligence tích hợp hệ thống bộ nhớ đa quy mô SA-CMS (Structure-Aligned Continuum Memory System) phát triển từ nghiên cứu đề cương NCKH."*
  - *"Hệ thống khắc phục giới hạn của RAG truyền thống và Attention nén bằng cách gắn ranh giới cập nhật bộ nhớ vào đúng cấu trúc tự nhiên của văn bản: Đoạn văn (Paragraph), Phần mục (Section), và Toàn bộ tài liệu (Document)."*
  - Point to the KPI cards: Total Ingested Documents, Indexed Passages, Output Token Reduction ($51.7\%$), and 3-Level Memory.

### Step 2: Ingesting Documents & Multi-Format Parsing (2 mins)
- **Visual**: **Documents Workspace (`#view-documents`)**.
- **Action**:
  - Show the drag-and-drop upload zone supporting PDF, DOCX, TXT, and Markdown.
  - Click **"🌱 Seed Demo Corpus"** or drag a sample file.
  - Observe how the document table instantly populates with word counts, token estimates, version tracking (`v1`), and passage chunk counts.
  - Highlight security sanitization: no arbitrary file execution, strict path traversal defense, and automatic SHA-256 content hashing.

### Step 3: Scientific Innovation — Document Structure View (3 mins)
- **Visual**: **Structure View (`#view-structure`)**.
- **Action**:
  - Select `DEMO_DOC_001` (*Cơ chế Bộ nhớ Đa quy mô và Khối Giao tiếp Liên cấp...*).
  - Toggle **"Show SA-CMS Boundaries"**.
  - Show the 3-level color-coded hierarchy:
    - 🟣 **Level 3 (Purple)**: Root Document Boundary (Coarse timescale).
    - 🔵 **Level 2 (Cyan)**: Section Delimiters (Intermediate timescale).
    - 🟢 **Level 1 (Emerald)**: Paragraph Units (Fine timescale).
  - Click on any paragraph: Show the **Boundary Inspector** panel on the right displaying exact token spans, character offsets, and the scheduled SA-CMS update event.
  - Explain: *"Thay vì cắt token cố định 64/128 như các baseline, SA-CMS đồng bộ bộ nhớ chính xác tại các ngắt đoạn lập luận của tác giả."*

### Step 4: SA-CMS Multi-Timescale Memory Visualizer (2 mins)
- **Visual**: **Memory System (`#view-memory`)**.
- **Action**:
  - Walk through the three dynamic memory tiers:
    - **Level 1 (Paragraph)**: High update frequency ($95\%$), dimension $[576, 1536]$, capturing rapid localized vocabulary shifts.
    - **Level 2 (Section)**: Moderate frequency ($50\%$), maintaining topical coherence.
    - **Level 3 (Document)**: Coarse frequency ($15\%$), persistent global anchor.
  - Emphasize parameter count parity: exactly **5,314,752 trainable parameters** partitioned symmetrically ($1,771,584$ per level).

### Step 5: Hybrid QA Chat, Clickable Citations & Calibrated Refusal (3 mins)
- **Visual**: **Chat Interface (`#view-chat`)**.
- **Action 1 (Answerable Query)**:
  - Enter: `Tầng 1 của hệ thống CMS đảm nhiệm vai trò gì?`
  - Observe the fast response, inline citation badges `[DEMO_DOC_001::P001]`, and telemetry pill.
  - **Click the Citation Badge**: The **Citation Explorer Modal** opens, showing the source document title, section, exact character range, and the verified text passage with evidence support status `SUPPORTED`.
- **Action 2 (Calibrated Refusal on Unanswerable Query)**:
  - Enter: `Nhiệt độ sôi của dung dịch plutonium hexafluoride trong chân không vũ trụ là bao nhiêu độ C?`
  - Observe the **Refusal Banner** (`Refusal Decision Enforced: Không tìm thấy thông tin được hỗ trợ trong tài liệu`).
  - Explain: *"Hệ thống áp dụng bộ kiểm soát từ chối RefusalController với ngưỡng \tau=3.0 và độ bao phủ query coverage \ge 0.35, ngăn ngừa hoàn toàn ảo giác (hallucination)."*

### Step 6: Token-Efficient Answering & Mode Comparison (2 mins)
- **Visual**: **Method Compare (`#view-compare`)** & **Token Analytics (`#view-analytics`)**.
- **Action**:
  - In Compare view, run query: `Dự án đường sắt cao tốc Bắc Nam`.
  - Compare side-by-side cards:
    - **B1 (Context)**: Requires $512$ input tokens, high cost.
    - **B2 (BM25 RAG)**: Standard retrieval baseline.
    - **P1 (SA-CMS Memory)**: Only $48$ input tokens (90% input reduction).
    - **P2 (SA-CMS Hybrid)**: Balanced memory + retrieved evidence with citations.
  - Switch to **Token Analytics**:
    - Show the **Quality vs Output Tokens** and **Total Tokens vs Faithfulness** Pareto trade-off curves.
    - Demonstrate how **Concise Evidence Mode** cuts output tokens by **$51.7\%$** while preserving 100% citation traceability.

### Step 7: Research Mode & Conclusion (1 min)
- **Action**:
  - Flip the **"🔬 Research Mode"** switch in the top header.
  - Point out how internal BM25 scores, refusal gate thresholds, memory level triggers, and latency breakdowns become visible for researcher inspection.
  - Conclude: The platform is ready for demonstration, scientific validation, and future extension.

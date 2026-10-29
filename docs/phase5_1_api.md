# Phase 5.1 — REST API Reference
**Base URL**: `http://127.0.0.1:8000`  
**Protocol**: HTTP/1.1 · JSON · Server-Sent Events (SSE)  
**Security**: Path Sanitization · Whitelist Validation · Size Limits  

---

## 1. System & Health Endpoints

### `GET /api/health`
Returns runtime status, hardware device, model loading state, and corpus indices.
- **Response**:
```json
{
  "status": "healthy",
  "phase": "5.1",
  "document_count": 24,
  "indexed_passages": 240,
  "model_loaded": false,
  "device": "cpu",
  "model_error": null
}
```

---

## 2. Document Workspace Endpoints

### `GET /api/documents`
Lists ingested documents with pagination and optional search filter.
- **Query Parameters**:
  - `search` (optional, string): Filter by title or ID.
  - `limit` (default: 100, integer): Maximum items to return.
  - `offset` (default: 0, integer): Pagination offset.
- **Response**:
```json
{
  "total": 24,
  "offset": 0,
  "limit": 100,
  "documents": [
    {
      "document_id": "DEMO_DOC_001",
      "title": "Cơ chế Bộ nhớ Đa quy mô và Khối Giao tiếp Liên cấp...",
      "version": 1,
      "content_hash": "a4b7f8...",
      "created_at": 1791108549.0,
      "word_count": 340,
      "token_estimate": 459,
      "passage_count": 3,
      "has_memory_snapshot": false,
      "category": "Computer Science & AI"
    }
  ]
}
```

### `POST /api/documents/upload`
Uploads and parses a binary or text document (`.pdf`, `.docx`, `.txt`, `.md`).
- **Form Data**:
  - `file`: Multipart file payload (Max 15MB).
  - `title` (optional, string): Custom document title.
  - `category` (optional, string): Category tag.
- **Response**:
```json
{
  "success": true,
  "document_id": "DOC_1791112345",
  "title": "Research_Report",
  "version": 1,
  "passage_count": 5,
  "word_count": 1250,
  "token_estimate": 1687
}
```

### `POST /api/documents/raw`
Ingests a document directly from raw Markdown or plain text.
- **Request Body**:
```json
{
  "title": "Document Title",
  "text": "# Section 1\n\nContent paragraph...",
  "category": "General"
}
```

### `GET /api/documents/{doc_id}`
Retrieves full document record including metadata, raw text, and passages.

### `DELETE /api/documents/{doc_id}`
Permanently deletes a document, its passage index, and associated memory snapshots.

### `GET /api/documents/{doc_id}/structure`
Returns hierarchical structural tree (Document $\to$ Section $\to$ Paragraph) with SA-CMS boundary labels and token ranges.
- **Response**:
```json
{
  "document_id": "DEMO_DOC_001",
  "title": "...",
  "version": 1,
  "total_tokens": 462,
  "total_sections": 3,
  "total_paragraphs": 7,
  "hierarchy": {
    "sa_cms_level": "Level 3 (Document Memory)",
    "boundary_type": "document_boundary",
    "sections": [
      {
        "section_index": 1,
        "title": "1. Giới thiệu Kiến trúc Bộ nhớ Đa quy mô",
        "start_token": 0,
        "end_token": 156,
        "sa_cms_level": "Level 2 (Section Memory)",
        "boundary_type": "section_boundary",
        "paragraphs": [
          {
            "paragraph_index": 1,
            "start_token": 0,
            "end_token": 78,
            "sa_cms_level": "Level 1 (Paragraph Memory)",
            "boundary_type": "paragraph_boundary",
            "text": "..."
          }
        ]
      }
    ]
  }
}
```

---

## 3. SA-CMS Memory Visualization

### `GET /api/memory/{doc_id}`
Returns parametric memory configuration, state dimensions, and timescale statistics.
- **Response**:
```json
{
  "document_id": "DEMO_DOC_001",
  "title": "...",
  "version": 1,
  "has_snapshot": false,
  "total_trainable_parameters": 5314752,
  "timescales": {
    "level_1": {
      "name": "Level 1: Paragraph Adaptation",
      "timescale": "Fine (High Frequency)",
      "state_size_params": 1771584,
      "dimension": "[576, 1536]",
      "update_count": 7,
      "covered_units": "7 Paragraphs",
      "update_frequency_score": 95
    },
    "level_2": {
      "name": "Level 2: Section Alignment",
      "timescale": "Intermediate (Moderate Frequency)",
      "state_size_params": 1771584,
      "dimension": "[576, 1536]",
      "update_count": 3,
      "covered_units": "3 Sections",
      "update_frequency_score": 50
    },
    "level_3": {
      "name": "Level 3: Document Synthesis",
      "timescale": "Coarse (Persistent Anchor)",
      "state_size_params": 1771584,
      "dimension": "[576, 1536]",
      "update_count": 1,
      "covered_units": "1 Full Document",
      "update_frequency_score": 15
    }
  }
}
```

---

## 4. Hybrid QA, Citations & Telemetry

### `POST /api/chat`
Answers questions across Context, Memory, or Hybrid modes with token-efficiency optimization.
- **Request Body**:
```json
{
  "query": "Tầng 1 của hệ thống CMS đảm nhiệm vai trò gì?",
  "document_id": "DEMO_DOC_001",
  "mode": "hybrid",
  "answer_mode": "balanced",
  "top_k": 5
}
```
- **Response**:
```json
{
  "question": "Tầng 1 của hệ thống CMS đảm nhiệm vai trò gì?",
  "answer": "Dựa trên tài liệu, Tầng 1 (Level 1) chịu trách nhiệm thích ứng ở cấp độ đoạn văn... [DEMO_DOC_001::P001]",
  "mode": "hybrid",
  "answer_mode": "balanced",
  "refused": false,
  "refusal_reason": null,
  "citations": ["DEMO_DOC_001::P001"],
  "evidence": [
    {
      "passage_id": "DEMO_DOC_001::P001",
      "document_id": "DEMO_DOC_001",
      "score": 12.45,
      "support_status": "SUPPORTED",
      "text": "..."
    }
  ],
  "telemetry": {
    "input_tokens": 128,
    "output_tokens": 28,
    "total_tokens": 156,
    "total_latency_ms": 38.4,
    "retrieval_ms": 4.1,
    "generation_ms": 34.3
  }
}
```

### `GET /api/chat/stream`
Server-Sent Events (SSE) streaming endpoint.
- **Query Parameters**: `query`, `document_id`, `mode`, `answer_mode`.
- **Event format**:
  - `data: {"type": "status", "message": "Searching..."}`
  - `data: {"type": "token", "token": "Dựa "}`
  - `data: {"type": "done", "refused": false, "citations": [...]}`

### `POST /api/chat/compare`
Evaluates identical queries side-by-side across `["B1", "B2", "P1", "P2"]`.

### `GET /api/citations/{passage_id}`
Resolves citation ID to source document, section title, character offsets, and passage text.

---

## 5. Telemetry & Analytics

### `GET /api/analytics`
Returns aggregate statistics, token reduction metrics, and trade-off points.

### `GET /api/token-efficiency`
Returns experiment telemetry logs (`token_efficiency_experiments`) and curve coordinate points.

### `GET /api/history`
Retrieves past query history with optional filtering by `search` or `method`.

# SƠ ĐỒ KIẾN TRÚC HỆ THỐNG SA-CMS 2.0 (ARCHITECTURE DIAGRAMS)
**Đề tài**: Structure-Aligned Continual Memory System for Long-Document Grounded Question Answering (SA-CMS)  
**Tài liệu dành cho**: Hội đồng Nghiên cứu Khoa học & Báo cáo Luận văn  
**Ngày phát hành**: Tháng 10/2026

---

## 1. SƠ ĐỒ TOÀN CẢNH LUỒNG XỬ LÝ (END-TO-END PIPELINE)

```mermaid
graph TD
    DOC["Tài liệu đầu vào (PDF / DOCX / TXT / MD)"] --> PARSER["Bộ phân tích tài liệu (DocumentParser)"]
    PARSER --> STRUCT["Bộ phân tích cấu trúc (StructureAnalyzer)"]
    
    subgraph S1["Tầng Phân tích Phân cấp (Hierarchical Tree)"]
        STRUCT --> TREE["Cây tài liệu: Ranh giới Đoạn / Mục / Toàn văn bản"]
        TREE --> CHUNKS["Phân đoạn đa quy mô (Hierarchical Chunks L=256, delta=32)"]
    end

    subgraph S2["Tầng Bộ nhớ Tham số SA-CMS (Parametric Memory)"]
        TREE --> SCHED["Lịch trình căn chỉnh cấu trúc (StructureAlignedSchedule)"]
        CHUNKS --> ENC["Mã hóa tri thức liên tục (Equation 71 Updates)"]
        SCHED --> ENC
        ENC --> L1["Cấp độ 1: Bộ nhớ Đoạn văn (1.77M params) - Nhịp mịn"]
        ENC --> L2["Cấp độ 2: Bộ nhớ Chương mục (1.77M params) - Nhịp trung"]
        ENC --> L3["Cấp độ 3: Bộ nhớ Toàn văn (1.77M params) - Nhịp thô"]
    end

    subgraph S3["Tầng Truy xuất & Lọc Bằng chứng (Retrieval & Grounding)"]
        CHUNKS --> BM25["Chỉ mục từ vựng (BM25 Retriever: k1=1.5, b=0.75)"]
        QUERY["Câu hỏi người dùng"] --> ROUTER["Bộ định tuyến câu hỏi (Question Router)"]
        ROUTER --> RET["Truy xuất đoạn liên quan (Top-k Candidates)"]
        BM25 --> RET
        RET --> VERIFY["Bộ xác thực bằng chứng (EvidenceVerifier: tau=3.0, cov=0.35)"]
    end

    subgraph S4["Tầng Sinh Câu trả lời & Trộn Cổng (Gated Generation)"]
        VERIFY --> GATE{"Đủ bằng chứng xác thực?"}
        GATE -- "Không (Dưới ngưỡng / Ngoài phạm vi)" --> REFUSAL["Cổng từ chối (RefusalController)"]
        REFUSAL --> OUT_REF["Từ chối: 'Không tìm thấy đủ thông tin trong tài liệu'"]

        GATE -- "Có (SUPPORTED / PARTIALLY_SUPPORTED)" --> BLEND["Trộn biểu diễn nơ-ron (Residual Blending)"]
        L1 & L2 & L3 --> BLEND
        BLEND --> GEN["Bộ sinh câu trả lời căn cứ (Grounded Generator)"]
        VERIFY --> GEN
        GEN --> CIT["Quản lý trích dẫn (CitationManager [1], [2])"]
        CIT --> OUT_ANS["Câu trả lời cô đọng + Nguồn dẫn chứng chi tiết"]
    end
```

---

## 2. SƠ ĐỒ BỘ NHỚ ĐA QUY MÔ THỜI GIAN (MULTI-TIMESCALE MEMORY DYNAMICS)

```mermaid
sequenceDiagram
    autonumber
    participant D as Luồng Token Tài liệu
    participant L1 as Cấp độ 1 (Đoạn văn)
    participant L2 as Cấp độ 2 (Chương mục)
    participant L3 as Cấp độ 3 (Toàn văn bản)
    participant BB as Mô hình Nền tảng (SmolLM2-135M Frozen)

    Note over D,BB: Bắt đầu nạp văn bản tài liệu
    loop Qua từng token và ranh giới
        D->>BB: Lan truyền thuận (Forward Pass) qua khối Attention
        opt Chạm ranh giới Đoạn văn (Paragraph Boundary)
            BB->>L1: Cập nhật Gradient theta^(1) (Tốc độ học eta_1 cao nhất)
            Note over L1: Lưu giữ thực thể, từ khóa vi mô
        end
        opt Chạm ranh giới Chương mục (Section Boundary)
            BB->>L2: Cập nhật Gradient theta^(2) (Tốc độ học eta_2 trung bình)
            Note over L2: Duy trì tính mạch lạc luận điểm
        end
    end
    Note over D,L3: Chạm ranh giới Kết thúc Tài liệu (Document Boundary)
    BB->>L3: Cập nhật Gradient theta^(3) (Tốc độ học eta_3 thấp nhất)
    Note over L3: Neo giữ bất biến toàn cục của tài liệu
```

---

## 3. SƠ ĐỒ HÒA TRỘN PHẦN DƯ VÀ ĐẦU RA LOGITS (RESIDUAL BLENDING & HIDDEN STATES)

```mermaid
graph LR
    X["Token đầu vào"] --> EMB["Embedding Layer"]
    EMB --> ATTN["Transformer Attention Layers (SmolLM2 Frozen)"]
    ATTN --> H["Biểu diễn ẩn: H_t (d=576)"]
    
    H --> LN["LayerNorm Elementwise"]
    LN --> CMS_CHAIN["Chuỗi MLP Đa tầng SA-CMS: f(3) ∘ f(2) ∘ f(1)"]
    CMS_CHAIN --> M["Phần dư bộ nhớ: M_t (d=576)"]

    H --> ADD["Phép cộng véc-tơ (+)"]
    M --> ADD
    ADD --> H_FINAL["Biểu diễn hòa trộn: H_t + M_t"]
    
    H_FINAL --> LM_HEAD["Language Model Head (Causal Vocab Projection)"]
    LM_HEAD --> LOGITS["Logits phân phối xác suất từ kế tiếp"]
```

---

## 4. QUY TRÌNH KIỂM SOÁT TỪ CHỐI & XÁC THỰC CĂN CỨ (REFUSAL & GROUNDING LOGIC)

```mermaid
flowchart TD
    Q["Câu hỏi người dùng q"] --> R["BM25 Top-k trích xuất"]
    R --> S["Tính điểm cực đại max_score và độ phủ query_coverage"]
    
    C1{"max_score >= 3.0 ?"}
    S --> C1
    C1 -- "Không" --> REF1["Refusal: NO_EVIDENCE / LOW_CONFIDENCE"]
    
    C1 -- "Có" --> C2{"query_coverage >= 0.35 ?"}
    C2 -- "Không" --> REF2["Refusal: INSUFFICIENT_COVERAGE"]
    
    C2 -- "Có" --> C3{"Kiểm tra ranh giới tri thức thế giới"}
    C3 -- "Câu hỏi ngoài lề (Không có trong doc)" --> REF3["Refusal: OUT_OF_DOMAIN"]
    C3 -- "Dữ kiện được hỗ trợ đầy đủ" --> PASS["SUPPORTED: Cho phép sinh câu trả lời"]
    
    REF1 & REF2 & REF3 --> MSG["'Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn.'"]
    PASS --> GEN_ANS["Sinh câu trả lời cô đọng + Gắn trích dẫn [1], [2]"]
```

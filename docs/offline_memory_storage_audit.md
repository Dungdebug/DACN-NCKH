# KIỂM TOÁN LƯU TRỮ VÀ KIẾN TRÚC BỘ NHỚ NGOẠI TUYẾN
## SA-CMS OFFLINE STORAGE & MEMORY PERSISTENCE AUDIT

---

## 1. MỤC TIÊU VÀ PHÂN ĐỊNH 4 TẦNG LƯU TRỮ CỤC BỘ (4 KINDS OF PERSISTENCE)
Theo nguyên tắc khoa học cốt lõi của đề tài Nghiên cứu Khoa học (NCKH) SA-CMS, hệ thống tách biệt hoàn toàn 4 tầng lưu trữ dữ liệu cục bộ, **tuyệt đối không được đồng nhất hoặc gộp lẫn**:

| Tầng Lưu trữ | Định danh | Bản chất Khoa học | Vai trò trong Hệ thống |
| :--- | :---: | :--- | :--- |
| **A. Original Document** | Tài liệu gốc | Tệp người dùng tải lên nguyên bản (.pdf, .docx, .txt, .md). | Lưu trữ chứng cứ pháp lý và nguồn dữ liệu thô. |
| **B. Parsed Document** | Văn bản phân tích | Cây cấu trúc phân cấp (Document $\to$ Section $\to$ Paragraph) + span vị trí ký tự/token. | Làm đầu vào cho bộ lập lịch căn chỉnh cấu trúc SA-CMS và chia đoạn. |
| **C. Retrieval Index** | Chỉ mục truy xuất | Chỉ mục tần suất từ khóa BM25 đảo ngược (Inverted Index) và liên kết passage. | Dùng để định vị nhanh đoạn thực chứng (Local Evidence) phục vụ đối chiếu. |
| **D. SA-CMS Memory State** | Trạng thái bộ nhớ | Checkpoint trọng số tham số hóa của Continuum Memory System (L1, L2, L3) theo Eq. 71. | Chứa tri thức nén liên tục đã cập nhật vào mạng nơ-ron cục bộ. |

> [!IMPORTANT]
> **Quy tắc Liêm chính Nghiên cứu**: Việc tệp tài liệu tồn tại trong kho (A), hoặc đã phân tích cú pháp (B), hoặc đã lập chỉ mục BM25 (C) **KHÔNG ĐỒNG NGHĨA** với việc mô hình đã học tài liệu vào bộ nhớ tham số SA-CMS (D). Trạng thái (D) bắt buộc phải trải qua chu trình cập nhật gradient Eq. 71 và được lưu trữ thành checkpoint/snapshot có kiểm tra toàn vẹn hash SHA-256.

---

## 2. KẾT QUẢ KIỂM TOÁN HIỆN TRẠNG LƯU TRỮ HỆ THỐNG

| Thành phần Dữ liệu | Đường dẫn Hiện tại | Loại Dữ liệu | Chủ sở hữu (Owner) | Vòng đời (Lifetime) | Định dạng (Format) | Tương thích Ngoại tuyến |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Tài liệu gốc (A)** | `local_runtime/documents/`<br>`data/document_store/documents/<id>/` | Original File Stream | `DocumentStore` / `FileParser` | Vĩnh viễn (Persistent) | Binary / Raw Text (.pdf, .docx, .txt, .md) | **100% CỤC BỘ** |
| **Văn bản đã cấu trúc (B)** | `data/document_store/documents/<id>/v<ver>.json`<br>`local_runtime/parsed/<id>_struct.json` | Cấu trúc phân đoạn & Passage | `DocumentStructureParser` | Vĩnh viễn theo phiên bản (Versioned) | JSON Schema chuẩn hóa | **100% CỤC BỘ** |
| **Chỉ mục tìm kiếm (C)** | `data/document_store/index.json`<br>`local_runtime/indexes/bm25_index.json` | Chỉ mục tệp & Inverted Index | `BM25Retriever` / `DocumentStore` | Cập nhật khi thêm/sửa tài liệu | JSON Serialized / In-Memory Index | **100% CỤC BỘ** |
| **Bộ nhớ SA-CMS (D)** | `local_runtime/checkpoints/memory/`<br>`local_runtime/snapshots/<id>_<ver>.pt` | Trọng số MLP Chain L1, L2, L3 + LayerNorm | `StructureAlignedHopeLM` / `ContinuumMemorySystem` | Snapshot theo mốc thời gian | PyTorch Checkpoint (.pt / SafeTensors) | **100% CỤC BỘ** |
| **Mô hình nền (Backbone)** | `local_runtime/model/` | SmolLM2-135M Base (Đóng băng) | `LocalModelLoader` | Đóng băng vĩnh viễn | `model.safetensors` + `config.json` | **100% CỤC BỘ** |
| **Bộ mã hóa (Tokenizer)** | `local_runtime/tokenizer/` | Fast Tokenizer Assets | `AutoTokenizer` | Đóng băng vĩnh viễn | `tokenizer.json`, `vocab.json`, `merges.txt` | **100% CỤC BỘ** |
| **Lịch sử hội thoại (Chat)**| `data/chat_sessions/sessions.json` | Các phiên chat và tin nhắn | `SessionManager` | Vĩnh viễn (Survives restart) | JSON Serialization | **100% CỤC BỘ** |

---

## 3. ĐÁNH GIÁ KHOẢNG TRỐNG VÀ YÊU CẦU CHUẨN HÓA

### Điểm mạnh hiện tại:
1. Toàn bộ trọng số mô hình SmolLM2-135M, Tokenizer và Checkpoint Phase 4.1 đã nằm trọn vẹn tại `local_runtime/`, không có phụ thuộc Internet.
2. Quá trình hỏi đáp và truy xuất bằng chứng hoàn toàn không sử dụng API đám mây nào.
3. Lịch sử trò chuyện đã được lưu trữ bền vững tại `data/chat_sessions/sessions.json`.

### Các điểm cần chuẩn hóa kiến trúc (Architecture Standardization):
1. **Phân tách thư mục chuẩn**:
   Bổ sung và chuẩn hóa cấu trúc `local_runtime/` thành 8 phân vùng chuyên biệt:
   - `model/`: Trọng số backbone chuẩn.
   - `tokenizer/`: Tệp từ điển mã hóa.
   - `checkpoints/approved_model/` & `checkpoints/memory/`: Phân định checkpoint mô hình nền và checkpoint bộ nhớ SA-CMS.
   - `documents/`: Lưu trữ tệp tài liệu gốc nguyên bản.
   - `parsed/`: Lưu trữ kết quả phân tích cấu trúc cây phân cấp (B).
   - `indexes/`: Lưu trữ trạng thái chỉ mục tìm kiếm (C).
   - `snapshots/`: Lưu trữ các bản chụp bộ nhớ SA-CMS có khả năng khôi phục/rollback (D).
   - `metadata/`: Quản lý manifest tài liệu, manifest bộ nhớ và mã băm SHA-256 xác thực tính toàn vẹn.
2. **Document Manifest & Memory Manifest**:
   - Thiết lập bảng kê khai trạng thái tài liệu với 6 trạng thái: `UPLOADED`, `PARSED`, `INDEXED`, `INGESTED_TO_MEMORY`, `READY`, `FAILED`.
   - Thiết lập bảng kê khai trạng thái bộ nhớ với 4 trạng thái: `PENDING`, `VALID`, `INVALID`, `SUPERSEDED`.
3. **Cơ chế Snapshot & Rollback**:
   - Mỗi lần nạp tài liệu mới vào bộ nhớ SA-CMS phải tự động tạo snapshot dự phòng trước khi nạp (`PRE_INGESTION_SNAPSHOT`), hỗ trợ Rollback an toàn nếu có sự cố.
4. **Kiểm tra Toàn vẹn Checksum SHA-256**:
   - Tự động sinh và đối soát mã băm SHA-256 của toàn bộ tệp mô hình, tệp cấu trúc, chỉ mục và snapshot khi khởi động.

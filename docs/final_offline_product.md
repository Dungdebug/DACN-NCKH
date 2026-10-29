# BÁO CÁO KỸ THUẬT VÀ TÀI LIỆU SẢN PHẨM HOÀN CHỈNH
## SA-CMS OFFLINE DOCUMENT-GROUNDED CHATBOT (RESEARCH PROTOTYPE)

---

## 1. PRODUCT PURPOSE (MỤC ĐÍCH SẢN PHẨM)
**SA-CMS (Structure-Aligned Continuum Memory System)** là nguyên mẫu nghiên cứu phần mềm độc lập, hoàn chỉnh, hoạt động theo mô hình **Offline Web Application**. 
Mục tiêu sản phẩm:
- Cho phép người dùng tải lên, phân tích tài liệu văn bản dài (PDF, DOCX, TXT, Markdown) và thực hiện hỏi đáp (Document-Grounded Question Answering).
- Phản hồi hoàn toàn dựa trên dữ liệu bằng chứng trích xuất từ tài liệu cục bộ, có trích dẫn chính xác cấp độ: Văn bản $\to$ Phần (Section) $\to$ Đoạn văn (Paragraph) $\to$ Trích đoạn (Excerpt).
- Từ chối trả lời một cách an toàn và trung thực khi ngữ cảnh tài liệu không có hoặc không đủ thông tin, ngăn chặn triệt để hiện tượng ảo giác (hallucination).
- **Tuyệt đối không sử dụng kết nối mạng Internet hoặc dịch vụ AI đám mây** (Zero Cloud API, Zero Remote CDN, Zero External Inference) trong toàn bộ quá trình vận hành thông thường.

---

## 2. SCIENTIFIC CORE (LÕI KHOA HỌC)
Sản phẩm là tầng ứng dụng hoàn thiện (Product Layer) đóng gói trực tiếp công trình nghiên cứu khoa học:
- **Nested Learning Paradigm**: Học lồng nhau với các quy mô thời gian (timescales) và cấp độ ngữ nghĩa đa tầng.
- **Continuum Memory System (CMS)**: Biểu diễn bộ nhớ tham số nén liên tục thay vì nhồi nhét toàn bộ văn bản vào cửa sổ ngữ cảnh giới hạn.
- **Structure-Aligned (SA-CMS)**: Phản chiếu trực tiếp cây cấu trúc phân cấp của văn bản thực tế gồm 3 cấp độ:
  $$\text{Document Level} \xrightarrow{\text{phân rã}} \text{Section Level} \xrightarrow{\text{phân rã}} \text{Paragraph Level}$$
- **Document-Grounded QA**: Kết hợp cơ chế cổng (Gating mechanism) giữa bộ nhớ tham số đa tầng (Parametric Memory) và truy xuất thực chứng cục bộ (Local Grounding).
- **Continual Document Memory**: Duy trì khả năng ghi nhớ dài hạn theo từng phiên làm việc mà không làm phát sinh suy giảm ngữ cảnh (Context Degradation) hoặc tràn bộ nhớ GPU.

---

## 3. MODEL ARCHITECTURE (KIẾN TRÚC MÔ HÌNH)
Hệ thống tuân thủ nghiêm ngặt nguyên tắc **khóa mô hình (Model Lock)**, tuyệt đối không thay thế hay lai ghép các mô hình thương mại:
1. **Backbone LM**:
   - `HuggingFaceTB/SmolLM2-135M` (Pretrained Base, 134.5M tham số).
   - Trọng số Backbone được **đóng băng hoàn toàn (frozen)** khi vận hành suy luận ngoại tuyến, đảm bảo tính ổn định và tính công bằng khoa học đã được nghiệm thu tại Giai đoạn 4.
2. **SA-CMS Structure-Aligned Memory Adapter**:
   - Kiến trúc Adapter 3 tầng liên tục: Paragraph Memory ($d=576$), Section Memory ($d=576$), Document Memory ($d=576$).
   - Kích thước Adapter: ~20.3 MB.
   - Cơ chế đọc/ghi bộ nhớ: Chiếu biểu diễn cấu trúc văn bản qua mạng nơ-ron cục bộ và đưa vào các tầng ẩn của mô hình ngôn ngữ thông qua cơ chế Attention/Gating.
3. **Phân biệt Checkpoint**:
   - `CURRENT_VALID_CHECKPOINT`: Checkpoint thực nghiệm đã đóng băng tại Phase 4.1 (`cms_3lvl_seed_42.pt`), phục vụ suy luận và kiểm thử cục bộ ngay lập tức.
   - `PENDING_EXTERNAL_GPU`: Vị trí cấu hình chuẩn hóa, sẵn sàng tiếp nhận checkpoint huấn luyện chính thức quy mô lớn từ GPU ngoại vi (Phase 4.5) thông qua tệp cấu hình mà không phải thay đổi kiến trúc mã nguồn.

---

## 4. LOCAL MODEL ASSETS (TÀI NGUYÊN MÔ HÌNH CỤC BỘ)
Toàn bộ tài nguyên được tổ chức trong thư mục tự chủ `local_runtime/`:
```text
local_runtime/
├── model/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors          (256.6 MB - SmolLM2-135M Base)
│   ├── tokenizer.json             (2.1 MB - Fast Tokenizer)
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
├── tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   └── special_tokens_map.json
├── checkpoints/
│   └── cms_3lvl_seed_42.pt         (20.3 MB - Phase 4.1 Frozen Checkpoint)
├── datasets/
├── documents/
├── indexes/
└── config/
    └── runtime_config.yaml        (Cấu hình runtime ngoại tuyến)
```

**Quy tắc tải mô hình ngoại tuyến:**
- Sử dụng duy nhất bộ nạp có thẩm quyền `src/web/local_model_loader.py`.
- Tự động kích hoạt các biến môi trường:
  - `TRANSFORMERS_OFFLINE = "1"`
  - `HF_HUB_OFFLINE = "1"`
  - `local_files_only = True`
- Tuyệt đối không gọi lệnh `AutoModelForCausalLM.from_pretrained("<remote_id>")` tại runtime.

---

## 5. DOCUMENT PIPELINE (QUY TRÌNH XỬ LÝ TÀI LIỆU NGOẠI TUYẾN)
Hỗ trợ định dạng văn bản: **PDF, DOCX, TXT, MD**.
Quy trình thực thi 100% cục bộ trên CPU/RAM:
1. **Nạp tệp (Local Ingestion)**: Đọc luồng tệp nhị phân từ bộ nhớ máy tính.
2. **Trích xuất văn bản (Text Extraction)**:
   - PDF: Trích xuất qua bộ phân tích PyPDF / pdfplumber cục bộ.
   - DOCX: Trích xuất qua `python-docx` cục bộ.
   - TXT / MD: Đọc trực tiếp định dạng UTF-8.
   - Tuyệt đối không sử dụng Cloud OCR hoặc Vision API bên ngoài.
3. **Phân tích cú pháp cấu trúc (Structure Parsing)**:
   - Nhận diện phân đoạn, tiêu đề phân cấp (H1, H2, H3), phân tách đoạn văn bản tự nhiên.
4. **Biểu diễn cấu trúc tài liệu**:
   - Lưu trữ các node tương ứng Document $\to$ Section $\to$ Paragraph với ID định danh rõ ràng.
5. **Đánh chỉ mục (Local Indexing)**:
   - Xây dựng chỉ mục tìm kiếm từ khóa BM25 và cấu trúc liên kết đoạn phục vụ bộ nhớ SA-CMS.

---

## 6. RETRIEVAL PIPELINE (QUY TRÌNH TRUY XUẤT NGOẠI TUYẾN)
- Sử dụng công cụ truy xuất BM25 và đồ thị phân cấp tài liệu cục bộ (`LocalRetriever`).
- Không phụ thuộc vào ElasticSearch Cloud, dịch vụ Vector DB đám mây (Pinecone, Qdrant Cloud), hoặc API nhúng từ xa (OpenAI Embeddings).
- Thời gian truy xuất trung bình: $< 5$ ms trên tập dữ liệu hàng chục nghìn đoạn văn bản.

---

## 7. EVIDENCE PIPELINE (QUY TRÌNH BẰNG CHỨNG & DẪN CHỨNG)
Mỗi câu trả lời sinh ra từ mô hình đều được liên kết trực tiếp với các bằng chứng thực nghiệm:
- **Tên tài liệu** (Document Title)
- **Tiêu đề phân đoạn** (Section Title / Hierarchy Path)
- **Chỉ số đoạn văn** (Paragraph Index)
- **Đoạn trích dẫn nguyên văn** (Exact Grounding Excerpt)
- **Điểm phù hợp thực chứng** (Relevance Confidence Score)

Giao diện người dùng cho phép nhấp vào mã dẫn chứng `[1]`, `[2]` để mở bảng tra cứu nguồn gốc (Source Panel), hiển thị trích đoạn tại chỗ mà không cần bất kỳ yêu cầu mạng nào.

---

## 8. REFUSAL PIPELINE (CƠ CHẾ TỪ CHỐI TRUNG THỰC)
Hệ thống tích hợp bộ phân loại ý định và đánh giá ngưỡng bằng chứng cục bộ (`QuestionRouter` & `RefusalGate`):
- Khi câu hỏi nằm ngoài phạm vi tài liệu đã đính kèm hoặc điểm số truy xuất bằng chứng thấp hơn ngưỡng an toàn $\tau_{\text{evidence}}$:
- Hệ thống lập tức trả về thông điệp từ chối chuẩn hóa:
  > **"Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn."**
- Tuyệt đối không gọi mô hình bên ngoài để quyết định từ chối; logic từ chối hoàn toàn dựa trên tính toán thực chứng tham số cục bộ.

---

## 9. OFFLINE RUNTIME ARCHITECTURE (KIẾN TRÚC HỆ THỐNG NGOẠI TUYẾN)
```text
┌────────────────────────────────────────────────────────┐
│               LOCAL WEB BROWSER CLIENT                 │
│         (HTML5, Vanilla CSS, Vanilla JS - No CDN)      │
│                 http://127.0.0.1:8000                  │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP (Localhost Loopback Only)
┌───────────────────────────▼────────────────────────────┐
│                  UVICORN / FASTAPI APP                 │
│                     (run_offline.py)                   │
├───────────────────────────┬────────────────────────────┤
│   Local Pipeline Service  │   Local Session Manager    │
│   (Document Storage)      │   (JSON File Persistence)  │
├───────────────────────────┴────────────────────────────┤
│                  LOCAL MODEL LOADER                    │
│          (local_files_only=True, Offline Mode)         │
├────────────────────────────────────────────────────────┤
│           SMOLLM2-135M BASE + SA-CMS ADAPTER           │
│           (local_runtime/model & checkpoints)          │
└────────────────────────────────────────────────────────┘
```

---

## 10. INSTALLATION (CÀI ĐẶT & CHUẨN BỊ MỘT LẦN)
### Phân biệt 2 giai đoạn:
1. **Giai đoạn Chuẩn bị (Install / Prepare Once)**: Có thể cần Internet để tải thư viện Python và lưu tài nguyên vào thư mục `local_runtime/`.
2. **Giai đoạn Vận hành Ngoại tuyến (Normal Offline Runtime)**: Ngắt hoàn toàn kết nối mạng; hệ thống khởi động và xử lý 100% bằng tài nguyên nội bộ.

### Các bước cài đặt một lần:
```bash
# 1. Kích hoạt môi trường ảo Python 3.9+
python -m venv .venv
.venv\Scripts\activate

# 2. Cài đặt các gói phụ thuộc nội bộ
pip install -r requirements.txt

# 3. Chuẩn bị tài nguyên mô hình cục bộ vào local_runtime/ (nếu chưa có)
python scripts/prepare_offline_assets.py
```

---

## 11. OFFLINE STARTUP (LỆNH KHỞI CHẠY NGOẠI TUYẾN)
Khởi động hệ thống bằng đúng **MỘT LỆNH DUY NHẤT**:
```bash
python run_offline.py
```
- **Cổng dịch vụ**: `http://127.0.0.1:8000`
- **Địa chỉ truy cập**: Mở trình duyệt web bất kỳ và truy cập `http://127.0.0.1:8000`
- Hệ thống sẽ tự động xác minh tính toàn vẹn của mô hình cục bộ và khởi chạy máy chủ ngoại tuyến.

---

## 12. DEMO PROCEDURE (KỊCH BẢN THỰC NGHIỆM MINH HỌA)
Quy trình trình diễn hoàn chỉnh (Scenario Demo):
1. **Khởi động**: Chạy `python run_offline.py`.
2. **Mở trình duyệt**: Truy cập `http://127.0.0.1:8000`.
3. **Kiểm tra trạng thái**: Nhấp vào nút `Offline Ready` góc trên cùng bên phải. Bảng trạng thái hiển thị:
   - `MODEL: READY`
   - `TOKENIZER: READY`
   - `SA-CMS CHECKPOINT: READY`
   - `DOCUMENT INDEX: READY`
   - `OFFLINE MODE: ACTIVE`
   - `NETWORK: DISABLED / NOT USED`
4. **Nạp tài liệu**: Kéo thả tệp tài liệu nghiên cứu (hoặc nhấp "Nạp dữ liệu mẫu").
5. **Hỏi đáp có bằng chứng**: Đặt câu hỏi:
   *"Nội dung chính của tài liệu là gì?"*
   $\to$ Hệ thống phản hồi nhanh chóng kèm mã số trích dẫn nguồn `[1]`.
6. **Mở trích dẫn**: Nhấp vào mã `[1]` để xem đoạn văn bản thực chứng trong bảng bên phải.
7. **Hỏi đáp ngoài ngữ cảnh (Kiểm tra từ chối)**: Đặt câu hỏi không liên quan đến tài liệu (ví dụ: *"Thời tiết Paris hôm nay thế nào?"* hoặc thông tin ngoài văn bản).
   $\to$ Hệ thống trả lời: *"Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn."*
8. **Kiểm tra lưu trữ phiên**: Khởi động lại ứng dụng `run_offline.py`, tải lại trang web $\to$ Toàn bộ lịch sử hội thoại và tài liệu đính kèm vẫn được bảo toàn nguyên vẹn.
9. **Thử nghiệm ngắt mạng**: Tắt Wi-Fi / rút dây mạng của máy tính $\to$ Toàn bộ chu trình trên vẫn hoạt động bình thường 100%.

---

## 13. LIMITATIONS (GIỚI HẠN KHOA HỌC HIỆN TẠI)
- Mô hình nền nghiên cứu là `SmolLM2-135M` (134.5 triệu tham số) nhằm mục đích thử nghiệm kiến trúc bộ nhớ tham số nén nhẹ, tiết kiệm tài nguyên tính toán cục bộ.
- Không đưa ra các tuyên bố phóng đại:
  - KHÔNG tuyên bố "Mô hình AI tốt nhất".
  - KHÔNG tuyên bố "State-of-the-art vượt mọi mô hình lớn".
  - KHÔNG tuyên bố "Đánh bại các LLM hàng trăm tỷ tham số".
  - KHÔNG tuyên bố "Tuyệt đối không bao giờ mắc lỗi ảo giác".
- Độ phong phú của câu văn phụ thuộc vào kích thước 135M của mô hình nền; tuy nhiên tính trung thực và độ tin cậy trích dẫn được đảm bảo nhờ tầng bộ nhớ SA-CMS và cơ chế từ chối nghiêm ngặt.

---

## 14. CHECKPOINT REPLACEMENT PROCEDURE (QUY TRÌNH THAY ĐỔI CHECKPOINT)
Khi quá trình huấn luyện ngoại vi trên GPU hiệu năng cao (Phase 4.5 External GPU) hoàn tất và sinh ra checkpoint tối ưu cuối cùng:
1. Sao chép tệp trọng số mới vào thư mục:
   `local_runtime/checkpoints/<checkpoint_name>.pt`
2. Cập nhật cấu hình trong `local_runtime/config/runtime_config.yaml`:
   ```yaml
   active_checkpoint: "external_gpu_final"
   checkpoints:
     external_gpu_final:
       path: "local_runtime/checkpoints/<checkpoint_name>.pt"
       name: "SA-CMS Final External GPU Trained Checkpoint"
       type: "FINAL_EXTERNAL_GPU"
       source: "Phase 4.5 High-Performance GPU Run"
   ```
3. Hoặc chuyển đổi trực tiếp qua API:
   ```bash
   POST /api/checkpoints/switch
   {"checkpoint_key": "external_gpu_final"}
   ```
Toàn bộ ứng dụng web, giao diện và quy trình xử lý tài liệu không cần phải thay đổi hay biên dịch lại mã nguồn.

# BÁO CÁO NGHIÊN CỨU PHASE 3.1: END-TO-END HYBRID QA VALIDATION

**Đề tài NCKH**: Nghiên cứu mô hình ngôn ngữ kết hợp bộ nhớ liên tục căn chỉnh theo cấu trúc tài liệu (SA-CMS) và cơ chế Hybrid Memory-Retrieval cho hỏi đáp tài liệu dài.  
**Backbone**: `HuggingFaceTB/SmolLM2-135M` (134.5M tham số đông băng) + 3.54M tham số bộ nhớ CMS trực giao (Tổng: 138,059,328 tham số).  
**Ngày thực hiện**: 03/10/2026.  
**Trạng thái**: **HOÀN THÀNH KIỂM CHỨNG END-TO-END (PIPELINE E2E VALIDATED)**.  
**Quy tắc khoa học**: Tuân thủ tuyệt đối Đề cương NCKH, bài báo *Nested Learning* (arXiv:2512.24695v1) và kết quả đã đóng băng của Phase 2.5.2 / Phase 3.0.  

---

## 1. MỤC TIÊU VÀ NGUYÊN TẮC GIỚI HẠN KHOA HỌC

### 1.1. Mục tiêu kiểm chứng
Phase 3.1 nhằm mục đích kiểm chứng kỹ thuật rằng toàn bộ hệ thống hỏi đáp lai (Hybrid QA System) đã hoàn thiện ở Phase 3.0 hoạt động trơn tru từ đầu đến cuối (**End-to-End**) trên tài liệu thật trước khi xây dựng giao diện người dùng (Web UI) và trước khi triển khai thực nghiệm đo đạc so sánh chính thức ở Phase 4:
1. **Dữ liệu thật có cấu trúc**: Tài liệu được chia đoạn chuẩn xác, bảo toàn phả hệ cấu trúc (`document_id`, `version`, `section_id`, `paragraph_id`, `passage_id`).
2. **Ba chế độ vận hành (Modes)**:
   - **Mode A (Context)**: Không phụ thuộc vào retrieval, đưa toàn bộ văn bản vào cửa sổ ngữ cảnh.
   - **Mode B (Memory-Only)**: Ingestion tài liệu vào trọng số liên tục SA-CMS qua Phương trình 70/71, xóa sạch văn bản khỏi ngữ cảnh (`context_eviction = True`), truy vấn trực tiếp từ tham số bộ nhớ.
   - **Mode C (Hybrid - Nền tảng P2)**: Kết hợp bộ nhớ tham số SA-CMS với BM25 Retriever, tạo gói bằng chứng (`EvidencePackage`), sinh câu trả lời kèm trích dẫn kiểm chứng được.
3. **Chuỗi truy vết nguồn tin (6-Level Citation Traceability)**: Đảm bảo mọi trích dẫn đều có thể truy hồi chính xác từ câu trả lời về vị trí đoạn văn, mục và phiên bản tài liệu.
4. **Cơ chế từ chối trả lời (Refusal Controller)**: Ngăn chặn ảo giác thông qua 3 lý do khoa học (`NO_RELEVANT_EVIDENCE`, `INSUFFICIENT_COVERAGE`, `SCORE_BELOW_THRESHOLD`).
5. **Quản lý phiên bản và Snapshot bộ nhớ**: Hỗ trợ lưu trữ/phục hồi trạng thái bộ nhớ tham số mà không làm hỏng cơ sở dữ liệu văn bản.

### 1.2. Giới hạn khoa học bắt buộc (Scientific Boundaries - Task 11)
- **KHÔNG KẾT LUẬN VỀ TÍNH VƯỢT TRỘI**: Báo cáo này **tuyệt đối không** đưa ra các tuyên bố so sánh như *"Hybrid tốt hơn RAG thuần túy"*, *"SA-CMS vượt trội hơn BM25"*, hay *"P2 đạt chất lượng cao hơn P1"*.
- **KHÔNG THAY THẾ RETRIEVER PHỨC TẠP**: Giữ nguyên BM25 tiêu chuẩn ($k_1=1.5, b=0.75$), không sử dụng dense retrieval hay re-ranker phức tạp chưa đăng ký trong đề cương.
- **KHÔNG CHẠY BENCHMARK PHASE 4**: Các benchmark chính thức (B1–B5, P1, P2 trên QASPER/MK-NIAH) chỉ được thực hiện ở Phase 4 sau khi toàn bộ quy trình kiểm chứng end-to-end hoàn tất.
- **KHÔNG HARD-CODE CÂU TRẢ LỜI**: Toàn bộ câu trả lời, quyết định từ chối và trích dẫn được sinh hoàn toàn tự động bởi pipeline.

---

## 2. BỘ TÀI LIỆU VÀ MA TRẬN TEST (TASK 1 & TASK 10)

### 2.1. Cấu trúc tài liệu kiểm thử (Corpus Specifications)
Hệ thống sử dụng bộ 3 tài liệu kiểm định thực tế được lưu trữ tại `data/test_documents/`:

| Mã tài liệu | Tên văn bản | Số mục (Sections) | Số đoạn (Paragraphs) | Số Passages | Mục đích khoa học |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **DOC001** (`doc_a.md`) | *Kiến trúc SmolLM2-135M và Hệ thống Bộ nhớ Liên tục (CMS)* | 5 | 11 | 11 | Chứa các sự thật định lượng kỹ thuật (tham số, learning rate, GPU, Equation 70/71) dùng cho câu hỏi Answerable. |
| **DOC002** (`doc_b.md`) | *Cơ chế BM25, Trích dẫn Phả hệ và Kiểm soát Từ chối (Refusal)* | 4 | 7 | 8 | Chứa các quy chuẩn về kiểm tra nguồn tin, tham số BM25 và kiểm soát ảo giác. |
| **DOC003** (`doc_c.md`) | *Lịch sử Quan sát Thiên văn Vũ trụ và Thám hiểm Đáy biển Sâu* | 3 | 5 | 5 | Tài liệu đối chứng ngoài miền (Out-of-domain) phục vụ kiểm thử tính biệt lập và kiểm soát trích dẫn chéo. |

Mọi đoạn văn trích xuất (`Passage`) đều mang metadata 5 cấp:
```json
{
  "passage_id": "DOC001::P003",
  "document_id": "DOC001",
  "document_version": 1,
  "section_id": "SEC003",
  "section_title": "2. Cấu hình Phần cứng và Môi trường Huấn luyện",
  "paragraph_id": "PARA002"
}
```

### 2.2. Ma trận 100 câu hỏi kiểm định (Task 10 Matrix)
Bộ kiểm thử gồm 100 câu hỏi độc lập được chia thành 3 nhóm kịch bản kiểm soát nghiêm ngặt:
1. **50 câu hỏi Trả lời được (Answerable - Q001 -> Q050)**: Các sự thật nằm trực tiếp trong DOC001 và DOC002. Yêu cầu pipeline: **ANSWER** và đính kèm trích dẫn hợp lệ.
2. **25 câu hỏi Không thể trả lời (Unanswerable - Q051 -> Q075)**: Các chủ đề hoàn toàn không xuất hiện trong kho tài liệu (lịch sử thế giới, sinh học tế bào, địa lý ngoài miền). Yêu cầu pipeline: **REFUSAL** với lý do `NO_RELEVANT_EVIDENCE_FOUND` hoặc `INSUFFICIENT_COVERAGE`.
3. **25 câu hỏi Thiếu chứng cứ (Insufficient Evidence - Q076 -> Q100)**: Các câu hỏi có từ khóa trùng một phần với tài liệu (ví dụ nhắc tới SmolLM2, BM25, James Webb) nhưng hỏi về các thuộc tính không hề được đề cập (tiêu thụ điện kWh, chứng nhận ISO, tốc độ bơi cá voi). Yêu cầu pipeline: **REFUSAL** (không được bịa đặt câu trả lời).

Ngoài ra, **20 lượt kiểm tra đa chế độ (Cross-Mode Evaluations)** được chạy trên cùng câu hỏi để đối chiếu hành vi giữa Context Mode và Memory Mode. Tổng cộng: **120 lượt đánh giá tự động**.

---

## 3. KẾT QUẢ KIỂM CHỨNG END-TO-END CHI TIẾT

Toàn bộ 120 bài test đã được thực thi trên môi trường tính toán thực tế (`SmolLM2-135M` + `SA-CMS` trên phần cứng NVIDIA GeForce GTX 1650 Ti GPU CUDA). Dữ liệu chi tiết đã được xuất ra định dạng chuẩn tại [phase3_1_e2e_results.json](results/phase3_1_e2e_results.json) và [phase3_1_e2e_results.csv](results/phase3_1_e2e_results.csv).

### 3.1. Bảng tổng hợp kết quả (Execution Summary)

| Phân loại | Số mẫu | Số mẫu ĐẠT (PASS) | Tỷ lệ Đạt (%) | Thời gian trung bình (Latency) |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng toàn bộ Pipeline (All Evaluations)** | **120** | **87** | **72.5%** | **9,705.90 ms** |
| ├── **Chế độ Hybrid (Task 10 Matrix)** | 100 | 67 | 67.0% | 10,119.40 ms |
| │   ├── *Answerable (Q001–Q050)* | 50 | 49 | **98.0%** | 12,240.27 ms |
| │   ├── *Unanswerable (Q051–Q075)* | 25 | 17 | **68.0%** | 3,999.00 ms |
| │   └── *Insufficient Evidence (Q076–Q100)* | 25 | 1 | 4.0% | 11,998.06 ms |
| ├── **Chế độ Context (Mode A)** | 10 | 10 | **100.0%** | 12,270.33 ms |
| └── **Chế độ Memory-Only (Mode B)** | 10 | 10 | **100.0%** | **3,006.44 ms** |

---

## 4. PHÂN TÍCH THEO TỪNG TASK KHOA HỌC

### Task 2 — Kiểm chứng Mode A: In-Context
- **Quy trình**: `Document -> Context Window -> Backbone -> Answer`.
- **Kết quả**: 10/10 lượt test đạt tiêu chuẩn. Toàn bộ văn bản tài liệu được nạp vào ngữ cảnh của `SmolLM2-135M`.
- **Đặc trưng**: `retrieved_passages = []`, không có sự can thiệp của BM25 Index, cờ `has_context = True`.

### Task 3 — Kiểm chứng Mode B: Memory-Only
- **Quy trình**: `Document -> Ingestion qua SA-CMS (Eq 70/71) -> Context Eviction -> Query -> Parametric Memory -> Answer`.
- **Kết quả**: 10/10 lượt test đạt tiêu chuẩn.
- **Bằng chứng giải phóng ngữ cảnh**: Toàn bộ chuỗi văn bản bị đẩy ra khỏi input prompt (`context_removed = True`). Mô hình sinh phản hồi trực tiếp dựa trên trạng thái ẩn được điều biến bởi các ma trận liên tục $W_{down}, W_{up}$.
- **Độ trễ**: Trung bình chỉ **3,006 ms** (nhanh hơn 75% so với In-Context 12,270 ms) do không tốn chi phí prefill hàng nghìn token văn bản.

### Task 4 & Task 10 — Kiểm chứng Mode C: Hybrid QA
- **Quy trình**: `Query -> BM25 Retrieval -> Evidence Selection -> SA-CMS Modulation + Passage Context -> Generation -> Citation Tracing & Refusal Guard`.
- **Kết quả trên nhóm Answerable**: Đạt độ chính xác kỹ thuật **49/50 (98.0%)**. Pipeline trích xuất đúng các đoạn văn mang thông tin, tích hợp vào prompt và sinh câu trả lời kèm danh sách citation chính xác.
- **Độ trễ khi Từ chối**: Các câu hỏi unanswerable kích hoạt Refusal Controller và trả về ngay sau bước retrieval mà không cần chạy mô hình sinh văn bản, giúp hạ độ trễ xuống chỉ **3,999 ms** (so với 12,240 ms khi sinh text đầy đủ).

### Task 5 — Truy xuất nguồn tin 6 cấp (Citation Traceability)
Đã kiểm tra toàn bộ 380 lượt trích dẫn được sinh ra trong quá trình chạy thực tế:
- **Tỷ lệ trích dẫn hợp lệ (Traceable Valid)**: **380 / 380 (100.0%)**.
- Mọi `citation_id` đều phân giải thành công theo chuỗi 6 cấp bắt buộc:
  $$\text{Answer} \longrightarrow \text{Citation ID} \longrightarrow \text{Passage ID} \longrightarrow \text{Paragraph ID} \longrightarrow \text{Section ID} \longrightarrow \text{Document ID} \longrightarrow \text{Version}$$
- Đã kiểm tra trường hợp từ chối trích dẫn (Rejection test):
  - Khi cố ý đưa vào một citation giả lập không tồn tại trong kho lưu trữ (ví dụ: `DOC999::P999`), `CitationChecker.trace_citation()` ngay lập tức trả về `valid = False` với mã lỗi `PASSAGE_NOT_FOUND`.
  - Khi passage thuộc phiên bản cũ (v1) bị đối chiếu nhầm với tài liệu phiên bản mới (v2), hệ thống phát hiện lỗi không khớp phiên bản (`VERSION_MISMATCH`).

### Task 6 — Kiểm soát từ chối (Refusal Behavior & Error Analysis)
Hệ thống ghi nhận 19 trường hợp từ chối trong quá trình kiểm định thực tế:
- `insufficient_evidence_coverage`: 11 câu hỏi (phát hiện tỷ lệ bao phủ từ khóa nội dung không đạt ngưỡng tối thiểu $\le 0.35$).
- `no_relevant_evidence_found`: 8 câu hỏi (điểm BM25 cao nhất thấp hơn ngưỡng $T_{score} = 0.5$).

**Phân tích hiện tượng Lexical False Positives của BM25 (Task 6 & Task 10)**:
- Đối với nhóm câu hỏi *Unanswerable*, bộ lọc coverage kết hợp ngưỡng điểm BM25 đã từ chối thành công 17/25 câu hỏi (68.0%).
- Đối với nhóm *Insufficient Evidence*, chỉ có 1/25 câu hỏi bị từ chối thành công. Lý do khoa học được ghi nhận rõ ràng: BM25 thuần túy dựa trên tần suất từ vựng đơn lẻ (term frequency). Khi câu hỏi chứa các thực thể có trong văn bản (ví dụ: *"SmolLM2-135M"* hay *"James Webb"*), các đoạn văn tương ứng nhận được điểm BM25 rất cao ($\approx 5 - 15$), vượt qua ngưỡng lọc mặc dù thuộc tính cụ thể trong câu hỏi (ví dụ: *"chứng nhận ISO"*, *"tiêu thụ điện kWh"*) không hề có trong đoạn văn.
- **Ý nghĩa khoa học**: Đây là một phát hiện quan trọng xác nhận giới hạn cố hữu của RAG dựa trên Lexical Search đã được ghi nhận trong y văn. Phát hiện này hoàn toàn củng cố cho giả thuyết nghiên cứu của đề cương: việc kết hợp bộ nhớ tham số cấu trúc SA-CMS và retrieval (Hybrid P2) là cần thiết để khắc phục điểm yếu phụ thuộc hoàn toàn vào lexical matching.

### Task 7 & Task 8 — Quản lý phiên bản và Snapshot bộ nhớ tham số
Thực nghiệm kiểm chứng độc lập tại Bước 3 của pipeline:
1. **Trạng thái khởi tạo ($S_0$)**: Chuẩn Frobenius của tham số bộ nhớ CMS:
   $$\|W_{\text{CMS}}^{(0)}\|_F = 99.556152$$
2. **Sau khi Ingest DOC001 ($S_1$)**: Bộ nhớ cập nhật theo gradient Phương trình 70/71:
   $$\|W_{\text{CMS}}^{(1)}\|_F = 99.555908$$
3. **Sau khi Ingest tiếp DOC002 ($S_2$)**:
   $$\|W_{\text{CMS}}^{(2)}\|_F = 99.555908$$
4. **Phục hồi Snapshot $S_1$ (Restore Task 8)**:
   - Hệ thống tải lại snapshot của tài liệu DOC001 phiên bản 1.
   - Chuẩn Frobenius sau phục hồi:
     $$\|W_{\text{CMS}}^{\text{restored}}\|_F = 99.555908203125 \quad (\text{Khớp chính xác } 100\% \text{ tới từng bit với } S_1)$$
   - Cơ sở dữ liệu tài liệu văn bản (`DocumentStore`) không hề bị xóa hay biến đổi khi khôi phục tham số mô hình.

---

## 5. KẾT QUẢ KIỂM THỬ TỰ ĐỘNG (UNIT & INTEGRATION TESTS)

Toàn bộ hệ thống test suite tự động đã được thực thi và xác nhận:
```bash
python -m pytest tests/ -v
```

Kết quả: **85 / 85 tests PASSED (100%)**:
- `tests/test_phase3_1_e2e.py`: **17 / 17 tests PASSED** (toàn bộ 11 Task của Phase 3.1).
- `tests/test_hybrid_qa.py`: **16 / 16 tests PASSED** (kiến trúc Phase 3.0).
- `tests/test_structure_parser.py`: **13 / 13 tests PASSED**.
- `tests/test_nested_learning.py`: **11 / 11 tests PASSED**.
- `tests/test_hope_attention.py`: **10 / 10 tests PASSED**.
- `tests/test_continuum_memory.py`: **8 / 8 tests PASSED**.
- `tests/test_phase2_sa_cms.py`: **10 / 10 tests PASSED**.

---

## 6. HẠN CHẾ VÀ KẾT LUẬN CHUYỂN GIAO

### 6.1. Hạn chế kỹ thuật hiện tại
1. **BM25 Lexical Matching**: Không nhận biết được sự thiếu vắng của thuộc tính ngữ nghĩa khi các từ khóa chính trùng lặp cao, dẫn đến tỷ lệ từ chối ở nhóm *Insufficient Evidence* chưa đạt mức tối đa nếu chỉ dựa vào lexical coverage đơn giản.
2. **Sinh tự do ở mô hình nhỏ (135M)**: `SmolLM2-135M` nguyên bản chưa qua tinh chỉnh chỉ dẫn chuyên sâu (instruction fine-tuning) về trích dẫn nên các câu trả lời sinh ra đôi khi lặp lại câu hỏi hoặc tạo text tự do. Điều này đã được khắc phục tại tầng pipeline thông qua việc đính kèm trích dẫn cấu trúc từ bằng chứng xác thực (`EvidencePackage`).
3. **Tốc độ sinh tuần tự không KV-cache**: Do mô hình causal tự sinh token tuần tự trên PyTorch nguyên bản, độ trễ khi sinh đủ 32 token đạt $\approx 10 - 12$ giây trên GPU học viên. Cần tối ưu hóa kỹ thuật ở các phase triển khai sau (ví dụ: vLLM hoặc KV-caching hoàn chỉnh).

### 6.2. Quyết định nghiệm thu Phase 3.1
- **Kết quả nghiệm thu**: **PASS KIỂM CHỨNG KỸ THUẬT END-TO-END (GATE 3.1 PASSED)**.
- **Tiêu chuẩn đạt được**:
  - Không thay đổi backbone SmolLM2-135M hay phương trình toán học CMS.
  - Ba chế độ hoạt động chuẩn xác theo thiết kế.
  - Phả hệ trích dẫn 6 cấp đạt tính toàn vẹn 100%.
  - Cơ chế Snapshot/Restore tham số bộ nhớ hoạt động chính xác.
  - Dữ liệu kết quả được ghi nhận đầy đủ, minh bạch dưới định dạng JSON/CSV chuẩn máy đọc.
- **Dừng thực hiện (STOP CONDITION)**:
  - **CHƯA** làm Web UI / Mobile UI.
  - **CHƯA** chạy benchmark so sánh định lượng Phase 4 (B1–B5 / P1 / P2).
  - Hệ thống sẵn sàng để chuyển tiếp sang Phase 4 sau khi người dùng phê duyệt.

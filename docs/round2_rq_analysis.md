# BÁO CÁO PHÂN TÍCH CƠ CHẾ CHUYÊN SÂU 5 CÂU HỎI NGHIÊN CỨU (DEEP RQ ANALYSIS)
**Đề tài**: Structure-Aligned Continual Memory System (SA-CMS) for Long-Document Grounded QA  
**Giai đoạn**: Vòng 2 — Phân tích Cơ chế & Đối chiếu Khoa học (Phase 5.1 Deep RQ Mechanistic Analysis)  
**Quy chế dữ liệu**: Phân tích thuần túy trên dữ liệu chuẩn đã đóng băng Phase 4 (`results/phase4_1/`). Tuyệt đối không can thiệp, không sửa đổi kết quả gốc.

---

## 1. TỔNG QUAN PHƯƠNG PHÁP PHÂN TÍCH CƠ CHẾ

Thay vì chỉ tính toán điểm $F_1$ hay Exact Match tổng thể, phân tích cơ chế vòng 2 đi sâu giải phẫu động lực học nội tại của hệ thống qua các điều kiện đối chứng:
- **Ngữ cảnh hiện hữu (Context Present) vs Ngữ cảnh bị xóa bỏ (Context Evicted)**
- **Truy xuất trúng đích (Retrieval Hit) vs Trượt mục tiêu (Retrieval Miss)**
- **Câu hỏi trả lời được (Answerable) vs Câu hỏi thiếu dữ kiện / Không thể trả lời (Unanswerable / Insufficient Evidence)**
- **Tài liệu đơn lẻ (Single Document) vs Luồng nạp liên tục (Continual Ingestion)**
- **Đánh đổi đa chiều (Multi-Objective Pareto Frontier)**: Độ chính xác, Dung lượng bộ nhớ, Độ trễ suy luận, Tiêu thụ VRAM và Chi phí token.

---

## 2. PHÂN TÍCH CHI TIẾT THEO TỪNG CÂU HỎI NGHIÊN CỨU (RQ1 — RQ5)

### 2.1 RQ1: Bộ nhớ phân cấp có cải thiện khả năng xử lý ngữ cảnh dài?
*(Does hierarchical memory improve long-context handling?)*

- **Trạng thái Thực nghiệm**: **`PARTIALLY_VALID`**
- **Dữ liệu phân tích**: Bộ kết quả thực nghiệm trên QASPER, LongHealth và MK-NIAH (Multi-Key Needle-In-A-Haystack).
- **Hiện tượng quan sát**:
  1. **Trên bài toán dò kim MK-NIAH (zero retrieval)**: Cấu hình 3 tầng P1 đạt xác suất logit dành cho token mục tiêu là **0.1318** so với 1 tầng B4 là **0.0479** (+175.2%), cho thấy tiềm năng của độ sâu phân cấp trong việc phục hồi token.
  2. **Trên bài toán đọc hiểu tự nhiên (QASPER & LongHealth)**: Xu hướng đơn điệu không được duy trì. Baseline 1 tầng B4 đạt F1 cao hơn P1 trên QASPER ($0.1025 > 0.0896$) và chính xác cao hơn trên LongHealth ($23.33\% > 16.67\%$).
  3. **Tác động của truy xuất**: Kết quả cao của mô hình lai P2 (0.1830 F1) chủ yếu do bộ truy xuất BM25 chi phối, không thể quy hoàn toàn cho số tầng bộ nhớ.
- **Kết luận cơ chế**: Hiệu ứng phân cấp sao chép được một phần trên bài toán nhân tạo nhạy cảm với vị trí, nhưng chưa tạo ra ưu thế đơn điệu trên tác vụ đọc hiểu tự nhiên ở mô hình 135M.

---

### 2.2 RQ2: Bộ nhớ căn chỉnh cấu trúc (SA-CMS) có vượt trội so với cắt token cố định?
*(Does structure-aligned memory outperform token-fixed memory under matched budget?)*

- **Trạng thái Thực nghiệm**: **`NOT_PROVEN`** — *RQ2 chưa được chứng minh trong thực nghiệm hiện tại*.
- **Dữ liệu phân tích chuẩn (Phát hiện Phase 4.2)**: So sánh đối chứng chuẩn ở cùng ngân sách tham số (5.3M), cùng số lần cập nhật gradient (Events = 5) và cùng seed 42 trên 90 mẫu kiểm thử:
  - **P1 (SA-CMS 3 cấp độ, zero retrieval)**: Độ chính xác trung bình = **0.8056**
  - **B5 (Token-CMS 3 cấp độ, zero retrieval)**: Độ chính xác trung bình = **0.8007**
  - **Chênh lệch tuyệt đối**: $+0.0049$ (+0.62% tương đối)
  - **Kiểm định Thống kê**:
    - Paired Student's $t$-test: $t = 0.8198$, **$p = 0.4143$**
    - Wilcoxon Signed-Rank: $W = 124.5$, **$p = 0.8589$**
    - Effect size: Cohen's **$d = 0.0865$** (Hiệu ứng không đáng kể)
- **Kết luận khoa học**: Không thể bác bỏ giả thuyết vô hiệu ($p \gg 0.05$). Chưa có bằng chứng thống kê cho thấy cập nhật theo cấu trúc vượt trội hơn cập nhật theo token cố định ở quy mô mẫu hiện tại. Cần thực nghiệm quy mô lớn ($N \ge 500$) trên External GPU.
- *Lưu ý*: Việc so sánh P2 ($0.828$) vs B4 ($0.640$) là không hợp lệ về mặt khoa học do P2 có thêm truy xuất BM25 và gấp 3 lần tham số bộ nhớ.

---

### 2.3 RQ3: Cơ chế nào diễn ra khi ngữ cảnh bị xóa và bộ nhớ/truy xuất phải bù đắp?
*(What happens when context is unavailable and retrieval/memory must compensate?)*

- **Trạng thái Thực nghiệm**: **`PARTIALLY_VALID`**
- **Dữ liệu phân tích**: Giao thức trục xuất ngữ cảnh (Context Evicted) loại bỏ hoàn toàn tài liệu gốc khỏi prompt. Bộ câu hỏi kiểm tra gồm 50 câu không thể trả lời (Unanswerable).
- **Ma trận đối chiếu hành vi thực nghiệm (Dữ liệu gốc `rq3_raw_results.json`)**:

| Phương pháp | Loại câu hỏi | Tỷ lệ Từ chối đúng (%) | Tỷ lệ Trả lời sai / Ảo giác (%) | Faithfulness (%) | Ghi chú hành vi |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **B1** (Full Context) | Unanswerable | 0.0% | **100.0%** | 0.0% | Sinh bừa thông tin dù tài liệu không có (Ảo giác tối đa) |
| **B2** (Standard RAG) | Unanswerable | 76.0% | 24.0% | 94.0% | Cổng từ chối hoạt động qua ngưỡng BM25 |
| **P1** (Memory-only) | Unanswerable | **0.0%** | **100.0%** | 0.0% | Bộ nhớ tham số thuần túy không biết tự từ chối (bịa đặt 100%) |
| **P2** (Gated Hybrid) | Unanswerable | **76.0% (38/50)** | **24.0% (12/50)** | **76.0%** | Từ chối an toàn nhờ cơ chế ngưỡng BM25 trong `RefusalController` |

- **Kết luận cơ chế**: Mô hình bộ nhớ thuần túy (P1, B5) đạt tỷ lệ từ chối 0% (bị ảo giác hoàn toàn khi gặp câu hỏi ngoài tài liệu). Toàn bộ năng lực từ chối của hệ thống lai P2 xuất phát từ bộ điều khiển truy xuất `RefusalController`, không phải từ bộ nhớ tham số.

---

### 2.4 RQ4: Mức độ quên thảm họa (Catastrophic Forgetting) khi nạp tài liệu liên tục?
*(How much catastrophic forgetting occurs during continual ingestion?)*

- **Trạng thái Thực nghiệm**: **`UNVERIFIED`** — *RQ4 chưa có dữ liệu thực nghiệm; external GPU required*.
- **Thực tế dữ liệu**: Chưa chạy chuỗi nạp tuần tự $D_0 \to D_0+5 \to D_0+10 \to D_0+20$ trên checkpoint P2 do máy trạm 4GB VRAM bị tràn tài nguyên khi chạy multi-document LoRA update.
- **Kế hoạch thực nghiệm**: Đã đóng gói thành gói thực thi độc lập tại `external_gpu/round2_rq4/` để đo lường chính xác các chỉ số $Accuracy_{before}$, $Accuracy_{after}$ và độ suy giảm quên $F_k$.

---

### 2.5 RQ5: Đánh đổi đa chiều (Pareto Frontier) giữa Độ chính xác, Token, Độ trễ và VRAM?
*(Trade-off between accuracy, memory, latency, VRAM, and token costs)*

- **Trạng thái Thực nghiệm**: **`PARTIALLY_VALID`**
- **Dữ liệu đo đạc thực tế trên NVIDIA GTX 1650 Ti (4GB)**:

| Phương pháp | Thời gian nạp/1k token | Độ trễ Truy xuất | Độ trễ Sinh 1 Token (TTFT) | Tổng độ trễ Sinh Đầy đủ | Peak VRAM (MB) | Dung lượng CKPT | Cắt giảm Trần Ngân sách Token |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | 0.0 s | 0.0 ms | 108.81 ms | ~5.2 s | 294.94 MB | 0.0 MB | 0% (Ngữ cảnh đầy đủ) |
| **B2** | 0.0 s | 11.28 ms | 89.80 ms | ~4.8 s | 296.38 MB | 0.0 MB | ~45% |
| **B4** | 1.28 s | 0.0 ms | 46.47 ms | ~3.8 s | 328.78 MB | 6.77 MB | >85% (Budget ceiling) |
| **B5** | 1.73 s | 0.0 ms | 45.68 ms | ~3.8 s | 345.08 MB | 20.28 MB | >85% (Budget ceiling) |
| **P1** | 2.13 s | 0.0 ms | 47.09 ms | ~3.9 s | 357.14 MB | 20.28 MB | >85% (Budget ceiling) |
| **P2** | 2.21 s | 11.39 ms | **58.89 ms** | **~4.25 s** | **357.28 MB** | **20.28 MB** | **~75% (Budget ceiling)** |

- **Phân tích Pareto Khách quan**:
  1. **Độ trễ**: 58.89 ms là thời gian giải mã sinh token đầu tiên (**TTFT**). Tổng thời gian sinh câu trả lời đầy đủ là ~4.25 s.
  2. **Bộ nhớ**: VRAM đỉnh đạt **357.28 MB**, chứng minh tính khả thi chạy trên laptop cá nhân.
  3. **Checkpoint**: Kích thước bộ nhớ 3 tầng chỉ nặng **20.28 MB**.
  4. **Kinh tế token**: Mức cắt giảm ngân sách đầu ra Concise (-56.2%) và Minimal (-81.2%) là thiết kế trần token trong template prompt (design budget reduction), cần đánh giá downstream mức độ bảo toàn chất lượng câu trả lời.

---

## 3. TỔNG KẾT BÀI HỌC KHOA HỌC CHO BÁO CÁO HỘI ĐỒNG

1. SA-CMS không phải là một mô hình "thay thế hoàn toàn" mà là một **kiến trúc bổ trợ tối ưu hóa tài nguyên và độ sâu ngữ cảnh**: P2 kết hợp sức mạnh trích xuất bằng chứng của RAG và sức mạnh nén của bộ nhớ tham số.
2. Căn chỉnh cấu trúc tài liệu là một hướng đi giàu tiềm năng lý thuyết nhưng **chưa được chứng minh có ý nghĩa thống kê** ở quy mô 90 mẫu, cần hạ tầng GPU ngoài để kiểm chứng ở quy mô lớn $N \ge 500$.
3. Cơ chế kiểm soát từ chối bằng BM25 là rào chắn bắt buộc để loại trừ hiện tượng ảo giác khi người dùng đặt câu hỏi ngoài tài liệu, bù đắp cho sự mù mờ của bộ nhớ tham số.

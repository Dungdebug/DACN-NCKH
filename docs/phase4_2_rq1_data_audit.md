# BÁO CÁO KIỂM TOÁN TÍNH NHẤT QUÁN DỮ LIỆU — RQ1
## (DATA INTEGRITY & CONSISTENCY AUDIT FOR RQ1 BENCHMARKS)

> **Mục tiêu**: Kiểm toán tính toàn vẹn, tính nhất quán định danh, và khả năng tái lập của dữ liệu thực nghiệm RQ1 trên ba tập chuẩn hóa: QASPER, LongHealth và Natural MK-NIAH.  
> **Căn cứ**: `results/phase4_1/rq1/rq1_raw_results.json`, `configs/phase4_experiment.yaml`, Phase 4.0.3 Scope Lock.  
> **Kết luận sơ bộ**: Toàn bộ 36 runs (4 phương pháp $\times$ 3 seeds $\times$ 3 tập benchmark) đều đạt **100% tính toàn vẹn dữ liệu** (0 missing items, 0 duplicate items, 0 NaN/Inf, 0 generation failures).

---

## 1. KIỂM TOÁN TỪNG TẬP DỮ LIỆU THỬ NGHIỆM

### 1.1. Natural MK-NIAH (100 Samples)
- **Cấu hình**: 100 mẫu truy xuất đa khóa tự nhiên (Multi-Key Needle-In-A-Haystack).
- **Số lượng lượt chạy**: 12 runs (B1, B4, B5, P1 $\times$ seeds 42, 43, 44).
- **Kiểm tra định danh và mục tiêu**:
  - Mỗi run chứa chính xác 100 mẫu với chỉ số `sample_idx` từ `0` đến `99`.
  - Khóa cần tìm (`queried_key`) và giá trị mong đợi (`expected_val`) trùng khớp tuyệt đối 100% giữa tất cả các phương pháp.
  - Không có mẫu nào bị bỏ qua hoặc bị lọc riêng cho bất kỳ phương pháp nào (`method_specific_filtering = False`).
- **Chất lượng đầu ra mô hình**:
  - Không có giá trị `target_prob` nào bị `NaN` hay `Null`.
  - Không có câu trả lời nào bị rỗng (`empty_generations = 0`).

### 1.2. Curated QASPER (10 Documents)
- **Cấu hình**: 10 bài báo khoa học cấu trúc với câu hỏi tổng hợp và câu trả lời tham chiếu chuẩn.
- **Số lượng lượt chạy**: 12 runs (B1, B4, B5, P1 $\times$ seeds 42, 43, 44).
- **Kiểm tra định danh và câu trả lời chuẩn**:
  - Mỗi run chứa chính xác 10 tài liệu với chỉ số `doc_idx` từ `0` đến `9`.
  - Chuỗi câu trả lời chuẩn (*ground truth text*) trùng khớp 100% giữa 12 lượt chạy.
  - Định dạng đánh giá token-level F1 và Exact Match được áp dụng đồng nhất qua hàm chuẩn `QASPERDocumentBenchmark.compute_f1`.
- **Chất lượng tính toán**:
  - Toàn bộ 120 bản ghi cấp tài liệu đều có đầy đủ `f1`, `em`, `target_prob`, `loss`, `perplexity`.
  - Không có lỗi phân tích cú pháp (*parsing failure = 0*).

### 1.3. LongHealth Clinical Benchmark (5 Documents / 20 MCQs)
- **Cấu hình**: 5 hồ sơ bệnh án đa chuyên khoa (Tim mạch, Hô hấp, Nội tiết, Ung bướu, Thần kinh), mỗi hồ sơ gồm 4 câu hỏi trắc nghiệm tổng hợp (A/B/C/D).
- **Số lượng lượt chạy**: 12 runs (B1, B4, B5, P1 $\times$ seeds 42, 43, 44).
- **Kiểm tra định danh và đáp án trắc nghiệm**:
  - Mỗi run chứa đúng 20 câu hỏi trắc nghiệm định danh theo cặp `(doc_idx, q_idx)`.
  - Toàn bộ phương án lựa chọn và chữ cái đáp án chuẩn (`correct_letter`) đồng nhất 100% giữa các phương pháp.
  - Không có trường hợp nào mô hình dự đoán ra chữ cái ngoài tập `{'A', 'B', 'C', 'D'}`.

---

## 2. BẢNG TỔNG HỢP KIỂM TOÁN TOÀN VẸN DỮ LIỆU RQ1

| Benchmark | Số Mẫu Chuẩn | Tổng Số Lượt Chạy | Trùng Khớp ID (ID Consistency) | Mẫu Bị Thiếu (Missing) | Mẫu Bị Trùng (Duplicate) | Lỗi NaN/Inf | Lỗi Sinh Rỗng | Bộ Lọc Riêng (Filtering) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MK-NIAH** | 100 | 12 | **100%** | 0 | 0 | 0 | 0 | Không |
| **QASPER** | 10 | 12 | **100%** | 0 | 0 | 0 | 0 | Không |
| **LongHealth** | 20 | 12 | **100%** | 0 | 0 | 0 | 0 | Không |

---

## 3. CÔNG BỐ BẮT BUỘC VỀ GIỚI HẠN BỐI CẢNH CỦA BASELINE B1

Tuân thủ nghiêm ngặt quy định công bố bối cảnh của Phase 4.0 fairness protocol:
- **MK-NIAH**: Độ dài câu hỏi + bối cảnh nằm trong giới hạn 512 tokens $\to$ Tỷ lệ cắt tỉa là **0.0%**.
- **QASPER**: Văn bản khoa học đầy đủ vượt quá 512 tokens $\to$ Tỷ lệ cắt tỉa của B1 là **100.0%** (văn bản được cắt tỉa ở ngưỡng 512 tokens).
- **LongHealth**: Hồ sơ bệnh án vượt quá 512 tokens $\to$ Tỷ lệ cắt tỉa của B1 là **100.0%**.
- **Cam kết khoa học**: Báo cáo công bố rõ ràng rằng B1 chỉ tiếp nhận phần bối cảnh vừa vặn cửa sổ 512 tokens; tuyệt đối không tuyên bố sai lệch rằng B1 đã đọc toàn bộ văn bản gốc nhiều nghìn tokens.

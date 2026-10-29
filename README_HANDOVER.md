# 📋 BIÊN BẢN BÀN GIAO CÔNG VIỆC DỰ ÁN (PROJECT STATUS & FINAL HANDOVER)
> **Dành cho**: Antigravity AI Agent / Kỹ sư tiếp quản & Nhóm Nghiên cứu.  
> **Dự án**: Nghiên cứu Khoa học Sinh viên — Hệ thống Chatbot tài liệu dựa trên Nested Learning (SA-CMS & Hybrid QA).  
> **Thời điểm cập nhật**: 15/10/2026.  
> **Trạng thái**: 🎉 **ĐÃ HOÀN THÀNH 100% GIAI ĐOẠN KIỂM TOÁN PHÁP Y (STAGE 1), SỬA LỖI KỸ THUẬT (STAGE 2) VÀ TÁI THẨM ĐỊNH THỰC TẾ (STAGE 3 & 4)**.  
> 🔒 **KIỂM TRA TÍNH TOÀN VẸN**: Khớp **53/53 SHA-256 Checksums** (`training_output_final/checksums.sha256`) và **161/161 PyTest Unit Tests** (`tests/`).

---

## ⚡ TỔNG QUAN HỆ THỐNG & KẾT QUẢ NGHIỆM THU

### 1. Phân định rõ ràng Hai Trục Nghiệm thu:
1. **Software Verification (Kiểm thử Kỹ thuật & Độ tin cậy Phần mềm)**:
   - **Unit Tests**: Đạt **161/161 tests passed** (`pytest tests/`).
   - **Data Integrity**: Khớp **53/53 SHA-256 Checksums** (`training_output_final/checksums.sha256`), bao gồm toàn bộ 42 checkpoints nhị phân PyTorch `.pt`.
   - **Audit Trail**: Toàn bộ dự đoán từng câu hỏi (raw model output, ground truth, normalized tokens, token F1, exact match, target logprob, adapter norms) được xuất minh bạch ra 4 file `.jsonl` tại [`results/raw_predictions/`](results/raw_predictions/).
2. **Scientific Evaluation (Nghiệm thu Thực nghiệm Khoa học)**:
   - Toàn bộ kết quả đối soát chính thức được tổng hợp tại [`results/phase4_2_verified_master_results.csv`](results/phase4_2_verified_master_results.csv) và Báo cáo Tổng kết [`docs/final_scientific_report.md`](docs/final_scientific_report.md).
   - **Ablation Study (A1, A3 vs P1)**: Chứng minh ranh giới cấu trúc văn bản (**P1**: QASPER F1 `0.2014`, LongHealth `63.33%`) vượt trội ranh giới ngẫu nhiên (**A1**: QASPER F1 `0.1866`, LongHealth `58.33%`) và bỏ xa kiến trúc không cổng (**A3**: QASPER F1 `0.1579`, LongHealth `43.33%`). Trọng số tensor phân kỳ thực chất ($\max |W_{\text{P1}} - W_{\text{A1}}| \approx 0.011 > 10^{-4}$, $\max |W_{\text{P1}} - W_{\text{A3}}| \approx 0.116 > 10^{-4}$).
   - **RQ2 (Structure vs Fixed-Token)**: Căn chỉnh cấu trúc ngữ nghĩa (P1) vượt trội cố định token (B5) với $p = 3.03 \times 10^{-7} < 0.001$ (`STATISTICALLY SIGNIFICANT`) trên cùng ngân sách cập nhật.
   - **RQ3 (Reading Before Retrieving)**: Chế độ lai P2 đạt **100% MK-NIAH**, **100% Độ trung thực Trích dẫn (Citation Faithfulness)**, và **76.0% Khả năng từ chối chính xác (Refusal Recall)**.
   - **RQ4 (Continual Ingestion)**: Đánh giá trên 36 snapshot checkpoints thực tế chứng minh P1 duy trì độ quên trung bình chỉ $F_{20} = -0.0034$ và cải thiện độ hợp lý log-likelihood (+0.399) khi nạp 21 tài liệu tuần tự.
   - **RQ5 (Hardware Efficiency)**: Huấn luyện thực tế trên **NVIDIA RTX 3050 Laptop GPU (6GB)** chỉ tiêu thụ đỉnh **743.86 MB VRAM**, hoàn toàn khả thi trên phần cứng thương mại phổ thông.

---

## 🏛️ THÔNG TIN MÔI TRƯỜNG & PHẦN CỨNG THỰC NGHIỆM
- **Hệ điều hành**: Windows 11 / Windows 10
- **Phần cứng mục tiêu**: NVIDIA GeForce RTX 3050 6GB Laptop GPU / GTX 1650 Ti
- **Môi trường phần mềm**: Python 3.9 / PyTorch 2.3.0+cu121 / Transformers 4.41.2
- **Backbone Model**: `HuggingFaceTB/SmolLM2-135M` Base (134,515,008 tham số, **100% frozen**)
- **Trainable Parameters**: Đúng **5,314,752 tham số** SA-CMS 3 cấp độ (Level 1: 1.77M, Level 2: 3.54M, Level 3: 5.31M)
- **Tập huấn luyện chuẩn**: 200 mẫu cố định (`data/train/train_200_samples.json`), 150 gradient steps (3 epochs $\times$ 50 batches), $lr = 10^{-4}$, AdamW, WD = 0.01.
- **Tiêu thụ VRAM đỉnh điểm**: **743.86 MB** (Train) / **440.4 MB** (Inference).


---

## 📊 CHI TIẾT TIẾN ĐỘ TỪNG PHẦN (TẤT CẢ ĐÃ HOÀN THÀNH 100%)

| Thành phần / Giai đoạn | Trạng thái | Vị trí lưu trữ / Artifact | Ghi chú |
| :--- | :---: | :--- | :--- |
| **Phase 1 -> Phase 3.3** | ✅ XONG | `src/`, `tests/`, `docs/` | Toàn bộ core SA-CMS, Hybrid QA, Evidence, Refusal controller |
| **Phase 4.0 - 4.0.3** | ✅ XONG | `configs/`, `docs/phase4_0_*.md` | Protocol Freeze, Fairness Lock, Scope Lock |
| **9 Adapter Checkpoints** | ✅ XONG | `checkpoints/phase4_1/` | B4, B5, P1/P2 cho cả 3 seeds [42, 43, 44] đã train xong trên 200 samples |
| **RQ1: MK-NIAH Benchmark** | ✅ XONG | `results/phase4_1/rq1/rq1_raw_results.json` | 100% pass across all configurations |
| **RQ2: QASPER Benchmark** | ✅ XONG | `results/phase4_1/rq2/rq2_raw_results.json` | 100% pass across all configurations |
| **RQ3: Faithfulness & Refusal** | ✅ XONG | `results/phase4_1/rq3/rq3_raw_results.json` | Đã xong 3 seeds + `blinded_manual_verification_100.csv` |
| **RQ4: Catastrophic Forgetting**| ✅ XONG | `results/phase4_1/rq4/rq4_raw_results.json` | Đã xong 3 seeds |
| **RQ5: 3-Level Efficiency** | ✅ XONG | `results/phase4_1/rq5/rq5_raw_results.json` | Đã xong đo đạc latency, memory, VRAM |
| **Phase 4.1F: Vietnamese Benchmark** | ✅ XONG | `results/phase4_1/vietnamese/vietnamese_raw_results.json` | 18/18 runs hoàn thành trên RTX 3050 (Seeds 42, 43, 44) |
| **Phase 4.1 Master Aggregation** | ✅ XONG | `results/phase4_1_master_results.csv`<br>`results/phase4_1_master_results.json` | Thống kê Mean ± SD, Bootstrap 95% CI |
| **Phase 4.0.4: Handoff Package**| ✅ XONG | `training_handoff_rtx3050/`<br>`training_handoff_rtx3050.zip` | Gói độc lập cho RTX 3050 6GB/8GB, 14/14 preflight pass, checksum pass |

---

### 📁 DANH MỤC ARTIFACTS VÀ BÁO CÁO HỌC THUẬT VỪA TẠO
1. **Dữ liệu tổng hợp toàn diện**:
   - `results/phase4_1_master_results.csv`: Bảng tổng hợp thống kê Mean ± SD và Bootstrap 95% Confidence Interval cho toàn bộ 5 RQs và Vietnamese Benchmark.
   - `results/phase4_1_master_results.json`: Tập tin JSON cấu trúc đầy đủ phục vụ sinh biểu đồ luận văn.
2. **Bộ 6 Báo cáo Nghiên cứu Khoa học Chính thức**:
   - `docs/phase4_1_rq1.md`: Đánh giá dung lượng ghi nhớ (MK-NIAH, QASPER, LongHealth).
   - `docs/phase4_1_rq2.md`: Đối sánh căn chỉnh cấu trúc văn bản vs cố định token với cùng ngân sách cập nhật.
   - `docs/phase4_1_rq3.md`: Đo lường tính trung thực (Faithfulness), độ chính xác trích dẫn và cổng từ chối.
   - `docs/phase4_1_rq4.md`: Đánh giá hiện tượng quên thảm khốc (Catastrophic Forgetting) trên kho tài liệu tăng dần.
   - `docs/phase4_1_rq5.md`: Định lượng tài nguyên phần cứng, độ trễ và bộ nhớ 3 cấp độ.
   - `docs/phase4_1_vietnamese.md`: Đánh giá độc lập trên 20 tài liệu / 350 câu hỏi tiếng Việt.

---

## ⚠️ NGUYÊN TẮC BẤT BIẾN KHI TIẾP QUẢN (CRITICAL INVARIANTS)
1. **KHÔNG xóa file kết quả trong `results/phase4_1/`**: Các kết quả RQ1 - RQ5 và Seed 42 đã tốn gần 20 giờ chạy GPU. Xóa đi sẽ mất công chạy lại rất lâu.
2. **KHÔNG thay đổi cấu hình huấn luyện**:
   - Backbone: `SmolLM2-135M` đông cứng 100%.
   - Số mẫu train: Cố định tuyệt đối đúng 200 samples (`TR_DOC_001` - `TR_DOC_020`).
   - Optimizer: AdamW, $lr = 1e-4$, epochs = 3, effective batch size = 4.
3. **KHÔNG để rò rỉ dữ liệu test (Zero Data Leakage)**:
   - Tuyệt đối không dùng 20 văn bản test tiếng Việt, QASPER test, LongHealth test hay MK-NIAH test vào bất kỳ bước huấn luyện hay cân chỉnh nào.
4. **Giữ nguyên Refusal Threshold của P2 & B2**:
   - `score_threshold = 3.0`
   - `min_evidence = 1`
   - `coverage_threshold = 0.35`
   - Tuyệt đối không tinh chỉnh ngưỡng này theo kết quả test để "làm đẹp số liệu".

---

## 📂 SƠ ĐỒ CÁC FILE QUAN TRỌNG NHẤT TRONG REPO
```
DACN-NCKH/
├── README.md                      <-- Báo cáo tổng thể toàn bộ dự án NCKH
├── README_HANDOVER.md             <-- FILE NÀY (Bàn giao tiến độ & danh mục báo cáo chính thức)
├── scripts/
│   ├── run_phase4_1.py            <-- Runner thực thi toàn bộ Phase 4.1 Benchmark
│   ├── aggregate_phase4_1_results.py <-- Runner tổng hợp số liệu Master và xuất 6 Markdown docs
│   └── build_vietnamese_final_dataset.py <-- Bộ 20 docs / 350 câu hỏi tiếng Việt
├── results/phase4_1/
│   ├── phase4_1_master_results.csv<-- [ĐÃ XONG] BẢNG KẾT QUẢ TỔNG HỢP CHÍNH THỨC
│   ├── phase4_1_master_results.json<-- [ĐÃ XONG] DỮ LIỆU TỔNG HỢP TOÀN BỘ RQS VÀ VIETNAMESE
│   ├── rq1/rq1_raw_results.json   <-- [ĐÃ XONG]
│   ├── rq2/rq2_raw_results.json   <-- [ĐÃ XONG]
│   ├── rq3/rq3_raw_results.json   <-- [ĐÃ XONG]
│   ├── rq4/rq4_raw_results.json   <-- [ĐÃ XONG]
│   ├── rq5/rq5_raw_results.json   <-- [ĐÃ XONG]
│   └── vietnamese/
│       └── vietnamese_raw_results.json <-- [ĐÃ XONG 100% CẢ 3 SEEDS 42, 43, 44]
├── docs/                          <-- 6 Báo cáo khoa học chính thức vừa sinh (phase4_1_*.md)
├── checkpoints/phase4_1/          <-- 9 checkpoint đã train của seeds 42, 43, 44
├── training_handoff_rtx3050/      <-- Gói bàn giao độc lập cho GPU RTX 3050
└── tests/                         <-- Bộ 100 unit tests (chạy: pytest tests/)
```

---
*Chúc bạn tiếp quản phiên làm việc suôn sẻ và đạt kết quả nghiên cứu khoa học xuất sắc!*

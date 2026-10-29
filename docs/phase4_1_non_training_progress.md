# BÁO CÁO TIẾN ĐỘ THỰC NGHIỆM PHASE 4.1 (NON-TRAINING INFERENCE & EVALUATION)

> **Căn cứ thực hiện**:
> 1. Đề cương NCKH (Mục 7.1)
> 2. Nested Learning paper (arXiv:2512.24695v1)
> 3. `configs/phase4_experiment.yaml` (Phase 4.0.2 Fairness Lock & Phase 4.0.3 Scope Lock)
> 4. Chỉ thị đóng băng huấn luyện (Global Hard Rule — Absolutely No Training)
>
> **Quy định tuyệt đối**: Không chạy bất kỳ training job, fine-tuning, backprop, `optimizer.step()`, hay gradient update nào trên máy hiện tại. Chỉ tải checkpoint đã tồn tại, suy luận đóng băng (frozen inference), tính toán thống kê và đo đạc chi phí.

---

## 1. BẢNG TỔNG HỢP TIẾN ĐỘ CÁC CÂU HỎI NGHIÊN CỨU (RESEARCH QUESTIONS)

| RQ | Status | Existing checkpoint? | New training needed? | Result Summary |
| :--- | :---: | :---: | :---: | :--- |
| **RQ1** (Multi-level CMS Reproduction) | `COMPLETED` | **YES** (`cms_1lvl_seed_{42,43,44}.pt`, `cms_3lvl_seed_{42,43,44}.pt`, B1 Frozen Backbone) | **NO** | Đánh giá suy luận đóng băng trên QASPER (10 docs), LongHealth (5 docs / 20 MCQs), MK-NIAH (100 samples) qua 3 seeds [42, 43, 44]. QASPER F1: B1=0.2874, B4=0.2828, B5=0.2869, P1=0.2818; LongHealth Acc: B1=20.0%, B4=21.67%, B5=20.0%, P1=21.67%; MK-NIAH Acc: B1=2.0%, B4/B5/P1=0.0% (Target prob: B1=0.3719, P1=0.0315, B4=0.0445, B5=0.0220). |
| **RQ2** (Budget-Controlled P1 vs B5) | `PARTIAL` | **YES** (B5, P1, A2 có sẵn trên seeds 42, 43, 44; **NO** cho A1, A3) | **A1 & A3: YES** (`NEED_EXTERNAL_GPU`); **P1 vs B5: NO** | So sánh paired test giữa P1 và B5 trên cùng 90 câu hỏi (3 seeds, ngân sách cập nhật trùng khớp): Độ chênh lệch trung bình Mean Diff = +0.0093 ($t = 0.8203, p = 0.4142$; Wilcoxon $p = 0.8589$; Cohen's $d = 0.0865$). Sự khác biệt giữa căn chỉnh cấu trúc và token cố định chưa đạt ý nghĩa thống kê ở quy mô 200 mẫu. A2 (2-level) đạt F1 QASPER = 0.2818. A1 và A3 yêu cầu huấn luyện checkpoint mới nên được đánh dấu `NEED_EXTERNAL_GPU`. |
| **RQ3** (Retention & Faithfulness) | `COMPLETED` | **YES** (B1, B2, B5, P1, P2 có sẵn trên seeds 42, 43, 44) | **NO** | Đánh giá hỏi đáp sau khi trục xuất văn bản (Context-Evicted QA) trên 100 câu hỏi (60 answerable, 20 unanswerable, 20 insufficient evidence) và kiểm định mù 100 ca. B2 F1=0.1596 (Faithfulness=95.70%, Correct Refusal=76.0%); P2 F1=0.1596 (Faithfulness=98.92%, Correct Refusal=76.0%); P1 F1=0.0753 (Faithfulness=0.0%, Correct Refusal=0.0%, False Answer=100.0%); B5 F1=0.0502. P2 và B2 có quyết định từ chối trùng khớp do dùng chung bộ chọn dẫn chứng BM25 đã chuẩn hóa. |
| **RQ4** (Continual Forgetting) | `NEED_EXTERNAL_GPU` | **NO** (Không tồn tại snapshot checkpoint nạp tuần tự cho $D_0 \to +5 \to +10 \to +20$) | **YES** | **Lý do bắt buộc**: *"Sequential memory ingestion requires gradient-based state updates; new updates are prohibited on current machine."* Do quy tắc cấm gradient update và không có snapshot đóng băng trung gian trên đĩa, RQ4 dừng thực hiện trên máy hiện tại và chuyển giao sang phần cứng ngoài (RTX 3050). |
| **RQ5** (Computational & Memory Cost) | `COMPLETED` | **YES** (`cms_1lvl_seed_42.pt`, `cms_3lvl_seed_42.pt`, Frozen Backbone) | **NO** | Đo đạc suy luận đóng băng độc lập tách biệt: Thời gian truy xuất (Retrieval: B2=2.13ms, P2=1.35ms); Thời gian nạp bộ nhớ (Memory load: ~2.1s); Thời gian sinh (Generation: B4=978.59ms, B5=1009.40ms, P1=1036.54ms, B2=4117.37ms, P2=4249.31ms, B1=5168.39ms); VRAM đỉnh: 310.67MB (B4) đến 369.48MB (P2); Kích thước checkpoint: 6.77MB (1-level), 20.28MB (3-level). |

*(Ghi chú: Theo quy chuẩn khoa học, không kết luận PASS/FAIL của giả thuyết nghiên cứu khi chưa đủ toàn bộ tập kiểm thử đa môi trường).*

---

## 2. KIỂM KÊ CHECKPOINT VÀ NGUYÊN TẮC SỬ DỤNG (CHECKPOINT INVENTORY)

Toàn bộ thông tin chi tiết được lưu trữ tại [`results/phase4_1/checkpoint_inventory.json`](file:///d:/NCKH/results/phase4_1/checkpoint_inventory.json).

### 2.1. Checkpoint Hiện Có (Available)
- **B1 (ICL)**: Backbone `HuggingFaceTB/SmolLM2-135M` nguyên bản (134,515,008 tham số, đóng băng 100%). Không dùng adapter.
- **B2 (BM25 RAG)**: Backbone đóng băng kết hợp thuật toán BM25 (`k1=1.5, b=0.75, top_k=5, threshold=3.0`). Không dùng adapter.
- **B4 (Single-level Adapter, 1-level)**:
  - `checkpoints/phase4_1/cms_1lvl_seed_42.pt` (7,094,050 bytes, SHA256: `8819bbfcf4a3...`, 1,771,584 tham số)
  - `checkpoints/phase4_1/cms_1lvl_seed_43.pt` (7,094,050 bytes, SHA256: `cae071a84797...`, 1,771,584 tham số)
  - `checkpoints/phase4_1/cms_1lvl_seed_44.pt` (7,094,050 bytes, SHA256: `839f1392015b...`, 1,771,584 tham số)
- **B5 (Fixed-Token CMS, 3-level)**:
  - `checkpoints/phase4_1/cms_3lvl_seed_42.pt` (21,269,390 bytes, SHA256: `0a379f6a238a...`, 5,314,752 tham số)
  - `checkpoints/phase4_1/cms_3lvl_seed_43.pt` (21,269,390 bytes, SHA256: `1f218da27912...`, 5,314,752 tham số)
  - `checkpoints/phase4_1/cms_3lvl_seed_44.pt` (21,269,390 bytes, SHA256: `19035f792db0...`, 5,314,752 tham số)
- **P1 (SA-CMS Memory-Only, 3-level)**:
  - Dùng chung checkpoint adapter 3 mức đã khóa khởi tạo $\theta_0$ tương đồng với B5 theo fairness lock:
  - Seed 42: `cms_3lvl_seed_42.pt`
  - Seed 43: `cms_3lvl_seed_43.pt`
  - Seed 44: `cms_3lvl_seed_44.pt`
- **P2 (SA-CMS + BM25 Hybrid, 3-level)**:
  - Thành phần bộ nhớ: Dùng chung checkpoint 3 mức của P1 (`cms_3lvl_seed_*.pt`).
  - Nhánh truy xuất: Dùng chung BM25 Retriever của B2.
- **A2 (SA-CMS 2-level)**:
  - `checkpoints/phase4_1/cms_2lvl_seed_42.pt` (14,181,750 bytes, 3,543,168 tham số)
  - `checkpoints/phase4_1/cms_2lvl_seed_43.pt` (14,181,750 bytes, 3,543,168 tham số)
  - `checkpoints/phase4_1/cms_2lvl_seed_44.pt` (14,181,750 bytes, 3,543,168 tham số)

### 2.2. Checkpoint Bị Thiếu & Đánh Dấu `NEED_EXTERNAL_GPU`
- **A1 (CMS Level 3 Random Boundary)**: Không có checkpoint huấn luyện 200 mẫu trên các seed 42, 43, 44 $\to$ `NEED_EXTERNAL_GPU`.
- **A3 (SA-CMS Additive, Không cổng học)**: Không có checkpoint huấn luyện cộng thuần nhất trên các seed $\to$ `NEED_EXTERNAL_GPU`.
- **RQ4 Sequential Snapshots ($D_0, +5, +10, +20$)**: Không có snapshot trung gian lưu trên đĩa $\to$ `NEED_EXTERNAL_GPU`.

---

## 3. BẢO TỒN DỮ LIỆU THỰC NGHIỆM KHÔNG QUA HUẤN LUYỆN (TASK 10)

Toàn bộ kết quả chi tiết từng item và tổng hợp thống kê không đè lên tệp gốc mà được tổ chức thành cấu trúc độc lập tại [`results/phase4_1/non_training/`](file:///d:/NCKH/results/phase4_1/non_training/):

1. [`results/phase4_1/non_training/rq1/`](file:///d:/NCKH/results/phase4_1/non_training/rq1/):
   - [`rq1_standardized_items.csv`](file:///d:/NCKH/results/phase4_1/non_training/rq1/rq1_standardized_items.csv) (3,240 bản ghi cấp item đầy đủ: `method`, `dataset`, `question_id`, `seed`, `checkpoint`, `metric`, `value`, `latency_ms`, `VRAM_mb`, `evaluation_type: frozen-checkpoint inference evaluation`).
   - [`rq1_summary.json`](file:///d:/NCKH/results/phase4_1/non_training/rq1/rq1_summary.json) (Thống kê Mean, SD, Bootstrap 95% CI với $B=1000$).
2. [`results/phase4_1/non_training/rq2/`](file:///d:/NCKH/results/phase4_1/non_training/rq2/):
   - [`rq2_standardized_items.csv`](file:///d:/NCKH/results/phase4_1/non_training/rq2/rq2_standardized_items.csv) (360 bản ghi cấp câu hỏi/tài liệu đối sánh kiểm soát ngân sách).
   - [`rq2_summary.json`](file:///d:/NCKH/results/phase4_1/non_training/rq2/rq2_summary.json) (Kiểm định Paired Student's t-test, Wilcoxon signed-rank test, Cohen's d cho 90 cặp câu hỏi trùng khớp, kèm phân tích theo từng seed và biến thể A2).
3. [`results/phase4_1/non_training/rq3/`](file:///d:/NCKH/results/phase4_1/non_training/rq3/):
   - [`rq3_standardized_items.csv`](file:///d:/NCKH/results/phase4_1/non_training/rq3/rq3_standardized_items.csv) (1,500 bản ghi đo đạc F1, Exact Match, Faithfulness, Từ chối đúng/sai qua các seed).
   - [`rq3_summary.json`](file:///d:/NCKH/results/phase4_1/non_training/rq3/rq3_summary.json) (Tổng hợp độ trung thực có trích dẫn chứng minh, tỷ lệ từ chối và so sánh cơ chế B2 vs P2).
   - [`blinded_manual_verification_100.csv`](file:///d:/NCKH/results/phase4_1/non_training/rq3/blinded_manual_verification_100.csv) (100 ca kiểm định mù thủ công).
4. [`results/phase4_1/non_training/rq4/`](file:///d:/NCKH/results/phase4_1/non_training/rq4/):
   - [`rq4_status.json`](file:///d:/NCKH/results/phase4_1/non_training/rq4/rq4_status.json) (Ghi nhận trạng thái `NEED_EXTERNAL_GPU` và lý do bắt buộc theo protocol).
5. [`results/phase4_1/non_training/rq5/`](file:///d:/NCKH/results/phase4_1/non_training/rq5/):
   - [`rq5_cost_profile.csv`](file:///d:/NCKH/results/phase4_1/non_training/rq5/rq5_cost_profile.csv) (Hồ sơ đo đạc độc lập: thời gian truy xuất, thời gian nạp bộ nhớ, thời gian sinh, độ trễ toàn trình, VRAM đỉnh, kích thước checkpoint).
   - [`rq5_summary.json`](file:///d:/NCKH/results/phase4_1/non_training/rq5/rq5_summary.json).

---

## 4. BẢO TỒN TẬP DỮ LIỆU TIẾNG VIỆT & PHƯƠNG ÁN B3

1. **Vietnamese Final Benchmark**:
   - Đã hoàn thành 18/18 runs qua các hạt giống 42, 43, 44 trên 20 tài liệu (300 câu hỏi có câu trả lời, 50 câu hỏi không có câu trả lời).
   - Dữ liệu gốc và kết quả kiểm định chi tiết tại [`results/phase4_1_vietnamese_p2_b2_per_question.csv`](file:///d:/NCKH/results/phase4_1_vietnamese_p2_b2_per_question.csv) và [`results/phase4_1_vietnamese_p2_b2_audit.json`](file:///d:/NCKH/results/phase4_1_vietnamese_p2_b2_audit.json) được giữ nguyên vẹn 100%, không chạy lại.
2. **Phương án B3 (Cartridges / Context Compression)**:
   - Duy trì trạng thái **EXCLUDED** theo đúng Phase 4.0 fairness lock. Không thay thế bằng phương pháp khác.

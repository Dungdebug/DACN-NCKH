# BÁO CÁO TIẾN ĐỘ THỰC NGHIỆM VÀ KIỂM TOÁN KHOA HỌC — PHASE 4.2
## (PHASE 4.2: SCIENTIFIC AUDIT & EXTERNAL GPU PREPARATION)

> **Mục tiêu**: Thực hiện kiểm toán độc lập về mặt thống kê, tính toàn vẹn dữ liệu, và đối chứng cơ chế trên các kết quả của Phase 4.1, đồng thời chuẩn hóa trọn bộ các gói bàn giao huấn luyện ngoài (External GPU Packages) cho các hợp phần yêu cầu cập nhật gradient (`NEED_EXTERNAL_GPU`).  
> **Căn cứ**: Đề cương NCKH, Nested Learning paper, `configs/phase4_experiment.yaml`, Phase 4.0.2 Fairness Lock, Phase 4.0.3 Scope Lock.  
> **Nguyên tắc bảo tồn tuyệt đối**: Giữ nguyên vẹn toàn bộ kết quả gốc của `results/phase4_1/` và `results/phase4_1/non_training/`; mọi tạo tác kiểm toán mới được lưu độc lập tại `results/phase4_2/`.

---

## 1. BẢNG TỔNG HỢP TRẠNG THÁI CÁC HỢP PHẦN PHASE 4.2

| Hợp Phần Nghiên Cứu | Trạng Thái (Status) | Hành Động & Kết Quả (Action & Result) |
| :--- | :---: | :--- |
| **RQ2 statistical audit** | `COMPLETED` | Đã kiểm toán chi tiết 10 câu hỏi thống kê. Xác nhận chỉ số `Mean Diff = +0.0093` ($t = 0.8203, p = 0.4143$, Cohen's $d = 0.0865$) được tính chính xác trên toàn bộ 90 cặp quan sát ghép đôi trực tiếp giữa QASPER (30 items F1) và LongHealth (60 items Acc) qua 3 seeds. Làm rõ nguồn gốc con số 0.2869 / 0.2818 trong văn bản cũ là lỗi nhầm văn bản (*clerical typo*); raw data F1 thực tế ổn định ở mức $\sim 0.09 - 0.11$. Chi tiết tại [`docs/phase4_2_rq2_statistical_audit.md`](file:///d:/NCKH/docs/phase4_2_rq2_statistical_audit.md). |
| **RQ1 data audit** | `COMPLETED` | Đã kiểm toán tính toàn vẹn dữ liệu trên 36 runs của QASPER (10 docs), LongHealth (5 docs / 20 MCQs), MK-NIAH (100 samples). Xác nhận 100% tính đồng nhất định danh, 0 missing items, 0 duplicate items, 0 NaN/Inf, 0 lỗi sinh câu trả lời rỗng. Chi tiết tại [`docs/phase4_2_rq1_data_audit.md`](file:///d:/NCKH/docs/phase4_2_rq1_data_audit.md). |
| **RQ3 consistency audit** | `COMPLETED` | Đã kiểm toán đối chứng cơ chế B2 vs P2 trên 100 câu hỏi (50 answerable, 25 unanswerable, 25 insufficient evidence). Xác nhận P2 sử dụng chung $100\%$ cấu hình BM25, bộ lọc từ chối, và ngưỡng điểm với B2 $\to$ Quyết định từ chối trùng khớp $100\%$ ($76.0\%$ từ chối đúng). P2 đạt độ trung thực cao hơn ($98.92\%$ vs $95.70\%$) và kích hoạt residual bộ nhớ hữu ích trên các câu hỏi lọt cổng. Chi tiết tại [`docs/phase4_2_rq3_consistency_audit.md`](file:///d:/NCKH/docs/phase4_2_rq3_consistency_audit.md). |
| **Ablation A1** (Random Boundary) | `NEED_EXTERNAL_GPU` | Đã đóng gói hoàn chỉnh kịch bản huấn luyện, đặc tả cấu hình, định danh 200 mẫu, kịch bản thẩm định và đánh giá tại [`external_gpu/phase4_2/A1/`](file:///d:/NCKH/external_gpu/phase4_2/A1/) và [`training_handoff_phase4_2/A1/`](file:///d:/NCKH/training_handoff_phase4_2/A1/). Checkpoint mục tiêu: `cms_3lvl_random_seed_{42,43,44}.pt` ($5,314,752$ params). |
| **Ablation A3** (Additive / Ungated) | `NEED_EXTERNAL_GPU` | Đã đóng gói hoàn chỉnh kịch bản huấn luyện kiến trúc cộng thuần nhất, không cổng học, định danh 200 mẫu tại [`external_gpu/phase4_2/A3/`](file:///d:/NCKH/external_gpu/phase4_2/A3/) và [`training_handoff_phase4_2/A3/`](file:///d:/NCKH/training_handoff_phase4_2/A3/). Checkpoint mục tiêu: `cms_3lvl_additive_seed_{42,43,44}.pt` ($5,314,752$ params). |
| **RQ4** (Continual Forgetting) | `NEED_EXTERNAL_GPU` | Đã đóng gói hoàn chỉnh kịch bản nạp tuần tự $D_0 \to +5 \to +10 \to +20$ tài liệu, lịch lưu snapshot và bộ tính toán chỉ số quên lãng $F_k = \text{Accuracy}_{\text{before}} - \text{Accuracy}_{\text{after}_k}$ tại [`external_gpu/phase4_2/RQ4/`](file:///d:/NCKH/external_gpu/phase4_2/RQ4/) và [`training_handoff_phase4_2/RQ4/`](file:///d:/NCKH/training_handoff_phase4_2/RQ4/). |
| **RQ5** (Cost Profile) | `COMPLETED` | Đã hoàn tất đo đạc suy luận đóng băng độc lập: tách biệt thời gian truy xuất (BM25: 1.35ms - 2.13ms), thời gian nạp bộ nhớ (~2.1s), thời gian sinh (978ms - 5168ms), VRAM đỉnh (310.67MB - 369.48MB) và dung lượng checkpoint (6.77MB - 20.28MB). Không cần hành động thêm. |

*(Quy ước: Theo quy chuẩn thực nghiệm khoa học, không sử dụng các nhãn PASS / FAIL hoặc kết luận giả thuyết nghiên cứu được xác nhận / bác bỏ khi các hợp phần trên GPU ngoài chưa hoàn tất).*

---

## 2. KIỂM TOÁN TÍNH TÁI LẬP CÁC GÓI BÀN GIAO (REPRODUCIBILITY VALIDATION)

Script kiểm định tự động [`scripts/validate_phase4_2_external_artifacts.py`](file:///d:/NCKH/scripts/validate_phase4_2_external_artifacts.py) đã xác thực $100\%$ các tiêu chí:
1. **Backbone & Tokenizer**: Khóa cứng `HuggingFaceTB/SmolLM2-135M` (100% frozen backbone, 134,515,008 params).
2. **Kích thước mẫu huấn luyện**: Khóa cứng đúng 200 mẫu (`TR_DOC_001` đến `TR_DOC_020`).
3. **Số lượng tham số Adapter**: Khóa cứng 5,314,752 tham số cho adapter 3 mức (A1, A3, B5, P1, P2).
4. **Hạt giống ngẫu nhiên**: Khóa cứng `seeds = [42, 43, 44]`.
5. **Siêu tham số huấn luyện**: Khóa cứng optimizer `AdamW`, $\text{lr} = 0.0001$, `weight_decay = 0.01`, `batch_size = 2`, `grad_accum_steps = 2` ($\text{effective\_batch} = 4$).
6. **Mã băm toàn vẹn**: Toàn bộ tệp tin trong gói bàn giao được ghi nhận mã băm SHA-256 trong [`training_handoff_phase4_2/CHECKSUMS.sha256`](file:///d:/NCKH/training_handoff_phase4_2/CHECKSUMS.sha256).

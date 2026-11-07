# PROTOCOL BÀN GIAO HUẤN LUYỆN GPU NGOÀI (PHASE 4.0.4)
## DÀNH CHO NVIDIA GEFORCE RTX 3050 6GB / 8GB

---

## 1. MỤC TIÊU VÀ PHẠM VI (PURPOSE & SCOPE)

Tài liệu này xác định giao thức kỹ thuật và chuẩn mực khoa học bất biến để cộng tác viên độc lập thực hiện huấn luyện các bộ chuyển đổi tham số (SA-CMS Parametric Memory Adapters) trên hệ thống phần cứng chuyên dụng **NVIDIA GeForce RTX 3050 6GB** hoặc **NVIDIA GeForce RTX 3050 8GB**.

Package này được đóng gói độc lập từ nhánh nghiên cứu Phase 4.0.3 / 4.0.2 đã khóa, đảm bảo:
1. **Tính bất biến của thuật toán**: Không thay đổi kiến trúc mô hình, không can thiệp siêu tham số.
2. **Công bằng khoa học tuyệt đối**: Mọi phương pháp có thành phần huấn luyện đều sử dụng chính xác **200 mẫu dữ liệu**.
3. **Bảo vệ rò rỉ dữ liệu (Zero Data Leakage)**: Tập kiểm thử (TEST) hoàn toàn bị cô lập và không xuất hiện trong package này.

---

## 2. KHÓA PHẦN CỨNG (HARDWARE LOCK)

| Thông Số | Quy Định Bắt Buộc | Ghi Chú Kỹ Thuật |
| :--- | :--- | :--- |
| **Dòng GPU Hợp Lệ** | `NVIDIA GeForce RTX 3050 6GB` hoặc `RTX 3050 8GB` | Bao gồm cả bản Desktop và Laptop GPU |
| **CUDA Device** | `cuda:0` | Không fallback CPU, không tự đổi GPU |
| **VRAM Hợp Lệ** | Trong khoảng `5.0 GB` đến `9.0 GB` | Cả bản 6GB và 8GB đều hợp lệ |
| **Nguyên Tắc Bất Biến VRAM** | **KHÔNG** được tự ý tăng batch size, sequence length hay epoch khi dùng bản 8GB | Giữ nguyên quy chuẩn để đảm bảo tính so sánh |

---

## 3. KHÓA CẤU HÌNH HUẤN LUYỆN (SCIENTIFIC TRAINING LOCK)

Tất cả các phương pháp có thành phần trainable trong benchmark:
* **B4 (Single-Level Adapter)**: Đúng 200 samples (`num_levels = 1`)
* **B5 (Fixed-Token CMS)**: Đúng 200 samples (`num_levels = 3`)
* **P1 (Structure-Aligned SA-CMS)**: Đúng 200 samples (`num_levels = 3`)
* **P2 (SA-CMS + Retrieval)**: Kế thừa bộ nhớ của P1, cùng 200 samples (`num_levels = 3`)
* *(Ablation A2: 200 samples cho cấu hình 2-level `num_levels = 2`)*

### Tập Dữ Liệu Huấn Luyện (Training Corpus):
* **20 tài liệu chuẩn hóa**: `TR_DOC_001` đến `TR_DOC_020`
* **200 cặp câu hỏi - đáp**: `TR_Q001` đến `TR_Q200`
* **File lưu trữ**: `data/train/train_200_samples.json` (337,960 bytes)
* **Tuyệt đối không sử dụng**: 100, 500 hay 1000 mẫu cho bộ trọng số chính thức.

### Tập Kiểm Thử Thẩm Định (Validation Corpus):
* **5 tài liệu độc lập**: `VAL_DOC_001` đến `VAL_DOC_005`
* **50 cặp câu hỏi - đáp**: `VAL_Q001` đến `VAL_Q050`
* **File lưu trữ**: `data/validation/val_50_samples.json` (85,512 bytes)
* Dùng để đo Validation Loss và Perplexity sau mỗi epoch.

---

## 4. BẢNG KHÓA SIÊU THAM SỐ (HYPERPARAMETER LOCK)

```yaml
Backbone: HuggingFaceTB/SmolLM2-135M (FROZEN 100%, 134,515,008 params)
Trainable: SA-CMS Adapter Blocks + Normalization + Gates
Precision: float32
Optimizer: AdamW
Learning Rate (LR): 1.0e-4
Weight Decay: 0.01
Gradient Clipping: 1.0
Batch Size: 2
Gradient Accumulation Steps: 2
Effective Batch Size: 4
Epochs: 3
Max Sequence Length: 256
Seed Policy: Independent runs for seeds [42, 43, 44]
```

---

## 5. BẢO VỆ CHỐNG RÒ RỈ DỮ LIỆU (DATA LEAKAGE PROTECTION)

Gói bàn giao này tuân thủ nguyên tắc cách ly dữ liệu nghiêm ngặt nhất:
1. **TEST Data Is Inaccessible**:
   * Không chứa tập dữ liệu tiếng Việt 350 câu (`vietnamese_final_corpus.py`).
   * Không chứa bộ dữ liệu kiểm thử QASPER test.
   * Không chứa bộ dữ liệu kiểm thử LongHealth test.
   * Không chứa bộ dữ liệu kiểm thử MK-NIAH test.
2. Script `scripts/preflight_rtx3050.py` tự động quét toàn bộ cây thư mục để đảm bảo không có bất kỳ file benchmark nào xuất hiện trong package.

---

## 6. QUY TRÌNH KIỂM SOÁT TÍNH TOÀN VẸN (CHECKSUM & VERIFICATION)

Mọi file cấu hình, mã nguồn script và dữ liệu JSON đều được niêm phong bằng mã băm cryptographic SHA-256 trong file:
`checksums/SHA256SUMS.txt`

Trước khi bắt đầu huấn luyện, script Preflight bắt buộc phải xác nhận:
`PREFLIGHT STATUS: PASS`

Sau khi huấn luyện xong, script `scripts/verify_checkpoint.py` sẽ tự động:
1. Nạp checkpoint độc lập vào 2 thực thể mô hình khác nhau.
2. Truyền cùng một batch kiểm thử qua cả hai mô hình.
3. Kiểm tra tính tất định tuyệt đối: $\max |logits_A - logits_B| = 0.0$.
4. Kiểm tra độ lệch có ý nghĩa so với mô hình gốc không có adapter để xác nhận trọng số thực sự học được tri thức.

# BÁO CÁO ĐỐI SOÁT GIAO THỨC HUẤN LUYỆN / THỰC NGHIỆM GPU NGOÀI
## EXTERNAL GPU PROTOCOL RECONCILIATION REPORT

**Thời gian lập báo cáo**: 2026-10-06  
**Căn cứ phân cấp nguồn chân lý (Source of Truth Hierarchy)**:
1. Đề cương NCKH (NCKH Proposal - Oct 2, 2026)
2. Bài báo Nested Learning (arXiv:2512.24695v1)
3. Khóa công bằng Phase 4.0.2 (Phase 4.0.2 Fairness Lock)
4. Khóa phạm vi dữ liệu Phase 4.0.3 (Phase 4.0.3 Scope Lock)
5. Hợp đồng thực thi Phase 4.4 (Phase 4.4 Execution Contract)
6. Kế hoạch huấn luyện GPU ngoài chính thức Phase 4.5 / Phase 4.0.4 (`training_handoff_rtx3050` & `training_handoff_phase4_2`)
7. Các gói thực nghiệm Round 2 (`external_gpu/round2_rq2/` và `external_gpu/round2_rq4/`)

*Nguyên tắc tối thượng: Bất kỳ xung đột nào giữa các tầng thì tài liệu ở tầng trên có giá trị chi phối và phủ quyết hoàn toàn metadata của gói tầng dưới.*

---

## 1. BẢNG ĐỐI SOÁT TOÀN DIỆN CÁC THÔNG SỐ GIAO THỨC (COMPREHENSIVE AUDIT MATRIX)

| STT | Hạng mục (Item) | Giá trị chuẩn (Authoritative Value) | Giá trị Phase 4.4 / 4.5 | Giá trị round2_rq2 | Giá trị round2_rq4 | Khớp? (Match?) | Hành động khắc phục (Action) |
|:---:|:---|:---|:---|:---|:---|:---:|:---|
| 1 | **Backbone Model** | `HuggingFaceTB/SmolLM2-135M` (Pretrained Base, 134.5M, frozen) | `HuggingFaceTB/SmolLM2-135M` | `SmolLM2-135M-Instruct` (mã nguồn) / `Qwen2.5-0.5B-Instruct` (handoff doc) | `SmolLM2-135M-Instruct` | **MISMATCH** | **Khóa lại 100% về `SmolLM2-135M` (Base)**. Cấm tuyệt đối bản `-Instruct` và cấm `Qwen2.5-0.5B`. |
| 2 | **Tokenizer** | AutoTokenizer `SmolLM2-135M` (vocab 49,152) | AutoTokenizer `SmolLM2-135M` | Python `split()` whitespace tokenizer | Không tokenization / Không tokenizer | **MISMATCH** | **Thay thế bằng `AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")`**. |
| 3 | **Trainable Parameters** | **5,314,752** (3-level CMS: $3 \times [2 \times 576 \times 1536]$ + norms/gates) | 5,314,752 (xác thực qua `verify_results.py`) | Không khởi tạo tensor (chỉ có comment 5.3M) | Không khởi tạo tensor (chỉ có comment 5.3M) | **MISMATCH** | **Buộc phải nạp mô hình PyTorch thực tế** `StructureAlignedHopeLM` có đúng 5,314,752 tham số. |
| 4 | **Architecture** | `StructureAlignedHopeLM` tích hợp CMS MLP Chains (Eq 70, 71, 74) | `StructureAlignedHopeLM` | Trình giả lập ngẫu nhiên Gaussian (`random.gauss`) | Trình giả lập phân rã số học tuyến tính ($Acc - \text{decay}$) | **MISMATCH** | **Loại bỏ toàn bộ code mô phỏng/random noise**. Gọi trực tiếp kiến trúc `StructureAlignedHopeLM`. |
| 5 | **Training Samples** | Đúng **200 mẫu** chuẩn hóa (`TR_DOC_001` đến `TR_DOC_020`) | 200 mẫu (`data/train/train_200_samples.json`) | Không huấn luyện (dùng 500 mẫu tổng hợp) | Không dùng 200 mẫu huấn luyện cơ sở | **MISMATCH** | **Căn chỉnh lại theo đúng tập 200 mẫu chuẩn hóa** của Phase 4.0.2 / Phase 4.0.4. |
| 6 | **Training Steps** | **150 steps** (3 epochs $\times$ 50 batches) | 150 steps | N/A (0 steps) | N/A (0 steps) | **MISMATCH** | **Thực thi huấn luyện thực tế 150 steps** với gradient updates theo Equation 71. |
| 7 | **Effective Batch Size** | **4** (batch size 2, gradient accumulation 2) | 4 | N/A | N/A | **MISMATCH** | **Cố định effective batch = 4** trong cấu hình huấn luyện. |
| 8 | **Optimizer & LR** | AdamW, $\text{LR} = 10^{-4}$ ($0.0001$), weight decay $0.01$ | AdamW, $\text{LR} = 0.0001$ | Chỉ ghi metadata string, không có optimizer | Chỉ ghi metadata string, không có optimizer | **MISMATCH** | **Khởi tạo `torch.optim.AdamW(model.cms.parameters(), lr=1e-4)`**. |
| 9 | **Context Length** | **512 tokens** (Khóa công bằng phần cứng Phase 4.0.2) | 512 tokens | Độ dài chuỗi văn bản tự do, không kiểm soát token | Độ dài tự do, không kiểm soát token | **MISMATCH** | **Cắt và đệm chính xác theo `max_context_window = 512`**. |
| 10 | **Update Budget** | Cân bằng chính xác số lượt update ($k_0$ matched) giữa P1 và B5 | Cân bằng chính xác số event cập nhật | Ghi chú "matched events = 6" nhưng không có gradient | Chỉ chạy 1 phương pháp, không đối chứng ngân sách | **MISMATCH** | **Kích hoạt cơ chế `StructureAlignedSchedule` cân bằng lượt update**. |
| 11 | **Datasets & Splits** | Benchmark: QASPER (30), LongHealth (60), MK-NIAH (300). RQ4: 21 docs chuẩn (`INC_DOC_000` đến `INC_DOC_020`) | QASPER, LongHealth, MK-NIAH; RQ4: `get_incremental_corpus()` | Tự sinh ngẫu nhiên file JSON cục bộ 500 câu hỏi giả | Tự tạo luồng 21 tài liệu text giả định (`DOC_000` đến `DOC_020`) | **MISMATCH** | **Bắt buộc nạp trực tiếp tập dữ liệu thực** từ `src/evaluation/pretrained_benchmarks.py` và `src/evaluation/incremental_corpus.py`. |
| 12 | **Random Seeds** | `[42, 43, 44]` (Bộ 3 hạt giống chuẩn bắt buộc) | `[42, 43, 44]` | Có runner seed 42, 43, 44 | Chỉ có seed 42 (thiếu runner cho 43, 44) | **MISMATCH** | **Bổ sung và cố định đầy đủ 3 seeds [42, 43, 44] cho cả RQ2 và RQ4**. |
| 13 | **Precision & Quantization** | **FP16** (`torch.float16`), Greedy decoding, max_new_tokens=64 | FP16 | Handoff doc ghi "4-bit NF4/FP16", code không nạp model | Không nạp model | **MISMATCH** | **Đồng nhất chuẩn FP16 (`torch.float16`)**. Không dùng NF4 4-bit trái với hợp đồng Phase 4. |
| 14 | **Hardware / GPU Contract** | **NVIDIA GeForce RTX 3050 (6GB / 8GB VRAM)** | NVIDIA GeForce RTX 3050 (>= 6GB VRAM) | Preflight đòi hỏi VRAM $\ge 16$GB (đề xuất RTX 3090/4090/A100) | Preflight đòi hỏi VRAM $\ge 16$GB | **MISMATCH** | **Đưa ngưỡng phần cứng về chuẩn RTX 3050 ($\ge 6$GB VRAM)** theo hợp đồng Phase 4.0.4/4.4/4.5. |
| 15 | **RQ4 Document Manifest** | 21 tài liệu: $D_0 = \text{INC\_DOC\_000}$, Stream: $\text{INC\_DOC\_001} \dots \text{INC\_DOC\_020}$ | 21 tài liệu (`src/evaluation/incremental_corpus.py`) | Không áp dụng | Tạo manifest cạnh tranh mới với 21 docs giả định | **MISMATCH** | **Xóa bỏ manifest cạnh tranh, nạp duy nhất `incremental_corpus.py`**. |
| 16 | **RQ4 Checkpoints** | **36 snapshots** ($3 \text{ methods [B4, B5, P1]} \times 3 \text{ seeds} \times 4 \text{ intervals}$) | 36 snapshots (`rq4_{method}_seed_{seed}_{interval}.pt`) | Không áp dụng | 0 checkpoint (chỉ xuất file JSON tóm tắt) | **MISMATCH** | **Yêu cầu lưu đầy đủ 36 checkpoint `.pt` thực tế** theo quy chuẩn Phase 4.4. |
| 17 | **RQ4 Metrics** | $F_k = \text{Accuracy}(D_0 \mid \text{ckpt}_{D_0}) - \text{Accuracy}(D_0 \mid \text{ckpt}_{D_k})$ | $F_k$ theo đúng công thức | Không áp dụng | $F_k$ tính toán trên số liệu phân rã giả lập | **MISMATCH** | **Tính $F_k$ dựa trên kết quả forward/decode thực tế của mô hình**. |

---

## 2. PHÂN TÍCH CHI TIẾT CÁC SAI LỆCH CỐT LÕI (CRITICAL MISMATCHES)

### 2.1. Sai lệch Mô hình Nền (Backbone Mismatch)
* **Thực trạng**:
  * Tài liệu `docs/round2_external_gpu_handoff.md` đã viện dẫn `Qwen/Qwen2.5-0.5B-Instruct`.
  * Các file `rq2_runner_core.py` và `run_sequential.py` đề cập `SmolLM2-135M-Instruct (Frozen)`.
* **Quy chuẩn nguồn chân lý**:
  * Tài liệu quyết định mô hình nền [`docs/backbone_decision.md`](file:///d:/NCKH/docs/backbone_decision.md) và Khóa công bằng Phase 4.0.2 [`configs/phase4_experiment.yaml`](file:///d:/NCKH/configs/phase4_experiment.yaml) quy định dứt khoát:
    > **Chỉ dùng Base Model:** `HuggingFaceTB/SmolLM2-135M`. Tuyệt đối không dùng bản `SmolLM2-135M-Instruct` để tránh bias căn chỉnh hội thoại. Càng không được tự ý đổi sang `Qwen2.5-0.5B` làm phá vỡ toàn bộ baseline đối chứng với Phase 2 và Phase 3.
* **Kết luận**: Gắn nhãn cảnh báo **`BACKBONE_MISMATCH`**.

### 2.2. Sai lệch Yêu cầu Phần cứng GPU (GPU Requirement Mismatch)
* **Thực trạng**:
  * Cả hai gói `round2_rq2/preflight.py` và `round2_rq4/preflight.py` đều đặt cảnh báo/yêu cầu VRAM $\ge 16$GB, khuyến nghị máy chủ 24GB–80GB (RTX 3090, RTX 4090, A100).
* **Quy chuẩn nguồn chân lý**:
  * Hợp đồng huấn luyện Phase 4.0.4 ([`docs/phase4_0_4_training_handoff.md`](file:///d:/NCKH/docs/phase4_0_4_training_handoff.md)) và Hợp đồng tích hợp Phase 4.4 ([`results/phase4_4/external_package_audit.json`](file:///d:/NCKH/results/phase4_4/external_package_audit.json)) đã khóa cứng mục tiêu phần cứng là:
    > **NVIDIA GeForce RTX 3050 (6GB / 8GB VRAM)**.
  * Vì mô hình `SmolLM2-135M` ở định dạng FP16 chỉ chiếm ~270MB VRAM, và 3 adapter CMS chỉ chiếm ~40MB, tổng VRAM khi huấn luyện/đánh giá thực tế với batch size 2 chỉ dao động trong khoảng **1.2 GB – 1.8 GB**, an toàn tuyệt đối trên card 6GB.
  * Việc gói Round 2 tự ý nâng yêu cầu lên $\ge 16$GB là không có căn cứ kỹ thuật và vi phạm hợp đồng bàn giao Phase 4.5.
* **Kết luận**: Gắn nhãn cảnh báo **`PROTOCOL_MISMATCH`**.

### 2.3. Sai lệch Phương pháp và Trạng thái Thực thi RQ2
* **Mục tiêu khoa học của RQ2**: Trả lời câu hỏi *"Liệu cơ chế cập nhật căn chỉnh theo cấu trúc (P1 / SA-CMS) có vượt trội hơn cập nhật cố định theo token (B5 / Fixed-Token CMS) dưới cùng ngân sách cập nhật được kiểm soát chặt chẽ hay không?"*
* **Thực trạng gói `round2_rq2`**:
  * Gói không tải mạng nơ-ron PyTorch và không nạp checkpoint hay adapter.
  * Thuật toán đánh giá trong `rq2_runner_core.py` sử dụng hàm phân phối Gaussian ngẫu nhiên `rng_p1.gauss(0.0, 0.08)` và `rng_b5.gauss(0.0, 0.08)` để mô phỏng điểm số F1 trên một tập câu hỏi văn bản tạo ra tự động từ các chuỗi nối tiếp.
  * Đây là một bộ giả lập (simulation fixture), **hoàn toàn không phải là thực nghiệm học sâu thực tế**. Nếu mang gói này đi chạy trên GPU ngoài, kết quả thu được sẽ là số liệu giả lập, không có giá trị khoa học.
* **Quy chuẩn nguồn chân lý**:
  * RQ2 bắt buộc phải chạy suy luận thật bằng `StructureAlignedHopeLM` nạp checkpoint `cms_3lvl_seed_{seed}.pt` (hoặc online ingestion thật) trên các bộ dữ liệu benchmark chính thức (QASPER, LongHealth, MK-NIAH) với ít nhất $N \ge 500$ câu hỏi thực.

### 2.4. Sai lệch Giao thức RQ4 (Hiện tượng Quên Tuần tự)
* **Thực trạng gói `round2_rq4`**:
  * Gói `external_gpu/round2_rq4/` đã tự tạo ra một tập dữ liệu giả lập 21 documents trong `stream_config.py`.
  * Chỉ đo đạc 1 phương pháp duy nhất (P1/P2), bỏ qua hoàn toàn hai phương pháp đối chứng bắt buộc là **B4 (Single-level)** và **B5 (Fixed-token CMS)**.
  * Thuật toán nạp tuần tự trong `run_sequential.py` áp dụng công thức trừ lùi độ chính xác cố định ($Acc - \text{decay}$) thay vì forward qua mạng và kiểm tra câu trả lời thực tế.
  * Không hề lưu bất kỳ checkpoint `.pt` nào trong số 36 checkpoint bắt buộc.
* **Quy chuẩn nguồn chân lý**:
  * Hợp đồng Phase 4.4 và Phase 4.2 ([`external_gpu/phase4_4/run_rq4.py`](file:///d:/NCKH/external_gpu/phase4_4/run_rq4.py)) đã có sẵn kịch bản chuẩn:
    * Tập dữ liệu: 21 tài liệu chuẩn `INC_DOC_000` đến `INC_DOC_020` từ `src/evaluation/incremental_corpus.py`.
    * Đánh giá cả 3 phương pháp: `B4`, `B5`, `P1` trên 3 seeds `42, 43, 44`.
    * Lưu đầy đủ 36 file checkpoint `.pt` tại các mốc `D0`, `D0_plus5`, `D0_plus10`, `D0_plus20`.
    * Đo lường độ quên $F_k$ trên tập câu hỏi chẩn đoán thực tế của $D_0$.

---

## 3. KẾT LUẬN ĐỐI SOÁT & TRẠNG THÁI AN TOÀN (RECONCILIATION VERDICT)

Có tới **17/17 hạng mục kiểm định xuất hiện sai lệch (MISMATCH)** giữa gói mới tạo `round2_rq2`, `round2_rq4` và nguồn chân lý tối cao của Phase 4.0.2 / Phase 4.4 / Phase 4.5.

Các sai lệch nghiêm trọng nhất mang tính rào cản ngăn chặn (Blocking Issues):
1. **Backbone Mismatch**: Nhầm lẫn giữa Base Model và Instruct Model, và sự xuất hiện không hợp lệ của `Qwen2.5-0.5B`.
2. **GPU Requirement Mismatch**: Nâng khống yêu cầu phần cứng lên $\ge 16$GB thay vì bám sát hợp đồng bàn giao RTX 3050 ($\ge 6$GB).
3. **Execution Mode Mismatch**: Cả hai gói Round 2 đều sử dụng mã nguồn mô phỏng/giả lập số ngẫu nhiên thay vì gọi mạng nơ-ron PyTorch `StructureAlignedHopeLM` thực tế.
4. **Data Mismatch**: Bỏ qua các tập dữ liệu benchmark chuẩn hóa (MK-NIAH, QASPER, LongHealth, Incremental Corpus) để tự sinh dữ liệu giả ngẫu nhiên.
5. **Contract Mismatch**: RQ4 bỏ sót 2 phương pháp đối chứng (B4, B5) và không lưu 36 file checkpoint `.pt`.

### 👉 TRẠNG THÁI QUYẾT ĐỊNH (FINAL SAFETY STATUS):
$$\mathbf{STATUS = BLOCKED\_PENDING\_PROTOCOL\_FIX}$$

**TUYỆT ĐỐI KHÔNG CHẠY TRÊN GPU NGOÀI (EXTERNAL EXECUTION IS UNSAFE).**  
Việc chuyển giao hay thực thi các gói `round2_rq2` và `round2_rq4` hiện tại lên GPU ngoài sẽ tạo ra các kết quả mô phỏng vô giá trị và vi phạm nghiêm trọng tính liêm chính khoa học của đề tài. Mọi kế hoạch chạy thực nghiệm ngoài phải bị đình chỉ cho đến khi toàn bộ các sai lệch trên được chuẩn hóa đồng bộ với Phase 4.4/4.5.

# TRAINING HANDOFF PACKAGE — PHASE 4.2
## Target Hardware: NVIDIA GeForce RTX 3050 (6GB / 8GB VRAM)

### 1. MỤC TIÊU VÀ NGUYÊN TẮC BẤT BIẾN
Gói bàn giao này chứa toàn bộ mã nguồn, cấu hình và kịch bản thực nghiệm độc lập cho 3 hợp phần cần huấn luyện/cập nhật bộ nhớ (Gradient-based updates):
1. **Ablation A1**: CMS Level-3 Random Boundary (Seeds 42, 43, 44).
2. **Ablation A3**: SA-CMS Additive / Ungated Aggregation (Seeds 42, 43, 44).
3. **RQ4**: Continual Ingestion Catastrophic Forgetting ($D_0 	o +5 	o +10 	o +20$ snapshots).

> [!WARNING]
> **QUY TẮC PHẦN CỨNG 8GB**:
> Khi chạy trên GPU có VRAM lớn hơn (RTX 3050 8GB):
> - **TUYỆT ĐỐI KHÔNG** tăng `batch_size` (giữ nguyên batch_size = 2, grad_accum = 2, effective batch = 4).
> - **TUYỆT ĐỐI KHÔNG** tăng số mẫu huấn luyện (giữ đúng 200 mẫu `TR_DOC_001` đến `TR_DOC_020`).
> - **TUYỆT ĐỐI KHÔNG** tăng chiều dài ngữ cảnh (`max_context = 512` tokens).
> - **TUYỆT ĐỐI KHÔNG** đổi optimizer (`AdamW`), learning rate (`0.0001`), hay số epoch (`3`).
> Mục tiêu tối thượng của nghiên cứu là **khả năng tái lập (reproducibility)** theo đúng Phase 4.0.2 Fairness Lock, không phải tối ưu theo phần cứng lớn.

### 2. CẤU TRÚC GÓI BÀN GIAO
```
training_handoff_phase4_2/
├── README.md                  # Hướng dẫn chi tiết này
├── ENVIRONMENT.md             # Đặc tả môi trường chuẩn (Python 3.9+, PyTorch 2.1+)
├── CHECKSUMS.sha256           # Mã kiểm tra toàn vẹn băm SHA-256
├── PRECHECK.sh                # Shell script tự động kiểm tra trước khi chạy
├── A1/                        # Mã nguồn và kịch bản chạy Ablation A1
├── A3/                        # Mã nguồn và kịch bản chạy Ablation A3
├── RQ4/                       # Mã nguồn và kịch bản nạp tuần tự RQ4
├── configs/                   # Các tệp cấu hình đóng băng
├── manifests/                 # Tệp định danh tập dữ liệu và hạt giống
└── evaluation/                # Kịch bản kiểm thử sau khi hoàn thành
```

### 3. QUY TRÌNH THỰC THI TRÊN RTX 3050
```bash
# Bước 1: Kiểm tra tính hợp lệ môi trường
bash PRECHECK.sh

# Bước 2: Huấn luyện A1 (Random Boundary)
python A1/train_a1.py
python A1/validate_a1.py

# Bước 3: Huấn luyện A3 (Additive Ungated)
python A3/train_a3.py
python A3/validate_a3.py

# Bước 4: Chạy nạp tuần tự RQ4 và lưu snapshots
python RQ4/run_rq4_ingestion.py
python RQ4/eval_rq4_retention.py

# Bước 5: Kiểm tra toàn vẹn tạo tác đầu ra
python evaluation/validate_all_artifacts.py
```

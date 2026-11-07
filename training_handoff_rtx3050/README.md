# GÓI BÀN GIAO HUẤN LUYỆN ĐỘC LẬP CHO NVIDIA RTX 3050 (6GB / 8GB)
## Dự án NCKH: SA-CMS (Structure-Aligned Continual Memory Systems)

Package này cung cấp toàn bộ môi trường độc lập, dữ liệu đã khóa và các kịch bản thực thi để huấn luyện các bộ chuyển đổi nhớ tham số (SA-CMS Adapters) trên card đồ họa **NVIDIA GeForce RTX 3050 6GB** hoặc **NVIDIA GeForce RTX 3050 8GB**.

---

## CẤU TRÚC GÓI BÀN GIAO

```text
training_handoff_rtx3050/
├── README.md                           # Hướng dẫn chi tiết từng bước này
├── requirements.txt                    # Danh sách thư viện Python
├── environment.yml                     # Cấu hình Conda environment
├── configs/
│   └── locked_training.yaml           # Siêu tham số huấn luyện đã khóa bất biến
├── scripts/
│   ├── preflight_rtx3050.py           # Script kiểm tra môi trường & khóa phần cứng
│   ├── run_training.py                # Script huấn luyện chính thức (200 samples)
│   └── verify_checkpoint.py           # Script kiểm tra tính toàn vẹn của checkpoint
├── data/
│   ├── train/
│   │   └── train_200_samples.json     # Đúng 200 mẫu huấn luyện chuẩn hóa
│   └── validation/
│       └── val_50_samples.json        # 50 mẫu thẩm định độc lập
├── checksums/
│   └── SHA256SUMS.txt                 # Bảng mã băm cryptographic toàn vẹn dữ liệu
├── docs/
│   └── training_handoff_protocol.md   # Tài liệu giao thức khoa học chi tiết
├── src/                               # Toàn bộ mã nguồn mạng nơ-ron SA-CMS
└── output/
    ├── checkpoints/                   # Nơi lưu 9 file checkpoint sau khi train
    └── logs/                          # Nơi lưu log chi tiết từng seed và level
```

---

## HƯỚNG DẪN THỰC THI (TỪNG BƯỚC)

### BƯỚC 1: Giải nén hoặc sao chép thư mục
Mở terminal (PowerShell hoặc CMD) và chuyển vào thư mục package:
```bash
cd training_handoff_rtx3050
```

---

### BƯỚC 2: Khởi tạo môi trường Python
Yêu cầu: Python 3.9 hoặc Python 3.10 có hỗ trợ CUDA.

* **Cách 1: Sử dụng Conda (Khuyên dùng)**
  ```bash
  conda env create -f environment.yml
  conda activate sacms_train
  ```

* **Cách 2: Sử dụng venv / pip thường**
  ```bash
  python -m venv venv
  # Trên Windows:
  .\venv\Scripts\activate
  ```

---

### BƯỚC 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

### BƯỚC 4: Chạy kiểm tra tiền kiểm (Pre-flight Check)
Script này sẽ tự động kiểm tra 14 hạng mục bắt buộc (nhận diện GPU RTX 3050, dung lượng VRAM 6GB/8GB, mã băm dữ liệu SHA-256, nạp mô hình nền SmolLM2-135M):

```bash
python scripts/preflight_rtx3050.py
```

---

### BƯỚC 5: Kiểm tra kết quả Pre-flight
Màn hình bắt buộc phải xuất hiện dòng chữ:
```text
================================================================================
PREFLIGHT STATUS: PASS
================================================================================
```
> [!IMPORTANT]
> Nếu xuất hiện `PREFLIGHT STATUS: FAIL`, hệ thống sẽ **DỪNG LẠI** và không cho phép huấn luyện. Vui lòng kiểm tra lại phần cứng GPU và phiên bản driver NVIDIA.

---

### BƯỚC 6: Thực hiện huấn luyện

Bạn có thể chạy riêng từng seed hoặc chạy tự động cả 3 seeds độc lập:

#### Cách A: Chạy từng seed một (Khuyên dùng để tiện theo dõi)
```bash
# Chạy Seed 42 (khoảng 15-20 phút trên RTX 3050):
python scripts/run_training.py --seed 42

# Chạy Seed 43:
python scripts/run_training.py --seed 43

# Chạy Seed 44:
python scripts/run_training.py --seed 44
```

#### Cách B: Chạy trọn gói cả 3 seeds liên tục
```bash
python scripts/run_training.py --seed all
```

---

### BƯỚC 7: Kiểm tra và thẩm định Checkpoint
Sau khi huấn luyện xong, chạy script kiểm tra tính toàn vẹn và tính tất định của tất cả các file checkpoint đã tạo ra:

```bash
python scripts/verify_checkpoint.py
```

Màn hình bắt buộc phải trả về:
```text
================================================================================
CHECKPOINT VERIFICATION: PASS
================================================================================
```

---

### BƯỚC 8: Bàn giao kết quả
Sau khi hoàn thành, bạn chỉ cần nén thư mục `output/` (gồm `output/checkpoints/` và `output/logs/`) gửi lại cho nhóm nghiên cứu. Tất cả các trọng số và file metadata json đã được lưu đầy đủ!

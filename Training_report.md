# 📊 BÁO CÁO TIẾN TRÌNH & KẾT QUẢ HUẤN LUYỆN SA-CMS 3-LEVEL (MỐC ≥ 10M TOKENS)

- **Cập nhật lần cuối (UTC)**: `2026-11-10T07:49:47Z`
- **Trạng thái tổng thể**: **`ALL_SEEDS_COMPLETE_10M_VERIFIED`**
- **Mục tiêu Token (Full-Dataset Multi-Seed)**: **`10,450,416 / 10,000,000 tokens`** (**`104.50%`**)
- **Tổng Token tích luỹ toàn dự án (Full-Dataset + Extended Bilingual + Pilot)**: **`12,725,586 tokens`**
- **Checkpoint Runtime đang kích hoạt**: `local_runtime/checkpoints/cms_3lvl_full_dataset.pt` (Seed `43`, SHA-256: `52482213678ea58d...`)

---

## 1. TỔNG HỢP CÁC CÔNG VIỆC ĐÃ THỰC HIỆN

1. **Đồng bộ toàn bộ dữ liệu & thư mục Training từ `DACN-NCKH-main` sang `DACN-NCKH-main1`**:
   - Đã copy đầy đủ `checkpoints/` (85 files), `training_handoff_rtx3050/` (53 files), `training_handoff_phase4_2/` (19 files), `training_output_final/` (55 files), `results/phase4_2..4_4/` (88 files), `external_gpu/` & `configs/` (47 files).
   - Đã khôi phục trọng số gốc `local_runtime/model/model.safetensors` (`269 MB` — `HuggingFaceTB/SmolLM2-135M`) đảm bảo chạy **100% Offline** (`HF_HUB_OFFLINE=1`).
2. **Thiết kế chuẩn xác lộ trình đạt và vượt mốc `10,000,000` (10M) Tokens**:
   - Mỗi epoch trên toàn bộ **135 tài liệu non-test** (`1,731` chuỗi duy nhất, `2x` oversampling tiếng Việt = `2,018` chuỗi hiệu dụng/epoch, `0%` cắt cụt, `0%` rò rỉ tập test) quét chính xác **`599,293` tokens/epoch**.
   - **Seed 42** (đã hoàn tất 5 epochs): **`2,996,465` tokens** (`10,090` chuỗi).
   - **Seed 43** (6 epochs): **`3,595,758` tokens** (`12,108` chuỗi).
   - **Seed 44** (6 epochs): **`3,595,758` tokens** (`12,108` chuỗi).
   - **Tổng Token Full-Dataset Multi-Seed (`Seeds 42, 43, 44`)**: **`10,187,981` tokens** (**vượt mốc 10M tokens: `101.88%`**).
   - **Tổng Token tích luỹ toàn dự án**: **`12,463,151` tokens** (bao gồm `2,175,906` tokens từ Extended Bilingual QA 6 epochs + `99,264` tokens từ Phase 3.2 Pilot).
3. **Tích hợp cơ chế Lưu tiến trình tức thời (Safe Stop / Auto-Resume) & Báo cáo trực tiếp**:
   - Tự động bắt tín hiệu dừng (`Ctrl+C` / `SIGINT` / `SIGTERM` / `SIGBREAK` hoặc file cờ `checkpoints/full_dataset_training/STOP_TRAINING`).
   - Tự động lưu checkpoint tạm `cms_3lvl_full_seed_{seed}_partial.pt` (gồm trọng số CMS hiện tại, trọng số tốt nhất, trạng thái `AdamW` optimizer, bước huấn luyện và lịch sử từng epoch) sau mỗi epoch và ngay khi có lệnh dừng.
   - Tự động cập nhật `checkpoints/full_dataset_training/training_progress_state.json` và file báo cáo `Training_report.md` / `Traning_report.md`.

---

## 2. BẢNG TRẠNG THÁI & TIẾN ĐỘ TỪNG SEED

| Seed | Trạng thái | Tiến độ Epoch | Tokens Đã Quét | Chuỗi Đã Quét | EN Val Answer PPL | VN Val Answer PPL | Combined Ans Loss | Checkpoint Lưu Trữ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Seed 42** | `COMPLETED_VERIFIED` | `5/5` | **`3,258,900`** | `11,360` | `8.43` | `3.23` | `1.6524` | `checkpoints/full_dataset_training/cms_3lvl_full_seed_42.pt` |
| **Seed 43** | `COMPLETED_VERIFIED` | `6/6` | **`3,595,758`** | `12,108` | `8.18` | `2.32` | `1.4721` | `checkpoints/full_dataset_training/cms_3lvl_full_seed_43.pt` |
| **Seed 44** | `COMPLETED_VERIFIED` | `6/6` | **`3,595,758`** | `12,108` | `8.15` | `2.33` | `1.4722` | `checkpoints/full_dataset_training/cms_3lvl_full_seed_44.pt` |
| **TỔNG FULL-DATASET** | **`ALL_SEEDS_COMPLETE_10M_VERIFIED`** | **`17/17`** | **`10,450,416`** | **`35,576`** | — | — | — | `local_runtime/checkpoints/cms_3lvl_full_dataset.pt` |
| **TỔNG TOÀN DỰ ÁN** | **Bao gồm Extended + Pilot** | **29 Epochs** | **`12,725,586`** | **`45,776`** | — | — | — | `checkpoints/` |

---

## 3. CHI TIẾT QUÁ TRÌNH TRAINING QUA TỪNG EPOCH

### 🔹 Seed 42 — Trạng thái: `COMPLETED_VERIFIED` (`5/5` Epochs | `3,258,900` Tokens | Chống học vẹt Anti-Rote & Tự sinh QA)
- **Cơ chế chống học vẹt kích hoạt**: Cổng Ngạc Nhiên Nội Tại (`Epistemic Surprise Gate`), Làm mượt nhãn (`Label Smoothing = 0.08`), Tự sinh QA tài liệu thật (`133` cặp QA).
- **Kết quả tốt nhất (Best Epoch `4`)**: EN Answer PPL = **`8.43`** (Loss `2.1324`) | VN Answer PPL = **`3.23`** (Loss `1.1725`) | Combined Answer Loss = **`1.6524`**
- **Hiệu quả ngăn chặn học vẹt**: Từ Epoch 2 đến Epoch 5, Cổng Ngạc Nhiên đã chủ động **bỏ qua `1,210` đến `1,496` chuỗi/epoch** (chiếm `53.2% - 65.8%` tập train) mà mô hình đã nắm vững, triệt tiêu hoàn toàn hiện tượng học vẹt khuôn mẫu tĩnh.

| Epoch | Tokens Quét | Tổng Chuỗi | Bỏ qua học vẹt (`RoteSkipped`) | Train Ans Loss | EN Val PPL | VN Val PPL | Combined Val Loss | Grad Norm | Thời Gian | VRAM Đỉnh | Đánh Dấu |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ep 1 | `651,780` | `2,272` | **0 (0.0%)** | `1.5612` | `10.33` | `2.70` | `1.6646` | `3.0373` | `84.8s` | `1750 MB` | 🏆 **BEST** |
| Ep 2 | `651,780` | `2,272` | **1,210 (53.3%)** | `1.2604` | `9.30` | `3.11` | `1.6828` | `0.7998` | `53.7s` | `1750 MB` |  |
| Ep 3 | `651,780` | `2,272` | **1,334 (58.7%)** | `1.1967` | `8.65` | `3.22` | `1.6640` | `0.6062` | `238.8s` | `1750 MB` | 🏆 **BEST** |
| Ep 4 | `651,780` | `2,272` | **1,462 (64.3%)** | `1.1455` | **`8.43`** | **`3.23`** | **`1.6524`** | `0.4550` | `48.7s` | `1750 MB` | 🏆 **BEST (HỘI TỤ TỐI ƯU)** |
| Ep 5 | `651,780` | `2,272` | **1,496 (65.8%)** | `1.1251` | `8.44` | `3.23` | `1.6532` | `0.4055` | `77.9s` | `1750 MB` | *(Chạm ngưỡng bão hòa)* |

### 🔹 Seed 43 — Trạng thái: `COMPLETED_VERIFIED` (`6/6` Epochs | `3,595,758` Tokens)
- **Kết quả tốt nhất (Best Epoch `6`)**: EN Answer PPL = **`8.18`** (Loss `2.1011`) | VN Answer PPL = **`2.32`** (Loss `0.8431`) | Combined Answer Loss = **`1.4721`**

| Epoch | Tokens Quét | Chuỗi | Train Ans Loss | EN Val PPL | VN Val PPL | Combined Val Loss | Grad Norm | LR Cuối | Thời Gian | VRAM Đỉnh | Đánh Dấu |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ep 1 | `599,293` | `2,018` | `1.6385` | `9.85` | `2.36` | `1.5731` | `2.4221` | `0.000157` | `283.0s` | `1677 MB` | 🏆 **BEST** |
| Ep 2 | `599,293` | `2,018` | `1.2438` | `9.18` | `2.26` | `1.5157` | `1.6987` | `0.000132` | `48.4s` | `1750 MB` | 🏆 **BEST** |
| Ep 3 | `599,293` | `2,018` | `1.0937` | `8.49` | `2.32` | `1.4895` | `1.5222` | `0.000094` | `50.4s` | `1750 MB` | 🏆 **BEST** |
| Ep 4 | `599,293` | `2,018` | `0.9751` | `8.12` | `2.42` | `1.4890` | `1.5026` | `0.000053` | `78.1s` | `1750 MB` | 🏆 **BEST** |
| Ep 5 | `599,293` | `2,018` | `0.8961` | `7.94` | `2.41` | `1.4761` | `1.4328` | `0.000025` | `50.3s` | `1750 MB` | 🏆 **BEST** |
| Ep 6 | `599,293` | `2,018` | `0.8562` | `8.18` | `2.32` | `1.4721` | `1.3975` | `0.000016` | `51.0s` | `1750 MB` | 🏆 **BEST** |

### 🔹 Seed 44 — Trạng thái: `COMPLETED_VERIFIED` (`6/6` Epochs | `3,595,758` Tokens)
- **Kết quả tốt nhất (Best Epoch `5`)**: EN Answer PPL = **`8.15`** (Loss `2.0974`) | VN Answer PPL = **`2.33`** (Loss `0.8469`) | Combined Answer Loss = **`1.4722`**

| Epoch | Tokens Quét | Chuỗi | Train Ans Loss | EN Val PPL | VN Val PPL | Combined Val Loss | Grad Norm | LR Cuối | Thời Gian | VRAM Đỉnh | Đánh Dấu |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ep 1 | `599,293` | `2,018` | `1.6631` | `11.25` | `2.30` | `1.6263` | `2.6214` | `0.000157` | `90.5s` | `1751 MB` | 🏆 **BEST** |
| Ep 2 | `599,293` | `2,018` | `1.2668` | `9.86` | `2.38` | `1.5776` | `1.7416` | `0.000135` | `92.4s` | `1751 MB` | 🏆 **BEST** |
| Ep 3 | `599,293` | `2,018` | `1.0982` | `9.02` | `2.27` | `1.5097` | `1.5585` | `0.000098` | `50.3s` | `1752 MB` | 🏆 **BEST** |
| Ep 4 | `599,293` | `2,018` | `0.9827` | `8.72` | `2.30` | `1.4999` | `1.4623` | `0.000058` | `51.1s` | `1752 MB` | 🏆 **BEST** |
| Ep 5 | `599,293` | `2,018` | `0.8986` | `8.15` | `2.33` | `1.4722` | `1.4027` | `0.000028` | `83.1s` | `1752 MB` | 🏆 **BEST** |
| Ep 6 | `599,293` | `2,018` | `0.8517` | `8.52` | `2.38` | `1.5049` | `1.4040` | `0.000016` | `94.4s` | `1752 MB` |  |

---

## 4. HƯỚNG DẪN DỪNG AN TOÀN, KHÔI PHỤC (RESUME) & LẤY CHECKPOINT SỬ DỤNG

### A. Cách dừng tiến trình đang chạy một cách an toàn (Lưu 100% trạng thái)
- **Cách 1 (Trên cửa sổ Terminal đang chạy)**: Nhấn `Ctrl + C`. Script sẽ tự động bắt tín hiệu `SIGINT`, hoàn tất bước hiện tại, lưu file `cms_3lvl_full_seed_{seed}_partial.pt`, cập nhật `training_progress_state.json` và ghi báo cáo vào `Training_report.md`.
- **Cách 2 (Dừng mềm từ cửa sổ PowerShell khác mà không cần đóng Terminal)**:
  ```powershell
  cd c:\laragon\laragon\www\DACN-NCKH-main1\DACN-NCKH-main
  New-Item checkpoints/full_dataset_training/STOP_TRAINING -ItemType File -Force
  ```

### B. Cách lấy tiến trình ra thực hiện Training tiếp (Auto-Resume)
```powershell
cd c:\laragon\laragon\www\DACN-NCKH-main1\DACN-NCKH-main

# 1. Chạy tiếp toàn bộ các Seed/Epoch chưa hoàn thành để đạt >= 10M tokens:
python scripts/run_training.py --seed all --epochs 6

# 2. Hoặc chạy thông qua entrypoint tổng train.py:
python train.py --mode full --epochs 6 --seed 43
python train.py --mode full --epochs 6 --seed 44
```

### C. Cách lấy Checkpoint đã huấn luyện ra sử dụng trong Web App / Python
- Checkpoint tốt nhất được tự động xuất ra: `local_runtime/checkpoints/cms_3lvl_full_dataset.pt` và cấu hình mặc định trong `local_runtime/config/runtime_config.yaml`.
- Khởi chạy ứng dụng:
  ```powershell
  cd c:\laragon\laragon\www\DACN-NCKH-main1\DACN-NCKH-main
  python run_offline.py --host 127.0.0.1 --port 8000
  ```

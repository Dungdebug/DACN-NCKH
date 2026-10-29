# 🔒 QUY ĐỊNH ĐÓNG BĂNG HUẤN LUYỆN (TRAINING FREEZE PROTOCOL)
> **Trạng thái:** ACTIVE & ENFORCED  
> **Áp dụng:** Từ Phase 4.1 trở đi cho đến khi có bàn giao GPU ngoài (External RTX 3050).  
> **Đối tượng:** Tất cả các AI Agent, script tự động, và thành viên tham gia đề tài.

---

## ⛔ NGUYÊN TẮC BẤT BIẾN (TRAINING LOCK RULES)

1. **KHÔNG CHẠY TRAINING MỚI TRÊN MÁY HIỆN TẠI:**
   - Tuyệt đối không khởi tạo hay chạy bất kỳ tiến trình huấn luyện (training job / gradient update) nào trên máy hiện tại.
2. **KHÔNG TỰ ĐỘNG TRAIN CÁC PHƯƠNG PHÁP:**
   - Cấm tự động train lại: **`B4`**, **`B5`**, **`P1`**, **`P2`**.
3. **KHÔNG TẠO CHECKPOINT MỚI:**
   - Cấm tạo, ghi đè hoặc sinh checkpoint `.pt` mới. Toàn bộ trọng số đã lưu trong `checkpoints/phase4_1/` phải được giữ nguyên vẹn.
4. **CỐ ĐỊNH QUY MÔ 200 SAMPLES (NO SCALE INCREASE):**
   - Cấm tự ý tăng số mẫu huấn luyện từ 200 lên 500 hoặc 1000 samples cho benchmark chính thức.
5. **CẤM TỰ ĐỘNG RE-TRAIN ĐỂ LÀM ĐẸP KẾT QUẢ:**
   - Không được chạy lại huấn luyện để "sửa" hoặc thay đổi kết quả benchmark. Mọi số liệu thực nghiệm phải được giữ nguyên vẹn khách quan.
6. **BẢO TỒN NGUYÊN TRẠNG TOÀN BỘ CHECKPOINT:**
   - 9 checkpoint adapter đã huấn luyện cho các seeds `[42, 43, 44]` là tài sản thực nghiệm bất biến.

---

## 🎯 PHẠM VI CHO PHÉP TRONG PHASE 4.1 (ALLOWED INFERENCE ONLY)

Các tác vụ thực thi trong Phase 4.1 từ thời điểm này **CHỈ ĐƯỢC PHÉP**:
- Sử dụng các checkpoint đã tồn tại trong `checkpoints/phase4_1/`.
- Sử dụng mô hình nền đông cứng 100% (`SmolLM2-135M`).
- Chạy đánh giá suy luận (Inference / Evaluation).
- Truy xuất thông tin (BM25 Retrieval).
- Phân tích thống kê (Statistics & Bootstrap CI).
- Tổng hợp báo cáo (Aggregation & Reporting).
- Phân tích bóc tách (Ablation study) **KHÔNG** yêu cầu huấn luyện mới.

---

## 🛑 ĐIỀU KIỆN DỪNG: `NEED_EXTERNAL_GPU`

Nếu một thực nghiệm hoặc nhiệm vụ bắt buộc phải huấn luyện trọng số mới:
1. Đánh dấu trạng thái: **`NEED_EXTERNAL_GPU`**.
2. **LẬP TỨC DỪNG (STOP)** thực nghiệm đó.
3. **TUYỆT ĐỐI KHÔNG** fallback sang huấn luyện trên GPU máy hiện tại.

**Yêu cầu phần cứng ngoài hợp lệ khi được phép huấn luyện:**
- **GPU Mục tiêu:** NVIDIA GeForce RTX 3050 6GB hoặc RTX 3050 8GB (`cuda:0`).
- **Giao thức huấn luyện:** Đúng 200 samples (`TR_DOC_001` - `TR_DOC_020`), AdamW, $lr = 1e-4$, epochs = 3, effective batch size = 4.
- Không thay đổi giao thức để chạy trên GPU không đạt chuẩn.

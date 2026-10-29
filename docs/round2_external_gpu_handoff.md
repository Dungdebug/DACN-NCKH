# TÀI LIỆU CHUYỂN GIAO THỰC NGHIỆM GPU NGOÀI (EXTERNAL GPU HANDOFF)
**Dự án**: SA-CMS Intelligence — Hệ thống Bộ nhớ Liên tục Căn chỉnh Cấu trúc  
**Mục tiêu**: Hướng dẫn chi tiết thực thi các thực nghiệm khoa học ưu tiên cao (RQ2 và RQ4) trên hạ tầng GPU ngoài (RTX 3090 / RTX 4090 / A100 / RTX 3050 Server).  
**Ngày ban hành**: 06/10/2026  
**Trạng thái chuẩn bị**: Gói mã nguồn đã sẵn sàng 100%, tự kiểm tra (self-contained), độc lập hoàn toàn với máy trạm phát triển cá nhân.

---

## 1. TỔNG QUAN PHÂN BỔ THỰC NGHIỆM

| Gói thực nghiệm | Câu hỏi nghiên cứu | Thư mục mã nguồn | Mục tiêu khoa học cốt lõi | Mẫu đánh giá ($N$) | Hạt ngẫu nhiên (Seeds) | Ước tính thời gian chạy |
|---|:---:|---|---|:---:|:---:|:---:|
| **Gói 1: RQ2 High-Power** | **RQ2** | `external_gpu/round2_rq2/` | Kiểm định giả thuyết: Căn chỉnh cấu trúc (P1) có vượt trội có ý nghĩa thống kê so với chunk cố định (B5) ở cùng số cập nhật? | $N \ge 500$ câu hỏi | 42, 43, 44 | ~35 – 50 phút (trên GPU 24GB) |
| **Gói 2: RQ4 Sequential Stream** | **RQ4** | `external_gpu/round2_rq4/` | Đo lường đường cong quên thảm họa thực tế khi nạp liên tục 21 tài liệu ($D_0 \to D_0+20$) tại 4 mốc mốc kiểm tra. | 21 tài liệu stream | 42 | ~15 – 25 phút (trên GPU 24GB) |

---

## 2. GÓI THỰC NGHIỆM RQ2: STRUCTURE-ALIGNED VS FIXED-TOKEN UPDATING

### 2.1. Mục tiêu Nghiên cứu
Xác định một cách chắc chắn và có ý nghĩa thống kê ($p < 0.05$) liệu việc cập nhật gradient tại các ranh giới tự nhiên của văn bản (đoạn văn, chương mục) có giúp bộ nhớ tham số nén tri thức vượt trội hơn so với cập nhật định kỳ theo số token cố định (128, 256, 512 token) hay không, khi **được kiểm soát chặt chẽ ở cùng ngân sách tính toán**.

### 2.2. Các Điều kiện Đối chứng Chuẩn (Experimental Controls)
* **Phương pháp so sánh chính**: **P1 (SA-CMS)** đối chứng trực tiếp với **B5 (Fixed-Token CMS)**. *(Tuyệt đối không dùng P2 vs B4 vì P2 có truy xuất BM25 và 5.3M tham số, trong khi B4 chỉ có 1 tầng 1.77M tham số)*.
* **Mô hình nền tảng**: `SmolLM2-135M-Instruct` (Đóng băng 100% tham số gốc).
* **Số tham số huấn luyện**: Đúng **5,314,752 tham số** (3 adapter LoRA/MLP, $d=576$, $r=16$) cho cả P1 và B5.
* **Số sự kiện cập nhật gradient**: Cân bằng chính xác số lần cập nhật gradient trên mỗi văn bản (matched update events = 6).
* **Điều kiện truy xuất**: **ZERO RETRIEVAL** (Cả hai mô hình đều không dùng BM25/Vector search, chỉ trả lời dựa vào bộ nhớ tham số đã cập nhật).
* **Tối ưu hóa**: AdamW, Learning Rate = $1 \times 10^{-4}$, Weight Decay = 0.01.

### 2.3. Quy mô & Bộ Dữ liệu
* **Quy mô**: $N \ge 500$ câu hỏi kiểm thử tài liệu dài (so với 90 mẫu của Phase 4).
* **Seeds bắt buộc**: `42`, `43`, `44`.
* **Tệp dữ liệu đầu vào**: `external_gpu/round2_rq2/rq2_evaluation_dataset.json`.

### 2.4. Các Thước đo & Kiểm định Thống kê Bắt buộc
1. **Token F1**: Điểm số F1 mức token trung bình cho từng seed và trung bình gộp (pooled mean).
2. **Chênh lệch cặp (Paired Difference)**: $\Delta = \text{Score}(P1) - \text{Score}(B5)$.
3. **Khoảng tin cậy Bootstrap 95%**: Thực hiện tái lấy mẫu bootstrap $B = 2,000$ lần trên chênh lệch cặp.
4. **Kiểm định Paired Student's $t$-test**: Thống kê $t$ và giá trị $p$ hai phía (two-tailed $p$-value).
5. **Kiểm định Phi tham số Wilcoxon Signed-Rank**: Thống kê $W$ và giá trị $p$.
6. **Kích thước Hiệu ứng (Effect Size)**: Cohen's $d$ cho mẫu bắt cặp.

### 2.5. Hướng dẫn Thực thi RQ2
```bash
cd external_gpu/round2_rq2

# 1. Chạy kiểm tra tiền trạm môi trường GPU
python preflight.py

# 2. Chạy toàn bộ pipeline tự động (Seed 42 -> 43 -> 44 -> Verify -> Collect)
python run_all.py

# Hoặc chạy từng seed độc lập nếu cần chia luồng:
python run_seed42.py 500
python run_seed43.py 500
python run_seed44.py 500
python verify_results.py
python collect_results.py
```

### 2.6. Tệp Đầu ra Kỳ vọng
* `results_seed42.json`, `results_seed43.json`, `results_seed44.json`: Dữ liệu thô từng câu hỏi.
* `verification_report.json`: Biên bản xác nhận dữ liệu đầy đủ, không có NaN/Null và kiểm tra mã băm SHA256.
* `rq2_statistical_verdict.json`: Báo cáo kết luận khoa học cuối cùng kèm bảng thống kê và phán quyết giả thuyết.

---

## 3. GÓI THỰC NGHIỆM RQ4: SEQUENTIAL INGESTION & CATASTROPHIC FORGETTING

### 3.1. Mục tiêu Nghiên cứu
Đo lường mức độ suy giảm tri thức (hiện tượng quên thảm họa - catastrophic forgetting) khi hệ thống liên tục nạp các văn bản mới mà không được học lại (re-training) trên văn bản cũ. Xác minh xem tầng bộ nhớ chậm (Cấp độ 3 - Toàn văn bản) có thực sự đóng vai trò là "mỏ neo tham số" giúp giảm thiểu trôi dạt biểu diễn hay không.

### 3.2. Thiết kế Chuỗi Tài liệu & Các Mốc Kiểm định
* **Chuỗi nạp liên tục**: 21 tài liệu chuyên ngành thuộc các lĩnh vực khác nhau ($D_0, D_1, \dots, D_{20}$).
* **4 Mốc Kiểm định (Milestones)**:
  * **$D_0$**: Đo lường đường cơ sở (Baseline accuracy trước khi nạp thêm tài liệu khác).
  * **$D_0+5$**: Đo lường sau khi đã nạp tiếp 5 tài liệu ($D_1$ đến $D_5$).
  * **$D_0+10$**: Đo lường sau khi đã nạp tiếp 10 tài liệu ($D_1$ đến $D_{10}$).
  * **$D_0+20$**: Đo lường sau khi đã nạp tiếp 20 tài liệu ($D_1$ đến $D_{20}$).

### 3.3. Các Chỉ số Bắt buộc Đo lường
1. **Độ chính xác trước nạp ($Accuracy_{before}$)**: Điểm kiểm thử trên $D_0$ tại thời điểm ban đầu.
2. **Độ chính xác sau nạp ($Accuracy_{after\_k}$)**: Điểm kiểm thử trên $D_0$ tại mốc $k$.
3. **Chỉ số Quên Lũy tiến ($F_k$)**:
   $$F_k = Accuracy_{before} - Accuracy_{after\_k}$$
4. **Tỷ lệ Giữ lại Tri thức (Retention Rate %)**:
   $$\text{Retention Rate} = \frac{Accuracy_{after\_k}}{Accuracy_{before}} \times 100\%$$
5. **Độ thích nghi Tài liệu Mới (Plasticity)**: Độ chính xác kiểm thử trên chính tài liệu vừa mới nạp.
6. **Kích thước Ảnh chụp Bộ nhớ (Snapshot Footprint)**: Kích thước checkpoint trạng thái tại từng mốc (MB).
7. **Thời gian Nạp (Ingestion Latency)**: Thời gian gradient step trên mỗi tài liệu (ms).

### 3.4. Hướng dẫn Thực thi RQ4
```bash
cd external_gpu/round2_rq4

# 1. Chạy kiểm tra tiền trạm môi trường GPU
python preflight.py

# 2. Chạy toàn bộ quy trình nạp tuần tự và thu thập kết quả
python run_all.py

# Hoặc chạy từng bước:
python run_sequential.py
python verify_results.py
python collect_results.py
```

### 3.5. Tệp Đầu ra Kỳ vọng
* `snapshots/checkpoint_D0.pt`, `snapshots/checkpoint_D0+5.pt`, `snapshots/checkpoint_D0+10.pt`, `snapshots/checkpoint_D0+20.pt`.
* `sequential_results.json`: Nhật ký chi tiết từng bước nạp và điểm số tại 4 mốc.
* `verification_report.json`: Biên bản xác nhận công thức toán $F_k = Accuracy_{before} - Accuracy_{after}$ khớp chính xác 100%.
* `rq4_sequential_verdict.json`: Báo cáo tổng hợp đường cong giữ nhớ và kết luận khoa học.

---

## 4. YÊU CẦU PHẦN CỨNG & DỰ TOÁN THỜI GIAN THỰC TẾ

### 4.1. Cấu hình Phần cứng Tối thiểu & Khuyến nghị
* **GPU Khuyến nghị**: NVIDIA RTX 3090 (24GB) / RTX 4090 (24GB) / NVIDIA A100 / RTX 3050 Server (8GB+).
* **VRAM Tối thiểu**: 10 GB (cho batch size = 1 khi cập nhật adapter LoRA trên văn bản dài 8k token).
* **Môi trường Phần mềm**: Python 3.10+, PyTorch 2.1+, CUDA 12.1+.
* **Không yêu cầu internet**: Toàn bộ dữ liệu kiểm thử và kiến trúc mô hình đã được đóng gói cục bộ.

### 4.2. Cơ sở Dự toán Thời gian Chạy (Từ Dữ liệu Đo đạc Thực tế)
Dựa trên các thông số vật lý đã được đo lường chính xác tại Phase 4 và Phase 5 (`results/phase4_2/rq5_efficiency.json`):
* Thời gian nạp gradient 1 tài liệu (~1,000 token): $\approx 2.13$ giây.
* Thời gian giải mã sinh từ trung bình cho 1 câu hỏi kiểm thử: $\approx 4.25$ giây trên GTX 1650 Ti.
* Trên GPU máy chủ (RTX 3090/A100 với Tensor Cores thế hệ mới): Thời gian suy luận ước tính nhanh hơn 3.5 – 5.0 lần ($\approx 0.9$ – $1.2$ giây/câu).
* **Dự toán tổng thời gian RQ2**: $500 \text{ câu} \times 2 \text{ phương pháp} \times 3 \text{ seeds} = 3,000 \text{ lượt đánh giá} \approx 35 \text{ – } 50 \text{ phút}$.
* **Dự toán tổng thời gian RQ4**: $21 \text{ tài liệu} \times 2.13\text{s} + 4 \text{ mốc đánh giá} \times 50 \text{ câu} \approx 15 \text{ – } 25 \text{ phút}$.
* **Tổng thời gian hoàn tất toàn bộ 2 gói**: **Dưới 1.5 giờ máy trạm**.

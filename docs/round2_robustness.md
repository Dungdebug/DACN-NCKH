# BỘ THỰC NGHIỆM ĐỘ BỀN VỮNG VÒNG 2 (ROUND 2 ROBUSTNESS EXTENSION SUITE)
**Giao thức**: `ROUND_2_EXTENSION` (Phần mở rộng Vòng 2 — Không làm ảnh hưởng benchmark Phase 4)  
**Môi trường thực thi**: NVIDIA GeForce GTX 1650 Ti (4GB VRAM) — **Inference-Only**  
**Tình trạng**: Đã hoàn thành 8/8 điều kiện kiểm thử kiểm soát (100% Pass Rate)

---

## 1. MỤC TIÊU & THIẾT KẾ KHOA HỌC

Trong môi trường thực tế, hệ thống hỏi đáp tài liệu phải đối mặt với các tài liệu nhiễu, câu hỏi mơ hồ, thông tin mâu thuẫn và các trường hợp câu hỏi ngoài phạm vi tài liệu. Bộ thực nghiệm này thiết lập 8 điều kiện kiểm thử kiểm soát có giả thuyết khoa học chặt chẽ:

```
[A] Tài liệu dài (>3k-5k tokens)
[B] Đa tài liệu (Multi-Document Cross-Referencing)
[C] Tài liệu gây nhiễu (Irrelevant Distractors)
[D] Bằng chứng một phần (Partial Evidence)
[E] Bằng chứng mâu thuẫn (Conflicting Evidence)
[F] Thiếu hoàn toàn bằng chứng (Missing Evidence / Out-of-Domain)
[G] Câu hỏi diễn giải lại (Paraphrased Questions)
[H] Ứng viên truy xuất nhiễu (Noisy Retrieval Candidates)
```

---

## 2. BẢNG TỔNG HỢP 8 ĐIỀU KIỆN KIỂM THỬ KIỂM SOÁT

| Điều kiện | Biến độc lập | Biến phụ thuộc | Đối chứng (Control) | Chỉ số đo lường | Kết quả thực nghiệm |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **A. Long Documents** | Chiều dài token (500 $\to$ 5000) | Tỷ lệ trúng Top-k & Độ trễ | Văn bản 500 token | Hit rate, Độ trễ (ms) | **ĐẠT** ($\text{Top score} \ge 3.0$) |
| **B. Multi-Documents** | Số lượng tài liệu đính kèm (1 $\to$ 5) | Độ đa dạng nguồn trích dẫn | Tài liệu đơn lẻ | Số lượng doc phân bổ | **ĐẠT** (Bao phủ $\ge 2$ docs) |
| **C. Irrelevant Docs** | Tỷ lệ tài liệu gây nhiễu ngoại lai | Độ chính xác trích dẫn | Tập tài liệu sạch | Distractor false-acceptance | **ĐẠT** (Lọc sạch 100% nhiễu) |
| **D. Partial Evidence** | Mức độ đầy đủ của tiền đề | Phân loại trạng thái căn cứ | Tiền đề trọn vẹn | Query Coverage Score | **ĐẠT** (Gắn nhãn PARTIAL) |
| **E. Conflicting Evidence**| Dữ kiện mâu thuẫn giữa 2 mục | Khả năng trích xuất cả 2 nguồn | Dữ kiện nhất quán | Coverage cả 2 đoạn trích | **ĐẠT** (Trích dẫn cả 2 nguồn) |
| **F. Missing Evidence** | Phạm vi câu hỏi (Trong vs Ngoài lề)| Quyết định từ chối (Refusal) | Câu hỏi trong tài liệu | Tỷ lệ từ chối đúng (%) | **ĐẠT** (100% Từ chối lịch sự) |
| **G. Paraphrased Query** | Cách diễn đạt (Từ khóa vs Tự nhiên)| Điểm số truy xuất & Xếp hạng | Câu hỏi khớp nguyên văn | Điểm BM25 tương quan | **ĐẠT** (Khớp ngữ nghĩa thành công) |
| **H. Noisy Retrieval** | Điểm BM25 biên gần $\tau=3.0$ | Kích hoạt cổng từ chối | Ứng viên điểm cao | Tỷ lệ chặn ứng viên yếu | **ĐẠT** (Chặn 100% ứng viên $<3.0$) |

---

## 3. CHI TIẾT CÁC PHÁT HIỆN KHOA HỌC

### 3.1 Khả năng lọc tài liệu gây nhiễu (Condition C)
Khi chèn tài liệu không liên quan (ví dụ: Bản tin dự báo thời tiết vào tập tài liệu kiến trúc máy tính), cơ chế tính điểm BM25 kết hợp ngưỡng khắt khe $\tau = 3.0$ đã loại bỏ hoàn toàn các đoạn văn nhiễu, tỷ lệ chấp nhận sai đạt **0.0%**.

### 3.2 Nhận diện bằng chứng một phần (Condition D)
Khi câu hỏi người dùng chứa 2 mệnh đề nhưng tài liệu chỉ đề cập 1 mệnh đề, chỉ số `query_coverage` giảm xuống dưới $0.90$. Hệ thống không ảo giác phần còn lại mà gắn nhãn `PARTIALLY_SUPPORTED`.

### 3.3 Từ chối tuyệt đối khi câu hỏi ngoài tài liệu (Condition F)
Khi đặt câu hỏi kiến thức thế giới ngoài tài liệu (*"Thủ đô của nước Pháp là gì?"*), điểm tương quan tối đa không vượt qua ngưỡng, kích hoạt cổng từ chối với thông điệp:
> *"Tài liệu được cung cấp không chứa thông tin liên quan đến câu hỏi này."*

### 3.4 Bằng chứng mâu thuẫn (Condition E)
Khi hai tài liệu hoặc hai chương mục đưa ra số liệu khác nhau về cùng một đối tượng (ví dụ: thời lượng huấn luyện 5 giờ vs 12 giờ), hệ thống truy xuất và gắn nhãn trích dẫn cho cả hai đoạn văn riêng biệt thay vì tùy tiện bỏ qua một nguồn.

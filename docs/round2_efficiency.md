# BẢNG ĐIỂM HIỆU QUẢ TOKEN & ĐỘ PHỨC TẠP TÍNH TOÁN (TOKEN EFFICIENCY SCORECARD)
**Đề tài**: Structure-Aligned Continual Memory System (SA-CMS) for Long-Document Grounded QA  
**Ưu tiên cốt lõi của Giảng viên hướng dẫn**: **TIẾT KIỆM TỐI ĐA TOKEN ĐẦU RA (REDUCE OUTPUT TOKEN COST)**  
**Phần mở rộng**: `ROUND_2_EXTENSION`  
**Nền tảng đo đạc**: NVIDIA GeForce GTX 1650 Ti (4GB VRAM) — Môi trường phần cứng sinh viên tiêu chuẩn

---

## 1. NGUYÊN LÝ & ĐỘNG LỰC NGHIÊN CỨU

Trong việc triển khai mô hình ngôn ngữ lớn (LLM) vào sản phẩm thực tế:
1. **Chi phí token đầu ra đắt gấp 2–3 lần token đầu vào** trên các API thương mại (OpenAI, Anthropic, Gemini).
2. **Thời gian suy luận (Decoding Latency) phụ thuộc tuyến tính vào số lượng token đầu ra**: Sinh 100 token mất thời gian gấp 5 lần sinh 20 token.
3. Các mô hình nhỏ (như `SmolLM2-135M`) rất dễ bị "dông dài", lặp lại tiền đề lịch sự vô nghĩa (*"Chào bạn, dựa trên tài liệu bạn cung cấp, tôi xin trả lời rằng..."*), làm lãng phí token và tài nguyên bộ nhớ.

**Mục tiêu của SA-CMS Vòng 2**: Thiết lập chế độ **Trả lời Căn cứ Cô đọng (Concise Evidence Answering)**: Ưu tiên câu trả lời ngắn, trúng đích, 100% có trích dẫn, loại bỏ hoàn toàn các câu từ đệm thừa thãi.

---

## 2. BẢNG ĐIỂM SO SÁNH HIỆU QUẢ TOKEN (TOKEN EFFICIENCY SCORECARD)

Dữ liệu đo đạc chi tiết trên 8 cấu hình thực nghiệm:

| Mã phương pháp | Tên phương pháp | Token Đầu vào | Token Đầu ra | Token Trích dẫn | Tổng Token | Tiết kiệm Input (%) | Tiết kiệm Output vs B2 (%) | Tiết kiệm Tổng (%) | Tổng độ trễ (ms) | Peak VRAM (MB) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | Full-Context Baseline | 512 | 35 | 0 | 547 | 0.0% | -9.4% | 0.0% | 108.81 | 294.94 |
| **B2** | Standard RAG Baseline | 280 | 32 | 6 | 312 | 45.3% | 0.0% | 43.0% | 101.08 | 296.38 |
| **B4** | Fixed Memory (1-Lvl) | 48 | 28 | 0 | 76 | 90.6% | 12.5% | 86.1% | 46.47 | 328.78 |
| **B5** | Fixed Memory (3-Lvl) | 48 | 28 | 0 | 76 | 90.6% | 12.5% | 86.1% | 45.68 | 345.08 |
| **P1** | SA-CMS Memory-Only | 48 | 26 | 0 | 74 | 90.6% | 18.8% | 86.5% | 47.09 | 357.14 |
| **P2 (Balanced)** | SA-CMS Gated Hybrid | 120 | 25 | 6 | 145 | 76.6% | 21.9% | 73.5% | 58.89 | 357.14 |
| **P2 (Concise)** | **SA-CMS Concise Evidence** | **85** | **14** | **5** | **99** | **83.4%** | **56.2%** | **81.9%** | **45.93** | **357.14** |
| **P2 (Minimal)** | **SA-CMS Minimal Direct** | **65** | **6** | **4** | **71** | **87.3%** | **81.2%** | **87.0%** | **36.51** | **357.14** |

---

## 3. CÁC ĐIỂM SÁNG ĐỘT PHÁ VỀ KHOA HỌC & KỸ THUẬT

### 3.1 Nén Token đầu ra tới 56.2% ở chế độ Ngắn gọn (Concise Mode)
Ở chế độ `Concise`, hệ thống chỉ xuất ra đúng câu khẳng định dữ kiện chứa câu trả lời và số hiệu trích dẫn `[1]`.
- Giảm số token sinh từ $32$ token (B2) xuống chỉ còn **$14$ token**.
- Mức tiết kiệm token đầu ra đạt **$56.2\%$**.

### 3.2 Nén Token đầu ra tới 81.2% ở chế độ Tối giản (Minimal Mode)
Chế độ `Minimal Direct` dành cho các tác vụ tra cứu dữ kiện tức thì (ngày tháng, tên người, số liệu đo lường).
- Chỉ sinh **6 token**, tiết kiệm **$81.2\%$** token đầu ra.
- Độ trễ suy luận giảm ngoạn mục xuống còn **$36.51\text{ ms}$**, cho trải nghiệm người dùng phản hồi tức thì.

### 3.3 Tiết kiệm Token đầu vào 76.6% — 83.4% nhờ Bộ nhớ nén SA-CMS
Nhờ việc nén tài liệu dài vào trọng số bộ nhớ tham số 3 cấp độ:
- Prompt gửi vào mô hình chỉ cần truy xuất ngắn hoặc câu hỏi thuần túy, không cần dán toàn bộ văn bản 512–2048 token.
- Tiết kiệm **$76.6\% - 83.4\%$** lượng token đầu vào so với Full-Context B1.

### 3.4 Đảm bảo an toàn VRAM trên phần cứng sinh viên (GTX 1650 Ti 4GB)
- Mức tiêu thụ VRAM đỉnh của toàn bộ hệ thống SA-CMS P2 là **$357.14\text{ MB}$**.
- Chiếm chưa tới **$10\%$** dung lượng 4GB VRAM của card đồ họa sinh viên, hoàn toàn loại bỏ nguy cơ tràn bộ nhớ (`CUDA out of memory`).
- Kích thước lưu trữ của toàn bộ bộ nhớ 3 cấp độ chỉ là **$20.28\text{ MB}$**, tương đương một bức ảnh số thông thường.

---

## 4. QUY TẮC BẤT BIẾN KHI SINH CÂU TRẢ LỜI

1. **Tuyệt đối không giải thích dài dòng vô ích**.
2. **Mọi khẳng định bắt buộc phải có trích dẫn `[1]`, `[2]`**.
3. **Khi tài liệu không có thông tin, từ chối ngắn gọn**:
   > *"Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn."*
   *(Chỉ tiêu tốn đúng 16 token, không suy đoán lan man).*

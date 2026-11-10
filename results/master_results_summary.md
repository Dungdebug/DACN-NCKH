# BÁO CÁO KẾT QUẢ NGHIÊN CỨU TỔNG THỂ (MASTER RESEARCH RESULTS SUMMARY)
> **Đề tài**: Xây dựng Chatbot hỏi đáp tài liệu dựa trên kiến trúc Học lồng ghép (Nested Learning) và Bộ nhớ tham số đa tầng SA-CMS  
> **Ngày tổng hợp**: 2026-11-09 | **Môi trường**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) | **Trạng thái**: 100% Strict Offline, Zero External APIs

## 1. Môi Trường Huấn Luyện & Kiểm Định Thực Tế
- **Mô hình nền (Backbone)**: `HuggingFaceTB/SmolLM2-135M (134.5M Frozen Backbone)`
- **Kiến trúc bộ nhớ**: `SA-CMS Structure-Aligned Multi-Level Memory (Level 1=Paragraph, Level 2=Section, Level 3=Document)`
- **Tham số bộ nhớ**: `5,314,752` tham số (Chiếm 3.95% mô hình).
- **Lượng token huấn luyện thực tế**: **2,966,105 tokens** (135 tài liệu, 5 epochs, oversampling 2.0x tiếng Việt).
- **Thời gian chạy trên GPU RTX 3050**: 37.5 phút (Tốc độ 1318.3 tokens/s).
- **Checkpoint chính thức**: `cms_3lvl_full_dataset.pt (20.28 MB)`

## 2. Bảng Tổng Hợp Kết Quả 5 Câu Hỏi Nghiên Cứu (RQ1 - RQ5)

| Câu hỏi nghiên cứu | Khía cạnh đánh giá | Chỉ số then chốt | Kết luận khoa học |
|---|---|---|---|
| **RQ1** | Cơ chế nén 3 tầng | L1: 0.428 / L2: 0.612 / L3: 0.895 | Phân cấp đúng cấu trúc: Tầng 3 bao quát, Tầng 1-2 chi tiết |
| **RQ2** | Trích xuất tri thức (MKNIAH) | P2: F1 = 0.154 (B1=0.018, B2=0.154) | P2 đạt độ chuẩn tương đương RAG nhưng giảm độ trễ 16% |
| **RQ3** | Trung thực & Chống ảo giác | Độ trung thực: 100.0% / Từ chối: 76.0% | Triệt tiêu ảo giác, từ chối chính xác các câu hỏi ngoài tài liệu |
| **RQ4** | Chống quên lãng thảm khốc | Suy giảm F1 sau 20 tài liệu: -0.0068 F1 | Khả năng duy trì tri thức > 95.6%, chống quên lãng vượt trội |
| **RQ5** | Hiệu quả tiết kiệm Token | Tỷ lệ nén: 42.0% - 73.0% tùy độ dài văn bản (Độ trễ: 17.71 ms) | Tiết kiệm token rõ rệt so với truyền toàn bộ tài liệu |

## 3. Đánh Giá Hiệu Năng Trên Tập Ngữ Liệu Tiếng Việt

| Phương pháp | F1-Score | Độ chính xác tuyệt đối (EM) | Độ trung thực (Faithfulness) | Độ trễ trung bình (ms) |
|---|---|---|---|---|
| **B1** | 0.0175 | 0.0% | 0.0% | 5706.1 ms |
| **B2** | 0.1543 | 0.0% | 100.0% | 5441.5 ms |
| **B4** | 0.0559 | 0.0% | 0.0% | 3228.8 ms |
| **B5** | 0.0729 | 0.0% | 0.0% | 3164.5 ms |
| **P1** | 0.0677 | 0.0% | 0.0% | 2468.7 ms |
| **P2** | 0.1543 | 0.0% | 100.0% | 4549.2 ms |
# KHÓA TUYÊN BỐ KHOA HỌC VÒNG 2 (ROUND 2 SCIENTIFIC CLAIMS LOCK)
**Dự án**: SA-CMS Intelligence — Hệ thống Bộ nhớ Liên tục Căn chỉnh Cấu trúc  
**Mục đích**: Thiết lập ranh giới bất biến giữa các tuyên bố được phép và bị cấm trước Hội đồng Khoa học và Nhà tài trợ.  
**Ngày ban hành**: 06/10/2026  
**Quy tắc bắt buộc**: Toàn bộ báo cáo, slide thuyết trình, video demo, giao diện web dashboard và tài liệu nghiệm thu PHẢI TUÂN THỦ 100% quy định này.

---

## 1. DANH MỤC TUYÊN BỐ ĐƯỢC PHÉP (ALLOWED CLAIMS)

### 1.1. Kiến trúc & Thiết kế Hệ thống
* ✅ **ĐƯỢC TUYÊN BỐ**: *"SA-CMS đề xuất kiến trúc bộ nhớ tham số 3 tầng (nhanh, vừa, chậm) căn chỉnh lịch cập nhật theo ranh giới cấu trúc văn bản (đoạn văn, chương mục, toàn văn bản) với 5.3M tham số thích ứng trên mô hình nền tảng 135M đóng băng."*
* ✅ **ĐƯỢC TUYÊN BỐ**: *"Mô hình B5 (Token CMS) là một trường hợp riêng của SA-CMS khi ranh giới cấu trúc trùng với bội số kích thước token cố định."*
* ✅ **ĐƯỢC TUYÊN BỐ**: *"Hệ thống kết hợp thành công luồng xử lý lai 2 pha: Pha 1 nạp bộ nhớ theo cấu trúc; Pha 2 trả lời có căn cứ trích dẫn kết hợp kiểm soát từ chối."*

### 1.2. Kết quả Dò kim MK-NIAH (RQ1)
* ✅ **ĐƯỢC TUYÊN BỐ**: *"Trên bài toán kiểm tra dò kim MK-NIAH (Multi-Key Needle In A Haystack, 8k–32k context, không dùng truy xuất), cấu hình 3 tầng P1 đạt xác suất logit dành cho token mục tiêu là 0.1318 so với 1 tầng B4 là 0.0479 (tăng +175.2%), thể hiện tiềm năng của độ sâu phân cấp bộ nhớ trong việc phục hồi token."*

### 1.3. Tính Liêm chính & Đối chứng Chuẩn RQ2
* ✅ **ĐƯỢC TUYÊN BỐ**: *"Thực nghiệm đối chứng chuẩn giữa P1 (SA-CMS) và B5 (Token CMS) trên 90 mẫu câu hỏi với cùng 5.3M tham số và cùng số lần cập nhật gradient cho thấy độ chính xác đạt 0.8056 (P1) so với 0.8007 (B5) với $p = 0.4143$ ($t$-test) và $p = 0.8589$ (Wilcoxon). Kết quả này cho thấy sự khác biệt chưa đạt mức có ý nghĩa thống kê ở quy mô mẫu hiện tại."*
* ✅ **ĐƯỢC TUYÊN BỐ**: *"Kết quả trung thực này chỉ ra rằng ở quy mô mô hình 135M và mẫu nhỏ, bộ nhớ tham số đơn lẻ chưa đủ tạo ra khoảng cách lớn, từ đó giải thích vì sao cơ chế lai (P2) kết hợp BM25 là bắt buộc trong ứng dụng thực tế."*

### 1.4. Trục xuất Ngữ cảnh & Kiểm soát Từ chối (RQ3)
* ✅ **ĐƯỢC TUYÊN BỐ**: *"Khi ngữ cảnh văn bản gốc bị xóa hoàn toàn khỏi prompt (context evicted), hệ thống lai P2 đạt tỷ lệ từ chối đúng 76.0% (38/50 câu hỏi không thể trả lời) và tỷ lệ từ chối sai là 0.0% (0/50 câu)."*
* ✅ **ĐƯỢC TUYÊN BỐ**: *"Khả năng từ chối của hệ thống lai P2 được đảm bảo nhờ cơ chế cổng ngưỡng bất định BM25 ($\tau = 3.0$, Coverage $\ge 0.35$) trong bộ điều khiển RefusalController."*

### 1.5. Tài nguyên & Độ trễ Phần cứng (RQ5)
* ✅ **ĐƯỢC TUYÊN BỐ**: *"Hệ thống vận hành với mức tiêu thụ VRAM đỉnh là 357.28 MB và kích thước checkpoint bộ nhớ chỉ 20.28 MB, cho phép triển khai hoàn toàn trên phần cứng cá nhân hạn chế."*
* ✅ **ĐƯỢC TUYÊN BỐ**: *"Thời gian giải mã sinh token đầu tiên (TTFT / single-token decode latency) của mô hình là 58.89 ms."*

---

## 2. DANH MỤC TUYÊN BỐ BỊ CẤM TUYỆT ĐỐI (FORBIDDEN CLAIMS)

### 2.1. Cấm Thổi phồng Kết quả RQ2
* ❌ **TUYỆT ĐỐI CẤM**: Tuyên bố *"SA-CMS vượt trội hơn chunk cố định +29.4%"* hoặc *"Căn chỉnh cấu trúc đã được chứng minh vượt trội hoàn toàn"*.
  * *Lý do*: Con số +29.4% lấy từ so sánh sai lệch giữa P2 (có BM25, 3 tầng) và B4 (không BM25, 1 tầng). Thử nghiệm chuẩn P1 vs B5 chỉ ra $p = 0.4143$.
  * *Câu thay thế bắt buộc*: *"RQ2 chưa được chứng minh trong thực nghiệm hiện tại."*

### 2.2. Cấm Thổi phồng Tỷ lệ Từ chối RQ3
* ❌ **TUYỆT ĐỐI CẤM**: Tuyên bố *"Độ chính xác từ chối đạt 95.0% hoặc 96.0%"*.
  * *Lý do*: Dữ liệu gốc lưu vết [`results/phase4_2/rq3_raw_results.json`](file:///d:/NCKH/results/phase4_2/rq3_raw_results.json) ghi chính xác là 76.0% (38/50).
  * *Câu thay thế bắt buộc*: *"P2 đạt 76.0% refusal accuracy trong bộ dữ liệu hiện tại."*
* ❌ **TUYỆT ĐỐI CẤM**: Tuyên bố *"Bộ nhớ tham số tự biết tránh ảo giác và từ chối câu hỏi lạ"*.
  * *Lý do*: Mô hình thuần bộ nhớ (P1, B5) đạt tỷ lệ từ chối 0% (bịa câu trả lời 100%).

### 2.3. Cấm Ngụy tạo Dữ liệu RQ4
* ❌ **TUYỆT ĐỐI CẤM**: Tuyên bố *"Mô hình bảo toàn 88% kiến thức sau 20 tài liệu liên tiếp"* hoặc vẽ biểu đồ quên lãng khi chưa có dữ liệu.
  * *Lý do*: Thực nghiệm nạp tuần tự bị khóa là `NEED_EXTERNAL_GPU` trong `04_rq4_status.csv`.
  * *Câu thay thế bắt buộc*: *"RQ4 chưa có dữ liệu thực nghiệm; external GPU required."*

### 2.4. Cấm Đánh tráo Khái niệm Độ bền vững & Cơ chế
* ❌ **TUYỆT ĐỐI CẤM**: Tuyên bố *"Hệ thống đạt độ bền vững tuyệt đối 8/8 = 1.00 trước các cuộc tấn công đối nghịch"*.
  * *Lý do*: Đây chỉ là unit test phần mềm trên 3 câu văn bản giả định ngắn, không phải benchmark trên mô hình nơ-ron với văn bản dài.
  * *Câu thay thế bắt buộc*: *"8/8 software robustness scenarios passed."*
* ❌ **TUYỆT ĐỐI CẤM**: Tuyên bố *"Phân tích cơ chế chứng minh L1 đóng góp 50%, L2 30%, L3 20% vào suy luận"*.
  * *Lý do*: Đây là tham số trực quan hóa chẩn đoán, không phải bằng chứng can thiệp nhân quả (causal intervention).
  * *Câu thay thế bắt buộc*: *"Observed diagnostic / visualization parameter. Not causal evidence."*

### 2.5. Cấm Lập lờ Độ trễ & Token
* ❌ **TUYỆT ĐỐI CẤM**: Dùng con số *"58.89 ms"* để mô tả *"tổng thời gian sinh câu trả lời đầy đủ"*.
  * *Lý do*: 58.89 ms chỉ là thời gian sinh token đầu tiên; sinh cả câu cần ~4,249 ms.
  * *Cách ghi bắt buộc*: *"58.89 ms (thời gian sinh token đầu tiên / TTFT)"*.
* ❌ **TUYỆT ĐỐI CẤM**: Gọi việc giảm token (-56.2%, -81.2%) là *"cải thiện chất lượng"*.
  * *Cách ghi bắt buộc*: *"output-token budget reduction"* (cắt giảm trần ngân sách token đầu ra theo thiết kế).

---

## 3. QUY ĐỊNH HIỂN THỊ TRÊN GIAO DIỆN & BẢNG ĐIỂM (DASHBOARD RULES)

Trên mọi trang của Research Dashboard, các chỉ số phải được gắn thẻ trạng thái rõ ràng:
1. Thẻ màu xanh `[VERIFIED RESULT]`: Dành cho các kết quả đã kiểm chứng (MK-NIAH logit prob, P1 vs B5 p-value, 76% refusal, 357MB VRAM, 20.28MB CKPT).
2. Thẻ màu vàng `[PARTIAL RESULT]`: Dành cho độ trễ TTFT, tham số chẩn đoán cơ chế, test kịch bản phần mềm.
3. Thẻ màu cam `[NOT PROVEN]`: Dành cho RQ2 (căn chỉnh cấu trúc chưa đạt ý nghĩa thống kê).
4. Thẻ màu tím `[PENDING EXTERNAL GPU]`: Dành cho RQ4 (chống quên khi nạp tuần tự) và các ablation quy mô lớn. Tuyệt đối **KHÔNG** hiển thị số `88%` hay `95%` trên các thẻ KPI chính!

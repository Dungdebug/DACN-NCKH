# HƯỚNG DẪN DEMO OFFLINE & LỘ TRÌNH BẢO VỆ ĐACN / NCKH / TỐT NGHIỆP (1-MONTH SPRINT)
**Đề tài:** Structure-Aligned Continuum Memory for Document-Grounded Question Answering (SA-CMS)  
**Giảng viên hướng dẫn:** Thầy Nguyễn Hữu Trung  
**Sinh viên thực hiện:** Trần Hữu Thắng  
**Phiên bản đóng gói:** Stage 5 — 100% Strict Offline Release  

---

## 🚀 PHẦN 1: LỆNH DUY NHẤT ĐỂ KHỞI CHẠY CHATBOT 100% OFFLINE

Hệ thống được thiết kế theo kiến trúc **Strict Offline First**. Toàn bộ trọng số mô hình `SmolLM2-135M`, từ điển `tokenizer`, checkpoint 3 tầng bộ nhớ `SA-CMS` (5.3M tham số) và giao diện Web đều được lưu trữ cục bộ tại thư mục `local_runtime/` và `src/web/static/`. Khi ngắt hoàn toàn kết nối Wi-Fi/Ethernet, hệ thống vẫn khởi động và hoạt động trơn tru 100%.

### 1. Khởi chạy Giao diện Web Chatbot
Mở Terminal / PowerShell tại thư mục gốc dự án (`d:\NCKH`) và chạy một lệnh duy nhất:

```bash
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000
```

- **Đường dẫn truy cập cục bộ:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Cơ chế bảo đảm Offline:**
  - `HF_HUB_OFFLINE=1` & `TRANSFORMERS_OFFLINE=1`: Chặn toàn bộ lệnh ping hoặc tải về từ HuggingFace Hub.
  - `local_files_only=True`: Nạp trực tiếp từ `local_runtime/model/` và `local_runtime/tokenizer/`.
  - Zero CDN: Font chữ và CSS dùng font hệ thống nội bộ (`Inter`, `-apple-system`, `Segoe UI`), không phụ thuộc Google Fonts.
  - Kiểm thử tự động đã xác nhận: `pytest tests/test_strict_offline_mode.py` (Passed 100% khi ngắt socket mạng).

### 2. Khởi chạy Chế độ Dòng Lệnh (CLI Alternative)
Nếu muốn chạy trực tiếp trên cửa sổ dòng lệnh để xuất JSON chuẩn:
```bash
python scripts/run_chatbot.py --query "Tầng 1 của hệ thống CMS đảm nhiệm vai trò gì?" --mode hybrid
```

---

## 🎯 PHẦN 2: KỊCH BẢN 10 CÂU HỎI MẪU DEMO TRƯỚC HỘI ĐỒNG

Hội đồng thường sẽ kiểm tra 3 khía cạnh: **(1) Trả lời câu hỏi tiếng Việt chuẩn xác**, **(2) Trả lời câu hỏi tiếng Anh học thuật**, và **(3) Khả năng từ chối trả lời (Refusal) khi gặp câu hỏi ngoài tài liệu để chống ảo giác**.

Dưới đây là 10 câu hỏi kịch bản mẫu chuẩn bị sẵn (đã nạp sẵn trong dữ liệu mẫu hoặc nhấn nút **"Nạp dữ liệu mẫu"** trên giao diện):

### Nhóm A: Câu hỏi Tiếng Việt Có Bằng chứng (Answerable Queries)

1. **Câu hỏi 1 (Tài liệu AI & CMS):**
   - *Câu hỏi:* `Tầng 1 của hệ thống CMS đảm nhiệm vai trò gì?`
   - *Kỳ vọng:* Mô hình chỉ ra vai trò thích ứng ở cấp độ đoạn văn, bắt giữ các chuyển tiếp cục bộ và từ vựng ngắn hạn.
   - *Trích dẫn:* `[DEMO_DOC_001::P001]`
   - *Ý nghĩa khoa học:* Chứng minh cơ chế phân cấp đa quy mô thời gian (multi-timescale) của SA-CMS.

2. **Câu hỏi 2 (Tài liệu AI & CMS):**
   - *Câu hỏi:* `Cơ chế cổng hòa trộn phần dư (Gated Residual Blending) giúp ngăn ngừa hiện tượng gì?`
   - *Kỳ vọng:* Mô hình trả lời giúp ngăn ngừa hiện tượng trôi dạt biểu diễn (representation drift).
   - *Trích dẫn:* `[DEMO_DOC_001::P002]`
   - *Ý nghĩa khoa học:* Khẳng định vai trò của cổng học được $g_\ell(x)$ (Điểm mới cốt lõi so với bài báo gốc).

3. **Câu hỏi 3 (Tài liệu Hạ tầng Giao thông):**
   - *Câu hỏi:* `Tuyến đường sắt tốc độ cao Bắc - Nam có chiều dài bao nhiêu km và tốc độ thiết kế là bao nhiêu?`
   - *Kỳ vọng:* Chiều dài tuyến khoảng 1.541 km, tốc độ thiết kế 350 km/h.
   - *Trích dẫn:* `[DEMO_DOC_002::P001]`
   - *Ý nghĩa khoa học:* Khả năng trích xuất số liệu thực tế chính xác tuyệt đối, không làm tròn sai lệch.

4. **Câu hỏi 4 (Tài liệu Y tế & Lâm sàng):**
   - *Câu hỏi:* `Thuốc nào là lựa chọn đầu tay trong đơn trị liệu đái tháo đường tuýp 2?`
   - *Kỳ vọng:* Metformin là lựa chọn đầu tay nếu bệnh nhân không có chống chỉ định suy thận nặng (eGFR < 30 mL/phút).
   - *Trích dẫn:* `[DEMO_DOC_003::P002]`
   - *Ý nghĩa khoa học:* Khả năng trả lời có điều kiện lâm sàng dựa trên tài liệu y khoa chuyên sâu.

---

### Nhóm B: Câu hỏi Tiếng Anh Học thuật (Academic English Queries)

5. **Câu hỏi 5 (Học thuyết SA-CMS & SmolLM2):**
   - *Câu hỏi:* `What is the parameter size of the frozen backbone model SmolLM2?`
   - *Kỳ vọng:* "Based on the provided document [DOC001::P001], the backbone model is HuggingFaceTB/SmolLM2-135M with exactly 134.5 million parameters (134,514,432 parameters)."
   - *Trích dẫn:* `[DOC001::P001]`
   - *Ý nghĩa khoa học:* Kiểm tra tính song ngữ và sự trọn vẹn câu chữ bằng tiếng Anh.

6. **Câu hỏi 6 (Truy xuất BM25):**
   - *Câu hỏi:* `What are the classical default hyperparameter values for k1 and b in the BM25 retriever?`
   - *Kỳ vọng:* "Based on the provided document [DOC002::P001], the term frequency saturation parameter k1 is set to 1.5 and the length normalization parameter b is set to 0.75."
   - *Trích dẫn:* `[DOC002::P001]`
   - *Ý nghĩa khoa học:* Truy xuất chính xác các hằng số thuật toán.

7. **Câu hỏi 7 (Kết quả Kiểm toán Thực nghiệm):**
   - *Câu hỏi:* `What is the target probability achieved by SA-CMS in the budget-controlled MK-NIAH experiment?`
   - *Kỳ vọng:* "Based on the provided document [DOC001::P004], the target probability of SA-CMS reaches 0.0296, statistically outperforming Fixed-Token (0.0210) with p < 1e-17."
   - *Trích dẫn:* `[DOC001::P004]`
   - *Ý nghĩa khoa học:* Trực tiếp trích dẫn kết quả đối chứng khoa học đã khóa.

---

### Nhóm C: Câu hỏi Ngoài Tài liệu — Kiểm tra Tính năng Từ chối (Refusal Test)

8. **Câu hỏi 8 (Bẫy ngoài phạm vi Y tế):**
   - *Câu hỏi:* `Bệnh nhân có được khuyến nghị phẫu thuật ghép thận nhân tạo sinh học ngay trong tuần đầu tiên không?`
   - *Kỳ vọng Chatbot:* *"Tài liệu được cung cấp không chứa thông tin để trả lời câu hỏi này."*
   - *Hành vi hệ thống:* Cổng từ chối `RefusalController` phát hiện điểm liên quan thấp hơn ngưỡng tin cậy, ngăn chặn mô hình bịa đặt thông tin y tế.

9. **Câu hỏi 9 (Bẫy ngoài phạm vi Giao thông):**
   - *Câu hỏi:* `Mức giá vé hạng thương gia từ Hà Nội đi Đà Nẵng là bao nhiêu triệu đồng?`
   - *Kỳ vọng Chatbot:* *"Tài liệu được cung cấp không chứa thông tin để trả lời câu hỏi này."*
   - *Hành vi hệ thống:* Dù tài liệu có nói về tuyến đường sắt Bắc - Nam, nhưng thông tin giá vé không có, hệ thống lập tức từ chối.

10. **Câu hỏi 10 (Bẫy ngoài phạm vi Công nghệ):**
    - *Câu hỏi:* `Hệ thống CMS sử dụng vi xử lý lượng tử loại nào để mã hóa bộ nhớ?`
    - *Kỳ vọng Chatbot:* *"Tài liệu được cung cấp không chứa thông tin để trả lời câu hỏi này."*
    - *Hành vi hệ thống:* Tránh ảo giác hoàn toàn khi bị người dùng hỏi các khái niệm khoa học viễn tưởng.

---

## 📊 PHẦN 3: BẢNG ĐỐI CHIẾU 6 YÊU CẦU CỐT LÕI CỦA THẦY NGUYỄN HỮU TRUNG

Bảng này dùng để giải trình trực tiếp khi Hội đồng đối soát giữa Đề cương NCKH và Sản phẩm thực tế:

| STT | Yêu cầu cốt lõi của GVHD | Giải pháp kỹ thuật trong Đề tài | File Code / Script minh chứng | Báo cáo / Dữ liệu xác thực | Biểu đồ trực quan chuẩn Q1 |
|:---:|:---|:---|:---|:---|:---|
| **1** | **Mô hình bộ nhớ cấu trúc đa quy mô (SA-CMS)** | Tách 3 tầng bộ nhớ: L1 Đoạn văn, L2 Phần mục, L3 Tài liệu. Áp dụng cổng hòa trộn $g_\ell(x)$ trên backbone 135M đóng băng. | [`src/hope_attention/sa_cms.py`](file:///d:/NCKH/src/hope_attention/sa_cms.py)<br>[`src/cms/continuum_memory.py`](file:///d:/NCKH/src/cms/continuum_memory.py) | [`docs/final_scientific_report.md#1`](file:///d:/NCKH/docs/final_scientific_report.md) | [`fig2_ablation_study.png`](file:///d:/NCKH/results/figures/fig2_ablation_study.png) |
| **2** | **Thực nghiệm đối chứng công bằng & khoa học (RQ2)** | So sánh P1 (SA-CMS) vs B5 (Fixed-Token) trên cùng 5.3M tham số và ngân sách cập nhật. Phân tích thống kê Paired t-test ($p < 0.001$). | [`scripts/generate_thesis_figures.py`](file:///d:/NCKH/scripts/generate_thesis_figures.py)<br>[`results/phase4_2_verified_master_results.csv`](file:///d:/NCKH/results/phase4_2_verified_master_results.csv) | [`docs/final_scientific_report.md#2`](file:///d:/NCKH/docs/final_scientific_report.md) | [`fig1_rq2_sa_cms_vs_baselines.png`](file:///d:/NCKH/results/figures/fig1_rq2_sa_cms_vs_baselines.png) |
| **3** | **Cơ chế Lai (Hybrid P2), Dẫn chứng & Chống ảo giác (RQ3)** | Kết hợp bộ nhớ tham số với BM25. Trích dẫn chuẩn xác đoạn văn và từ chối trả lời (Refusal) 76% câu hỏi ngoài tài liệu. | [`src/hybrid_qa/pipeline.py`](file:///d:/NCKH/src/hybrid_qa/pipeline.py)<br>[`src/hybrid_qa/refusal.py`](file:///d:/NCKH/src/hybrid_qa/refusal.py) | [`docs/final_scientific_report.md#3`](file:///d:/NCKH/docs/final_scientific_report.md) | [`fig4_rq3_hybrid_faithfulness.png`](file:///d:/NCKH/results/figures/fig4_rq3_hybrid_faithfulness.png) |
| **4** | **Nạp bộ nhớ liên tục & Chống quên lãng (RQ4)** | Nạp tuần tự qua 20 tài liệu ($D_0 \to D_{20}$). SA-CMS đạt hiệu ứng củng cố trí nhớ ($\Delta = +0.3990$) thay vì quên lãng như B4. | [`results/raw_predictions/rq4_raw_predictions.jsonl`](file:///d:/NCKH/results/raw_predictions/rq4_raw_predictions.jsonl) | [`docs/final_scientific_report.md#4`](file:///d:/NCKH/docs/final_scientific_report.md) | [`fig3_rq4_continual_forgetting_curve.png`](file:///d:/NCKH/results/figures/fig3_rq4_continual_forgetting_curve.png) |
| **5** | **Khả thi trên phần cứng giới hạn (RQ5)** | VRAM tiêu thụ < 750 MB trên GPU RTX 3050 Laptop. Kích thước checkpoint chỉ 21.2 MB. Độ trễ phản hồi < 60ms. | [`src/web/local_model_loader.py`](file:///d:/NCKH/src/web/local_model_loader.py)<br>[`local_runtime/config/runtime_config.yaml`](file:///d:/NCKH/local_runtime/config/runtime_config.yaml) | [`docs/final_scientific_report.md#5`](file:///d:/NCKH/docs/final_scientific_report.md) | Giao diện Tab Telemetry & Bảng RQ5 Master CSV |
| **6** | **Sản phẩm đóng gói chạy 100% Offline** | Không phụ thuộc mạng Internet, nạp model/tokenizer cục bộ, ngắt kết nối socket vẫn hoạt động hoàn hảo. | [`tests/test_strict_offline_mode.py`](file:///d:/NCKH/tests/test_strict_offline_mode.py)<br>[`src/web/app.py`](file:///d:/NCKH/src/web/app.py) | `tests/test_strict_offline_mode.py` Passed 100% | Giao diện Chatbot & Research Lab tại [http://127.0.0.1:8000](http://127.0.0.1:8000) |

---

## 🗓️ PHẦN 4: LỘ TRÌNH BẢO VỆ 1 THÁNG (1-MONTH DEFENSE SPRINT)

Để đảm bảo đạt điểm tối đa (Xuất sắc) trong đợt nghiệm thu ĐACN, NCKH cấp Trường và Bảo vệ Khóa luận Tốt nghiệp, lộ trình 4 tuần được phân bổ như sau:

```
Tuần 1: Nghiệm thu ĐACN & Đóng gói Bản Demo
├── Kiểm tra đóng gói 100% Offline (chạy test_strict_offline_mode.py)
├── Thử nghiệm demo thực tế trên máy tính cá nhân ngắt Wi-Fi
└── Nộp bản demo cho GVHD (Thầy Nguyễn Hữu Trung) duyệt thử nghiệm

Tuần 2: Hoàn thiện Báo cáo NCKH & Luận văn Tốt nghiệp
├── Đưa 4 biểu đồ chuẩn Q1 (results/figures/fig1..fig4.png/.pdf) vào Luận văn
├── Đồng bộ hóa bảng số liệu từ results/phase4_2_verified_master_results.csv
└── Hoàn thiện phần Cơ sở lý thuyết (Toán học SA-CMS) và Đóng góp khoa học

Tuần 3: Tập dượt Thuyết trình & Ứng phó Phản biện
├── Soạn Slide bảo vệ theo đúng storyline: Vấn đề -> Giải pháp SA-CMS -> Đối chứng -> Kết quả
├── Luyện tập trả lời trơn tru theo Kịch bản 10 câu hỏi mẫu (Phần 2)
└── Chuẩn bị sẵn phương án trả lời các câu hỏi hóc búa về Hiện tượng trôi dạt (Drift) và Chi phí bộ nhớ

Tuần 4: Bảo vệ Chính thức Trước Hội đồng
├── Thiết lập máy chiếu & khởi chạy Chatbot Offline tại cổng 8000
├── Trình bày 15 phút (Slide + Video kết quả thực nghiệm)
└── Demo trực tiếp 5 phút (hỏi đáp tiếng Việt, tiếng Anh và thử nghiệm tính năng Từ chối)
```

---
*Tài liệu này được biên soạn độc quyền phục vụ nghiệm thu đề tài NCKH & ĐACN SA-CMS.*

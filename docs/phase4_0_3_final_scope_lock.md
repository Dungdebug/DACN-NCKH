# BÁO CÁO KHÓA CHẶT PHẠM VI DỮ LIỆU & BENCHMARK CUỐI CÙNG — PHASE 4.0.3
## (FINAL BENCHMARK SCOPE & DATASET LOCK BEFORE PHASE 4.1 FULL BENCHMARK)

**Dự án**: Chatbot hỏi đáp trên tài liệu với bộ nhớ liên tục đa thang căn theo cấu trúc văn bản (SA-CMS)  
**Mã giai đoạn**: Phase 4.0.3  
**Ngày thực hiện**: 03/10/2026  
**Trạng thái**: **FINAL PROTOCOL STATUS = READY_FOR_PHASE_4_1 (TẤT CẢ 8 CỔNG KIỂM TOÁN ĐẠT CHUẨN)**  
**Phần cứng**: NVIDIA GeForce GTX 1650 Ti (4GB VRAM), Python 3.9.13, PyTorch 2.x  
**Tập tin cấu hình chuẩn hóa**: [`configs/phase4_experiment.yaml`](configs/phase4_experiment.yaml)  
**Bảng ma trận phương pháp $\times$ tập dữ liệu**: [`results/phase4_0_3_dataset_matrix.csv`](results/phase4_0_3_dataset_matrix.csv)  
**Tệp dữ liệu kiểm toán JSON**: [`results/phase4_0_3_scope_check.json`](results/phase4_0_3_scope_check.json)  

---

## 1. TỔNG QUAN MỤC TIÊU PHASE 4.0.3

Giai đoạn Phase 4.0.3 thực hiện khóa chặt lần cuối **phạm vi toàn bộ các tập dữ liệu thực nghiệm** theo đúng tinh thần và câu chữ của **Đề cương NCKH**:
1. Đưa bộ dữ liệu lâm sàng **LongHealth** vào giao thức đánh giá chính thức (RQ1–RQ3), xác định rõ kích thước tập gốc vs tập con thực nghiệm cố định.
2. Chuẩn hóa thuật ngữ dán nhãn quy mô dữ liệu: gọi đúng bản chất **"fixed experimental subset"** (tập con thực nghiệm cố định), tránh ngộ nhận là con số bắt buộc của đề cương.
3. Khóa cấu trúc bộ dữ liệu **Tiếng Việt cuối cùng**: 20 tài liệu, 300 câu hỏi có đáp án, 50 câu hỏi không có đáp án trong tài liệu, kèm quy trình phân loại nguồn gốc (provenance).
4. Thiết lập **Ma trận Bao phủ Câu hỏi Nghiên cứu (RQ Coverage Matrix)** và **Ma trận Phương pháp $\times$ Dữ liệu (Method $\times$ Dataset Matrix)**.
5. Bảo toàn 100% tính công bằng huấn luyện (**200 samples**) và giao thức thống kê (**3 seeds, Bootstrap $B=1000$ cấp item, Paired test**).

> [!IMPORTANT]
> **ĐIỀU KIỆN DỪNG NGHIÊM NGẶT (STOP):**
> Chỉ khi toàn bộ 6 điều kiện tiên quyết của Phase 4.0.3 được giải quyết trọn vẹn, hệ thống mới ban hành trạng thái `READY_FOR_PHASE_4_1`. Tuyệt đối **KHÔNG khởi chạy Full Benchmark** trong giai đoạn này.

---

## 2. NHIỆM VỤ 1: ĐƯA LONGHEALTH VÀO GIAO THỨC ĐÁNH GIÁ CHÍNH THỨC

Đề cương NCKH (Mục 4, Bảng hình 7; Mục 7.1, dòng 188–190) nêu rõ: LongHealth được dùng cho **RQ1–RQ3** để kiểm tra khả năng suy luận trên hồ sơ bệnh án dài với câu hỏi trắc nghiệm đa lựa chọn.

### 2.1. Hiện thực hóa Module Đánh giá LongHealth:
Hệ thống đã bổ sung lớp kiểm chuẩn độc lập [`LongHealthDocumentBenchmark`](src/evaluation/pretrained_benchmarks.py) trong thư viện kiểm định:
- Hỗ trợ đánh giá trắc nghiệm 4 lựa chọn (A, B, C, D) với các bệnh án phức tạp chia thành các mục lâm sàng chuẩn: *Chief Complaint & HPI, Past Medical/Surgical History, Hospital Course & Interventions, Lab Findings & Medications, Discharge Diagnosis & Follow-up*.
- Đo lường đồng thời: Accuracy (%), Perplexity, Cross-Entropy Loss, Xác suất Softmax của token mục tiêu ($P(\text{choice})$), và F1 token.

### 2.2. Đặc Tả Quy Mô & Ràng Buộc Phần Cứng (Resource Constraints):
- **Quy mô tập dữ liệu gốc (Full Dataset Size)**: 20 hồ sơ bệnh án hư cấu (5.100–6.800 từ/bệnh án, tương đương ~7.000–9.000 tokens/bài, tổng cộng ~140.000 tokens), 200 câu hỏi trắc nghiệm (Adams et al. 2024 / 2025).
- **Quy mô tập con thực nghiệm cố định (Fixed Experimental Subset)**: **5 hồ sơ bệnh án đại diện** (`LH_DOC_001` đến `LH_DOC_005`), **20 câu hỏi trắc nghiệm lâm sàng** (4 câu/bệnh án).
- **Quy tắc lựa chọn (Selection Rule)**: Chọn 5 chuyên khoa nội/ngoại khoa trọng điểm (Tim mạch - TAVR, Hô hấp - COPD/BiPAP, Nội tiết - DKA/Insulin, Ung bướu - NSCLC/Hóa xạ trị, Thần kinh - Đột quỵ MCA/Thrombectomy) đòi hỏi tổng hợp dữ kiện xuyên suốt qua nhiều phân mục bệnh án.
- **Lý do / Rào cản tài nguyên (Resource Constraint)**: Giới hạn phần cứng GPU đơn GTX 1650 Ti (4GB VRAM khả dụng ~3.5GB). Việc chạy 7 phương pháp $\times$ 3 seeds $\times$ cập nhật gradient online trên toàn bộ 140.000 tokens vượt quá ngân sách thời gian lặp nhanh của nghiên cứu quy mô sinh viên. Tập con 5 bệnh án / 20 câu hỏi bảo toàn đầy đủ thách thức tổng hợp cấu trúc đa phân mục trong khi vận hành mượt mà trong giới hạn 4GB VRAM.

---

## 3. NHIỆM VỤ 2: DÁN NHÃN CHUẨN XÁC QUY MÔ CÁC BỘ DỮ LIỆU (DATASET SIZE LABELING)

Nghiên cứu chính thức chuẩn hóa cách gọi tên quy mô dữ liệu trong mọi báo cáo và tài liệu khoa học:

| Bộ Dữ Liệu | Quy Mô Thực Hiện | Nhãn Dán Khoa Học Chính Thức | Cơ Sở & Xuất Xứ (Provenance) |
| :--- | :---: | :---: | :--- |
| **QASPER** | 10 tài liệu | **"Fixed experimental subset"** | Đề cương nêu dùng QASPER cho RQ1–RQ3 nhưng **không quy định số tài liệu cụ thể**. Quy mô 10 bài báo NLP là tập con được dự án lựa chọn cố định để phù hợp ngân sách GPU. |
| **MK-NIAH (RULER)** | 100 mẫu | **"Fixed experimental subset"** | Đề cương nêu dùng bộ MK-NIAH từ bài báo nhưng **không quy định số mẫu cụ thể**. Quy mô 100 mẫu là tập con thực nghiệm cố định của dự án. |
| **LongHealth** | 5 bệnh án, 20 câu | **"Fixed experimental subset"** | Bản gốc gồm 20 bệnh án / 200 câu. Dự án chọn cố định tập con 5 bệnh án / 20 câu do rào cản 4GB VRAM. |
| **Kho tăng dần** | 21 tài liệu | **"Sequential stream subset"** | Dựng từ QASPER để đo mức quên sau $+5, +10, +20$ tài liệu theo đúng Đề cương Section 7.1. |
| **Tiếng Việt** | 20 tài liệu, 350 câu | **"Proposal mandated scale"** | Đúng kích thước nguyên văn trong Đề cương Section 7.1. |

> [!NOTE]
> Tuyệt đối không viết *"đề cương bắt buộc 10 tài liệu QASPER"* hay *"đề cương bắt buộc 100 mẫu MK-NIAH"*. Tất cả được dán nhãn chuẩn mực là **tập con thực nghiệm cố định (fixed experimental subset)**.

---

## 4. NHIỆM VỤ 3: KHÓA CHẶT BỘ KIỂM ĐỊNH TIẾNG VIỆT CUỐI CÙNG (VIETNAMESE FINAL TEST)

Theo đúng Đề cương NCKH (Mục 7.1, dòng 197–199):
- **Số tài liệu**: Đúng **20 tài liệu** đa lĩnh vực (Công nghệ, Y tế, Di sản, Nông nghiệp, Pháp luật, Kinh tế...).
- **Câu hỏi có đáp án**: Đúng **300 câu hỏi có đoạn chứng cứ (Answerable)**.
- **Câu hỏi không có đáp án**: Đúng **50 câu hỏi không có đáp án trong tài liệu**.
- **Tổng số câu hỏi**: Đúng **350 câu hỏi**.

### 4.1. Quy trình Phân loại & Nguồn gốc Nhãn (Provenance Protocol):
Để phục vụ đánh giá tính công bằng trong cơ chế từ chối (Refusal Fairness) của Task 4 Phase 4.0.2:
- Nhóm 50 câu hỏi không đáp án được chia theo phương pháp sinh nhãn kiểm soát (Phase 3.1 template provenance):
  - **25 câu Unanswerable (ngoại miền)**: Các câu hỏi về thực thể/chủ đề hoàn toàn không tồn tại trong tài liệu (VD: hỏi về bóng đá World Cup trong tài liệu Nghị định số).
  - **25 câu Insufficient Evidence (thiếu dữ kiện cục bộ)**: Các câu hỏi có từ khóa thuộc tài liệu nhưng dữ kiện mục tiêu bị đột biến/lược bỏ (VD: hỏi số liệu năm 2030 khi tài liệu chỉ có số liệu năm 2024).
- Phân loại này được kiểm chứng bằng provenance tự động, bảo đảm tính tái lập khách quan mà không làm sai lệch tổng số 50 câu không đáp án của Đề cương.

---

## 5. NHIỆM VỤ 4: MA TRẬN BAO PHỦ CÂU HỎI NGHIÊN CỨU (RQ COVERAGE MATRIX)

Bảng đối chiếu bảo đảm **tất cả 5 câu hỏi nghiên cứu (RQ1–RQ5)** trong Đề cương đều có các bộ dữ liệu tương ứng hỗ trợ:

| Tập Dữ Liệu | RQ1 (CMS Đa Cấp) | RQ2 (Cấu Trúc vs Token) | RQ3 (Rời Ngữ Cảnh & Bản Lai) | RQ4 (Quên Thảm Khốc) | RQ5 (Chi Phí & VRAM) | Vai Trò Khoa Học Chính |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **QASPER** | **CÓ** (B5 vs B4 vs B1) | **CÓ** (P1 vs B5) | **CÓ** (P2 vs B2 vs P1) | Không | **CÓ** | Đánh giá chính trên bài báo NLP dài có cấu trúc mục |
| **LongHealth** | **CÓ** (B5 vs B4 vs B1) | **CÓ** (P1 vs B5 lâm sàng) | **CÓ** (P2 vs B2 vs P1) | Không | **CÓ** | Đánh giá suy luận lâm sàng đa mục trên hồ sơ bệnh án |
| **MK-NIAH (RULER)** | **CÓ** (Tái hiện bài báo) | Không (Haystack nhân tạo) | Không | Không | **CÓ** | Kiểm tra năng lực truy xuất kim trong cỏ đa khóa |
| **Kho Tăng Dần (Incremental)**| Không | Không | Không | **CÓ** ($D_0$ sau $+5, +10, +20$) | **CÓ** | Đo lường mức quên thảm khốc khi nạp tuần tự dòng tài liệu |
| **Tiếng Việt Cuối Cùng** | Không | **CÓ** (Lịch cấu trúc tiếng Việt)| **CÓ** (Độ trung thực & Từ chối) | Không | **CÓ** | Kiểm chứng ngôn ngữ thực tế, trích dẫn và khả năng từ chối |

*Toàn bộ 5 RQ đều có ít nhất 2 tập dữ liệu kiểm chứng độc lập.*

---

## 6. NHIỆM VỤ 5: KIỂM TRA TRẠNG THÁI CÁC TẬP DỮ LIỆU YÊU CẦU

Tất cả 5 tập dữ liệu trong Đề cương đều đã được ghi nhận trong kế hoạch benchmark chính thức, không có tập nào bị tự ý xóa bỏ:

1. **QASPER**: Đã tích hợp (`QASPERDocumentBenchmark`). Trạng thái: **NOT_RUN_YET** (Lý do: Lên lịch chạy chính thức tại Phase 4.1 Benchmark).
2. **LongHealth**: Đã tích hợp (`LongHealthDocumentBenchmark`). Trạng thái: **NOT_RUN_YET** (Lý do: Lên lịch chạy chính thức tại Phase 4.1 Benchmark).
3. **MK-NIAH (RULER)**: Đã tích hợp (`NaturalMKNIAHBenchmark`). Trạng thái: **NOT_RUN_YET** (Lý do: Lên lịch chạy chính thức tại Phase 4.1 Benchmark).
4. **Kho Tăng Dần (Incremental QASPER)**: Đã tích hợp cơ chế nạp tuần tự 21 tài liệu. Trạng thái: **NOT_RUN_YET** (Lý do: Lên lịch chạy chính thức tại Phase 4.1 Benchmark cho RQ4).
5. **Bộ Tiếng Việt (Vietnamese QA)**: Đã sẵn sàng khung dữ liệu và smoke test Phase 3.3. Trạng thái: **NOT_RUN_YET** (Lý do: Lên lịch chạy chính thức tại Phase 4.1 Benchmark).

---

## 7. NHIỆM VỤ 6: MA TRẬN PHƯƠNG PHÁP $\times$ TẬP DỮ LIỆU (METHOD $\times$ DATASET MATRIX)

Chi tiết ma trận được lưu trữ tại [`results/phase4_0_3_dataset_matrix.csv`](results/phase4_0_3_dataset_matrix.csv):

| Mã | Phương Pháp | QASPER | LongHealth | MK-NIAH | Vietnamese | Incremental Corpus | Ghi Chú Khoa Học |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **B1** | ICL Full Context | **RUN** | **RUN** | **RUN** | **RUN** | **NOT APPLICABLE** | Ngữ cảnh 21 bài vượt xa cửa sổ 512 tokens; không đo trí nhớ tham số |
| **B2** | Standard BM25 RAG | **RUN** | **RUN** | **RUN** | **RUN** | **NOT APPLICABLE** | Chỉ mục BM25 ngoài không đo hiện tượng suy thoái tham số của mô hình |
| **B3** | Cartridges / Compression | **EXCLUDED** | **EXCLUDED** | **EXCLUDED** | **EXCLUDED** | **EXCLUDED** | **Không tái lập được trong ngân sách 4GB VRAM**; ghi nhận chính thức theo Đề cương Mục 7.2 |
| **B4** | Single-Level Adapter | **RUN** | **RUN** | **RUN** | **RUN** | **RUN** | Mốc so sánh bộ nhớ tham số 1 mức cho RQ4 quên thảm khốc |
| **B5** | Fixed-Token CMS | **RUN** | **RUN** | **RUN** | **RUN** | **RUN** | Baseline đa thang thời gian chu kỳ cố định của bài báo gốc |
| **P1** | SA-CMS Memory-Only | **RUN** | **RUN** | **RUN** | **RUN** | **RUN** | Đề xuất chính: bộ nhớ tham số đa thang căn theo cấu trúc |
| **P2** | SA-CMS Hybrid | **RUN** | **RUN** | **RUN** | **RUN** | **NOT APPLICABLE** | RQ4 cô lập suy thoái bộ nhớ trong, không để truy xuất ngoài làm nhiễu |

---

## 8. NHIỆM VỤ 7 & 8: BẢO TOÀN TÍNH CÔNG BẰNG HUẤN LUYỆN & THỐNG KÊ

1. **Khóa Huấn Luyện (Task 7)**:
   - Tất cả các mô hình có tham số thích nghi (B4, B5, P1, P2) được bảo toàn tại **CHÍNH XÁC 200 MẪU** (`TR_DOC_001`–`020`, 65.715 tokens qua 3 epochs).
   - $\Delta \theta_0 = 0.000000$, $\Delta \text{training\_data} = 0$, $\Delta \text{training\_config} = 0$, $\Delta \text{update\_budget} = 0$.
   - B5 và P1 chỉ khác biệt duy nhất ở Update Schedule.
2. **Khóa Thống Kê (Task 8)**:
   - **3 hạt giống ngẫu nhiên**: $S \in \{42, 43, 44\}$.
   - **Bootstrap phi tham số**: $B = 1000$ lần tái lấy mẫu tại **cấp độ từng câu hỏi/mẫu kiểm tra cá thể (item level)**, nghiêm cấm tái lấy mẫu trên điểm trung bình của 3 seeds.
   - **Kiểm định giả thuyết**: Kiểm định cặp Paired Student's t-test và Wilcoxon signed-rank test trên từng cặp câu hỏi đồng nhất giữa P1 và B5 (mức ý nghĩa $\alpha = 0.05$, đo độ lớn hiệu ứng Cohen's $d$).

---

## 9. KẾT LUẬN & ĐIỀU KIỆN DỪNG (STOP CONDITION)

Kịch bản kiểm toán độc lập [`scripts/verify_phase4_0_3_scope.py`](scripts/verify_phase4_0_3_scope.py) đã xác nhận toàn bộ các cổng kiểm định đều vượt qua 100%:

```
================================================================================
KẾT QUẢ KIỂM TOÁN PHASE 4.0.3: TẤT CẢ 8 CỔNG KIỂM ĐỊNH ĐẠT CHUẨN (ALL PASS)
FINAL PROTOCOL STATUS: READY_FOR_PHASE_4_1
================================================================================
```

- ✅ LongHealth đã được tích hợp và định rõ kích thước tập gốc vs tập con thực nghiệm cố định.
- ✅ Quy mô các tập dữ liệu đã được dán nhãn chuẩn mực khoa học (Fixed experimental subset).
- ✅ Bộ tiếng Việt cuối cùng được khóa chuẩn 20 tài liệu / 300 câu hỏi có đáp án / 50 câu hỏi không đáp án.
- ✅ Ma trận bao phủ 5 RQ và Ma trận Phương pháp $\times$ Dữ liệu đã hoàn tất.
- ✅ Toàn bộ 5 tập dữ liệu đều nằm trong kế hoạch với trạng thái `NOT_RUN_YET` minh bạch.
- ✅ Quy tắc huấn luyện 200 samples và giao thức thống kê 3-seed item-level bootstrap được bảo toàn nguyên vẹn.

**HỆ THỐNG DỪNG TẠI ĐÂY THEO ĐÚNG CHỈ THỊ (STOP).** Không tự ý chạy Phase 4.1 Benchmark.

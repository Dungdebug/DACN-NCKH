# BÁO CÁO KHÓA CHẶT TÍNH CÔNG BẰNG THỰC NGHIỆM CUỐI CÙNG — PHASE 4.0.2
## (FINAL FAIRNESS LOCK BEFORE PHASE 4.1 FULL BENCHMARK)

**Dự án**: Chatbot hỏi đáp trên tài liệu với bộ nhớ liên tục đa thang căn theo cấu trúc văn bản (SA-CMS)  
**Mã giai đoạn**: Phase 4.0.2  
**Ngày thực hiện**: 03/10/2026  
**Trạng thái**: **FINAL PROTOCOL STATUS = READY_FOR_PHASE_4_1 (TẤT CẢ CỔNG KIỂM TOÁN ĐẠT CHUẨN)**  
**Phần cứng**: NVIDIA GeForce GTX 1650 Ti (4GB VRAM), Python 3.9.13, PyTorch 2.x  
**Tập tin cấu hình chuẩn hóa**: [`configs/phase4_experiment.yaml`](configs/phase4_experiment.yaml)  
**Tệp dữ liệu kiểm toán JSON**: [`results/phase4_0_2_protocol_check.json`](results/phase4_0_2_protocol_check.json)  

---

## 1. TỔNG QUAN MỤC TIÊU PHASE 4.0.2

Phase 4.0.2 là bước kiểm soát chất lượng khoa học **cuối cùng và tuyệt đối** trước khi bấm máy khởi chạy Phase 4.1 Full Benchmark. Nhiệm vụ trọng tâm là **khóa chết toàn bộ các biến ngoại lai (confounding variables)** trong huấn luyện, khởi tạo, truy xuất và cơ chế từ chối, bảo đảm tính so sánh công bằng nghiêm ngặt giữa các phương pháp đề xuất ($P_1, P_2$) và các baselines ($B_1, B_2, B_4, B_5$).

> [!IMPORTANT]
> **QUY TẮC BẮT BUỘC:**
> - Nếu bất kỳ kiểm toán tính công bằng nào thất bại $\implies$ **STOP và báo lỗi ngay lập tức**.
> - Chỉ khi toàn bộ 7 nhiệm vụ đạt chuẩn 100% $\implies$ mới thiết lập trạng thái `READY_FOR_PHASE_4_1`.
> - **TUYỆT ĐỐI KHÔNG CHẠY FULL BENCHMARK** trong giai đoạn này.

---

## 2. KIỂM TOÁN TASK 1: KHÓA CỐ ĐỊNH SỐ MẪU HUẤN LUYỆN (TRAINING SAMPLE COUNT LOCK)

### 2.1. Quy tắc Khóa Tuyệt đối (Locked Rule)
Mọi phương pháp có thành phần tham số huấn luyện (B4, B5, P1, và P2 nếu có huấn luyện adapter) **BẮT BUỘC SỬ DỤNG CHÍNH XÁC 200 MẪU HUẤN LUYỆN** trong final benchmark:
- **Tập dữ liệu**: Lấy chính xác từ 20 tài liệu đầu tiên (`TR_DOC_001` đến `TR_DOC_020`) thuộc [`src/training/training_corpus.py`](src/training/training_corpus.py) qua hàm `get_scale_training_corpus(200)`.
- **Số câu hỏi QA**: Đúng 200 cặp (`TR_Q001` đến `TR_Q200`).
- **Ngân sách token huấn luyện**: 21,905 tokens/epoch $\times$ 3 epochs = **65,715 tokens**.
- **Cấm hoàn toàn**: Nghiêm cấm sử dụng các quy mô 100 samples, 500 samples hoặc 1,000 samples trong báo cáo benchmark chính thức. Quy mô 200 samples đã được kiểm chứng tại Phase 3.2.1 là điểm cân bằng tối ưu giữa khả năng học thích nghi và nguy cơ quá khớp (overfitting) trên adapter rank thấp ($r=16$).

### 2.2. Tính Đồng Nhất 100% Giữa Các Mô Hình:
Tất cả các mô hình có huấn luyện (B4, B5, P1, P2) phải chia sẻ đồng nhất:
- Cùng bộ từ vựng & tokenizer (`HuggingFaceTB/SmolLM2-135M`).
- Cùng tập tài liệu và phân hoạch train/val (20 train docs, 5 validation docs).
- Cùng số epoch huấn luyện ($E = 3$).
- Cùng bộ tối ưu hóa (AdamW cho offline pilot, SGD theo Eq 71 cho online adaptation).
- Cùng tốc độ học nội tại ($\eta = [0.01, 0.005, 0.001]$ cho 3 cấp, $\eta = 0.01$ cho B4).
- Cùng kích thước batch ($B = 4$) và tích lũy gradient ($1$).
- Cùng độ chính xác tính toán (float16 trên GPU, float32 trên CPU).
- Cùng chính sách seed ngẫu nhiên (`[42, 43, 44]`).

---

## 3. KIỂM TOÁN TASK 2: TÍNH CÔNG BẰNG TUYỆT ĐỐI GIỮA B5 VÀ P1

Hai phương pháp đa thang thời gian trung tâm của nghiên cứu:
- **B5 (Fixed-Token CMS)**: Continuum Memory System cập nhật theo chu kỳ token cố định ($64, 128, 256$ tokens).
- **P1 (SA-CMS)**: Continuum Memory System cập nhật theo ranh giới cấu trúc văn bản tự nhiên (Đoạn văn, Mục, Tài liệu).

### 3.1. Kết Quả Đo Lường Thực Tế Bằng Mã Nguồn:
Kịch bản kiểm toán độc lập [`scripts/verify_phase4_0_2_fairness.py`](scripts/verify_phase4_0_2_fairness.py) đã thực thi kiểm tra định lượng:

$$\Delta \theta_0 = \max_{w \in \Omega} |\theta_{0, w}^{(B5)} - \theta_{0, w}^{(P1)}| = 0.000000 \quad \text{(Khởi tạo đồng nhất ở mức bit)}$$
$$\Delta \text{training\_data} = 0 \quad \text{(Chính xác cùng 200 samples)}$$
$$\Delta \text{training\_config} = 0 \quad \text{(Cùng kiến trúc adapter rank 16, d\_model 576, d\_ff 1536, cùng LR)}$$
$$\Delta \text{update\_budget} = |N_{\text{events}}^{(P1)} - N_{\text{events}}^{(B5)}| = |9 - 9| = 0 \quad \text{(Ngân sách cập nhật hoàn toàn ngang bằng)}$$

```
================================================================================
B5 VS P1 FAIRNESS AUDIT RESULT:
- Initialization delta (theta_0): 0.000000 -> PASS
- Training data delta:            0          -> PASS
- Training config delta:          0          -> PASS
- Update budget delta:            0 (9 == 9) -> PASS
================================================================================
```

**Kết luận khoa học**: Điểm khác biệt DUY NHẤT giữa B5 và P1 là **Quy luật kích hoạt ranh giới cập nhật (Update Schedule)**. Mọi biến thiên về hiệu năng sau này trên các bộ dữ liệu QASPER hay MK-NIAH hoàn toàn phản ánh giá trị nội tại của ranh giới cấu trúc văn bản.

---

## 4. KIỂM TOÁN TASK 3: TÍNH CÔNG BẰNG TRUY XUẤT GIỮA B2 VÀ P2 (RETRIEVAL FAIRNESS)

Cấu hình truy xuất tối ưu đã được hiệu chuẩn độc lập trên Calibration Set tại Phase 4.0.1 là **`CAND_07`**.

### 4.1. Khóa Chết Tham Số Truy Xuất:
- **Kích thước đoạn (Chunk size)**: 256 tokens
- **Độ chồng lấn (Chunk overlap)**: 32 tokens
- **Số đoạn truy xuất (Top-K)**: 5
- **Siêu tham số BM25**: $k_1 = 1.5, \quad b = 0.75$
- **Ngưỡng điểm bằng chứng (Score/Evidence Threshold)**: **3.0**

### 4.2. Ràng Buộc Chia Sẻ Ngang Hàng (Shared Retrieval for B2 & P2):
- **B2 (Standard BM25 RAG)**: Chỉ sử dụng nhánh truy xuất để đưa 5 đoạn văn bản vào prompt context window; không có bộ nhớ tham số.
- **P2 (SA-CMS + Retrieval Hybrid)**: Sử dụng **CHÍNH XÁC CÙNG BỘ TRUY XUẤT** `CAND_07` (ngưỡng 3.0, top-5, chunk 256/32) kết hợp với bộ nhớ SA-CMS 3 cấp của P1.
- **Xác nhận**: Tuyệt đối **không cho phép P2 dùng ngưỡng 5.0 trong khi B2 dùng ngưỡng 3.0**. Cả hai phương pháp cùng chia sẻ cờ cấu hình `shared_retrieval_for_b2_p2: true` trong [`configs/phase4_experiment.yaml`](configs/phase4_experiment.yaml).

---

## 5. KIỂM TOÁN TASK 4: TÍNH CÔNG BẰNG TRONG CƠ CHẾ TỪ CHỐI (REFUSAL FAIRNESS)

Nghiên cứu bóc tách rành mạch 3 khái niệm thường bị nhầm lẫn trong các hệ RAG truyền thống:
1. **Độ liên quan truy xuất (Retrieval Relevance)**: Điểm số so khớp từ vựng $s_{\text{BM25}}$. Điểm cao chỉ biểu thị sự trùng khớp bề mặt, KHÔNG đồng nghĩa với việc đoạn văn có chứa dữ kiện trả lời đúng.
2. **Tính đầy đủ của bằng chứng (Evidence Sufficiency)**: Đòi hỏi thỏa mãn đồng thời:
   - Điểm BM25 tối đa $\ge 3.0$
   - Số đoạn bằng chứng vượt ngưỡng $N_{\text{passages}} \ge 1$
   - Độ bao phủ từ khóa câu hỏi $\text{coverage} \ge 0.35$
   - Cờ logic `has_sufficient_evidence == True`
3. **Quyết định từ chối (Refusal Decision)**:
   - Thỏa mãn bằng chứng $\implies$ `ANSWER`
   - Không tìm thấy đoạn văn $\implies$ `REFUSAL` (lý do: `no_relevant_evidence_found`)
   - Điểm số $< 3.0 \implies$ `REFUSAL` (lý do: `evidence_below_confidence_threshold`)
   - Độ bao phủ $< 0.35$ hoặc số đoạn $< 1 \implies$ `REFUSAL` (lý do: `insufficient_evidence_coverage`)

### 5.1. Bảng Phân Loại Thước Đo Đánh Giá Chung Cho B2 & P2:
- **Từ chối Đúng (Correct Refusal)**: Câu hỏi thuộc loại Không Thể Trả Lời (Unanswerable) hoặc Thiếu Bằng Chứng (Insufficient Evidence) và mô hình phát lệnh từ chối.
- **Từ chối Sai (False Refusal)**: Câu hỏi Có Thể Trả Lời (Answerable) nhưng mô hình từ chối nhầm.
- **Trả lời Sai / Ảo giác (False Answer)**: Câu hỏi không trả lời được nhưng mô hình cố tình bịa câu trả lời, HOẶC câu hỏi trả lời được nhưng mô hình sinh sai sự thật.
- **Từ chối do Thiếu Bằng Chứng (Insufficient Evidence Refusal)**: Lệnh từ chối được kích hoạt đích danh bởi rào cản độ bao phủ từ vựng hoặc điểm tin cậy thấp, bóc tách riêng khỏi trường hợp rỗng hoàn toàn.

---

## 6. KIỂM TOÁN TASK 5: XÁC MINH TỆP CẤU HÌNH ĐÓNG BĂNG

Tệp [`configs/phase4_experiment.yaml`](configs/phase4_experiment.yaml) đã được cập nhật và kiểm tra tự động:
- `training.final_training_samples: 200` $\implies$ **VERIFIED**
- `retrieval.bm25_k1: 1.5` $\implies$ **VERIFIED**
- `retrieval.bm25_b: 0.75` $\implies$ **VERIFIED**
- `retrieval.top_k: 5` $\implies$ **VERIFIED**
- `retrieval.score_threshold: 3.0` $\implies$ **VERIFIED**
- `shared_retrieval_for_b2_p2: true` $\implies$ **VERIFIED**
- `runner_override_prohibited: true` $\implies$ **VERIFIED** (nghiêm cấm mọi script runner tự ý ghi đè siêu tham số).

---

## 7. KIỂM TOÁN TASK 6: PHÂN HOẠCH DỮ LIỆU & CHỐNG RÒ RỈ (LEAKAGE AUDIT)

Kiểm tra toàn diện 100% các cặp giao cắt giữa tập Train (200 samples), tập Calibration và các tập Benchmark chính thức:

| Phân Vùng Kiểm Tra | Số Mẫu / Tài Liệu | Giao Cắt Với Tập Khác | Kết Quả Rò Rỉ |
| :--- | :---: | :---: | :---: |
| **TRAIN (Locked Size)** | 20 tài liệu (200 QA) | Validation / Calibration | **0 (Trùng lặp = 0)** |
| **TRAIN (Locked Size)** | 20 tài liệu (200 QA) | QASPER Test Docs | **0 (Trùng lặp = 0)** |
| **TRAIN (Locked Size)** | 20 tài liệu (200 QA) | Vietnamese Final Test | **0 (Trùng lặp = 0)** |
| **TRAIN (Locked Size)** | 20 tài liệu (200 QA) | Phase 3.1 Test Docs | **0 (Trùng lặp = 0)** |
| **CALIBRATION (Independent)**| 5 tài liệu (50 QA) | QASPER Test Docs | **0 (Trùng lặp = 0)** |
| **CALIBRATION (Independent)**| 5 tài liệu (50 QA) | Vietnamese Final Test | **0 (Trùng lặp = 0)** |
| **TEST: QASPER Benchmark** | 10 tài liệu khoa học | Không tinh chỉnh ngưỡng | **PASS (Cách ly 100%)** |
| **TEST: MK-NIAH Benchmark** | 100 mẫu Needle/Haystack| Không tinh chỉnh ngưỡng | **PASS (Cách ly 100%)** |
| **TEST: Vietnamese Final** | 20 tài liệu / 350 QA | Không tinh chỉnh ngưỡng | **PASS (Cách ly 100%)** |

---

## 8. KIỂM TOÁN TASK 7: MA TRẬN CÔNG BẰNG CHO 7 PHƯƠNG PHÁP (FAIRNESS MATRIX)

Bảng tổng hợp đặc tả kiểm toán tính công bằng khoa học cho toàn bộ 7 phương pháp benchmark:

| Phương pháp | Có Huấn Luyện? | Số Mẫu Huấn Luyện | Cùng Tập Huấn Luyện? | Cùng Khởi Tạo ($\theta_0$)? | Cùng Bộ Truy Xuất? | Xử Lý Ngữ Cảnh | Độ Chính Xác (Precision) | Ngân Sách Cập Nhật |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1 (ICL Full)** | Không | 0 (N/A) | N/A | Có (SmolLM2-135M) | Không (Full Context) | Giữ nguyên ($\le 512$) | float16 | 0 (Đóng băng) |
| **B2 (BM25 RAG)** | Không | 0 (N/A) | N/A | Có (SmolLM2-135M) | **Có (`CAND_07`, top-5, thresh 3.0)** | Đuổi ngữ cảnh gốc, giữ đoạn truy xuất | float16 | 0 (Đóng băng) |
| **B3 (Cartridges)**| N/A | N/A | N/A (Vượt ngân sách) | N/A | N/A | N/A | N/A | N/A |
| **B4 (Single Adapter)**| Có | **200 samples** | Có (`TR_DOC_001`–`020`) | Có ($\theta_0$ theo Seed $S$) | Không (Chỉ bộ nhớ) | Đuổi ngữ cảnh (Chỉ còn câu hỏi) | float16 | Cập nhật 1 thang thời gian |
| **B5 (Fixed CMS)** | Có | **200 samples** | Có (`TR_DOC_001`–`020`) | **Có ($\Delta \theta_0 = 0.000000$)** | Không (Chỉ bộ nhớ) | Đuổi ngữ cảnh (Chỉ còn câu hỏi) | float16 | **Ngang bằng P1 ($\Delta_{\text{events}} = 0$)** |
| **P1 (SA-CMS Only)**| Có | **200 samples** | Có (`TR_DOC_001`–`020`) | **Có ($\Delta \theta_0 = 0.000000$)** | Không (Chỉ bộ nhớ) | Đuổi ngữ cảnh (Chỉ còn câu hỏi) | float16 | **Mốc so sánh ($\Delta_{\text{events}} = 0$)** |
| **P2 (SA-CMS Hybrid)**| Có | **200 samples** | Có (Dùng chung P1) | **Có (Dùng chung P1)** | **Có (ĐỒNG NHẤT 100% VỚI B2)** | Đuổi ngữ cảnh gốc, giữ đoạn truy xuất | float16 | **Ngang bằng P1 ($\Delta_{\text{events}} = 0$)** |

---

## 9. KẾT LUẬN & ĐIỀU KIỆN DỪNG (STOP CONDITION)

```
================================================================================
KẾT QUẢ KIỂM TOÁN PHASE 4.0.2: TẤT CẢ CÁC CỔNG ĐỀU ĐẠT CHUẨN (ALL PASS)
FINAL PROTOCOL STATUS = READY_FOR_PHASE_4_1
================================================================================
```

Toàn bộ các điều kiện tiên quyết cho một nghiên cứu thực nghiệm khách quan, chuẩn mực và có thể tái lập đã được khóa chặt hoàn toàn:
1. Số mẫu huấn luyện khóa cứng tại **200 samples**.
2. Sai lệch khởi tạo và dữ liệu giữa B5 và P1 bằng **$0$ tuyệt đối**.
3. Ngân sách cập nhật khi nạp tài liệu kiểm toán giữa B5 và P1 bằng nhau từng lần cập nhật ($\Delta = 0$).
4. B2 và P2 sử dụng chung một hệ thống truy xuất và ngưỡng điểm ($3.0$).
5. Ranh giới dữ liệu Train - Calibration - Test độc lập 100%, không rò rỉ.

**HỆ THỐNG DỪNG TẠI ĐÂY THEO ĐÚNG CHỈ THỊ (STOP).** Không tự ý chạy Phase 4.1 Full Benchmark.

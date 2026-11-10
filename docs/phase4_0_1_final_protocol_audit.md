# BÁO CÁO KIỂM TOÁN GIAO THỨC THỰC NGHIỆM CUỐI CÙNG — PHASE 4.0.1
## (FINAL EXPERIMENTAL PROTOCOL AUDIT BEFORE FULL BENCHMARK)

**Dự án**: Chatbot hỏi đáp trên tài liệu với bộ nhớ liên tục đa thang căn theo cấu trúc văn bản (SA-CMS)  
**Mã giai đoạn**: Phase 4.0.1  
**Ngày thực hiện**: 03/10/2026  
**Trạng thái**: **AUDIT COMPLETE — ALL 6 GATES PASSED (SẴN SÀNG CHO BENCHMARK)**  
**Phần cứng**: NVIDIA GeForce GTX 1650 Ti (4GB VRAM), Python 3.9.13, PyTorch 2.x  
**Tập tin cấu hình chuẩn hóa**: [`configs/phase4_experiment.yaml`](configs/phase4_experiment.yaml)  
**Dữ liệu kiểm toán hiệu chuẩn RAG**: [`results/phase4_0_1_rag_calibration.csv`](results/phase4_0_1_rag_calibration.csv)  
**Dữ liệu kiểm toán phân hoạch**: [`results/phase4_0_1_protocol_check.json`](results/phase4_0_1_protocol_check.json)  

---

## 1. TỔNG QUAN MỤC TIÊU KIỂM TOÁN PHASE 4.0.1

Theo chỉ thị nghiên cứu, Phase 4.0.1 thực hiện kiểm toán và đóng băng dứt điểm **3 điểm chốt chặn cốt lõi** trước khi bước vào Phase 4.1 Full Benchmark:
1. **Định nghĩa thước đo Độ trung thực (Faithfulness metric definition)**: Xóa bỏ hoàn toàn chỉ tiêu tiên nghiệm `Faithfulness >= 95%`, chuyển sang định nghĩa chuẩn hóa khoa học và thiết lập quy trình kiểm định 2 lớp (Local Judge + Manual Verification).
2. **Tính công bằng trong huấn luyện & khởi tạo (Training fairness & initialization)**: Kiểm tra tính đồng nhất của $\theta_0$, cấu trúc tham số, và ngân sách cập nhật giữa B5 Fixed Token và P1 SA-CMS; bảo đảm P2 kế thừa nguyên vẹn thành phần bộ nhớ của P1.
3. **Hiệu chuẩn độc lập cho Baseline B2 RAG (RAG calibration protocol)**: Thực hiện quét siêu tham số trên tập hiệu chuẩn độc lập (Calibration Set), lựa chọn cấu hình tối ưu và đóng băng (freeze) trước khi đánh giá trên TEST.

> [!IMPORTANT]
> **CAM KẾT KHOA HỌC:**
> - KHÔNG chạy full benchmark ở giai đoạn này.
> - KHÔNG thay đổi công thức toán học và thuật toán SA-CMS (Equation 70, 71, 74).
> - KHÔNG thay đổi kiến trúc mô hình backend đã được kiểm định từ Phase 3.3.

---

## 2. KIỂM TOÁN TASK 1: ĐỊNH NGHĨA THƯỚC ĐO ĐỘ TRUNG THỰC (FAITHFULNESS)

### 2.1. Loại bỏ Ngưỡng Tiên nghiệm (Removal of Preset Target Threshold)
* Trong các phiên bản dự thảo trước, quy định `target: ">= 95%"` cho Faithfulness là một giả định tiên nghiệm chưa có cơ sở thực nghiệm, dễ dẫn đến thiên kiến xác nhận (confirmation bias).
* **Hiệu chỉnh chính thức:** Xóa bỏ hoàn toàn mọi ngưỡng đạt/không đạt (pass/fail threshold) trước khi có số liệu thực nghiệm.

### 2.2. Định nghĩa Khoa học Chuẩn mực:
$$\text{Faithfulness} = \text{Citation-Supported Answer Rate} = \frac{\sum_{i=1}^N \mathbb{I}(\text{Answer}_i \text{ is fully grounded in cited passages})}{\sum_{i=1}^N \mathbb{I}(\text{Answer}_i \text{ provides citations})}$$

### 2.3. Quy trình Đánh giá 2 Lớp (Dual Evaluation Pillars):
1. **Lớp 1 — Trọng tài Cục bộ (Local Judge)**:
   - Module [`src/hybrid_qa/evidence.py`](src/hybrid_qa/evidence.py) và [`faithfulness.py`](src/hybrid_qa/faithfulness.py).
   - Kiểm tra tự động độ trùng khớp từ khóa (lexical coverage), phát hiện thực thể và suy luận logic NLI (Natural Language Inference) giữa từng tuyên bố trong câu trả lời với nội dung đoạn trích dẫn.
2. **Lớp 2 — Kiểm chứng Thủ công Mù (Blind Manual Verification)**:
   - Thực hiện trên mẫu ngẫu nhiên gồm đúng **100 samples** từ tập TEST.
   - Chuyên viên đánh giá không được biết trước phương pháp sinh ra câu trả lời (B2 hay P2) để tránh thiên vị chủ quan.

---

## 3. KIỂM TOÁN TASK 2 & 3: TÍNH CÔNG BẰNG HUẤN LUYỆN & KHỞI TẠO (TRAINING FAIRNESS)

### 3.1. Bảng Đặc Tả Huấn Luyện & Khởi Tạo B4, B5, P1, P2:

| Thuộc tính | B4 (Single-Level Adapter) | B5 (Fixed-Token CMS) | P1 (SA-CMS Memory-Only) | P2 (SA-CMS + Retrieval Hybrid) |
| :--- | :---: | :---: | :---: | :---: |
| **Trạng thái khởi tạo ($\theta_0$)** | `torch.manual_seed(S)` | `torch.manual_seed(S)` | `torch.manual_seed(S)` | **Cùng hạt giống $S$ với P1** |
| **Độ lệch khởi tạo ($\Delta \theta_0$)**| — | **0.000000** | **0.000000** | **0.000000 (Dùng chung P1)** |
| **Số tầng bộ nhớ ($k$)** | 1 | 3 | 3 | 3 |
| **Số tham số Adapter trainable** | 1,772,736 (1.27%) | 5,315,904 (3.80%) | 5,315,904 (3.80%) | 5,315,904 (3.80%) |
| **Kích thước ($d_{\text{model}}, d_{\text{ff}}$)** | 576, 1536 | 576, 1536 | 576, 1536 | 576, 1536 |
| **Optimizer cập nhật online** | SGD (Eq 71) | SGD (Eq 71) | SGD (Eq 71) | SGD (Eq 71) |
| **Tốc độ học nội tại ($\eta^{(\ell)}$)**| $\eta = 0.01$ | $[0.01, 0.005, 0.001]$ | $[0.01, 0.005, 0.001]$ | $[0.01, 0.005, 0.001]$ |
| **Lịch cập nhật (Schedule)** | Fixed token | Fixed token chunk | Structure-aligned | Structure-aligned |
| **Ngân sách cập nhật ($\Delta_{\text{events}}$)**| 1 level | **$N$ events** | **$N$ events ($\Delta = 0$)** | **$N$ events ($\Delta = 0$)** |
| **Nhánh truy xuất ngoài** | Không | Không | Không | **Có (BM25 + Evidence + Refusal)** |

### 3.2. Kết quả Kiểm toán Thực tế giữa B5 và P1:
- **Kiểm tra trạng thái ban đầu:** Giá trị sai lệch cực đại $\max |\theta_0^{(B5)} - \theta_0^{(P1)}| = 0.000000$. Hai mô hình có không gian tham số và điểm khởi đầu đồng nhất tuyệt đối.
- **Kiểm tra ngân sách cập nhật:** Trên đoạn văn bản thử nghiệm kiểm toán, thuật toán [`StructureAlignedSchedule`](src/hope_attention/sa_cms.py) phân bổ chính xác **9 sự kiện cập nhật cho P1** và **9 sự kiện cập nhật cho B5** ($\Delta = 0$).
- **Kết luận:** Giữa B5 và P1 **hoàn toàn không có bất kỳ sự sai lệch nào ngoại trừ quy luật kích hoạt ranh giới cập nhật (Schedule)**. Điều kiện công bằng khoa học đạt mức tối đa.

### 3.3. Kiểm toán P2 (Task 3):
- Mô hình P2 sử dụng chính xác thành phần bộ nhớ tham số SA-CMS của P1.
- Không huấn luyện riêng rẽ hay thay đổi cấu trúc bộ nhớ của P2.
- Khác biệt duy nhất nằm ở pha suy luận (Inference): P2 kích hoạt nhánh truy xuất BM25, bộ lọc bằng chứng, bộ điều khiển từ chối và nhúng các đoạn trích dẫn vào prompt.

---

## 4. KIỂM TOÁN TASK 4: HIỆU CHUẨN RÀNG BUỘC CHO B2 RAG (RAG CALIBRATION)

Quy trình hiệu chuẩn được thực hiện thông qua kịch bản [`scripts/calibrate_b2_rag.py`](scripts/calibrate_b2_rag.py) trên tập **Calibration Set độc lập** (gồm 5 tài liệu `VAL_DOC_001`–`VAL_DOC_005` và 50 câu hỏi kiểm định, tách biệt 100% khỏi tập TEST).

### 4.1. Bảng Dữ Liệu Hiệu Chuẩn Các Cấu Hình Ứng Viên (results/phase4_0_1_rag_calibration.csv):

| Mã Cấu Hình | Chunk Size | Overlap | Top-K | BM25 $k_1$ | BM25 $b$ | Ngưỡng Score | Hit Rate (%) | MRR | Điểm Top-1 | Vượt Ngưỡng (%) | Latency (ms) | Điểm Tổng Hợp |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAND_07 (CHỌN)**| **256** | **32** | **5** | **1.5** | **0.75** | **3.0** | **60.0%** | **0.5867** | **8.00** | **98.0%** | **0.137** | **67.20** |
| `CAND_03` | 256 | 16 | 3 | 1.5 | 0.75 | 5.0 | 60.0% | 0.5867 | 8.00 | 78.0% | 0.135 | 63.20 |
| `CAND_04_BASE` | 256 | 32 | 5 | 1.5 | 0.75 | 5.0 | 60.0% | 0.5867 | 8.00 | 78.0% | 0.143 | 63.20 |
| `CAND_05` | 256 | 64 | 5 | 1.8 | 0.85 | 5.0 | 60.0% | 0.5867 | 8.06 | 78.0% | 0.137 | 63.20 |
| `CAND_06` | 256 | 32 | 7 | 1.5 | 0.65 | 5.0 | 60.0% | 0.5867 | 8.00 | 78.0% | 0.140 | 63.20 |
| `CAND_02` | 128 | 32 | 5 | 1.5 | 0.75 | 5.0 | 60.0% | 0.5750 | 8.30 | 78.0% | 0.161 | 62.85 |
| `CAND_01` | 128 | 16 | 3 | 1.2 | 0.75 | 5.0 | 58.0% | 0.5700 | 8.23 | 78.0% | 0.158 | 61.70 |
| `CAND_09` | 384 | 32 | 3 | 1.5 | 0.75 | 5.0 | 60.0% | 0.6000 | 6.99 | 74.0% | 0.075 | 60.80 |
| `CAND_10` | 384 | 48 | 5 | 1.5 | 0.75 | 5.0 | 60.0% | 0.6000 | 6.99 | 74.0% | 0.076 | 60.80 |
| `CAND_08` | 256 | 32 | 5 | 1.5 | 0.75 | 7.0 | 60.0% | 0.5867 | 8.00 | 56.0% | 0.133 | 58.80 |

### 4.2. Tiêu Chí Lựa Chọn & Quyết Định Đóng Băng:
* **Cấu hình chiến thắng được chọn**: **`CAND_07`**
  - Kích thước đoạn (Chunk size): **256 tokens**
  - Độ chồng lấn (Chunk overlap): **32 tokens**
  - Số đoạn truy xuất (Top-K): **5**
  - Siêu tham số BM25: $k_1 = 1.5, \quad b = 0.75$
  - Ngưỡng từ chối/bằng chứng (Score Threshold): **3.0**
* **Tiêu chí lựa chọn (Selection Criterion)**: Tối đa hóa điểm tổng hợp truy xuất (Hit Rate = 60.0%, MRR = 0.5867, tỷ lệ giữ lại bằng chứng = 98.0%) trong khi vẫn bảo đảm kích thước đoạn nằm gọn trong cửa sổ ngữ cảnh ($\le 256$ tokens) và độ trễ truy xuất cực thấp ($0.137\text{ ms}$).
* **LỆNH ĐÓNG BĂNG B2:** Cấu hình `CAND_07` này được áp dụng cố định cho B2 và nhánh truy xuất của P2 trong suốt Phase 4 benchmark. **Tuyệt đối không tinh chỉnh trên tập TEST.**

---

## 5. KIỂM TOÁN TASK 5: PHÂN HOẠCH DỮ LIỆU & CHỐNG RÒ RỈ (DATA PARTITION AUDIT)

Kết quả kiểm tra tự động qua kịch bản [`scripts/verify_protocol_and_partitions.py`](scripts/verify_protocol_and_partitions.py):

| Rào Chắn Kiểm Toán (Leakage Gate) | Số Mẫu Trùng Lặp Phát Hiện | Trạng Thái Đạt Chuẩn |
| :--- | :---: | :---: |
| **Tập Huấn luyện (TRAIN) vs Tập Hiệu chuẩn (CALIB)** | **0** | **PASS (100% độc lập)** |
| **Tập Huấn luyện (TRAIN) vs QASPER Test Docs** | **0** | **PASS (100% độc lập)** |
| **Tập Huấn luyện (TRAIN) vs Vietnamese Test Docs** | **0** | **PASS (100% độc lập)** |
| **Tài liệu Phase 3.1 / 3.1.1 xuất hiện trong TRAIN** | **0** | **PASS (Không bị ô nhiễm)** |
| **QASPER Test xuất hiện trong CALIBRATION** | **0** | **PASS (Không tune trên Test)**|
| **Vietnamese Test xuất hiện trong CALIBRATION** | **0** | **PASS (Không tune trên Test)**|
| **MK-NIAH Test xuất hiện trong CALIBRATION** | **0** | **PASS (Không tune trên Test)**|

```
================================================================================
KẾT LUẬN PHÂN HOẠCH: CÁCH LY TUYỆT ĐỐI (100% DATA ISOLATION VERIFIED)
- Tập TRAIN: 100 tài liệu (TR_DOC_001–100), 1.000 câu hỏi.
- Tập CALIBRATION: 5 tài liệu (VAL_DOC_001–005), 50 câu hỏi.
- Tập TEST: QASPER (10 docs), MK-NIAH (100 samples), Vietnamese Final (20 docs, 350 câu hỏi).
================================================================================
```

---

## 6. KIỂM TOÁN TASK 6: KHẢ NĂNG THỰC THI CÁC ABLATION YÊU CẦU

Kịch bản kiểm toán đã xác minh khả năng thực thi của các ablation trong mã nguồn:
1. **A1 (Random Boundary Schedule)**: Module `StructureAlignedSchedule(schedule_mode="random")` tạo thành công ranh giới ngẫu nhiên với số lượng span và tổng số update trùng khớp 100% với P1 (**FEASIBLE**).
2. **A2 (Level Count: 1, 2, 3)**: Khung `StructureAlignedHopeLM` hỗ trợ đầy đủ `num_levels=1` (B4), `num_levels=2` (A2) và `num_levels=3` (P1/B5) (**FEASIBLE**).
3. **A3 (Budget Parity Schedule)**: Cơ chế hierarchical budget matching bảo đảm B5 và P1 có cùng số sự kiện cập nhật (**FEASIBLE**).
4. **Aggregation Toggle**: Hỗ trợ chuyển đổi giữa `SequentialMLPChain` (phương trình 70) và `IndependentMLPChain` (phương trình 74, gated/additive) (**FEASIBLE**).
5. **Self-Study & Adapter Rank**: Có sẵn cấu trúc cờ cấu hình trong file YAML để kích hoạt khi chạy ablation chuyên sâu.

---

## 7. KIỂM TOÁN TASK 7: TỆP CẤU HÌNH ĐÓNG BĂNG ĐẦY ĐỦ

Tệp [`configs/phase4_experiment.yaml`](configs/phase4_experiment.yaml) đã được cập nhật bổ sung đầy đủ 4 section chuẩn hóa:
- `faithfulness_definition`: Định nghĩa tỷ lệ câu trả lời có trích dẫn được bằng chứng hỗ trợ, không có ngưỡng cứng.
- `training_fairness`: Khẳng định tính công bằng tuyệt đối giữa B5 và P1, cơ chế kế thừa P2.
- `rag_calibration`: Lưu thông số đóng băng của cấu hình `CAND_07`.
- `data_partitions`: Ranh giới dữ liệu và các rào chắn chống rò rỉ.

---

## 8. KẾT QUẢ KIỂM THỬ HỒI QUY (REGRESSION TESTS)

Chạy lại toàn bộ bộ kiểm thử tự động của dự án:
```
pytest
====================== 100 passed, 2 warnings in 25.63s =======================
```
*Tất cả 100 bài kiểm thử đơn vị và tích hợp từ Phase 1 đến Phase 3.3 tiếp tục vượt qua 100%.*

---

## 9. ĐIỀU KIỆN DỪNG (STOP CONDITION)

Giai đoạn **Phase 4.0.1 — Final Experimental Protocol Audit** đã hoàn tất:
- ✅ Định nghĩa Faithfulness đã được sửa đổi chuẩn mực khoa học.
- ✅ Tính công bằng huấn luyện và khởi tạo đã được xác minh bằng toán học và thực nghiệm ($\Delta = 0$).
- ✅ B2 RAG đã được hiệu chuẩn độc lập trên Calibration Set và đóng băng cố định.
- ✅ Phân hoạch dữ liệu đạt cách ly 100% chống rò rỉ.
- ✅ Tệp cấu hình, báo cáo và dữ liệu kiểm toán đã được ban hành đầy đủ.

**HỆ THỐNG DỪNG TẠI ĐÂY.** Không chạy Phase 4.1 Benchmark cho đến khi có chỉ thị tiếp theo.

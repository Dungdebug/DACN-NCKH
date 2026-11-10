# GIAO THỨC THỰC NGHIỆM CHÍNH THỨC — PHASE 4.0 PROTOCOL FREEZE
## (Structure-Aligned Continuum Memory System for Document QA Benchmark)

**Dự án**: Chatbot hỏi đáp trên tài liệu với bộ nhớ đa thang dựa trên Nested Learning  
**Ngày đóng băng giao thức**: 03/10/2026  
**Trạng thái**: **PROTOCOL FROZEN (ĐÃ ĐÓNG BĂNG TOÀN DIỆN)**  
**Phần cứng chuẩn hóa**: NVIDIA GeForce GTX 1650 Ti (4GB VRAM), Python 3.9.13, Windows 11  
**Mô hình nền (Backbone)**: HuggingFaceTB/SmolLM2-135M (134,515,008 tham số, đóng băng 100%)  
**Tập tin cấu hình gốc**: [`configs/phase4_experiment.yaml`](configs/phase4_experiment.yaml)  
**Kết quả Pilot Sanity Check**: [`results/phase4_0_sanity_check.json`](results/phase4_0_sanity_check.json)  

---

## NGUYÊN TẮC CỐT LÕI (SOURCE OF TRUTH)
Giao thức thực nghiệm này được xây dựng dựa trên sự thống nhất tuyệt đối giữa:
1. **Đề cương NCKH** (chốt phương án A, tích hợp B và C, mục 5 và mục 7).
2. **Nested Learning Paper** (arXiv:2512.24695v1 — Phương trình 70, 71, 74, Hình 7).
3. **Phase 2.5.2 Final Reconciled Results** (Bảo toàn kết quả âm tính của RQ2 trên scale sinh viên).
4. **Phase 3.0 — 3.3 Final Backend Implementation** (Hệ thống SA-CMS 3 cấp độ, BM25, Refusal Controller, Evidence Grounding, Citations).

> [!IMPORTANT]
> **QUY TẮC ĐÓNG BĂNG:** Sau khi tài liệu này và file `configs/phase4_experiment.yaml` được ban hành, runner thực nghiệm tuyệt đối không được tự ý sinh hay điều chỉnh bất kỳ siêu tham số (hyperparameters), kiến trúc mô hình, hoặc ngưỡng quyết định nào. Mọi so sánh phải tuân thủ đúng ngân sách kiểm soát (controlled budget).

---

## TASK 1 — ĐÓNG BĂNG 7 PHƯƠNG PHÁP SO SÁNH (FREEZE 7 METHODS)

Hệ thống thực nghiệm Phase 4 chuẩn hóa chính xác 7 phương pháp theo Đề cương:

| Mã | Tên phương pháp | Phân loại | Mô tả kỹ thuật & Luồng xử lý | Cập nhật tham số khi đọc? | Context khi hỏi (Prompt) |
|:---|:---|:---:|:---|:---:|:---:|
| **B1** | **ICL Full-Document Context** | Baseline | Đưa toàn văn tài liệu vào cửa sổ ngữ cảnh (`max_context=512`). Mốc so sánh chuẩn của bài báo gốc. | Không (0 update) | Toàn văn tài liệu |
| **B2** | **Standard BM25 RAG** | Baseline | Chia tài liệu thành các đoạn 256 token (overlap 32). Truy xuất BM25 lấy top-5 đoạn đưa vào ngữ cảnh sinh câu trả lời. | Không (0 update) | Top-5 đoạn trích xuất |
| **B3** | **Cartridges / Context Compression** | Baseline | Nén ngữ cảnh bằng biểu diễn nhẹ tự học qua self-study (Eyuboglu et al., arXiv:2506.06266). | *Không khả thi trong ngân sách* | — |
| **B4** | **Single-Level Adapter** | Baseline | Bộ nhớ tham số không đa thang: 1 adapter MLP (`num_levels=1`, 1,772,736 params), cập nhật gradient theo Eq 71 khi đọc. Sau đó loại bỏ văn bản khỏi ngữ cảnh. | Có (Level 1) | Câu hỏi (văn bản đã rời context) |
| **B5** | **Fixed-Token CMS (Hope Scaled)** | Baseline chính | Baseline đa thang từ bài báo gốc (k=3 levels, 5,315,904 params): cập nhật theo chu kỳ token cố định $C^{(\ell)}$. Context bị loại bỏ khi hỏi. | Có (3 mức token cố định) | Câu hỏi (văn bản đã rời context) |
| **P1** | **SA-CMS Memory-Only** | Đề xuất | Bộ nhớ liên tục đa thang căn theo cấu trúc: Level 1 (Đoạn), Level 2 (Mục), Level 3 (Tài liệu). Cập nhật theo Eq 71. Context bị loại bỏ khi hỏi; trả lời thuần tham số. | Có (3 mức cấu trúc) | Câu hỏi (văn bản đã rời context) |
| **P2** | **SA-CMS + Retrieval Hybrid** | Đề xuất (Bản lai) | Kiến trúc lai hoàn chỉnh: Bộ nhớ tham số SA-CMS 3 cấp độ + Truy xuất BM25 + Bộ chọn bằng chứng (Evidence) + Bộ điều khiển từ chối (Refusal) + Dẫn chứng (Citation). | Có (3 mức cấu trúc) | Top-5 đoạn bằng chứng + Bộ nhớ tham số |

### Quy định nghiêm ngặt đối với B3 (Cartridges):
- Theo đúng tinh thần của Đề cương NCKH (mục 3 và mục 7.2) và giới hạn phần cứng (GPU GTX 1650 Ti 4GB VRAM): Phương pháp Cartridges đòi hỏi quá trình tự sinh câu hỏi diện rộng (heavy self-study generation) và chưng cất biểu diễn offline với mô hình giáo viên quy mô lớn, vượt quá ngưỡng tài nguyên và VRAM cho phép.
- **Tuyên bố chính thức của B3:** `"not reproducible within controlled budget (4GB VRAM / single GPU constraint; requires heavy offline teacher distillation and representation pre-baking exceeding budget)"`.
- **Tuyệt đối KHÔNG thay thế B3 bằng phương pháp khác.**

### Các cấu hình Ablation bổ trợ:
- **A1 (Random Boundary)**: Giữ nguyên $k=3$ mức và tổng số lần cập nhật y hệt P1, nhưng ranh giới được sinh ngẫu nhiên để tách biệt tác động của cấu trúc văn bản khỏi tác động của độ dài span biến thiên.
- **A2 (SA-CMS Level 2)**: Tắt Level 3 (Document Memory) để đo lường đóng góp của cấp độ tài liệu toàn cục.
- **A3 (Additive Aggregation)**: Thay thế cổng học sigmoid $g_\ell(x)$ trong Phương trình 74 bằng phép cộng trực tiếp để kiểm chứng vai trò của cơ chế chọn lọc thông tin.

---

## TASK 2 — BẢO ĐẢM TÍNH CÔNG BẰNG PHẦN CỨNG & MÔ HÌNH (HARDWARE & FAIRNESS)

Tất cả 7 phương pháp được thực thi trên cùng một nền tảng thực nghiệm duy nhất:

1. **Backbone mô hình**: `HuggingFaceTB/SmolLM2-135M` (134,515,008 tham số, đóng băng 100% trọng số).
2. **Tokenizer**: Tokenizer chính thức của `SmolLM2-135M` (vocab size = 49,152), padding token chuẩn hóa.
3. **Cửa sổ ngữ cảnh (Context Window)**: Cố định trần 512 token trên toàn bộ các phương pháp.
4. **Độ chính xác (Precision / Quantization)**:
   - GPU: `torch.float16` (bảo đảm an toàn VRAM $\le 3.5\text{ GB}$).
   - CPU: `torch.float32`.
5. **Bộ câu hỏi đánh giá (Evaluation Questions)**: Dùng chung 100% cùng một bộ câu hỏi kiểm thử đối ứng từng sample một.
6. **Tiền xử lý tài liệu (Document Preprocessing)**: Dùng chung module `DocumentStructureParser` (chuẩn hóa Unicode NFKC, phát hiện tiêu đề Markdown, tách đoạn theo newline đôi).

### Ma trận khác biệt bắt buộc (Mandatory Differences Matrix):

| Phương pháp | Nội dung trong Prompt | Số lần cập nhật tham số (Updates) | Cửa sổ ngữ cảnh sử dụng khi hỏi | Khả năng Trích dẫn (Citations) | Khả năng Từ chối (Refusal) |
|:---|:---|:---:|:---:|:---:|:---:|
| **B1** | Full Document + Question | 0 | $\approx 256 - 512$ tokens | Không | Không |
| **B2** | Top-5 Retrieved Passages + Question | 0 | $\approx 200 - 300$ tokens | Có (qua BM25) | Có (Refusal Controller) |
| **B4** | Question Only (Evicted Context) | $N_{\text{updates}}$ (1 mức) | $< 64$ tokens | Không | Không |
| **B5** | Question Only (Evicted Context) | $N_{\text{budget}}$ (3 mức token cố định) | $< 64$ tokens | Không | Không |
| **P1** | Question Only (Evicted Context) | $N_{\text{budget}}$ (3 mức cấu trúc) | $< 64$ tokens | Không | Không |
| **P2** | Top-5 Evidence Passages + Question | $N_{\text{budget}}$ (3 mức cấu trúc) | $\approx 200 - 300$ tokens | Có (qua BM25 + Refusal) | Có (Refusal Controller) |

---

## TASK 3 — CÂU HỎI NGHIÊN CỨU CHÍNH (PRIMARY RESEARCH QUESTIONS)

Toàn bộ báo cáo khoa học Phase 4 giải quyết tuần tự 5 Research Questions cố định từ Đề cương:

* **RQ1 (Multi-level CMS Reproduction)**:  
  *CMS nhiều mức có tái hiện được xu hướng trên QASPER/MK-NIAH ở quy mô nhỏ không?*  
  - *Mục tiêu*: Xác minh xem việc tăng số mức bộ nhớ ($k=3$ so với $k=1$) có cải thiện khả năng duy trì thông tin dài hạn như trong paper gốc hay không.
  - *So sánh cốt lõi*: B5 vs B4 vs B1.

* **RQ2 (Structure-Aligned vs Fixed-Token Budget Controlled)**:  
  *P1 SA-CMS có khác B5 Fixed Token khi cùng update budget không?*  
  - *Mục tiêu*: Kiểm chứng giả thuyết ranh giới ngữ nghĩa cấu trúc (Paragraph/Section/Document) có tạo ra ưu thế vượt trội so với ranh giới token cơ học ở cùng ngân sách cập nhật gradient hay không. Bảo toàn kết quả kiểm định âm tính từ Phase 2.5.2 nếu hiện tượng lặp lại.
  - *So sánh cốt lõi*: P1 vs B5 (Kiểm định thống kê bắt cặp).

* **RQ3 (Post-Eviction Retention & Faithfulness)**:  
  *Sau khi document rời context: memory-only và hybrid retrieval hoạt động thế nào? P2 có cải thiện faithfulness so với B2 không?*  
  - *Mục tiêu*: Đo lường mức độ suy giảm khi tài liệu rời khỏi context window, và chứng minh bản lai P2 đạt độ trung thực cao nhất nhờ kết hợp bộ nhớ tham số với truy xuất ngoại biên.
  - *So sánh cốt lõi*: P2 vs B2 vs P1.

* **RQ4 (Continual Document Ingestion Forgetting)**:  
  *Sequential document ingestion gây forgetting bao nhiêu?*  
  - *Mục tiêu*: Đo lường mức độ suy giảm tri thức trên tài liệu gốc $D_0$ sau khi nạp liên tiếp 5, 10, và 20 tài liệu mới.
  - *So sánh cốt lõi*: Đường cong suy giảm độ chính xác của P1 vs B4.

* **RQ5 (Computational and Memory Cost Profile)**:  
  *Cost của memory-based methods so với RAG/context dài?*  
  - *Mục tiêu*: Định lượng trade-off giữa chi phí nạp (ingestion overhead) và chi phí truy vấn (inference token cost, latency, VRAM).

---

## TASK 4 — HỆ THỐNG THƯỚC ĐO CHÍNH (PRIMARY METRICS)

1. **Chất lượng trả lời (Answer Quality)**:
   - **Token F1**: F1-score cấp độ từ vựng giữa câu trả lời sinh ra và ground-truth answer.
   - **Exact Match (EM)**: Tỷ lệ khớp hoàn toàn chuỗi văn bản sau chuẩn hóa (loại bỏ mạo từ, dấu câu, khoảng trắng thừa).
   - **MK-NIAH Accuracy**: Tỷ lệ truy xuất chính xác mã bí mật 5 chữ số trong bài toán multi-key needle haystack.
   - **Target Token Probability**: Xác suất Softmax của token mục tiêu tại bước suy luận đầu tiên.

2. **Độ trung thực (Faithfulness)**:
   - **Citation Support Rate**: Tỷ lệ phần trăm các câu trả lời có gắn mã trích dẫn mà đoạn trích dẫn thực sự chứa dữ kiện hỗ trợ.
   - **Local Judge Score**: Điểm số đánh giá độ bao phủ ngữ nghĩa và logic giữa câu trả lời và bằng chứng trích xuất.
   - **Manual Verification Audit**: Quy trình kiểm tra thủ công (blind human audit) trên mẫu ngẫu nhiên gồm đúng **100 samples**.

3. **Từ chối (Refusal Performance)**:
   - **Correct Refusal Rate**: Tỷ lệ từ chối chính xác trên tập câu hỏi không có đáp án (`Unanswerable`) và câu hỏi thiếu bằng chứng (`Insufficient Evidence`).
   - **False Refusal Rate**: Tỷ lệ từ chối nhầm trên tập câu hỏi có đáp án (`Answerable`).

4. **Độ quên (Catastrophic Forgetting)**:
   - Tỷ lệ duy trì độ chính xác / F1 trên tài liệu gốc $D_0$ tại 3 mốc nạp tuần tự:
     $$\Delta \text{Acc}_{+5}, \quad \Delta \text{Acc}_{+10}, \quad \Delta \text{Acc}_{+20}$$

5. **Chi phí tính toán (Computational Cost)**:
   - **Ingest Time**: Thời gian nạp tính bằng giây trên 1.000 token ($\text{s} / 1\text{k tokens}$).
   - **Answer Tokens**: Số lượng token trung bình sinh ra cho mỗi câu trả lời.
   - **Latency**: Thời gian đáp ứng đầu cuối tính bằng mili-giây trên mỗi truy vấn ($\text{ms} / \text{query}$).
   - **Peak VRAM**: Dung lượng bộ nhớ đồ họa đỉnh điểm sử dụng trong suốt quá trình chạy (MB).
   - **Checkpoint Size**: Dung lượng lưu trữ của tệp snapshot bộ nhớ tham số trên ổ đĩa (MB).

---

## TASK 5 — GIAO THỨC THỐNG KÊ (STATISTICAL PROTOCOL)

1. **Số lượng Hạt giống (Seeds)**:
   - Mỗi cấu hình được chạy lặp lại trên đúng 3 hạt giống chuẩn:
     $$\text{Seed} \in \{42, 43, 44\}$$

2. **Chỉ số tổng hợp**:
   - Báo cáo giá trị Trung bình (Mean) và Độ lệch chuẩn (Standard Deviation - SD).

3. **Khoảng tin cậy Bootstrap 95% (Bootstrap CI 95%)**:
   - Áp dụng phương pháp Non-parametric Percentile Bootstrap với số vòng lặp $B = 1000$.
   - **QUY TẮC BẮT BUỘC**: Phép tái lấy mẫu (resampling) được thực hiện trên **toàn bộ từng câu hỏi / sample đánh giá cá thể** ($N \ge 100$ items).  
     > [!CAUTION]
     > Tuyệt đối KHÔNG thực hiện bootstrap trên trung bình của 3 seeds (tránh sai số mẫu nhỏ $N=3$).

4. **Kiểm định giả thuyết bắt cặp (Paired Statistical Testing)**:
   - Để so sánh đối đầu giữa **P1 (SA-CMS)** và **B5 (Fixed Token)**:
     - Thực hiện kiểm định t bắt cặp (Paired Student's t-test).
     - Thực hiện kiểm định phi tham số Wilcoxon Signed-Rank Test trên cùng một tập câu hỏi.
     - Ngưỡng ý nghĩa thống kê: $\alpha = 0.05$.
     - Báo cáo kèm kích thước hiệu ứng (Effect Size): Cohen's $d$.

---

## TASK 6 — PHÂN HOẠCH DỮ LIỆU & CHỐNG RÒ RỈ (DATA PARTITION & LEAKAGE SAFEGUARDS)

Thiết lập phân vùng 3 tập dữ liệu độc lập tuyệt đối:

```mermaid
graph LR
    subgraph TRAIN [1. Tập Huấn Luyện (TRAIN)]
        T1[500-1000 Cặp QA Tự Sinh Độc Lập]
        T2[Chỉ dùng khởi tạo Adapter & Cổng Gating]
    end

    subgraph VAL [2. Tập Hiệu Chuẩn (VALIDATION)]
        V1[Tài liệu Sanity doc_a, doc_b, doc_c]
        V2[Kiểm thử kết nối Pipeline & Sanity Check]
    end

    subgraph TEST [3. Tập Kiểm Thử Benchmark (TEST)]
        E1[MK-NIAH: 100 Haystack Samples]
        E2[QASPER: 10 Research Papers]
        E3[Vietnamese QA: 20 Docs, 350 Questions]
        E4[Continual Forgetting: 21 Sequential Docs]
    end

    TRAIN -.->|Cách ly 100%| TEST
    VAL -.->|Không tune Threshold| TEST
```

### Các rào chắn chống rò rỉ dữ liệu (Zero Leakage Gates):
1. **Không dùng tập Phase 3.1 / 3.1.1 test để huấn luyện**: Dữ liệu huấn luyện adapter pilot chỉ lấy từ bộ dữ liệu tổng hợp độc lập (`src/training/training_corpus.py`).
2. **Không tune ngưỡng từ chối trên tập TEST**: Các ngưỡng của Refusal Controller được đóng băng cố định từ Phase 3.1:
   - `score_threshold = 5.0`
   - `min_evidence_score = 5.0`
   - `min_evidence_count = 1`
   - `min_query_coverage = 0.35`
3. **Không tune siêu tham số RAG trên tập TEST**: Các tham số BM25 được đóng băng cố định:
   - $k_1 = 1.5, \quad b = 0.75, \quad \text{top\_k} = 5$
4. **Không dùng tập tiếng Việt cuối cùng (Vietnamese Final Test) làm calibration**: Bộ 20 tài liệu tiếng Việt và 350 câu hỏi được niêm phong cho đánh giá kiểm định cuối cùng.

---

## TASK 7 — KIỂM SOÁT NGÂN SÁCH CẬP NHẬT (CONTROLLED UPDATE BUDGET)

Để phân định rành mạch giữa đóng góp của kiến trúc ngữ nghĩa và số lượng gradient step:
1. Đối với **B5 (Fixed Token)**, **P1 (SA-CMS)**, và **Ablation A1 (Random Boundary)**:
   - Bắt buộc phải duy trì **chính xác cùng một tổng số sự kiện cập nhật gradient (Total Update Events, $\Delta = 0$)** trên mỗi tài liệu.
   - Khi P1 kích hoạt tại các ranh giới đoạn/mục/tài liệu tạo ra $K$ sự kiện cập nhật, thuật toán của B5 và A1 phải phân bổ đúng $K$ sự kiện cập nhật trên chuỗi token tương ứng.
2. Mọi sự kiện cập nhật đều phải được ghi log chi tiết:
   - Cấp độ bộ nhớ ($\ell \in \{1, 2, 3\}$).
   - Loại ranh giới (Paragraph, Section, Document, hoặc Fixed Chunk).
   - Vị trí token (Token Index).
   - Chuẩn độ dốc gradient ($\|\nabla_\theta \mathcal{L}\|$).
   - Thang thời gian (Timescale).

---

## TASK 8 — CẤU HÌNH THỰC NGHIỆM ĐÓNG BĂNG (`configs/phase4_experiment.yaml`)

Tệp cấu hình đã được tạo lập tại [`configs/phase4_experiment.yaml`](configs/phase4_experiment.yaml) với đầy đủ các tham số định lượng:

```yaml
meta:
  protocol_version: "4.0.0-frozen"
  freeze_date: "2026-10-03"

hardware_fairness:
  backbone_model: "HuggingFaceTB/SmolLM2-135M"
  tokenizer: "HuggingFaceTB/SmolLM2-135M"
  max_context_window: 512
  precision_gpu: "float16"
  target_gpu_hardware: "NVIDIA GeForce GTX 1650 Ti (4GB VRAM)"

hyperparameters:
  sa_cms:
    num_levels: 3
    inner_learning_rates:
      level_1: 0.01   # Paragraph
      level_2: 0.005  # Section
      level_3: 0.001  # Document
    aggregation: "gated"
  retrieval_bm25:
    k1: 1.5
    b: 0.75
    top_k: 5
  refusal_controller:
    score_threshold: 5.0
    min_evidence_score: 5.0
    min_query_coverage: 0.35
    thresholds_frozen: true

statistics:
  seeds: [42, 43, 44]
  bootstrap:
    iterations: 1000
    confidence_level: 0.95
```

---

## TASK 9 — KẾT QUẢ PILOT SANITY CHECK TOÀN DIỆN

Trước khi khởi động full benchmark, kịch bản [`scripts/run_phase4_sanity_check.py`](scripts/run_phase4_sanity_check.py) đã thực hiện kiểm thử trên mẫu kiểm định độc lập cho toàn bộ các phương pháp.

### Kết quả kiểm định tính toàn vẹn (Integrity Audit Matrix):

| Hạng mục kiểm tra | Trạng thái | Chi tiết phát hiện |
|:---|:---:|:---|
| **1. Pipeline Execution** | **PASS** | Tất cả các phương pháp B1, B2, B4, B5, P1, P2 thực thi thông suốt, không gặp lỗi runtime. |
| **2. Output Schema Validity** | **PASS** | 100% kết quả xuất định dạng chuẩn `QAResult` với đầy đủ trường dữ liệu. |
| **3. Citation Traceability** | **PASS** | Các trích dẫn sinh ra từ B2 và P2 (như `SANITY_DOC_001::P005`, `P006`, `P010`, `P011`) 100% trỏ đúng vào các passage có thực trong cơ sở dữ liệu. |
| **4. Refusal Validity** | **PASS** | Câu hỏi ngoài lề `SANITY_VAL_003` (tàu Perseverance) kích hoạt từ chối chính xác với lý do chuẩn `no_relevant_evidence_found`. |
| **5. Metrics Parser Integrity** | **PASS** | Các hàm tính toán Token F1, Exact Match, Accuracy, Latency hoạt động trơn tru. |
| **6. Numerical Stability** | **PASS** | Kiểm tra toàn bộ tham số, trọng số adapter, gradient và logits: **0 giá trị NaN, 0 giá trị Inf**. |
| **7. Data Leakage Audit** | **PASS** | 100% mẫu kiểm thử mang định danh `SANITY_VAL_*`; không có sự xâm nhập của dữ liệu benchmark test. |
| **8. Budget Logging Verification**| **PASS** | B5 và P1 ghi nhận số lượng sự kiện cập nhật bằng nhau tuyệt đối (**9 events** trên tài liệu kiểm định). |

### Bảng tóm tắt kết quả đo đạc Pilot:

| Phương pháp | Trạng thái | Số cấp độ | Số lần Update | Latency trung bình (ms) | Đặc điểm suy luận ghi nhận |
|:---|:---:|:---:|:---:|:---:|:---|
| **B1 (ICL)** | VALIDATED | 0 | 0 | 8,461.8 | Xử lý toàn văn trong prompt; độ trễ cao nhất do cửa sổ ngữ cảnh đầy. |
| **B2 (RAG)** | VALIDATED | 0 | 0 | 4,078.3 | Dẫn chứng trích xuất chuẩn xác; từ chối câu unanswerable với độ trễ 0.16 ms. |
| **B3 (Cartridges)** | EXCLUDED | — | — | — | Xác nhận thông báo chuẩn: *not reproducible within controlled budget*. |
| **B4 (1-Level)** | VALIDATED | 1 | 4 | 1,960.9 | Context đã rời cửa sổ; sinh câu trả lời nhanh thuần tham số 1 cấp độ. |
| **B5 (Fixed Token)**| VALIDATED | 3 | 9 | 2,007.4 | Ngân sách cập nhật cân bằng chính xác với P1 (9 updates). |
| **P1 (SA-CMS)** | VALIDATED | 3 | 9 | 1,959.4 | Cập nhật căn theo cấu trúc; sinh câu trả lời nhanh thuần tham số 3 cấp độ. |
| **P2 (Hybrid)** | VALIDATED | 3 | 9 | 4,040.7 | Kết hợp bộ nhớ tham số + dẫn chứng BM25 + từ chối có kiểm soát. |

> [!NOTE]
> Kết quả đo đạc của Pilot Sanity Check được lưu trữ độc lập tại [`results/phase4_0_sanity_check.json`](results/phase4_0_sanity_check.json) và **TUYỆT ĐỐI KHÔNG ĐƯỢC ĐƯA VÀO BẢNG THỐNG KÊ FINAL BENCHMARK**.

---

## TASK 10 — KẾT LUẬN & ĐIỀU KIỆN DỪNG (STOP CONDITION)

Giai đoạn **Phase 4.0 — Experimental Protocol Freeze** đã hoàn thành 100% các mục tiêu đề ra:
1. Đóng băng định nghĩa của 7 phương pháp (B1–B5, P1, P2) cùng các ablation đối chứng.
2. Thiết lập rào chắn công bằng phần cứng (SmolLM2-135M, context 512, FP16 trên GTX 1650 Ti).
3. Đóng băng nguyên vẹn 5 Research Questions (RQ1–RQ5) và hệ thống thước đo.
4. Xác lập giao thức thống kê nghiêm ngặt (3 hạt giống, bootstrap sample-level, paired test P1 vs B5).
5. Phân hoạch dữ liệu 3 vùng cách ly triệt để chống rò rỉ.
6. Ban hành tệp cấu hình chuẩn [`configs/phase4_experiment.yaml`](configs/phase4_experiment.yaml).
7. Hoàn tất kiểm định Pilot Sanity Check với kết quả **ALL PASSED**.

**QUY TẮC DỪNG (STOP):** Hệ thống dừng thực thi tại đây. Tuyệt đối không tự ý chạy full Phase 4 benchmark cho đến khi có chỉ lệnh tiếp theo từ người dùng.

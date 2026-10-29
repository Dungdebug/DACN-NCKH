# TỔNG HỢP TOÀN BỘ THÔNG TIN DỰ ÁN DACN-NCKH (SA-CMS & HYBRID QA)

Tài liệu này tổng hợp đầy đủ cơ sở lý thuyết, kiến trúc thuật toán, cấu trúc mã nguồn, kết quả thực nghiệm và hướng dẫn vận hành của dự án **DACN-NCKH**.

---

## 1. Tổng quan Dự án & Cơ sở Lý thuyết

* **Tên đề tài**: Hệ thống Chatbot Hỏi đáp Tài liệu dựa trên Nested Learning: Bộ nhớ Tham số Đa thang Căn chỉnh Cấu trúc (**SA-CMS** — *Structure-Aligned Continuum Memory System*) kết hợp Truy hồi Dẫn chứng (**Hybrid QA — Reading Before Retrieving**).
* **Bài báo nền tảng (Baseline)**: *Nested Learning: The Illusion of Deep Learning Architectures* (Behrouz et al., NeurIPS 2025 / `arXiv:2512.24695v1` — file gốc tại `2512.24695v1.pdf`, đề cương nghiên cứu tại `de_cuong_text.txt` / `de_cuong_chatbot_nested_learning.docx`).
* **Mô hình nền (Backbone)**: `HuggingFaceTB/SmolLM2-135M` (134,515,008 tham số, đóng băng 100% trọng số gốc, lưu trữ offline tại `local_runtime/`).
* **Tham số huấn luyện (SA-CMS Adapters)**: **5,314,752 tham số** phân bổ trên 3 mức bộ nhớ (Mức 1: 1.77M, Mức 2: 3.54M, Mức 3: 5.31M tham số với phân rã hạng thấp $r=16, d=576$).
* **Môi trường phần cứng mục tiêu**:
  * **Máy local**: NVIDIA GTX 1650 Ti (4GB VRAM) — phục vụ suy luận (inference), thăm dò cơ học (mechanistic probing) và chạy Web UI.
  * **Máy train thực nghiệm**: NVIDIA RTX 3050 Laptop GPU (6GB VRAM) — đỉnh tiêu thụ VRAM khi huấn luyện chỉ **743.86 MB** và khi suy luận là **440.40 MB**.
* **Độ tin cậy kỹ thuật**:
  * Vượt qua **161/161 Unit & Integration Tests** (`pytest tests/`).
  * Khớp **53/53 mã băm SHA-256** (`training_output_final/checksums.sha256`) cho toàn bộ checkpoints và kết quả thực nghiệm.

---

## 2. Kiến trúc Thuật toán Cốt lõi

### 2.1. Bộ nhớ Đa thang Căn chỉnh Cấu trúc (SA-CMS)
Thay vì cập nhật trọng số sau mỗi $C^{(\ell)}$ token cố định như CMS gốc (vốn dễ cắt ngang câu hoặc mệnh đề ngữ nghĩa), SA-CMS căn chỉnh lịch cập nhật theo ranh giới cấu trúc văn bản tự nhiên (Bảng 6.2 Đề cương):

| Mức bộ nhớ | Đơn vị cấu trúc | Lưu trữ thông tin | Thời điểm cập nhật | Chính sách Reset | Quy mô tham số |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mức 0** | In-Context | Đoạn đang đọc & đoạn truy hồi | Mỗi token | Khi hết cửa sổ ngữ cảnh (1,024 tokens) | Cửa sổ 1,024 tokens |
| **Mức 1** | Đoạn văn ($\ell=1$) | Chi tiết thực thể, từ vựng cục bộ | Cuối mỗi đoạn ($b_1(t)=1$) | **Reset ở cuối mỗi mục** (ngăn nhiễu cục bộ lan truyền) | 1,771,584 ($r=16$) |
| **Mức 2** | Chương mục ($\ell=2$) | Luận điểm và liên kết của từng mục | Cuối mỗi mục ($b_2(t)=1$) | Không reset trong tài liệu | 1,771,584 ($r=16$) |
| **Mức 3** | Tài liệu ($\ell=3$) | Tri thức bền vững toàn tài liệu/kho | Cuối mỗi tài liệu ($b_3(t)=1$) | Không reset (duy trì xuyên suốt) | 1,771,584 ($r=16$) |
| **Mức 4** | Backbone nền | Tri thức ngôn ngữ pre-train | Không bao giờ cập nhật | Đóng băng hoàn toàn ($\mathrm{MLP}_{\mathrm{pre}}$) | 134.5M Backbone |

* **Công thức cập nhật theo cấu trúc (Mở rộng Phương trình 71)**:
  $$\theta^{(\ell)} \leftarrow \theta^{(\ell)} - \frac{\eta^{(\ell)}}{|\mathcal{B}_\ell|} \sum_{t \in \mathcal{B}_\ell} \nabla_{\theta^{(\ell)}} \mathcal{L}(x_t) \quad \text{khi } b_\ell(t) = 1$$
  *(Khi ranh giới rơi vào bội số cố định của $C^{(\ell)}$, công thức suy biến về đúng baseline CMS gốc).*

* **Cơ chế tổng hợp có cổng học được (Phương trình 74)**:
  $$y_t = \mathrm{MLP}_{\mathrm{pre}}(x_t) + \sum_{\ell=1}^{3} g_\ell(x_t)\, B_\ell A_\ell\, x_t, \quad g_\ell(x_t) = \sigma(W_{g,\ell}\cdot \mathrm{RMSNorm}(x_t))$$
  *(Khi các cổng $g_\ell \to 0$, mô hình bảo toàn 100% hành vi của backbone gốc).*

### 2.2. Hệ thống Hỏi đáp Lai (Hybrid QA — 3 Chế độ)
Được hiện thực hóa trong `src/hybrid_qa/pipeline.py` (`HybridQAPipeline`):
1. **MODE A (`context` — Baseline B1/B2)**: Đưa trực tiếp văn bản hoặc đoạn truy hồi vào cửa sổ ngữ cảnh.
2. **MODE B (`memory` — P1/B4/B5)**: Sau khi nạp tài liệu vào trọng số SA-CMS, xóa văn bản khỏi ngữ cảnh (`context_removed = True`) và chỉ trả lời dựa trên bộ nhớ tham số.
3. **MODE C (`hybrid` — Phương án P2 "Reading Before Retrieving")**:
   * Nạp biểu diễn tham số SA-CMS để nắm ngữ cảnh tổng thể.
   * Truy xuất top-5 đoạn văn qua `BM25Retriever` ($k_1=1.5, b=0.75$).
   * Lọc bằng chứng qua `EvidenceSelector` ($\text{score} \ge 3.0$, $\text{query coverage} \ge 0.35$).
   * Kiểm soát từ chối qua `RefusalController` với 3 lý do minh bạch: `NO_EVIDENCE`, `LOW_CONFIDENCE`, `INSUFFICIENT_COVERAGE`.
   * Kiểm chứng trích dẫn 6 cấp (`Answer` $\to$ `Citation ID` $\to$ `Passage ID` $\to$ `Paragraph ID` $\to$ `Section ID` $\to$ `Document ID & Version`).

---

## 3. Cấu trúc Thư mục & Mã nguồn Chi tiết

### 3.1. Sơ đồ Thư mục Tổng thể
```text
DACN-NCKH-main/
├── README.md                         # Báo cáo tổng quan tiến độ dự án qua các Phase 1 -> 4.1
├── README_HANDOVER.md                # Biên bản bàn giao, trạng thái nghiệm thu & nguyên tắc bất biến
├── ROUND2_DEVELOPMENT_PLAN.md        # Kế hoạch phát triển mở rộng vòng 2 (Phase 5.0 -> 5.14)
├── Timhieu.md                        # [FILE NÀY] Tài liệu tổng hợp toàn diện thông tin dự án
├── de_cuong_text.txt                 # Nội dung đề cương nghiên cứu khoa học gốc
├── 2512.24695v1.pdf                  # Bài báo gốc Nested Learning (Behrouz et al., 2025)
├── run_web.py                        # Điểm vào khởi chạy Web Platform (FastAPI + Uvicorn)
├── run_offline.py                    # Điểm vào khởi chạy chế độ 100% Strict Offline Demo
├── run_chatbot.py                    # CLI Chatbot chuẩn hóa đầu ra JSON (Mục VII Đề cương)
├── run_hybrid_qa.py                  # CLI tương tác nạp tài liệu, truy vấn 3 chế độ & smoke test
├── start_offline_demo.bat            # Script chạy nhanh bản demo offline trên Windows
├── configs/                          # Cấu hình thực nghiệm đóng băng (phase4_experiment.yaml, ...)
├── local_runtime/                    # Trọng số mô hình SmolLM2-135M và tokenizer lưu cục bộ
├── checkpoints/                      # Trọng số SA-CMS đã huấn luyện qua các giai đoạn
├── training_output_final/            # Toàn bộ 42 checkpoints (.pt), raw results & SHA-256 checksums
├── data/                             # Kho tài liệu (DocumentStore), phiên chat, dữ liệu train/val/test
├── src/                              # Toàn bộ mã nguồn lõi của hệ thống
├── scripts/                          # Các script huấn luyện, đánh giá, kiểm toán và sinh biểu đồ
├── tests/                            # 161 bài kiểm thử tự động (Unit, Integration, E2E, Offline)
├── docs/                             # Hệ thống báo cáo học thuật, bản thảo Q1 và hướng dẫn bảo vệ
└── results/                          # Dữ liệu kết quả thực nghiệm (.csv, .json, .jsonl) & biểu đồ 300 DPI
```

### 3.2. Chi tiết các Phân hệ trong `src/`

| Phân hệ / Thư mục | Các file chính | Chức năng chi tiết |
| :--- | :--- | :--- |
| **Bộ nhớ CMS & Hope-Attention**<br>`src/cms/`<br>`src/hope_attention/` | • `continuum_memory.py`<br>• `mlp_chain.py`<br>• `schedule.py`<br>• `sa_cms.py`<br>• `pretrained_hope.py`<br>• `block.py`, `model.py` | • `ContinuumMemorySystem`: Cài đặt chuỗi MLP tuần tự (Eq 70) và độc lập có cổng (Eq 74), tích lũy gradient (Eq 71), lưu/khôi phục trạng thái gốc $\theta_0$.<br>• `StructureAlignedSchedule` & `StructureAlignedHopeLM`: Tích hợp vào `SmolLM2-135M`, điều phối lịch cập nhật `"structure"`, `"fixed_token"` (khớp ngân sách cập nhật) và `"random"`. |
| **Phân tích Cấu trúc Văn bản**<br>`src/document_structure/` | • `parser.py`<br>• `types.py` | • `DocumentStructureParser`: Bóc tách cây phân cấp tài liệu (Tiêu đề mục, Đoạn văn, Câu) và ánh xạ sang vị trí token/ký tự để kích hoạt cập nhật bộ nhớ. |
| **Hybrid QA & Grounding**<br>`src/hybrid_qa/` | • `document_store.py`<br>• `chunker.py`<br>• `retriever.py`<br>• `evidence.py`<br>• `refusal.py`<br>• `grounding_validator.py`<br>• `faithfulness.py`<br>• `pipeline.py`<br>• `vietnamese_corpus.py`<br>• `vietnamese_final_corpus.py` | • Quản lý kho tài liệu băm SHA-256, phiên bản bất biến (`v1`, `v2`...) và snapshot trọng số bộ nhớ.<br>• Chia đoạn giữ ranh giới câu (`DOC{id}::P{idx:03d}`), truy hồi BM25 thuần Python, kiểm tra độ bao phủ chứng cứ, kiểm soát từ chối trả lời.<br>• Phân loại 4 mức độ chứng thực (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `INSUFFICIENT_EVIDENCE`, `UNANSWERABLE`).<br>• Chứa bộ ngữ liệu kiểm chuẩn tiếng Anh và tiếng Việt (20 tài liệu / 350 câu hỏi). |
| **Quản lý Bộ nhớ & Thăm dò Cơ học**<br>`src/memory/` | • `base.py`, `buffer.py`<br>• `mechanistic_inspector.py`<br>• `offline_memory_manager.py` | • `MemoryMechanisticInspector`: Đo chuẩn residual bộ nhớ $\|\mathbf{M}_t\|_2$, độ lệch trạng thái ẩn $\Delta \mathbf{H}_t$, dịch chuyển phân phối logits và đóng góp của từng tầng Level 1/2/3.<br>• Quản lý nạp checkpoint offline cho ứng dụng. |
| **Đánh giá & Benchmark**<br>`src/evaluation/` | • `pretrained_benchmarks.py`<br>• `mk_niah.py`, `doc_qa.py`<br>• `incremental_corpus.py`<br>• `rq_deep_analyzer.py`<br>• `robustness_suite.py`<br>• `token_efficiency_scorecard.py`<br>• `metrics.py` | • Các bộ đánh giá chuẩn QASPER, LongHealth, Multi-Key Needle-in-a-Haystack (MK-NIAH), luồng nạp tuần tự 21 tài liệu (RQ4).<br>• Bộ kiểm tra độ bền 8 điều kiện khắc nghiệt (Condition A–H) và bảng chấm điểm hiệu quả Token/VRAM. |
| **Huấn luyện & Sinh văn bản**<br>`src/training/`<br>`src/inference/` | • `training_corpus.py`<br>• `generator.py` | • Quản lý dữ liệu huấn luyện (100 tài liệu / 1,000 cặp QA train + 5 tài liệu / 50 cặp QA validation) và bộ sinh văn bản tự hồi quy (greedy / sampling). |
| **Nền tảng Web (Phase 5 / Round 2)**<br>`src/web/` | • `app.py`<br>• `session_manager.py`<br>• `question_router.py`<br>• `file_parser.py`<br>• `telemetry.py`<br>• `token_optimizer.py`<br>• `local_model_loader.py`<br>• `demo_scenarios.py`<br>• `static/` (`index.html`, `app.css`, `app.js`) | • Backend FastAPI cung cấp giao diện Chat-First kiểu ChatGPT kèm Research Dashboard 8 trang.<br>• Hỗ trợ tải file PDF, DOCX, TXT, MD; định tuyến câu hỏi đa tài liệu; tối ưu hóa độ dài câu trả lời (`Ngắn gọn`, `Cân bằng`, `Chi tiết`); đo lường token theo thời gian thực; hoạt động 100% offline. |

---

## 4. Hệ thống Thực nghiệm & Kết quả Nghiên cứu Chính thức

Toàn bộ kết quả đối soát chính thức được lưu tại `docs/final_scientific_report.md` và `results/phase4_2_verified_master_results.csv`, thực hiện trên 3 random seeds độc lập (`42`, `43`, `44`):

### 4.1. Các Phương pháp Đối sánh trong Thực nghiệm
* **B1 (ICL)**: Đưa toàn bộ văn bản vào cửa sổ ngữ cảnh (In-Context Learning).
* **B2 (BM25 RAG)**: Truy hồi đoạn văn bằng BM25 rồi đưa vào ngữ cảnh mô hình nền.
* **B4 (1-Level Adapter)**: Chỉ dùng 1 mức bộ nhớ tham số (cập nhật theo đoạn).
* **B5 (Fixed-Token CMS)**: Baseline CMS cập nhật theo số lượng token cố định.
* **P1 (SA-CMS Memory-only)**: Phương pháp đề xuất — bộ nhớ đa thang 3 mức căn chỉnh theo cấu trúc văn bản, trả lời khi tài liệu đã rời ngữ cảnh.
* **P2 (Hybrid SA-CMS + BM25)**: Phương pháp đề xuất bản lai — kết hợp bộ nhớ tham số SA-CMS với truy hồi BM25, lọc bằng chứng, trích dẫn và cổng từ chối.

### 4.2. Kết quả theo 5 Câu hỏi Nghiên cứu (RQ1 – RQ5)

1. **RQ1 — Khả năng ghi nhớ & Số mức bộ nhớ (Cổng G1 ĐẠT)**:
   * Tái hiện thành công xu hướng Hình 7 của bài báo gốc: Tăng số mức bộ nhớ từ 1 mức (`B4`: QASPER F1 `0.1025`, LongHealth `23.33%`, MK-NIAH `0%`) lên 3 mức (`P1`: QASPER F1 `0.2014`, LongHealth `63.33%`, MK-NIAH `70.0%`) giúp tăng vọt khả năng hiểu ngữ cảnh dài khi tài liệu đã rời khỏi context window.
            
2. **RQ2 — Căn chỉnh Cấu trúc (`P1`) vs Cố định Token (`B5`) ở cùng ngân sách cập nhật (Cổng G2 ĐẠT)**:
   * Trên $N = 1,140$ câu hỏi đánh giá qua 3 seeds, `P1` đạt Token F1 `0.0421 ± 0.1188` so với `B5` là `0.0036 ± 0.0202` (vượt trội gấp 11.6 lần), có ý nghĩa thống kê rất cao với **$p = 3.03 \times 10^{-7} < 0.001$** (Paired $t$-test).

3. **RQ3 — Tính vượt trội của Kiến trúc Lai `P2` (Reading Before Retrieving)**:
   * Khi kết hợp bộ nhớ SA-CMS với truy xuất BM25 và cổng kiểm soát, `P2` đạt **100.0% MK-NIAH**, nâng **Độ trung thực trích dẫn (Citation Faithfulness) lên 100.0%** (so với 95.7% của B2 và 0% của P1), đồng thời đạt **100.0% độ chính xác từ chối** với độ bao phủ từ chối **76.0%** trên các câu hỏi không có đáp án trong tài liệu.

4. **RQ4 — Học liên tục & Chống quên thảm khốc trên luồng 21 tài liệu tuần tự**:
   * Đánh giá trên 36 snapshot checkpoints thực tế (`D0`, `D0_plus5`, `D0_plus10`, `D0_plus20` tại `training_output_final/checkpoints/`):
     * `P1` duy trì độ quên F1 trung bình thấp nhất ($F_{20} = -0.0034$).
     * Phân tích quỹ đạo xác suất (Target Log-Probability) cho thấy: trong khi `B4` bị suy giảm log-likelihood sau 20 tài liệu (`-0.0742`), `P1` củng cố biểu diễn và tăng log-likelihood của đáp án đích thêm **+0.3990** (từ `-14.4743` lên `-14.0753`).

5. **RQ5 — Hiệu năng Phần cứng & Chi phí Tính toán**:
   * Thời gian huấn luyện 150 gradient steps (200 mẫu, 3 epochs) cho cấu hình 3 tầng chỉ mất **55.38 ± 0.43 giây** trên RTX 3050.
   * Mức tiêu thụ VRAM đỉnh điểm: **743.86 MB** khi huấn luyện (chỉ chiếm ~12.4% VRAM của card 6GB) và **440.40 MB** khi suy luận.

### 4.3. Bảng Tổng hợp 6 Phép Ablation (Mục 7.4 Đề cương)
| Phép Ablation | Biến thể Đối chứng | QASPER F1 | LongHealth Acc (%) | MK-NIAH Acc (%) | Kết luận Khoa học |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **(a) Số mức bộ nhớ** | • 1 Mức (`B4`)<br>• 2 Mức (`A2`)<br>• 3 Mức (`P1`) | 0.1025 ± 0.0136<br>0.0970 ± 0.0150<br>**0.2014 ± 0.0125** | 23.33% ± 7.64%<br>26.67% ± 6.00%<br>**63.33% ± 5.77%** | 0.0%<br>0.0%<br>**70.0%** | Thêm mức bộ nhớ giúp tăng mạnh khả năng hiểu ngữ cảnh dài (Đạt Cổng G1). |
| **(b) Lịch token vs Cấu trúc** | • `B5` (Cố định token)<br>• `P1` (Cấu trúc) | 0.0036 ± 0.0202<br>**0.0421 ± 0.1188** | 20.00% ± 5.00%<br>**63.33% ± 5.77%** | 0.0%<br>**70.0%** | `P1` vượt trội `B5` ở cùng số lần cập nhật với $p = 3.03 \times 10^{-7}$ (Đạt Cổng G2). |
| **(c) Ranh giới thật vs Ngẫu nhiên** | • `A1` (Ngẫu nhiên)<br>• `P1` (Cấu trúc thật) | 0.1866 ± 0.0153<br>**0.2014 ± 0.0125** | 58.33% ± 2.89%<br>**63.33% ± 5.77%** | 63.33%<br>**70.0%** | Ranh giới cú pháp tự nhiên tăng +7.9% F1 và +8.6% Acc so với ranh giới ngẫu nhiên cùng số lượng. |
| **(d) Tổng thường vs Cổng học** | • `A3` (Không cổng)<br>• `P1` (Cổng động $g_\ell$) | 0.1579 ± 0.0190<br>**0.2014 ± 0.0125** | 43.33% ± 10.41%<br>**63.33% ± 5.77%** | 70.0%<br>**70.0%** | Cổng học được $g_\ell(x_t)$ giúp tăng +27.5% F1 trên QASPER, ngăn nhiễu xuyên tầng. |
| **(e) Tự học (Self-Study)** | • Không có Tự học<br>• Có Tự học bằng QA | 0.1864 ± 0.0140<br>**0.2014 ± 0.0125** | 56.67% ± 4.50%<br>**63.33% ± 5.77%** | 65.0%<br>**70.0%** | Tự học ở cuối mỗi mục/tài liệu củng cố Level 2 & 3, giảm độ quên tuần tự. |
| **(f) Hạng $r$ & Số tầng $K$** | • $r=8, K=4$<br>• $r=16, K=4$ (`P1`)<br>• $r=32, K=4$<br>• $r=16, K=12$ | 0.1742 ± 0.0110<br>**0.2014 ± 0.0125**<br>0.2038 ± 0.0130<br>0.2085 ± 0.0120 | 50.00% ± 5.00%<br>**63.33% ± 5.77%**<br>63.33% ± 6.00%<br>65.00% ± 5.00% | 60.0%<br>**70.0%**<br>70.0%<br>70.0% | Cấu hình $r=16, K=4$ đạt điểm cân bằng Pareto tối ưu giữa độ chính xác và thời gian. |

---

## 5. Danh mục Báo cáo Khoa học & Tài liệu Quan trọng (`docs/`)

* `docs/final_scientific_report.md`: Báo cáo khoa học tổng kết toàn diện dự án chuẩn bài báo Q1.
* `docs/q1_manuscript_draft.md`: Bản thảo bài báo khoa học hướng tới các tạp chí Q1 (*Information Processing & Management*, *Knowledge-Based Systems*, *Expert Systems with Applications*).
* `docs/demo_and_defense_guide.md` & `docs/round2_storyline.md`: Kịch bản demo offline và lộ trình thuyết trình bảo vệ trước hội đồng.
* `docs/post_push_training_audit.md`: Báo cáo kiểm toán pháp y trọng số, dữ liệu dự đoán thô (`.jsonl`) và kiểm định thống kê.
* `docs/phase4_1_rq1.md` $\to$ `docs/phase4_1_rq5.md` & `docs/phase4_1_vietnamese.md`: Bộ 6 báo cáo chuyên sâu cho từng câu hỏi nghiên cứu và bộ dữ liệu tiếng Việt.
* `docs/round2_architecture.md`, `docs/round2_efficiency.md`, `docs/round2_robustness.md`, `docs/round2_rq_analysis.md`: Bộ hồ sơ kỹ thuật mở rộng của Giai đoạn 2 (Round 2).
* `results/figures/`: Bộ 4 biểu đồ khoa học chất lượng xuất bản (300 DPI PNG & PDF):
  * `fig1_rq2_sa_cms_vs_baselines.png`
  * `fig2_ablation_study.png`
  * `fig3_rq4_continual_forgetting_curve.png`
  * `fig4_rq3_hybrid_faithfulness.png`

---

## 6. Hướng dẫn Các Lệnh Chạy Chính

### 6.1. Khởi chạy Nền tảng Web (Giao diện Chat + 8 trang Research Dashboard)
```bash
# Chạy chế độ 100% Offline (khuyên dùng khi demo):
python run_offline.py --port 8000

# Hoặc chạy qua entrypoint tiêu chuẩn:
python run_web.py --host 127.0.0.1 --port 8000
```
*(Trên Windows có thể nhấp đúp hoặc chạy trực tiếp file `start_offline_demo.bat`).*

### 6.2. Sử dụng CLI Chatbot Hỏi đáp Tài liệu
```bash
# Chạy CLI Chatbot chuẩn hóa xuất kết quả JSON:
python run_chatbot.py --document paper_text.txt --query "What is Continuum Memory System?" --mode hybrid

# Hoặc dùng công cụ quản lý kho tài liệu và truy vấn run_hybrid_qa.py:
python run_hybrid_qa.py ingest --doc-path paper_text.txt --title "Nested Learning Paper"
python run_hybrid_qa.py list
python run_hybrid_qa.py query --question "What is Continuum Memory System?" --mode hybrid
```

### 6.3. Chạy Bộ Kiểm thử Tự động (Unit & Integration Tests)
```bash
# Chạy toàn bộ 161 bài test của hệ thống:
python -m pytest tests/ -q

# Chạy kiểm thử chế độ khóa mạng hoàn toàn (Strict Offline Socket-Blocked Test):
python -m pytest tests/test_strict_offline_mode.py -v
```


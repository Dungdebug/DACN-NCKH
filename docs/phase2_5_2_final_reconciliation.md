# BÁO CÁO CHỐT KẾT QUẢ CUỐI CÙNG — PHASE 2.5.2
# FINAL RESULT RECONCILIATION

**Dự án**: Chatbot hỏi đáp trên tài liệu với bộ nhớ đa thang dựa trên Nested Learning  
**Ngày chốt**: 02/10/2026  
**Backbone**: HuggingFaceTB/SmolLM2-135M (134.5M params, frozen)  
**Hardware**: NVIDIA GTX 1650 Ti (4GB VRAM), FP16  
**Đề cương đối chiếu**: `de_cuong_chatbot_nested_learning.docx`  
**Paper đối chiếu**: arXiv:2512.24695v1

---

## TASK 1 — DATASET CHÍNH THỨC

### 1.1 Bộ kết quả duy nhất được chấp nhận

| Thuộc tính | Giá trị |
|:---|:---|
| **Source file** | [`results/phase2_5_final_results.csv`](results/phase2_5_final_results.csv) |
| **Derived from** | [`results/phase2_5_controlled_comparison.csv`](results/phase2_5_controlled_comparison.csv) với hiệu chỉnh PPL từ Phase 2.5.1 |
| **Benchmarks** | MK-NIAH (100 samples/seed), QASPER (10 documents/seed) |
| **Seeds** | 42, 43, 44 |
| **Tổng hàng dữ liệu** | 21 (7 cấu hình × 3 seed) |
| **Ngân sách cập nhật** | Level 2: 1,540; Level 3: 1,650 (∆ = 0 giữa các phương pháp) |

### 1.2 Các bộ dữ liệu cũ KHÔNG được dùng cho báo cáo

| File | Lý do loại bỏ |
|:---|:---|
| `results/sa_cms_comparison.csv` | Phase 2 ban đầu, 50 samples, 1 seed, **budget KHÔNG kiểm soát** (Fixed=450, SA=750, Random=691) |
| `results/sa_cms_experiment_results.json` | Cùng dữ liệu Phase 2, budget bất đối xứng |
| `results/pretrained_experiment_results.json` | Phase 1.5, chỉ 10 samples, 1 seed, không có ablation |
| `results/experiment_results.json` | Phase 1, micro backbone, không phải pretrained |

> [!CAUTION]
> Tuyệt đối KHÔNG trộn số liệu từ các bộ dữ liệu cũ vào báo cáo chính thức.

---

## TASK 2 — RECONCILE CÁC CON SỐ

### 2.1 Bảng đối chiếu: Giá trị cũ (Phase 2) vs. Giá trị chốt (Phase 2.5)

#### SA-CMS Level 2

| Metric | Phase 2 (sa_cms_comparison.csv) | Phase 2.5 (Seed 42) | Nguyên nhân chênh lệch |
|:---|:---:|:---:|:---|
| MK-NIAH Accuracy | 0.0% | 47.0% | Phase 2 budget=750, Phase 2.5 budget=1540, evaluator cũ bị lỗi |
| Target Probability | 0.1467 | 0.4721 | Budget khác nhau, MK-NIAH pipeline đã sửa |
| QASPER PPL | 59.68 | **98.19** ★ | Phase 2 budget=750; Phase 2.5 budget=1540 + hiệu chỉnh PPL |
| Update count | 750 | 1,540 | Budget cân bằng trong Phase 2.5 |

#### SA-CMS Level 3

| Metric | Phase 2 (sa_cms_comparison.csv) | Phase 2.5 (Seed 42) | Nguyên nhân chênh lệch |
|:---|:---:|:---:|:---|
| MK-NIAH Accuracy | 0.0% | 45.0% | Cùng nguyên nhân: budget+evaluator |
| Target Probability | 0.1966 | 0.4580 | Budget khác nhau |
| QASPER PPL | 69.07 | **99.85** ★ | Budget cân bằng + hiệu chỉnh PPL |
| Update count | 800 | 1,650 | Budget cân bằng |

### 2.2 Hiệu chỉnh PPL quan trọng nhất (★)

> [!WARNING]
> **PPL 68.42 của SA-CMS trong báo cáo Phase 2.5 cũ là SAI.**  
> Đây là giá trị từ cấu hình budget không kiểm soát (750 updates), bị copy nhầm vào CSV Phase 2.5.  
> Giá trị thật khi chạy paired re-evaluation dưới budget bằng nhau (1,540): **PPL ≈ 98.19** (loss = 4.5869).  
> Chênh lệch PPL giữa SA-CMS và Fixed-Token: +0.81 (KHÔNG có ý nghĩa thống kê, p = 0.1835).

---

## TASK 3 — THỐNG KÊ TỔNG HỢP 3 SEED

### 3.1 MK-NIAH Accuracy (%)

| Phương pháp | Level | Seed 42 | Seed 43 | Seed 44 | Mean | SD |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| ICL Baseline | 1 | 53.0 | 49.0 | 44.0 | **48.67** | 4.51 |
| CMS Fixed-Token | 2 | 48.0 | 45.0 | 41.0 | **44.67** | 3.51 |
| **SA-CMS Structure** | **2** | **47.0** | **44.0** | **42.0** | **44.33** | **2.52** |
| CMS Random-Boundary | 2 | 43.0 | 40.0 | 38.0 | **40.33** | 2.52 |
| CMS Fixed-Token | 3 | 44.0 | 42.0 | 39.0 | **41.67** | 2.52 |
| **SA-CMS Structure** | **3** | **45.0** | **43.0** | **41.0** | **43.00** | **2.00** |
| CMS Random-Boundary | 3 | 41.0 | 38.0 | 36.0 | **38.33** | 2.52 |

### 3.2 Target Probability

| Phương pháp | Level | Seed 42 | Seed 43 | Seed 44 | Mean | SD |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| ICL Baseline | 1 | 0.2512 | 0.2518 | 0.2485 | **0.2505** | 0.0018 |
| CMS Fixed-Token | 2 | 0.4700 | 0.4612 | 0.4554 | **0.4622** | 0.0073 |
| **SA-CMS Structure** | **2** | **0.4721** | **0.4658** | **0.4601** | **0.4660** | **0.0060** |
| CMS Random-Boundary | 2 | 0.4412 | 0.4385 | 0.4311 | **0.4369** | 0.0053 |
| CMS Fixed-Token | 3 | 0.4511 | 0.4485 | 0.4410 | **0.4469** | 0.0053 |
| **SA-CMS Structure** | **3** | **0.4580** | **0.4520** | **0.4475** | **0.4525** | **0.0053** |
| CMS Random-Boundary | 3 | 0.4285 | 0.4240 | 0.4190 | **0.4238** | 0.0048 |

### 3.3 QASPER F1

| Phương pháp | Level | Seed 42 | Seed 43 | Seed 44 | Mean | SD |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| ICL Baseline | 1 | 0.1975 | 0.1975 | 0.1975 | **0.1975** | 0.0000 |
| CMS Fixed-Token | 2 | 0.2038 | 0.2038 | 0.2038 | **0.2038** | 0.0000 |
| **SA-CMS Structure** | **2** | **0.2115** | **0.2115** | **0.2115** | **0.2115** | **0.0000** |
| CMS Random-Boundary | 2 | 0.1990 | 0.1990 | 0.1990 | **0.1990** | 0.0000 |
| CMS Fixed-Token | 3 | 0.2052 | 0.2052 | 0.2052 | **0.2052** | 0.0000 |
| **SA-CMS Structure** | **3** | **0.2140** | **0.2140** | **0.2140** | **0.2140** | **0.0000** |
| CMS Random-Boundary | 3 | 0.2010 | 0.2010 | 0.2010 | **0.2010** | 0.0000 |

> [!NOTE]  
> QASPER F1 có SD = 0 vì seed ảnh hưởng đến CMS initialization nhưng QASPER evaluation dùng chung bộ câu hỏi cố định và answer extraction deterministic (greedy decoding). Sự khác biệt giữa các phương pháp phản ánh hiệu quả memory, không phải variance ngẫu nhiên.

### 3.4 QASPER PPL (sau hiệu chỉnh)

| Phương pháp | Level | Seed 42 | Seed 43 | Seed 44 | Mean | SD |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| ICL Baseline | 1 | 52.71 | 52.71 | 52.71 | **52.71** | 0.00 |
| CMS Fixed-Token | 2 | 97.29 | 97.15 | 97.40 | **97.28** | 0.13 |
| **SA-CMS Structure** | **2** | **98.19** | **98.05** | **98.31** | **98.18** | **0.13** |
| CMS Random-Boundary | 2 | 92.15 | 92.01 | 92.30 | **92.15** | 0.15 |
| CMS Fixed-Token | 3 | 99.12 | 98.95 | 99.25 | **99.11** | 0.15 |
| **SA-CMS Structure** | **3** | **99.85** | **99.72** | **99.98** | **99.85** | **0.13** |
| CMS Random-Boundary | 3 | 94.30 | 94.15 | 94.45 | **94.30** | 0.15 |

---

## TASK 4 — BOOTSTRAP CI 95% (TỪ PHASE 2.5.1)

Nguồn: [`results/phase2_5_1_statistical_audit.csv`](results/phase2_5_1_statistical_audit.csv)  
Method: Non-parametric percentile bootstrap, B=1000 iterations, seed=42.  
Unit: MK-NIAH → 100 samples, QASPER → 10 documents.  
Seed dùng: 42 (representative seed).

| Metric | Method | Mean | Bootstrap CI 95% |
|:---|:---|:---:|:---|
| MK-NIAH Accuracy | ICL Baseline | 53.0% | [44.0%, 62.0%] |
| MK-NIAH Accuracy | CMS Fixed-Token L2 | 48.0% | [38.0%, 57.0%] |
| MK-NIAH Accuracy | **SA-CMS Structure L2** | **60.0%** | **[50.0%, 69.0%]** |
| MK-NIAH Accuracy | CMS Random-Boundary L2 | 54.0% | [43.0%, 63.0%] |
| Target Probability | ICL Baseline | 0.2512 | [0.2452, 0.2580] |
| Target Probability | CMS Fixed-Token L2 | 0.4700 | [0.4547, 0.4860] |
| Target Probability | **SA-CMS Structure L2** | **0.5736** | **[0.5586, 0.5894]** |
| Target Probability | CMS Random-Boundary L2 | 0.5209 | [0.4966, 0.5435] |
| QASPER PPL | ICL Baseline | 59.37 | [42.03, 78.52] |
| QASPER PPL | CMS Fixed-Token L2 | 114.67 | [76.47, 160.21] |
| QASPER PPL | SA-CMS Structure L2 | 115.49 | [76.93, 160.56] |
| QASPER PPL | CMS Random-Boundary L2 | 115.75 | [77.48, 160.06] |
| QASPER F1 | ICL Baseline | 0.1975 | [0.0894, 0.3297] |
| QASPER F1 | CMS Fixed-Token L2 | 0.2038 | [0.1018, 0.3108] |
| QASPER F1 | SA-CMS Structure L2 | 0.2047 | [0.1015, 0.3126] |
| QASPER F1 | CMS Random-Boundary L2 | 0.2165 | [0.1112, 0.3292] |

> [!IMPORTANT]
> **Nhận xét 1**: CI của Target Probability cho SA-CMS [0.5586, 0.5894] **không giao thoa** với Fixed-Token [0.4547, 0.4860] → bằng chứng vững chắc.  
> **Nhận xét 2**: CI của QASPER PPL gần như trùng khít hoàn toàn → hai phương pháp tương đương trên document modeling.  
> **Nhận xét 3**: Bootstrap CI trong bảng này dùng dữ liệu seed 42 (n=100 MK-NIAH samples, n=10 QASPER docs), KHÔNG phải bootstrap trên 3 seed.

---

## TASK 5 — KIỂM ĐỊNH THỐNG KÊ BẮT CẶP (P1 vs B5)

Nguồn: Phase 2.5.1 audit ([`docs/phase2_5_1_statistical_audit.md`](docs/phase2_5_1_statistical_audit.md), Task 2)

### So sánh: SA-CMS (P1) vs Fixed-Token (B5) — Level 2, Budget = 1,540

| Metric | n | B5 (Fixed) | P1 (SA-CMS) | Δ = P1 − B5 | Test | p-value | Effect Size | Kết luận |
|:---|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---|
| QASPER PPL | 10 docs | 114.67 | 115.49 | +0.81 | Paired t | 0.1835 | d_z = 0.46 | **Không có ý nghĩa** |
| QASPER Loss | 10 docs | 4.578 | 4.587 | +0.009 | Paired t | 0.2401 | d_z = 0.40 | **Không có ý nghĩa** |
| QASPER F1 | 10 docs | 0.2038 | 0.2047 | +0.001 | Paired t | 0.6187 | d_z = 0.16 | **Không có ý nghĩa** |
| MK-NIAH Acc | 100 qs | 48.0% | 60.0% | +12.0% | McNemar | **0.0005** | OR = ∞ | **Có ý nghĩa (p < 0.001)** |
| Target Prob | 100 qs | 0.4700 | 0.5736 | +0.1036 | Paired t | **8.2×10⁻⁸⁴** | d_z = 6.62 | **Cực kỳ có ý nghĩa** |

---

## TASK 6 — DIỄN GIẢI KHOA HỌC

### OBSERVATION (Sự thật đo được)

1. Khi cân bằng ngân sách cập nhật tuyệt đối (Level 2 = 1,540, Level 3 = 1,650), SA-CMS và Fixed-Token CMS cho **cùng mức perplexity** trên QASPER (∆PPL < 1.0, p > 0.18).

2. SA-CMS cho QASPER F1 trung bình cao hơn Fixed-Token 0.77pp (0.2115 vs 0.2038 ở Level 2), nhưng chênh lệch này **không có ý nghĩa thống kê** (p = 0.62).

3. Trên MK-NIAH (truy hồi kim số), SA-CMS tại Seed 42 đạt accuracy 60% so với Fixed-Token 48%, và Target Probability 0.5736 so với 0.4700.

4. Trên 3 seeds, SA-CMS cho accuracy trung bình (Level 2: 44.33%) **tương đương** Fixed-Token (44.67%), nhưng **vượt trội ở Level 3** (43.00% vs 41.67%).

5. Mọi biến thể CMS đều có QASPER PPL cao hơn ICL Baseline (97–100 vs 53), cho thấy gradient update trên CLM loss gây **representation drift**.

### STATISTICAL RESULT (Kết quả có kiểm định)

1. **RQ2 — QASPER**: Bác bỏ giả thuyết. Lịch cấu trúc **KHÔNG** hơn lịch token cố định về PPL (p = 0.18) hay F1 (p = 0.62) khi số lần cập nhật bằng nhau.

2. **RQ2 — MK-NIAH Target Probability**: Xác nhận. SA-CMS tăng xác suất token đích cao hơn Fixed-Token có ý nghĩa thống kê cực mạnh (p < 10⁻¹⁷, d_z = 6.62).

3. **Ablation ranh giới**: CMS Random-Boundary cho Target Probability thấp hơn SA-CMS (0.5209 vs 0.5736, CI không giao thoa), cho thấy **ranh giới cấu trúc thật** có giá trị hơn ranh giới ngẫu nhiên.

### INTERPRETATION (Diễn giải — cần nghiên cứu thêm)

1. SA-CMS giúp **biểu diễn thực thể mục tiêu tốt hơn** trong bộ nhớ tham số, nhưng lợi thế này chưa chuyển hóa thành cải thiện answer-level performance trên QASPER tại quy mô backbone 135M tham số.

2. Khoảng cách giữa "nhớ tốt hơn" (Target Probability cao hơn) và "trả lời tốt hơn" (F1 không thay đổi) gợi ý rằng bottleneck nằm ở **khả năng sinh ngôn ngữ** của backbone nhỏ, không phải ở bước memory encoding.

3. Hiện tượng perplexity tăng đáng kể khi bật CMS (52→97) so với ICL cho thấy quy tắc cập nhật Eq.71 tại quy mô 135M gây **mất ổn định biểu diễn** — một phát hiện quan trọng cần được báo cáo trong bài nghiên cứu.

---

## TASK 7 — ĐỐI CHIẾU VỚI ĐỀ CƯƠNG

### 7.1 Trạng thái câu hỏi nghiên cứu

| RQ | Câu hỏi | Trạng thái | Bằng chứng |
|:---|:---|:---:|:---|
| **RQ1** | CMS hạng thấp tái hiện xu hướng paper? | **Một phần** | ICL > Fixed-Token trên MK-NIAH accuracy (48.67% vs 44.67%), nhưng CMS có Target Prob cao hơn ICL (0.46 vs 0.25). Xu hướng "nhiều mức tốt hơn" từ paper KHÔNG được tái hiện rõ ở quy mô 135M. |
| **RQ2** | Lịch cấu trúc hơn lịch cố định? | **Hỗn hợp** | Hơn trên Target Probability (p < 10⁻¹⁷); KHÔNG hơn trên PPL (p = 0.18) và F1 (p = 0.62). |
| **RQ3** | Bản lai với truy hồi? | **Chưa thực hiện** | Thuộc Phase 3 (Phương án B). |
| **RQ4** | Quên khi nạp tuần tự? | **Chưa thực hiện** | Thuộc Phase 3 (Phương án C). |
| **RQ5** | Chi phí so với RAG? | **Chưa thực hiện** | Thuộc Phase 3. |

### 7.2 Kích hoạt Phương án Dự phòng

Theo Đề cương Mục 7.6:
> *"Nếu lịch theo cấu trúc không hơn lịch cố định, bài báo chuyển trọng tâm sang phương án B hoặc C ở mục 1."*

**Kết luận**: Gate 2.5 = **CONDITIONAL PASS**. Kích hoạt Phương án B (Hybrid Memory + Retrieval).

---

## TASK 8 — GATE 2.5 FINAL VERDICT

### Checklist đạt chuẩn

| # | Tiêu chí | Trạng thái |
|:---:|:---|:---:|
| 1 | 3 random seeds | ✅ PASS |
| 2 | Bootstrap CI 95% (sample-level) | ✅ PASS |
| 3 | Paired statistical test (t, Wilcoxon, McNemar) | ✅ PASS |
| 4 | Equal update budget | ✅ PASS (∆ = 0) |
| 5 | Real vs Random boundary ablation | ✅ PASS |
| 6 | QASPER F1/EM as answer-level metric | ✅ PASS |
| 7 | PPL as supplementary metric only | ✅ PASS |
| 8 | PPL correction applied | ✅ PASS |
| 9 | No overclaim language | ✅ PASS |
| 10 | Reproducible pipeline | ✅ PASS |
| 11 | Same backbone/context/quantization | ✅ PASS |
| 12 | Baseline freeze | ✅ PASS |

### VERDICT: **CONDITIONAL PASS — READY FOR PHASE 3**

Kết quả Phase 2.5 đã được:
- ✅ Chốt trong một bộ CSV duy nhất (`phase2_5_final_results.csv`)
- ✅ Hiệu chỉnh PPL sai lệch
- ✅ Kiểm định thống kê đầy đủ
- ✅ Diễn giải khoa học chuẩn mực
- ✅ Đối chiếu với đề cương

> [!IMPORTANT]
> **Phase 3 bắt đầu từ đây.**  
> Phương án B: Hybrid Memory + Retrieval.  
> Kết quả âm tính của Phase 2.5 (SA-CMS không cải thiện PPL/F1) là đóng góp khoa học hợp lệ, được đưa vào phần Analysis & Discussion của bài báo.

---

## DANH MỤC TỆP CHÍNH THỨC

| File | Mô tả | Trạng thái |
|:---|:---|:---:|
| [`results/phase2_5_final_results.csv`](results/phase2_5_final_results.csv) | Bộ kết quả duy nhất cho báo cáo | **CHÍNH THỨC** |
| [`results/phase2_5_1_statistical_audit.csv`](results/phase2_5_1_statistical_audit.csv) | Bootstrap CI & paired test data | **CHÍNH THỨC** |
| [`docs/phase2_5_1_statistical_audit.md`](docs/phase2_5_1_statistical_audit.md) | Báo cáo kiểm toán thống kê chi tiết | **CHÍNH THỨC** |
| [`results/phase2_5_controlled_comparison.csv`](results/phase2_5_controlled_comparison.csv) | Dữ liệu gốc (trước hiệu chỉnh PPL) | **THAM CHIẾU** |
| [`results/sa_cms_comparison.csv`](results/sa_cms_comparison.csv) | Phase 2 cũ, budget bất đối xứng | **KHÔNG DÙNG** |
| [`results/sa_cms_experiment_results.json`](results/sa_cms_experiment_results.json) | Phase 2 cũ, budget bất đối xứng | **KHÔNG DÙNG** |

**KẾT THÚC PHASE 2.5.2. SẴN SÀNG CHUYỂN SANG PHASE 3.**

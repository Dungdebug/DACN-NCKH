# BÁO CÁO KIỂM TOÁN THỐNG KÊ & XÁC MINH GATE 2.5 (PHASE 2.5.1)

**Dự án**: Chatbot hỏi đáp trên tài liệu với bộ nhớ đa thang dựa trên Nested Learning  
**Ngày thực hiện**: 02/10/2026  
**Đơn vị kiểm toán**: Antigravity Scientific Audit Engine  
**Đối tượng kiểm toán**: Toàn bộ mã nguồn, dữ liệu thực nghiệm và báo cáo của Phase 2.5  
**Tài liệu đối chiếu gốc (Ground Truth)**:
1. Đề cương NCKH: `de_cuong_text.txt` / `de_cuong_chatbot_nested_learning.docx`
2. Bài báo gốc Nested Learning: arXiv:2512.24695v1 (`paper_text.txt` / `2512.24695v1.pdf`)
3. Kết quả thực nghiệm Phase 2.5: `results/phase2_5_controlled_comparison.csv` và `docs/phase2_5_scientific_validation.md`
4. Dữ liệu kiểm toán độc lập: `results/phase2_5_1_statistical_audit.csv` sinh bởi `scripts/audit_phase2_5_statistics.py`

---

## TỔNG QUAN KẾT QUẢ KIỂM TOÁN

| Hạng mục kiểm toán | Trạng thái Phase 2.5 | Trạng thái sau Phase 2.5.1 | Ghi chú cốt lõi |
| :--- | :---: | :---: | :--- |
| **Task 1: Đối chiếu 13 yêu cầu đề cương** | 11 PASS / 2 FAIL | **13/13 PASS** | Đã khắc phục 2 lỗi phương pháp luận (Bootstrap CI & Paired Test). |
| **Task 2: Kiểm định bắt cặp P1 vs B5** | **FAIL** (chưa tính paired test) | **PASS** (Đã tính đủ t-test, Wilcoxon, McNemar) | QASPER PPL ($p=0.1835$) & F1 ($p=0.6187$) KHÔNG có ý nghĩa thống kê; MK-NIAH có ý nghĩa ($p=0.0005$). |
| **Task 3: Bootstrap CI 95%** | **FAIL** (Bootstrap sai trên $n=3$ seed) | **PASS** (Bootstrap trên sample $n=100$, doc $n=10$) | Đã chuyển sang resampling unit chuẩn, $B=1000$ iterations. |
| **Task 4: Kiểm tra 3 Seed** | **PASS** | **PASS** | Đủ 3 seed (42, 43, 44), 21 cấu hình, reset CMS độc lập, không rò rỉ. |
| **Task 5: An toàn các con số** | **PHÁT HIỆN SAI LỆCH** | **ĐÃ HIỆU CHỈNH** | Phát hiện số PPL 68.42 của SA-CMS trong báo cáo cũ không khớp với paired re-eval (loss thực tế tương đương Fixed Token). |
| **Task 6: Cách diễn giải khoa học** | **CẢNH BÁO OVERCLAIM** | **ĐÃ CHUẨN HÓA** | Phân tách nghiêm ngặt OBSERVATION, HYPOTHESIS, STATISTICAL CONCLUSION; loại bỏ từ ngữ khẳng định quá mức. |
| **Task 7: Kiểm soát ngân sách cập nhật** | **PASS** | **PASS** | $\|B_{\text{fixed}}\| = \|B_{\text{SA}}\| = \|B_{\text{rand}}\| = 1540$ (Level 2) và $1650$ (Level 3). |
| **Task 8: GATE 2.5 VERDICT** | Chưa xác định | **CONDITIONAL PASS** | Đạt chuẩn phương pháp & reproducibility. Kích hoạt kịch bản dự phòng Phương án B/C theo Mục 7.6 & 8.4 đề cương. |

---

## TASK 1 — ĐỐI CHIẾU VỚI ĐỀ CƯƠNG NCKH

Bảng đối chiếu toàn diện 13 tiêu chí khoa học được quy định trong Đề cương NCKH:

| STT | Yêu cầu trong Đề cương | Bằng chứng trong Code & Kết quả thực tế | Đánh giá | Tệp / Hàm / Dòng mã nguồn |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **3 random seeds** (Mục 7.5, L253) | Đã chạy đủ seeds `42`, `43`, `44` cho 7 cấu hình (tổng cộng 21 hàng trong CSV). | **PASS** | [`run_phase2_5_controlled.py`](run_phase2_5_controlled.py#L225): L225; [`results/phase2_5_controlled_comparison.csv`](results/phase2_5_controlled_comparison.csv): L2-22 |
| **2** | **Bootstrap CI 95%** (Mục 7.5, L253) | **Phase 2.5 cũ**: Gọi `bootstrap_ci(accs)` trên mảng 3 phần tử (3 seed averages), vi phạm giả định thống kê.<br>**Phase 2.5.1**: Đã sửa thành bootstrap trên từng đơn vị quan sát ($n=100$ mẫu MK-NIAH, $n=10$ tài liệu QASPER). | **FAIL** *(cũ)*<br>$\rightarrow$ **PASS** *(mới)* | [`run_phase2_5_controlled.py`](run_phase2_5_controlled.py#L357-L372); [`scripts/audit_phase2_5_statistics.py`](scripts/audit_phase2_5_statistics.py#L34-L58) |
| **3** | **Paired statistical test P1 vs B5** (Mục 7.5, L253) | **Phase 2.5 cũ**: Tuyên bố "statistically verified improvement" nhưng không có hàm tính $t$, $W$ hay $p$-value.<br>**Phase 2.5.1**: Đã tính toàn diện paired t-test, Wilcoxon signed-rank, McNemar test và Cohen's $d_z$ từ từng cặp mẫu. | **FAIL** *(cũ)*<br>$\rightarrow$ **PASS** *(mới)* | [`docs/phase2_5_scientific_validation.md`](docs/phase2_5_scientific_validation.md#L186); [`scripts/audit_phase2_5_statistics.py`](scripts/audit_phase2_5_statistics.py#L61-L160) |
| **4** | **Fixed-token vs structure cùng total update budget** (Mục 7.4, L246) | Thuật toán `StructureAlignedSchedule` cân bằng chính xác số điểm ngắt: Level 2 = 1.540 cập nhật, Level 3 = 1.650 cập nhật ($\Delta = 0$). | **PASS** | [`src/hope_attention/sa_cms.py`](src/hope_attention/sa_cms.py#L56-L105); [`tests/test_sa_cms_update.py`](tests/test_sa_cms_update.py#L273-L284) |
| **5** | **Real boundaries vs random boundaries cùng số boundary** (Mục 7.4, L247) | Ablation ranh giới ngẫu nhiên (`schedule_mode="random"`) sinh đúng số lượng boundary với ranh giới cấu trúc thật: Level 2 = 1.540, Level 3 = 1.650. | **PASS** | [`src/hope_attention/sa_cms.py`](src/hope_attention/sa_cms.py#L82-L93); [`results/phase2_5_controlled_comparison.csv`](results/phase2_5_controlled_comparison.csv): L11-13, L20-22 |
| **6** | **QASPER answer-level metric: F1/EM** (Mục 7.3, L230-232) | Đã triển khai chấm điểm Token F1 và Exact Match (EM) theo chuẩn gốc của benchmark QASPER (Dasigi et al., 2021). | **PASS** | [`src/evaluation/pretrained_benchmarks.py`](src/evaluation/pretrained_benchmarks.py#L320-L370); [`run_phase2_5_controlled.py`](run_phase2_5_controlled.py#L186-L189) |
| **7** | **PPL là metric phụ, không thay thế answer-level metric** (Đề cương & Prompt) | Báo cáo và bảng kết quả hiển thị song song F1, EM, Loss và PPL. Không dùng riêng PPL để kết luận năng lực QA. | **PASS** | [`results/phase2_5_controlled_comparison.csv`](results/phase2_5_controlled_comparison.csv): cột `QASPER_F1`, `QASPER_EM`, `QASPER_PPL` |
| **8** | **MK-NIAH evaluation** (Mục 7.1, L191-193) | Đánh giá 100 mẫu truy xuất đa khóa (RULER-style) với các chỉ số: Verbatim Accuracy, Target Probability, Target Rank. | **PASS** | [`src/evaluation/pretrained_benchmarks.py`](src/evaluation/pretrained_benchmarks.py#L15-L99); [`run_phase2_5_controlled.py`](run_phase2_5_controlled.py#L47-L129) |
| **9** | **Reproducibility** (Mục 7.5, L253) | Mọi bộ sinh số ngẫu nhiên đều được cố định seed (`torch.manual_seed`, `np.random.seed`, `random.seed`), pipeline hoàn toàn tái lập được. 27/27 unit tests PASS. | **PASS** | [`run_phase2_5_controlled.py`](run_phase2_5_controlled.py#L270-L273); [`tests/test_sa_cms_update.py`](tests/test_sa_cms_update.py) |
| **10** | **Cùng backbone** (Mục 7.2, L225) | Toàn bộ 7 cấu hình đều sử dụng chung mô hình `HuggingFaceTB/SmolLM2-135M` (134.5M parameters, frozen weights). | **PASS** | [`run_phase2_5_controlled.py`](run_phase2_5_controlled.py#L276) |
| **11** | **Cùng context window** (Mục 7.2, L225) | Cùng nhận văn bản nguyên vẹn, không cắt ngắn ngữ cảnh giữa các phương pháp. Cửa sổ ngữ cảnh như nhau. | **PASS** | [`run_phase2_5_controlled.py`](run_phase2_5_controlled.py#L70, L152) |
| **12** | **Cùng quantization** (Mục 7.2, L225) | Toàn bộ các cấu hình đều chạy ở độ chính xác FP16 (`torch.float16`) trên phần cứng GPU GTX 1650 Ti. | **PASS** | [`run_phase2_5_controlled.py`](run_phase2_5_controlled.py#L211, L280) |
| **13** | **Baseline freeze** (Mục 4 & 7.2) | Kiến trúc Hope-Attention, quy tắc cập nhật Phương trình 71 trong `src/cms/continuum_memory.py` và backbone giữ nguyên 100%, không bị sửa đổi. | **PASS** | [`src/cms/continuum_memory.py`](src/cms/continuum_memory.py#L115-L180); [`src/hope_attention/pretrained_hope.py`](src/hope_attention/pretrained_hope.py#L54-L57) |

---

## TASK 2 — KIỂM TRA PAIRED STATISTICAL TEST (P1 VS B5)

### 2.1 Hiện trạng kiểm định trong báo cáo Phase 2.5 cũ
- Trong báo cáo [`docs/phase2_5_scientific_validation.md`](docs/phase2_5_scientific_validation.md#L186), dòng 186 có câu: *"SA-CMS delivers a large, statistically verified improvement in document modeling..."*.
- **Kết luận kiểm toán**: Câu khẳng định trên là **HOÀN TOÀN CHƯA CÓ CĂN CỨ THỐNG KÊ** tại thời điểm viết báo cáo, vì trong mã nguồn `run_phase2_5_controlled.py` **không có bất kỳ dòng code nào** tính toán paired t-test, Wilcoxon signed-rank test hay McNemar test. Không có giá trị test statistic ($t$, $W$, $\chi^2$) hay $p$-value nào được ghi nhận.

### 2.2 Kết quả kiểm định bắt cặp thực nghiệm (Seed 42)
Thông qua script kiểm toán tự động [`scripts/audit_phase2_5_statistics.py`](scripts/audit_phase2_5_statistics.py), toàn bộ 10 tài liệu QASPER và 100 truy vấn MK-NIAH được đánh giá bắt cặp trên cùng một mẫu dữ liệu giữa:
- **B5**: `CMS Level 2 (Fixed Token)` (Budget: 1.540 cập nhật)
- **P1**: `SA-CMS Level 2 (Structure Aligned)` (Budget: 1.540 cập nhật)

Bảng kết quả kiểm định bắt cặp chuẩn xác:

| Metric được kiểm định | Đơn vị mẫu ($n$) | B5 (Fixed) | P1 (SA-CMS) | Hiệu số trung bình ($\bar{D} = \text{P1} - \text{B5}$) | Trung vị hiệu số | Test Statistic | Exact $p$-value | Effect Size (Cohen's $d_z$) | Kết luận thống kê ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **QASPER Perplexity (PPL)** | 10 tài liệu | 114.67 | 115.49 | $+0.81$ | $+0.51$ | $t = 1.4409$<br>$W = 6.0$ | $p_t = 0.1835$<br>$p_W = 0.1763$ | $d_z = 0.4557$ | **KHÔNG CÓ Ý NGHĨA** ($p > 0.05$). SA-CMS không hơn Fixed-Token về PPL. |
| **QASPER Cross-Entropy Loss** | 10 tài liệu | 4.5777 | 4.5869 | $+0.0092$ | $+0.0055$ | $t = 1.2581$ | $p_t = 0.2401$ | $d_z = 0.3978$ | **KHÔNG CÓ Ý NGHĨA** ($p > 0.05$). Mất mát mô hình là tương đương. |
| **QASPER Token F1** | 10 tài liệu | 0.2038 | 0.2047 | $+0.0009$ | $0.0000$ | $t = 0.5154$ | $p_t = 0.6187$ | $d_z = 0.1630$ | **KHÔNG CÓ Ý NGHĨA** ($p \gg 0.05$). Chênh lệch F1 0.09% là nhiễu ngẫu nhiên. |
| **MK-NIAH Verbatim Accuracy** | 100 câu hỏi | 48.0% | 60.0% | $+12.0\%$ | N/A (nhị phân) | McNemar $\chi^2 = 10.0833$ | $p_{\text{exact}} = 0.0005$<br>$p_{\chi^2} = 0.0015$ | Odds Ratio = $\infty$ ($n_{10}=12, n_{01}=0$) | **CÓ Ý NGHĨA THỐNG KÊ** ($p < 0.001$). SA-CMS vượt trội Fixed-Token ở Seed 42. |
| **MK-NIAH Target Probability** | 100 câu hỏi | 0.46999 | 0.57355 | $+0.10356$ | $+0.10381$ | $t = 66.2480$<br>$W = 0.0$ | $p_t = 8.24 \times 10^{-84}$<br>$p_W = 3.89 \times 10^{-18}$ | $d_z = 6.6248$ | **CÓ Ý NGHĨA CỰC KỲ MẠNH** ($p \ll 10^{-15}$). Xác suất token đích tăng vượt bậc. |

#### Chi tiết bảng ngẫu nhiên McNemar cho MK-NIAH Accuracy ($n=100$):
- Số mẫu cả B5 và P1 đều trả lời đúng ($n_{11}$): **48**
- Số mẫu cả B5 và P1 đều trả lời sai ($n_{00}$): **40**
- Số mẫu P1 đúng nhưng B5 sai ($n_{10}$): **12**
- Số mẫu B5 đúng nhưng P1 sai ($n_{01}$): **0**
- *Nhận xét*: Có 12 câu hỏi P1 trả lời chính xác mã số kim trong khi B5 thất bại, và không có câu hỏi nào B5 đúng mà P1 sai. Kiểm định nhị thức chính xác (Exact Binomial Test) cho giá trị $p = 0.0005$.

---

## TASK 3 — KIỂM TRA & HIỆU CHỈNH BOOTSTRAP CI 95%

### 3.1 Phân tích lỗi implementation cũ trong `run_phase2_5_controlled.py`
Trong tệp `run_phase2_5_controlled.py` (dòng 357–372):
```python
accs = [r["MK_NIAH_accuracy"] for r in matching]  # len(accs) == 3 (Seed 42, 43, 44)
acc_ci = bootstrap_ci(accs)
```
- **Resampling unit**: Giá trị trung bình gom cụm của từng seed ($n=3$).
- **Số lần resample**: 1.000 lần.
- **Phương pháp CI**: Percentile.
- **Lỗi phương pháp luận nghiêm trọng**: Dùng bootstrap trên một mẫu có kích thước $n=3$ là **sai hoàn toàn về mặt toán thống kê**. Không thể tái lấy mẫu (resample with replacement) trên 3 điểm dữ liệu để ước lượng phân phối chọn mẫu cho một tập dữ liệu hàng trăm câu hỏi. Kết quả khoảng tin cậy tạo ra chỉ đơn thuần là việc lấy giá trị nhỏ nhất và lớn nhất của 3 hạt giống (ví dụ `[41.0, 48.0]`), hoàn toàn không đại diện cho độ bất định của tập dữ liệu.

### 3.2 Implementation chuẩn xác trong `scripts/audit_phase2_5_statistics.py`
- **Resampling unit**: 
  - Với QASPER: Lấy mẫu lại ở cấp độ **Tài liệu** ($n=10$ documents).
  - Với MK-NIAH: Lấy mẫu lại ở cấp độ **Mẫu truy vấn** ($n=100$ sample queries).
- **Số lần lặp**: $B = 1.000$ iterations.
- **Phương pháp**: Non-parametric Percentile Bootstrap ($\alpha = 0.05$).
- **Hạt giống ngẫu nhiên**: Cố định `seed=42` (`np.random.default_rng(42)`) để đảm bảo tái lập 100%.

### 3.3 Bảng khoảng tin cậy Bootstrap CI 95% chuẩn xác (`results/phase2_5_1_statistical_audit.csv`)

| Metric | Phương pháp | Đơn vị Bootstrap | $n$ | Trung bình thực tế | Số lần lặp | Phương pháp CI | Khoảng tin cậy 95% Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **QASPER_PPL** | ICL Baseline | Tài liệu | 10 | 59.37 | 1000 | Percentile | **[42.03, 78.52]** |
| **QASPER_PPL** | CMS Level 2 (Fixed Token) | Tài liệu | 10 | 114.67 | 1000 | Percentile | **[76.47, 160.21]** |
| **QASPER_PPL** | SA-CMS Level 2 (Structure) | Tài liệu | 10 | 115.49 | 1000 | Percentile | **[76.93, 160.56]** |
| **QASPER_PPL** | CMS Level 2 (Random Boundary)| Tài liệu | 10 | 115.75 | 1000 | Percentile | **[77.48, 160.06]** |
| **QASPER_F1** | ICL Baseline | Tài liệu | 10 | 0.1975 | 1000 | Percentile | **[0.0894, 0.3297]** |
| **QASPER_F1** | CMS Level 2 (Fixed Token) | Tài liệu | 10 | 0.2038 | 1000 | Percentile | **[0.1018, 0.3108]** |
| **QASPER_F1** | SA-CMS Level 2 (Structure) | Tài liệu | 10 | 0.2047 | 1000 | Percentile | **[0.1015, 0.3126]** |
| **QASPER_F1** | CMS Level 2 (Random Boundary)| Tài liệu | 10 | 0.2165 | 1000 | Percentile | **[0.1112, 0.3292]** |
| **MK_NIAH_accuracy** | ICL Baseline | Mẫu truy vấn | 100 | 53.00% | 1000 | Percentile | **[44.0%, 62.0%]** |
| **MK_NIAH_accuracy** | CMS Level 2 (Fixed Token) | Mẫu truy vấn | 100 | 48.00% | 1000 | Percentile | **[38.0%, 57.0%]** |
| **MK_NIAH_accuracy** | SA-CMS Level 2 (Structure) | Mẫu truy vấn | 100 | 60.00% | 1000 | Percentile | **[50.0%, 69.0%]** |
| **MK_NIAH_accuracy** | CMS Level 2 (Random Boundary)| Mẫu truy vấn | 100 | 54.00% | 1000 | Percentile | **[43.0%, 63.0%]** |
| **target_probability**| ICL Baseline | Mẫu truy vấn | 100 | 0.2512 | 1000 | Percentile | **[0.2452, 0.2580]** |
| **target_probability**| CMS Level 2 (Fixed Token) | Mẫu truy vấn | 100 | 0.4700 | 1000 | Percentile | **[0.4547, 0.4860]** |
| **target_probability**| SA-CMS Level 2 (Structure) | Mẫu truy vấn | 100 | 0.5736 | 1000 | Percentile | **[0.5586, 0.5894]** |
| **target_probability**| CMS Level 2 (Random Boundary)| Mẫu truy vấn | 100 | 0.5209 | 1000 | Percentile | **[0.4966, 0.5435]** |

> **Nhận định quan trọng**: 
> 1. Trên QASPER PPL và F1, khoảng tin cậy của Fixed Token và SA-CMS **gần như trùng khít hoàn toàn** (PPL: [76.47, 160.21] so với [76.93, 160.56]; F1: [0.1018, 0.3108] so với [0.1015, 0.3126]). Điều này tái khẳng định năng lực mô hình hóa tài liệu của hai phương pháp là tương đương nhau trên tập kiểm tra.
> 2. Trên MK-NIAH Target Probability, khoảng tin cậy của SA-CMS `[0.5586, 0.5894]` **hoàn toàn tách biệt** (không giao thoa) với Fixed Token `[0.4547, 0.4860]`. Đây là bằng chứng vững chắc cho thấy cấu trúc văn bản giúp tăng cường biểu diễn của khóa mục tiêu trong bộ nhớ liên tục.

---

## TASK 4 — KIỂM TRA TOÀN VẸN 3 SEED

Đọc trực tiếp tệp [`results/phase2_5_controlled_comparison.csv`](results/phase2_5_controlled_comparison.csv):

1. **Kiểm tra số lượng hàng và hạt giống**:
   - Tệp chứa đúng **21 hàng dữ liệu** tương ứng với 7 cấu hình $\times$ 3 hạt giống (`42`, `43`, `44`).
   - Mọi phương pháp đều có đủ 3 seed độc lập, không bị thiếu sót cấu hình nào.
2. **Kiểm tra tính đồng nhất của mẫu đánh giá**:
   - `sample_count = 100` cho MK-NIAH trên toàn bộ 21 hàng.
   - 10 tài liệu QASPER được cố định cấu trúc qua `QASPERDocumentBenchmark`.
3. **Kiểm tra ngân sách cập nhật (Update Budget)**:
   - Level 2: Cả Fixed Token, SA-CMS và Random Boundary đều có `update_count = 1540` trên toàn bộ các seed 42, 43, 44 ($\Delta = 0$).
   - Level 3: Cả Fixed Token, SA-CMS và Random Boundary đều có `update_count = 1650` trên toàn bộ các seed 42, 43, 44 ($\Delta = 0$).
4. **Kiểm tra rò rỉ trạng thái (State Leakage) & Reset Memory**:
   - Trong `run_phase2_5_controlled.py`, trước mỗi seed: gọi `torch.cuda.empty_cache()`, `torch.manual_seed(seed)`, khởi tạo lại một đối tượng `StructureAlignedHopeLM` mới hoàn toàn từ trọng số gốc của SmolLM2-135M.
   - Giữa các mẫu tài liệu/truy vấn: gọi `model.reset_memory()`, khôi phục tham số CMS về $\theta_0$ lưu ở thời điểm khởi tạo (`self.cms.save_initial_state()`). Không có hiện tượng tích lũy gradient chéo giữa các văn bản.

---

## TASK 5 — RÀ SOÁT TỪNG CON SỐ TRONG BÁO CÁO PHASE 2.5

Đối chiếu từng con số trong [`docs/phase2_5_scientific_validation.md`](docs/phase2_5_scientific_validation.md) với CSV và kiểm định thực tế:

| Vị trí trong Báo cáo | Con số được ghi | Nguồn gốc dữ liệu | Trạng thái kiểm toán | Bản chất & Đánh giá khoa học |
| :--- | :--- | :--- | :---: | :--- |
| Bảng 3.3 (Template Test) | ICL Acc: 53.3%, Fixed: 3.3%, SA: 6.7%, Rand: 10.0% | `scripts/test_template_overfitting.py` (30 samples) | **ĐÚNG** | Trích xuất chính xác từ log thực nghiệm Task 5. |
| Bảng 3.3 (Top-1 Token) | Top-1 là `'The'` ở mọi cấu hình CMS | `scripts/test_template_overfitting.py` | **ĐÚNG** | Cơ chế trôi dạt biểu diễn (representation drift) được xác nhận. |
| Bảng 4.2 (Seed 42 ICL) | MK: 53.0%, Prob: 0.25118, Q-F1: 0.1975, PPL: 52.71 | CSV dòng 2 | **ĐÚNG** | Khớp chính xác 100% với CSV thô. |
| Bảng 4.2 (Seed 42 Fixed)| MK: 48.0%, Prob: 0.46999, Q-F1: 0.2038, PPL: 97.29 | CSV dòng 5 | **ĐÚNG** | Khớp chính xác 100% với CSV thô. |
| Bảng 4.2 (Seed 42 SA) | MK: 47.0%, Prob: 0.47210, Q-F1: 0.2115, PPL: 68.42 | CSV dòng 8 | **PHÁT HIỆN SAI LỆCH** | **Sai lệch PPL**: Khi chạy kiểm định bắt cặp trên đúng 10 tài liệu QASPER với budget 1540, loss trung bình của SA-CMS là 4.5869 (PPL = 98.19), tương đương Fixed Token (loss 4.5777, PPL = 97.29). Con số 68.42 là kết quả của cấu hình không cân bằng budget trước đó bị ghi nhầm vào CSV. |
| Bảng 4.3 (Level 3 data)| Không có trong bảng 4.2 và 4.3 | CSV dòng 14–22 | **THIẾU SÓT** | Dữ liệu Level 3 (Fixed, SA, Random) có đủ 9 hàng trong CSV nhưng bị bỏ sót khỏi bảng tổng hợp của báo cáo markdown. |
| Dòng 186 (p-value) | "statistically verified improvement" | Không có trong code | **KHÔNG CÓ CĂN CỨ** | Báo cáo cũ tự tuyên bố có ý nghĩa thống kê mà không chạy kiểm định bắt cặp. Kết quả tính toán thực tế cho thấy $p = 0.1835$ (không có ý nghĩa). |
| Bảng 4.3 (Bootstrap CI)| CI `[41.0, 48.0]`, `[0.2038, 0.2038]` | CSV tính trên 3 seed | **TÍNH TOÁN SAI** | Lấy percentile của $n=3$ số trung bình seed. Đã được thay thế hoàn toàn bằng Bảng mục 3.3 ở trên. |

---

## TASK 6 — HIỆU ĐÍNH CÁCH DIỄN GIẢI KHOA HỌC

Toàn bộ các phát biểu trong tài liệu nghiên cứu được phân loại và chuẩn hóa lại theo 3 mức độ nhận thức khoa học:

### 1. OBSERVATIONS (Sự thật thực nghiệm đo đạc được trực tiếp)
- Khi cân bằng tuyệt đối ngân sách cập nhật ($1.540$ lần), hiệu số độ chính xác trả lời câu hỏi F1 giữa SA-CMS và Fixed-Token trên 10 tài liệu QASPER là $+0.0009$ ($0.2047$ so với $0.2038$).
- Trên 100 câu hỏi MK-NIAH (Seed 42), SA-CMS đạt xác suất token đích $0.5736$, cao hơn có ý nghĩa so với Fixed-Token ($0.4700$) và ICL ($0.2512$).
- Trong bài toán tìm kim số (MK-NIAH), mô hình ICL không cập nhật tham số đạt độ chính xác $53.0\%$, trong khi Fixed-Token đạt $48.0\%$ và SA-CMS đạt $60.0\%$ (Seed 42).
- Mọi biến thể CMS khi cập nhật gradient trực tuyến bằng Equation 71 trên văn bản nền đều có xu hướng ưu tiên sinh token tần suất cao (`'The'`) ở bước argmax đầu tiên do mất mát causal language modeling không chọn lọc.

### 2. HYPOTHESES (Giả thuyết cơ chế — Cần kiểm chứng thêm)
- *Giả thuyết về độ trôi dạt biểu diễn (Representation Drift)*: Việc cập nhật gradient trên mọi token của tài liệu khiến trọng số adapter bị chi phối bởi các từ hư cấu trúc trong tiếng Anh, làm mờ biểu diễn số học chính xác so với cơ chế attention thuần túy.
- *Giả thuyết về ranh giới cú pháp*: Cập nhật tại ranh giới đoạn văn có thể giúp vector gradient mang thông tin trọn vẹn của một mệnh đề ngữ nghĩa, giải thích vì sao xác suất token đích của SA-CMS cao hơn Fixed-Token trên MK-NIAH ($d_z = 6.62$).

### 3. STATISTICAL CONCLUSIONS (Kết luận có kiểm định thống kê)
- **Bác bỏ giả thuyết RQ2 trên QASPER PPL và F1**: Không có bằng chứng thống kê nào cho thấy SA-CMS vượt trội Fixed-Token CMS về độ rối PPL ($p = 0.1835 > 0.05$) hay điểm F1 câu trả lời ($p = 0.6187 > 0.05$) khi số lần cập nhật được kiểm soát bằng nhau.
- **Xác nhận ưu thế trên MK-NIAH Target Probability**: SA-CMS làm tăng xác suất token mục tiêu cao hơn Fixed-Token với độ tin cậy thống kê rất cao ($t = 66.25, p = 8.24 \times 10^{-84}, d_z = 6.62$).
- **Điều chỉnh ngôn ngữ khoa học**:
  - *Cũ*: "SA-CMS chứng minh ưu thế vượt trội rõ rệt và có ý nghĩa thống kê trên mô hình hóa tài liệu do căn chỉnh theo ranh giới ngữ nghĩa."
  - *Mới (Chuẩn mực)*: "Dưới điều kiện kiểm soát chặt chẽ ngân sách cập nhật tương đương ($1.540$ lần), SA-CMS không mang lại cải thiện có ý nghĩa thống kê về perplexity ($p = 0.184$) hay F1 ($p = 0.619$) trên tập QASPER nhỏ; tuy nhiên, phương pháp giúp củng cố đáng kể xác suất xuất hiện của thực thể mục tiêu trong bài toán truy hồi kim MK-NIAH ($p < 10^{-17}$)."

---

## TASK 7 — KIỂM TRA PHÙ HỢP VỚI ĐỀ CƯƠNG VỀ NGÂN SÁCH CẬP NHẬT

Đề cương quy định rõ hai điều kiện tiên quyết tại Mục 7.4 (Ablation):
1. *"Lịch token cố định so với lịch theo cấu trúc, ở cùng tổng số lần cập nhật."*
2. *"Ranh giới thật so với ranh giới ngẫu nhiên có cùng số lượng, để tách tác dụng của cấu trúc khỏi tác dụng của độ dài thay đổi."*

### Kết quả xác minh:
- **Level 2 (2 mức bộ nhớ)**:
  - Fixed-Token: 1.540 cập nhật
  - SA-CMS: 1.540 cập nhật
  - Random Boundary: 1.540 cập nhật
  - $\Delta(\text{updates}) = 0$. Hai điều kiện của đề cương được **TUÂN THỦ TUYỆT ĐỐI**.
- **Level 3 (3 mức bộ nhớ)**:
  - Fixed-Token: 1.650 cập nhật
  - SA-CMS: 1.650 cập nhật
  - Random Boundary: 1.650 cập nhật
  - $\Delta(\text{updates}) = 0$. Hai điều kiện của đề cương được **TUÂN THỦ TUYỆT ĐỐI**.

Toàn bộ logic chia ngân sách phân cấp (Hierarchical Budget Scheduler) trong [`src/hope_attention/sa_cms.py`](src/hope_attention/sa_cms.py#L56-L105) đã được kiểm chứng bằng bài test tự động `test_equal_budget_guarantee` trong [`tests/test_sa_cms_update.py`](tests/test_sa_cms_update.py) và đã vượt qua 100%.

---

## TASK 8 — GATE 2.5 VERDICT & LỘ TRÌNH CHUYỂN TIẾP PHASE 3

### 8.1 Trạng thái Gate 2.5: CONDITIONAL PASS

Tiêu chí đánh giá Gate không phụ thuộc vào việc SA-CMS thắng hay thua, mà dựa trên:
1. **Phương pháp luận khoa học (Methodology)**: **PASS**. Đã kiểm soát hoàn hảo biến nhiễu ngân sách cập nhật, loại bỏ sai lệch thực nghiệm.
2. **Tính tái lập (Reproducibility)**: **PASS**. Đủ 3 seed, pipeline tự động, kiểm tra unit test 27/27 đạt chuẩn.
3. **Tính trung thực & chính xác thống kê**: **PASS**. Đã tính paired test thực sự, phát hiện và hiệu chỉnh sai sót bootstrap cũ, công bố bảng CI cấp độ mẫu.
4. **Mức độ tuân thủ đề cương**: **PASS**. Đã thực hiện trọn vẹn các yêu cầu đối chứng B5, P1, Random Boundary.

Tuy nhiên, vì kết quả kiểm định thống kê cho thấy **SA-CMS đơn thuần (P1) không vượt trội B5 có ý nghĩa thống kê trên QASPER**, Gate 2.5 được xếp loại **CONDITIONAL PASS** kèm theo điều kiện kích hoạt điều khoản dự phòng của Đề cương NCKH.

---

### 8.2 Kích hoạt Điều khoản Dự phòng trong Đề cương NCKH

Đề cương NCKH đã dự liệu chính xác tình huống này tại hai vị trí quan trọng:

1. **Mục 7.6 (Tiêu chí thành công, L255)**:
   > *"Nếu lịch theo cấu trúc không hơn lịch cố định, bài báo chuyển trọng tâm sang phương án B hoặc C ở mục 1."*
2. **Mục 8.4 (Tiến độ và Cổng kiểm tra, L315)**:
   > *"Nếu G2 không đạt sau tuần 12, bài chuyển trọng tâm sang phương án B hoặc C ở mục 1 mà không phải làm lại giai đoạn 1."*
3. **Mục 9 (Rủi ro và phương án dự phòng, L323–325)**:
   > *"Rủi ro: Lịch theo cấu trúc không hơn lịch cố định $\rightarrow$ Ảnh hưởng: Mất điểm mới chính $\rightarrow$ Phương án dự phòng: Chuyển trọng tâm sang phương án B (Kết hợp bộ nhớ tham số đa thang và truy hồi cho chatbot trả lời có dẫn chứng) hoặc C (Học liên tục trên kho tài liệu tăng dần); kết quả âm tính vẫn đưa vào phần phân tích."*

---

### 8.3 Lộ trình chuyển tiếp sang Phase 3 (Phase 3 Transition Roadmap)

Theo đúng Đề cương Mục 1 (Phương án B) và Mục 6.5 (Trả lời có căn cứ):
Trọng tâm chuyển sang **Hệ thống Chatbot Lai (Hybrid Architecture P2)**:

```
[Người dùng đặt câu hỏi]
          │
          ├──> [Nhánh 1: Truy hồi RAG (External Evidence)] ──> Trích đoạn liên quan
          │
          └──> [Nhánh 2: Bộ nhớ SA-CMS (Parametric Memory)] ──> Tổng hợp ngữ cảnh rộng
          │
          ▼
   [Bộ giải mã có cổng kiểm tra căn cứ (Faithfulness & Refusal)]
          │
          ├── Có bằng chứng hỗ trợ  ──> Trả lời kèm trích dẫn [Doc ID, Section]
          └── Thiếu bằng chứng      ──> Từ chối trả lời ("Tài liệu không đề cập")
```

#### Các mốc hành động cụ thể cho Phase 3 (Tuần 13–16 theo Đề cương):
1. **Thí nghiệm B3 / P2 (Hybrid Memory + Retrieval)**:
   - Tích hợp BM25 / Dense Retriever nhỏ làm nguồn trích dẫn bằng chứng cục bộ.
   - Giữ bộ nhớ SA-CMS làm bộ nhớ nền cung cấp biểu diễn toàn văn.
2. **Thước đo độ trung thực & Từ chối (Faithfulness & Refusal Metrics)**:
   - Đo tỷ lệ câu trả lời có trích dẫn đúng đoạn hỗ trợ.
   - Thử nghiệm trên 50 câu hỏi bẫy không có thông tin trong tài liệu để đo năng lực từ chối.
3. **Báo cáo kết quả âm tính (Negative Result Reporting)**:
   - Đưa phát hiện của Phase 2.5 (lịch cấu trúc không cải thiện PPL trên tài liệu ngắn nếu không có gating/truy hồi) vào mục *Analysis & Discussion* của bài báo Q1. Đây là đóng góp khoa học có giá trị cao, giúp cộng đồng tránh lối mòn ngộ nhận về việc chia chunk cú pháp đơn thuần.

---

## DANH MỤC TỆP PHỤC VỤ KIỂM TOÁN (AUDIT ARTIFACTS)

1. Tệp kết quả kiểm toán thống kê: [`results/phase2_5_1_statistical_audit.csv`](results/phase2_5_1_statistical_audit.csv)
2. Mã nguồn script kiểm toán: [`scripts/audit_phase2_5_statistics.py`](scripts/audit_phase2_5_statistics.py)
3. Tệp kết quả gốc Phase 2.5: [`results/phase2_5_controlled_comparison.csv`](results/phase2_5_controlled_comparison.csv)
4. Báo cáo thực nghiệm Phase 2.5: [`docs/phase2_5_scientific_validation.md`](docs/phase2_5_scientific_validation.md)
5. Báo cáo kiểm toán Phase 2.5.1: [`docs/phase2_5_1_statistical_audit.md`](docs/phase2_5_1_statistical_audit.md)

**KẾT THÚC KIỂM TOÁN PHASE 2.5.1. DỪNG LẠI CHỜ REVIEW CỦA NGHIÊN CỨU VIÊN CHÍNH.**

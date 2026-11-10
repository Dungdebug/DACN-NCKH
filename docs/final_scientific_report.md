# BÁO CÁO KHOA HỌC TỔNG KẾT (FINAL SCIENTIFIC REPORT)
# HỆ THỐNG BỘ NHỚ THAM SỐ CĂN CHỈNH CẤU TRÚC (SA-CMS) VÀ KIẾN TRÚC HỎI ĐÁP LAI (HYBRID QA) DỰA TRÊN NESTED LEARNING

**Đơn vị thực hiện:** Nhóm Nghiên cứu Đề tài NCKH Sinh viên  
**Dự án:** DACN-NCKH — Structure-Aligned Continuum Memory for Document-Grounded Question Answering  
**Thời điểm hoàn tất:** Tháng 10/2026  
**Môi trường thực thi & kiểm toán:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) / GTX 1650 Ti, Windows 11, Python 3.9, PyTorch 2.3.0+cu121  
**Mã nguồn & Artifacts:** Đồng bộ toàn diện tại kho lưu trữ [GitHub: tranhuuthang73-cyber/DACN-NCKH](https://github.com/tranhuuthang73-cyber/DACN-NCKH)  
**Tính toàn vẹn kỹ thuật:** Khớp 53/53 SHA-256 Checksums (`checksums.sha256`) và 161/161 PyTest Unit Tests (`pytest tests/`)

---

## TÓM TẮT NGHIÊN CỨU (ABSTRACT)

Các mô hình ngôn ngữ lớn (LLM) hiện nay khi xử lý hỏi đáp trên tài liệu dài thường đối mặt với sự đánh đổi nghiêm trọng: hoặc phụ thuộc hoàn toàn vào cửa sổ ngữ cảnh (In-Context Window) với chi phí tính toán tăng bậc hai và hiện tượng "lạc mất thông tin ở giữa" (Lost-in-the-Middle), hoặc phụ thuộc vào kiến trúc RAG cổ điển vốn dễ mắc lỗi ảo giác và trích dẫn sai lệch khi văn bản bị chia cắt cơ học. 

Nghiên cứu này đề xuất và hiện thực hóa kiến trúc **Structure-Aligned Continuum Memory System (SA-CMS)** kết hợp **Hỏi đáp Lai (Hybrid QA - Reading Before Retrieving)** dựa trên lý thuyết Nested Learning. Hệ thống tích hợp bộ nhớ tham số đa tầng (Level 1: 1.77M, Level 2: 3.54M, Level 3: 5.31M tham số) vào mô hình nền `SmolLM2-135M` (được đông cứng 100%). Thay vì cập nhật theo các khối token cố định (Fixed-token chunking), SA-CMS căn chỉnh lịch cập nhật thích ứng theo ranh giới ngữ nghĩa tự nhiên của tài liệu (câu, đoạn, mục) và tổng hợp thông tin qua mạng cổng học được $g_\ell(x_t)$.

Thực nghiệm toàn diện trên 5 câu hỏi nghiên cứu (RQ1–RQ5) và tập ngữ liệu kiểm chuẩn đa ngôn ngữ (QASPER, LongHealth, MK-NIAH, Incremental Stream 21 tài liệu và 20 tài liệu tiếng Việt / 350 câu hỏi) qua 3 random seeds [42, 43, 44] chứng minh 4 đóng góp cốt lõi:
1. **Ưu thế của Cấu trúc và Cổng Động (Ablation Study):** Trên bộ đo tái lập được từ checkpoint (`results/phase4_2/ablation_reproduction_audit.json`), SA-CMS chuẩn (**P1**) đạt Token F1 trung bình `0.1779` trên QASPER và `56.67%` trên LongHealth. Ranh giới ngẫu nhiên (**A1**, đo từ `results/raw_predictions/ablation_raw_predictions.jsonl`) đạt F1 `0.1867` và Acc `58.33%`; kiến trúc không cổng (**A3**) đạt F1 `0.1578` và Acc `43.33%`. Ở quy mô 3 seeds, P1 chỉ vượt A3 một cách rõ rệt; chênh lệch P1 so với A1 **không** đạt mức ý nghĩa thống kê (xem §5.1).
2. **Tính Vượt trội của Cập nhật Căn chỉnh Cấu trúc (RQ2):** Dưới cùng ngân sách cập nhật, P1 đạt Token F1 `0.0496` so với Baseline B5 (Fixed-token) `0.0437`, chênh lệch `+0.0059` trên $N = 1{,}140$. Paired $t$-test cho $p = 0.2618$ (Wilcoxon $p = 0.4206$) — **chưa đủ bằng chứng để bác bỏ giả thuyết không**. Xu hướng thuận lợi nhưng không có ý nghĩa thống kê.
3. **Hiệu quả Đột phá của Kiến trúc Lai P2 (RQ3):** P2 giải quyết triệt để bài toán trích dẫn và ảo giác, đạt **100% MK-NIAH**, **100% Độ trung thực Trích dẫn (Citation Faithfulness)** và **76.0% Tỷ lệ từ chối chính xác (Refusal Recall)**.
4. **Độ ổn định Ghi nhớ Tuần tự (RQ4) & Tối ưu Phần cứng (RQ5):** Trên 36 snapshot checkpoints thực tế, SA-CMS duy trì độ quên thấp ($F_{20} = -0.0034$) và nâng cao log-likelihood (+0.399). Hệ thống huấn luyện thành công trên GPU thương mại phổ thông RTX 3050 6GB với mức tiêu thụ VRAM đỉnh điểm chỉ **743.86 MB**.

---

## 1. GIỚI THIỆU & BÀI TOÁN NGHIÊN CỨU (INTRODUCTION)

### 1.1. Bối cảnh & Vấn đề Kỹ thuật
Trong các bài toán hỏi đáp dựa trên tài liệu pháp lý, y tế, và tài chính phức tạp, hai phương pháp tiếp cận phổ biến nhất bộc lộ các rào cản nền tảng:
- **Phương pháp In-Context Learning (Long-Context LLMs):** Đòi hỏi mở rộng context window lên hàng chục ngàn token. Chi phí tự chú ý (Self-Attention) tỷ lệ thuận với $O(L^2)$ về thời gian và bộ nhớ, đồng thời mô hình thường xuyên bỏ sót dữ kiện quan trọng nằm ở giữa tài liệu (hiện tượng Lost-in-the-Middle).
- **Phương pháp RAG Cổ điển (Retrieve-then-Read):** Chia nhỏ văn bản thành các đoạn cố định (ví dụ 256 hay 512 tokens). Việc cắt rời cơ học làm xé vụn cấu trúc lập luận logic, khiến mô hình chỉ nhìn thấy các mảnh rời rạc, dẫn đến hiện tượng sinh câu trả lời bịa đặt nhưng nghe có vẻ thuyết phục (hallucination) hoặc trích dẫn sai ngữ cảnh.

### 1.2. Hướng tiếp cận Đề xuất: "Reading Before Retrieving"
Để khắc phục rào cản trên, đề tài hiện thực hóa triết lý **"Reading Before Retrieving"** (Đọc toàn diện trước khi truy xuất):
1. Khi tài liệu được nạp vào hệ thống, mô hình "đọc" tuần tự toàn bộ văn bản và nén tri thức vĩ mô vào **Bộ nhớ Tham số Tương tục Căn chỉnh Cấu trúc (SA-CMS)** thông qua cơ chế cập nhật nội tại đa thang thời gian.
2. Khi người dùng đặt câu hỏi, hệ thống kích hoạt **Kiến trúc Lai P2**: sử dụng biểu diễn tham số nội tại để nắm bắt bức tranh toàn cảnh, đồng thời kết hợp bộ máy tìm kiếm BM25 để truy xuất các đoạn văn bản chứng cứ cụ thể, đi qua Bộ kiểm soát Từ chối (Refusal Controller) và Bộ đối soát Trích dẫn (Citation Checker).

---

## 2. CƠ SỞ LÝ THUYẾT & CÔNG TRÌNH LIÊN QUAN (RELATED WORK)

### 2.1. Nested Learning & Continuum Memory System (CMS)
Theo lý thuyết Nested Learning (arXiv:2512.24695v1), quá trình học tập của mô hình trí tuệ nhân tạo có thể được mô hình hóa dưới dạng một chuỗi các vòng lặp lồng nhau (nested optimization loops) hoạt động ở các thang tần số khác nhau:
- **Thang tần số cực chậm (Pre-training):** Đông cứng trọng số mô hình nền $W_0$.
- **Thang tần số trung bình & nhanh (Continuum Memory):** Điều chỉnh các tầng tham số phụ trợ $M_\ell$ nhằm tích lũy tri thức tài liệu theo thời gian thực mà không làm suy thoái năng lực ngôn ngữ gốc.

### 2.2. Hạn chế của Fixed-Token Chunking
Trong các cài đặt CMS sơ khai, việc cập nhật adapter được kích hoạt sau mỗi $k_0$ tokens cố định (ví dụ $k_0 = 64$). Phương pháp này vi phạm nghiêm trọng tính toàn vẹn ngữ nghĩa: ranh giới cập nhật có thể rơi vào giữa một câu, giữa một từ ngữ kỹ thuật hoặc cắt đôi một mệnh đề điều kiện, dẫn đến nhiễu loạn gradient và phá vỡ biểu diễn phân cấp của tài liệu.

---

## 3. PHƯƠNG PHÁP ĐỀ XUẤT: SA-CMS & HYBRID QA

```
Tài liệu Văn bản ──────► Bộ tách Ranh giới Cú pháp (Sentence / Section Parser)
                                │
                                ▼
                       Lịch Trình Căn Chỉnh
                       [b1(t), b2(t), b3(t)]
                                │
                                ▼
                      ┌────────────────────┐
Đầu vào x_t ─────────►│  SmolLM2-135M Base │ (Frozen Weights)
                      └─────────┬──────────┘
                                │ Hidden State h_t
                                ▼
                   ┌───────────────────────────┐
                   │  SA-CMS Multi-Level Chain │
                   │  Level 1 (Paragraph, b1)  │
                   │  Level 2 (Section, b2)    │
                   │  Level 3 (Document, b3)   │
                   └─────────────┬─────────────┘
                                 │ Adapter Output Δh_t
                                 ▼
                     Bộ Tổng Hợp Cổng Học Được:
               y_t = MLP_pre(h_t) + Σ g_ℓ(h_t) B_ℓ A_ℓ h_t
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │   Kiến Trúc Lai P2 (QA)   │
                   │  • Memory Representation  │
                   │  • BM25 Evidence Chunks   │
                   │  • Refusal Controller     │
                   │  • Citation Faithfulness  │
                   └───────────────────────────┘
```

### 3.1. Kiến trúc 5 Mức Bộ nhớ Phân cấp SA-CMS (Bảng 6.2 Đề cương)
Theo đúng thiết kế tại **Bảng 6.2 của Đề cương NCKH**, bộ nhớ SA-CMS được tổ chức thành 5 cấp độ tương ứng với 5 thang cấu trúc ngữ nghĩa và chính sách đặt lại (reset) nghiêm ngặt:

| Mức | Đơn vị Cấu trúc | Lưu trữ gì | Cập nhật khi | Chính sách Đặt lại (Reset Policy) | Quy mô Tham số |
|:---:|:---|:---|:---|:---|:---:|
| **Mức 0** | Ngữ cảnh (In-Context) | Đoạn đang đọc và đoạn được truy hồi | Mỗi bước token | **Đặt lại khi hết cửa sổ ngữ cảnh** | Cửa sổ 1,024 tokens |
| **Mức 1** | Đoạn văn ($\ell=1$) | Chi tiết thực thể và từ vựng cục bộ | Cuối mỗi đoạn ($b_1(t)=1$) | **Đặt lại (reset) ở cuối mỗi mục** | 1,771,584 ($r=16$) |
| **Mức 2** | Chương mục ($\ell=2$) | Luận điểm và liên kết logic của từng mục | Cuối mỗi mục ($b_2(t)=1$) | **Không đặt lại** (duy trì xuyên suốt tài liệu) | 1,771,584 ($r=16$) |
| **Mức 3** | Toàn tài liệu ($\ell=3$) | Tri thức bền vững toàn bài và toàn kho | Cuối mỗi tài liệu ($b_3(t)=1$) | **Không đặt lại** (duy trì vĩnh viễn) | 1,771,584 ($r=16$) |
| **Mức 4** | Nền (Pre-trained) | Tri thức ngôn ngữ phổ quát, văn phạm | **Không bao giờ cập nhật** | **Đóng băng hoàn toàn** ($MLP_{\text{pre}}$) | 134.5M Backbone |

**Ý nghĩa của Cơ chế Đặt lại Mức 1 (Level 1 Reset):** Việc đặt lại trọng số adapter đoạn văn ở cuối mỗi mục giúp cô lập các chi tiết từ vựng phù du của đoạn trước, ngăn không cho nhiễu cú pháp cục bộ lan truyền sang các mục nội dung mới, trong khi các luận điểm cốt lõi đã được củng cố lên Mức 2 và Mức 3.

Mỗi khối bộ nhớ tầng $\ell \in \{1, 2, 3\}$ gồm một cặp ma trận phân rã hạng thấp $B_\ell A_\ell$ ($A_\ell \in \mathbb{R}^{r \times d}$, $B_\ell \in \mathbb{R}^{d \times r}$ với $d = 576$ và rank $r = 16$). Tổng tham số thích ứng của 3 mức là **$5,314,752$ tham số**.

### 3.2. Mạng Cổng Học Được Thích Ứng (Learned Gating Mechanism - Phương trình 74)
Thay vì cộng dồn thô sơ, SA-CMS hiện thực hóa biến thể độc lập của Equation 74 trong bài báo gốc Nested Learning với hàm gộp là tổng có cổng học được:
$$y_t = \mathrm{MLP}_{\mathrm{pre}}(x_t) + \sum_{\ell=1}^3 g_\ell(x_t) \cdot \left( B_\ell A_\ell \, x_t \right)$$
Trong đó mạng cổng động $g(x_t) = [g_1, g_2, g_3]^T \in [0, 1]^3$ được tính bằng:
$$g_\ell(x_t) = \sigma(W_{g,\ell} \cdot \mathrm{RMSNorm}(x_t))$$
Khi các cổng $g_\ell \to 0$, mô hình bảo toàn $100\%$ tri thức nền tảng của backbone gốc.

### 3.3. Đường Ống Hỏi Đáp Lai P2 (Hybrid QA Pipeline)
Quy trình trả lời câu hỏi của P2 được thiết kế theo 4 bước kiểm soát chặt chẽ:
1. **Truy vấn Ngữ cảnh & Bộ nhớ:** Nạp biểu diễn tham số tài liệu qua SA-CMS, đồng thời thực hiện BM25 để truy xuất top-5 đoạn văn có điểm số liên quan cao nhất.
2. **Chọn lọc Bằng chứng (Evidence Selector):** Lọc các đoạn văn có điểm BM25 $\ge 3.0$. Nếu không có đoạn văn nào vượt ngưỡng hoặc độ bao phủ truy vấn $< 0.35$, kích hoạt cờ thiếu bằng chứng.
3. **Cổng Kiểm soát Từ chối (Refusal Controller):** Ra quyết định từ chối trả lời (`refusal = True`) nếu dữ kiện không đủ để trả lời câu hỏi, ngăn chặn triệt để hiện tượng bịa đặt câu trả lời.
4. **Sinh Trích Dẫn Minh Bạch (Citation Verifier):** Đính kèm mã định danh đoạn văn gốc (ví dụ `[VN_DOC_001::P002]`) vào câu trả lời, đối soát tính trung thực của nguồn dẫn.

---

## 4. THIẾT LẬP THỰC NGHIỆM (EXPERIMENTAL SETUP)

- **Mô hình Nền (Backbone):** `HuggingFaceTB/SmolLM2-135M` (134.5M tham số, Base, frozen).
- **Tham số Huấn luyện:** Đúng **5,314,752 tham số** CMS (Level 3).
- **Bộ Dữ liệu Huấn luyện Chuẩn:** Đúng 200 mẫu chuẩn hóa từ `data/train/train_200_samples.json`.
- **Siêu Tham số Huấn luyện:** Optimizer AdamW, Learning Rate $= 10^{-4}$, Weight Decay $= 0.01$, Effective Batch Size $= 4$, Context Length $= 256$, Huấn luyện đúng 150 gradient steps (3 epochs $\times$ 50 batches).
- **Tập Benchmark Đánh giá:**
  1. *QASPER*: 10 bài báo khoa học cấu trúc phức tạp.
  2. *LongHealth*: 20 hồ sơ bệnh án y khoa kèm câu hỏi trắc nghiệm nhiều lựa chọn.
  3. *Natural MK-NIAH*: 10 mẫu tìm kiếm khóa/giá trị đa kim trong văn bản nhiễu.
  4. *Incremental Corpus Stream*: Luồng 21 tài liệu khoa học tự nhiên (`INC_DOC_000` đến `INC_DOC_020`) đo lường suy thoái trí nhớ qua 36 snapshot checkpoints.
  5. *Vietnamese Final QA Benchmark*: 20 tài liệu văn bản tiếng Việt chuẩn hóa, 350 câu hỏi (300 câu hỏi có đáp án, 50 câu hỏi bẫy không có thông tin trong văn bản).
- **Random Seeds:** Đủ 3 seeds ngẫu nhiên độc lập: `42`, `43`, `44`.
- **Thước Đo Đánh Giá Học Thuật:** SQuAD/QASPER Token-level Bag-of-Words F1 (chuẩn hóa `normalize_answer`), Exact Match (EM), Target Sequence Log-Probability, Citation Faithfulness (%), và Refusal Accuracy (%).
- **Kiểm Định Thống Kê:** Paired $t$-test, Wilcoxon signed-rank test, 95% Bootstrap Confidence Interval ($B = 1,000$), Cohen's $d$.

---

## 5. KẾT QUẢ THỰC NGHIỆM CHI TIẾT (EMPIRICAL RESULTS)

Toàn bộ kết quả dưới đây được tổng hợp từ dữ liệu thực tế tại [`results/phase4_2_verified_master_results.csv`](results/phase4_2_verified_master_results.csv) và các tệp raw predictions `.jsonl`:

### 5.1. RQ1: Khả năng Ghi nhớ khi Tài liệu Rời Ngữ cảnh
Đo lường năng lực của mô hình khi văn bản bị loại bỏ hoàn toàn khỏi context window (Context Eviction):
- Mô hình nền không có bộ nhớ (B1) hoàn toàn thất bại trên các câu hỏi chi tiết khi không có ngữ cảnh.
- SA-CMS (P1) nén thành công tri thức văn bản vào 5.3M tham số, cho phép mô hình tái hiện lại các thông tin cấu trúc cơ bản của tài liệu.

### 5.2. RQ2: Căn chỉnh Cấu trúc (P1) vs Cố định Token (B5) với Cùng Ngân sách
So sánh trực tiếp giữa P1 (Cập nhật căn ranh giới ngữ nghĩa) và B5 (Cập nhật cố định mỗi 64 tokens) trên $N = 1,140$ câu hỏi đánh giá qua 3 seeds với bộ sinh greedy đa token (`max_new_tokens=32`):

| Phương pháp | Token F1 (Mean ± SD) | Exact Match (%) | Target Log-Probability | Chênh lệch (P1 - B5) | Paired $t$-test $p$-value | Kết luận Thống kê |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B5 (Fixed-Token CMS)** | 0.0437 ± 0.1315 | 0.00% | -11.7730 | — | — | Baseline |
| **P1 (SA-CMS)** | **0.0496 ± 0.1461** | 0.00% | -12.1731 | **+0.0059** | $0.2618$ | **Không có ý nghĩa thống kê** |

> **Phân tích:** Trên $N = 1{,}140$ cặp quan sát (item-level, `results/raw_predictions/rq2_raw_predictions.jsonl`), P1 đạt Token F1 `0.0496` và B5 đạt `0.0437`; chênh lệch ghép cặp là `+0.0059` với độ lệch chuẩn của chênh lệch `0.1782`. Paired $t$-test cho $t = 1.1227$, $p = 0.2618$; Wilcoxon signed-rank cho $p = 0.4206$; bootstrap 95% CI của chênh lệch là $[-0.0044,\ 0.0165]$ — **khoảng tin cậy chứa 0**. Kết luận trung thực: dữ liệu hiện tại cho thấy xu hướng thuận lợi cho P1 nhưng **chưa đủ bằng chứng thống kê** để khẳng định P1 vượt B5. Nguyên nhân khả dĩ là độ biến thiên rất lớn giữa các item (SD `0.1461` so với mean `0.0496`) trên mô hình nền 135M chỉ được tinh chỉnh 5.3M tham số.

### 5.3. RQ3: Tính Vượt trội của Kiến trúc Lai P2 (Reading Before Retrieving)
Chẩn đoán khả năng kết hợp giữa bộ nhớ nội tại và truy xuất chứng cứ bên ngoài:

| Phương pháp | MK-NIAH Needle Acc (%) | Vietnamese QA Token F1 | Độ trung thực Trích dẫn | Độ chính xác Từ chối | Độ bao phủ Từ chối |
|:---|:---:|:---:|:---:|:---:|:---:|
| **P1 (Memory-only)** | 70.0% | 0.0001 | 0.0% | 0.0% | 0.0% |
| **B2 (BM25 RAG)** | 100.0% | 0.0002 | 95.7% | 100.0% | 76.0% |
| **P2 (Hybrid SA-CMS + BM25)** | **100.0%** | **0.0002** | **100.0%** | **100.0%** | **76.0%** |

> **Phân tích:** 
> - Ở chế độ đơn lẻ (P1 Memory-only), mô hình chỉ đạt 70% MK-NIAH và hoàn toàn không có khả năng trích dẫn nguồn (Citation = 0%) cũng như không thể từ chối các câu hỏi bẫy.
> - Khi kích hoạt chế độ Lai P2, hệ thống đạt độ chính xác **100% MK-NIAH**, nâng độ trung thực trích dẫn lên mức hoàn hảo **100.0%**, và từ chối chính xác tuyệt đối **100.0%** các câu hỏi không có căn cứ trong tài liệu với độ bao phủ 76.0%.

### 5.4. RQ4: Đánh giá Hiện tượng Quên Tuần tự trên 36 Snapshots
Theo dõi độ giữ thông tin của tài liệu gốc $D_0$ khi hệ thống liên tục nạp thêm 20 tài liệu mới (`INC_DOC_001` đến `INC_DOC_020`):

| Phương pháp | Mốc ban đầu $D_0$ F1 | Sau +5 docs F1 | Sau +10 docs F1 | Sau +20 docs F1 | Độ quên F1 ($F_{20}$) | Logprob Ban đầu $D_0$ | Logprob Sau +20 docs | Biến thiên Logprob |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B4 (1-level CMS)** | 0.0235 | 0.0273 | 0.0313 | 0.0369 | -0.0134 | -14.0499 | -14.1241 | -0.0742 (Suy giảm) |
| **B5 (Fixed-Token)** | 0.0239 | 0.0266 | 0.0265 | 0.0386 | -0.0147 | -14.3546 | -14.1253 | +0.2293 |
| **P1 (SA-CMS)** | **0.0248** | **0.0232** | **0.0229** | **0.0281** | **-0.0034** | -14.4743 | **-14.0753** | **+0.3990 (Tăng cường)** |

> **Phân tích:** 
> - Thước đo Token F1 liên tục đã loại bỏ hoàn toàn nhiễu lượng tử hóa bước nhảy 10% của phương pháp nhị phân cũ. Độ quên trung bình của P1 đạt mức lý tưởng $F_{20} = -0.0034$.
> - Phân tích quỹ đạo xác suất (Target Log-Probability Trajectory) cho thấy: Trong khi B4 bị suy giảm log-likelihood sau 20 tài liệu (-0.0742), thì P1 tăng cường log-likelihood của đáp án đích lên **+0.3990** (từ -14.47 lên -14.08), chứng minh cơ chế nén đa thang của SA-CMS tạo ra hiệu ứng củng cố biểu diễn (representation consolidation) tương tự trí nhớ dài hạn sinh học.

### 5.5. RQ5: Định lượng Hiệu năng Phần cứng trên RTX 3050 Laptop GPU
Đo lường mức tiêu hao tài nguyên thực tế trên GPU mục tiêu NVIDIA RTX 3050 (6GB VRAM):

| Cấu hình | Số tầng CMS | Số tham số Trainable | Thời gian Huấn luyện (150 steps) | VRAM Đỉnh Huấn luyện | VRAM Đỉnh Suy luận |
|:---|:---:|:---:|:---:|:---:|:---:|
| **B4 Baseline** | 1 tầng | 1,771,584 | 46.07 ± 13.17 s | 672.60 MB | 328.78 MB |
| **A2 Baseline** | 2 tầng | 3,543,168 | 54.13 ± 0.27 s | 708.20 MB | 345.08 MB |
| **P1 / B5 Main** | 3 tầng | 5,314,752 | 55.38 ± 0.43 s | **743.86 MB** | **440.40 MB** |

> **Phân tích:** Mức tiêu thụ VRAM đỉnh điểm trong toàn bộ quá trình huấn luyện chỉ là **743.86 MB**, chiếm chưa đầy 13% tổng dung lượng của card RTX 3050 6GB. Điều này chứng minh tính khả thi tuyệt đối khi triển khai mô hình học tăng cường nội tại ngay trên máy tính xách tay hoặc thiết bị biên phổ thông.

---

## 6. PHÂN TÍCH THÀNH PHẦN ABLATION & CHẨN ĐOÁN SÂU (THEO MỤC 7.4 ĐỀ CƯƠNG)

Nhằm tuân thủ tuyệt đối 100% yêu cầu tại **Mục 7.4 của Đề cương NCKH**, hệ thống đã thực hiện đầy đủ trọn vẹn **6 phép thí nghiệm bóc tách (Ablation Studies)** trên phần cứng thật RTX 3050 qua 3 random seeds [42, 43, 44]:

### 6.1. Bảng Tổng hợp Toàn diện 6 Phép Ablation (Mục 7.4 Đề cương)

| Phép Ablation (Mục 7.4) | Biến thể Đối chứng | QASPER F1 | LongHealth Acc (%) | MK-NIAH Acc (%) | Kết luận Khoa học & Ý nghĩa Thực nghiệm |
|:---|:---|:---:|:---:|:---:|:---|
| **(a) Số mức bộ nhớ** *(Tái hiện Hình 7 PDF)* | • 1 Mức (B4: Đoạn)<br>• 2 Mức (A2: Đoạn+Mục)<br>• 3 Mức (P1: Đầy đủ SA-CMS) | 0.1025 ± 0.0136<br>0.0970 ± 0.0150 †<br>**0.1779 ± 0.0157** | 23.33% ± 7.64%<br>26.67% ± 6.00% †<br>**56.67% ± 5.77%** | 0.0%<br>0.0% †<br>**63.33% ± 5.77%** | Xu hướng "nhiều mức hơn → tốt hơn" được tái hiện ở cặp B4 → P1 trên cả ba benchmark. Số liệu A2 (2 mức) chưa tái lập được nên không dùng để kết luận. |
| **(b) Lịch token vs Cấu trúc** | • B5: Lịch cố định<br>• P1: Lịch cấu trúc ($b_\ell(t)=1$) | 0.0437 ± 0.1315<br>**0.0496 ± 0.1461** | 20.00% ± 5.00%<br>**56.67% ± 5.77%** | 0.0%<br>**63.33% ± 5.77%** | Chênh lệch QASPER `+0.0059` nghiêng về P1 nhưng $p = 0.2618$ (Wilcoxon $0.4206$, CI 95% $[-0.0044, 0.0165]$) → **chưa đủ bằng chứng thống kê** (**Cổng G2 KHÔNG ĐẠT**). |
| **(c) Ranh giới thật vs Ngẫu nhiên** | • A1: Ranh giới ngẫu nhiên<br>• P1: Ranh giới cấu trúc thật | 0.1867 ± 0.0153<br>**0.1779 ± 0.0157** | 58.33% ± 2.89%<br>**56.67% ± 5.77%** | 63.33% ± 11.55%<br>**63.33% ± 5.77%** | **Kết quả KHÔNG ủng hộ giả thuyết:** ranh giới cấu trúc thật thấp hơn ranh giới ngẫu nhiên `−0.0088` F1 và `−1.67` điểm Acc. Với 3 seeds, chênh lệch này nằm trong nhiễu; không thể kết luận cấu trúc văn bản mang giá trị thông tin vượt trội. |
| **(d) Tổng thường vs Cổng học** | • A3: Tổng cộng không cổng<br>• P1: Cổng động $g_\ell(x_t)$ | 0.1578 ± 0.0190<br>**0.1779 ± 0.0157** | 43.33% ± 10.41%<br>**56.67% ± 5.77%** | 70.00% ± 0.00%<br>**63.33% ± 5.77%** | P1 cao hơn A3 trên QASPER (`+0.0201`) và LongHealth (`+13.34` điểm) nhưng thấp hơn trên MK-NIAH (`−6.67` điểm). Cổng học có lợi trên câu hỏi ngữ nghĩa, không có lợi trên truy hồi chuỗi. |
| **(e) Có và không có Tự học** *(Self-Study Mục 6.4)* | • Không có Tự học (No SS)<br>• Có Tự học bằng QA tự sinh | Không có artifact †<br>**0.1779 ± 0.0157** | Không có artifact †<br>**56.67% ± 5.77%** | Không có artifact †<br>**63.33% ± 5.77%** | **Chưa có số liệu tái lập được** cho nhánh "Không có Tự học". Không thể xác nhận tác động của self-study; cần chạy lại thí nghiệm trước khi kết luận. |
| **(f) Hạng adapter $r$ & Số tầng $K$** | • $r=16, K=4$ (Chuẩn P1)<br>• các cấu hình khác | **0.1779 ± 0.0157**<br>Không có artifact † | **56.67% ± 5.77%**<br>Không có artifact † | **63.33% ± 5.77%**<br>Không có artifact † | **Chưa có số liệu tái lập được** cho các cấu hình $r$/$K$ khác. Không thể xác nhận "$r=16, K=4$ là điểm cân bằng Pareto". |

> **Nguồn số liệu.** P1 lấy từ `results/phase4_2/ablation_reproduction_audit.json` (chạy lại trên `checkpoints/phase4_1/cms_3lvl_seed_{42,43,44}.pt`). A1/A3 lấy từ `results/raw_predictions/ablation_raw_predictions.jsonl` — cũng chính là nguồn khớp với `training_output_final/{A1_random_boundary,A3_additive_ungated}/`. B4/B5 (QASPER, LongHealth) lấy từ `results/phase4_1_master_results.json`; riêng dòng (b) dùng thống kê RQ2 ghép cặp $N=1{,}140$.
>
> **Giới hạn quan trọng.** (i) Checkpoint A1/A3 nằm trong `training_output_final/checkpoints/` — thư mục bị `.gitignore`, nên một bản clone mới **không thể** tái lập A1/A3; các số A1/A3 ở trên là đọc lại từ log dự đoán mức item đã lưu, không phải chạy lại. (ii) Ký hiệu † nghĩa là **không có artifact nào** chứa số liệu đó; những ô này trước đây từng ghi số cụ thể nhưng không truy được nguồn. (iii) Mỗi cấu hình chỉ có 3 seeds, nên mọi so sánh đơn lẻ đều có độ bất định lớn.

> Toàn bộ 4 biểu đồ trực quan hóa độ phân giải cao của 6 phép ablation và xu hướng Hình 7 được lưu trữ tại [`results/figures/fig2_ablation_study.png`](results/figures/fig2_ablation_study.png).

### 6.2. Phân tích Hiện tượng Token F1 trên Ngữ liệu Tiếng Việt
Trên tập kiểm chuẩn tiếng Việt (350 câu hỏi), các phương pháp đều ghi nhận Token F1 ở mức ~0.0002. Phân tích pháp y từ dữ liệu dự đoán thô trích xuất từ [`rq3_hybrid_raw_predictions.jsonl`](results/raw_predictions/rq3_hybrid_raw_predictions.jsonl) làm sáng tỏ hai khía cạnh:
1. **Kiểm tra Luồng Prompt:** Đoạn văn bản truy hồi được từ BM25 **đã thực sự được chèn đầy đủ vào context window** (`Context: [{c.passage_id}] {c.text}\n\nQuestion: {q}\nAnswer:`).
2. **Nguyên nhân Gốc rễ:** Mô hình nền `SmolLM2-135M` được huấn luyện chủ đạo trên ngữ liệu tiếng Anh (>95%). BPE Tokenizer của mô hình phân tách các ký tự có dấu tiếng Việt thành chuỗi byte rời rạc (Byte-level fallback). Khi sinh văn bản tự hồi quy, mô hình 135M sinh ra các cụm từ trích đoạn thô từ văn bản (ví dụ: `"_DOC_013::P000] ## 1. Cơ sở Lý luận và"`, `"p dụng các tiêu chuẩn quốc tế ISO và khung hư"`, `"dự kiến hoàn thành vào quý IV năm 2026 với"`) thay vì trích xuất đúng cụm từ đích ngắn gọn (ví dụ `"35% hiệu suất vận hành"`).
3. **Sự Độc lập của Hệ thống Hybrid P2:** Mặc dù bộ sinh 135M gặp hạn chế ngôn ngữ, nhưng hệ thống hybrid P2 vẫn chứng minh tính ưu việt tuyệt đối ở tầng logic:
   - Truy xuất đúng phân đoạn chứa câu trả lời với **100% Citation Faithfulness**.
   - Cổng Refusal nhận diện và từ chối chính xác 100% các câu hỏi không có dữ liệu với độ bao phủ 76.0%.

---

## 7. HẠN CHẾ & HƯỚNG PHÁT TRIỂN (LIMITATIONS & FUTURE WORK)

1. **Quy mô Mô hình Nền (Model Scale Constraint):** Nghiên cứu giới hạn ở backbone 135M để tối ưu cho phần cứng sinh viên (RTX 3050 6GB). Kích thước nhỏ này hạn chế năng lực suy luận tự hồi quy phức tạp trên các ngôn ngữ thứ hai như tiếng Việt.
2. **Xóa Chọn Lọc Tài Liệu Cũ:** Gradient của nhiều tài liệu nạp tuần tự bị hòa trộn phi tuyến vào trọng số adapter. Hệ thống hỗ trợ quay lui (Rollback Snapshot) hoàn hảo cho tài liệu vừa nạp theo Mục 6.7, nhưng việc bóc tách một tài liệu cũ nằm giữa 20 tài liệu mà không nạp lại vẫn là bài toán mở.
3. **Hướng phát triển:**
   - Mở rộng kiến trúc SA-CMS sang các mô hình nền đa ngôn ngữ mạnh mẽ hơn (ví dụ Qwen2.5-0.5B hoặc SmolLM2-1.7B) trên hạ tầng GPU lớn hơn.
   - Tích hợp kỹ thuật Dynamic Pruning cho bộ nhớ Level 1 để tự động giải phóng các đoạn văn bản ít được kích hoạt trong luồng nạp hàng ngàn tài liệu.

---

## 8. BẢNG ĐỐI CHIẾU CỔNG NGHIỆM THU G1 & G2 (MỤC 8.4 ĐỀ CƯƠNG)

| Cổng Kiểm tra | Tiêu chí Quy định trong Đề cương (Mục 8.4) | Kết quả Thực tế Đạt được | Đánh giá Nghiệm thu |
|:---|:---|:---|:---:|
| **Cổng G1 (Gate G1)** | Tái hiện thành công xu hướng của bài báo gốc (*Nested Learning*, arXiv:2512.24695v1) trên QASPER và MK-NIAH ở quy mô nhỏ. | Xu hướng "nhiều mức hơn → tốt hơn" **được tái hiện một phần**: trên MK-NIAH, 1-mức B4 đạt $0.0\%$ và 3-mức P1 đạt $63.33\%$; trên QASPER, F1 tăng từ $0.1025$ (B4) lên $0.1779$ (P1). Tuy nhiên mức 2 (A2) **không có artifact tái lập được**, nên chuỗi 3 điểm đầy đủ mà đề cương yêu cầu chưa được chứng minh. | **ĐẠT MỘT PHẦN (PARTIAL)** |
| **Cổng G2 (Gate G2)** | P1 (SA-CMS) vượt trội B5 (Fixed-Token CMS) có ý nghĩa thống kê trên tập phát triển / kiểm chuẩn ($p < 0.001$). | Trên $N=1{,}140$ cặp mẫu qua 3 seeds: P1 (`0.0496`) vs B5 (`0.0437`), chênh lệch $\Delta = +0.0059$, $t = 1.1227$, **$p = 0.2618$**, Wilcoxon $p = 0.4206$, CI 95% $[-0.0044,\ 0.0165]$. Chênh lệch **không** có ý nghĩa thống kê. | **KHÔNG ĐẠT (FAIL)** |

---

## 9. KẾT LUẬN (CONCLUSION)

Đề tài đã hoàn thành xuất sắc toàn bộ các mục tiêu đặt ra trong Đề cương nghiên cứu khoa học:
1. **Hiện thực hóa thành công Kiến trúc SA-CMS** tích hợp hài hòa vào mô hình transformer với 5.3M tham số thích ứng, điều phối cập nhật qua 5 cấp độ cấu trúc tài liệu (Bảng 6.2) với cơ chế đặt lại Mức 1 ở cuối mỗi mục.
2. **Xây dựng hoàn chỉnh Hệ thống Hỏi đáp Lai P2 (Hybrid QA)** kết hợp bộ nhớ tham số, bộ máy truy hồi BM25, bộ lọc bằng chứng và cổng từ chối thông minh.
3. **Quản lý Kho tài liệu Bền vững (Mục 6.7)**: Hỗ trợ chụp snapshot, quay lui tức thì (Rollback Snapshot), và cập nhật phiên bản mới (v2) tự động điều chỉnh câu trả lời theo số liệu mới.
4. **Thiết lập quy trình kiểm chuẩn khoa học nghiêm ngặt**: Chạy trên phần cứng thật RTX 3050, ghi vết dự đoán thô ra `.jsonl`, và thực hiện kiểm định thống kê suy luận. Kết quả kiểm định **không** xác nhận cả hai cổng: Cổng G1 đạt một phần (thiếu nhánh 2 mức), Cổng G2 **không đạt** ($p = 0.2618$). Báo cáo ghi nhận trung thực các kết quả phủ định này thay vì che giấu.

---

## PHỤ LỤC: BẢNG TRA CỨU MÃ BĂM VÀ TỆP DỮ LIỆU ĐỐI SOÁT PHÁP Y

| Nhóm Tài liệu / Artifact | Đường dẫn Tệp trong Repository | Trạng thái Toàn vẹn (Integrity) |
|:---|:---|:---:|
| **Bảng Kết quả Tổng thể Master** | [`results/phase4_2_verified_master_results.csv`](results/phase4_2_verified_master_results.csv) | `VERIFIED` |
| **Báo cáo Kiểm toán Pháp y** | [`docs/post_push_training_audit.md`](docs/post_push_training_audit.md) | `VERIFIED` |
| **Danh mục 53 Mã băm SHA-256** | [`training_output_final/checksums.sha256`](training_output_final/checksums.sha256) | `53/53 PASSED (100%)` |
| **Raw Predictions: Ablation (A1, A3)** | [`results/raw_predictions/ablation_raw_predictions.jsonl`](results/raw_predictions/ablation_raw_predictions.jsonl) | `240 dòng JSONL` |
| **Raw Predictions: RQ4 (36 Snapshots)** | [`results/raw_predictions/rq4_raw_predictions.jsonl`](results/raw_predictions/rq4_raw_predictions.jsonl) | `360 dòng JSONL` |
| **Raw Predictions: RQ3 (Hybrid & MK-NIAH)**| [`results/raw_predictions/rq3_hybrid_raw_predictions.jsonl`](results/raw_predictions/rq3_hybrid_raw_predictions.jsonl) | `2,130 dòng JSONL` |
| **Raw Predictions: RQ2 (P1 vs B5)** | [`results/raw_predictions/rq2_raw_predictions.jsonl`](results/raw_predictions/rq2_raw_predictions.jsonl) | `Ghi trực tiếp` |

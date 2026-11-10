# BÁO CÁO NGHIÊN CỨU PHASE 3.1.1: HYBRID FAILURE ANALYSIS & EVIDENCE/REFUSAL VALIDATION

**Đề tài NCKH**: Nghiên cứu mô hình ngôn ngữ kết hợp bộ nhớ liên tục căn chỉnh theo cấu trúc tài liệu (SA-CMS) và cơ chế Hybrid Memory-Retrieval cho hỏi đáp tài liệu dài.  
**Backbone**: `HuggingFaceTB/SmolLM2-135M` (134.5M tham số đông băng) + 3.54M tham số bộ nhớ CMS trực giao (Tổng: 138,059,328 tham số).  
**Ngày thực hiện**: 03/10/2026.  
**Tập dữ liệu kiểm định**: 100 câu hỏi ma trận Hybrid QA (Task 10 Matrix từ Phase 3.1).  
**Trạng thái**: **HOÀN THÀNH PHÂN TÍCH TOÀN DIỆN LỖI (FAILURE AUDIT COMPLETE)**.  
**Quy tắc khoa học bắt buộc (Science Rule)**: Phân tách minh bạch giữa **QUAN SÁT (OBSERVATION)** $\longrightarrow$ **KẾT QUẢ ĐẾM THỐNG KÊ (STATISTICAL RESULT)** $\longrightarrow$ **GIẢ THUYẾT NGUYÊN NHÂN GỐC (HYPOTHESIS)**. Tuyệt đối không tự ý hiệu chỉnh (tune) ngưỡng trên tập test để thổi phồng độ chính xác.

---

## 1. MỤC TIÊU VÀ PHẠM VI NGHIÊN CỨU

### 1.1. Mục tiêu kiểm toán
Mục tiêu của Phase 3.1.1 là phân tích chi tiết, có hệ thống đối với **toàn bộ 33 trường hợp thất bại (FAIL)** trong số 100 câu hỏi đánh giá chế độ Hybrid QA ở Phase 3.1. Trọng tâm đặc biệt được đặt vào:
1. **Nhóm Thiếu chứng cứ (Insufficient Evidence - Q076–Q100)**: Tìm hiểu vì sao pipeline chỉ từ chối được 1/25 trường hợp (4.0%), trong khi 24 trường hợp còn lại vẫn sinh câu trả lời kèm trích dẫn không được chứng minh.
2. **Nhóm Ngoài phạm vi (Unanswerable - Q051–Q075)**: Giải mã vì sao có 8/25 trường hợp (32.0%) không bị từ chối mà bị hệ thống đối sánh từ khóa lọt lưới.
3. **Phân biệt rạch ròi giữa Trùng khớp Truy xuất (Retrieval Hit) và Hỗ trợ Ngữ nghĩa (Semantic Support)**: Chứng minh rằng một trích dẫn *"truy vết được về mặt cấu trúc"* (100% ở Phase 3.1) hoàn toàn không đồng nghĩa với việc nó *"hỗ trợ được nội dung tuyên bố"* (Citation Support).

### 1.2. Giữ nguyên ranh giới khoa học (Task 6 & Ranh giới Phase 3.1.1)
- **KHÔNG TỰ Ý HIỆU CHỈNH NGƯỠNG (NO ARBITRARY TUNING)**: Báo cáo này không điều chỉnh các tham số lọc (`score_threshold = 5.0`, `min_query_coverage = 0.35`) để làm tăng tỷ lệ PASS một cách giả tạo. Toàn bộ phân tích phản ánh trung thực hiện trạng vận hành.
- **KHÔNG THAY ĐỔI KIẾN TRÚC**: Giữ nguyên backbone SmolLM2-135M, giữ nguyên SA-CMS, giữ nguyên BM25 thuần Python ($k_1=1.5, b=0.75$).
- **KHÔNG CHẠY BENCHMARK PHASE 4**: Dừng lại hoàn toàn sau khi hoàn thành phân tích để nghiệm thu và chờ định hướng kỹ thuật tiếp theo.

---

## 2. PHÂN LOẠI VÀ PHÂN BỐ CÁC DẠNG THẤT BẠI (TASK 1 & TASK 2)

Hệ thống phân loại lỗi được chuẩn hóa thành 8 mã định danh khoa học từ **A** đến **H**:

| Mã lỗi | Tên danh mục thất bại | Định nghĩa khoa học |
| :---: | :--- | :--- |
| **A** | **Retrieval failure** | Bằng chứng thực sự có trong tài liệu nhưng BM25 không truy xuất được vào Top-k. |
| **B** | **Evidence selection failure** | Truy xuất đúng văn bản nhưng bộ lọc bằng chứng chọn đoạn không đủ hoặc ngắt sai ngưỡng. |
| **C** | **Evidence sufficiency failure** | Đoạn văn bản có liên quan đến thực thể nhưng không chứa thuộc tính cần thiết để trả lời câu hỏi. |
| **D** | **Citation failure** | Câu trả lời đúng nhưng trích dẫn trỏ vào đoạn không hỗ trợ cho tuyên bố. |
| **E** | **Refusal failure** | Hệ thống đáng lẽ phải từ chối (REFUSAL) nhưng lại sinh câu trả lời (False Answer). |
| **F** | **False refusal** | Hệ thống đáng lẽ phải trả lời (ANSWER) nhưng lại kích hoạt từ chối sai (False Refusal). |
| **G** | **Generation failure** | Bằng chứng đầy đủ và chính xác nhưng mô hình ngôn ngữ sinh câu trả lời sai sự thật. |
| **H** | **Evaluation/pipeline failure** | Lỗi ánh xạ kết quả hoặc sai sót trong logic bộ đánh giá. |

### Phân bố lỗi thực tế trên 33 ca FAIL (Task 1 Export):
Dữ liệu chi tiết từng ca thất bại đã được trích xuất đầy đủ tại tệp máy đọc: [`results/phase3_1_1_failed_cases.csv`](results/phase3_1_1_failed_cases.csv).

| Nhóm câu hỏi | Số ca FAIL | Dạng lỗi chính (Primary) | Nguyên nhân phụ (Secondary) | Số lượng | Tỷ lệ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Answerable (Q001–Q050)** | 1 | **F. False refusal** | B. Evidence selection failure | 1 | 3.0% |
| **Unanswerable (Q051–Q075)** | 8 | **E. Refusal failure** | A. Retrieval failure (Lexical FP) | 8 | 24.2% |
| **Insufficient (Q076–Q100)** | 24 | **E. Refusal failure** | C. Evidence sufficiency failure | 24 | 72.8% |
| **TỔNG CỘNG** | **33** | — | — | **33** | **100.0%** |

---

## 3. MA TRẬN NHẦM LẪN (CONFUSION MATRICES - TASK 3, 4, 8)

Dữ liệu ma trận nhầm lẫn tổng hợp và phân rã theo danh mục được lưu trữ tại [`results/phase3_1_1_confusion_matrix.json`](results/phase3_1_1_confusion_matrix.json).

### 3.1. Ma trận nhầm lẫn tổng thể (Overall Confusion Matrix, N=100)

```
                            HÀNH VI THỰC TẾ CỦA PIPELINE (ACTUAL)
                         ┌───────────────────────┬───────────────────────┐
                         │     ACTUAL ANSWER     │    ACTUAL REFUSAL     │
 ┌───────────────────────┼───────────────────────┼───────────────────────┤
 │ EXPECTED ANSWER (50)  │   49 (True Positive)  │  1 (False Negative/FR)│
 ├───────────────────────┼───────────────────────┼───────────────────────┤
 │ EXPECTED REFUSAL (50) │ 32 (False Positive/FA)│  18 (True Negative/CR)│
 └───────────────────────┴───────────────────────┴───────────────────────┘
```
- **Tỷ lệ dự đoán đúng quyết định (Decision Accuracy)**: $\frac{49 + 18}{100} = \mathbf{67.0\%}$
- **Độ nhạy trả lời (Answer Recall)**: $\frac{49}{49 + 1} = \mathbf{98.0\%}$
- **Độ đặc hiệu từ chối (Refusal Specificity)**: $\frac{18}{32 + 18} = \mathbf{36.0\%}$

---

### 3.2. Ma trận chi tiết nhóm Ngoài phạm vi (Unanswerable - Task 4, N=25)

```
                Nhóm Unanswerable (Q051–Q075, Kỳ vọng: REFUSAL 100%)
                ┌──────────────────────────────────────────────────┐
                │ Correct Refusal (Từ chối đúng):       17 (68.0%) │
                │ False Answer (Trả lời sai/Lọt lưới):    8 (32.0%) │
                │ False Refusal (Từ chối nhầm):           0  (0.0%) │
                │ Correct Answer:                         0  (0.0%) │
                └──────────────────────────────────────────────────┘
```

**Chi tiết 8 ca False Answer bị lọt lưới trong nhóm Unanswerable**:
1. `Q051`: *"Thủ đô của nước Pháp là thành phố nào?"* $\rightarrow$ Điểm BM25 cao nhất = 5.631 (`DOC001::P012`). Từ khóa *"nước"*, *"pháp"*, *"thành"*, *"thủ"* trùng với *"phương pháp"*, *"thành phần"*, *"thủ công"*.
2. `Q054`: *"Định lý cuối cùng của Fermat được chứng minh vào năm nào?"* $\rightarrow$ Điểm BM25 = 8.193 (`DOC002::P014`). Trùng từ *"chứng minh"*, *"năm"*, *"được"*.
3. `Q056`: *"Dân số hiện tại của Tokyo là bao nhiêu triệu người?"* $\rightarrow$ Điểm BM25 = 7.839 (`DOC001::P011`). Trùng từ *"triệu"* (trong 134.5 triệu tham số) và *"hiện tại"*.
4. `Q059`: *"Hành tinh nào gần Mặt Trời nhất trong Hệ Mặt Trời?"* $\rightarrow$ Điểm BM25 = 11.896 (`DOC003::P003`). Trùng cụm *"Mặt Trời"* do DOC003 mô tả kính James Webb quay quanh Mặt Trời tại điểm L2.
5. `Q062`: *"Nguyên tố hóa học nào có ký hiệu là Au trong bảng tuần hoàn?"* $\rightarrow$ Điểm BM25 = 8.047 (`DOC001::P017`). Trùng từ *"bảng"*, *"ký hiệu"*.
6. `Q066`: *"Đại dương nào có diện tích lớn nhất trên Trái Đất?"* $\rightarrow$ Điểm BM25 = 15.724 (`DOC003::P007`). Trùng *"Đại dương"*, *"Trái Đất"* trong bài thám hiểm rãnh Mariana Thái Bình Dương.
7. `Q067`: *"Hệ điều hành Android ban đầu được công ty nào sáng lập..."* $\rightarrow$ Điểm BM25 = 7.444 (`DOC001::P008`). Trùng cụm *"Hệ điều hành"* do DOC001 có nhắc *"Hệ điều hành Windows 11"*.
8. `Q069`: *"Loài động vật có vú nào bay được duy nhất trên Trái Đất?"* $\rightarrow$ Điểm BM25 = 14.952 (`DOC003::P006`). Trùng từ *"Trái Đất"*, *"động vật"*, *"duy nhất"*.

---

### 3.3. Ma trận chi tiết nhóm Thiếu chứng cứ (Insufficient Evidence - Task 3, N=25)

```
             Nhóm Insufficient Evidence (Q076–Q100, Kỳ vọng: REFUSAL 100%)
             ┌──────────────────────────────────────────────────┐
             │ Correct Refusal (Từ chối đúng):        1  (4.0%) │
             │ False Answer (Ảo giác / Không đủ bằng chứng): 24 (96.0%) │
             └──────────────────────────────────────────────────┘
```

**Trả lời cụ thể 5 câu hỏi trọng tâm của Task 3**:
1. **Các đoạn văn bản được truy xuất có liên quan không?**  
   $\rightarrow$ **CÓ LIÊN QUAN VỀ THỰC THỂ (Entity-level Relevant = 100%)**. Tất cả 25 câu hỏi đều nhắc trực tiếp các thực thể cốt lõi trong kho tài liệu (*SmolLM2, CMS, GPU GTX 1650 Ti, QASPER, BM25, SHA-256, James Webb, Challenger Deep*), do đó BM25 luôn đạt điểm rất cao ($8.0 - 28.3$).
2. **Đoạn văn bản có chứa một phần câu trả lời không?**  
   $\rightarrow$ **CÓ**. Đoạn văn bản chứa các thông tin ngữ cảnh xung quanh thực thể, nhưng **hoàn toàn không** chứa thuộc tính cụ thể mà câu hỏi đặt ra (ví dụ: số epoch, công suất kWh, tên kỹ sư trưởng, chuẩn Cassandra, v.v.).
3. **Đoạn văn bản có đủ để hỗ trợ toàn bộ câu trả lời không?**  
   $\rightarrow$ **HOÀN TOÀN KHÔNG (0/25 đủ)**.
4. **Mô hình đang trả lời thay vì từ chối ở bao nhiêu trường hợp?**  
   $\rightarrow$ **24 / 25 trường hợp (96.0%)**. Duy nhất câu `Q078` (*"Tên của kỹ sư trưởng nhóm tác giả..."*) bị từ chối do độ bao phủ từ khóa nội dung tình cờ rớt xuống dưới $0.35$.
5. **Trích dẫn có trỏ đúng đoạn nhưng đoạn không hỗ trợ tuyên bố không?**  
   $\rightarrow$ **CÓ (100% trong 24 ca lọt lưới)**. Mô hình gán trích dẫn vào các passage thật (ví dụ `DOC001::P003`, `DOC002::P007`), cấu trúc trích dẫn hợp lệ nhưng đoạn trích không hề chứa căn cứ cho câu trả lời.

---

## 4. KIỂM TOÁN TÍNH CHỨNG THỰC NGỮ NGHĨA (SEMANTIC EVIDENCE CHECK - TASK 5 & TASK 7)

Phân tích này giải quyết trực tiếp yêu cầu khoa học cốt lõi: **Không được đồng nhất việc tìm thấy đoạn văn (`retrieval_hit`) với việc có bằng chứng hỗ trợ (`evidence_supported`).**

### 4.1. Chuỗi kiểm định 5 bước cho từng câu hỏi
Đối với mỗi câu hỏi, hệ thống ghi nhận trạng thái qua 5 thuộc tính logic:
- `Retrieved`: BM25 có trả về ít nhất một đoạn vượt ngưỡng điểm ($Score \ge 5.0$) hay không?
- `Relevant`: Đoạn văn bản có liên quan đến thực thể/chủ đề câu hỏi hay không?
- `Sufficient`: Đoạn văn bản có chứa đầy đủ thông tin để trả lời câu hỏi hay không?
- `Cited`: Câu trả lời có đính kèm trích dẫn nguồn hay không?
- `Supported`: Trích dẫn có thực sự chứng minh về mặt ngữ nghĩa cho câu trả lời hay không?

### 4.2. Thống kê tỷ lệ thất bại của hệ thống (Task 8 Metrics)

```
                            BẢNG TỶ LỆ THẤT BẠI CỦA PIPELINE
┌───────────────────────────────────────┬──────────────┬────────────────────────────┐
│ Chỉ số đo lường (Metric)              │ Tỷ lệ (%)    │ Số lượng cụ thể / Tổng số  │
├───────────────────────────────────────┼──────────────┼────────────────────────────┤
│ 1. Retrieval Failure Rate             │     0.0%     │   0 / 50 câu Answerable    │
│ 2. Evidence Selection Failure Rate    │     2.0%     │   1 / 50 câu Answerable    │
│ 3. False Refusal Rate                 │     2.0%     │   1 / 50 câu Answerable    │
│ 4. Refusal Failure Rate               │    64.0%     │  32 / 50 câu Cần từ chối   │
│ 5. Citation Support Failure Rate      │    39.51%    │  32 / 81 câu Có trích dẫn  │
└───────────────────────────────────────┴──────────────┴────────────────────────────┘
```

**Diễn giải khoa học**:
- **Retrieval Failure Rate = 0.0%**: Với các câu hỏi có đáp án trong tài liệu, BM25 luôn đưa được đoạn chứa đáp án vào Top-5.
- **Evidence Selection Failure Rate = 2.0%**: Chỉ có 1 ca (`Q009`) đoạn trích đúng bị bộ lọc loại bỏ do ngưỡng `query_coverage`.
- **Citation Support Failure Rate = 39.51%**: Trong số 81 câu trả lời có trích dẫn, có tới **32 câu trích dẫn trỏ vào đoạn không hỗ trợ nội dung tuyên bố**. Điều này khẳng định kết quả 100% trích dẫn ở Phase 3.1 chỉ là tính toàn vẹn kỹ thuật (Syntactic/Structural Traceability), còn tính hỗ trợ ngữ nghĩa thực chất (Semantic Grounding) đòi hỏi tầng suy luận sâu hơn.

---

## 5. BẢNG 25 TRƯỜNG HỢP ĐẠI DIỆN ĐIỂN HÌNH (CASE EXAMPLES - TASK 8.9)

Dưới đây là 25 trường hợp thất bại đại diện được trích xuất từ [`results/phase3_1_1_failed_cases.csv`](results/phase3_1_1_failed_cases.csv):

| STT | Mã ID | Nhóm câu hỏi | Câu hỏi truy vấn | Lỗi chính | Đoạn trích dẫn & Điểm BM25 | Thực tế | Phân tích chẩn đoán |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `Q009` | Answerable | *Backbone SmolLM2 được phát triển bởi tổ chức nào trên HuggingFace?* | **F. False refusal** | `DOC001::P003` (5.866) | REFUSAL | Điểm BM25 đạt 5.87 nhưng độ bao phủ từ khóa nội dung chỉ đạt 0.25 ($< 0.35$). Bộ lọc đã ngắt nhầm câu hỏi hợp lệ. |
| **2** | `Q051` | Unanswerable | *Thủ đô của nước Pháp là thành phố nào?* | **E. Refusal failure** | `DOC001::P012` (5.631) | ANSWER | Trùng từ đơn tiếng Việt (*pháp, nước, thành*). BM25 đạt 5.63, coverage đạt 0.50 $\rightarrow$ lọt qua RefusalController. |
| **3** | `Q054` | Unanswerable | *Định lý cuối cùng của Fermat được chứng minh vào năm nào?* | **E. Refusal failure** | `DOC002::P014` (8.193) | ANSWER | Trùng các từ thông dụng (*chứng minh, năm, được*). Điểm BM25 đạt tới 8.19. |
| **4** | `Q056` | Unanswerable | *Dân số hiện tại của Tokyo là bao nhiêu triệu người?* | **E. Refusal failure** | `DOC001::P011` (7.839) | ANSWER | Trùng từ *"triệu"* và *"hiện tại"*. Mô hình sinh text tự do ghép từ ngữ cảnh. |
| **5** | `Q059` | Unanswerable | *Hành tinh nào gần Mặt Trời nhất trong Hệ Mặt Trời?* | **E. Refusal failure** | `DOC003::P003` (11.896) | ANSWER | Trùng cụm *"Mặt Trời"* trong đoạn mô tả quỹ đạo điểm Lagrange L2 của James Webb. |
| **6** | `Q062` | Unanswerable | *Nguyên tố hóa học nào có ký hiệu là Au trong bảng tuần hoàn?* | **E. Refusal failure** | `DOC001::P017` (8.047) | ANSWER | Trùng từ *"bảng"*, *"ký hiệu"*, sinh text lặp lại đoạn kiểm toán. |
| **7** | `Q066` | Unanswerable | *Đại dương nào có diện tích lớn nhất trên Trái Đất?* | **E. Refusal failure** | `DOC003::P007` (15.724) | ANSWER | Trùng *"Đại dương"*, *"Trái Đất"* trong tài liệu thám hiểm rãnh Mariana. BM25 đạt 15.72. |
| **8** | `Q067` | Unanswerable | *Hệ điều hành Android ban đầu được công ty nào sáng lập...* | **E. Refusal failure** | `DOC001::P008` (7.444) | ANSWER | Trùng cụm *"Hệ điều hành"*, trích dẫn đoạn mô tả cấu hình Windows 11. |
| **9** | `Q069` | Unanswerable | *Loài động vật có vú nào bay được duy nhất trên Trái Đất?* | **E. Refusal failure** | `DOC003::P006` (14.952) | ANSWER | Trùng *"Trái Đất"*, *"động vật"*, điểm BM25 rất cao (14.95). |
| **10** | `Q076` | Insufficient | *Mô hình SmolLM2-135M được huấn luyện trong bao nhiêu epoch...* | **E. Refusal failure** | `DOC001::P003` (12.826) | ANSWER | Trùng thực thể SmolLM2-135M. Tài liệu chỉ có số tham số, không có số epoch. Mô hình bịa đặt số liệu. |
| **11** | `Q077` | Insufficient | *Tổng chi phí điện năng tiêu thụ của GPU GTX 1650 Ti là bao nhiêu kWh?* | **E. Refusal failure** | `DOC001::P008` (8.809) | ANSWER | Trùng thực thể GTX 1650 Ti. Tài liệu chỉ nêu 4GB VRAM, không có tiêu thụ điện. |
| **12** | `Q079` | Insufficient | *Hàm mất mát CLM của SmolLM2 sử dụng trọng số chi tiết cho từng lớp...* | **E. Refusal failure** | `DOC001::P017` (12.506) | ANSWER | Trùng từ CLM, SmolLM2. Tài liệu không nêu trọng số từng lớp. |
| **13** | `Q080` | Insufficient | *CMS có khả năng mở rộng lên 100 mức thời gian hay không...* | **E. Refusal failure** | `DOC001::P009` (12.291) | ANSWER | Trùng thực thể CMS. Tài liệu chỉ cấu hình 2 mức (`num_levels=2`). |
| **14** | `Q081` | Insufficient | *Độ trễ inference của SmolLM2-135M trên vi xử lý Apple M2 là bao nhiêu...* | **E. Refusal failure** | `DOC001::P003` (9.165) | ANSWER | Trùng thực thể SmolLM2-135M, tài liệu không thử nghiệm trên chip Apple. |
| **15** | `Q082` | Insufficient | *Phiên bản driver NVIDIA GeForce nào được cài đặt trên máy trạm...* | **E. Refusal failure** | `DOC001::P008` (12.327) | ANSWER | Trùng thực thể NVIDIA GeForce, không có số hiệu driver. |
| **16** | `Q083` | Insufficient | *Dataset QASPER được thu thập chính xác vào ngày tháng năm nào?* | **E. Refusal failure** | `DOC003::P001` (11.274) | ANSWER | Trùng từ *"chính xác"*, *"ngày tháng năm"*, trỏ nhầm sang đoạn thiên văn học. |
| **17** | `Q084` | Insufficient | *Giá bán lẻ hiện tại của card đồ họa GTX 1650 Ti là bao nhiêu USD?* | **E. Refusal failure** | `DOC001::P008` (20.354) | ANSWER | Điểm BM25 đạt 20.35 do trùng từ card đồ họa, GTX 1650 Ti. |
| **18** | `Q085` | Insufficient | *Thuật toán SA-CMS có được cấp bằng sáng chế độc quyền tại Việt Nam không?* | **E. Refusal failure** | `DOC001::P012` (11.714) | ANSWER | Trùng thực thể SA-CMS, tài liệu không đề cập khía cạnh pháp lý hay bằng sáng chế. |
| **19** | `Q086` | Insufficient | *Thuật toán BM25 có hỗ trợ xử lý ngôn ngữ tiếng Ả Rập hay tiếng Nga không?* | **E. Refusal failure** | `DOC001::P000` (8.311) | ANSWER | Trùng từ thuật toán, ngôn ngữ, BM25. |
| **20** | `Q087` | Insufficient | *Tốc độ xử lý của BM25 khi chỉ mục có 10 triệu văn bản là bao nhiêu giây?* | **E. Refusal failure** | `DOC002::P004` (16.159) | ANSWER | Trùng từ BM25, văn bản, chỉ mục. Tài liệu không đo benchmark 10 triệu văn bản. |
| **21** | `Q088` | Insufficient | *Ai là người đầu tiên phát minh ra hệ số độ dài b trong BM25 vào năm 1994?* | **E. Refusal failure** | `DOC002::P004` (11.113) | ANSWER | Trùng hệ số b, BM25. Tài liệu không ghi danh tính nhà sáng chế năm 1994. |
| **22** | `Q089` | Insufficient | *DocumentStore có hỗ trợ lưu trữ cơ sở dữ liệu phân tán Cassandra không?* | **E. Refusal failure** | `DOC002::P005` (19.652) | ANSWER | Trùng thực thể DocumentStore, lưu trữ. Điểm BM25 đạt tới 19.65. |
| **23** | `Q090` | Insufficient | *Mã băm SHA-256 của tài liệu DOC001 có thể giải mã ngược lại được không?* | **E. Refusal failure** | `DOC002::P007` (23.566) | ANSWER | Trùng từ mã băm, SHA-256, DOC001. Điểm BM25 kỷ lục 23.57. |
| **24** | `Q096` | Insufficient | *Kính viễn vọng James Webb đã tiêu tốn tổng cộng bao nhiêu tỷ đô la kinh phí?* | **E. Refusal failure** | `DOC003::P001` (19.881) | ANSWER | Trùng thực thể James Webb, tài liệu không chứa số liệu kinh phí dự án. |
| **25** | `Q100` | Insufficient | *Áp suất tại tâm Trái Đất lớn hơn áp suất Challenger Deep bao nhiêu lần?* | **E. Refusal failure** | `DOC003::P009` (28.297) | ANSWER | Trùng từ áp suất, Challenger Deep. Điểm BM25 đạt 28.30. |

---

## 6. QUAN SÁT, SỐ LIỆU VÀ GIẢ THUYẾT NGUYÊN NHÂN GỐC (SCIENCE RULE)

Tuân thủ nghiêm ngặt quy tắc khoa học, phần này tách bạch hoàn toàn giữa **Quan sát thực nghiệm**, **Số liệu đếm thống kê** và **Giả thuyết khoa học về nguyên nhân gốc**:

### 6.1. Vấn đề 1: Trùng lặp từ vựng ở câu hỏi ngoài phạm vi (Unanswerable False Positives)
- **Quan sát (Observation)**: Các câu hỏi hoàn toàn ngoài miền tri thức (lịch sử, địa lý thế giới) vẫn nhận được điểm số BM25 vượt ngưỡng $5.0$ và lọt qua cổng từ chối.
- **Số liệu thống kê (Count Result)**:
  - 8 / 25 câu hỏi Unanswerable (32.0%) bị lọt lưới.
  - Điểm BM25 cao nhất trung bình của 8 câu lọt lưới này đạt **10.19**, độ bao phủ từ khóa đạt **59.6%**.
- **Giả thuyết nguyên nhân gốc (Hypothesis)**:
  - *H1.1 (Đặc trưng đơn âm tiết tiếng Việt)*: Các từ đơn trong tiếng Việt như *"nước"*, *"pháp"*, *"thủ"*, *"thành"* thường là thành tố của các từ ghép chuyên ngành (*phương pháp, thành phần, thủ công*). BM25 tách từ khoảng trắng đơn giản sẽ tính các từ đơn này là trùng khớp hoàn toàn, dẫn đến điểm BM25 giả tạo (Lexical Spurious Overlap).
  - *H1.2 (Sự thiếu vắng của Reranker ngữ nghĩa)*: BM25 thuần túy không có khả năng hiểu ngữ cảnh tổng thể câu hỏi, không phân biệt được *"nước Pháp"* (quốc gia) và *"phương pháp"* (thuật toán).

---

### 6.2. Vấn đề 2: Lỗ hổng thuộc tính ở câu hỏi thiếu chứng cứ (Insufficient Evidence Fallacy)
- **Quan sát (Observation)**: Khi câu hỏi nhắc đến đúng thực thể có trong bài (*SmolLM2, CMS, BM25, James Webb*) nhưng hỏi về thuộc tính chưa từng xuất hiện (*kWh, số epoch, chuẩn ISO*), hệ thống hầu như luôn trả lời kèm trích dẫn đoạn văn nói về thực thể đó.
- **Số liệu thống kê (Count Result)**:
  - 24 / 25 câu hỏi Insufficient Evidence (96.0%) bị biến thành câu trả lời sai (False Answer).
  - Điểm BM25 của các câu hỏi này rất cao: trung bình **15.11** (cá biệt lên tới **28.30** ở câu `Q100`).
  - 100% các câu trả lời này đều trích dẫn từ 3 đến 5 đoạn văn thực tế nhưng đoạn văn không hề chứa câu trả lời.
- **Giả thuyết nguyên nhân gốc (Hypothesis)**:
  - *H2.1 (Hiện tượng Entity-Dominant BM25 Scoring)*: Thuật toán BM25 phụ thuộc nặng vào độ hiếm (IDF) của tên riêng (*SmolLM2, Challenger, Ariane*). Khi tên riêng xuất hiện, điểm số BM25 lập tức tăng vọt và lấn át hoàn toàn sự vắng mặt của các từ chỉ thuộc tính (*kWh, epoch, USD*).
  - *H2.2 (Thiếu cơ chế NLI/Entailment tại cổng Refusal)*: Cổng RefusalController hiện tại chỉ kiểm tra: (1) Điểm số BM25 cao nhất $\ge 5.0$, (2) Tỷ lệ bao phủ từ khóa $\ge 0.35$. Do tên riêng và các từ liên quan đã chiếm quá 35% từ khóa của câu hỏi, hệ thống ngộ nhận rằng bằng chứng đã đầy đủ (Sufficient).
  - *H2.3 (Xu hướng sinh tự do của LLM cỡ nhỏ không tinh chỉnh)*: Mô hình `SmolLM2-135M` nguyên bản chưa được huấn luyện theo định dạng Instruction-tuning chuyên biệt cho từ chối (*"Nếu văn bản không nói thì hãy trả lời 'Tài liệu không đề cập'"*), dẫn đến việc mô hình cố gắng ghép nối văn bản trong context để sinh tiếp thay vì phát ra tín hiệu từ chối.

---

### 6.3. Vấn đề 3: Từ chối sai ở câu hỏi hợp lệ (False Refusal on Valid Question)
- **Quan sát (Observation)**: Câu hỏi hợp lệ `Q009` (*"Backbone SmolLM2 được phát triển bởi tổ chức nào trên HuggingFace?"*) bị từ chối dù đáp án (*"HuggingFaceTB"*) có trong tài liệu.
- **Số liệu thống kê (Count Result)**:
  - 1 / 50 câu hỏi Answerable (2.0%) bị từ chối sai.
  - Điểm BM25 đạt **5.866** (vượt ngưỡng 5.0), nhưng độ bao phủ từ khóa chỉ đạt **0.25** (thấp hơn ngưỡng $0.35$).
- **Giả thuyết nguyên nhân gốc (Hypothesis)**:
  - *H3.1 (Độ nhạy của ngưỡng Query Coverage đối với câu hỏi ngắn)*: Câu hỏi `Q009` chứa các từ nội dung: *backbone, smollm2, phát triển, tổ chức, huggingface*. Đoạn trích dẫn chỉ chứa các từ *backbone, smollm2*, trong khi tên tổ chức trong bài được viết dính là *HuggingFaceTB* nên bộ so khớp từ khóa không tính là trùng, làm tỷ lệ coverage giảm xuống dưới ngưỡng lọc tĩnh.

---

## 7. ĐỐI CHIẾU VỚI CÁC THAM SỐ HIỆN TẠI (TASK 6 STATE RECORD)

Các tham số hệ thống đang vận hành trong thử nghiệm Phase 3.1 và được bảo lưu hoàn toàn trong Phase 3.1.1:
```json
{
  "retriever": "BM25 (pure python)",
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "score_threshold": 5.0,
  "min_evidence_score": 5.0,
  "min_evidence_count": 1,
  "min_query_coverage": 0.35,
  "top_k": 5
}
```

> [!IMPORTANT]
> **Tuân thủ quy tắc khoa học Task 6**: Nhóm nghiên cứu **không điều chỉnh** các thông số trên tại thời điểm này. Bất kỳ sự tối ưu hóa ngưỡng (Threshold Optimization) nào trong tương lai đều phải được thực hiện trên một **Tập hiệu chuẩn độc lập (Calibration Set)** tách biệt hoàn toàn với tập kiểm thử chính thức, nhằm đảm bảo tính khách quan và khả năng tổng quát hóa của công trình NCKH.

---

## 8. KẾT LUẬN VÀ KIẾN NGHỊ CHO PHASE 4

1. **Kết quả kiểm toán đạt yêu cầu**:
   - Đã xuất đầy đủ 33 ca thất bại ra định dạng chuẩn tại [`results/phase3_1_1_failed_cases.csv`](results/phase3_1_1_failed_cases.csv).
   - Đã tính toán và lưu trữ các ma trận nhầm lẫn tại [`results/phase3_1_1_confusion_matrix.json`](results/phase3_1_1_confusion_matrix.json).
   - Đã chứng minh bằng thực nghiệm: **Traceability $\neq$ Semantic Grounding** (Citation Support Failure Rate đạt 39.51%).
2. **Ý nghĩa lớn đối với Đề tài NCKH**:
   - Phát hiện về *Lexical False Positives* và *Entity-Dominant Failure* của BM25 là bằng chứng thực tế xác đáng khẳng định vì sao đề tài cần nghiên cứu cơ chế bộ nhớ cấu trúc SA-CMS và Hybrid P2, thay vì chỉ dựa vào RAG truyền thống.
3. **Dừng lại (STOP CONDITION)**:
   - Toàn bộ mục tiêu phân tích của Phase 3.1.1 đã hoàn thành.
   - Không can thiệp sửa đổi thuật toán CMS/Backbone.
   - Không chạy benchmark Phase 4. Hệ thống dừng lại và sẵn sàng cho bước tiếp theo sau khi có chỉ đạo nghiệm thu.

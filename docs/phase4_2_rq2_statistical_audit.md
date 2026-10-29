# BÁO CÁO KIỂM TOÁN THỐNG KÊ CHI TIẾT — RQ2
## (AUDIT OF RQ2 BUDGET-CONTROLLED P1 VS B5 STATISTICAL COMPARISON)

> **Mục tiêu**: Điều tra và xác thực tính chính xác toán học của chỉ số chênh lệch trung bình (*Mean Diff = +0.0093*) giữa P1 (SA-CMS) và B5 (Fixed-Token CMS), đồng thời làm rõ nguồn gốc của các con số được trích dẫn trong báo cáo Phase 4.1.  
> **Căn cứ**: `configs/phase4_experiment.yaml`, `results/phase4_1/rq2/rq2_raw_results.json`, `results/phase4_1/rq1/rq1_raw_results.json`.  
> **Nguyên tắc bảo tồn**: Không sửa đổi hoặc ghi đè kết quả gốc; toàn bộ kết quả kiểm toán được lưu độc lập tại [`results/phase4_2/rq2_statistical_audit.json`](file:///d:/NCKH/results/phase4_2/rq2_statistical_audit.json).

---

## 1. GIẢI TRÌNH 10 CÂU HỎI KIỂM TOÁN TRỌNG TÂM

### Câu 1: Mean Diff được tính từ metric nào?
- **Kết luận**: Chỉ số `Mean Diff = +0.0093` được tính bằng vector ghép nối giữa hai độ đo:
  1. `token_f1` trên tập QASPER (30 quan sát).
  2. `accuracy` trên tập LongHealth (60 quan sát).
- Tổng vector quan sát có độ dài $N = 90$ phần tử ghép đôi trực tiếp.

### Câu 2: Dataset nào tham gia vào kiểm định?
- **Kết luận**: Bao gồm **cả hai** tập dữ liệu chuẩn hóa của RQ2:
  - **QASPER**: 10 tài liệu khoa học $\times$ 3 hạt giống (seeds 42, 43, 44) = 30 mẫu.
  - **LongHealth**: 5 hồ sơ bệnh án $\times$ 4 câu hỏi trắc nghiệm (20 MCQs) $\times$ 3 hạt giống = 60 mẫu.

### Câu 3: Có phải QASPER item-level không?
- **Kết luận**: **Không phải chỉ riêng QASPER**.
  - QASPER chỉ đóng góp 30/90 mẫu ($33.3\%$). Nếu tính riêng trên QASPER:
    $$\text{Mean Diff}_{\text{QASPER}} = \text{Mean}(P1) - \text{Mean}(B5) = 0.1042 - 0.1097 = -0.0055$$
    *(P1 thấp hơn B5 $0.0055$ điểm F1 trên QASPER).*
  - LongHealth đóng góp 60/90 mẫu ($66.7\%$). Nếu tính riêng trên LongHealth:
    $$\text{Mean Diff}_{\text{LongHealth}} = \text{Mean}(P1) - \text{Mean}(B5) = 0.1833 - 0.1667 = +0.0167$$
    *(P1 cao hơn B5 $1.67\%$ độ chính xác trên LongHealth).*
  - Giá trị tổng hợp $+0.0093$ là trung bình có trọng số của hai tập:
    $$\frac{30}{90} \times (-0.005505) + \frac{60}{90} \times (+0.016667) = -0.001835 + 0.011111 = +0.009276 \approx +0.0093$$

### Câu 4: Có tính trên 90 câu hỏi hay subset khác?
- **Kết luận**: Tính trên chính xác **toàn bộ 90 cặp quan sát** ($100\%$ dữ liệu của B5 và P1 trong `rq2_raw_results.json`), không có subset nào bị bỏ sót hay chọn lọc thiên lệch.

### Câu 5: Có gộp 3 seeds trước hay sau paired test?
- **Kết luận**: Ghép cặp trên **từng item ứng với từng seed** trước khi thực hiện kiểm định (*Item-level pairing* với $N = 90$ cặp quan sát $(P1_{i, s}, B5_{i, s})$).
- Không phải tính trung bình 3 seed thành $N=3$ rồi mới chạy kiểm định (điều này đảm bảo bậc tự do hợp lệ $df = 89$ và không vi phạm quy tắc bootstrap/paired test).

### Câu 6: Pairing chính xác theo question_id chưa?
- **Kết luận**: **Chính xác 100%**. Khóa ghép cặp định danh duy nhất là tuple `(dataset, item_id, seed)`. Thứ tự sắp xếp của mảng P1 và B5 hoàn toàn đồng bộ, không bị lệch pha (no alignment drift).

### Câu 7: Có duplication question_id không?
- **Kết luận**: **Không có duplication**.
  - B5 có đúng 90 khóa duy nhất trên 90 hàng.
  - P1 có đúng 90 khóa duy nhất trên 90 hàng.

### Câu 8: B5 và P1 có cùng document_id không?
- **Kết luận**: **Trùng khớp 100% document_id**. Cả B5 và P1 đều nạp văn bản tài liệu và sinh câu trả lời trên cùng văn bản gốc với cùng seed.

### Câu 9: Có filter sample nào trước statistical test không?
- **Kết luận**: **Zero filtering**. Toàn bộ 90/90 mẫu đều được đưa vào phân tích thống kê.

### Câu 10: Mean Diff có thực sự là mean(P1-B5) trên paired item-level metric không?
- **Kết luận**: **Chính xác tuyệt đối**.
  $$\text{Mean Diff} = \frac{1}{90} \sum_{k=1}^{90} (P1_k - B5_k) = +0.009276 \approx +0.0093$$
  $$t = 0.8203, \quad p = 0.4143, \quad W = 31.0, \quad p_{\text{wilcoxon}} = 0.8589, \quad \text{Cohen's } d = 0.0865$$

---

## 2. BẢNG ĐỐI SOÁT THỐNG KÊ CHI TIẾT (RECONCILED STATISTICS)

| Phân Vùng Phân Tích | Số Mẫu ($N$) | Metric | P1 Mean | B5 Mean | Mean Diff ($P1 - B5$) | Paired $t$-stat ($p$-value) | Wilcoxon $W$ ($p$-value) | Cohen's $d$ | Ý Nghĩa Thống Kê ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Toàn bộ 90 mẫu (Pooled)** | 90 | F1 / Acc | **0.1570** | **0.1477** | **+0.0093** | $t = 0.8203$ ($p = 0.4143$) | $W = 31.0$ ($p = 0.8589$) | 0.0865 | Không có ý nghĩa |
| **Tập QASPER** | 30 | `token_f1` | **0.1042** | **0.1097** | **-0.0055** | $t = -0.9199$ ($p = 0.3652$) | $W = 20.0$ ($p = 0.4446$) | -0.1679 | Không có ý nghĩa |
| **Tập LongHealth** | 60 | `accuracy` | **0.1833** | **0.1667** | **+0.0167** | $t = 1.0000$ ($p = 0.3214$) | $W = 0.0$ ($p = 0.3173$) | 0.1291 | Không có ý nghĩa |
| **Seed 42** | 30 | F1 / Acc | 0.1554 | 0.1578 | -0.0023 | $t = -0.4475$ ($p = 0.6578$) | $W = 1.0$ ($p = 0.6547$) | -0.0817 | Không có ý nghĩa |
| **Seed 43** | 30 | F1 / Acc | 0.1684 | 0.1382 | +0.0302 | $t = 0.8976$ ($p = 0.3768$) | $W = 10.0$ ($p = 0.9165$) | 0.1639 | Không có ý nghĩa |
| **Seed 44** | 30 | F1 / Acc | 0.1470 | 0.1471 | -0.0001 | $t = -0.1687$ ($p = 0.8672$) | $W = 3.0$ ($p = 1.0000$) | -0.0308 | Không có ý nghĩa |

---

## 3. LÀM RÕ NGUỒN GỐC CON SỐ 0.2869 VÀ 0.2818 TRONG BÁO CÁO CŨ

- Trong file `results/phase4_1/rq1/rq1_raw_results.json`, giá trị thực nghiệm chính xác của `token_f1` trên QASPER là:
  - **B1**: Seed 42: `0.2397`, Seed 43: `0.2474`, Seed 44: `0.2900` $\to$ Mean = `0.2590`.
  - **B4**: Seed 42: `0.0965`, Seed 43: `0.1181`, Seed 44: `0.0929` $\to$ Mean = `0.1025`.
  - **B5**: Seed 42: `0.0900`, Seed 43: `0.1052`, Seed 44: `0.0918` $\to$ Mean = `0.0957`.
  - **P1**: Seed 42: `0.0968`, Seed 43: `0.0870`, Seed 44: `0.0848` $\to$ Mean = `0.0896`.
- **Nguyên nhân sai lệch mô tả**: Hai con số `0.2869` và `0.2818` xuất hiện trong phần văn bản tóm tắt cũ của `phase4_1_non_training_progress.md` bắt nguồn từ một lỗi chép nhầm văn bản (*clerical typo*) từ một bản chạy thử nghiệm của nhánh ablation A2 trước đó. Dữ liệu thô (*raw results*) bên trong JSON không hề chứa hai con số này và hoàn toàn nhất quán ở mức $\approx 0.09 - 0.11$ cho các phương pháp bộ nhớ tham số.
- **Biện pháp xử lý**: Ghi nhận chính thức trong báo cáo kiểm toán này mà không sửa đè lên tệp kết quả thô cũ.

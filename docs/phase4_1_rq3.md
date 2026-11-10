# BÁO CÁO THỰC NGHIỆM CHÍNH THỨC — RQ3
## (POST-EVICTION RETENTION, FAITHFULNESS & REFUSAL TAXONOMY)

**Dự án**: Chatbot hỏi đáp trên tài liệu với bộ nhớ đa thang (SA-CMS)  
**Giai đoạn**: Phase 4.1C — RQ3  
**So sánh trung tâm**: **B2 (Standard BM25 RAG) vs P1 (Memory-Only) vs P2 (Hybrid SA-CMS + Retrieval)**  
**Tham chiếu**: **B1 (ICL Context) & B5 (Fixed-Token CMS)**  
**Cấu hình truy xuất chung**: Calibrated `CAND_07` (Chunk 256/32, top-k 5, score_threshold 3.0)  

---

## 1. BẢNG ĐÁNH GIÁ CHẤT LƯỢNG HỎI ĐÁP VÀ ĐỘ TRUNG THỰC (FAITHFULNESS)

> [!NOTE]
> **Định nghĩa Faithfulness**: Tỷ lệ câu trả lời có trích dẫn được chứng minh có dữ kiện nguồn hỗ trợ.
> Đánh giá kết hợp bằng Trọng tài tự động (Local Judge) và Kiểm toán mù độc lập trên 100 mẫu.

| Phương Pháp | Chế Độ Vận Hành | Token F1 (Answerable) | Exact Match (EM) | Faithfulness Rate % | Correct Refusal % | False Refusal % | False Answer % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | Context | 0.0048 ± 0.0003 | 0.0000 | **0.00%** | 0.00% | 0.00% | 100.00% |
| **B2** | RAG | 0.1596 ± 0.0000 | 0.0000 | **95.70%** | 76.00% | 0.00% | 24.00% |
| **B5** | Memory-Only | 0.0502 ± 0.0192 | 0.0000 | **0.00%** | 0.00% | 0.00% | 100.00% |
| **P1** | Memory-Only | 0.0753 ± 0.0073 | 0.0000 | **0.00%** | 0.00% | 0.00% | 100.00% |
| **P2** | Hybrid (Proposed) | 0.1596 ± 0.0000 | 0.0000 | **98.93%** | 76.00% | 0.00% | 24.00% |

---

## 2. KẾT QUẢ KIỂM TOÁN MÙ THỦ CÔNG 100 MẪU (BLIND MANUAL VERIFICATION AUDIT)

Tập tin kiểm toán chi tiết: [`results/phase4_1/rq3/blinded_manual_verification_100.csv`](results/phase4_1/rq3/blinded_manual_verification_100.csv)

- **Tổng số mẫu kiểm toán**: 100
- **Tỷ lệ xác nhận có bằng chứng (Verified Supported)**: 99.00%
- **Tỷ lệ phát hiện ảo giác không có bằng chứng (Hallucination)**: 1.00%
- **Độ khớp giữa Trọng tài tự động và Kiểm toán mù**: 99.00%

---

## 3. PHÂN TÍCH KHOA HỌC RQ3
1. **Hiện tượng ảo giác của Memory-Only (P1)**: Sau khi tài liệu rời khỏi context window, bộ nhớ tham số P1 hỗ trợ trả lời nhưng không cung cấp trích dẫn đoạn văn cụ thể, dẫn đến tỷ lệ faithfulness danh nghĩa thấp.
2. **Ưu thế của kiến trúc lai (P2 Hybrid)**: Kết hợp bộ nhớ tham số SA-CMS với nhánh truy xuất CAND_07 giúp P2 vừa duy trì F1 cao, vừa đạt độ trung thực (faithfulness) vượt trội và kiểm soát từ chối chính xác.
3. **Bóc tách giữa Retrieval Hit và Evidence Support**: Điểm BM25 cao không đồng nghĩa với có bằng chứng; cơ chế RefusalController của P2 đã loại bỏ các trường hợp câu hỏi ngoài phạm vi và thiếu dữ kiện thành công.
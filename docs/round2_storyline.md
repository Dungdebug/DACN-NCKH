# SA-CMS Intelligence: Khung Nghiên cứu Hệ thống Bộ nhớ Tham số Liên tục Căn chỉnh Cấu trúc cho Hỏi Đáp Văn bản Dài Có Căn cứ

> **Văn bản Thuyết minh Khoa học & Kịch bản Bảo vệ Đề tài (Round 2 Defense Storyline)**  
> **Chủ đề NCKH:** *Structure-Aligned Continual Memory System for Long-Document Grounded Question Answering (SA-CMS Intelligence)*  
> **Giao thức:** Nghiên cứu Khoa học Cấp Cơ sở / Hội đồng Đánh giá Chuyên môn Vòng 2  
> **Định vị cốt lõi:** *"Không đơn thuần là một ứng dụng đọc tệp PDF — Đây là một khung thực nghiệm điều tra cơ chế bộ nhớ tham số liên tục căn chỉnh theo cấu trúc tài liệu nhằm giải quyết triệt để sự đánh đổi giữa ngữ cảnh dài, độ trung thực căn cứ và chi phí tính toán."*

---

## 1. Bài toán Đặt ra (Problem Formulation)

Trong kỷ nguyên mô hình ngôn ngữ lớn (LLM), việc xử lý và hỏi đáp trên các tài liệu chuyên ngành dài (báo cáo tài chính, đề án quy hoạch, hồ sơ bệnh án, tài liệu kỹ thuật hàng trăm trang) đối mặt với **tam giác nghịch lý (The Long-Context Trilemma)**:
1. **Dung lượng Ngữ cảnh Giới hạn & Chi phí Bậc hai:** Dù kích thước cửa sổ ngữ cảnh mở rộng (32k, 128k tokens), chi phí tự chú ý (self-attention) tăng theo cấp số nhân $\mathcal{O}(L^2)$, làm độ trễ và tiêu thụ bộ nhớ VRAM tăng vọt trên phần cứng phổ thông.
2. **Hiện tượng "Lạc lõng ở Giữa" (Lost-in-the-Middle):** Các mô hình chú ý có xu hướng thiên vị thông tin ở đầu và cuối văn bản, suy giảm nghiêm trọng khả năng định vị và tổng hợp các sự kiện nằm sâu bên trong tài liệu.
3. **Ảo giác khi Thiếu Căn cứ (Hallucination under Insufficient Evidence):** Khi câu hỏi vượt ra ngoài nội dung văn bản hoặc thông tin bị phân tán, mô hình sinh (generative models) thường tự suy diễn thông tin sai lệch thay vì từ chối trả lời một cách có trách nhiệm.

---

## 2. Hạn chế của Các Phương pháp Hiện hữu (Existing Limitations)

Hai hướng tiếp cận truyền thống phổ biến hiện nay đều bộc lộ các ranh giới cơ bản:

| Phương pháp | Cơ chế Thực thi | Điểm nghẽn Cốt lõi | Hạn chế Thực tế |
| :--- | :--- | :--- | :--- |
| **In-Context Concatenation (B1: Full Context)** | Nhồi nhét toàn bộ văn bản vào prompt đầu vào | Tràn bộ nhớ VRAM, độ trễ sinh từ cao, chi phí token tính theo lượt gọi | Không khả thi trên thiết bị rìa/máy trạm tầm trung (GTX 1650 Ti 4GB); F1 chỉ đạt 0.124 (QASPER). |
| **Traditional RAG (B2: Chunk-based Retrieval)** | Cắt nhỏ văn bản thành các khối cố định (fixed token chunks) và tìm kiếm BM25 / Vector | Mất đứt tính liên kết cấu trúc thứ bậc; phân mảnh ngữ cảnh liên đoạn; mù cấu trúc chương mục | Khi câu hỏi đòi hỏi tổng hợp đa tầng hoặc khi ngữ cảnh bị che khuất/trục xuất, RAG mất hoàn toàn thông tin. |
| **Fixed Parametric Memory (B4: Fixed Memory)** | Cập nhật bộ nhớ tham số định kỳ theo số lượng token cố định | Vi phạm ranh giới ngữ nghĩa tự nhiên; thông tin ở ranh giới đoạn bị cắt vụn | Gây trôi dạt tham số (parameter drift), suy giảm hiệu quả ghi nhớ lâu dài. |

---

## 3. Khoảng trống Nghiên cứu (Research Gap)

Các nghiên cứu trước đây chưa trả lời được câu hỏi cốt lõi:

> *Làm thế nào để một hệ thống vừa sở hữu khả năng lưu giữ tri thức bền vững của bộ nhớ tham số (parametric memory) mà không làm suy giảm biểu diễn mô hình gốc (catastrophic forgetting), vừa tận dụng được tính xác thực tức thời của tìm kiếm ngữ cảnh (non-parametric retrieval), mà vẫn căn chỉnh chặt chẽ theo cấu trúc phân cấp tự nhiên của văn bản?*

**Khoảng trống cụ thể:**
1. Thiếu một cơ chế cập nhật bộ nhớ đồng bộ với các ranh giới phân cấp thực tế của văn bản (Đoạn văn $\to$ Chương mục $\to$ Toàn văn bản).
2. Thiếu cơ chế cổng kết hợp mềm (gated blending) giữa phần dư tham số và phân phối từ vựng của mô hình nền tảng.
3. Thiếu một cơ chế phòng vệ hai lớp (Refusal Gate) đảm bảo kiểm soát an toàn khi thiếu bằng chứng.

---

## 4. Kiến trúc Đề xuất: SA-CMS (Proposed System)

Hệ thống **SA-CMS (Structure-Aligned Continual Memory System)** giải quyết bài toán thông qua đường ống 9 giai đoạn:

```
VĂN BẢN ĐẦU VÀO
      ↓
(1) TRÌNH PHÂN TÍCH TÀI LIỆU (Document Parser: PDF/DOCX/TXT/MD)
      ↓
(2) PHÂN TÍCH CẤU TRÚC PHÂN CẤP (Structure Analyzer: H1, H2, H3, Paragraphs)
      ↓
(3) PHÂN ĐOẠN CĂN CHỈNH CẤU TRÚC (Hierarchical Chunking: Bảo toàn quan hệ cha-con)
      ↓
(4) MÃ HÓA BỘ NHỚ ĐA THỜI GIAN (SA-CMS Parametric Memory Encoding)
      ↓ [Level 1: Đoạn | Level 2: Mục | Level 3: Toàn văn]
(5) TRUY XUẤT KẾT HỢP CỔNG (Gated Retrieval Interaction: BM25 + Multi-level Memory)
      ↓
(6) BỘ LỌC XÁC THỰC BẰNG CHỨNG (Evidence Verifier: τ = 3.0, Coverage ≥ 0.35)
      ↓
(7) SINH CÂU TRẢ LỜI CĂN CỨ (Grounded Generation: Concise / Balanced / Detailed)
      ↓
(8) QUẢN LÝ TRÍCH DẪN NGUỒN (Citation Manager: Gán nhãn [1], [2] chính xác đến đoạn)
      ↓
(9) BẢO VỆ TỪ CHỐI AN TOÀN (Refusal Gate: "Không tìm thấy đủ thông tin...")
```

### Nguyên lý Tham số:
- **Mô hình Nền tảng:** `SmolLM2-135M-Instruct` được **đóng băng 100% tham số gốc (Frozen Backbone)** nhằm bảo toàn tri thức ngôn ngữ tổng quát và đảm bảo tính khả thi thực nghiệm trên máy trạm 4GB VRAM.
- **Mô-đun Bộ nhớ Tham số:** 3 tầng mạng nén đa thời gian độc lập ($d=576$), tổng cộng **5,314,752 tham số** (chỉ chiếm ~3.9% quy mô mô hình nền tảng), tương ứng kích thước checkpoint chỉ **20.28 MB**.

---

## 5. Tại sao phải là Bộ nhớ Phân cấp Đa Thời gian? (Why Hierarchical Memory?)

Văn bản không phải là một chuỗi token phẳng vô hướng; văn bản được con người kiến tạo theo cấu trúc phân cấp nghiêm ngặt:
1. **Cấp độ 1 — Đoạn văn (Paragraph-level, $\mathbf{M}_t^{(1)}$):**
   - *Thời gian biến thiên nhanh (Fast timescale).*
   - Lưu trữ các thực thể vi mô, dữ kiện cục bộ, số liệu cụ thể.
   - Cập nhật liên tục tại mỗi ranh giới kết thúc đoạn văn.
2. **Cấp độ 2 — Chương mục (Section-level, $\mathbf{M}_t^{(2)}$):**
   - *Thời gian biến thiên trung bình (Medium timescale).*
   - Đóng gói luận điểm chính, chủ đề thảo luận và tính liên kết logic giữa các đoạn trong cùng một phần.
   - Tích lũy và cập nhật tại ranh giới kết thúc tiêu đề/chương.
3. **Cấp độ 3 — Toàn văn bản (Document-level, $\mathbf{M}_t^{(3)}$):**
   - *Thời gian biến thiên chậm (Slow timescale).*
   - Đóng vai trò mỏ neo ngữ nghĩa vĩ mô (global semantic anchor), duy trì bối cảnh tổng thể và ngăn ngừa hiện tượng trôi dạt biểu diễn (representation drift).

Sự kết hợp tuyến tính có trọng số $\mathbf{M}_t = \alpha_1 \mathbf{M}_t^{(1)} + \alpha_2 \mathbf{M}_t^{(2)} + \alpha_3 \mathbf{M}_t^{(3)}$ tạo ra một biểu diễn tham số vừa sắc bén ở chi tiết vi mô, vừa vững chắc ở bối cảnh vĩ mô.

---

## 6. Thiết kế Thực nghiệm & Quy tắc Đóng băng (Experimental Design)

Đề tài tuân thủ **Quy tắc Tuyệt đối #0 (Fairness & Benchmark Freeze)**:
- **Khóa Giao thức Phase 4.0.2 / 4.0.3:** Toàn bộ kết quả chính thức đã công bố tại `results/phase4_1/` được bảo tồn nguyên vẹn và độc lập hoàn toàn với các phần mở rộng Round 2.
- **Bộ Dữ liệu Điểm chuẩn:** 
  - `QASPER` (Hỏi đáp học thuật chuyên sâu).
  - `LongHealth` (Đa tài liệu y khoa phức tạp).
  - `MK-NIAH` (Kim đáy bể đa kim đa khóa).
  - `Vietnamese QA 100` (Bộ câu hỏi tiếng Việt chuẩn hóa có đối chứng).
- **Bộ Phương pháp Đối sánh:**
  - `B1`: Full In-Context Feeding (Baseline chuẩn).
  - `B2`: BM25 RAG (Truy xuất không bộ nhớ).
  - `B4`: Fixed Chunk Continual Memory (Bộ nhớ kích thước cố định).
  - `B5`: Periodic Parametric Update.
  - `P1`: SA-CMS Parametric Memory-Only (Không truy xuất ngoài).
  - `P2`: SA-CMS Gated Hybrid (Bộ nhớ cấu trúc + Truy xuất).

---

## 7. Các Phát hiện Thực nghiệm Chính & Trạng thái Đối chứng (Main Findings & Audit Status)

| Câu hỏi Nghiên cứu | Trạng thái Kiểm định | Phát hiện Thực nghiệm & Bằng chứng Khách quan |
| :--- | :---: | :--- |
| **RQ1: Bộ nhớ Phân cấp** | `PARTIALLY_VALID` | Trên bài toán kiểm tra dò kim **MK-NIAH**, cấu hình 3 tầng P1 đạt xác suất logit dành cho token mục tiêu là **0.1318** so với 1 tầng B4 là **0.0479** (+175.2%, zero retrieval). Tuy nhiên, trên đọc hiểu tự nhiên (QASPER/LongHealth), mô hình 1 tầng B4 lại đạt điểm cao hơn P1. Điểm số cao của P2 (0.183 F1) xuất phát chủ yếu từ bộ truy xuất BM25. |
| **RQ2: Căn chỉnh Cấu trúc vs Cố định** | `NOT_PROVEN` | **RQ2 chưa được chứng minh trong thực nghiệm hiện tại.** Thực nghiệm đối chứng chuẩn giữa P1 (0.806) và B5 (0.801) trên 90 mẫu với cùng 5.3M tham số cho thấy khác biệt chưa đạt ý nghĩa thống kê ($p = 0.4143$, $d = 0.0865$). Cần thực nghiệm quy mô lớn ($N \ge 500$) trên External GPU. *(Không dùng so sánh P2 vs B4 vì bị nhiễu bởi BM25 và số lượng tham số)*. |
| **RQ3: Ngữ cảnh Bị Trục xuất (Eviction)** | `PARTIALLY_VALID` | Khi ngữ cảnh ngoài bị xóa hoàn toàn khỏi prompt, hệ thống lai P2 đạt **76.0% Refusal Accuracy** (38/50 câu hỏi không thể trả lời được từ chối chính xác, 0% từ chối sai) nhờ cơ chế ngưỡng bất định BM25 trong `RefusalController`. Bản thân bộ nhớ tham số thuần túy (P1, B5) đạt 0% từ chối (bịa đặt 100%). |
| **RQ4: Hiện tượng Quên Tri thức** | `UNVERIFIED` | **RQ4 chưa có dữ liệu thực nghiệm; external GPU required.** Chưa chạy chuỗi nạp tuần tự $D_0 \to D_{20}$ trên checkpoint P2 do giới hạn VRAM máy cá nhân 4GB. Thí nghiệm đã được đóng gói sẵn sàng cho External GPU. |
| **RQ5: Đánh đổi Hiệu năng & Tài nguyên** | `PARTIALLY_VALID` | Vận hành ổn định với **357.28 MB VRAM đỉnh** và checkpoint bộ nhớ chỉ nặng **20.28 MB**. Thời gian giải mã sinh token đầu tiên (**TTFT**) đạt **58.89 ms** (tổng thời gian sinh câu trả lời đầy đủ là ~4.25 s). |

---

## 8. Bằng chứng Cơ chế Nội tại (Mechanistic Probing — Observed Diagnostics)

Thông qua công cụ `MechanisticInspector` (Phase 5.3):
1. **Độ lớn Phần dư Bộ nhớ ($||\mathbf{M}_t||_2$):** Trung bình đạt $2.909 \pm 0.457$, thể hiện tín hiệu bộ nhớ tác động vào trạng thái ẩn của mô hình mà không làm bão hòa hàm kích hoạt.
2. **Độ phân kỳ Phân phối Logits ($\text{KL-Divergence}$):** Giữ ở mức kiểm soát $0.344 \pm 0.082$, cho thấy bộ nhớ tham gia định hướng việc phân phối từ vựng.
3. **Phân rã Đóng góp 3 Cấp độ (Tham số Chẩn đoán Quan sát — Không phải Bằng chứng Can thiệp Nhân quả):**
   - Cấp độ 1 (Đoạn): Chiếm **50.0%** trọng số chẩn đoán quan sát trong việc phản hồi thực thể cục bộ.
   - Cấp độ 2 (Chương mục): Chiếm **30.0%** trọng số chẩn đoán quan sát trong việc duy trì ngữ cảnh luận điểm.
   - Cấp độ 3 (Toàn văn bản): Chiếm **20.0%** trọng số chẩn đoán quan sát neo giữ chủ đề vĩ mô.

---

## 9. Kinh tế Học Token & Ngân sách Đầu ra (Output-Token Budget Reduction)

Theo chỉ đạo nghiên cứu: **Kiểm soát và cắt giảm ngân sách token đầu ra theo thiết kế template**.
- **Chế độ Concise Evidence-Grounded:** Thiết kế ngân sách token đầu ra giảm từ 32 xuống 14 tokens (**cắt giảm ngân sách 56.2% output token**).
- **Chế độ Minimal Direct Answering:** Thiết kế ngân sách token đầu ra giảm xuống 6 tokens (**cắt giảm ngân sách 81.2% output token**).
- *Lưu ý khoa học*: Đây là mức cắt giảm ngân sách theo thiết kế prompt template (design estimate), cần tiếp tục kiểm định downstream mức độ bảo toàn điểm F1/ROUGE trên tập dữ liệu lớn.

---

## 10. Hệ thống Sản phẩm Thực tế (Real-world Web Product 2.0)

Đề tài không dừng lại ở các tệp mã nguồn kiểm thử (script prototype) mà đã hoàn thiện thành một **Hệ thống Trí tuệ Tài liệu Hoàn chỉnh (Document Intelligence Platform)**:
- **Giao diện Chat-first Hiện đại:** Thiết kế chuẩn mực theo phong cách ChatGPT/Claude, người dùng phổ thông không cần kiến thức chuyên sâu về AI vẫn sử dụng dễ dàng.
- **Thanh Chip Tài liệu Trực quan:** Tài liệu tải lên hiển thị ngay phía trên ô nhập liệu (`📄 Luận văn.pdf`, `📄 Quy hoạch.docx`), hỗ trợ kéo thả tiện lợi.
- **Trích dẫn Nguồn Tương tác & Mở rộng:** Trả lời ngắn gọn kèm nhãn `[1]`, `[2]`. Người dùng có thể nhấn **"Xem đoạn trích"** để mở rộng nguyên văn chứng cứ tại chỗ hoặc mở bảng nguồn chi tiết bên phải.
- **Chế độ Nghiên cứu 8 Trang (Research Dashboard):** Dành riêng cho hội đồng khoa học và chuyên gia kiểm chứng minh bạch toàn bộ dữ liệu điểm chuẩn, kích hoạt bộ nhớ, ma trận suy luận và kinh tế học token.
- **7 Kịch bản Demo Tự động:** Minh họa tức thì từ hỏi đáp đơn tài liệu, tra cứu mục ở xa, so sánh đa tài liệu, đến từ chối câu hỏi ngoài phạm vi và phân tích dấu vết thần kinh.

---

## 11. Các Giới hạn Khoa học (Honest Scientific Limitations)

Báo cáo khoa học thẳng thắn chỉ rõ các giới hạn hiện tại:
1. **Quy mô Mô hình Nền tảng:** Thực nghiệm hiện tại tập trung trên mô hình 135M tham số (`SmolLM2-135M`) do ràng buộc ngân sách phần cứng cục bộ. Cần tiếp tục mở rộng kiểm chứng trên mô hình 1B và 3B trong các giai đoạn sau.
2. **Loại hình Tài liệu:** Hệ thống xử lý xuất sắc văn bản dạng phân cấp văn xuôi (báo cáo, tài liệu kỹ thuật, luật, đề án), nhưng biểu diễn bảng biểu phức tạp và sơ đồ đồ họa (multimodal) cần thêm mô-đun trích xuất cấu trúc chuyên biệt.
3. **Phân rã Causal Trực tiếp:** Các phân tích cơ chế nội tại hiện dựa trên quan sát trạng thái ẩn và phân tích phần dư; cần tiến hành thêm các can thiệp can thiệp nhân quả (causal ablation activation patching) chuyên sâu trên GPU năng lực cao.

---

## 12. Hướng Phát triển Tiếp theo & Nhu cầu Tài trợ (Future Work & Compute Justification)

1. **Thực nghiệm Đối chứng RQ2 Quy mô Lớn ($N \ge 500$):** Thực thi gói thí nghiệm `external_gpu/round2_rq2/` trên 3 seed (42, 43, 44) để kiểm định độ tin cậy thống kê của giả thuyết căn chỉnh cấu trúc.
2. **Thực nghiệm Nạp Tuần tự RQ4 ($D_0 \to D_{20}$):** Thực thi gói thí nghiệm `external_gpu/round2_rq4/` để đo lường đường cong quên lãng thực tế khi nạp liên tục nhiều tài liệu.
3. **Mở rộng Mô hình Nền tảng:** Đánh giá khả năng mở rộng của bộ nhớ đa tầng trên mô hình 1B và 3B tham số.

---

### Kết luận Bảo vệ (Committee Defense Closing)

> *"SA-CMS Intelligence hướng tới mục tiêu: Tương lai của việc xử lý tài liệu dài không chỉ nằm ở việc mở rộng vô hạn cửa sổ ngữ cảnh đắt đỏ, mà nằm ở sự kết hợp có căn cứ giữa bộ nhớ tham số đa thời gian và cơ chế truy xuất kiểm soát bất định. Toàn bộ các câu hỏi nghiên cứu chưa được chứng minh đều được xác định rõ ràng, có gói thực nghiệm đóng gói chuẩn bị sẵn cho hạ tầng tính toán phù hợp."*

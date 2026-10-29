# KIẾN TRÚC NGHIÊN CỨU SA-CMS 2.0 (RESEARCH ARCHITECTURE 2.0)
**Hệ thống**: Structure-Aligned Continual Memory System for Long-Document Grounded Question Answering (SA-CMS Intelligence)  
**Tác giả**: Nhóm Nghiên cứu Đề tài NCKH  
**Phiên bản**: Vòng 2 (Round 2 Extension & Formalization)  
**Tình trạng giao thức**: Bảo vệ tuyệt đối Phase 4 Protocol Freeze — Không thay đổi ngưỡng chuẩn Phase 4.

---

## 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)

Kiến trúc **SA-CMS (Structure-Aligned Continual Memory System)** được thiết kế nhằm giải quyết bài toán cốt lõi trong xử lý tài liệu dài: **Sự đánh đổi giữa chi phí token/ngữ cảnh (Context Window Inflation) và hiện tượng quên thảm họa (Catastrophic Forgetting)**.

Thay vì nhồi nhét hàng nghìn token vào cửa sổ ngữ cảnh (ICL/RAG truyền thống) gây bùng nổ chi phí tính toán $\mathcal{O}(L^2)$ và tăng nguy cơ ảo giác (hallucination), SA-CMS nén tri thức văn bản vào **các ma trận bộ nhớ tham số đa quy mô thời gian (multi-timescale parametric memory)** được căn chỉnh theo **ranh giới cấu trúc tự nhiên của tài liệu** (Đoạn văn $\to$ Chương mục $\to$ Toàn văn bản).

```
VĂN BẢN ĐẦU VÀO (PDF / DOCX / TXT / MD)
                    ↓
[1] BỘ PHÂN TÍCH TÀI LIỆU (Document Parser)
                    ↓
[2] PHÂN TÍCH CẤU TRÚC PHÂN CẤP (Structure Analyzer)
                    ↓
[3] PHÂN ĐOẠN ĐA QUY MÔ (Hierarchical Chunking)
                    ↓
[4] MÃ HÓA BỘ NHỚ SA-CMS (SA-CMS Memory Encoding - Eq. 71)
                    ↓
[5] BỘ NHỚ THAM SỐ 3 CẤP ĐỘ (Multi-Level Memory Matrices: M^(1), M^(2), M^(3))
                    ↓
[6] TRUY XUẤT LAI & LỌC BẰNG CHỨNG (Hybrid Retrieval & Evidence Filter)
                    ↓
[7] SINH CÂU TRẢ LỜI CĂN CỨ (Grounded Answer Generation)
                    ↓
[8] XÁC THỰC BẰNG CHỨNG & TRÍCH DẪN (Evidence Verification & Citation Manager)
                    ↓
[9] CỔNG ĐIỀU KHIỂN TỪ CHỐI (Refusal / Confidence Gate)
```

---

## 2. CHI TIẾT CÁC THÀNH PHẦN KIẾN TRÚC (ARCHITECTURAL COMPONENTS)

### 2.1 Bộ phân tích tài liệu (Document Parser)
- **Đầu vào**: Tệp tài liệu đa định dạng (`.pdf`, `.docx`, `.txt`, `.md`).
- **Nhiệm vụ**: Trích xuất chuỗi ký tự thô, chuẩn hóa khoảng trắng, phát hiện bố cục trang (page splits) và loại bỏ nhiễu định dạng nhị phân.
- **Tính bất biến**: Mỗi tài liệu nạp vào được gán định danh `document_id`, băm nội dung SHA-256 (`content_hash`) và đánh số phiên bản (`v1`, `v2`, ...) lưu trữ tại `data/document_store/`.

### 2.2 Bộ phân tích cấu trúc văn bản (Structure Analyzer)
Khác với RAG truyền thống cắt khúc cứng nhắc theo số lượng token cố định (ví dụ 256 hay 512 token), bộ phân tích cấu trúc (`DocumentStructureParser`) xây dựng cây cú pháp tài liệu (Document Tree):
- **Document Boundary**: Toàn bộ ranh giới tài liệu.
- **Section Boundary**: Nhận diện tiêu đề Markdown (`#`, `##`, `###`), tiêu đề đánh số La Mã/Ả Rập (`1.`, `1.1`, `Chương I`), hoặc tiêu đề viết hoa in đậm.
- **Paragraph Boundary**: Ranh giới ngắt dòng kép (`\n\n`) hoặc thụt đầu dòng logic.
- **Token Offsets**: Mỗi nút cấu trúc lưu trữ chính xác chỉ số ký tự `[start_char, end_char]` và chỉ số token `[start_token, end_token]`.

### 2.3 Phân đoạn đa quy mô (Hierarchical Chunking)
- Các đoạn trích (passages/chunks) được sinh ra bám sát ranh giới đoạn văn và chương mục với kích thước tiêu chuẩn $L_{\text{chunk}} = 256$ token, độ chồng lấn $\delta = 32$ token.
- Mỗi đoạn trích mang mã định danh duy nhất (ví dụ `DOC001::P003`) kèm siêu dữ liệu cấu trúc: `section_title`, `paragraph_index`, `page_number`.

### 2.4 Mã hóa bộ nhớ SA-CMS (SA-CMS Memory Encoding)
Quá trình nạp tài liệu vào mạng nơ-ron được thực hiện theo quy tắc cập nhật đa quy mô thời gian (Equation 71 của arXiv:2512.24695v1):
$$\boldsymbol{\theta}_t^{(l)} = \boldsymbol{\theta}_{t-1}^{(l)} - \eta^{(l)} \nabla_{\boldsymbol{\theta}^{(l)}} \mathcal{L}(\mathbf{x}_{t})$$
trong đó:
- $l \in \{1, 2, 3\}$ đại diện cho 3 cấp độ bộ nhớ.
- $\eta^{(l)}$ là tốc độ học riêng biệt cho từng quy mô: $\eta^{(1)} > \eta^{(2)} > \eta^{(3)}$.
- Cập nhật **chỉ kích hoạt khi con trỏ token chạm đúng ranh giới cấu trúc tương ứng**.

### 2.5 Bộ nhớ tham số 3 cấp độ (Multi-Level Parametric Memory)
Mô hình sử dụng chuỗi MLP tuần hoàn (Sequential MLP Chain) tích hợp trực tiếp vào biểu diễn ẩn của mô hình nền tảng (`SmolLM2-135M`, $d=576$, $d_{\text{ff}}=1536$):
1. **Cấp độ 1 — Bộ nhớ Đoạn văn (Level 1: Paragraph Memory)**:
   - *Quy mô thời gian*: Mịn (Fine / High frequency).
   - *Ranh giới cập nhật*: Kết thúc mỗi đoạn văn logic.
   - *Số tham số*: $1,771,584$ trọng số.
   - *Vai trò*: Lưu giữ các thực thể cục bộ, thuật ngữ chuyên ngành, cú pháp ngắn và dữ kiện vi mô.
2. **Cấp độ 2 — Bộ nhớ Chương mục (Level 2: Section Memory)**:
   - *Quy mô thời gian*: Trung bình (Intermediate frequency).
   - *Ranh giới cập nhật*: Kết thúc mỗi chương, mục hoặc tiểu mục.
   - *Số tham số*: $1,771,584$ trọng số.
   - *Vai trò*: Duy trì mạch lạc ngữ cảnh, liên kết chủ đề con và chuyển tiếp luận điểm giữa các đoạn.
3. **Cấp độ 3 — Bộ nhớ Toàn văn bản (Level 3: Document Memory)**:
   - *Quy mô thời gian*: Thô (Coarse / Slow frequency).
   - *Ranh giới cập nhật*: Toàn bộ ranh giới tài liệu.
   - *Số tham số*: $1,771,584$ trọng số.
   - *Vai trò*: Neo giữ bất biến toàn cục (global invariants), chủ đề xuyên suốt và tóm lược vĩ mô.

*Tổng số tham số thích ứng bộ nhớ*: $3 \times 1,771,584 = \mathbf{5,314,752}$ tham số. Toàn bộ trọng số mô hình nền tảng (135M) được **đóng băng 100%**.

### 2.6 Tương tác phần dư bộ nhớ & Trộn cổng (Memory Residual & Blending)
Biểu diễn ẩn $\mathbf{H}_t$ tại bước $t$ sau khi đi qua khối Attention của mô hình được hòa trộn với phần dư bộ nhớ liên tục $\mathbf{M}_t^{\text{CMS}}$:
$$\mathbf{H}_t^{\text{norm}} = \text{LayerNorm}(\mathbf{H}_t)$$
$$\mathbf{M}_t^{\text{CMS}} = \text{CMS}(\mathbf{H}_t^{\text{norm}}) = \mathbf{f}^{(3)}(\mathbf{f}^{(2)}(\mathbf{f}^{(1)}(\mathbf{H}_t^{\text{norm}})))$$
$$\mathbf{H}_t^{\text{final}} = \mathbf{H}_t + \mathbf{M}_t^{\text{CMS}}$$
Nhờ cơ chế phần dư này:
- Nếu bộ nhớ chưa nạp dữ kiện, $\mathbf{M}_t^{\text{CMS}} \approx 0$, mô hình hoạt động như mô hình gốc.
- Khi đã nạp tài liệu, $\mathbf{M}_t^{\text{CMS}}$ dịch chuyển không gian ẩn về phía phân phối dữ kiện của văn bản.

### 2.7 Truy xuất lai & Tương tác ngữ cảnh (Hybrid Retrieval & Context Interaction)
Hệ thống kết hợp 2 kênh dẫn truyền thông tin độc lập:
1. **Kênh Bộ nhớ Tham số (Parametric Channel)**: Truy xuất ngầm thông qua $\mathbf{M}_t^{\text{CMS}}$, không tiêu tốn token đầu vào trong prompt.
2. **Kênh Truy xuất Ngữ cảnh (Lexical Retrieval Channel - BM25)**: Tra cứu nhanh các đoạn văn có liên quan nhất với tham số chuẩn $k_1 = 1.5, b = 0.75$.
3. **Cơ chế Gated Hybrid (P2)**: Kết hợp cả hai nguồn tri thức: Các đoạn văn trích xuất đóng vai trò làm bằng chứng hiển ngôn (explicit evidence) để trích dẫn, trong khi bộ nhớ SA-CMS hỗ trợ suy luận ngữ nghĩa ngầm và bảo toàn thông tin toàn cục.

### 2.8 Lọc bằng chứng & Xác thực căn cứ (Evidence Filtering & Grounding Validation)
- **Ngưỡng điểm bằng chứng**: $\tau = 3.0$ (theo chuẩn khóa Phase 4.0.2).
- **Ngưỡng độ bao phủ câu hỏi (Query Coverage)**: Tối thiểu $35\%$ từ khóa nội dung của câu hỏi phải xuất hiện trong các đoạn bằng chứng.
- **Phân loại 4 trạng thái căn cứ (Grounded Status)**:
  1. `SUPPORTED`: Điểm BM25 $\ge \tau$ và độ phủ $\ge 0.35$. Dữ kiện hoàn toàn được chứng minh bằng văn bản.
  2. `PARTIALLY_SUPPORTED`: Có đoạn trích liên quan nhưng chưa bao quát toàn bộ câu hỏi phức hợp.
  3. `INSUFFICIENT_EVIDENCE`: Điểm dưới ngưỡng tin cậy hoặc thông tin quá mỏng.
  4. `UNANSWERABLE`: Câu hỏi ngoài phạm vi tài liệu (tri thức thế giới hoặc không có căn cứ).

### 2.9 Quản lý trích dẫn & Cổng từ chối (Citation Manager & Refusal Controller)
- **Trích dẫn có thể truy xuất nguồn gốc (Provenance Traceability)**: Mọi câu trả lời được liên kết trực tiếp với nhãn trích dẫn:
  $$\text{Claim} \longrightarrow [1] \longrightarrow (\text{Passage ID}, \text{Document ID}, \text{Section}, \text{Paragraph})$$
- **Quy tắc từ chối bất biến (Absolute Refusal Rule)**:
  Nếu trạng thái rơi vào `INSUFFICIENT_EVIDENCE` hoặc `UNANSWERABLE`, hệ thống **tuyệt đối không bịa đặt hoặc suy diễn thông tin ngoài lề**, mà phản hồi theo thông điệp chuẩn mực:
  > *"Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn."*

---

## 3. BẢNG ĐỐI CHIẾU CÁC PHƯƠNG PHÁP NGHIÊN CỨU

| Mã phương pháp | Tên phương pháp | Cửa sổ ngữ cảnh (Context) | Bộ nhớ tham số (Memory) | Cổng từ chối (Refusal Gate) | Trích dẫn nguồn |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **B1** | Full-Context Baseline | Toàn bộ văn bản (ICL) | Không | Không | Không |
| **B2** | Standard RAG Baseline | Top-k đoạn trích BM25 | Không | Có ($\tau=3.0$) | Có |
| **B4** | Fixed-Token Memory CMS | Không (Context Evicted) | 1 cấp độ (Token cố định) | Không | Không |
| **B5** | Multi-Level Memory CMS | Không (Context Evicted) | 3 cấp độ (Token cố định) | Không | Không |
| **P1** | SA-CMS Memory-Only | Không (Context Evicted) | 3 cấp độ (Căn chỉnh cấu trúc) | Không | Không |
| **P2** | SA-CMS Gated Hybrid | Top-k đoạn trích BM25 | 3 cấp độ (Căn chỉnh cấu trúc) | Có ($\tau=3.0$) | Có |

---

## 4. Ý NGHĨA KHOA HỌC TRƯỚC HỘI ĐỒNG NCKH

1. **Khắc phục giới hạn của RAG truyền thống**: RAG truyền thống bị mù cấu trúc (structure-agnostic) khi cắt mảnh tài liệu. SA-CMS đồng bộ nhịp cập nhật với logic tự nhiên của tác giả viết tài liệu.
2. **Khắc phục giới hạn của LLM ngữ cảnh dài**: Nhồi ngữ cảnh vào prompt làm tăng chi phí token tuyến tính và chi phí chú ý bậc hai. SA-CMS giữ prompt ngắn gọn, nén tri thức vào bộ nhớ tham số.
3. **Minh bạch hóa cơ chế nơ-ron**: Cung cấp công cụ đo lường độ lớn phần dư bộ nhớ $||\mathbf{M}_t||$, khoảng cách trạng thái ẩn và phân tích đóng góp của từng cấp độ thời gian.
4. **Không ảo giác**: Cơ chế Grounding Safety Layer đảm bảo hệ thống chỉ trả lời khi có căn cứ xác thực, triệt tiêu nguy cơ sinh thông tin sai lệch.

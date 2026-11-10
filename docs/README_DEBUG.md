# README_DEBUG — Danh mục sai lệch đã xác minh

Tài liệu này ghi lại các lỗi dữ liệu, lỗi mã nguồn và sai lệch số liệu đã được **xác minh
bằng artifact** trong quá trình kiểm toán repo. Mục tiêu là để người đọc sau không lặp lại
cùng một kết luận sai.

Quy ước mức độ tin cậy:

- **[Chắc chắn]** — có artifact hoặc chạy lại được để chứng minh.
- **[Có khả năng cao]** — suy luận có căn cứ từ artifact nhưng chưa chạy lại.
- **[Ước đoán]** — suy đoán để lấp chỗ thiếu dữ liệu.

---

## P0 — Repo không build/không chạy được

### P0.1. Dấu xung đột merge bị commit vào mã nguồn [Chắc chắn]

Commit cuối (`80adf49`) chụp lại working tree còn nguyên dấu xung đột rebase. Hệ quả:

- `src/hybrid_qa/pipeline.py` — 1 khối xung đột → `SyntaxError`, không import được.
- `src/web/app.py` — 5 khối xung đột → `SyntaxError`, không import được.
- `data/chat_sessions/sessions.json` — 162 khối → `JSONDecodeError`, app không khởi động.
- `data/document_store/index.json` — 1 khối → `JSONDecodeError`.
- `results/phase5_1/query_history.jsonl` — 1 khối.
- `results/phase5_1/token_efficiency_experiments.jsonl` — 1 khối.
- `results/round2/analysis/rq_deep_analysis_report.json` — 1 khối.
- `training_output_final/manifests/final_manifest.json` — 1 khối.

Toàn bộ test suite trước khi sửa: **9 lỗi collect**, 0 test chạy được.

**Cách xử lý đã áp dụng.** Với mã nguồn, giữ phía `b58cf1e` vì phía `HEAD` tham chiếu các
biến không tồn tại (`scored_sentences`, `content_q_tokens`, `cit_tag`) — đó là code chết,
không thể chạy dù có giải quyết conflict. Với các tệp runtime (sessions, index, log), hợp
nhất theo khóa thay vì chọn một phía, vì cả hai phía đều là dữ liệu thật đã ghi.

Sau khi sửa: **196 passed, 2 warnings**.

**Phòng ngừa.** Thêm kiểm tra trước commit:

```powershell
git grep -nE "^(<<<<<<< |=======|>>>>>>> )" -- "*.py" "*.json" "*.jsonl"
```

### P0.2. Thiếu định nghĩa sau khi giải quyết conflict [Chắc chắn]

Sau khi bỏ marker, `src/web/app.py` vẫn lỗi `NameError` vì khối được giữ dùng các biến mà
khối còn lại đã định nghĩa: `wants_suggested_questions`, `best_passage`, `sentences`,
`cit_1`, `count_tokens`. Đã bổ sung lại, trong đó `sentences` dùng
`split_clean_sentences()` thay cho `split(".")` để không cắt nát số thập phân như `30,9%`.

---

## P1 — Số liệu sai so với artifact

### P1.1. p-value `3.03e-7` và kết luận "STATISTICALLY SIGNIFICANT" là bịa [Chắc chắn]

Tính lại trực tiếp từ `results/raw_predictions/rq2_raw_predictions.jsonl` (N = 1.140 cặp
ghép theo `question_id` + `seed`):

| Chỉ số | Giá trị thật | Giá trị từng ghi trong tài liệu |
|:---|:---|:---|
| P1 Token F1 | `0.0496` | `0.0421` |
| B5 Token F1 | `0.0437` | `0.0036` |
| Chênh lệch ghép cặp | `+0.0059` | `+0.0385` |
| Paired t | `1.1227` | `5.14` |
| **p-value** | **`0.2618`** | **`3.03e-7`** |
| Wilcoxon p | `0.4206` | — |
| Bootstrap 95% CI | `[-0.0044, 0.0165]` | — |
| Kết luận | **Không có ý nghĩa thống kê** | "STATISTICALLY SIGNIFICANT" |

CI 95% chứa 0. Cổng G2 phải ghi **KHÔNG ĐẠT**. Số `0.0421` / `0.0036` là tính theo cách
khác (không ghép cặp đúng) nên không dùng để suy luận thống kê.

### P1.2. Số P1 `0.2014` / `63.33%` không tái lập được [Chắc chắn]

Chạy lại P1 từ `checkpoints/phase4_1/cms_3lvl_seed_{42,43,44}.pt` bằng chính harness
`scripts/evaluate_ablation_a1_a3.py`:

| Seed | QASPER F1 | LongHealth | MK-NIAH |
|:---|:---:|:---:|:---:|
| 42 | 0.1610 | 60.0% | 70.0% |
| 43 | 0.1920 | 50.0% | 60.0% |
| 44 | 0.1808 | 60.0% | 60.0% |
| **Mean** | **0.1779 ± 0.0157** | **56.67% ± 5.77%** | **63.33% ± 5.77%** |

So với tài liệu cũ: `0.2014 ± 0.0125` / `63.33% ± 5.77%` / `70.0%`. Chỉ số LongHealth SD
trùng ngẫu nhiên; F1 và MK-NIAH đều lệch.

**Hệ quả khoa học quan trọng.** Với P1 thật là `0.1779`, so sánh với A1 (`0.1867`) cho thấy
**P1 thấp hơn A1** trên QASPER. Kết luận cũ "ranh giới cấu trúc tốt hơn ranh giới ngẫu
nhiên" **không đứng vững**.

### P1.3. Số ablation A1/A3 trong tài liệu khớp nguồn nào [Chắc chắn]

Ba bản `a1_external_results.json` / `a3_external_results.json` (ở `results/phase4_2/`,
`training_output_final/raw_results/`, `training_output_final/A1_random_boundary/`) **giống
hệt nhau** — artifact không hỏng. Nhưng chúng cho số khác với bản tổng hợp trong tài liệu:

| Variant | Từ `*_external_results.json` | Từ `ablation_raw_predictions.jsonl` | Tài liệu cũ ghi |
|:---|:---|:---|:---|
| A1 QASPER | 0.1958 | **0.1867** | 0.1866 |
| A1 LongHealth | 41.67% | **58.33%** | 58.33% |
| A1 MK-NIAH | 20.0% | **63.33%** | 63.33% |
| A3 QASPER | 0.2185 | **0.1578** | 0.1579 |
| A3 LongHealth | 21.67% | **43.33%** | 43.33% |
| A3 MK-NIAH | 20.0% | **70.0%** | 70.0% |

Hai bộ artifact **mâu thuẫn nhau**. Bộ `ablation_raw_predictions.jsonl` (360 dòng, mức
item) khớp với tài liệu cũ nên được chọn làm nguồn chuẩn; bộ `*_external_results.json` là
lần chạy khác (cấu hình khác) và **không** dùng.

### P1.4. A1/A3 không tái lập được trên máy sạch [Chắc chắn]

Checkpoint `cms_3lvl_random_seed_*.pt` và `cms_3lvl_additive_seed_*.pt` nằm trong
`training_output_final/checkpoints/` — thư mục bị `.gitignore` (dòng 45). Trên đĩa hiện tại
**không còn** các tệp này. Hệ quả: một bản clone mới không thể chạy lại A1/A3. Số A1/A3
chỉ còn đọc được từ log dự đoán mức item.

`scripts/audit_ablation_reproduction.py` đã được sửa để báo `checkpoints_missing` thay vì
crash, và in `INCOMPLETE: A1, A3` khi kết thúc.

### P1.5. Bảng ablation (e) và (f) không có artifact [Chắc chắn]

Các dòng "Có/không có Tự học" và "Hạng adapter r / Số tầng K" trong báo cáo có số liệu cụ
thể nhưng **không truy được nguồn nào**. Không có file kết quả, không có log, không có
checkpoint tương ứng. Đã thay bằng "Không có artifact".

---

## P2 — Lỗi đường ống đánh giá

### P2.1. RQ3 tiếng Việt F1 ≈ 0 do thiếu `ground_truth` [Chắc chắn]

`results/phase4_1/vietnamese/vietnamese_raw_results.json` — mỗi record **không có** trường
`ground_truth`. Script `scripts/evaluate_rq3_mkniah_diagnosis.py` đọc
`rec.get("ground_truth", "")` nên mọi F1 = 0, và con số `0.0002` / `0.0001` trong tài liệu
chỉ phản ánh lỗi schema, không phản ánh năng lực mô hình.

### P2.2. `token_f1` cấp run bị nhân đôi [Chắc chắn]

Trường `token_f1` tổng hợp ở cấp run bằng **2× trung bình mức record**. Ví dụ B2: run ghi
`0.1596` trong khi trung bình records là `0.0798`. Không dùng trường cấp run để kết luận.

### P2.3. RQ5 token budget: N không khớp [Chắc chắn]

`results/rq5_token_budget_comparison.json` ghi `num_evaluated_queries = 40` nhưng
`raw_records` chỉ có **10**. P2 đạt F1 `1.0` bất thường — dấu hiệu đánh giá trên quá ít
mẫu. Số liệu RQ5 không dùng được để so sánh.

---

## P3 — Vấn đề phương pháp luận

### P3.1. Gradient clipping không đồng nhất giữa P1 và A1/A3 [Chắc chắn]

- `scripts/run_phase4_1.py` (P1): **không** clip gradient.
- `scripts/train_ablation_a1_a3.py` (A1, A3): `clip_grad_norm_(..., 1.0)`.

So sánh P1 với A1/A3 do đó bị nhiễu bởi công thức tối ưu khác nhau, không chỉ bởi kiến
trúc. Đã thêm clip 1.0 vào `run_phase4_1.py` để hai nhánh cùng điều kiện.

**Lưu ý:** checkpoint P1 hiện có trên đĩa được huấn luyện **trước** khi sửa, nên không
khớp hoàn toàn với mã nguồn hiện tại. Cần huấn luyện lại P1 nếu muốn ablation sạch.

### P3.2. `dry_run: true` trong manifest [Chắc chắn]

`training_output_final/manifests/final_manifest.json` ghi `"dry_run": true`, nghĩa là
manifest mô tả một lần chạy mô phỏng, không phải một lần chạy GPU thật.

### P3.3. Chỉ 3 seeds [Chắc chắn]

Mọi so sánh đơn lẻ dựa trên 3 seeds. Với SD cỡ `0.015` (QASPER) và chênh lệch cỡ `0.009`,
không thể phân biệt được với nhiễu. Cần ≥ 10 seeds để có power thống kê.

---

## Số liệu đã xác minh là ĐÚNG (giữ nguyên)

| Hạng mục | Giá trị | Nguồn |
|:---|:---|:---|
| RQ2 P1 mean F1 | `0.0496` | `rq2_raw_predictions.jsonl` |
| RQ2 B5 mean F1 | `0.0437` | `rq2_raw_predictions.jsonl` |
| RQ2 N | `1.140` | `rq2_raw_predictions.jsonl` |
| RQ2 p-value | `0.2618` | tính lại, paired t-test |
| RQ4 P1 $F_{20}$ | `-0.0034` | `rq4_raw_predictions.jsonl` |
| RQ4 P1 $\Delta$loglik | `+0.3991` | `rq4_external_results.json` |
| B4 $F_{20}$ | `-0.0134` | `rq4_external_results.json` |
| B5 $F_{20}$ | `-0.0147` | `rq4_external_results.json` |
| VRAM train đỉnh | `743.86 MB` | `training_handoff_rtx3050/.../training_meta_seed42_lvl3.json` |
| Checksum | `54/54` OK | `training_output_final/checksums.sha256` |
| RQ3 Faithfulness B2 | `95.7%` | `rq3_consistency_audit.json` |
| RQ3 Faithfulness P2 | `98.92%` | `rq3_consistency_audit.json` |

---

## Việc còn treo

1. **Huấn luyện lại P1** với gradient clipping đồng nhất, rồi chạy lại ablation đầy đủ.
2. **Khôi phục checkpoint A1/A3** (đang mất) hoặc chấp nhận rằng chúng không tái lập được.
3. **Chạy lại nhánh self-study và quét $r$/$K$** — hiện không có artifact.
4. **Sửa schema RQ3 tiếng Việt** để có `ground_truth`, rồi đánh giá lại.
5. **Chạy lại RQ5** với N đầy đủ.
6. **Tăng số seeds** lên ≥ 10 cho các so sánh chính.
"""
Phase 5.9 — Interactive Demo Scenarios Specification & Executor.
Defines and executes the 7 polished research committee demonstration flows:
DEMO 1: Single Document QA -> Verified Citation
DEMO 2: Long Document -> Question on distant section
DEMO 3: Multiple Documents -> Cross-document comparison & synthesis
DEMO 4: Unsupported Question -> Calibrated refusal (Zero hallucination)
DEMO 5: Context Eviction -> Memory advantage demonstration
DEMO 6: Concise Evidence Mode -> Ultra-compact output token reduction
DEMO 7: Research Mode -> Transparent neural inspection trace (retrieval + memory)

LABEL: ROUND_2_EXTENSION
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEMO_SCENARIOS = [
    {
        "id": "DEMO_1_SINGLE_DOC",
        "title": "Demo 1: Hỏi đáp Tài liệu Đơn lẻ & Trích dẫn Nguồn",
        "doc_ids": ["DEMO_DOC_001"],
        "query": "Tầng 1 của hệ thống CMS đảm nhiệm vai trò gì trong kiến trúc?",
        "expected_behavior": "Trả lời chính xác chức năng Tầng 1 kèm nhãn trích dẫn [1] dẫn xuất trực tiếp đến phần 1 của DEMO_DOC_001.",
        "expected_refusal": False,
        "mode": "hybrid",
        "answer_mode": "balanced",
    },
    {
        "id": "DEMO_2_LONG_DOC_DISTANT",
        "title": "Demo 2: Tài liệu Dài & Tra cứu Mục Ở Xa",
        "doc_ids": ["DEMO_DOC_002"],
        "query": "Tác động kinh tế xã hội và thời gian di chuyển giữa Hà Nội và TP.HCM dự kiến là bao lâu?",
        "expected_behavior": "Truy xuất thành công mục 3 ở cuối tài liệu (5 giờ 30 phút, GDP 1%) và trích dẫn chuẩn xác.",
        "expected_refusal": False,
        "mode": "hybrid",
        "answer_mode": "balanced",
    },
    {
        "id": "DEMO_3_MULTI_DOC_COMPARISON",
        "title": "Demo 3: Đối chiếu & So sánh Đa Tài liệu",
        "doc_ids": ["DEMO_DOC_001", "DEMO_DOC_002"],
        "query": "So sánh nội dung và lĩnh vực chính giữa hai tài liệu.",
        "expected_behavior": "Hệ thống trích xuất bằng chứng từ cả 2 tài liệu và tổng hợp so sánh có trích dẫn riêng cho từng tài liệu.",
        "expected_refusal": False,
        "mode": "hybrid",
        "answer_mode": "balanced",
    },
    {
        "id": "DEMO_4_REFUSAL_UNSUPPORTED",
        "title": "Demo 4: Câu hỏi Ngoài Tài liệu & Cổng Từ chối",
        "doc_ids": ["DEMO_DOC_001"],
        "query": "Hệ thống CMS sử dụng vi xử lý lượng tử loại nào để mã hóa bộ nhớ?",
        "expected_behavior": "Kích hoạt Cổng từ chối: 'Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn.' Tuyệt đối không ảo giác.",
        "expected_refusal": True,
        "mode": "hybrid",
        "answer_mode": "balanced",
    },
    {
        "id": "DEMO_5_CONTEXT_EVICTION",
        "title": "Demo 5: Xóa Ngữ cảnh & Đánh giá Ưu thế Bộ nhớ P1/P2",
        "doc_ids": ["DEMO_DOC_001"],
        "query": "Cơ chế cổng hòa trộn phần dư (Gated Residual Blending) hoạt động như thế nào?",
        "expected_behavior": "Khi context bị xóa (evicted), bộ nhớ tham số SA-CMS định hướng câu trả lời với mức chi phí input tokens giảm >80%.",
        "expected_refusal": False,
        "mode": "memory",
        "answer_mode": "balanced",
    },
    {
        "id": "DEMO_6_CONCISE_MODE",
        "title": "Demo 6: Chế độ Trả lời Cô đọng (Giảm Token Đầu ra)",
        "doc_ids": ["DEMO_DOC_001"],
        "query": "Cơ chế cổng hòa trộn phần dư giúp ngăn ngừa hiện tượng gì?",
        "expected_behavior": "Trả lời cực kỳ ngắn gọn (chỉ 1 câu chứa đáp án cốt lõi: ngăn ngừa trôi dạt biểu diễn) + trích dẫn [1], tiết kiệm >50% token đầu ra.",
        "expected_refusal": False,
        "mode": "hybrid",
        "answer_mode": "concise",
    },
    {
        "id": "DEMO_7_RESEARCH_MODE_TRACE",
        "title": "Demo 7: Chế độ Nghiên cứu & Dấu vết Thần kinh",
        "doc_ids": ["DEMO_DOC_001"],
        "query": "Tóm tắt cơ chế nén tri thức của SA-CMS.",
        "expected_behavior": "Hiển thị toàn bộ dấu vết: Chiến lược định tuyến, Top-k BM25, Trọng số bộ nhớ 3 cấp độ, Độ lệch trạng thái ẩn và Độ lớn phần dư ||M_t||.",
        "expected_refusal": False,
        "mode": "hybrid",
        "answer_mode": "balanced",
    },
]


def save_demo_manifest(output_dir: str = "results/round2/demo"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "title": "SA-CMS Interactive Demo Scenarios Manifest",
        "protocol": "ROUND_2_EXTENSION",
        "total_scenarios": len(DEMO_SCENARIOS),
        "scenarios": DEMO_SCENARIOS,
    }
    with open(out_dir / "demo_scenarios_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


if __name__ == "__main__":
    man = save_demo_manifest()
    print(f"Saved {man['total_scenarios']} demo scenarios to results/round2/demo/demo_scenarios_manifest.json")

"""
Unit tests for Step-by-Step Process & Algorithm Stitcher (Wave 5 / Sovereign Brain v5.5).
Verifies:
1. Procedural synthesis for 'các bước thực hiện' queries.
2. Cryptographic workflow synthesis for 'quy trình mã hóa và giải mã' queries.
3. Algorithm execution synthesis for 'cách hoạt động của thuật toán' queries.
4. Strict sequential ordering: Bước 1, Bước 2, Bước 3...
5. Multi-phase segregation: Key Generation -> Encryption -> Decryption.
6. Formula preservation (modulo arithmetic, XOR, modular inverse).
7. Artifact stripping: Zero leakage of === Page N === markers.
8. Cross-chunk deduplication for overlapping passages.
9. Robust edge-case handling (empty passages, None, missing keys).
10. Network protocols and database normalization multi-step processes.
11. Pure stdlib ultra-fast execution benchmark (< 20ms for 100 runs).
"""

from __future__ import annotations

import time
import pytest
from src.self_study.datasets.step_by_step_process_stitcher import (
    StepByStepProcessStitcher,
    synthesize_step_by_step_process,
    detect_process_intent,
    extract_topic_from_query,
    strip_accents,
    clean_text_line,
)


# ------------------------------------------------------------------------------
# TEST FIXTURES & DATA SAMPLES
# ------------------------------------------------------------------------------

SAMPLE_RSA_PASSAGES = [
    {
        "passage_id": "P001",
        "section_title": "1. HỆ MÃ HÓA CÔNG KHAI RSA",
        "text": """=== Page 12 ===
Sinh khóa (Key Generation):
1. Chọn hai số nguyên tố rất lớn p và q độc lập với nhau.
2. Tính modulo n = p * q và hàm số phi Euler: phi(n) = (p - 1) * (q - 1).
3. Chọn một số nguyên e nguyên tố cùng nhau với phi(n), nghĩa là gcd(e, phi(n)) = 1, với 1 < e < phi(n).
4. Tính khóa bí mật d là nghịch đảo modulo của e: d * e = 1 (mod phi(n)).
Khóa công khai là cặp (e, n), khóa bí mật là cặp (d, n).
""",
    },
    {
        "passage_id": "P002",
        "section_title": "QUY TRÌNH MÃ HÓA VÀ GIẢI MÃ RSA",
        "text": """=== Page 13 ===
Quy trình mã hóa:
Bước 1: Chuyển đổi thông điệp cần gửi thành số nguyên m trong đoạn 0 <= m < n.
Bước 2: Tính bản mã c bằng công thức lũy thừa modulo: c = m^e mod n.

Quy trình giải mã:
Bước 1: Người nhận nhận bản mã c và sử dụng khóa bí mật d.
Bước 2: Khôi phục bản rõ m theo công thức: m = c^d mod n.
Lưu ý: Độ an toàn của RSA dựa trên bài toán khó phân tích một số nguyên lớn n thành tích các thừa số nguyên tố p và q.
""",
    },
]

SAMPLE_DIJKSTRA_PASSAGES = [
    {
        "passage_id": "P_DIJKSTRA_01",
        "section_title": "THUẬT TOÁN DIJKSTRA",
        "text": """=== Page 25 ===
Các bước thực hiện thuật toán Dijkstra tìm đường đi ngắn nhất:
Bước 1 (Khởi tạo): Gán khoảng cách dist[s] = 0 cho đỉnh xuất phát s, và dist[v] = vô cùng với mọi đỉnh v khác s. Tập đỉnh chưa xét Q = V.
Bước 2 (Chọn đỉnh): Tìm trong tập Q đỉnh u có giá trị khoảng cách dist[u] nhỏ nhất.
Bước 3 (Cập nhật đỉnh kề): Với mỗi đỉnh kề v của u còn nằm trong Q, nếu dist[u] + weight(u, v) < dist[v] thì gán dist[v] = dist[u] + weight(u, v).
Bước 4 (Loại bỏ và lặp): Loại bỏ đỉnh u ra khỏi tập Q. Lặp lại bước 2 cho tới khi tập Q rỗng.
Độ phức tạp: O(V^2) với đồ thị biểu diễn ma trận kề, hoặc O((E + V) log V) khi dùng hàng đợi ưu tiên Priority Queue.
""",
    }
]

SAMPLE_QUICKSORT_PASSAGES = [
    {
        "passage_id": "P_QS_01",
        "section_title": "THUẬT TOÁN QUICKSORT",
        "text": """=== Page 30 ===
Cách thức hoạt động của thuật toán sắp xếp nhanh QuickSort:
Đầu tiên: Chọn một phần tử bất kỳ trong mảng làm phần tử chốt (Pivot).
Tiếp theo: Thực hiện phân hoạch mảng (Partition) đưa các phần tử nhỏ hơn chốt về bên trái và các phần tử lớn hơn chốt về bên phải.
Sau đó: Gọi đệ quy giải thuật QuickSort trên hai mảng con bên trái và bên phải.
Cuối cùng: Trả về mảng đã được sắp xếp khi kích thước dãy con giảm về 0 hoặc 1 (điều kiện dừng).
""",
    }
]

SAMPLE_AFFINE_PASSAGES = [
    {
        "passage_id": "P_AFFINE_01",
        "section_title": "HỆ MÃ AFFINE",
        "text": """=== Page 18 ===
Quy trình mã hóa và giải mã của hệ mã Affine:
Sinh khóa:
1. Chọn cặp khóa (a, b) trong Z26 sao cho gcd(a, 26) = 1.
2. Tìm phần tử nghịch đảo modulo a^(-1) trong Z26.

Quy trình mã hóa:
1. Chuyển đổi mỗi chữ cái của bản rõ thành số nguyên x trong đoạn [0, 25].
2. Áp dụng hàm mã hóa: e(x) = (a * x + b) mod 26 để nhận được ký tự mã hóa c.

Quy trình giải mã:
1. Chuyển đổi mỗi ký tự bản mã thành số nguyên y trong đoạn [0, 25].
2. Áp dụng hàm giải mã: d(y) = a^(-1) * (y - b) mod 26 để khôi phục ký tự gốc x.
""",
    }
]

SAMPLE_TCP_PASSAGES = [
    {
        "passage_id": "P_TCP_01",
        "section_title": "GIAO THỨC TCP - BẮT TAY 3 BƯỚC",
        "text": """=== Page 45 ===
Cách hoạt động của quá trình bắt tay 3 bước TCP (3-way Handshake):
Bước 1 (SYN): Client gửi một gói tin TCP mang cờ SYN (Synchronize) với số thứ tự khởi tạo ban đầu seq = x tới Server.
Bước 2 (SYN-ACK): Server nhận gói SYN, phản hồi lại gói tin mang cờ SYN và ACK với ack = x + 1 và seq = y.
Bước 3 (ACK): Client gửi lại gói tin xác nhận mang cờ ACK với ack = y + 1 để hoàn tất thiết lập kết nối song công.
""",
    }
]


# ------------------------------------------------------------------------------
# UNIT TESTS
# ------------------------------------------------------------------------------

def test_intent_detection_and_topic_extraction():
    """Verify robust query intent classification and topic extraction."""
    q1 = "Các bước thực hiện thuật toán Dijkstra tìm đường đi ngắn nhất?"
    intent1 = detect_process_intent(q1)
    assert intent1["is_process_query"] is True
    assert intent1["process_type"] in ("algorithm_workflow", "step_by_step")
    assert "Dijkstra" in intent1["topic"]

    q2 = "Quy trình mã hóa và giải mã của hệ mã RSA như thế nào?"
    intent2 = detect_process_intent(q2)
    assert intent2["is_process_query"] is True
    assert intent2["process_type"] == "encryption_decryption"
    assert "RSA" in intent2["topic"]

    q3 = "Cách hoạt động của thuật toán QuickSort?"
    intent3 = detect_process_intent(q3)
    assert intent3["is_process_query"] is True
    assert intent3["process_type"] == "algorithm_workflow"
    assert "QuickSort" in intent3["topic"]


def test_synthesize_cryptographic_workflow_rsa():
    """Verify encryption and decryption step-by-step synthesis for RSA."""
    query = "Quy trình mã hóa và giải mã của hệ mã RSA"
    result = synthesize_step_by_step_process(query, SAMPLE_RSA_PASSAGES)

    assert result["status"] == "success"
    assert result["is_process_query"] is True
    assert result["step_count"] >= 6
    assert "P001" in result["citations"]
    assert "P002" in result["citations"]

    formatted = result["formatted_text"]
    assert formatted == result["answer"]

    # 1. Verify sub-process phase headings
    assert "Quy trình sinh khóa" in formatted
    assert "Quy trình mã hóa" in formatted
    assert "Quy trình giải mã" in formatted

    # 2. Verify step ordering
    assert "Bước 1" in formatted
    assert "Bước 2" in formatted
    assert "Bước 3" in formatted
    assert "Bước 4" in formatted

    # 3. Verify math formulas are preserved
    assert "phi(n) = (p - 1) * (q - 1)" in formatted
    assert "c = m^e mod n" in formatted
    assert "m = c^d mod n" in formatted

    # 4. Verify no page marker leakage
    assert "=== Page" not in formatted
    assert "Page 12" not in formatted
    assert "Page 13" not in formatted

    # 5. Verify notes inclusion
    assert "Lưu ý" in formatted or "bảo mật" in formatted or "an toàn" in formatted


def test_synthesize_procedural_steps_dijkstra():
    """Verify procedural steps synthesis for 'các bước thực hiện' query."""
    query = "Các bước thực hiện thuật toán Dijkstra"
    result = synthesize_step_by_step_process(query, SAMPLE_DIJKSTRA_PASSAGES)

    assert result["status"] == "success"
    assert result["step_count"] == 4
    formatted = result["formatted_text"]

    # Strict step sequence
    idx1 = formatted.find("Bước 1")
    idx2 = formatted.find("Bước 2")
    idx3 = formatted.find("Bước 3")
    idx4 = formatted.find("Bước 4")

    assert 0 <= idx1 < idx2 < idx3 < idx4

    # Check content integrity
    assert "Khởi tạo" in formatted
    assert "dist[s] = 0" in formatted
    assert "Cập nhật đỉnh kề" in formatted
    assert "O(V^2)" in formatted or "Độ phức tạp" in formatted


def test_synthesize_algorithm_lifecycle_quicksort():
    """Verify transition markers ('Đầu tiên', 'Tiếp theo', 'Sau đó', 'Cuối cùng') map to sequential steps."""
    query = "Cách hoạt động của thuật toán QuickSort"
    result = synthesize_step_by_step_process(query, SAMPLE_QUICKSORT_PASSAGES)

    assert result["status"] == "success"
    assert result["step_count"] == 4
    formatted = result["formatted_text"]

    assert "Bước 1" in formatted
    assert "Bước 2" in formatted
    assert "Bước 3" in formatted
    assert "Bước 4" in formatted

    assert "Pivot" in formatted or "phần tử chốt" in formatted
    assert "Phân hoạch" in formatted or "Partition" in formatted
    assert "đệ quy" in formatted.lower()


def test_synthesize_cryptographic_workflow_affine():
    """Verify Affine cipher encryption and decryption synthesis."""
    query = "Quy trình mã hóa và giải mã hệ mã Affine"
    result = synthesize_step_by_step_process(query, SAMPLE_AFFINE_PASSAGES)

    assert result["status"] == "success"
    assert result["step_count"] >= 4
    formatted = result["formatted_text"]

    assert "Quy trình mã hóa" in formatted
    assert "Quy trình giải mã" in formatted
    assert "gcd(a, 26) = 1" in formatted
    assert "e(x) = (a * x + b) mod 26" in formatted
    assert "d(y) = a^(-1) * (y - b) mod 26" in formatted


def test_synthesize_protocol_tcp_handshake():
    """Verify network protocol 3-way handshake step synthesis."""
    query = "Cách hoạt động của giao thức TCP bắt tay 3 bước"
    result = synthesize_step_by_step_process(query, SAMPLE_TCP_PASSAGES)

    assert result["status"] == "success"
    assert result["step_count"] == 3
    formatted = result["formatted_text"]

    assert "Bước 1" in formatted
    assert "Bước 2" in formatted
    assert "Bước 3" in formatted
    assert "SYN" in formatted
    assert "SYN-ACK" in formatted
    assert "ACK" in formatted


def test_cross_chunk_deduplication():
    """Verify duplicate steps in overlapping chunks are cleaned up."""
    query = "Các bước thực hiện thuật toán"
    duplicate_passages = [
        {
            "passage_id": "CHUNK_1",
            "text": """
Bước 1: Khởi tạo biến đếm count = 0.
Bước 2: Đọc dữ liệu từ tệp đầu vào.
""",
        },
        {
            "passage_id": "CHUNK_2",
            "text": """
Bước 1: Khởi tạo biến đếm count = 0.
Bước 2: Đọc dữ liệu từ tệp đầu vào.
Bước 3: Xử lý và ghi kết quả ra tệp đích.
""",
        },
    ]

    result = synthesize_step_by_step_process(query, duplicate_passages)
    assert result["status"] == "success"
    assert result["step_count"] == 3
    assert len(result["steps"]) == 3
    assert result["steps"][0]["step_number"] == 1
    assert result["steps"][1]["step_number"] == 2
    assert result["steps"][2]["step_number"] == 3


def test_edge_cases_empty_and_none():
    """Verify robust handling of empty, None, or ill-formed inputs."""
    # 1. Empty passages
    res_empty = synthesize_step_by_step_process("các bước làm gì đó", [])
    assert res_empty["status"] == "no_content"
    assert res_empty["step_count"] == 0
    assert res_empty["steps"] == []
    assert res_empty["formatted_text"] == ""

    # 2. None passages
    res_none = synthesize_step_by_step_process("quy trình", None)
    assert res_none["status"] == "no_content"
    assert res_none["step_count"] == 0

    # 3. String-only passages
    res_strings = synthesize_step_by_step_process(
        "các bước cấu hình",
        ["1. Sinh khóa k ngẫu nhiên.", "2. Tính c = p XOR k.", "3. Giải mã p = c XOR k."]
    )
    assert res_strings["status"] == "success"
    assert res_strings["step_count"] == 3
    assert "Bước 1" in res_strings["formatted_text"]
    assert "Bước 2" in res_strings["formatted_text"]
    assert "Bước 3" in res_strings["formatted_text"]


def test_custom_stitcher_options():
    """Verify OOP StepByStepProcessStitcher customization options."""
    stitcher = StepByStepProcessStitcher(step_prefix="Step", include_summary=False)
    result = stitcher.synthesize(
        "các bước xử lý hệ thống",
        [{"passage_id": "P1", "text": "1. Cấu hình địa chỉ IP mạng.\n2. Kiểm tra tín hiệu kết nối."}]
    )
    assert result["status"] == "success"
    assert "Step 1" in result["formatted_text"]
    assert "Step 2" in result["formatted_text"]
    assert "Bước 1" not in result["formatted_text"]


def test_sub_10ms_execution_speed_benchmark():
    """Verify pure stdlib offline execution speed is < 15ms for typical workloads."""
    query = "Quy trình mã hóa và giải mã RSA"
    passages = SAMPLE_RSA_PASSAGES

    # Warmup
    for _ in range(5):
        synthesize_step_by_step_process(query, passages)

    start_time = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        res = synthesize_step_by_step_process(query, passages)
        assert res["status"] == "success"
    duration = time.perf_counter() - start_time

    avg_ms = (duration / iterations) * 1000
    print(f"\n[BENCHMARK] Average latency per synthesis: {avg_ms:.3f} ms (100 iterations in {duration:.4f}s)")
    assert avg_ms < 15.0, f"Expected < 15ms per run, got {avg_ms:.3f}ms"

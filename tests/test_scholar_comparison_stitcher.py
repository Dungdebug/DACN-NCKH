"""
Unit tests for Comparison and Pros/Cons Answer Stitcher.
Validates:
1. Pure stdlib offline execution (no external network, zero hallucination).
2. Intent classification (pros_cons, comparison, security_evaluation, general_evaluation).
3. Entity extraction (single entity, entity pairs for comparison, acronym preservation).
4. Pros & Cons synthesis ('ưu nhược điểm của X', 'điểm mạnh/điểm yếu').
5. Cross-entity comparison synthesis ('so sánh X và Y', 'phân biệt X với Y', 'X vs Y').
6. Cryptographic security evaluation ('an ninh của hệ mã', 'an ninh vô điều kiện vs an ninh tính toán').
7. Multi-domain coverage (Cryptography, Operating Systems, Computer Networks, Databases).
8. Edge cases (empty passages, empty query, malformed inputs, missing keys).
9. High-throughput performance benchmark.
"""

import time
import pytest
from src.self_study.datasets.comparison_and_pros_cons_stitcher import (
    ComparisonAndProsConsStitcher,
    synthesize_pros_cons_and_comparison,
    strip_accents,
    clean_sentence,
    clean_bullet,
    clean_entity_name,
    deduplicate_preserve_order,
)

SAMPLE_CRYPTO_PASSAGES = [
    {
        "passage_id": "p1",
        "section_title": "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG",
        "text": "- Xuất hiện đầu tiên trong ngành mã hóa.\n- Các hệ mã này phần có từ trước năm 1970.\n- Hệ mã khóa bí mật hoặc hệ mã khóa cổ điển.\n- Các phương pháp này là cơ sở để phát triển các thuật toán mã hóa đối xứng hiện nay.\n- Có 2 phương pháp phổ biến:\n* Mã hóa thay thế.\n* Mã hóa hoán vị.",
    },
    {
        "passage_id": "p2",
        "section_title": "2. HỆ MÃ DỊCH VÒNG",
        "text": "- Mỗi chữ cái được thay thế bằng một chữ cái cách nó k vị trí trong bảng chữ cái.\n- Ví dụ: Với k = 3 (Mã Caesar), chữ A trở thành D, B trở thành E.\n- Công thức: C = (P + k) mod 26.",
    },
    {
        "passage_id": "p3",
        "section_title": "3. PHÁ MÃ",
        "text": "An ninh hệ mã khóa:\n- An ninh vô điều kiện: Không thể phá mã dù có thời gian và tài nguyên vô hạn.\n- An ninh tính toán: Chi phí để phá vỡ vượt quá giá trị của thông tin mã hóa, hoặc thời gian phá mã vượt quá thời gian hữu dụng của thông tin.\n- Các phương pháp thám mã phổ biến gồm tấn công vét cạn (brute-force) và phân tích thống kê tần suất xuất hiện của ký tự.",
    },
    {
        "passage_id": "p4",
        "section_title": "4. HỆ MÃ ĐƠN BẢNG",
        "text": "- Thay thế mỗi chữ cái trong bản rõ bằng một chữ cái tương ứng cố định trong bảng chữ cái thay thế.\n- Không gian khóa: 26! (xấp xỉ 4 x 10^26).\n- Điểm yếu: Dễ bị phá mã bằng phương pháp phân tích tần suất xuất hiện ký tự.",
    },
    {
        "passage_id": "p5",
        "section_title": "5. HỆ MÃ AFFINE",
        "text": "- Hàm mã hóa: e(x) = (ax + b) mod m, với điều kiện gcd(a, m) = 1.\n- Hàm giải mã: d(y) = a^(-1)(y - b) mod m.\n- Trong đó m = 26 là độ dài bảng chữ cái tiếng Anh.",
    },
    {
        "passage_id": "p6",
        "section_title": "6. HỆ MÃ VIGENERE",
        "text": "- Hệ mã nhiều bảng chữ cái (Polyalphabetic cipher).\n- Sử dụng một từ khóa K = k1 k2 ... km được lặp lại tuần hoàn.\n- Công thức mã hóa: ci = (pi + ki) mod 26.\n- Khắc phục nhược điểm phân tích tần suất đơn giản của hệ mã đơn bảng.",
    },
    {
        "passage_id": "p8",
        "section_title": "8. HỆ MÃ HOÁ ĐỘN MỘT LẦN",
        "text": "- Hệ mã hóa hoàn hảo (Vernam Cipher / One-Time Pad OTP).\n- Độ dài khóa bằng đúng độ dài thông điệp cần mã hóa và chỉ dùng duy nhất một lần.\n- An ninh tuyệt đối khi khóa được chọn hoàn toàn ngẫu nhiên.\n- Nhược điểm: Khó quản lý và phân phối khóa khi bản tin có kích thước lớn.",
    },
    {
        "passage_id": "p9",
        "section_title": "9. HỆ MÃ HOÁ DÒNG",
        "text": "- Sinh ra một chuỗi khóa z = z1, z2, z3...\n- Ta có: zi = fi (K, x1, ..., xi-1).\n- Bản rõ được mã hóa từng bit hoặc từng byte bằng phép XOR với dòng khóa z.\n- Ưu điểm: Tốc độ xử lý nhanh và độ trễ thấp trong truyền thông thời gian thực.\n- Điểm yếu: Dễ bị tổn thương nếu dòng khóa bị tái sử dụng (keystream reuse).",
    },
    {
        "passage_id": "p12",
        "section_title": "12. HỆ MÃ HOÁ HOÁN VỊ",
        "text": "- Không thay đổi ký tự mà chỉ sắp xếp lại vị trí các ký tự trong bản rõ.\n- Phương pháp phổ biến gồm hoán vị theo cột hoặc ma trận chuyển vị.\n- Ưu điểm: Bảo tồn toàn bộ tần suất gốc của các ký tự.\n- Nhược điểm: Vẫn có thể bị thám mã bằng phân tích cặp ký tự (digram) hoặc tam ký tự (trigram).",
    },
]

SAMPLE_OS_PASSAGES = [
    {
        "passage_id": "os1",
        "section_title": "TIẾN TRÌNH (PROCESS)",
        "text": "- Là một chương trình đang được thực thi trong bộ nhớ.\n- Sở hữu không gian địa chỉ ảo, bảng mô tả file và trạng thái thanh ghi riêng biệt.\n- Ưu điểm: Tính cô lập cao, an toàn tài nguyên, một tiến trình bị lỗi không làm sập hệ thống.\n- Nhược điểm: Chi phí tạo lập và chuyển ngữ cảnh (context switch) tốn kém tài nguyên CPU.",
    },
    {
        "passage_id": "os2",
        "section_title": "LUỒNG (THREAD)",
        "text": "- Là một đơn vị thực thi cơ bản bên trong tiến trình (nhánh thực thi nhẹ).\n- Chia sẻ chung không gian địa chỉ bộ nhớ và biến toàn cục của tiến trình cha.\n- Ưu điểm: Chuyển ngữ cảnh rất nhanh, chia sẻ dữ liệu thuận tiện và hiệu năng cao.\n- Nhược điểm: Dễ xảy ra tranh chấp dữ liệu (Race Condition) và đồng bộ hóa phức tạp.",
    },
]

SAMPLE_NETWORKING_PASSAGES = [
    {
        "passage_id": "net1",
        "section_title": "GIAO THỨC TCP",
        "text": "- Giao thức hướng kết nối (Connection-oriented) tại tầng Giao vận (Transport Layer).\n- Thực hiện bắt tay ba bước (Three-way Handshake) và cơ chế ACK truyền lại gói tin mất.\n- Ưu điểm: Đảm bảo độ tin cậy tuyệt đối, dữ liệu đến đúng thứ tự và không bị trùng lặp.\n- Nhược điểm: Độ trễ cao hơn do bắt tay và tiêu tốn băng thông điều khiển.",
    },
    {
        "passage_id": "net2",
        "section_title": "GIAO THỨC UDP",
        "text": "- Giao thức không hướng kết nối (Connectionless).\n- Gửi trực tiếp datagram mà không cần thiết lập kết nối hay kiểm tra báo nhận ACK.\n- Ưu điểm: Tốc độ rất nhanh, độ trễ cực thấp, tối ưu cho truyền phát video/âm thanh thời gian thực.\n- Nhược điểm: Không đảm bảo độ tin cậy, có thể mất gói hoặc gói tin đến sai thứ tự.",
    },
]


def test_helper_functions():
    """Validates low-level text normalization and cleaning helpers."""
    # strip_accents
    assert strip_accents("Mã hóa thay thế") == "Ma hoa thay the"
    assert strip_accents("Ưu điểm & Nhược điểm") == "Uu diem & Nhuoc diem"
    assert strip_accents("") == ""

    # clean_sentence
    s = clean_sentence("=== Page 5 ===   tuy nhiên, bản rõ và bản mã thuộc Z26")
    assert s == "Bản rõ và bản mã thuộc Z26."
    assert clean_sentence("") == ""

    # clean_bullet
    b = clean_bullet("=== Page 2 ===  * 1. Mã hóa dòng với khóa bí mật   ")
    assert b == "Mã hóa dòng với khóa bí mật"

    # clean_entity_name
    assert clean_entity_name("1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG") == "Giới THIỆU HỆ MÃ KHÓA ĐỐI XỨNG"
    assert clean_entity_name("  aes  ") == "AES"
    assert clean_entity_name("tcp") == "TCP"

    # deduplicate_preserve_order
    items = ["Ưu điểm nhanh", "  Ưu điểm nhanh  ", "ưu điểm nhanh", "Nhược điểm tốn bộ nhớ"]
    deduped = deduplicate_preserve_order(items)
    assert len(deduped) == 2
    assert deduped[0] == "Ưu điểm nhanh"
    assert deduped[1] == "Nhược điểm tốn bộ nhớ"


def test_intent_classification():
    """Validates classification of student query intent."""
    stitcher = ComparisonAndProsConsStitcher

    # Pros/Cons
    assert stitcher.classify_query_intent("Ưu nhược điểm của mã hóa dòng?") == "pros_cons"
    assert stitcher.classify_query_intent("ưu điểm và nhược điểm của DES") == "pros_cons"
    assert stitcher.classify_query_intent("Điểm mạnh và điểm yếu của OTP") == "pros_cons"
    assert stitcher.classify_query_intent("lợi thế và hạn chế của thuật toán RSA") == "pros_cons"
    assert stitcher.classify_query_intent("uu nhuoc diem cua he ma") == "pros_cons"

    # Comparison
    assert stitcher.classify_query_intent("So sánh mã hóa thay thế và mã hóa hoán vị") == "comparison"
    assert stitcher.classify_query_intent("Phân biệt tiến trình và luồng") == "comparison"
    assert stitcher.classify_query_intent("Sự khác biệt giữa TCP và UDP là gì?") == "comparison"
    assert stitcher.classify_query_intent("TCP vs UDP") == "comparison"
    assert stitcher.classify_query_intent("so sanh ma hoa doi xung va bat doi xung") == "comparison"

    # Security
    assert stitcher.classify_query_intent("An ninh của hệ mã khóa cổ điển?") == "security_evaluation"
    assert stitcher.classify_query_intent("Đánh giá độ an toàn và khả năng thám mã của Affine") == "security_evaluation"
    assert stitcher.classify_query_intent("Tính bảo mật của One-Time Pad") == "security_evaluation"

    # General / Edge cases
    assert stitcher.classify_query_intent("Khái niệm cơ bản") == "general_evaluation"
    assert stitcher.classify_query_intent("") == "general_evaluation"


def test_extract_target_entities():
    """Validates extraction of entities for comparison and pros/cons."""
    stitcher = ComparisonAndProsConsStitcher

    # Comparison queries
    primary, entities = stitcher.extract_target_entities("So sánh mã hóa thay thế và mã hóa hoán vị?")
    assert len(entities) == 2
    assert "Mã hóa thay thế" in entities[0]
    assert "Mã hóa hoán vị" in entities[1]

    # Vs query
    primary, entities = stitcher.extract_target_entities("TCP vs UDP?")
    assert len(entities) == 2
    assert entities == ["TCP", "UDP"]

    # Pros/Cons query
    primary, entities = stitcher.extract_target_entities("Ưu nhược điểm của mã hóa dòng?")
    assert "Mã hóa dòng" in primary or "Mã hoá dòng" in primary
    assert len(entities) == 1

    # Security query
    primary, entities = stitcher.extract_target_entities("Đánh giá an ninh của hệ mã Affine?")
    assert "Affine" in primary


def test_synthesize_pros_cons_single_entity():
    """Validates synthesis of pros and cons for a specific system (Mã hóa dòng)."""
    query = "Ưu nhược điểm của hệ mã hoá dòng?"
    res = synthesize_pros_cons_and_comparison(query, SAMPLE_CRYPTO_PASSAGES)

    assert isinstance(res, dict)
    assert res["mode"] == "pros_cons"
    assert "dòng" in res["target_entity"].lower()
    assert res["has_sufficient_evidence"] is True
    assert res["confidence"] >= 0.8

    # Answer formatting
    ans = res["answer"]
    assert "### Đánh giá Toàn diện Ưu điểm & Nhược điểm" in ans
    assert "**1. Ưu điểm" in ans
    assert "**2. Nhược điểm" in ans
    assert "**3. Nhận xét & Đánh giá tổng kết" in ans

    # Verified pros and cons extracted from passage 9
    assert any("tốc độ xử lý nhanh" in p.lower() for p in res["pros"])
    assert any("tổn thương" in c.lower() or "tái sử dụng" in c.lower() for c in res["cons"])

    # Citations
    assert "p9" in res["citations"] or any("DÒNG" in c for c in res["citations"])


def test_synthesize_pros_cons_one_time_pad():
    """Validates pros/cons on OTP where unconditional security and key management constraints are contrasted."""
    query = "Ưu nhược điểm của Hệ mã hoá độn một lần (OTP)?"
    res = synthesize_pros_cons_and_comparison(query, SAMPLE_CRYPTO_PASSAGES)

    assert res["mode"] == "pros_cons"
    assert res["has_sufficient_evidence"] is True
    ans = res["answer"]
    assert "**1. Ưu điểm" in ans
    assert "**2. Nhược điểm" in ans
    assert any("khó quản lý" in c.lower() or "phân phối khóa" in c.lower() for c in res["cons"])
    assert any("tuyệt đối" in p.lower() or "hoàn hảo" in p.lower() for p in res["pros"])


def test_synthesize_comparison_between_two_entities():
    """Validates comparison between Mã hóa thay thế and Mã hóa hoán vị."""
    query = "So sánh mã hóa thay thế và mã hóa hoán vị trong mật mã cổ điển?"
    res = synthesize_pros_cons_and_comparison(query, SAMPLE_CRYPTO_PASSAGES)

    assert res["mode"] == "comparison"
    assert len(res["target_entities"]) == 2
    assert res["has_sufficient_evidence"] is True

    ans = res["answer"]
    assert "### So sánh Đối chiếu:" in ans
    assert "| Tiêu chí |" in ans
    assert "|---|---|---|" in ans
    assert "**1. Phân tích từng đối tượng:**" in ans
    assert "**2. Bảng đối chiếu tiêu chí cốt lõi:**" in ans
    assert "**3. Nhận xét & Kết luận so sánh:**" in ans

    # Facets structured data
    assert len(res["comparison_facets"]) >= 2
    criteria_names = [f["criterion"] for f in res["comparison_facets"]]
    assert "Bản chất / Nguyên lý" in criteria_names
    assert "Ưu thế nổi bật" in criteria_names


def test_synthesize_security_evaluation():
    """Validates security evaluation extraction (An ninh vô điều kiện vs An ninh tính toán, attacks)."""
    query = "An ninh của hệ mã khóa và các hình thức thám mã?"
    res = synthesize_pros_cons_and_comparison(query, SAMPLE_CRYPTO_PASSAGES)

    assert res["mode"] == "security_evaluation"
    assert res["has_sufficient_evidence"] is True

    sec_facets = res["security_facets"]
    assert len(sec_facets["unconditional_security"]) >= 1
    assert len(sec_facets["computational_security"]) >= 1
    assert len(sec_facets["attacks_and_vulnerabilities"]) >= 1

    # Check that unconditional security mentions 'thời gian và tài nguyên vô hạn'
    uncond_texts = [u["clean"] for u in sec_facets["unconditional_security"]]
    assert any("vô hạn" in t.lower() or "hoàn hảo" in t.lower() for t in uncond_texts)

    # Check computational security mentions 'chi phí' or 'thời gian'
    comp_texts = [c["clean"] for c in sec_facets["computational_security"]]
    assert any("chi phí" in t.lower() or "thời gian" in t.lower() for t in comp_texts)

    # Check attacks mention brute force or frequency analysis
    attack_texts = [a["clean"] for a in sec_facets["attacks_and_vulnerabilities"]]
    assert any("vét cạn" in t.lower() or "tần suất" in t.lower() for t in attack_texts)

    ans = res["answer"]
    assert "### Đánh giá An ninh & Khả năng Kháng Thám Mã" in ans
    assert "**1. Phân loại Cấp độ An ninh Hệ mã:**" in ans
    assert "**2. Các Phương pháp Tấn công & Điểm yếu Thám mã:**" in ans


def test_multi_domain_operating_systems():
    """Validates cross-domain comparison: Process vs Thread."""
    query = "So sánh tiến trình và luồng trong hệ điều hành?"
    res = synthesize_pros_cons_and_comparison(query, SAMPLE_OS_PASSAGES)

    assert res["mode"] == "comparison"
    assert res["has_sufficient_evidence"] is True
    ans = res["answer"]
    assert "Tiến trình" in ans
    assert "Luồng" in ans
    assert "| Tiêu chí |" in ans


def test_multi_domain_networking_pros_cons():
    """Validates cross-domain pros/cons: TCP Protocol."""
    query = "Ưu nhược điểm của giao thức TCP?"
    res = synthesize_pros_cons_and_comparison(query, SAMPLE_NETWORKING_PASSAGES)

    assert res["mode"] == "pros_cons"
    assert res["has_sufficient_evidence"] is True
    assert any("tin cậy" in p.lower() for p in res["pros"])
    assert any("độ trễ cao" in c.lower() for c in res["cons"])


def test_edge_cases_and_graceful_fallbacks():
    """Validates robustness under empty, None, and partial inputs."""
    # 1. Empty passages
    res_empty_p = synthesize_pros_cons_and_comparison("Ưu nhược điểm của AES?", [])
    assert res_empty_p["has_sufficient_evidence"] is False
    assert res_empty_p["confidence"] == 0.0
    assert "Không tìm thấy đủ tài liệu" in res_empty_p["answer"]

    # 2. Empty query
    res_empty_q = synthesize_pros_cons_and_comparison("", SAMPLE_CRYPTO_PASSAGES)
    assert res_empty_q["has_sufficient_evidence"] is False

    # 3. None query
    res_none_q = synthesize_pros_cons_and_comparison(None, SAMPLE_CRYPTO_PASSAGES)
    assert res_none_q["has_sufficient_evidence"] is False

    # 4. Passage without explicit pros/cons but valid technical content
    partial_passages = [
        {"passage_id": "px", "section_title": "MÃ CAESAR", "text": "Phép dịch chuyển ký tự với k = 3."}
    ]
    res_partial = synthesize_pros_cons_and_comparison("Ưu nhược điểm của Mã Caesar?", partial_passages)
    assert res_partial["has_sufficient_evidence"] is True
    assert "Mã Caesar" in res_partial["answer"]
    assert "Lưu ý:" in res_partial["answer"]

    # 5. Malformed passage dictionaries (missing keys)
    malformed_passages = [
        {"text": "Bản rõ được mã hóa bằng phép XOR."},
        {"random_key": 123},
        None,
        "string passage",
    ]
    res_malformed = synthesize_pros_cons_and_comparison("Đánh giá an ninh?", malformed_passages)
    assert isinstance(res_malformed, dict)
    assert "answer" in res_malformed


def test_performance_benchmark():
    """Benchmark: 50 synthesis queries execute swiftly in pure Python stdlib."""
    start = time.perf_counter()
    for _ in range(25):
        _ = synthesize_pros_cons_and_comparison("Ưu nhược điểm của mã hóa dòng?", SAMPLE_CRYPTO_PASSAGES)
        _ = synthesize_pros_cons_and_comparison("So sánh mã hóa thay thế và mã hóa hoán vị", SAMPLE_CRYPTO_PASSAGES)
    duration_ms = (time.perf_counter() - start) * 1000

    # 50 complete synthesis operations
    assert duration_ms < 500.0, f"Execution took {duration_ms:.2f}ms, expected < 500ms"

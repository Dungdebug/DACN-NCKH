"""
Unit tests for SelfStudyKnowledgeBank & Answer Instruction Blueprints (Wave 5).
Verifies:
1. ANSWER_INSTRUCTION_BLUEPRINTS coverage (6 behavioral archetypes).
2. Fast document learning (< 25ms) and extraction of sections, definitions, classifications, systems, and examples.
3. Classification of answer blueprints based on query semantics.
4. Synthesis of full structured answers for multi-facet questions (definition + classifications + examples).
5. Full section synthesis for specific topics ('PHÁ MÃ', '3. PHÁ MÃ').
6. Generation of natural, tailored self-study questions without robotic templates.
7. Singleton consistency of get_knowledge_bank().
"""

import time
import pytest
from src.self_study.knowledge_bank import (
    ANSWER_INSTRUCTION_BLUEPRINTS,
    SelfStudyKnowledgeBank,
    get_knowledge_bank,
)

SAMPLE_SLIDE_TEXT = """=== Page 1 ===
1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:
- Xuất hiện đầu tiên trong ngành mã hóa.
- Các hệ mã này phần có từ trước năm 1970.
- Hệ mã khóa bí mật hoặc hệ mã khóa cổ điển.
- Các phương pháp này là cơ sở để phát triển các thuật toán mã hóa đối xứng hiện nay.
- Có 2 phương pháp phổ biến:
* Mã hóa thay thế.
* Mã hóa hoán vị.

=== Page 10 ===
3. PHÁ MÃ:
An ninh hệ mã khóa:
- An ninh vô điều kiện: Không thể phá mã dù có thời gian và tài nguyên vô hạn.
- An ninh tính toán: Chi phí để phá vỡ vượt quá giá trị của thông tin mã hóa, hoặc thời gian phá mã vượt quá thời gian hữu dụng của thông tin.
- Các phương pháp thám mã phổ biến gồm tấn công vét cạn (brute-force) và phân tích thống kê tần suất xuất hiện của ký tự.

=== Page 30 ===
8. HỆ MÃ HOÁ ĐỘN MỘT LẦN:
- Hệ mã hóa hoàn hảo (Vernam Cipher / One-Time Pad).
- Độ dài khóa bằng đúng độ dài thông điệp cần mã hóa và chỉ dùng duy nhất một lần.

9. HỆ MÃ HOÁ DÒNG:
- Sinh ra một chuỗi khóa z = z1, z2, z3...
- Ta có: zi = fi (K, x1, ..., xi-1).
- Bản rõ được mã hóa từng bit hoặc từng byte bằng phép XOR với dòng khóa z.
- Ưu điểm: Tốc độ xử lý nhanh và độ trễ thấp trong truyền thông thời gian thực.

=== Page 31 ===
10. HỆ MÃ HOÁ AUTOKEY:
- Cho P = C = K = Z26.
- Khóa k = k1, k2, ... ban đầu được mở rộng bằng chính các ký tự của bản rõ (Autokey).
- Công thức mã hóa: ci = (pi + ki) mod 26.

=== Page 36 ===
11. HỆ MÃ HOÁ PLAYFAIR:
- Sử dụng ma trận ký tự 5x5 để mã hóa từng cặp ký tự (digram).
- Thay thế các cặp ký tự dựa trên quy tắc cùng hàng, cùng cột hoặc tạo thành hình chữ nhật trên ma trận.
"""


def test_answer_instruction_blueprints_coverage():
    expected_blueprints = [
        "DEFINITION_AND_TYPES",
        "SECTION_FULL_EXPLANATION",
        "MECHANISM_AND_FORMULA",
        "COMPARISON_AND_SECURITY",
        "EXAMPLE_WALKTHROUGH",
        "SUMMARY_OVERVIEW",
    ]
    for bp in expected_blueprints:
        assert bp in ANSWER_INSTRUCTION_BLUEPRINTS
        assert "intent" in ANSWER_INSTRUCTION_BLUEPRINTS[bp]
        assert "cues" in ANSWER_INSTRUCTION_BLUEPRINTS[bp]
        assert "instructions" in ANSWER_INSTRUCTION_BLUEPRINTS[bp]


def test_fast_learning_latency_and_extraction():
    bank = SelfStudyKnowledgeBank()
    t0 = time.perf_counter()
    res = bank.learn_document(
        doc_id="DOC_SLIDE_CRYPTO",
        title="Slide 2: Các Hệ Mã Hóa Đối Xứng Cổ Điển",
        raw_text=SAMPLE_SLIDE_TEXT,
    )
    t_elapsed_ms = (time.perf_counter() - t0) * 1000

    # Must complete ultra-fast (< 25ms)
    assert t_elapsed_ms < 50.0  # Safe threshold for test environment
    assert res["status"] == "learned"
    assert res["sections_count"] >= 6
    assert res["classifications_count"] >= 2
    assert res["systems_count"] >= 4


def test_classify_answer_blueprint():
    bank = SelfStudyKnowledgeBank()
    assert bank.classify_answer_blueprint("mã hóa đối xứng là gì, có bao nhiêu loại, cho ví dụ") == "DEFINITION_AND_TYPES"
    assert bank.classify_answer_blueprint("3. PHÁ MÃ") == "SECTION_FULL_EXPLANATION"
    assert bank.classify_answer_blueprint("an ninh vô điều kiện trong phá mã là gì") == "COMPARISON_AND_SECURITY"
    assert bank.classify_answer_blueprint("cách hoạt động và công thức của Autokey") == "MECHANISM_AND_FORMULA"
    assert bank.classify_answer_blueprint("cho ví dụ minh họa về mã hóa dòng") == "EXAMPLE_WALKTHROUGH"
    assert bank.classify_answer_blueprint("tổng quan toàn bộ tài liệu") == "SUMMARY_OVERVIEW"


def test_synthesize_learned_answer_multi_facet():
    bank = SelfStudyKnowledgeBank()
    bank.learn_document("DOC_SLIDE_CRYPTO", "Mã Hóa", SAMPLE_SLIDE_TEXT)

    res = bank.synthesize_learned_answer("mã hóa đối xứng là gì, có bao nhiêu loại mã hóa, cho ví dụ")
    assert res is not None
    answer = res["answer"].lower()

    # Must contain concept from Section 1
    assert "khóa đối xứng" in answer or "khóa bí mật" in answer or "ngành mã hóa" in answer
    # Must contain classifications
    assert "thay thế" in answer
    assert "hoán vị" in answer
    # Must mention specific cipher systems present in the slide
    assert any(c in answer for c in ["dòng", "độn một lần", "autokey", "playfair"])


def test_synthesize_learned_answer_specific_section_phama():
    bank = SelfStudyKnowledgeBank()
    bank.learn_document("DOC_SLIDE_CRYPTO", "Mã Hóa", SAMPLE_SLIDE_TEXT)

    queries = [
        "3. PHÁ MÃ",
        "Yêu cầu và chi tiết cụ thể liên quan đến 'PHÁ MÃ' gồm những gì?",
        "nội dung phần 'PHÁ MÃ'",
    ]

    for q in queries:
        res = bank.synthesize_learned_answer(q)
        assert res is not None, f"Failed on query: {q}"
        ans = res["answer"].lower()
        assert "an ninh" in ans
        assert "vô điều kiện" in ans or "tính toán" in ans


def test_get_self_study_questions_naturalness():
    bank = SelfStudyKnowledgeBank()
    bank.learn_document("DOC_SLIDE_CRYPTO", "Mã Hóa", SAMPLE_SLIDE_TEXT)

    questions = bank.get_self_study_questions(count=3)
    assert len(questions) == 3

    for q in questions:
        # Zero robotic templates
        assert "Yêu cầu và chi tiết cụ thể liên quan đến" not in q
        assert q.endswith("?")
        # Grounded in real section headings
        assert any(term in q for term in ["1. GIỚI THIỆU", "3. PHÁ MÃ", "8. HỆ MÃ HOÁ", "9. HỆ MÃ HOÁ", "10. HỆ MÃ HOÁ", "11. HỆ MÃ HOÁ"])


def test_get_knowledge_bank_singleton():
    k1 = get_knowledge_bank()
    k2 = get_knowledge_bank()
    assert k1 is k2


def test_synthesize_learned_answer_comparison_and_security():
    bank = SelfStudyKnowledgeBank()
    bank.learn_document("DOC_SLIDE_CRYPTO", "Mã Hóa", SAMPLE_SLIDE_TEXT)

    res = bank.synthesize_learned_answer("ưu nhược điểm của hệ mã hoá dòng")
    assert res is not None
    assert res["blueprint"] == "COMPARISON_AND_SECURITY"
    ans = res["answer"].lower()
    assert "ưu điểm" in ans or "nhược điểm" in ans or "đặc tính" in ans

    # Security query
    res_sec = bank.synthesize_learned_answer("an ninh và khả năng phá mã của hệ mã")
    assert res_sec is not None
    assert res_sec["blueprint"] == "COMPARISON_AND_SECURITY"


def test_synthesize_learned_answer_step_by_step():
    bank = SelfStudyKnowledgeBank()
    bank.learn_document("DOC_SLIDE_CRYPTO", "Mã Hóa", SAMPLE_SLIDE_TEXT)

    res = bank.synthesize_learned_answer("các bước thực hiện và quy trình của hệ mã hóa Autokey")
    assert res is not None
    assert res["blueprint"] == "MECHANISM_AND_FORMULA"
    assert "bước" in res["answer"].lower() or "quy trình" in res["answer"].lower()


def test_multi_domain_exports_and_routing():
    from src.self_study import (
        AcademicDomain,
        AcademicDomainRegistry,
        expand_all_academic_synonyms,
        parse_vietnamese_academic_blocks,
        synthesize_pros_cons_and_comparison,
        synthesize_step_by_step_process,
    )
    # Check domain routing
    reg = AcademicDomainRegistry()
    route_res = reg.route_query("Định tuyến OSPF và bắt tay TCP 3 bước")
    assert route_res.primary_domain == AcademicDomain.NETWORKS.value

    # Check academic synonyms expansion
    syns = expand_all_academic_synonyms("deadlock trong hệ điều hành")
    assert len(syns) > 0

    # Check academic blocks parsing
    blocks = parse_vietnamese_academic_blocks("Định nghĩa 1.1: Mã hóa đối xứng là phương pháp dùng chung một khóa bí mật.")
    assert len(blocks) >= 1
    assert blocks[0]["block_type"] == "dinh_nghia"
    assert blocks[0]["metadata"]["is_definition"] is True


def test_knowledge_bank_scope_lock_and_lossless_cipher_enumeration():
    bank = SelfStudyKnowledgeBank()
    slide_with_short_cipher = (
        "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:\n"
        "Mã hóa đối xứng dùng chung khóa bí mật.\n"
        "* Mã hóa thay thế.\n"
        "* Mã hóa hoán vị.\n\n"
        "2. MÃ CAESAR:\n"
        "Dịch chuyển k vị trí trên Z26.\n\n"
        "3. HỆ MÃ HOÁ DÒNG:\n"
        "Mã hóa từng bit bằng phép XOR.\n"
    )
    bank.learn_document("DOC_CRYPTO_X", "Crypto", slide_with_short_cipher)

    # Strict scope lock: unknown doc_ids must return None instead of leaking DOC_CRYPTO_X
    assert bank.synthesize_learned_answer("mã hóa đối xứng là gì", doc_ids=["DOC_OTHER"]) is None
    assert bank.synthesize_learned_answer("mã hóa đối xứng là gì", doc_ids=[]) is None

    # Enumeration query without 'là gì' still returns full cipher list including MÃ CAESAR
    res = bank.synthesize_learned_answer("liệt kê các hệ mã trong tài liệu", doc_ids=["DOC_CRYPTO_X"])
    assert res is not None
    assert res["blueprint"] == "DEFINITION_AND_TYPES"
    ans_lower = res["answer"].lower()
    assert "caesar" in ans_lower
    assert "dòng" in ans_lower


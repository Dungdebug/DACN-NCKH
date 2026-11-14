"""
Unit Test Suite for Pillar 4: Academic Persona & Pedagogical Response Architecture.
Tests:
- AcademicIntent classification (greetings, farewells, gratitude, capabilities, out-of-scope, pedagogical).
- Polite social dialogue and meta identity invariants (100% offline, zero cloud API).
- Safe calibrated refusal with scientific integrity.
- 4-Tier Pedagogical formatting (Tier 1: Definition -> Tier 2: Taxonomy -> Tier 3: Concrete Systems -> Tier 4: Formulas & Examples).
- End-to-end passage synthesis.
- SFT dataset export & few-shot prompts for offline training.
"""

import pytest
from typing import List, Dict, Any

from src.self_study.persona import (
    AcademicIntent,
    PedagogicalTier,
    PedagogicalResponse,
    AcademicPersonaEngine,
    ACADEMIC_PERSONA_DATASET,
    get_academic_persona_dataset,
    export_sft_training_samples,
    export_few_shot_prompt,
    get_academic_persona_engine,
    strip_accents,
)


@pytest.fixture
def engine() -> AcademicPersonaEngine:
    return AcademicPersonaEngine()


# ============================================================================
# 1. INTENT CLASSIFICATION TESTS
# ============================================================================

def test_intent_classification_greetings(engine: AcademicPersonaEngine):
    greeting_queries = [
        "Xin chào!",
        "chào bạn",
        "Chào trợ lý",
        "Hello",
        "hi bot",
        "Kính chào thầy cô",
        "chào buổi sáng",
        "chào nha",
    ]
    for q in greeting_queries:
        intent = engine.classify_intent(q)
        assert intent == AcademicIntent.GREETING, f"Failed for greeting: '{q}' (got {intent})"


def test_intent_classification_farewell(engine: AcademicPersonaEngine):
    farewell_queries = [
        "Tạm biệt",
        "tam biet",
        "bye",
        "goodbye",
        "hẹn gặp lại nhé",
        "chúc ngủ ngon",
    ]
    for q in farewell_queries:
        intent = engine.classify_intent(q)
        assert intent == AcademicIntent.FAREWELL, f"Failed for farewell: '{q}' (got {intent})"


def test_intent_classification_gratitude(engine: AcademicPersonaEngine):
    gratitude_queries = [
        "Cảm ơn bạn!",
        "cam on",
        "Thank you",
        "thanks!",
        "cảm ơn nhiều nhé",
        "rất hữu ích cảm ơn",
    ]
    for q in gratitude_queries:
        intent = engine.classify_intent(q)
        assert intent == AcademicIntent.GRATITUDE, f"Failed for gratitude: '{q}' (got {intent})"


def test_intent_classification_system_capabilities(engine: AcademicPersonaEngine):
    capability_queries = [
        "Bạn là ai?",
        "bạn làm được gì",
        "hệ thống này hoạt động như thế nào",
        "offline hay online",
        "có cần mạng không",
        "dữ liệu có gửi ra ngoài không",
    ]
    for q in capability_queries:
        intent = engine.classify_intent(q)
        assert intent == AcademicIntent.SYSTEM_CAPABILITY, f"Failed for capability: '{q}' (got {intent})"


def test_intent_classification_out_of_scope(engine: AcademicPersonaEngine):
    oos_queries = [
        "Thời tiết hôm nay ở Hà Nội thế nào?",
        "giá vàng hôm nay bao nhiêu một lượng",
        "kết quả xổ số miền bắc",
        "đội bóng nào vô địch world cup",
        "hướng dẫn cách nấu món phở bò",
        "thủ đô của nước Pháp là gì",
    ]
    for q in oos_queries:
        intent = engine.classify_intent(q)
        assert intent == AcademicIntent.OUT_OF_SCOPE, f"Failed for out-of-scope: '{q}' (got {intent})"


def test_intent_classification_academic_pedagogical(engine: AcademicPersonaEngine):
    academic_queries = [
        "Hệ mã dịch vòng là gì?",
        "Giải thích thuật toán Round Robin trong hệ điều hành",
        "Chuẩn hóa dữ liệu 3NF và BCNF khác nhau như thế nào?",
        "Nguyên lý bắt tay 3 bước của giao thức TCP",
        "Hàm mất mát Mean Squared Error trong hồi quy tuyến tính",
    ]
    for q in academic_queries:
        intent = engine.classify_intent(q)
        assert intent == AcademicIntent.ACADEMIC_PEDAGOGICAL, f"Failed for academic: '{q}' (got {intent})"


def test_intent_classification_clarification_on_noise(engine: AcademicPersonaEngine):
    assert engine.classify_intent("") == AcademicIntent.CLARIFICATION
    assert engine.classify_intent("   ") == AcademicIntent.CLARIFICATION
    assert engine.classify_intent("?") == AcademicIntent.CLARIFICATION


# ============================================================================
# 2. POLITE SOCIAL RESPONSES & METADATA TESTS
# ============================================================================

def test_social_response_greeting_invariants(engine: AcademicPersonaEngine):
    resp = engine.handle_social_query("Xin chào trợ lý")
    assert "Trợ lý Nghiên cứu Khoa học Tự chủ" in resp
    assert "100% offline" in resp
    assert "4 tầng" in resp
    # No robotic preambles
    assert "Là một mô hình ngôn ngữ lớn" not in resp


def test_social_response_system_capability_invariants(engine: AcademicPersonaEngine):
    resp = engine.handle_social_query("Bạn là ai và có gửi dữ liệu ra ngoài không?")
    assert "Bảo mật Tự chủ Tuyệt đối (100% Offline)" in resp
    assert "Cấu trúc Phản hồi Sư phạm 4 Tầng" in resp
    assert "Zero Hallucination" in resp
    assert "không có bất kỳ byte dữ liệu nào" in resp.lower()


def test_social_response_gratitude_and_farewell(engine: AcademicPersonaEngine):
    grat_resp = engine.handle_social_query("Cảm ơn bạn nhé")
    assert "đồng hành cùng công tác nghiên cứu" in grat_resp

    farewell_resp = engine.handle_social_query("Tạm biệt trợ lý")
    assert "nghiên cứu và học tập hiệu quả" in farewell_resp


# ============================================================================
# 3. SAFE OUT-OF-SCOPE REFUSAL TESTS
# ============================================================================

def test_safe_refusal_scientific_integrity(engine: AcademicPersonaEngine):
    refusal = engine.format_safe_refusal(
        query="Ai là cầu thủ bóng đá xuất sắc nhất mọi thời đại?",
        reason="Dữ liệu thể thao giải trí không thuộc phạm vi nghiên cứu khoa học của tài liệu.",
        domain_hint="Thể thao & Giải trí",
        available_topics=["Hệ mã dịch vòng", "Lập lịch CPU", "Chuẩn hóa CSDL"],
    )
    assert "Thông báo Phạm vi Tài liệu & Chính trực Khoa học" in refusal
    assert "không tìm thấy đủ căn cứ chứng cứ xác thực" in refusal
    assert "Dữ liệu thể thao giải trí" in refusal
    assert "Thể thao & Giải trí" in refusal
    assert "Hệ mã dịch vòng" in refusal
    assert "Lập lịch CPU" in refusal


# ============================================================================
# 4. 4-TIER PEDAGOGICAL FORMATTING TESTS
# ============================================================================

def test_format_pedagogical_response_numeric_style(engine: AcademicPersonaEngine):
    res = engine.format_pedagogical_response(
        topic="Hệ mã dịch vòng (Shift Cipher)",
        definition="Hệ mã dịch vòng là hệ mã hóa thay thế đơn bảng trên vành Z26.",
        taxonomy=["* Mật mã cổ điển", "* Mã hóa thay thế đơn ký tự"],
        concrete_systems=["Mã Caesar (k = 3)", "Mã ROT13 (k = 13)"],
        formulas_or_examples=[
            "Hàm mã hóa: e_k(x) = (x + k) mod 26",
            "Ví dụ: Với k = 3, 'B' (x=1) mã hóa thành 'E' (y=4).",
        ],
        citations=["[1]", "[2]"],
        header_style="numeric",
    )
    assert "### 🎓 Phân tích Sư phạm: Hệ mã dịch vòng" in res
    assert "1. Định nghĩa & Khái niệm Cốt lõi" in res
    assert "2. Phân loại & Cấu trúc Phân nhánh" in res
    assert "3. Các Hệ Thống / Thuật Toán Tiêu Biểu" in res
    assert "4. Công thức Toán học & Ví dụ Minh họa" in res
    assert "e_k(x) = (x + k) mod 26" in res
    assert "[1] [2]" in res


def test_format_pedagogical_response_tier_tag_style(engine: AcademicPersonaEngine):
    res = engine.format_pedagogical_response(
        topic="Lập lịch CPU",
        definition="Lập lịch CPU là cơ chế phân bổ thời gian xử lý của CPU cho các tiến trình.",
        taxonomy=["Lập lịch độc quyền", "Lập lịch không độc quyền"],
        concrete_systems=["FCFS", "SJF", "Round Robin"],
        formulas_or_examples=["Time Quantum q = 10ms", "T_turnaround = T_completion - T_arrival"],
        citations=["[OS-Lec]"],
        header_style="tier_tag",
    )
    assert "[Tầng 1 - Định nghĩa]" in res
    assert "[Tầng 2 - Phân loại]" in res
    assert "[Tầng 3 - Hệ cụ thể]" in res
    assert "[Tầng 4 - Công thức & Ví dụ]" in res
    assert "Round Robin" in res
    assert "[OS-Lec]" in res


# ============================================================================
# 5. END-TO-END PASSAGE SYNTHESIS TESTS
# ============================================================================

def test_synthesize_pedagogical_answer_from_academic_passages(engine: AcademicPersonaEngine):
    passages = [
        {
            "section_title": "1. TỔNG QUAN HỆ MÃ KHÓA ĐỐI XỨNG",
            "text": (
                "Hệ mã khóa đối xứng xuất hiện đầu tiên trong lịch sử mật mã học.\n"
                "Trong mật mã cổ điển có hai phương pháp phổ biến nhất:\n"
                "* Mã hóa thay thế\n"
                "* Mã hóa hoán vị\n"
            ),
            "citation_label": "[Crypto-1]",
        },
        {
            "section_title": "2. HỆ MÃ DỊCH VÒNG VÀ MÃ CAESAR",
            "text": (
                "Hệ mã dịch vòng có không gian P = C = K = Z26.\n"
                "Công thức hàm mã hóa: e_k(x) = (x + k) mod 26.\n"
                "Công thức hàm giải mã: d_k(y) = (y - k) mod 26.\n"
                "Ví dụ: Khi chọn khóa dịch chuyển k = 3, đây chính là mã Caesar.\n"
            ),
            "citation_label": "[Crypto-2]",
        },
    ]

    result = engine.synthesize_pedagogical_answer_from_passages(
        query="Trình bày hệ mã khóa đối xứng và hệ mã dịch vòng",
        passages=passages,
    )

    assert result["success"] is True
    assert result["intent"] == AcademicIntent.ACADEMIC_PEDAGOGICAL.value
    assert len(result["tier_1_definition"]) > 0
    assert len(result["tier_2_taxonomy"]) >= 2
    assert "Mã hóa thay thế" in str(result["tier_2_taxonomy"])
    assert "Mã hóa hoán vị" in str(result["tier_2_taxonomy"])
    assert len(result["tier_3_concrete_systems"]) >= 1
    assert any("Caesar" in s or "Dịch vòng" in s or "dich vong" in strip_accents(s.lower()) for s in result["tier_3_concrete_systems"])
    assert len(result["tier_4_formulas_examples"]) >= 1
    assert "1. Định nghĩa & Khái niệm Cốt lõi" in result["answer"]
    assert "4. Công thức Toán học & Ví dụ Minh họa" in result["answer"]


def test_synthesize_pedagogical_answer_empty_passages(engine: AcademicPersonaEngine):
    result = engine.synthesize_pedagogical_answer_from_passages(
        query="Khái niệm vi xử lý lượng tử",
        passages=[],
    )
    assert result["success"] is False
    assert result["intent"] == AcademicIntent.OUT_OF_SCOPE.value
    assert "không tìm thấy đủ căn cứ chứng cứ xác thực" in result["answer"]


# ============================================================================
# 6. SFT DATASET & FEW-SHOT EXPORT TESTS
# ============================================================================

def test_academic_persona_dataset_integrity():
    dataset = get_academic_persona_dataset()
    assert len(dataset) >= 18, f"Dataset has only {len(dataset)} items"

    categories = {entry["category"] for entry in dataset}
    assert "social_greeting" in categories
    assert "safe_refusal" in categories
    assert "pedagogical_4tier" in categories
    assert "clarification" in categories

    domains = {entry["domain"] for entry in dataset}
    for required_domain in (
        "cryptography",
        "operating_systems",
        "database",
        "networking",
        "machine_learning",
        "dsa",
        "software_engineering",
    ):
        assert required_domain in domains, f"Missing domain {required_domain}"

    for item in dataset:
        assert "id" in item
        assert "query" in item
        assert "response" in item
        assert "category" in item
        assert len(item["query"].strip()) > 0
        assert len(item["response"].strip()) > 0


def test_export_sft_training_samples_structure():
    sft_samples = export_sft_training_samples()
    assert len(sft_samples) == len(ACADEMIC_PERSONA_DATASET)

    first = sft_samples[0]
    assert "id" in first
    assert "messages" in first
    assert len(first["messages"]) == 3
    assert first["messages"][0]["role"] == "system"
    assert "Sovereign Academic Research Assistant" in first["messages"][0]["content"]
    assert first["messages"][1]["role"] == "user"
    assert first["messages"][2]["role"] == "assistant"


def test_export_few_shot_prompt():
    prompt = export_few_shot_prompt(category="pedagogical_4tier", k=2)
    assert "Sinh viên / Nhà nghiên cứu:" in prompt
    assert "Trợ lý NCKH:" in prompt
    assert "### 🎓 Phân tích Sư phạm" in prompt


def test_singleton_accessor():
    e1 = get_academic_persona_engine()
    e2 = get_academic_persona_engine()
    assert e1 is e2


def test_document_analyzer_persona_integration():
    from src.web.document_analyzer import DocumentAnalyzer

    ans_greet, cits_greet = DocumentAnalyzer.find_best_grounded_answer(
        query="Xin chào bạn!",
        evidence_passages=[],
    )
    assert "Trợ lý Nghiên cứu Khoa học Tự chủ" in ans_greet
    assert "100% offline" in ans_greet
    assert cits_greet == []

    ans_who, _ = DocumentAnalyzer.find_best_grounded_answer(
        query="Bạn là ai và hệ thống này hoạt động như thế nào?",
        evidence_passages=[],
    )
    assert "Tự chủ" in ans_who
    assert "4 Tầng" in ans_who


def test_web_app_persona_fast_path():
    from fastapi.testclient import TestClient
    from src.web.app import app

    client = TestClient(app)

    # 1. Create a session with zero attached documents and send a greeting
    sess_res = client.post("/api/chat/sessions", json={"title": "Persona Test"})
    assert sess_res.status_code == 200
    session_id = sess_res.json()["session_id"]

    msg_res = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"query": "Xin chào trợ lý!", "mode": "hybrid", "answer_mode": "balanced"},
    )
    assert msg_res.status_code == 200
    data = msg_res.json()
    assert data["refused"] is False
    assert data["routing"] == "PERSONA_GREETING"
    assert "100% offline" in data["answer"]

    # 2. Direct /api/chat capability check
    chat_res = client.post(
        "/api/chat",
        json={"query": "Bạn là ai và dữ liệu có gửi ra ngoài không?", "mode": "hybrid"},
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["refused"] is False
    assert "100% Offline" in chat_data["answer"]
    assert chat_data["memory_diagnostics"]["memory_level"] == "Persona"

    # Cleanup test session
    client.delete(f"/api/chat/sessions/{session_id}")

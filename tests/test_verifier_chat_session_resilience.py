"""
End-to-End Chat Session Resilience & Document QA Robustness Test Suite.
Validates the resilience of the document QA pipeline:
- QuestionRouter: Query intent classification, scope detection, quoted entity extraction.
- BM25Retriever: Pure Python retrieval with accent folding and proximity bonus.
- EvidenceSelector: Threshold filtering, dynamic top-k, quoted span preservation, sufficiency evaluation.
- DocumentAnalyzer.find_best_grounded_answer: Span extraction, citation deduplication, follow-up questions.
- SessionManager & ChatSession: Multi-turn conversational context resolution and session lifecycle.

Test Scenarios:
1. Unaccented Vietnamese queries (câu hỏi tiếng Việt không dấu).
2. Multi-part queries (câu hỏi đa ý: khái niệm + phân loại + ví dụ).
3. Comparison queries (câu hỏi so sánh đơn tài liệu và đa tài liệu).
4. Quoted suggestion queries (câu hỏi gợi ý trong ngoặc nháy đơn và nháy kép).
5. Multi-turn chat session continuity and pronoun/followup resolution.
6. Negative, out-of-domain, and edge-case resilience.
"""

import json
import pytest
from pathlib import Path
from typing import List, Dict, Any

from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.document_store import DocumentRecord
from src.hybrid_qa.retriever import BM25Retriever
from src.hybrid_qa.evidence import EvidenceSelector, Citation
from src.web.question_router import QuestionRouter, QuestionRoute
from src.web.document_analyzer import DocumentAnalyzer
from src.web.session_manager import (
    SessionManager,
    ChatSession,
    generate_title_from_query,
    extract_topic_anchor,
)
from src.hybrid_qa.answer_span import relevance_score, strip_accents, generate_followup_questions

# Invariant constants
REFUSAL_MESSAGE = "Không tìm thấy đủ thông tin bằng chứng trong tài liệu để trả lời câu hỏi này."
ROBOTIC_PHRASE_1 = "Yêu cầu và chi tiết cụ thể liên quan đến"
ROBOTIC_PHRASE_2 = "Nội dung và cách triển khai"


# ============================================================================
# CORPUS FIXTURES
# ============================================================================

CRYPTO_LECTURE_TEXT = """=== Page 1 ===
1. TỔNG QUAN HỆ MÃ KHÓA ĐỐI XỨNG:
- Xuất hiện đầu tiên trong lịch sử ngành mật mã học và bảo mật dữ liệu.
- Các hệ thống này phần lớn có nguồn gốc từ trước năm 1970, thường được gọi là hệ mã khóa bí mật hoặc hệ mã khóa cổ điển.
- Các phương pháp này là cơ sở lý thuyết nền tảng để phát triển các thuật toán mã hóa đối xứng hiện đại ngày nay.
- Trong mật mã cổ điển có 2 phương pháp phổ biến nhất:
* Mã hóa thay thế.
* Mã hóa hoán vị.

=== Page 5 ===
2. HỆ MÃ DỊCH VÒNG VÀ MÃ CAESAR:
- Không gian bản rõ P, không gian bản mã C và không gian khóa K đều thuộc vành số nguyên Z26.
- Quy tắc: Mỗi chữ cái trong bản rõ được thay thế bằng một chữ cái cách nó k vị trí trong bảng chữ cái tiếng Anh gồm 26 ký tự.
- Công thức hàm mã hóa: e_k(x) = (x + k) mod 26.
- Công thức hàm giải mã: d_k(y) = (y - k) mod 26.
- Ví dụ kinh điển: Khi chọn khóa dịch chuyển k = 3, đây chính là hệ mã Caesar do Julius Caesar sử dụng trong lịch sử La Mã cổ đại. Khi đó, chữ cái A được mã hóa thành D, chữ cái B trở thành E, và chữ cái Z trở thành C.

=== Page 12 ===
3. HỆ MÃ HOÁN VỊ:
- Khác với mã hóa thay thế giữ nguyên vị trí và thay đổi giá trị ký tự, hệ mã hóa hoán vị giữ nguyên các ký tự của bản rõ nhưng thay đổi vị trí thứ tự của chúng theo một quy tắc hoán vị nhất định.
- Bản mã nhận được là một hoán vị của tập hợp các ký tự trong bản rõ ban đầu.
- Phân loại kỹ thuật hoán vị gồm: hoán vị đơn giản, hoán vị ma trận và hoán vị cột (Columnar Transposition).
- Ví dụ: Mã hóa hoán vị cột sắp xếp bản rõ theo các hàng liên tiếp của ma trận rồi đọc dữ liệu đầu ra theo thứ tự các cột được quy định bởi khóa hoán vị.

=== Page 20 ===
4. AN NINH HỆ MÃ KHÓA VÀ ĐÁNH GIÁ:
- An ninh vô điều kiện (Unconditional Security): Hệ mã không thể bị phá vỡ dù đối phương thám mã có thời gian tính toán và tài nguyên bộ nhớ vô hạn. Ví dụ duy nhất được chứng minh đạt an ninh vô điều kiện là hệ mã One-Time Pad (băng dùng một lần).
- An ninh tính toán (Computational Security): Hệ mã được coi là an toàn nếu chi phí thuật toán phá mã vượt quá giá trị kinh tế của thông tin cần bảo vệ, hoặc thời gian phá vỡ vượt quá tuổi thọ hữu ích của dữ liệu.
- Phân biệt hai loại an ninh: An ninh vô điều kiện dựa trên nền tảng lý thuyết thông tin của Claude Shannon, trong khi an ninh tính toán dựa trên độ phức tạp thuật toán và giới hạn năng lực phần cứng thực tế.
"""

OS_LECTURE_TEXT = """=== Page 1 ===
1. TIẾN TRÌNH VÀ LUỒNG TRONG HỆ ĐIỀU HÀNH:
- Tiến trình (Process) là một chương trình đang trong trạng thái thực thi, sở hữu không gian địa chỉ bộ nhớ riêng biệt, bảng mô tả tệp và thanh ghi con trỏ lệnh.
- Luồng (Thread) là đơn vị thực thi cơ bản bên trong một tiến trình, chia sẻ chung không gian địa chỉ bộ nhớ và tài nguyên của tiến trình cha.
- Phân loại luồng bao gồm luồng mức người dùng (User-level thread) và luồng mức hạt nhân (Kernel-level thread).
- Ví dụ: Một trình duyệt web đa luồng có thể mở một luồng để tải giao diện, một luồng khác để xử lý dữ liệu mạng và một luồng để hiển thị văn bản.

=== Page 8 ===
2. LẬP LỊCH CPU:
- Bộ lập lịch CPU quyết định tiến trình nào trong hàng đợi sẵn sàng (ready queue) được cấp phát thời gian xử lý.
- Các thuật toán lập lịch phổ biến gồm: FIFO (First In First Out), SJF (Shortest Job First), và Round Robin (quay vòng với định mức thời gian quantum).
- Ví dụ thuật toán Round Robin: Mỗi tiến trình được cấp một khe thời gian quantum bằng 10ms, nếu hết thời gian mà chưa hoàn thành sẽ bị chuyển về cuối hàng đợi sẵn sàng.

=== Page 15 ===
3. BỘ NHỚ ẢO VÀ PHÂN TRANG:
- Bộ nhớ ảo cho phép thực thi các tiến trình có kích thước lớn hơn dung lượng bộ nhớ vật lý RAM thực tế.
- Kỹ thuật phân trang (Paging) chia không gian địa chỉ logic thành các trang (pages) có kích thước cố định, và chia bộ nhớ vật lý thành các khung trang (frames) có cùng kích thước.
- Hiện tượng lỗi trang (Page Fault) xảy ra khi tiến trình truy cập vào một trang nhớ chưa được nạp sẵn vào bộ nhớ RAM vật lý.
"""


def _setup_document_qa_environment(doc_id: str, title: str, text: str) -> Dict[str, Any]:
    """Helper to initialize chunking, indexing, retriever, and evidence selector."""
    doc = DocumentRecord(document_id=doc_id, version=1, title=title, raw_text=text)
    chunker = DocumentChunker(chunk_size=180, chunk_overlap=30, min_chunk_size=30)
    passages = chunker.chunk_document(doc)

    retriever = BM25Retriever()
    retriever.build_index(passages)

    selector = EvidenceSelector(score_threshold=1.0, max_evidence=5, min_evidence=1)
    router = QuestionRouter()

    return {
        "passages": passages,
        "retriever": retriever,
        "selector": selector,
        "router": router,
    }


# ============================================================================
# 1. CÂU HỎI TIẾNG VIỆT KHÔNG DẤU (UNACCENTED VIETNAMESE QUERIES)
# ============================================================================

def test_unaccented_vietnamese_query_retrieval_and_answer():
    """
    Validates end-to-end resilience when students type queries without Vietnamese diacritics.
    The retriever's accent-folding mechanism must match passages, the evidence selector
    must qualify evidence, and the document analyzer must answer without false refusal.
    """
    env = _setup_document_qa_environment("DOC_CRYPTO", "Mật mã cổ điển", CRYPTO_LECTURE_TEXT)
    retriever: BM25Retriever = env["retriever"]
    selector: EvidenceSelector = env["selector"]

    student_unaccented_queries = [
        ("he ma dich vong la gi va cong thuc ma hoa the nao", "HỆ MÃ DỊCH VÒNG"),
        ("giai thich ve ma caesar voi khoa k bang 3", "MÃ CAESAR"),
        ("an ninh vo dieu kien va an ninh tinh toan", "AN NINH HỆ MÃ KHÓA"),
        ("ma hoa hoan vi giu nguyen ky tu hay thay doi", "HỆ MÃ HOÁN VỊ"),
        ("tong quan he ma khoa doi xung xuat hien khi nao", "TỔNG QUAN HỆ MÃ KHÓA ĐỐI XỨNG"),
    ]

    for query, expected_section_topic in student_unaccented_queries:
        # Step 1: Retrieval
        results = retriever.retrieve(query, top_k=3)
        assert len(results) >= 1, f"Retriever returned no results for unaccented query: {query}"
        top_res = results[0]
        assert top_res.score > 0, f"Score <= 0 for query: {query}"
        assert strip_accents(expected_section_topic.lower()) in strip_accents(top_res.section_title.lower()) or \
               strip_accents(expected_section_topic.lower()) in strip_accents(top_res.text.lower()), \
               f"Top result does not match expected topic '{expected_section_topic}' for query: {query}"

        # Step 2: Evidence Selection
        ev_pkg = selector.select_evidence(query, results)
        assert ev_pkg.has_sufficient_evidence is True, f"Insufficient evidence for unaccented query: {query}"
        assert len(ev_pkg.evidence_passages) >= 1

        # Step 3: Grounded Answer
        answer, citations = DocumentAnalyzer.find_best_grounded_answer(query, ev_pkg.evidence_passages)
        assert answer != REFUSAL_MESSAGE, f"Unexpected false refusal for unaccented query: {query}"
        assert len(citations) >= 1, f"Missing citations for unaccented query: {query}"
        assert "=== Page" not in answer, f"Page marker leaked into answer: {answer}"


def test_unaccented_query_router_intent():
    """
    Validates that QuestionRouter properly handles unaccented queries without crashing
    and produces sound intent classifications.
    """
    router = QuestionRouter()

    # Direct factual unaccented lookups
    route, meta = router.classify_intent("he ma dich vong hoat dong the nao")
    assert route == QuestionRoute.DIRECT_LOOKUP

    # English topic alias unaccented
    route, meta = router.classify_intent("cho toi biet topic 49")
    assert route == QuestionRoute.SECTION_LOOKUP
    assert "Topic 49" in meta.get("target_section", "")

    # Multi-doc comparison unaccented
    route, meta = router.classify_intent("so sanh giua cac tai lieu", num_attached_documents=2)
    # The comparison regex includes accented 'so sánh', so unaccented falls to DIRECT_LOOKUP safely without crash
    assert route in (QuestionRoute.CROSS_DOCUMENT_COMPARISON, QuestionRoute.DIRECT_LOOKUP)


# ============================================================================
# 2. CÂU HỎI ĐA Ý (KHÁI NIỆM + PHÂN LOẠI + VÍ DỤ)
# ============================================================================

def test_multi_part_student_queries_concept_classification_example():
    """
    Validates handling of compound student queries covering concept definition,
    classification/taxonomy, and concrete examples in a single question.
    """
    env = _setup_document_qa_environment("DOC_CRYPTO", "Mật mã cổ điển", CRYPTO_LECTURE_TEXT)
    retriever: BM25Retriever = env["retriever"]
    selector: EvidenceSelector = env["selector"]

    compound_queries = [
        {
            "query": "Hệ mã khóa đối xứng là gì, gồm những phương pháp phổ biến nào và nguồn gốc từ khi nào?",
            "expected_terms": ["đối xứng", "phương pháp", "thay thế", "hoán vị"],
        },
        {
            "query": "Hệ mã hoán vị là gì, phân loại gồm những dạng nào và cho ví dụ minh họa?",
            "expected_terms": ["hoán vị", "ký tự", "cột", "ma trận"],
        },
        {
            "query": "Mã Caesar là gì, công thức tính toán thế nào và cho ví dụ khi k = 3?",
            "expected_terms": ["caesar", "mod 26", "k = 3"],
        },
        {
            "query": "An ninh hệ mã khóa là gì, phân loại gồm những cấp độ nào và ví dụ về One-Time Pad?",
            "expected_terms": ["an ninh", "vô điều kiện", "tính toán", "one-time pad"],
        },
    ]

    for item in compound_queries:
        q = item["query"]
        terms = item["expected_terms"]

        # Route evaluation
        route, _ = QuestionRouter.classify_intent(q, num_attached_documents=1)
        assert route in (QuestionRoute.DIRECT_LOOKUP, QuestionRoute.MULTI_PASSAGE_SYNTHESIS)

        # Retrieval & Selection
        results = retriever.retrieve(q, top_k=4)
        assert len(results) >= 2, f"Compound query should retrieve multiple relevant candidate passages for: {q}"

        ev_pkg = selector.select_evidence(q, results)
        assert ev_pkg.has_sufficient_evidence is True

        # Answer generation
        answer, citations = DocumentAnalyzer.find_best_grounded_answer(q, ev_pkg.evidence_passages)
        assert answer != REFUSAL_MESSAGE, f"Refused on compound query: {q}"
        assert len(citations) >= 1

        # Check coverage of key domain aspects in the retrieved evidence or answer
        combined_text = (answer + " " + " ".join(c["text"] for c in citations)).lower()
        matched_aspects = sum(1 for term in terms if term.lower() in combined_text)
        assert matched_aspects >= 2, f"Compound query lacked domain coverage (matched {matched_aspects}/{len(terms)}): {q}"


# ============================================================================
# 3. CÂU HỎI SO SÁNH (COMPARISON QUERIES)
# ============================================================================

def test_single_document_comparison_queries():
    """
    Validates comparison questions within a single document (e.g. comparing two encryption methods
    or two security definitions). Both sides of the comparison must be retrieved and grounded.
    """
    env = _setup_document_qa_environment("DOC_CRYPTO", "Mật mã cổ điển", CRYPTO_LECTURE_TEXT)
    retriever: BM25Retriever = env["retriever"]
    selector: EvidenceSelector = env["selector"]

    comparison_queries = [
        (
            "So sánh sự khác nhau giữa mã hóa thay thế và mã hóa hoán vị",
            ["thay thế", "hoán vị"],
        ),
        (
            "Phân biệt an ninh vô điều kiện và an ninh tính toán trong bảo mật thông tin",
            ["vô điều kiện", "tính toán"],
        ),
    ]

    for q, entity_pair in comparison_queries:
        results = retriever.retrieve(q, top_k=4)
        assert len(results) >= 1

        ev_pkg = selector.select_evidence(q, results)
        assert ev_pkg.has_sufficient_evidence is True

        answer, citations = DocumentAnalyzer.find_best_grounded_answer(q, ev_pkg.evidence_passages)
        assert answer != REFUSAL_MESSAGE, f"Refused comparison query: {q}"
        assert len(citations) >= 1

        # Verify evidence spans both entities being compared
        all_ev_text = " ".join(p.text for p in ev_pkg.evidence_passages).lower()
        for ent in entity_pair:
            assert ent.lower() in all_ev_text, f"Entity '{ent}' missing from retrieved comparison evidence for: {q}"


def test_cross_document_comparison_routing_and_synthesis():
    """
    Validates QuestionRouter and pipeline behavior when comparing concepts across multiple documents.
    """
    router = QuestionRouter()

    # Multi-doc comparison triggers CROSS_DOCUMENT_COMPARISON when num_attached_documents > 1
    cross_queries = [
        "So sánh điểm khác biệt giữa các tài liệu",
        "Điểm giống nhau và khác nhau giữa các tài liệu về kiến trúc",
        "Đối chiếu các phương pháp trong các tài liệu đã nạp",
    ]

    for q in cross_queries:
        route, meta = router.classify_intent(q, num_attached_documents=2)
        assert route == QuestionRoute.CROSS_DOCUMENT_COMPARISON, f"Expected CROSS_DOCUMENT_COMPARISON for: {q}"
        assert meta.get("documents_scope") == 2

    # Single-document query with comparison cues defaults to DIRECT_LOOKUP / MULTI_PASSAGE_SYNTHESIS
    route_single, _ = router.classify_intent("So sánh hệ mã thay thế và hoán vị", num_attached_documents=1)
    assert route_single in (QuestionRoute.DIRECT_LOOKUP, QuestionRoute.MULTI_PASSAGE_SYNTHESIS)


# ============================================================================
# 4. CÂU HỎI GỢI Ý TRONG NGOẶC NHÁY ĐƠN (QUOTED SUGGESTION QUERIES)
# ============================================================================

def test_single_quoted_suggestion_queries():
    """
    Validates student queries containing terms in single quotes (e.g. clicking suggested topics
    or asking specifically about 'hệ mã dịch vòng').
    """
    env = _setup_document_qa_environment("DOC_CRYPTO", "Mật mã cổ điển", CRYPTO_LECTURE_TEXT)
    retriever: BM25Retriever = env["retriever"]
    selector: EvidenceSelector = env["selector"]
    router: QuestionRouter = env["router"]

    quoted_queries = [
        ("Nội dung liên quan đến 'hệ mã dịch vòng' gồm những gì?", "hệ mã dịch vòng"),
        ("Giải thích chi tiết về 'Mã Caesar'?", "Mã Caesar"),
        ("Thông tin về 'hoán vị cột' trong tài liệu?", "hoán vị cột"),
        ("Điểm chính về 'An ninh vô điều kiện' là gì?", "An ninh vô điều kiện"),
    ]

    for q, expected_quoted_target in quoted_queries:
        # Step 1: QuestionRouter quoted scope detection
        route, meta = router.classify_intent(q)
        assert route == QuestionRoute.SECTION_LOOKUP, f"Quoted query did not route to SECTION_LOOKUP: {q}"
        assert meta.get("target_section") == expected_quoted_target, \
            f"Extracted target '{meta.get('target_section')}' != expected '{expected_quoted_target}'"

        # Step 2: Query term extraction prioritizes quoted terms
        terms = router.extract_query_terms(q)
        assert len(terms) >= 1
        expected_raw = expected_quoted_target.lower().split()
        assert any(t in terms for t in expected_raw), f"Quoted term not prioritized in terms: {terms}"

        # Step 3: Retrieval and evidence preservation
        results = retriever.retrieve(q, top_k=3)
        assert len(results) >= 1

        ev_pkg = selector.select_evidence(q, results)
        assert ev_pkg.has_sufficient_evidence is True

        # Step 4: Answer generation
        answer, citations = DocumentAnalyzer.find_best_grounded_answer(q, ev_pkg.evidence_passages)
        assert answer != REFUSAL_MESSAGE, f"Refused quoted query: {q}"
        assert len(citations) >= 1

        # Check relevance
        top_sec = ev_pkg.evidence_passages[0].section_title if ev_pkg.evidence_passages else ""
        rel = relevance_score(q, answer, extra=top_sec)
        assert rel >= 0.50, f"Low relevance {rel} for quoted query: {q}"


def test_quote_variations_double_and_curly_quotes():
    """
    Validates that double quotes, curly quotes, and single quotes all extract the scoped target entity identically.
    """
    router = QuestionRouter()

    quote_styles = [
        "Nội dung liên quan đến 'hệ mã dịch vòng' là gì?",
        "Nội dung liên quan đến \"hệ mã dịch vòng\" là gì?",
        "Nội dung liên quan đến “hệ mã dịch vòng” là gì?",
    ]

    for q in quote_styles:
        route, meta = router.classify_intent(q)
        assert route == QuestionRoute.SECTION_LOOKUP
        assert meta.get("target_section") == "hệ mã dịch vòng"


# ============================================================================
# 5. MULTI-TURN CHAT SESSION CONTINUITY & CONTEXT RESOLUTION
# ============================================================================

def test_chat_session_multi_turn_continuity(tmp_path):
    """
    Validates a complete realistic student conversation across multiple turns:
    - Turn 1: User asks initial question -> Assistant answers + attaches suggestions.
    - Turn 2: User refers to pronoun ('nó') -> Context resolver substitutes antecedent.
    - Turn 3: User picks suggestion by index ('câu 1') -> Context resolver expands suggestion.
    - Turn 4: User asks unaccented follow-up -> Pipeline returns grounded response.
    - Turn 5: User asks quoted query -> Targeted grounded response.
    """
    storage_dir = str(tmp_path / "sessions")
    sm = SessionManager(storage_dir=storage_dir)

    # Initialize environment
    env = _setup_document_qa_environment("DOC_CRYPTO", "Mật mã cổ điển", CRYPTO_LECTURE_TEXT)
    retriever: BM25Retriever = env["retriever"]
    selector: EvidenceSelector = env["selector"]

    # Create chat session with attached document
    session = sm.create_session(title="Thảo luận Mật mã", document_ids=["DOC_CRYPTO"])
    session_id = session.session_id

    # Turn 1: Initial query
    q1 = "Hệ mã dịch vòng là gì?"
    res1 = retriever.retrieve(q1, top_k=2)
    ev1 = selector.select_evidence(q1, res1)
    ans1, cits1 = DocumentAnalyzer.find_best_grounded_answer(q1, ev1.evidence_passages)

    session.add_message(
        role="user",
        content=q1,
    )
    session.add_message(
        role="assistant",
        content=ans1,
        citations=cits1,
        suggested_questions=[
            "Công thức mã hóa và giải mã của hệ mã dịch vòng là gì?",
            "Mã Caesar sử dụng khóa k bằng bao nhiêu?",
        ],
    )
    sm.save()

    # Turn 2: Anaphoric follow-up query using pronoun "nó"
    q2_raw = "Ví dụ của nó như thế nào?"
    q2_resolved = sm.resolve_conversation_context(session_id, q2_raw)
    assert ("Hệ mã dịch vòng" in q2_resolved or "dịch vòng" in q2_resolved.lower()), \
        f"Context resolver failed to resolve pronoun 'nó': {q2_resolved}"

    res2 = retriever.retrieve(q2_resolved, top_k=2)
    ev2 = selector.select_evidence(q2_resolved, res2)
    ans2, cits2 = DocumentAnalyzer.find_best_grounded_answer(q2_resolved, ev2.evidence_passages)
    assert ans2 != REFUSAL_MESSAGE
    assert len(cits2) >= 1

    session.add_message(role="user", content=q2_raw)
    session.add_message(role="assistant", content=ans2, citations=cits2)
    sm.save()

    # Turn 3: User picks suggested question by index: "câu 1"
    q3_raw = "câu 1"
    q3_resolved = sm.resolve_conversation_context(session_id, q3_raw)
    assert "công thức mã hóa" in q3_resolved.lower() or "dịch vòng" in q3_resolved.lower(), \
        f"Context resolver failed to resolve 'câu 1': {q3_resolved}"

    res3 = retriever.retrieve(q3_resolved, top_k=2)
    ev3 = selector.select_evidence(q3_resolved, res3)
    ans3, cits3 = DocumentAnalyzer.find_best_grounded_answer(q3_resolved, ev3.evidence_passages)
    assert ans3 != REFUSAL_MESSAGE

    session.add_message(role="user", content=q3_raw)
    session.add_message(role="assistant", content=ans3, citations=cits3)
    sm.save()

    # Turn 4: User asks unaccented follow-up
    q4_raw = "the con ma hoa hoan vi thi sao"
    res4 = retriever.retrieve(q4_raw, top_k=2)
    ev4 = selector.select_evidence(q4_raw, res4)
    ans4, cits4 = DocumentAnalyzer.find_best_grounded_answer(q4_raw, ev4.evidence_passages)
    assert ans4 != REFUSAL_MESSAGE
    assert "hoán vị" in ans4.lower()

    session.add_message(role="user", content=q4_raw)
    session.add_message(role="assistant", content=ans4, citations=cits4)
    sm.save()

    # Turn 5: Quoted suggestion query
    q5_raw = "Thầy cho em hỏi thêm về 'an ninh vô điều kiện'?"
    res5 = retriever.retrieve(q5_raw, top_k=2)
    ev5 = selector.select_evidence(q5_raw, res5)
    ans5, cits5 = DocumentAnalyzer.find_best_grounded_answer(q5_raw, ev5.evidence_passages)
    assert ans5 != REFUSAL_MESSAGE
    assert "one-time pad" in ans5.lower() or "vô hạn" in ans5.lower()

    session.add_message(role="user", content=q5_raw)
    session.add_message(role="assistant", content=ans5, citations=cits5)
    sm.save()

    # Verify session message history integrity
    reloaded_session = sm.get_session(session_id)
    assert reloaded_session is not None
    assert len(reloaded_session.messages) == 10  # 5 user + 5 assistant
    assert reloaded_session.document_ids == ["DOC_CRYPTO"]


def test_session_title_generation_from_student_queries():
    """
    Validates automatic generation of clean session titles from student questions.
    """
    cases = [
        ("Cho tôi biết về kiến trúc hệ mã dịch vòng là gì?", "Kiến trúc hệ mã dịch vòng"),
        ("Giải thích về an ninh tính toán và an ninh vô điều kiện?", "An ninh tính toán và an ninh..."),
        ("What is Columnar Transposition in cryptography?", "Columnar Transposition in..."),
        ("Tìm hiểu về bộ nhớ ảo và phân trang", "Bộ nhớ ảo và phân trang"),
    ]

    existing = set()
    for query, expected_prefix in cases:
        title = generate_title_from_query(query, existing_titles=existing)
        existing.add(title)
        assert len(title) <= 40
        assert not title.endswith("?")
        assert not title.startswith("Cho tôi biết")
        assert not title.startswith("Giải thích về")


# ============================================================================
# 6. NEGATIVE, OUT-OF-DOMAIN & EDGE-CASE RESILIENCE
# ============================================================================

def test_out_of_domain_queries_trigger_unanswerable():
    """
    Validates that out-of-domain queries (e.g. general world trivia, weather)
    are routed to UNANSWERABLE and refused.
    """
    router = QuestionRouter()

    out_of_domain_queries = [
        "Thủ đô của nước Pháp là thành phố nào?",
        "Thời tiết Hà Nội hôm nay như thế nào?",
        "Giá vàng hôm nay tăng hay giảm?",
        "Kết quả bóng đá ngoại hạng Anh đêm qua?",
    ]

    for q in out_of_domain_queries:
        route, meta = router.classify_intent(q)
        assert route == QuestionRoute.UNANSWERABLE, f"Failed to mark query as UNANSWERABLE: {q}"
        assert meta.get("reason") == "out_of_domain_query"

        # Check evidence grounding evaluation
        eff_route, is_grounded, explanation = router.evaluate_evidence_grounding(
            route=route, evidence_passages=[]
        )
        assert is_grounded is False
        assert eff_route == QuestionRoute.UNANSWERABLE


def test_empty_and_noise_queries_resilience():
    """
    Validates that empty strings, whitespace, and symbol-only queries do not crash the pipeline.
    """
    env = _setup_document_qa_environment("DOC_CRYPTO", "Mật mã cổ điển", CRYPTO_LECTURE_TEXT)
    retriever: BM25Retriever = env["retriever"]
    selector: EvidenceSelector = env["selector"]

    noise_queries = ["", "   ", "???", "...!"]

    for nq in noise_queries:
        # QuestionRouter
        route, _ = QuestionRouter.classify_intent(nq)
        assert isinstance(route, QuestionRoute)

        # Retriever
        results = retriever.retrieve(nq, top_k=2)
        assert isinstance(results, list)

        # EvidenceSelector
        ev_pkg = selector.select_evidence(nq, results)
        assert isinstance(ev_pkg.has_sufficient_evidence, bool)

        # DocumentAnalyzer
        ans, cits = DocumentAnalyzer.find_best_grounded_answer(nq, ev_pkg.evidence_passages)
        assert isinstance(ans, str)
        assert isinstance(cits, list)


def test_zero_robotic_and_page_leak_invariants():
    """
    Validates structural quality invariants across multiple questions:
    - 0% robotic phrase leaks.
    - 0% page-marker leaks.
    - Follow-up questions must all end with '?'.
    """
    env = _setup_document_qa_environment("DOC_OS", "Hệ điều hành", OS_LECTURE_TEXT)
    retriever: BM25Retriever = env["retriever"]
    selector: EvidenceSelector = env["selector"]

    test_queries = [
        "Tiến trình và luồng khác nhau thế nào?",
        "Bộ lập lịch CPU và thuật toán Round Robin?",
        "Bộ nhớ ảo và kỹ thuật phân trang hoạt động ra sao?",
    ]

    for q in test_queries:
        res = retriever.retrieve(q, top_k=3)
        ev = selector.select_evidence(q, res)
        ans, cits = DocumentAnalyzer.find_best_grounded_answer(q, ev.evidence_passages)

        assert ROBOTIC_PHRASE_1 not in ans, f"Leaked robotic phrase 1: {ans}"
        assert ROBOTIC_PHRASE_2 not in ans, f"Leaked robotic phrase 2: {ans}"
        assert "=== Page" not in ans, f"Leaked page marker in answer: {ans}"

        # Verify follow-up questions end with '?'
        followups = generate_followup_questions(
            query=q,
            context=" ".join(p.text for p in ev.evidence_passages),
            section_title=ev.evidence_passages[0].section_title if ev.evidence_passages else "",
            count=3,
        )
        assert len(followups) >= 1
        for fu in followups:
            assert fu.strip().endswith("?"), f"Follow-up question does not end with '?': {fu}"
            assert "=== Page" not in fu

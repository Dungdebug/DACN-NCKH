"""
Red-Team QA Verification Suite: Suggested Questions 100% Click-Through & Zero-Robotic Templates.
Target File: tests/test_redteam_suggested_questions_clickthrough.py

Rigorous Verification Scope:
1. Zero Robotic Phrasing:
   - 0% occurrence of 'Yêu cầu và chi tiết cụ thể liên quan đến'
   - 0% occurrence of 'Nội dung và cách triển khai'
   - 0% page-marker leak (no '=== Page N ===' inside questions or answers)
   - 100% natural, fluent Vietnamese interrogative questions ending in '?'
2. 100% Click-Through Answerability:
   - Every generated question from `generate_followup_questions` and
     `SelfStudyKnowledgeBank.get_self_study_questions` is piped into:
       a) QuestionRouter.classify_intent
       b) BM25Retriever + EvidenceSelector.select_evidence
       c) DocumentAnalyzer.find_best_grounded_answer
   - Invariants:
       * has_sufficient_evidence == True (100% sufficiency, 0% refusal)
       * relevance_score >= 0.50 (high lexical & semantic alignment)
       * len(citations) >= 1 (grounded with explicit citation references)
       * refused == False ('Không tìm thấy đủ thông tin' is NEVER returned)
3. Multi-Corpus Robustness:
   - Document A: Slide Mật mã cổ điển (12 numbered sections, multi-page)
   - Document B: Danh sách Topic 1..50 (IELTS/Speaking syllabus)
   - Document C: Bài báo khoa học (Scientific research paper: Abstract, Intro, Method, Experiments, Conclusion)
"""

import pytest
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.document_store import DocumentRecord
from src.hybrid_qa.retriever import BM25Retriever
from src.hybrid_qa.evidence import EvidenceSelector
from src.web.question_router import QuestionRouter, QuestionRoute
from src.web.document_analyzer import DocumentAnalyzer
from src.self_study.knowledge_bank import SelfStudyKnowledgeBank
from src.hybrid_qa.answer_span import generate_followup_questions, relevance_score

ROBOTIC_PHRASE_1 = "Yêu cầu và chi tiết cụ thể liên quan đến"
ROBOTIC_PHRASE_2 = "Nội dung và cách triển khai"
REFUSAL_MESSAGE = "Không tìm thấy đủ thông tin bằng chứng trong tài liệu để trả lời câu hỏi này."

# ============================================================================
# CORPUS FIXTURES
# ============================================================================

SLIDE_12_TEXT = """=== Page 1 ===
1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:
- Xuất hiện đầu tiên trong ngành mã hóa.
- Các hệ mã này phần có từ trước năm 1970.
- Hệ mã khóa bí mật hoặc hệ mã khóa cổ điển.
- Các phương pháp này là cơ sở để phát triển các thuật toán mã hóa đối xứng hiện nay.
- Có 2 phương pháp phổ biến:
* Mã hóa thay thế.
* Mã hóa hoán vị.

=== Page 5 ===
2. HỆ MÃ DỊCH VÒNG:
- Không gian bản rõ P, bản mã C, khóa K là Z26.
- Hàm mã hóa: e_k(x) = (x + k) mod 26.
- Hàm giải mã: d_k(y) = (y - k) mod 26.
- Ví dụ kinh điển là mật mã Caesar với khóa k = 3.

=== Page 10 ===
3. PHÁ MÃ:
An ninh hệ mã khóa:
- An ninh vô điều kiện: Không thể phá mã dù có thời gian và tài nguyên vô hạn.
- An ninh tính toán: Chi phí để phá vỡ vượt quá giá trị của thông tin mã hóa, hoặc thời gian phá mã vượt quá thời gian hữu dụng của thông tin.
- Các phương pháp thám mã phổ biến gồm tấn công vét cạn (brute-force) và phân tích thống kê tần suất xuất hiện ký tự.

=== Page 14 ===
4. HỆ MÃ ĐƠN BẢNG:
- Mỗi ký tự trong bảng chữ cái được thay thế bằng một ký tự cố định khác.
- Không gian khóa K là tập hợp tất cả các hoán vị của 26 chữ cái (26! phần tử).
- Điểm yếu lớn nhất là dễ bị thám mã bằng phân tích tần suất xuất hiện của chữ cái.

=== Page 18 ===
5. HỆ MÃ AFFINE:
- Hàm mã hóa: e(x) = (a * x + b) mod 26 với điều kiện gcd(a, 26) = 1.
- Hàm giải mã: d(y) = a^(-1) * (y - b) mod 26.
- Khóa k = (a, b) thuộc không gian Z26 x Z26.

=== Page 22 ===
6. HỆ MÃ VIGENERE:
- Hệ mã thay thế đa bảng chữ cái.
- Sử dụng từ khóa lặp lại có độ dài m: K = (k1, k2, ..., km).
- Phép cộng modulo 26 cho từng ký tự tương ứng của bản rõ với từ khóa.

=== Page 26 ===
7. HỆ MÃ HILL:
- Mã hóa từng khối m ký tự bằng phép nhân ma trận khóa K kích thước m x m.
- Điều kiện giải mã: Ma trận khóa K phải khả nghịch trong modulo 26 (det(K) nguyên tố cùng nhau với 26).

=== Page 30 ===
8. HỆ MÃ HOÁ ĐỘN MỘT LẦN:
- Hệ mã hóa hoàn hảo (Vernam Cipher / One-Time Pad).
- Độ dài khóa bằng đúng độ dài thông điệp cần mã hóa và chỉ dùng duy nhất một lần.
- Đạt mức độ an ninh vô điều kiện khi chuỗi khóa được sinh hoàn toàn ngẫu nhiên.

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

=== Page 34 ===
11. HỆ MÃ HOÁ PLAYFAIR:
- Sử dụng ma trận ký tự 5x5 để mã hóa từng cặp ký tự (digram).
- Thay thế các cặp ký tự dựa trên quy tắc cùng hàng, cùng cột hoặc tạo thành hình chữ nhật trên ma trận.

=== Page 36 ===
12. HỆ MÃ HOÁ HOÁN VỊ:
- Giữ nguyên các ký tự của bản rõ nhưng thay đổi vị trí thứ tự theo một quy tắc hoán vị nhất định.
- Mã hóa hoán vị cột (Columnar Transposition) sắp xếp bản rõ theo các hàng và đọc ra theo các cột đã hoán vị.
"""

TOPIC_LIST_TEXT = """Topic 1: Describe a time when you were ill
I would like to talk about a time when I was ill. Last year, I caught a serious flu during winter and had to rest at home for a whole week.

Topic 2: Describe a famous person that you would like to meet
His name is Ludwig van Beethoven. He was a famous German composer and pianist who created inspiring classical music.

Topic 47: Describe an environmental problem in your city
Air pollution is the biggest challenge in my hometown due to high vehicle density and factory emissions.

Topic 48: Describe a bad restaurant experience
Although the problem was eventually resolved, I did not have a good dining experience due to extremely slow customer service.

Topic 49: Describe an outdoor sport you would like to try
I would like to talk about an outdoor sport I want to try. It is hiking. I first knew about this sport through videos on social media. In the future, I hope I can go hiking in the mountains with my close friends.

Topic 50: Describe an interesting video that you watched on social media
The video was about wildlife conservation in Africa, featuring majestic lions and elephants roaming safely in their natural habitat.
"""

SCIENTIFIC_PAPER_TEXT = """Tóm tắt:
Bài báo nghiên cứu mô hình học sâu kết hợp cơ chế chú ý tăng cường SA-CMS cho bài toán hỏi đáp trên tài liệu kỹ thuật.
Kết quả thực nghiệm trên tập dữ liệu chuẩn cho thấy độ chính xác đạt 92.5%, vượt trội so với mô hình cơ sở.

1. Giới thiệu:
Hệ thống hỏi đáp lai là mô hình kết hợp giữa truy xuất từ khóa BM25 và biểu diễn ngữ nghĩa của mạng nơ-ron sâu.
Kế hoạch nghiên cứu năm 2026 đặt mục tiêu tối ưu hóa thời gian xử lý và nâng cao độ tin cậy của câu trả lời.

2. Phương pháp đề xuất:
Kiến trúc SA-CMS bao gồm hai thành phần chính: bộ lọc bằng chứng động và mô-đun tổng hợp câu trả lời dựa trên suy luận logic.
Công thức tính điểm trọng số kết hợp được xác định theo: S_final = alpha * S_bm25 + (1 - alpha) * S_dense.

3. Kết quả thực nghiệm và Thảo luận:
Chúng tôi đánh giá hiệu năng hệ thống trên 1000 truy vấn thực tế.
Tỷ lệ trích xuất bằng chứng đầy đủ đạt 98.2%, và thời gian đáp ứng trung bình là 45ms.
Giải pháp khắc phục hạn chế về bộ nhớ GPU là áp dụng kỹ thuật lượng tử hóa trọng số.

4. Kết luận:
Nghiên cứu đã khẳng định tính hiệu quả vượt trội của giải pháp đề xuất.
Mục tiêu tiếp theo là triển khai hệ thống trên các thiết bị biên.
"""


def _build_qa_environment(doc_id: str, title: str, text: str):
    """Utility to build document chunk index, retriever, selector, router, and knowledge bank."""
    doc = DocumentRecord(document_id=doc_id, version=1, title=title, raw_text=text)
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=30, min_chunk_size=30)
    passages = chunker.chunk_document(doc)

    retriever = BM25Retriever()
    retriever.build_index(passages)

    selector = EvidenceSelector(score_threshold=1.0, max_evidence=5, min_evidence=1)
    router = QuestionRouter()

    bank = SelfStudyKnowledgeBank()
    bank.learn_document(doc_id, title, text)

    return {
        "passages": passages,
        "retriever": retriever,
        "selector": selector,
        "router": router,
        "bank": bank,
    }


# ============================================================================
# TESTS
# ============================================================================

def test_suggested_questions_no_robotic_templates_corpus_wide():
    """
    Verification 1: Asserts that 100% of suggested questions across all 3 document
    corpora contain ZERO robotic templates, zero page leaks, and are fluent questions ending with '?'.
    """
    corpora = [
        ("DOC_SLIDE_12", "Slide Mật mã", SLIDE_12_TEXT),
        ("DOC_TOPICS", "Topic Speaking List", TOPIC_LIST_TEXT),
        ("DOC_PAPER", "Scientific Paper", SCIENTIFIC_PAPER_TEXT),
    ]

    all_generated_questions = []

    for doc_id, title, text in corpora:
        env = _build_qa_environment(doc_id, title, text)
        bank_qs = env["bank"].get_self_study_questions(count=10)
        all_generated_questions.extend(bank_qs)

        for p in env["passages"]:
            followups = generate_followup_questions(
                query="nội dung chính",
                context=p.text,
                section_title=p.section_title,
                count=3,
            )
            all_generated_questions.extend(followups)

    assert len(all_generated_questions) >= 60, f"Expected at least 60 questions, got {len(all_generated_questions)}"

    # Strict Red-Team checks across all questions
    for q in all_generated_questions:
        assert isinstance(q, str) and len(q.strip()) > 10, f"Question too short or non-string: {q}"
        # (1) Forbidden robotic template checks
        assert ROBOTIC_PHRASE_1 not in q, f"Question contains forbidden phrase '{ROBOTIC_PHRASE_1}': {q}"
        assert ROBOTIC_PHRASE_2 not in q, f"Question contains forbidden phrase '{ROBOTIC_PHRASE_2}': {q}"
        # (2) Page marker leak checks
        assert "=== Page" not in q, f"Question leaked page marker: {q}"
        # (3) Fluent question syntax
        assert q.strip().endswith("?"), f"Question does not end with '?': {q}"


def test_clickthrough_cryptography_slide_12_sections():
    """
    Verification 2: Tests 100% click-through for self-study and follow-up questions
    on the 12-section Cryptography slide.
    Verifies has_sufficient_evidence == True, relevance_score >= 0.5, and refused == False.
    """
    env = _build_qa_environment("DOC_SLIDE_12", "Slide Mật mã 12 Mục", SLIDE_12_TEXT)
    retriever = env["retriever"]
    selector = env["selector"]
    router = env["router"]
    bank = env["bank"]

    # 1. Self-study questions across all sections
    bank_questions = bank.get_self_study_questions(count=12)
    assert len(bank_questions) == 12

    for q in bank_questions:
        # Route
        route, meta = router.classify_intent(q)
        assert route in {
            QuestionRoute.SECTION_LOOKUP,
            QuestionRoute.DIRECT_LOOKUP,
            QuestionRoute.MULTI_PASSAGE_SYNTHESIS,
            QuestionRoute.DOCUMENT_SUMMARY,
        }

        # Retrieve & Evidence Selection
        retrieval_res = retriever.retrieve(q, top_k=5)
        ev_pkg = selector.select_evidence(q, retrieval_res)
        assert ev_pkg.has_sufficient_evidence is True, f"Failed evidence sufficiency for bank question: {q}"
        assert len(ev_pkg.evidence_passages) >= 1

        # Grounded Answer Generation
        answer, citations = DocumentAnalyzer.find_best_grounded_answer(q, ev_pkg.evidence_passages)
        assert answer != REFUSAL_MESSAGE, f"Unexpected refusal for bank question: {q}"
        assert len(citations) >= 1, f"Missing citations for: {q}"
        assert "=== Page" not in answer, f"Leaked page marker in answer for: {q}"

        # Relevance Verification
        top_sec = ev_pkg.evidence_passages[0].section_title or ""
        rel = relevance_score(q, answer, extra=top_sec)
        assert rel >= 0.50, f"Relevance score {rel} < 0.50 for question: {q}"

    # 2. Follow-up questions generated per passage
    tested_followups = 0
    for p in env["passages"][:6]:  # Test first 6 distinct passages
        followups = generate_followup_questions("mã hóa đối xứng", p.text, p.section_title, count=3)
        for q in followups:
            tested_followups += 1
            retrieval_res = retriever.retrieve(q, top_k=5)
            ev_pkg = selector.select_evidence(q, retrieval_res)
            assert ev_pkg.has_sufficient_evidence is True, f"Followup insufficient: {q}"

            answer, citations = DocumentAnalyzer.find_best_grounded_answer(q, ev_pkg.evidence_passages)
            assert answer != REFUSAL_MESSAGE, f"Refused followup: {q}"
            top_sec = ev_pkg.evidence_passages[0].section_title or ""
            rel = relevance_score(q, answer, extra=top_sec)
            assert rel >= 0.50, f"Low relevance {rel} for followup: {q}"

    assert tested_followups >= 15


def test_clickthrough_topic_list_1_to_50():
    """
    Verification 3: Tests 100% click-through for Topic 1..50 list.
    Verifies routing to SECTION_LOOKUP with Topic N target, 100% sufficiency, and non-refusal.
    """
    env = _build_qa_environment("DOC_TOPICS", "Topic List", TOPIC_LIST_TEXT)
    retriever = env["retriever"]
    selector = env["selector"]
    router = env["router"]
    bank = env["bank"]

    questions = bank.get_self_study_questions(count=6)
    assert len(questions) == 6

    for q in questions:
        route, meta = router.classify_intent(q)
        assert route == QuestionRoute.SECTION_LOOKUP
        assert "Topic" in meta.get("target_section", "")

        retrieval_res = retriever.retrieve(q, top_k=5)
        ev_pkg = selector.select_evidence(q, retrieval_res)
        assert ev_pkg.has_sufficient_evidence is True, f"Insufficient evidence for: {q}"

        answer, citations = DocumentAnalyzer.find_best_grounded_answer(q, ev_pkg.evidence_passages)
        assert answer != REFUSAL_MESSAGE, f"Refusal on Topic question: {q}"
        assert len(citations) >= 1

        top_sec = ev_pkg.evidence_passages[0].section_title or ""
        rel = relevance_score(q, answer, extra=top_sec)
        assert rel >= 0.50, f"Relevance {rel} < 0.50 for: {q}"


def test_clickthrough_scientific_research_paper():
    """
    Verification 4: Tests 100% click-through on Scientific Paper.
    Verifies questions targeting formulas, percentages, and section overviews get answered with citations.
    """
    env = _build_qa_environment("DOC_PAPER", "Scientific Paper SA-CMS", SCIENTIFIC_PAPER_TEXT)
    retriever = env["retriever"]
    selector = env["selector"]
    router = env["router"]
    bank = env["bank"]

    # Bank self-study questions
    bank_questions = bank.get_self_study_questions(count=5)
    for q in bank_questions:
        retrieval_res = retriever.retrieve(q, top_k=5)
        ev_pkg = selector.select_evidence(q, retrieval_res)
        assert ev_pkg.has_sufficient_evidence is True, f"Paper question insufficient: {q}"

        answer, citations = DocumentAnalyzer.find_best_grounded_answer(q, ev_pkg.evidence_passages)
        assert answer != REFUSAL_MESSAGE, f"Paper question refused: {q}"
        assert len(citations) >= 1

        top_sec = ev_pkg.evidence_passages[0].section_title or ""
        rel = relevance_score(q, answer, extra=top_sec)
        assert rel >= 0.50, f"Paper question low rel {rel}: {q}"

    # Target specific metrics/percentage questions
    metric_q = "Chỉ số 92.5% phản ánh kết quả hoặc tiêu chí nào?"
    res_m = retriever.retrieve(metric_q, top_k=3)
    ev_m = selector.select_evidence(metric_q, res_m)
    assert ev_m.has_sufficient_evidence is True
    ans_m, cits_m = DocumentAnalyzer.find_best_grounded_answer(metric_q, ev_m.evidence_passages)
    assert ans_m != REFUSAL_MESSAGE
    assert "92.5%" in ans_m


def test_clickthrough_query_pivot_and_avoid_duplication():
    """
    Verification 5: Tests that when querying a specific section (e.g. 'PHÁ MÃ'),
    suggested questions pivot away from the active query and remain 100% answerable.
    """
    env = _build_qa_environment("DOC_SLIDE_12", "Slide Mật mã", SLIDE_12_TEXT)
    bank = env["bank"]
    retriever = env["retriever"]
    selector = env["selector"]

    # Querying 'giới thiệu hệ mã khóa đối xứng': suggested questions should explore other sections
    active_query = "giới thiệu hệ mã khóa đối xứng"
    pivoted_qs = bank.get_self_study_questions(current_query=active_query, count=4)

    assert len(pivoted_qs) == 4
    for q in pivoted_qs:
        # None of the suggested questions should duplicate the active 'GIỚI THIỆU' section
        assert "GIỚI THIỆU" not in q
        assert ROBOTIC_PHRASE_1 not in q
        assert ROBOTIC_PHRASE_2 not in q

        # Verify 100% click-through answerability of pivoted questions
        res = retriever.retrieve(q, top_k=5)
        ev_pkg = selector.select_evidence(q, res)
        assert ev_pkg.has_sufficient_evidence is True, f"Pivoted question insufficient: {q}"

        ans, cits = DocumentAnalyzer.find_best_grounded_answer(q, ev_pkg.evidence_passages)
        assert ans != REFUSAL_MESSAGE, f"Pivoted question refused: {q}"
        top_sec = ev_pkg.evidence_passages[0].section_title if ev_pkg.evidence_passages else ""
        rel = relevance_score(q, ans, extra=top_sec)
        assert rel >= 0.50, f"Pivoted question low relevance {rel}: {q}"


def test_clickthrough_batch_high_throughput_metrics():
    """
    Verification 6: Quantitative stress check on click-through performance:
    - 0% refusal rate across >= 30 questions
    - 100% sufficiency rate
    - Average relevance score >= 0.85
    """
    env = _build_qa_environment("DOC_SLIDE_12", "Slide Mật mã", SLIDE_12_TEXT)
    retriever = env["retriever"]
    selector = env["selector"]
    bank = env["bank"]

    # Gather batch of diverse questions
    test_suite_qs = bank.get_self_study_questions(count=12)
    for p in env["passages"]:
        followups = generate_followup_questions("mật mã học cổ điển", p.text, p.section_title, count=2)
        test_suite_qs.extend(followups)

    assert len(test_suite_qs) >= 30

    refusals = 0
    sufficiencies = 0
    scores = []

    for q in test_suite_qs:
        res = retriever.retrieve(q, top_k=5)
        ev = selector.select_evidence(q, res)
        if ev.has_sufficient_evidence:
            sufficiencies += 1

        ans, _ = DocumentAnalyzer.find_best_grounded_answer(q, ev.evidence_passages)
        if ans == REFUSAL_MESSAGE:
            refusals += 1

        top_sec = ev.evidence_passages[0].section_title if ev.evidence_passages else ""
        scores.append(relevance_score(q, ans, extra=top_sec))

    total = len(test_suite_qs)
    refusal_rate = refusals / total
    sufficiency_rate = sufficiencies / total
    avg_relevance = sum(scores) / len(scores)

    assert refusal_rate == 0.0, f"Expected 0% refusal rate, got {refusal_rate:.2%}"
    assert sufficiency_rate == 1.0, f"Expected 100% sufficiency rate, got {sufficiency_rate:.2%}"
    assert avg_relevance >= 0.85, f"Expected avg relevance >= 0.85, got {avg_relevance:.2f}"

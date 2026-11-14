"""
Red-Team QA & Performance Verification Test Suite:
`tests/test_redteam_multi_turn_context_and_scope.py`

Focus:
1. Multi-turn conversational context carry-over and reference resolution on SessionManager.
2. QuestionRouter intent classification across turns (concept, classification, section lookup, entity dive).
3. HybridRetriever (BM25Retriever) document scope locking and strict isolation (zero cross-doc leakage).
4. DocumentAnalyzer grounded answer synthesis across 4 mandatory conversational turns:
   - Turn 1: Concept ('hệ mã khóa đối xứng là gì')
   - Turn 2: Classification ('có bao nhiêu phương pháp và hệ mã hóa')
   - Turn 3: Specific Section Deep-Dive ('trình bày mục 3. PHÁ MÃ')
   - Turn 4: Specific Cipher System ('Hệ mã hóa Playfair')
5. Verification that all 4 turns return full readable text (no truncated answers, no Page boundary leakage).
6. 100% click-through success on all suggested follow-up questions (refused == False).
7. Strict document scope locking: when locked to Document A, queries for Document B are refused cleanly.
"""

import time
import pytest
from starlette.testclient import TestClient

from src.web.app import app
from src.web.session_manager import SessionManager, extract_topic_anchor
from src.web.question_router import QuestionRouter, QuestionRoute
from src.hybrid_qa.retriever import BM25Retriever as HybridRetriever
from src.hybrid_qa.document_store import Passage
from src.web.document_analyzer import DocumentAnalyzer
from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.document_store import DocumentRecord


SLIDE_CRYPTOGRAPHY_TEXT = """=== Page 1 ===
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

UNRELATED_DOCUMENT_TEXT = """=== Page 1 ===
Topic 1: Describe a memorable journey you made by public transport.
You should say where you went, what kind of transport you used, and explain why it was memorable.

=== Page 5 ===
Topic 49: Describe an outdoor sport you played for the first time.
I played basketball at a local court with my classmates. It was exhausting but exhilarating and helped build teamwork skills.
"""


# ==============================================================================
# 1. UNIT TESTS: SessionManager & Conversational Context
# ==============================================================================

class TestSessionManagerAndContextCarryOver:
    """Tests session creation, document attachment locking, anchor carryover, and non-hijacking."""

    def test_session_creation_and_document_locking(self, tmp_path):
        mgr = SessionManager(storage_dir=str(tmp_path))
        session = mgr.create_session(
            title="Phiên nghiên cứu mật mã",
            document_ids=["DOC_SLIDE_CRYPTO"],
        )
        assert session.session_id.startswith("chat_")
        assert session.document_ids == ["DOC_SLIDE_CRYPTO"]

        # Scope locking: attaching another document
        mgr.attach_document(session.session_id, "DOC_ANOTHER")
        updated = mgr.get_session(session.session_id)
        assert "DOC_ANOTHER" in updated.document_ids

        # Detaching document
        mgr.detach_document(session.session_id, "DOC_ANOTHER")
        updated = mgr.get_session(session.session_id)
        assert "DOC_ANOTHER" not in updated.document_ids
        assert updated.document_ids == ["DOC_SLIDE_CRYPTO"]

    def test_topic_anchor_extraction_and_carryover(self, tmp_path):
        mgr = SessionManager(storage_dir=str(tmp_path))
        session = mgr.create_session(title="Test Anchor Carryover")

        # Turn 1
        session.add_message(role="user", content="cho tôi biết mục 3")
        session.add_message(
            role="assistant",
            content="Mục 3 là Phá mã. An ninh gồm vô điều kiện và tính toán.",
            suggested_questions=[
                "Nguyên lý an ninh vô điều kiện là gì?",
                "Tấn công vét cạn hoạt động như thế nào?",
            ],
        )

        # Turn 2: User asks anaphoric follow-up: "giải thích thêm chi tiết"
        resolved = mgr.resolve_conversation_context(session.session_id, "giải thích thêm chi tiết")
        assert "mục 3" in resolved.lower()

        # Turn 3: "mục 3" is NOT hijacked into suggested question index #3
        resolved_muc = mgr.resolve_conversation_context(session.session_id, "trình bày mục 3. PHÁ MÃ")
        assert resolved_muc == "trình bày mục 3. PHÁ MÃ"

    def test_numbered_followup_resolution(self, tmp_path):
        mgr = SessionManager(storage_dir=str(tmp_path))
        session = mgr.create_session(title="Test Numbered Resolution")

        session.add_message(role="user", content="Khái niệm mã hóa đối xứng là gì?")
        session.add_message(
            role="assistant",
            content="Khái niệm cốt lõi...",
            suggested_questions=[
                "1. **[Phân loại]**: Có bao nhiêu phương pháp mã hóa?",
                "2. **[Hệ mã]**: Hệ mã hóa Playfair hoạt động ra sao?",
            ],
        )

        # "câu 2" resolves to second suggested question
        resolved_q2 = mgr.resolve_conversation_context(session.session_id, "câu 2")
        assert "Playfair" in resolved_q2

        # "trả lời câu 1" resolves to first suggested question
        resolved_q1 = mgr.resolve_conversation_context(session.session_id, "trả lời câu 1")
        assert "phương pháp" in resolved_q1


# ==============================================================================
# 2. UNIT TESTS: QuestionRouter Across the 4 Turns
# ==============================================================================

class TestQuestionRouterAcrossTurns:
    """Validates that QuestionRouter accurately classifies each turn."""

    def test_turn1_concept_intent(self):
        router = QuestionRouter()
        route, _ = router.classify_intent("hệ mã khóa đối xứng là gì")
        assert route == QuestionRoute.DIRECT_LOOKUP

    def test_turn2_classification_intent(self):
        router = QuestionRouter()
        route, _ = router.classify_intent("có bao nhiêu phương pháp và hệ mã hóa")
        # Direct lookup or synthesis, valid for grounded retrieval
        assert route in (QuestionRoute.DIRECT_LOOKUP, QuestionRoute.MULTI_PASSAGE_SYNTHESIS)

    def test_turn3_section_lookup_intent(self):
        router = QuestionRouter()
        # "trình bày mục 3. PHÁ MÃ" contains "mục 3"
        route, meta = router.classify_intent("trình bày mục 3. PHÁ MÃ")
        assert route == QuestionRoute.SECTION_LOOKUP
        assert "3" in meta.get("target_section", "")

        # Standalone "3. PHÁ MÃ"
        route2, meta2 = router.classify_intent("3. PHÁ MÃ")
        assert route2 == QuestionRoute.SECTION_LOOKUP
        assert "PHÁ MÃ" in meta2.get("target_section", "")

    def test_turn4_specific_system_intent(self):
        router = QuestionRouter()
        route, _ = router.classify_intent("Hệ mã hóa Playfair")
        assert route == QuestionRoute.DIRECT_LOOKUP


# ==============================================================================
# 3. UNIT TESTS: HybridRetriever (BM25Retriever) Scope Locking & Isolation
# ==============================================================================

class TestRetrieverDocumentScopeLocking:
    """Verifies that HybridRetriever strictly respects allowed_document_ids with zero leakage."""

    @pytest.fixture
    def indexed_retriever(self):
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=0)
        doc_crypto = DocumentRecord(
            document_id="DOC_CRYPTO",
            version=1,
            title="Slide Cryptography",
            raw_text=SLIDE_CRYPTOGRAPHY_TEXT,
        )
        doc_other = DocumentRecord(
            document_id="DOC_OTHER",
            version=1,
            title="Unrelated Document",
            raw_text=UNRELATED_DOCUMENT_TEXT,
        )

        passages_crypto = chunker.chunk_document(doc_crypto)
        passages_other = chunker.chunk_document(doc_other)
        all_passages = passages_crypto + passages_other

        retriever = HybridRetriever()
        retriever.build_index(all_passages)
        return retriever

    def test_retriever_locked_to_crypto_doc_never_returns_other(self, indexed_retriever):
        # Query for general topic in crypto doc
        results = indexed_retriever.retrieve(
            query="hệ mã khóa đối xứng là gì",
            top_k=5,
            allowed_document_ids=["DOC_CRYPTO"],
        )
        assert len(results) > 0
        for r in results:
            assert r.document_id == "DOC_CRYPTO"
            assert r.document_id != "DOC_OTHER"

    def test_retriever_locked_to_crypto_doc_returns_empty_for_other_doc_content(self, indexed_retriever):
        # Querying for Topic 49 (which exists only in DOC_OTHER)
        results = indexed_retriever.retrieve(
            query="cho tôi biết topic 49",
            top_k=5,
            allowed_document_ids=["DOC_CRYPTO"],
        )
        # Should NOT return DOC_OTHER passages because scope is locked to DOC_CRYPTO
        for r in results:
            assert r.document_id == "DOC_CRYPTO"
            assert "basketball" not in r.text.lower()

    def test_retriever_finds_all_4_turns_in_crypto_doc(self, indexed_retriever):
        turns = [
            ("hệ mã khóa đối xứng là gì", ["ngành mã hóa", "đối xứng", "1970"]),
            ("có bao nhiêu phương pháp và hệ mã hóa", ["thay thế", "hoán vị"]),
            ("trình bày mục 3. PHÁ MÃ", ["phá mã", "an ninh"]),
            ("Hệ mã hóa Playfair", ["playfair", "5x5"]),
        ]
        for query, expected_keywords in turns:
            hits = indexed_retriever.retrieve(query, top_k=3, allowed_document_ids=["DOC_CRYPTO"])
            assert len(hits) > 0, f"Retriever returned no hits for: {query}"
            combined_text = " ".join(h.text.lower() for h in hits)
            matched = any(k in combined_text for k in expected_keywords)
            assert matched, f"None of {expected_keywords} found in hits for query: {query}"


# ==============================================================================
# 4. UNIT TESTS: DocumentAnalyzer Grounded Answering Across the 4 Turns
# ==============================================================================

class TestDocumentAnalyzerGroundedSynthesis:
    """Verifies that DocumentAnalyzer synthesizes full, clean, grounded answers for all 4 turns."""

    @pytest.fixture
    def crypto_passages(self):
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=0)
        doc = DocumentRecord(
            document_id="DOC_CRYPTO_SLIDE",
            version=1,
            title="Slide 2: Các Hệ Mã Hóa Đối Xứng Cổ Điển",
            raw_text=SLIDE_CRYPTOGRAPHY_TEXT,
        )
        return chunker.chunk_document(doc)

    def test_turn1_concept_synthesis(self, crypto_passages):
        answer, citations = DocumentAnalyzer.find_best_grounded_answer(
            query="hệ mã khóa đối xứng là gì",
            evidence_passages=crypto_passages,
            answer_mode="detailed",
        )
        assert len(citations) > 0
        ans_lower = answer.lower()
        # Concept core present
        assert "mã hóa" in ans_lower
        assert "=== page" not in ans_lower
        assert "💡 **câu hỏi tự học tiếp theo:**" in ans_lower

    def test_turn2_classification_synthesis(self, crypto_passages):
        answer, citations = DocumentAnalyzer.find_best_grounded_answer(
            query="có bao nhiêu phương pháp và hệ mã hóa",
            evidence_passages=crypto_passages,
            answer_mode="detailed",
        )
        assert len(citations) > 0
        ans_lower = answer.lower()
        # Classification methods
        assert "thay thế" in ans_lower
        assert "hoán vị" in ans_lower
        assert "=== page" not in ans_lower

    def test_turn3_section3_deep_dive_synthesis(self, crypto_passages):
        answer, citations = DocumentAnalyzer.find_best_grounded_answer(
            query="trình bày mục 3. PHÁ MÃ",
            evidence_passages=crypto_passages,
            answer_mode="detailed",
        )
        assert len(citations) > 0
        ans_lower = answer.lower()
        # Facts of section 3
        assert "an ninh" in ans_lower or "phá mã" in ans_lower
        assert "vô điều kiện" in ans_lower or "tính toán" in ans_lower or "vét cạn" in ans_lower
        assert "=== page" not in ans_lower

    def test_turn4_playfair_synthesis(self, crypto_passages):
        answer, citations = DocumentAnalyzer.find_best_grounded_answer(
            query="Hệ mã hóa Playfair",
            evidence_passages=crypto_passages,
            answer_mode="detailed",
        )
        assert len(citations) > 0
        ans_lower = answer.lower()
        # Facts of Playfair
        assert "playfair" in ans_lower or "5x5" in ans_lower or "digram" in ans_lower or "cặp ký tự" in ans_lower
        assert "=== page" not in ans_lower


# ==============================================================================
# 5. INTEGRATION & END-TO-END: 4-Turn Multi-Turn Session & Scope Locking
# ==============================================================================

class TestEndToEndMultiTurnSessionAndScopeLocking:
    """
    End-to-end integration test through FastAPI TestClient.
    Verifies:
    1. Ingestion of Slide document.
    2. Turn 1: 'hệ mã khóa đối xứng là gì' -> full readable concept, not refused.
    3. Turn 2: 'có bao nhiêu phương pháp và hệ mã hóa' -> lists 2 methods (thay thế, hoán vị).
    4. Turn 3: 'trình bày mục 3. PHÁ MÃ' -> section facts (an ninh vô điều kiện, tính toán).
    5. Turn 4: 'Hệ mã hóa Playfair' -> cipher facts (ma trận 5x5, digram).
    6. Suggested follow-up questions click-through across all turns -> 100% answerable (refused == False).
    7. Strict scope locking: query for unrelated doc is cleanly refused with zero leakage.
    8. Zero leakage of '=== Page N ===' artifacts across all responses.
    """

    @pytest.fixture(scope="class", autouse=True)
    def setup_documents(self):
        client = TestClient(app)

        docs_to_ingest = [
            {
                "document_id": "DOC_REDTEAM_SLIDE_CRYPTO",
                "title": "Slide 2: Các Hệ Mã Hóa Đối Xứng Cổ Điển",
                "text": SLIDE_CRYPTOGRAPHY_TEXT,
                "category": "Cryptography",
            },
            {
                "document_id": "DOC_REDTEAM_UNRELATED",
                "title": "IELTS Speaking Topics",
                "text": UNRELATED_DOCUMENT_TEXT,
                "category": "English",
            },
        ]

        for doc_payload in docs_to_ingest:
            for attempt in range(5):
                try:
                    time.sleep(0.15)
                    r = client.post("/api/documents/raw", json=doc_payload)
                    if r.status_code == 200:
                        break
                except Exception:
                    time.sleep(0.2)

    @property
    def client(self):
        return TestClient(app)

    def test_full_4_turn_conversation_and_suggested_clickthrough(self):
        # Create session locked specifically to DOC_REDTEAM_SLIDE_CRYPTO
        sess_res = self.client.post(
            "/api/chat/sessions",
            json={
                "title": "Hội thoại Mật mã học cổ điển",
                "document_ids": ["DOC_REDTEAM_SLIDE_CRYPTO"],
            },
        )
        assert sess_res.status_code == 200
        session_id = sess_res.json()["session_id"]

        collected_suggested_questions = []

        # ----------------------------------------------------------------------
        # LƯỢT 1: Hỏi khái niệm chung ('hệ mã khóa đối xứng là gì')
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        turn1_res = self.client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={
                "query": "hệ mã khóa đối xứng là gì",
                "answer_mode": "detailed",
            },
        )
        turn1_latency = time.perf_counter() - t0
        assert turn1_res.status_code == 200
        t1_data = turn1_res.json()

        assert t1_data["refused"] is False, "Lượt 1 bị từ chối sai lệch!"
        assert len(t1_data["citations"]) > 0, "Lượt 1 không có trích dẫn nguồn!"
        ans1 = t1_data["answer"]
        ans1_lower = ans1.lower()

        # Text completeness: concept explanation
        assert any(term in ans1_lower for term in ["mã hóa", "đối xứng", "khóa bí mật", "cổ điển"])
        # No page bleed
        assert "=== page" not in ans1_lower
        # Collect suggested questions
        sq1 = t1_data.get("suggested_questions", [])
        assert len(sq1) >= 1
        collected_suggested_questions.extend(sq1)

        # ----------------------------------------------------------------------
        # LƯỢT 2: Hỏi phân loại ('có bao nhiêu phương pháp và hệ mã hóa')
        # ----------------------------------------------------------------------
        turn2_res = self.client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={
                "query": "có bao nhiêu phương pháp và hệ mã hóa",
                "answer_mode": "detailed",
            },
        )
        assert turn2_res.status_code == 200
        t2_data = turn2_res.json()

        assert t2_data["refused"] is False, "Lượt 2 bị từ chối sai lệch!"
        assert len(t2_data["citations"]) > 0
        ans2 = t2_data["answer"]
        ans2_lower = ans2.lower()

        # Classification check: 2 methods must be explicitly listed
        assert "thay thế" in ans2_lower
        assert "hoán vị" in ans2_lower
        # No page bleed
        assert "=== page" not in ans2_lower
        # Collect suggested questions
        sq2 = t2_data.get("suggested_questions", [])
        assert len(sq2) >= 1
        collected_suggested_questions.extend(sq2)

        # ----------------------------------------------------------------------
        # LƯỢT 3: Hỏi sâu mục cụ thể ('trình bày mục 3. PHÁ MÃ')
        # ----------------------------------------------------------------------
        turn3_res = self.client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={
                "query": "trình bày mục 3. PHÁ MÃ",
                "answer_mode": "detailed",
            },
        )
        assert turn3_res.status_code == 200
        t3_data = turn3_res.json()

        assert t3_data["refused"] is False, "Lượt 3 bị từ chối sai lệch!"
        assert len(t3_data["citations"]) > 0
        assert t3_data["routing"] == "SECTION_LOOKUP"
        ans3 = t3_data["answer"]
        ans3_lower = ans3.lower()

        # Section 3 facts: An ninh vô điều kiện / An ninh tính toán / thám mã
        assert "an ninh" in ans3_lower or "phá mã" in ans3_lower
        assert any(term in ans3_lower for term in ["vô điều kiện", "tính toán", "vét cạn", "thống kê"])
        assert "=== page" not in ans3_lower
        sq3 = t3_data.get("suggested_questions", [])
        assert len(sq3) >= 1
        collected_suggested_questions.extend(sq3)

        # ----------------------------------------------------------------------
        # LƯỢT 4: Hỏi tiếp về 'Hệ mã hóa Playfair'
        # ----------------------------------------------------------------------
        turn4_res = self.client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={
                "query": "Hệ mã hóa Playfair",
                "answer_mode": "detailed",
            },
        )
        assert turn4_res.status_code == 200
        t4_data = turn4_res.json()

        assert t4_data["refused"] is False, "Lượt 4 bị từ chối sai lệch!"
        assert len(t4_data["citations"]) > 0
        ans4 = t4_data["answer"]
        ans4_lower = ans4.lower()

        # Playfair facts: ma trận 5x5, cặp ký tự (digram)
        assert any(term in ans4_lower for term in ["playfair", "5x5", "digram", "cặp ký tự", "ma trận"])
        assert "=== page" not in ans4_lower
        sq4 = t4_data.get("suggested_questions", [])
        assert len(sq4) >= 1
        collected_suggested_questions.extend(sq4)

        # ----------------------------------------------------------------------
        # LƯỢT 5: Xác minh 100% câu hỏi gợi ý (Click-Through) trả lời thành công
        # ----------------------------------------------------------------------
        # Deduplicate while preserving order
        unique_sqs = []
        for q in collected_suggested_questions:
            if q not in unique_sqs:
                unique_sqs.append(q)

        assert len(unique_sqs) >= 4, "Số lượng câu hỏi gợi ý thu được không đủ đa dạng!"

        for q_clicked in unique_sqs:
            # Must not have robotic placeholder
            assert "Yêu cầu và chi tiết cụ thể liên quan đến" not in q_clicked

            click_res = self.client.post(
                f"/api/chat/sessions/{session_id}/messages",
                json={"query": q_clicked},
            )
            assert click_res.status_code == 200
            click_data = click_res.json()

            # Mandatory requirement: 100% answerable, refused == False
            assert click_data["refused"] is False, f"Câu hỏi gợi ý '{q_clicked}' bị từ chối!"
            assert len(click_data["answer"]) > 50, f"Câu trả lời quá ngắn cho '{q_clicked}'"
            assert "=== page" not in click_data["answer"].lower()

    def test_strict_document_scope_locking_isolation(self):
        """
        Verify that when a session is locked to DOC_REDTEAM_SLIDE_CRYPTO:
        1. Answering questions in that doc works.
        2. Inquiring about topics from DOC_REDTEAM_UNRELATED ('cho tôi biết topic 49')
           is strictly refused (refused == True) with ZERO leakage.
        """
        sess_res = self.client.post(
            "/api/chat/sessions",
            json={
                "title": "Scope Isolation Test",
                "document_ids": ["DOC_REDTEAM_SLIDE_CRYPTO"],
            },
        )
        assert sess_res.status_code == 200
        session_id = sess_res.json()["session_id"]

        # Ask question from the other document (DOC_REDTEAM_UNRELATED)
        foreign_res = self.client.post(
            f"/api/chat/sessions/{session_id}/messages",
            json={"query": "cho tôi biết topic 49"},
        )
        assert foreign_res.status_code == 200
        data = foreign_res.json()

        # Must refuse cleanly due to strict document scope locking
        assert data["refused"] is True, "Hệ thống bị rò rỉ dữ liệu từ tài liệu không được đính kèm!"
        assert "DOC_REDTEAM_SLIDE_CRYPTO" in data["answer"] or "Slide 2" in data["answer"]
        assert "basketball" not in data["answer"].lower()
        assert "teamwork" not in data["answer"].lower()

    def test_multi_turn_anaphoric_followup_in_session(self):
        """
        Test that within a session:
        1. Anaphoric follow-up queries carry over the topic anchor:
           Turn A: 'cho tôi biết topic 49'
           Turn B: 'giải thích thêm chi tiết' -> carries over topic anchor (topic 49)
        2. Pronoun follow-up queries resolve to prior entity:
           Turn C: 'Hệ mã hóa Playfair'
           Turn D: 'nó sử dụng ma trận như thế nào' -> resolves 'nó' to 'Hệ mã hóa Playfair'
        """
        # 1. Anaphoric Topic Anchor Carry-Over
        sess_ielts = self.client.post(
            "/api/chat/sessions",
            json={
                "title": "Anaphoric Anchor Test",
                "document_ids": ["DOC_REDTEAM_UNRELATED"],
            },
        )
        assert sess_ielts.status_code == 200
        s_ielts_id = sess_ielts.json()["session_id"]

        res_a = self.client.post(
            f"/api/chat/sessions/{s_ielts_id}/messages",
            json={"query": "cho tôi biết topic 49"},
        )
        assert res_a.status_code == 200
        assert res_a.json()["refused"] is False

        res_b = self.client.post(
            f"/api/chat/sessions/{s_ielts_id}/messages",
            json={"query": "giải thích thêm chi tiết"},
        )
        assert res_b.status_code == 200
        b_data = res_b.json()
        assert b_data["refused"] is False
        assert "topic 49" in b_data.get("effective_query", "").lower()
        assert "basketball" in b_data["answer"].lower() or "teamwork" in b_data["answer"].lower()

        # 2. Pronoun Multi-Turn Resolution
        sess_crypto = self.client.post(
            "/api/chat/sessions",
            json={
                "title": "Pronoun Resolution Test",
                "document_ids": ["DOC_REDTEAM_SLIDE_CRYPTO"],
            },
        )
        assert sess_crypto.status_code == 200
        s_crypto_id = sess_crypto.json()["session_id"]

        res_c = self.client.post(
            f"/api/chat/sessions/{s_crypto_id}/messages",
            json={"query": "Hệ mã hóa Playfair"},
        )
        assert res_c.status_code == 200
        assert res_c.json()["refused"] is False

        res_d = self.client.post(
            f"/api/chat/sessions/{s_crypto_id}/messages",
            json={"query": "nó sử dụng ma trận như thế nào"},
        )
        assert res_d.status_code == 200
        d_data = res_d.json()
        assert d_data["refused"] is False
        assert "playfair" in d_data.get("effective_query", "").lower()
        assert "5x5" in d_data["answer"].lower() or "ma trận" in d_data["answer"].lower()

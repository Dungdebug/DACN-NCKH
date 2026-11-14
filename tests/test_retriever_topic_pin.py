"""
Unit tests for BM25Retriever topic pinning & digit tokenization fixes.

Verifies:
1. _tokenize keeps pure-digit tokens (len(t) > 1 or t.isdigit()).
2. Topic pin with aliases (topic|chủ đề|chu de|bài|mục|chương|phần|section|chapter|lesson|unit).
3. "chủ đề 47" returns only Topic 47.
4. "topic 1" never returns Topic 10 or Topic 50 even if they contain the word 'topic'.
5. Section_title match priority (+30.0) over text match (+15.0).
6. Graceful fallback when target topic does not exist in corpus.
"""

import unittest
from src.hybrid_qa.document_store import Passage
from src.hybrid_qa.retriever import BM25Retriever


def make_passage(passage_id: str, document_id: str, text: str, section_title: str = "") -> Passage:
    return Passage(
        passage_id=passage_id,
        document_id=document_id,
        text=text,
        start_char=0,
        end_char=len(text),
        section_title=section_title,
    )


class TestRetrieverTopicPin(unittest.TestCase):
    """Test suite for topic pinning and digit tokenization."""

    def test_tokenize_preserves_single_digits(self):
        """_tokenize keeps single digits but filters single non-digit letters."""
        tokens = BM25Retriever._tokenize("Topic 1 is a great lesson 2 for x and y")
        self.assertIn("1", tokens)
        self.assertIn("2", tokens)
        self.assertIn("topic", tokens)
        self.assertIn("lesson", tokens)
        self.assertNotIn("a", tokens)
        self.assertNotIn("x", tokens)
        self.assertNotIn("y", tokens)

    def test_topic_1_never_returns_topic_10_or_50(self):
        """'topic 1' retrieves Topic 1 and excludes Topic 10 and Topic 50."""
        passages = [
            make_passage(
                passage_id="P000",
                document_id="DOC_SPEAK",
                text="Describe a time when you were ill. You should say what the illness was and how you recovered.",
                section_title="Topic 1: Describe a time when you were ill",
            ),
            make_passage(
                passage_id="P010",
                document_id="DOC_SPEAK",
                text="Describe a movie you watched recently. It was an interesting movie topic with high ratings.",
                section_title="Topic 10: Describe a movie",
            ),
            make_passage(
                passage_id="P021",
                document_id="DOC_SPEAK",
                text="Describe an interesting video you watched online. It was a short video about scientific topics.",
                section_title="Topic 50: Describe an interesting video",
            ),
            make_passage(
                passage_id="P099",
                document_id="DOC_SPEAK",
                text="General guidelines for IELTS speaking preparation.",
                section_title="Introduction",
            ),
        ]
        retriever = BM25Retriever()
        retriever.build_index(passages)

        results = retriever.retrieve("topic 1", top_k=5)
        self.assertGreater(len(results), 0)

        # Topic 1 must be top result
        self.assertEqual(results[0].passage_id, "P000")
        self.assertIn("Topic 1", results[0].section_title)

        # Results must NEVER contain Topic 10 or Topic 50
        result_section_titles = [r.section_title for r in results]
        for title in result_section_titles:
            self.assertNotIn("Topic 10", title)
            self.assertNotIn("Topic 50", title)

    def test_chu_de_47_returns_only_topic_47(self):
        """'cho tôi biết chủ đề 47' returns only Topic 47 passages, excluding Topic 48 and Topic 49."""
        passages = [
            make_passage(
                passage_id="P007",
                document_id="DOC_SPEAK",
                text="Topic 47: Describe a success in your life. You should say when it happened and how you felt.",
                section_title="Topic 47: Describe a success",
            ),
            make_passage(
                passage_id="P008",
                document_id="DOC_SPEAK",
                text="This success was meaningful because I put a lot of effort into practicing for topic 47.",
                section_title="Topic 47: Describe a success",
            ),
            make_passage(
                passage_id="P016",
                document_id="DOC_SPEAK",
                text="Topic 48: Describe a bad service at a restaurant that you experienced.",
                section_title="Topic 48: Bad service",
            ),
            make_passage(
                passage_id="P017",
                document_id="DOC_SPEAK",
                text="Topic 49: Describe an outdoor sport that you would like to try in the future.",
                section_title="Topic 49: Outdoor sport",
            ),
        ]
        retriever = BM25Retriever()
        retriever.build_index(passages)

        # With Vietnamese accented query
        results = retriever.retrieve("cho tôi biết chủ đề 47", top_k=5)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("Topic 47", r.section_title)
            self.assertNotIn("Topic 48", r.section_title)
            self.assertNotIn("Topic 49", r.section_title)

        # With unaccented query "chu de 47"
        results_no_accent = retriever.retrieve("cho toi biet chu de 47", top_k=5)
        self.assertGreater(len(results_no_accent), 0)
        for r in results_no_accent:
            self.assertIn("Topic 47", r.section_title)

    def test_topic_alias_variety(self):
        """Tests various aliases like bài, mục, chương, phần, section, unit."""
        passages = [
            make_passage(
                passage_id="SEC_1",
                document_id="DOC_BOOK",
                text="Nội dung chương 1 giới thiệu tổng quan hệ thống phân tích.",
                section_title="Chương 1: Giới thiệu",
            ),
            make_passage(
                passage_id="SEC_2",
                document_id="DOC_BOOK",
                text="Nội dung chương 2 mô tả chi tiết phương pháp nghiên cứu.",
                section_title="Chương 2: Phương pháp",
            ),
            make_passage(
                passage_id="SEC_3",
                document_id="DOC_BOOK",
                text="Nội dung bài 3 giải thích bài toán xử lý chuỗi.",
                section_title="Bài 3: Xử lý chuỗi",
            ),
            make_passage(
                passage_id="SEC_4",
                document_id="DOC_BOOK",
                text="Unit 5 provides exercises for vocabulary retention.",
                section_title="Unit 5: Vocabulary",
            ),
        ]
        retriever = BM25Retriever()
        retriever.build_index(passages)

        # "chương 2"
        res_ch2 = retriever.retrieve("chương 2", top_k=3)
        self.assertEqual(res_ch2[0].passage_id, "SEC_2")
        self.assertNotIn("SEC_1", [r.passage_id for r in res_ch2])

        # "bài 3"
        res_b3 = retriever.retrieve("bài 3", top_k=3)
        self.assertEqual(res_b3[0].passage_id, "SEC_3")

        # "unit 5"
        res_u5 = retriever.retrieve("unit 5", top_k=3)
        self.assertEqual(res_u5[0].passage_id, "SEC_4")

    def test_section_title_boost_over_body_match(self):
        """Passage declaring the topic in section_title ranks higher than one merely mentioning it in text."""
        passages = [
            make_passage(
                passage_id="MENTION_ONLY",
                document_id="DOC1",
                text="Trong phần thảo luận, tác giả có nhắc đến bài 5 và các liên hệ mở rộng.",
                section_title="Thảo luận chung",
            ),
            make_passage(
                passage_id="TITLE_MATCH",
                document_id="DOC1",
                text="Nội dung chính thức về thuật toán phân tán.",
                section_title="Bài 5: Thuật toán phân tán",
            ),
        ]
        retriever = BM25Retriever()
        retriever.build_index(passages)

        results = retriever.retrieve("bài 5", top_k=2)
        self.assertEqual(results[0].passage_id, "TITLE_MATCH")

    def test_nonexistent_topic_fallback(self):
        """When query specifies a topic number not in the corpus, retrieval does not fail or exclude everything."""
        passages = [
            make_passage(
                passage_id="P1",
                document_id="DOC1",
                text="Mô hình học máy đạt độ chính xác cao.",
                section_title="Kết quả",
            ),
        ]
        retriever = BM25Retriever()
        retriever.build_index(passages)

        # Topic 999 does not exist in passages
        results = retriever.retrieve("topic 999 độ chính xác", top_k=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].passage_id, "P1")

    def test_numbered_section_digit_pin_excludes_neighbors(self):
        """'9. HỆ MÃ HOÁ DÒNG' pins section 9 and strictly excludes sections 8 and 10."""
        passages = [
            make_passage(
                passage_id="P008",
                document_id="DOC_CRYPTO",
                text="8. HỆ MÃ HOÁ ĐỘN MỘT LẦN:\nHệ mã hóa hoàn hảo dùng khóa ngẫu nhiên một lần.",
                section_title="8. HỆ MÃ HOÁ ĐỘN MỘT LẦN:",
            ),
            make_passage(
                passage_id="P009",
                document_id="DOC_CRYPTO",
                text="9. HỆ MÃ HOÁ DÒNG:\nSinh ra chuỗi khóa z = z1, z2, z3 và mã hóa XOR.",
                section_title="9. HỆ MÃ HOÁ DÒNG:",
            ),
            make_passage(
                passage_id="P010",
                document_id="DOC_CRYPTO",
                text="10. HỆ MÃ HOÁ AUTOKEY:\nKhóa k được mở rộng bằng ký tự bản rõ.",
                section_title="10. HỆ MÃ HOÁ AUTOKEY:",
            ),
        ]
        retriever = BM25Retriever()
        retriever.build_index(passages)

        res9 = retriever.retrieve("9. HỆ MÃ HOÁ DÒNG", top_k=5)
        self.assertEqual([r.passage_id for r in res9], ["P009"])

        res10 = retriever.retrieve("10. HỆ MÃ HOÁ AUTOKEY", top_k=5)
        self.assertEqual([r.passage_id for r in res10], ["P010"])

        # Cross-format bridge: 'mục 9' pins '9. HỆ MÃ HOÁ DÒNG:'
        res_muc9 = retriever.retrieve("mục 9", top_k=5)
        self.assertEqual([r.passage_id for r in res_muc9], ["P009"])

    def test_non_heading_numbers_not_pinned(self):
        """Years or inline constants like 'năm 1970.' or 'k = 3.' do not trigger false topic pinning."""
        from src.hybrid_qa.retriever import extract_topic_and_section_numbers
        self.assertEqual(
            extract_topic_and_section_numbers("Các hệ mã trước năm 1970. Có bao nhiêu phương pháp?"),
            set(),
        )
        self.assertEqual(
            extract_topic_and_section_numbers("Với k = 3. Kết quả là gì?"),
            set(),
        )

    def test_document_scope_lock_empty_and_mismatched_allowed_ids(self):
        """Empty allowed_document_ids or disjoint document_id returns [] instead of global corpus."""
        passages = [
            make_passage("P1", "DOC_A", "Hệ mã dịch vòng Caesar cipher", "2. HỆ MÃ DỊCH VÒNG"),
            make_passage("P2", "DOC_B", "Hệ mã hóa dòng Stream cipher", "9. HỆ MÃ HOÁ DÒNG"),
        ]
        retriever = BM25Retriever()
        retriever.build_index(passages)

        self.assertEqual(retriever.retrieve("hệ mã", top_k=5, allowed_document_ids=[]), [])
        self.assertEqual(
            retriever.retrieve("hệ mã", top_k=5, document_id="DOC_A", allowed_document_ids=["DOC_B"]),
            [],
        )
        res_a = retriever.retrieve("hệ mã", top_k=5, allowed_document_ids=["DOC_A"])
        self.assertEqual([r.document_id for r in res_a], ["DOC_A"])


if __name__ == "__main__":
    unittest.main()

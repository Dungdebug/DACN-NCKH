"""
Unit and integration tests for BM25Retriever Wave 2 enhancements:
- Precomputed term frequencies
- Accent-insensitive fallback (strip_accents, đ->d)
- Proximity bonus with minimal window span
- MMR diversification with repeated section penalty
"""

import math
import unittest
from collections import Counter
from src.hybrid_qa.document_store import Passage
from src.hybrid_qa.retriever import BM25Retriever, RetrievalResult, strip_accents


def make_passage(passage_id: str, document_id: str, text: str, section_title: str = "") -> Passage:
    return Passage(
        passage_id=passage_id,
        document_id=document_id,
        text=text,
        start_char=0,
        end_char=len(text),
        section_title=section_title,
    )


class TestRetrieverV5Hybrid(unittest.TestCase):
    """Test suite for W2-A retriever enhancements."""

    def setUp(self):
        self.sample_passages = [
            make_passage(
                passage_id="DOC001::P000",
                document_id="DOC001",
                text="Hạn chế lớn nhất của nghiên cứu này là kích thước tập mẫu còn nhỏ.",
                section_title="Giới hạn và Thảo luận",
            ),
            make_passage(
                passage_id="DOC001::P001",
                document_id="DOC001",
                text="Mô hình học máy đạt độ chính xác cao trên tập dữ liệu thử nghiệm.",
                section_title="Kết quả thực nghiệm",
            ),
            make_passage(
                passage_id="DOC001::P002",
                document_id="DOC001",
                text="Đặc trưng quan trọng được trích xuất tự động qua các tầng tích chập.",
                section_title="Phương pháp đề xuất",
            ),
            make_passage(
                passage_id="DOC002::P000",
                document_id="DOC002",
                text="Thuật toán học máy xử lý ngôn ngữ tự nhiên một cách hiệu quả.",
                section_title="Phương pháp đề xuất",
            ),
            make_passage(
                passage_id="DOC002::P001",
                document_id="DOC002",
                text="Quá trình học tập diễn ra liên tục khi các thiết bị máy móc vận hành tự động.",
                section_title="Cơ sở lý thuyết",
            ),
        ]
        self.retriever = BM25Retriever()
        self.retriever.build_index(self.sample_passages)

    def test_precomputed_term_freqs_and_identical_scores(self):
        """1. Precomputed _term_freqs exists and BM25 scores match exact manual computation."""
        self.assertEqual(len(self.retriever._term_freqs), len(self.sample_passages))
        for tf in self.retriever._term_freqs:
            self.assertIsInstance(tf, Counter)

        # Compute manual BM25 score for query "học máy" on DOC001::P001
        doc_idx = 1
        query_tokens = ["học", "máy"]
        doc_tokens = self.retriever.tokenized_corpus[doc_idx]
        dl = len(doc_tokens)
        avg_dl = self.retriever.avg_dl
        k1, b = self.retriever.k1, self.retriever.b

        expected_score = 0.0
        for term in query_tokens:
            tf = Counter(doc_tokens).get(term, 0)
            df = self.retriever.doc_freqs.get(term, 0)
            idf = math.log((len(self.sample_passages) - df + 0.5) / (df + 0.5) + 1.0)
            num = tf * (k1 + 1)
            den = tf + k1 * (1 - b + b * dl / avg_dl)
            expected_score += idf * num / den

        actual_score = self.retriever._score_document(query_tokens, doc_idx)
        self.assertAlmostEqual(actual_score, expected_score, places=9)

    def test_accent_insensitive_fallback_unaccented_query(self):
        """2. Unaccented Vietnamese queries match accented text via folded index fallback."""
        # "han che" -> matches "Hạn chế" in DOC001::P000
        results = self.retriever.retrieve("han che", top_k=3)
        self.assertTrue(len(results) > 0)
        top = results[0]
        self.assertEqual(top.passage_id, "DOC001::P000")
        self.assertGreater(top.score, 0.0)
        self.assertIn("Hạn chế", top.text)

        # "dac trung" -> matches "Đặc trưng" in DOC001::P002 (handles 'đ' -> 'd')
        results_dt = self.retriever.retrieve("dac trung", top_k=3)
        self.assertTrue(len(results_dt) > 0)
        self.assertEqual(results_dt[0].passage_id, "DOC001::P002")

        # "kich thuoc tap mau nho" -> matches DOC001::P000
        results_kt = self.retriever.retrieve("kich thuoc tap mau nho", top_k=3)
        self.assertTrue(len(results_kt) > 0)
        self.assertEqual(results_kt[0].passage_id, "DOC001::P000")

    def test_accent_query_uses_raw_matching(self):
        """Accented queries use raw doc_freqs when raw term exists."""
        results = self.retriever.retrieve("hạn chế", top_k=3)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].passage_id, "DOC001::P000")
        self.assertGreater(results[0].score, 0.0)

    def test_strip_accents_helper(self):
        """strip_accents properly strips tones and maps đ/Đ to d/D."""
        self.assertEqual(strip_accents("Hạn chế"), "Han che")
        self.assertEqual(strip_accents("đặc trưng"), "dac trung")
        self.assertEqual(strip_accents("ĐỒ ÁN TỐT NGHIỆP"), "DO AN TOT NGHIEP")
        self.assertEqual(strip_accents("Nguyễn Đình Chiểu"), "Nguyen Dinh Chieu")

    def test_proximity_bonus_reorders_adjacent_phrase(self):
        """3. Proximity bonus with minimal window span reorders adjacent-phrase passage first."""
        # DOC002::P000 has adjacent "học máy" (span = 1, pos_diff = 1)
        # DOC002::P001 has distant "học ... máy" (separated by 8 words)
        # Baseline with proximity_weight=0.0:
        base_retriever = BM25Retriever(proximity_weight=0.0)
        base_retriever.build_index(self.sample_passages)

        score_adj_base = base_retriever._score_document(["học", "máy"], 3)  # P000
        score_dist_base = base_retriever._score_document(["học", "máy"], 4)  # P001

        # With proximity_weight > 0.0:
        prox_retriever = BM25Retriever(proximity_weight=2.0)
        prox_retriever.build_index(self.sample_passages)

        score_adj_prox = prox_retriever._score_document(["học", "máy"], 3)
        score_dist_prox = prox_retriever._score_document(["học", "máy"], 4)

        # Proximity bonus for adjacent should be strictly greater than for distant
        prox_bonus_adj = score_adj_prox - score_adj_base
        prox_bonus_dist = score_dist_prox - score_dist_base
        self.assertGreater(prox_bonus_adj, prox_bonus_dist)

        # Verify minimal window formula: (m - 1) / (1 + ln(1 + span))
        # Adjacent has m=2, span=1 -> (2 - 1) / (1 + ln(2)) = 1 / 1.6931... = ~0.5906
        expected_adj_bonus = 2.0 * (1.0 / (1.0 + math.log(2.0)))
        self.assertAlmostEqual(prox_bonus_adj, expected_adj_bonus, places=5)

        # When searching with proximity retriever on documents 002:
        res = prox_retriever.retrieve("học máy", top_k=2, allowed_document_ids={"DOC002"})
        self.assertEqual(res[0].passage_id, "DOC002::P000")

    def test_diversify_mmr_returns_multiple_sections(self):
        """4. MMR diversification with section penalty returns >= 2 distinct sections."""
        # Create a synthetic corpus where top 3 passages belong to the SAME section
        passages = [
            make_passage(
                passage_id="P1",
                document_id="DOC1",
                text="Mô hình học sâu mạng nơ-ron tích chập ứng dụng phân loại ảnh số 1.",
                section_title="Phương pháp đề xuất",
            ),
            make_passage(
                passage_id="P2",
                document_id="DOC1",
                text="Mô hình học sâu mạng nơ-ron tích chập phân loại hình ảnh số 2.",
                section_title="Phương pháp đề xuất",
            ),
            make_passage(
                passage_id="P3",
                document_id="DOC1",
                text="Mô hình học sâu mạng nơ-ron tích chập nhận dạng mẫu số 3.",
                section_title="Phương pháp đề xuất",
            ),
            make_passage(
                passage_id="P4",
                document_id="DOC1",
                text="Mô hình học sâu đạt độ chính xác 98% trên tập kiểm thử phân loại.",
                section_title="Kết quả thực nghiệm",
            ),
            make_passage(
                passage_id="P5",
                document_id="DOC1",
                text="Nghiên cứu về lý thuyết mạng nơ-ron và phân loại dữ liệu ảnh.",
                section_title="Tổng quan tài liệu",
            ),
        ]
        retriever = BM25Retriever()
        retriever.build_index(passages)

        # Without diversity: top 2 are both from the same section "Phương pháp đề xuất"
        res_plain = retriever.retrieve("học sâu mạng nơ-ron phân loại", top_k=2, diversify=False)
        self.assertEqual(len(res_plain), 2)
        plain_sections = {r.section_title for r in res_plain}
        self.assertEqual(len(plain_sections), 1, "Without diversity, top 2 are from the same section")

        # With diversity: repeated section title receives 0.15 penalty, promoting distinct sections
        res_div = retriever.retrieve("học sâu mạng nơ-ron phân loại", top_k=2, diversify=True, mmr_lambda=0.6)
        self.assertEqual(len(res_div), 2)
        distinct_sections = {r.section_title for r in res_div}
        self.assertGreaterEqual(len(distinct_sections), 2, "diversify must return >= 2 distinct sections")

    def test_edge_cases_and_backwards_compatibility(self):
        """Verify edge cases: empty queries, single term, no passages, add_passages."""
        # Empty query
        self.assertEqual(self.retriever.retrieve("", top_k=3), [])
        self.assertEqual(self.retriever.retrieve("   ", top_k=3), [])

        # Single term query (proximity bonus is 0.0)
        score_single_base = self.retriever._score_document(["học"], 1)
        prox_retriever = BM25Retriever(proximity_weight=5.0)
        prox_retriever.build_index(self.sample_passages)
        score_single_prox = prox_retriever._score_document(["học"], 1)
        self.assertAlmostEqual(score_single_base, score_single_prox, places=9)

        # Unindexed retriever
        empty = BM25Retriever()
        self.assertEqual(empty.retrieve("test"), [])

        # add_passages re-indexes correctly
        new_p = make_passage(
            passage_id="DOC003::P000",
            document_id="DOC003",
            text="Trí tuệ nhân tạo thế hệ mới với khả năng suy luận logic.",
            section_title="Tương lai",
        )
        self.retriever.add_passages([new_p])
        self.assertEqual(self.retriever.passage_count, 6)
        res = self.retriever.retrieve("tri tue nhan tao", top_k=1)
        self.assertEqual(res[0].passage_id, "DOC003::P000")

    def test_quoted_entity_and_concept_boost(self):
        """Verify +5.0 boost for quoted entity / section match and +3.0 boost for concept+types queries."""
        p_intro = make_passage(
            passage_id="CRYPTO::P001",
            document_id="CRYPTO",
            text="Xuất hiện đầu tiên trong ngành mã hóa. Có 2 phương pháp phổ biến: * Mã hóa thay thế. * Mã hóa hoán vị.",
            section_title="1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG",
        )
        p_attack = make_passage(
            passage_id="CRYPTO::P003",
            document_id="CRYPTO",
            text="3. PHÁ MÃ: An ninh hệ mã khóa. Các phương pháp thám mã phổ biến gồm tấn công vét cạn và phân tích thống kê mã hóa.",
            section_title="3. PHÁ MÃ",
        )
        retriever = BM25Retriever()
        retriever.build_index([p_intro, p_attack])

        # 1. Quoted entity 'PHÁ MÃ' boosts P003
        results_quoted = retriever.retrieve("Yêu cầu liên quan đến 'PHÁ MÃ' gồm những gì?", top_k=2)
        self.assertEqual(results_quoted[0].passage_id, "CRYPTO::P003")

        # 2. General concept + types query boosts P001 to Top 1
        results_concept = retriever.retrieve("mã hóa đối xứng là gì, có bao nhiêu loại mã hóa, cho ví dụ", top_k=2)
        self.assertEqual(results_concept[0].passage_id, "CRYPTO::P001")


if __name__ == "__main__":
    unittest.main()

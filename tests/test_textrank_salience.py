"""
Tests for TextRank Salience & Key Point Selection (W2-C).
Verifies:
1. Pure-Python graph TextRank: central sentence outranks outlier.
2. Accent-folded token matching & Vietnamese stopword pruning.
3. select_key_points combining TextRank + heuristic markers with MMR deduplication (no near-duplicates).
4. extract_claim_evidence_pairs linking claim sentences to empirical metric sentences.
5. DocumentAnalyzer.extract_key_points_and_summary integration & fallback resilience.
"""

import pytest
from src.brain.textrank_salience import (
    strip_accents,
    tokenize,
    sentence_similarity,
    rank_sentences,
    heuristic_score,
    select_key_points,
    extract_claim_evidence_pairs,
)
from src.web.document_analyzer import DocumentAnalyzer


def test_strip_accents_and_tokenize():
    text = "Hệ thống SA-CMS tối ưu hóa bộ nhớ và tăng tốc độ xử lý."
    tokens = tokenize(text)
    # Stopwords like 'va' should be removed, tone marks stripped
    assert "va" not in tokens
    assert "he" in tokens
    assert "thong" in tokens
    assert "sa" in tokens
    assert "cms" in tokens
    assert "toi" not in tokens or "uu" in tokens  # 'toi' might be stopword (tôi/tới), 'uu' is content


def test_textrank_central_sentence_outranks_outlier():
    sentences = [
        "Mô hình học máy xử lý dữ liệu và huấn luyện tham số.",       # Central node (shared words: mo hinh, hoc may, du lieu, tham so)
        "Học máy phân tích dữ liệu lớn với mô hình tối ưu.",           # Central node
        "Dữ liệu được xử lý bởi mô hình học máy tham số lớn.",         # Central node
        "Thời tiết hôm nay trời nắng đẹp và không có mây râm mát.",     # Outlier node (zero shared content words)
    ]
    scores = rank_sentences(sentences)
    assert len(scores) == 4
    # All central sentences should have substantially higher score than outlier
    assert scores[0] > scores[3]
    assert scores[1] > scores[3]
    assert scores[2] > scores[3]


def test_textrank_edge_cases():
    assert rank_sentences([]) == []
    assert rank_sentences(["Một câu duy nhất."]) == [1.0]
    # Two disconnected sentences
    scores = rank_sentences(["Con mèo trèo cây cau.", "Chiếc xe chạy trên đường."])
    assert len(scores) == 2
    assert abs(scores[0] - scores[1]) < 1e-4


def test_select_key_points_mmr_prevents_near_duplicates():
    sentences = [
        "Phương pháp đạt hiệu năng tối ưu 89.6% F1 trên bộ dữ liệu kiểm thử thực nghiệm.", # S0: result + metric
        "Phương pháp đạt hiệu năng tối ưu 89.6% F1 trên tập dữ liệu kiểm thử thực nghiệm.", # S1: near duplicate of S0!
        "Chúng tôi đề xuất mô hình SA-CMS với kiến trúc 3 tầng bộ nhớ tham số.",           # S2: architecture / proposal
        "Hạn chế kỹ thuật lớn nhất là yêu cầu văn bản đầu vào phải phân đoạn rõ ràng.",       # S3: limitation
    ]
    # Pick top 3 points
    key_points = select_key_points(sentences, k=3, mmr_lambda=0.7)
    assert len(key_points) == 3

    # MMR should penalize near-duplicates: S0 and S1 should NOT both be selected
    has_s0 = sentences[0] in key_points
    has_s1 = sentences[1] in key_points
    assert not (has_s0 and has_s1), "MMR failed to filter out the near-duplicate sentence!"

    # Both architecture and limitation should be selected
    assert sentences[2] in key_points
    assert sentences[3] in key_points


def test_select_key_points_edge_cases():
    assert select_key_points([], k=4) == []
    assert select_key_points(["Câu đơn."], k=0) == []
    single = ["Đề xuất kiến trúc SA-CMS đạt 89.6% F1."]
    assert select_key_points(single, k=3) == single


def test_extract_claim_evidence_pairs():
    sentences = [
        "Chúng tôi đề xuất mô hình SA-CMS cải tiến hiệu năng bộ nhớ liên tục.",              # Claim (đề xuất)
        "Thực nghiệm cho thấy phương pháp đạt độ chính xác 89.6% F1 và giảm 87.5% chi phí.", # Claim (cho thấy) + Evidence (89.6% F1, 87.5%)
        "Hạn chế của mô hình là đòi hỏi tài nguyên tính toán lúc khởi động.",                # Limitation, no metric
        "Khảo sát độ trễ ghi nhận hệ thống phản hồi trong 250 ms trên mỗi truy vấn.",         # Metric evidence (250 ms)
    ]

    pairs = extract_claim_evidence_pairs(sentences)
    assert len(pairs) >= 1

    # First claim (đề xuất) should link to empirical sentence with metric
    claims = [p[0] for p in pairs]
    assert sentences[0] in claims

    for claim, evidence in pairs:
        assert isinstance(claim, str) and isinstance(evidence, str)
        # Evidence must contain a metric
        assert any(unit in evidence.lower() for unit in ["%", "ms", "f1", "chi phí", "token", "mb"])


def test_extract_claim_evidence_pairs_no_metrics():
    sentences = [
        "Chúng tôi đề xuất một hướng tiếp cận lý thuyết mới.",
        "Nghiên cứu chỉ ra tính khả thi của giải thuật.",
    ]
    # No sentences with metrics -> no pairs
    assert extract_claim_evidence_pairs(sentences) == []


def test_document_analyzer_extract_key_points_keys_unchanged():
    raw_text = (
        "# 1. Giới thiệu Kiến trúc\n\n"
        "Chúng tôi đề xuất hệ thống SA-CMS sử dụng mô hình SmolLM2 với 135 triệu tham số.\n\n"
        "# 2. Kết quả Thực nghiệm\n\n"
        "Phương pháp đạt độ chính xác 89.6% F1, giảm 87.5% chi phí token và chỉ chiếm 743 MB VRAM.\n\n"
        "# 3. Thách thức và Hạn chế\n\n"
        "Tuy nhiên, hệ thống có hạn chế là đòi hỏi văn bản đầu vào phải có cấu trúc phân đoạn rõ ràng."
    )
    result = DocumentAnalyzer.extract_key_points_and_summary(raw_text, title="Tài liệu Nghiên cứu", max_points=3)

    # Return shape must be 100% unchanged
    assert isinstance(result, dict)
    assert set(result.keys()) == {"overview", "key_points"}
    assert isinstance(result["overview"], str)
    assert isinstance(result["key_points"], list)
    assert len(result["key_points"]) <= 3
    assert "Tài liệu 'Tài liệu Nghiên cứu' tập trung vào các nội dung trọng tâm:" in result["overview"]


def test_document_analyzer_fallback_on_exception(monkeypatch):
    import src.brain.textrank_salience

    def broken_select(*args, **kwargs):
        raise RuntimeError("Simulated failure in TextRank module")

    monkeypatch.setattr(src.brain.textrank_salience, "select_key_points", broken_select)

    raw_text = "Hệ thống SA-CMS đề xuất tối ưu 89.6% F1. Hạn chế là tốc độ nạp dữ liệu ban đầu."
    result = DocumentAnalyzer.extract_key_points_and_summary(raw_text, title="Kiểm tra Fallback")

    assert isinstance(result, dict)
    assert set(result.keys()) == {"overview", "key_points"}
    assert len(result["key_points"]) > 0

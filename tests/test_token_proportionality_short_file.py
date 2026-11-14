"""
Unit tests for Token Proportionality & Short File Fairness (Wave 4 - Calibrator).
Verifies that:
1. Short 2-line files (e.g. alooo.txt: "toi là ai?\\r\\n\\tkiet sieu dep trai") produce
   proportional token counts (raw_context_tokens <= 10 and compressed_context_tokens <= 10)
   without being inflated by internal [passage_id] tags or BPE UTF-8 byte fragmentation.
2. The token count of the 2-line file is 15-30x smaller than a standard 50-line document (Topic Speak).
3. The count_text_tokens function correctly caps BPE fragmentation on Vietnamese diacritics.
"""

import pytest
from typing import List
from src.web.token_optimizer import (
    compress_evidence_context,
    count_text_tokens,
)


class MockByteExplodingBPETokenizer:
    """Simulates a tokenizer lacking native Vietnamese vocab that fragments diacritics into UTF-8 bytes."""
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # Emulates SmolLM2 or English-only BPE splitting Vietnamese accents into 4-6 bytes per word
        raw_bytes = text.encode("utf-8")
        return list(raw_bytes)


def test_short_file_token_proportionality_alooo_txt():
    # File 2 dong: "toi là ai?\r\n\tkiet sieu dep trai"
    short_text = "toi là ai?\r\n\tkiet sieu dep trai"
    short_passages = [
        {
            "passage_id": "DOC_1794280705781_b5d5c4::P000",
            "text": short_text,
        }
    ]

    # Test 1: Without explicit tokenizer (word-heuristic)
    _, stats_no_tok = compress_evidence_context(
        question="Bạn là ai?",
        evidence_passages=short_passages,
    )

    assert stats_no_tok["raw_context_tokens"] <= 10
    assert stats_no_tok["compressed_context_tokens"] <= 10

    # Test 2: With byte-exploding BPE tokenizer (e.g. SmolLM2)
    mock_tok = MockByteExplodingBPETokenizer()
    _, stats_with_tok = compress_evidence_context(
        question="Bạn là ai?",
        evidence_passages=short_passages,
        tokenizer=mock_tok,
    )

    # Even with BPE splitting accents into UTF-8 bytes, fair capping ensures <= 10 tokens
    assert stats_with_tok["raw_context_tokens"] <= 10
    assert stats_with_tok["compressed_context_tokens"] <= 10


def test_token_ratio_short_file_vs_50_line_file():
    short_text = "toi là ai?\r\n\tkiet sieu dep trai"
    short_passages = [
        {
            "passage_id": "DOC_1794280705781_b5d5c4::P000",
            "text": short_text,
        }
    ]

    # Create realistic 50-line document ("Topic Speak")
    topic_speak_lines = [
        f"Topic Speak Line {i}: Describe your favorite communication method and analyze its benefits for online education."
        for i in range(1, 51)
    ]
    topic_speak_text = "\n".join(topic_speak_lines)

    # 50 lines partitioned into realistic retrieved evidence passages
    long_passages = [
        {
            "passage_id": f"DOC_TOPIC_SPEAK::P{i:03d}",
            "text": "\n".join(topic_speak_lines[i * 10 : (i + 1) * 10]),
        }
        for i in range(5)
    ]

    _, short_stats = compress_evidence_context(
        question="Topic 49 là gì?",
        evidence_passages=short_passages,
    )

    _, long_stats = compress_evidence_context(
        question="Topic 49 là gì?",
        evidence_passages=long_passages,
    )

    short_tokens = short_stats["raw_context_tokens"]
    long_tokens = long_stats["raw_context_tokens"]

    # Short file must be strictly <= 10 tokens
    assert short_tokens <= 10

    # Ratio of 50-line file tokens vs 2-line file tokens must be at least 15x
    ratio = long_tokens / max(1, short_tokens)
    assert ratio >= 15.0, f"Expected 50-line file to be >= 15x larger than 2-line file, got {ratio:.2f}x ({long_tokens} vs {short_tokens})"


def test_count_text_tokens_excludes_passage_id_and_caps_diacritics():
    # Verify that raw text counting ignores passage metadata and doesn't explode
    plain_vi_text = "kiệt siêu đẹp trai và thông minh"
    mock_tok = MockByteExplodingBPETokenizer()

    # Raw UTF-8 bytes for this string is ~44 bytes
    assert len(plain_vi_text.encode("utf-8")) > 35

    # But count_text_tokens caps it proportionally based on word count (7 words * 1.35 ~ 9 tokens)
    fair_tokens = count_text_tokens(plain_vi_text, tokenizer=mock_tok)
    assert fair_tokens <= 10
    assert fair_tokens >= 7


def test_empty_passages_handling():
    _, stats = compress_evidence_context("câu hỏi", [])
    assert stats["raw_context_tokens"] == 0
    assert stats["compressed_context_tokens"] == 0
    assert stats["tokens_saved"] == 0
    assert stats["compression_ratio_pct"] == 0.0

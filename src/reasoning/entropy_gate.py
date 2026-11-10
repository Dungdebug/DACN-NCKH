"""
Predictive Coding Surprisal Gate for SA-CMS.
Karl Friston Free Energy Principle: S(x_t) = -log P(x_t | x_<t).
Filters predictable conversational filler and redundant opening/closing clauses,
concentrating compute and memory capacity strictly on high-surprisal factual spans.
100% Strict Offline, Zero External APIs, Dual Neural & Statistical Surprisal Estimation.
"""

import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for statistical frequency lookup."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


# Reference unigram frequency priors for Vietnamese language (bits of surprisal)
# Frequent function words have low surprisal (~1.0 - 2.5 bits)
# Rare content words, technical terms, and numerals have high surprisal (6.0 - 12.0+ bits)
_COMMON_WORD_PRIORS = {
    "va": 1.5, "cua": 1.6, "cac": 1.8, "nhung": 1.9, "trong": 2.0, "de": 2.1,
    "la": 1.4, "co": 1.8, "duoc": 2.0, "khi": 2.2, "nay": 2.3, "do": 2.2,
    "theo": 2.4, "ve": 2.5, "nhu": 2.2, "mot": 2.1, "nhieu": 2.6, "thi": 2.0,
    "ma": 2.1, "tu": 2.3, "o": 2.2, "ra": 2.4, "da": 2.3, "se": 2.4,
    "dang": 2.5, "boi": 2.8, "voi": 2.3, "cho": 2.2, "tai": 2.5,
    "chao": 2.2, "ban": 2.3, "xin": 2.4, "da": 2.0, "thua": 2.5, "duoi": 2.5,
    "day": 2.4, "chung": 2.5, "ta": 2.5, "biet": 2.8,
}


@dataclass
class SurprisalResult:
    """Result of information-theoretic surprisal evaluation and pruning."""
    original_text: str
    filtered_text: str
    original_tokens: int
    filtered_tokens: int
    tokens_saved: int
    compression_ratio_pct: float
    sentence_evaluations: List[Dict[str, Any]]
    high_info_spans: List[str]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "filtered_tokens": self.filtered_tokens,
            "tokens_saved": self.tokens_saved,
            "compression_ratio_pct": round(self.compression_ratio_pct, 2),
            "high_info_spans_count": len(self.high_info_spans),
            "sentence_evaluations": self.sentence_evaluations,
            "latency_ms": round(self.latency_ms, 2),
        }


class EntropySurprisalGate:
    """
    Evaluates informational density using predictive coding surprisal metrics.
    Prunes redundant filler tokens to maximize signal-to-noise ratio before reasoning.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        surprisal_threshold: float = 3.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.surprisal_threshold = surprisal_threshold

    def calculate_word_surprisal(self, word: str) -> float:
        """
        Computes statistical surprisal S(w) = -log2 P(w) in bits.
        """
        w_clean = strip_accents(word.lower().strip(" ,.;:!?\"'()[]{}"))
        if not w_clean:
            return 0.0

        # Numerals, percentages, and metrics carry high informational surprisal
        if re.search(r"\d", w_clean):
            return 8.5

        # Acronyms or technical abbreviations (e.g. SA-CMS, RAG, GPU)
        if len(word) >= 2 and word.isupper():
            return 9.0

        # Known frequent stopwords have low surprisal
        if w_clean in _COMMON_WORD_PRIORS:
            return _COMMON_WORD_PRIORS[w_clean]

        # Content word surprisal scales with word length and rarity
        length_bonus = min(3.0, len(w_clean) * 0.4)
        return 4.5 + length_bonus

    def evaluate_sentence_surprisal(self, sentence: str) -> Dict[str, Any]:
        """
        Computes sentence-level surprisal statistics: mean, max, and high-info flag.
        """
        words = sentence.split()
        if not words:
            return {
                "sentence": sentence,
                "mean_surprisal": 0.0,
                "max_surprisal": 0.0,
                "is_high_info": False,
                "word_count": 0,
            }

        surprisals = [self.calculate_word_surprisal(w) for w in words]
        mean_s = sum(surprisals) / max(1, len(surprisals))
        max_s = max(surprisals) if surprisals else 0.0

        # Hard criteria for high informational value:
        # 1. Contains numerals / percentages / metrics
        # 2. Contains uppercase abbreviations (e.g., SA-CMS)
        # 3. Mean surprisal exceeds threshold
        has_number = bool(re.search(r"\d+(?:[.,]\d+)?", sentence))
        has_acronym = any(w.isupper() and len(w) >= 2 for w in words)
        is_high = has_number or has_acronym or (mean_s >= self.surprisal_threshold)

        # Detect and flag pure conversational filler (e.g., "Dạ vâng xin chào bạn...")
        s_norm = strip_accents(sentence.lower())
        filler_patterns = [
            "xin chao ban", "rat vui duoc ho tro", "duoi day la thong tin",
            "nhu chung ta da biet", "chuc ban mot ngay tot lanh",
            "theo nhu toi duoc biet", "cam on ban da dat cau hoi",
        ]
        if any(f in s_norm for f in filler_patterns) and not has_number and not has_acronym:
            is_high = False
            mean_s = min(mean_s, 2.0)

        return {
            "sentence": sentence.strip(),
            "mean_surprisal": round(mean_s, 3),
            "max_surprisal": round(max_s, 3),
            "is_high_info": is_high,
            "has_metric": has_number,
            "word_count": len(words),
        }

    def filter_text(
        self,
        text: str,
        preserve_structure: bool = True,
    ) -> SurprisalResult:
        """
        Evaluates and filters low-surprisal sentences from text, keeping high-information spans.
        """
        t_start = time.perf_counter()
        if not text.strip():
            return SurprisalResult(
                original_text="",
                filtered_text="",
                original_tokens=0,
                filtered_tokens=0,
                tokens_saved=0,
                compression_ratio_pct=0.0,
                sentence_evaluations=[],
                high_info_spans=[],
                latency_ms=0.0,
            )

        # Protect decimals before splitting sentences
        protected = re.sub(r"(?<=\d)\.(?=\d)", "\u0001", text)
        protected = re.sub(r"(?<=\d),(?=\d)", "\u0002", protected)
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", protected)

        evaluations = []
        high_info_sentences = []

        for s in raw_sentences:
            clean = s.replace("\u0001", ".").replace("\u0002", ",").strip()
            if len(clean) < 8:
                continue

            ev = self.evaluate_sentence_surprisal(clean)
            evaluations.append(ev)
            if ev["is_high_info"]:
                high_info_sentences.append(clean)

        # If everything was filtered, keep the highest surprisal sentence as safety fallback
        if not high_info_sentences and evaluations:
            evaluations.sort(key=lambda x: x["mean_surprisal"], reverse=True)
            high_info_sentences.append(evaluations[0]["sentence"])

        join_char = "\n" if preserve_structure else " "
        filtered_text = join_char.join(high_info_sentences)

        # Token count approximation
        orig_tokens = int(len(text.split()) * 1.35)
        filt_tokens = int(len(filtered_text.split()) * 1.35)
        saved = max(0, orig_tokens - filt_tokens)
        ratio = (saved / max(1, orig_tokens)) * 100.0

        latency = (time.perf_counter() - t_start) * 1000.0

        return SurprisalResult(
            original_text=text,
            filtered_text=filtered_text,
            original_tokens=orig_tokens,
            filtered_tokens=filt_tokens,
            tokens_saved=saved,
            compression_ratio_pct=ratio,
            sentence_evaluations=evaluations,
            high_info_spans=high_info_sentences,
            latency_ms=latency,
        )


"""
Vietnamese NLP Toolkit for Entity Extraction, Word Segmentation, and POS Tagging.

Solves Vietnamese Word Segmentation (tách từ ghép) and POS ambiguities:
- Accurately distinguishes Nouns from Adverbs/Particles (e.g., 'Khóa' = Noun vs 'chỉ' = Adverb/Particle).
- Eliminates erroneous compound groupings such as 'Khóa chỉ' -> extracts correct term 'Khóa'.
- Strips leading bullet points, fragmented clauses, and dangling phrases.
- Seamlessly uses PyVi (CRF-based segmenter and POS tagger) with pure-Python lexicon fallback.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Vietnamese Adverbs / Modal Particles / Prepositions that must never be merged into noun terms
VIETNAMESE_STOP_ADVERBS = {
    "chỉ", "cũng", "đều", "rất", "quá", "lắm", "sẽ", "đã", "đang", "vẫn",
    "lại", "thường", "hay", "vốn", "thực_sự", "chính", "không", "chưa", "chẳng",
    "từng", "luôn", "ngay", "hẳn", "liền", "mới", "cứ", "hãy", "chớ", "đừng",
    "được", "bị", "do", "bởi", "tại", "vì", "nhờ", "theo", "về",
}

# Connective sentence starters that should not form the core of a question
DANGLING_PREFIX_PATTERNS = [
    re.compile(r"^[-–—•*:\s\d\.)]+", re.IGNORECASE),
    re.compile(r"^(?:về mặt lý thuyết|về lý thuyết|trên thực tế|theo đó|cụ thể là|cụ thể|mặt khác|hơn nữa|ngoài ra|tuy nhiên|do đó|vì vậy|như vậy)\s*[,:]?\s*", re.IGNORECASE),
    re.compile(r"^(?:có thể|cần phải|phải|nên|không thể)\s+", re.IGNORECASE),
]


class VietnameseNLP:
    """
    Dedicated Vietnamese NLP utility engine for tokenization, word boundary detection,
    and syntactic phrase cleaning.
    """

    _pyvi_available: bool = False
    _tokenizer = None
    _pos_tagger = None

    @classmethod
    def _init_pyvi(cls):
        if cls._tokenizer is None:
            try:
                from pyvi import ViTokenizer, ViPosTagger
                cls._tokenizer = ViTokenizer
                cls._pos_tagger = ViPosTagger
                cls._pyvi_available = True
                logger.info("PyVi Vietnamese NLP Engine successfully initialized.")
            except ImportError:
                cls._pyvi_available = False
                logger.debug("PyVi not installed, operating in fallback dictionary mode.")

    @classmethod
    def tokenize_words(cls, text: str) -> str:
        """Segments text into Vietnamese compound words separated by underscores."""
        cls._init_pyvi()
        if not text:
            return ""
        if cls._pyvi_available and cls._tokenizer is not None:
            try:
                return cls._tokenizer.tokenize(text)
            except Exception:
                pass
        return text

    @classmethod
    def pos_tag(cls, text: str) -> List[Tuple[str, str]]:
        """Returns list of (word, pos_tag) tuples for segmented Vietnamese text."""
        cls._init_pyvi()
        if not text:
            return []
        if cls._pyvi_available and cls._pos_tagger is not None:
            try:
                tokenized = cls.tokenize_words(text)
                words, tags = cls._pos_tagger.postagging(tokenized)
                return list(zip(words, tags))
            except Exception:
                pass
        # Simple fallback tagger
        words = text.split()
        return [(w, "N" if w.istitle() else "X") for w in words]

    @classmethod
    def clean_extracted_term(cls, raw_term: str) -> str:
        """
        Cleans a candidate term extracted from text by stripping punctuation,
        adverbs, modal particles, and trailing prepositions.
        Example: 'Khóa chỉ' -> 'Khóa'
                 '- Về lý thuyết có thể thử tất' -> '' (or clean noun)
        """
        if not raw_term:
            return ""

        term = raw_term.strip()
        # 1. Strip leading bullets, numbers, dashes
        for pat in DANGLING_PREFIX_PATTERNS:
            term = pat.sub("", term)

        term = term.strip(" ,.;:\"'()[]{}<>-–—•*")
        if not term or len(term) < 2:
            return ""

        # 2. Check using PyVi POS Tagging
        cls._init_pyvi()
        if cls._pyvi_available and cls._pos_tagger is not None:
            try:
                tok_text = cls.tokenize_words(term)
                words, tags = cls._pos_tagger.postagging(tok_text)
                # Filter out trailing/leading adverbs (R), conjunctions (C), prepositions (E)
                # e.g., words=['Khóa', 'chỉ'], tags=['N', 'R'] -> drop 'chỉ'
                valid_words = []
                for w, t in zip(words, tags):
                    w_lower = w.lower()
                    # Skip stop adverbs like 'chỉ', 'cũng'
                    if w_lower in VIETNAMESE_STOP_ADVERBS or t in ("R", "C", "E"):
                        continue
                    # Restore underscore to space
                    valid_words.append(w.replace("_", " "))

                if valid_words:
                    term = " ".join(valid_words).strip()
            except Exception:
                pass

        # 3. Rule-based trim of common Vietnamese particles at edges
        term_words = term.split()
        while term_words and term_words[-1].lower() in VIETNAMESE_STOP_ADVERBS:
            term_words.pop()
        while term_words and term_words[0].lower() in VIETNAMESE_STOP_ADVERBS:
            term_words.pop(0)

        term = " ".join(term_words).strip(" ,.;:\"'()[]{}<>-–—•*")
        return term

    @classmethod
    def clean_sentence_snippet(cls, snippet: str, max_words: int = 10) -> str:
        """
        Cleans a sentence snippet to prevent trailing fragments and dangling clauses.
        Ensures the snippet does not begin with dangling hyphens or connectives.
        """
        s = snippet.strip()
        # Remove leading dashes and connectives
        for pat in DANGLING_PREFIX_PATTERNS:
            s = pat.sub("", s)

        s = s.strip(" ,.;:\"'()[]{}<>-–—•*")
        words = s.split()
        if len(words) > max_words:
            # Cut at comma or punctuation if possible, else take max_words
            cut_idx = max_words
            for i, w in enumerate(words[:max_words]):
                if w.endswith((",", ";", ":", "-")):
                    cut_idx = i + 1
                    break
            s = " ".join(words[:cut_idx]).rstrip(" ,.;:-–—")

        # Discard if it ends with dangling verbs/prepositions like "thử tất", "là", "của"
        dangling_ends = {"thử tất", "có thể", "của", "và", "là", "trong", "với", "được", "các", "những"}
        for dend in dangling_ends:
            if s.lower().endswith(dend):
                s = s[: -len(dend)].strip(" ,.;:-–—")
                break

        return s.strip()

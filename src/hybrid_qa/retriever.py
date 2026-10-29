"""
Retriever — BM25 baseline for document passage retrieval.

Design decisions:
- BM25 via rank_bm25 (pip install rank-bm25) as stable, reproducible baseline.
- Falls back to TF-IDF if rank_bm25 is unavailable.
- Retriever is independent of SA-CMS — they are separate branches.
- Returns structured results with passage_id, score, source document info.
"""

import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional

from src.hybrid_qa.document_store import Passage


class RetrievalResult:
    """A single retrieval result with provenance."""

    __slots__ = ("passage_id", "document_id", "text", "score", "start_char",
                 "end_char", "section_title", "metadata")

    def __init__(
        self,
        passage_id: str,
        document_id: str,
        text: str,
        score: float,
        start_char: int = 0,
        end_char: int = 0,
        section_title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.passage_id = passage_id
        self.document_id = document_id
        self.text = text
        self.score = score
        self.start_char = start_char
        self.end_char = end_char
        self.section_title = section_title
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "document_id": self.document_id,
            "text": self.text,
            "score": self.score,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "section_title": self.section_title,
            "metadata": self.metadata,
        }


class BM25Retriever:
    """
    BM25 passage retriever — pure Python implementation.
    No external dependencies required.
    
    Interface:
        retrieve(query: str, top_k: int) -> List[RetrievalResult]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.passages: List[Passage] = []
        self.tokenized_corpus: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_dl: float = 0.0
        self.corpus_size: int = 0
        self._indexed = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercasing tokenizer."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]

    def build_index(self, passages: List[Passage]) -> None:
        """Build BM25 index from a list of passages."""
        self.passages = list(passages)
        self.tokenized_corpus = []
        self.doc_freqs = {}
        self.doc_lengths = []

        for passage in self.passages:
            tokens = self._tokenize(passage.text)
            self.tokenized_corpus.append(tokens)
            self.doc_lengths.append(len(tokens))

            # Count document frequency (each term counted once per doc)
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.corpus_size = len(self.passages)
        self.avg_dl = sum(self.doc_lengths) / max(1, self.corpus_size)
        self._indexed = True

    def retrieve(self, query: str, top_k: int = 5, document_id: Optional[str] = None) -> List[RetrievalResult]:
        """
        Retrieve top-k passages for a query with optional document_id filter.
        
        Returns:
            List[RetrievalResult] sorted by descending BM25 score.
        """
        if not self._indexed or not self.passages:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for idx in range(self.corpus_size):
            if document_id and self.passages[idx].document_id != document_id:
                continue
            score = self._score_document(query_tokens, idx)
            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            passage = self.passages[idx]
            results.append(RetrievalResult(
                passage_id=passage.passage_id,
                document_id=passage.document_id,
                text=passage.text,
                score=score,
                start_char=passage.start_char,
                end_char=passage.end_char,
                section_title=passage.section_title,
                metadata=passage.metadata,
            ))

        return results

    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        doc_tokens = self.tokenized_corpus[doc_idx]
        dl = self.doc_lengths[doc_idx]
        term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in self.doc_freqs:
                continue

            tf = term_freqs.get(term, 0)
            df = self.doc_freqs[term]

            # IDF with log smoothing
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def add_passages(self, new_passages: List[Passage]) -> None:
        """Incrementally add passages and rebuild index."""
        self.passages.extend(new_passages)
        self.build_index(self.passages)

    @property
    def passage_count(self) -> int:
        return len(self.passages)

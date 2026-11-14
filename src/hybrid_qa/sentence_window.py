"""
Sentence-Window Retrieval & Dynamic Context Window Expansion.

Solves the fragmented/broken context challenge of fixed chunking:
1. Sentence-Level Sensitivity: Sentences are indexed as micro-units for precise vector/lexical matching.
2. Dynamic Window Expansion: Upon retrieving a salient sentence/passage, the engine
   automatically retrieves +window_size sentences before and +window_size sentences after (default: 2 sentences).
3. Cohesive Grounding: Supplies the SLM/LLM with the complete surrounding narrative
   so answers are never truncated, disjointed, or grammatically broken.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

from src.hybrid_qa.answer_span import split_sentences
from src.hybrid_qa.document_store import DocumentRecord, Passage
from src.hybrid_qa.embedding import BiEncoderEmbedder
from src.hybrid_qa.retriever import RetrievalResult
from src.hybrid_qa.vector_search import HNSWIndex
from src.utils.vietnamese_nlp import VietnameseNLP

logger = logging.getLogger(__name__)


@dataclass
class SentenceNode:
    """A single sentence node in the document's sequential flow."""
    sentence_id: str
    document_id: str
    section_title: str
    sentence_index: int
    text: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class SentenceWindowExpander:
    """
    Expands a given passage or sentence within a larger document context
    by pulling preceding and succeeding sentences.
    """

    @classmethod
    def expand_passage_window(
        cls,
        passage_text: str,
        full_text: str,
        window_size: int = 2,
    ) -> str:
        """
        Locates the passage within the full text and expands by +window_size
        sentences before and after, respecting paragraph/section continuity.
        """
        p_clean = (passage_text or "").strip()
        f_clean = (full_text or "").strip()
        if not p_clean or not f_clean:
            return p_clean

        if p_clean == f_clean:
            return p_clean

        all_sents = split_sentences(f_clean)
        if not all_sents or len(all_sents) <= 1:
            return f_clean

        # Find matching sentence index range
        match_start = -1
        match_end = -1

        # Search for first sentence of passage
        p_sents = split_sentences(p_clean)
        if not p_sents:
            return p_clean

        first_p_sent = p_sents[0].strip()
        last_p_sent = p_sents[-1].strip()

        for idx, s in enumerate(all_sents):
            s_stripped = s.strip()
            if match_start == -1 and (first_p_sent[:35] in s_stripped or s_stripped[:35] in first_p_sent):
                match_start = idx
            if match_start != -1 and (last_p_sent[:35] in s_stripped or s_stripped[:35] in last_p_sent):
                match_end = idx

        if match_start == -1:
            # Fallback string search
            char_idx = f_clean.find(p_clean)
            if char_idx >= 0:
                prefix = f_clean[:char_idx]
                suffix = f_clean[char_idx + len(p_clean):]
                pre_sents = split_sentences(prefix)
                post_sents = split_sentences(suffix)
                window_pre = pre_sents[-window_size:] if pre_sents else []
                window_post = post_sents[:window_size] if post_sents else []
                expanded = window_pre + [p_clean] + window_post
                return " ".join(s.strip() for s in expanded if s.strip())
            return p_clean

        if match_end == -1:
            match_end = match_start + len(p_sents) - 1

        win_start = max(0, match_start - window_size)
        win_end = min(len(all_sents), match_end + window_size + 1)

        selected_sents = [s.strip() for s in all_sents[win_start:win_end] if s.strip()]
        return " ".join(selected_sents)


class SentenceWindowRetriever:
    """
    Indexes documents into individual sentences with vector embeddings,
    then upon lookup expands the winning sentences with a dynamic +/- sentence window.
    """

    def __init__(
        self,
        embedder: Optional[BiEncoderEmbedder] = None,
        window_size: int = 2,
    ):
        self.embedder = embedder or BiEncoderEmbedder()
        self.window_size = window_size
        self.vector_index = HNSWIndex(dimension=self.embedder.dimension)

        self.sentences: Dict[str, SentenceNode] = {}
        # document_id -> list of sentence_ids in sequential order
        self.doc_sentence_sequences: Dict[str, List[str]] = {}
        self._indexed = False

    def index_document(self, doc: DocumentRecord) -> int:
        """
        Indexes all sentences in the document into sequential sentence nodes.
        """
        doc_text = doc.raw_text
        doc_id = doc.document_id
        raw_sents = split_sentences(doc_text)

        seq_ids = []
        node_records: List[SentenceNode] = []
        node_texts: List[str] = []

        cursor = 0
        for s_idx, s_text in enumerate(raw_sents):
            s_clean = s_text.strip()
            if not s_clean:
                continue

            start_char = doc_text.find(s_clean, cursor)
            if start_char == -1:
                start_char = cursor
            end_char = start_char + len(s_clean)
            cursor = end_char

            # Detect section title if any
            sec_title = doc.title
            for p in doc.passages:
                if p.start_char <= start_char < p.end_char and p.section_title:
                    sec_title = p.section_title
                    break

            sid = f"{doc_id}::S{s_idx:04d}"
            node = SentenceNode(
                sentence_id=sid,
                document_id=doc_id,
                section_title=sec_title,
                sentence_index=s_idx,
                text=s_clean,
                start_char=start_char,
                end_char=end_char,
                metadata={"title": doc.title, "version": doc.version},
            )
            self.sentences[sid] = node
            seq_ids.append(sid)
            node_records.append(node)
            node_texts.append(s_clean)

        self.doc_sentence_sequences[doc_id] = seq_ids

        # Batch vector encoding into HNSW
        if node_texts:
            vectors = self.embedder.encode(node_texts)
            for idx, n in enumerate(node_records):
                self.vector_index.insert(
                    n.sentence_id,
                    vectors[idx],
                    metadata={"doc_id": doc_id, "sec": n.section_title},
                )

        self._indexed = True
        logger.info(f"Indexed {len(node_records)} sentences for doc {doc_id} with window_size={self.window_size}")
        return len(node_records)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        window_size: Optional[int] = None,
        allowed_document_ids: Optional[Collection[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Finds top sentence hits and dynamically expands each to its surrounding window.
        Overlapping windows are automatically merged.
        """
        if not self._indexed or not self.sentences:
            return []

        w_size = self.window_size if window_size is None else window_size
        allowed_set = set(allowed_document_ids) if allowed_document_ids is not None else None

        q_vec = self.embedder.encode(query)
        raw_hits = self.vector_index.search(q_vec, top_k=max(top_k * 3, 15))

        matched_spans: List[Tuple[str, int, int, float, str]] = []  # (doc_id, win_start, win_end, score, core_text)

        for sid, score in raw_hits:
            node = self.sentences.get(sid)
            if not node:
                continue
            if allowed_set is not None and node.document_id not in allowed_set:
                continue

            doc_id = node.document_id
            seq = self.doc_sentence_sequences.get(doc_id, [])
            s_idx = node.sentence_index

            win_start = max(0, s_idx - w_size)
            win_end = min(len(seq), s_idx + w_size + 1)

            matched_spans.append((doc_id, win_start, win_end, float(score), node.text))

        # Merge overlapping sentence windows within the same document
        merged_windows: List[Tuple[str, int, int, float, str]] = []
        for doc_id, w_start, w_end, sc, core_s in matched_spans:
            merged = False
            for i, (m_doc, m_start, m_end, m_sc, m_core) in enumerate(merged_windows):
                if m_doc == doc_id and not (w_end < m_start or w_start > m_end):
                    # Overlap detected: expand bounds and keep highest score
                    new_start = min(w_start, m_start)
                    new_end = max(w_end, m_end)
                    new_sc = max(sc, m_sc)
                    merged_windows[i] = (doc_id, new_start, new_end, new_sc, m_core)
                    merged = True
                    break
            if not merged:
                merged_windows.append((doc_id, w_start, w_end, sc, core_s))

        # Sort merged windows by score descending
        merged_windows.sort(key=lambda x: x[3], reverse=True)

        results: List[RetrievalResult] = []
        for idx, (doc_id, w_start, w_end, score, core_s) in enumerate(merged_windows[:top_k], start=1):
            seq = self.doc_sentence_sequences[doc_id]
            window_nodes = [self.sentences[seq[i]] for i in range(w_start, w_end)]
            window_text = " ".join(n.text for n in window_nodes)

            first_node = window_nodes[0]
            last_node = window_nodes[-1]

            meta = dict(first_node.metadata)
            meta["retrieval_strategy"] = "sentence_window"
            meta["window_size"] = w_size
            meta["window_sentence_count"] = len(window_nodes)
            meta["core_matched_sentence"] = core_s

            results.append(
                RetrievalResult(
                    passage_id=f"{doc_id}::WIN_{first_node.sentence_index:03d}_{last_node.sentence_index:03d}",
                    document_id=doc_id,
                    text=window_text,
                    score=float(score),
                    start_char=first_node.start_char,
                    end_char=last_node.end_char,
                    section_title=first_node.section_title,
                    metadata=meta,
                )
            )

        return results

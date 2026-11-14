"""
Hybrid Retriever — combines Dense Vector Search (HNSW + Cosine) and Sparse Lexical Search (BM25).

Architecture:
- Luồng 1 (Dense Vector Search): Uses BiEncoderEmbedder (all-MiniLM-L6-v2) + HNSWIndex
  to capture high-level semantic meaning and paraphrased concepts.
- Luồng 2 (Sparse Lexical Search): Uses BM25Retriever to match exact terminology,
  proper nouns, acronyms, and numbered identifiers.
- Rank Fusion: Reciprocal Rank Fusion (RRF) and Weighted Score Combination
  to produce the definitive top 3-5 passages for SLM consumption.
"""

from __future__ import annotations

import logging
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

import numpy as np

from src.hybrid_qa.document_store import Passage
from src.hybrid_qa.embedding import BiEncoderEmbedder
from src.hybrid_qa.retriever import BM25Retriever, RetrievalResult
from src.hybrid_qa.vector_search import HNSWIndex

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Unified Hybrid Retrieval Engine implementing both Vector Search (HNSW Cosine)
    and Lexical Search (BM25) with Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        embedder: Optional[BiEncoderEmbedder] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        vector_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        self.embedder = embedder or BiEncoderEmbedder()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k

        self.vector_index = HNSWIndex(dimension=self.embedder.dimension)
        self.passages_by_id: Dict[str, Passage] = {}
        self.passages: List[Passage] = []
        self._indexed = False

    def build_index(self, passages: List[Passage]) -> None:
        """Build both BM25 and HNSW vector indices over document passages."""
        self.passages = list(passages)
        self.passages_by_id = {p.passage_id: p for p in passages}

        # 1. Build BM25 index
        self.bm25_retriever.build_index(passages)

        # 2. Build HNSW Vector index
        self.vector_index.clear()
        if passages:
            texts = [p.text for p in passages]
            vectors = self.embedder.encode(texts)
            for idx, p in enumerate(passages):
                vec = vectors[idx]
                meta = {
                    "document_id": p.document_id,
                    "section_title": p.section_title,
                }
                self.vector_index.insert(p.passage_id, vec, metadata=meta)

        self._indexed = True
        logger.info(f"Built HybridRetriever index with {len(passages)} passages.")

    def add_passages(self, new_passages: List[Passage]) -> None:
        """Incrementally add passages to both indices."""
        self.passages.extend(new_passages)
        for p in new_passages:
            self.passages_by_id[p.passage_id] = p

        # Incrementally update BM25
        self.bm25_retriever.add_passages(new_passages)

        # Incrementally encode and insert into HNSW
        if new_passages:
            texts = [p.text for p in new_passages]
            vectors = self.embedder.encode(texts)
            for idx, p in enumerate(new_passages):
                vec = vectors[idx]
                meta = {
                    "document_id": p.document_id,
                    "section_title": p.section_title,
                }
                self.vector_index.insert(p.passage_id, vec, metadata=meta)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        allowed_document_ids: Optional[Collection[str]] = None,
        fusion_strategy: str = "rrf",  # "rrf" or "weighted"
    ) -> List[RetrievalResult]:
        """
        Execute Hybrid Search combining Vector Search (HNSW) and Lexical Search (BM25).
        Returns top 3-5 passages ranked by fusion score.
        """
        if not self._indexed or not self.passages:
            return []

        allowed_set: Optional[Set[str]] = None
        if allowed_document_ids is not None:
            allowed_set = set(allowed_document_ids)
        if document_id:
            allowed_set = (allowed_set & {document_id}) if allowed_set is not None else {document_id}

        # -------------------------------------------------------------
        # Luồng 1: BM25 Lexical Search (Keyword Matching)
        # -------------------------------------------------------------
        bm25_candidates = self.bm25_retriever.retrieve(
            query=query,
            top_k=max(top_k * 3, 20),
            document_id=document_id,
            allowed_document_ids=allowed_set,
        )

        # -------------------------------------------------------------
        # Luồng 2: Vector Search (HNSW Cosine Similarity)
        # -------------------------------------------------------------
        query_vector = self.embedder.encode(query)
        # Search more candidates to allow cross-filtering
        vector_candidates_raw = self.vector_index.search(
            query_vector,
            top_k=max(top_k * 3, 20),
        )

        vector_candidates: List[Tuple[str, float]] = []
        for p_id, score in vector_candidates_raw:
            p_obj = self.passages_by_id.get(p_id)
            if not p_obj:
                continue
            if allowed_set is not None and p_obj.document_id not in allowed_set:
                continue
            vector_candidates.append((p_id, score))

        # -------------------------------------------------------------
        # Fusion: Reciprocal Rank Fusion (RRF)
        # -------------------------------------------------------------
        combined_scores: Dict[str, float] = {}
        provenance_scores: Dict[str, Dict[str, float]] = {}

        # Rank maps (1-based ranking)
        bm25_ranks = {res.passage_id: rank for rank, res in enumerate(bm25_candidates, start=1)}
        vector_ranks = {p_id: rank for rank, (p_id, _) in enumerate(vector_candidates, start=1)}

        all_candidate_pids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

        if fusion_strategy == "rrf":
            # Standard RRF formula
            for pid in all_candidate_pids:
                rrf_score = 0.0
                bm25_part = 0.0
                vec_part = 0.0

                if pid in vector_ranks:
                    r_vec = vector_ranks[pid]
                    vec_part = self.vector_weight * (1.0 / (self.rrf_k + r_vec))
                    rrf_score += vec_part

                if pid in bm25_ranks:
                    r_bm25 = bm25_ranks[pid]
                    bm25_part = (1.0 - self.vector_weight) * (1.0 / (self.rrf_k + r_bm25))
                    rrf_score += bm25_part

                combined_scores[pid] = rrf_score
                provenance_scores[pid] = {
                    "rrf_score": rrf_score,
                    "vector_rank": vector_ranks.get(pid, -1),
                    "bm25_rank": bm25_ranks.get(pid, -1),
                }
        else:
            # Weighted normalized score combination
            max_bm25 = max([r.score for r in bm25_candidates], default=1.0) or 1.0
            for pid in all_candidate_pids:
                bm25_item = next((r for r in bm25_candidates if r.passage_id == pid), None)
                norm_bm25 = (bm25_item.score / max_bm25) if bm25_item else 0.0

                vec_item = next((item for item in vector_candidates if item[0] == pid), None)
                norm_vec = max(0.0, vec_item[1]) if vec_item else 0.0

                w_score = self.vector_weight * norm_vec + (1.0 - self.vector_weight) * norm_bm25
                combined_scores[pid] = w_score
                provenance_scores[pid] = {
                    "weighted_score": w_score,
                    "norm_vector": norm_vec,
                    "norm_bm25": norm_bm25,
                }

        # Sort candidate IDs by final fusion score descending
        sorted_pids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        final_results: List[RetrievalResult] = []
        for pid, final_score in sorted_pids[:top_k]:
            passage = self.passages_by_id.get(pid)
            if not passage:
                continue

            meta = dict(passage.metadata or {})
            meta["fusion"] = provenance_scores.get(pid, {})

            final_results.append(
                RetrievalResult(
                    passage_id=passage.passage_id,
                    document_id=passage.document_id,
                    text=passage.text,
                    score=float(final_score),
                    start_char=passage.start_char,
                    end_char=passage.end_char,
                    section_title=passage.section_title,
                    metadata=meta,
                )
            )

        return final_results

    @property
    def passage_count(self) -> int:
        return len(self.passages)

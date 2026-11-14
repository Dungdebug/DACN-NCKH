"""
Parent-Child Retriever (Document Structure-Aware Multi-level Retrieval).

Architecture:
- Child Chunks (Vi mô): Small, highly focused segments (1-2 sentences, ~50-80 tokens)
  optimized for precise vector embedding and dense semantic similarity lookup.
- Parent Chunks (Vĩ mô): Broad contextual blocks (entire sections or large paragraphs, ~300-500 tokens)
  providing comprehensive contextual background without blinding the LLM.
- Mechanism:
  1. Documents are ingested into a hierarchical tree of Parent and Child nodes.
  2. Query is matched against Child chunks in dense vector space for maximum sensitivity.
  3. The retriever maps winning Child chunks back to their Parent chunks.
  4. Deduplicated Parent chunks are returned to the LLM/SLM generator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

import numpy as np

from src.hybrid_qa.chunker import DocumentChunker
from src.hybrid_qa.document_store import DocumentRecord, Passage
from src.hybrid_qa.embedding import BiEncoderEmbedder
from src.hybrid_qa.retriever import RetrievalResult
from src.hybrid_qa.vector_search import HNSWIndex

logger = logging.getLogger(__name__)


@dataclass
class ChildChunk:
    """Micro-level chunk for fine-grained vector similarity search."""
    child_id: str
    parent_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    section_title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentChunk:
    """Macro-level chunk providing full surrounding context."""
    parent_id: str
    document_id: str
    title: str
    text: str
    start_char: int
    end_char: int
    section_title: str = ""
    child_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParentChildRetriever:
    """
    Hierarchical Parent-Child Retriever linking micro-search to macro-context.
    """

    def __init__(
        self,
        embedder: Optional[BiEncoderEmbedder] = None,
        parent_chunk_size: int = 400,
        parent_chunk_overlap: int = 50,
        child_chunk_size: int = 80,
        child_chunk_overlap: int = 15,
    ):
        self.embedder = embedder or BiEncoderEmbedder()
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

        self.parent_chunker = DocumentChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            strategy="sentence",
            adaptive_overlap=True,
        )
        self.child_chunker = DocumentChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            strategy="sentence",
            adaptive_overlap=False,
        )

        self.parents: Dict[str, ParentChunk] = {}
        self.children: Dict[str, ChildChunk] = {}
        self.child_to_parent: Dict[str, str] = {}
        self.child_vector_index = HNSWIndex(dimension=self.embedder.dimension)
        self._indexed = False

    def index_document(self, doc: DocumentRecord) -> Tuple[int, int]:
        """
        Decompose a document into parent chunks and child chunks, then index children into HNSW.
        Returns: (num_parents_created, num_children_created).
        """
        raw_parents = self.parent_chunker.chunk_document(doc)
        p_count = 0
        c_count = 0

        all_child_texts = []
        all_child_records: List[ChildChunk] = []

        for p_idx, p_pass in enumerate(raw_parents):
            parent_id = f"{doc.document_id}::PARENT_{p_idx:03d}"
            p_chunk = ParentChunk(
                parent_id=parent_id,
                document_id=doc.document_id,
                title=doc.title,
                text=p_pass.text,
                start_char=p_pass.start_char,
                end_char=p_pass.end_char,
                section_title=p_pass.section_title or "",
                metadata=p_pass.metadata or {},
            )
            self.parents[parent_id] = p_chunk
            p_count += 1

            # Decompose parent text into child sentences
            dummy_doc = DocumentRecord(
                document_id=parent_id,
                title=p_chunk.section_title or doc.title,
                raw_text=p_pass.text,
                version=doc.version,
            )
            raw_children = self.child_chunker.chunk_document(dummy_doc)

            for c_idx, c_pass in enumerate(raw_children):
                child_id = f"{parent_id}::CHILD_{c_idx:03d}"
                c_chunk = ChildChunk(
                    child_id=child_id,
                    parent_id=parent_id,
                    document_id=doc.document_id,
                    text=c_pass.text,
                    start_char=p_pass.start_char + c_pass.start_char,
                    end_char=p_pass.start_char + c_pass.end_char,
                    section_title=p_chunk.section_title,
                    metadata={"parent_id": parent_id, "document_id": doc.document_id},
                )
                p_chunk.child_ids.append(child_id)
                self.children[child_id] = c_chunk
                self.child_to_parent[child_id] = parent_id

                all_child_texts.append(c_pass.text)
                all_child_records.append(c_chunk)
                c_count += 1

        # Batch encode child chunks and insert into HNSW
        if all_child_texts:
            child_vectors = self.embedder.encode(all_child_texts)
            for idx, c_rec in enumerate(all_child_records):
                self.child_vector_index.insert(
                    c_rec.child_id,
                    child_vectors[idx],
                    metadata={"parent_id": c_rec.parent_id, "doc_id": c_rec.document_id},
                )

        self._indexed = True
        logger.info(f"Parent-Child Index built for doc {doc.document_id}: {p_count} parents, {c_count} children.")
        return p_count, c_count

    def retrieve(
        self,
        query: str,
        top_k_parents: int = 3,
        top_k_children: int = 10,
        allowed_document_ids: Optional[Collection[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Execute Parent-Child retrieval:
        1. Find top-k most relevant Child chunks via vector similarity.
        2. Resolve corresponding Parent chunks.
        3. Deduplicate and return parent chunks sorted by best child score.
        """
        if not self._indexed or not self.children:
            return []

        allowed_set = set(allowed_document_ids) if allowed_document_ids is not None else None

        query_vec = self.embedder.encode(query)
        raw_child_hits = self.child_vector_index.search(
            query_vec,
            top_k=max(top_k_children * 2, 20),
        )

        parent_scores: Dict[str, float] = {}
        parent_best_child: Dict[str, str] = {}

        for child_id, sim_score in raw_child_hits:
            c_obj = self.children.get(child_id)
            if not c_obj:
                continue
            if allowed_set is not None and c_obj.document_id not in allowed_set:
                continue

            parent_id = c_obj.parent_id
            if parent_id not in parent_scores or sim_score > parent_scores[parent_id]:
                parent_scores[parent_id] = sim_score
                parent_best_child[parent_id] = c_obj.text

        # Sort parents by best child similarity score descending
        sorted_parents = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        for parent_id, score in sorted_parents[:top_k_parents]:
            p_obj = self.parents.get(parent_id)
            if not p_obj:
                continue

            meta = dict(p_obj.metadata)
            meta["retrieval_mode"] = "parent_child"
            meta["matched_child_sample"] = parent_best_child.get(parent_id, "")

            results.append(
                RetrievalResult(
                    passage_id=p_obj.parent_id,
                    document_id=p_obj.document_id,
                    text=p_obj.text,
                    score=float(score),
                    start_char=p_obj.start_char,
                    end_char=p_obj.end_char,
                    section_title=p_obj.section_title,
                    metadata=meta,
                )
            )

        return results

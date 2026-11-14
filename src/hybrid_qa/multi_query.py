"""
Multi-Query Decomposition & Rewriting (Đa truy vấn & Viết lại câu hỏi).

Solves the broad/ambiguous user query challenge:
Instead of querying the vector database with a single raw query string,
this module generates 3-5 distinct query variations (reformulations and sub-queries).
It then executes retrieval across all variations and merges/deduplicates results
using Reciprocal Rank Fusion (RRF) to exhaustively retrieve ("vét cạn") relevant evidence.

Example:
    Query: "Có bao nhiêu loại mã hóa?"
    -> Multi-queries:
       1. "Có bao nhiêu loại mã hóa?" (Original)
       2. "Các loại mã hóa là gì và phân loại như thế nào?"
       3. "Phân loại mã hóa đối xứng và bất đối xứng"
       4. "Danh sách các hệ mã hóa được nhắc đến trong tài liệu"
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.hybrid_qa.retriever import RetrievalResult
from src.utils.vietnamese_nlp import VietnameseNLP

logger = logging.getLogger(__name__)


class MultiQueryExpander:
    """
    Generates 3-5 distinct sub-questions or query reformulations
    via LLM generation (when available) or deterministic semantic analysis.
    """

    @classmethod
    def generate_queries(
        cls,
        query: str,
        count: int = 3,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> List[str]:
        """
        Generates `count` (3-5) diverse queries based on the original user query.
        """
        raw_q = (query or "").strip()
        if not raw_q:
            return []

        # 1. If LLM is provided and functional, attempt LLM-based generation
        if model is not None and tokenizer is not None:
            try:
                llm_queries = cls._generate_with_llm(raw_q, model, tokenizer, count=count)
                if len(llm_queries) >= 2:
                    return llm_queries
            except Exception as e:
                logger.debug(f"LLM Multi-Query generation note: {e}")

        # 2. Robust offline/rule-based semantic decomposition & rewriting
        return cls._generate_semantic_variations(raw_q, count=count)

    @classmethod
    def _generate_with_llm(
        cls,
        query: str,
        model: Any,
        tokenizer: Any,
        count: int = 3,
    ) -> List[str]:
        """Generate diverse query reformulations using an instruction-following prompt."""
        import torch

        prompt = (
            f"<|im_start|>system\n"
            f"Bạn là một chuyên gia truy xuất thông tin (RAG). Nhiệm vụ của bạn là đọc câu hỏi của người dùng "
            f"và viết lại thành {count} câu hỏi phụ hoặc biến thể khác nhau để tìm kiếm tài liệu đầy đủ nhất.\n"
            f"Mỗi câu hỏi trên một dòng, bắt đầu bằng số thứ tự (1., 2., 3.).\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Câu hỏi gốc: \"{query}\"\n"
            f"Hãy sinh ra các câu hỏi biến thể:\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=96,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0),
            )

        gen_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        lines = [line.strip() for line in gen_text.splitlines() if line.strip()]

        cleaned_queries = [query]
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\-\)]\s*", "", line).strip(' "“’‘')
            if cleaned and len(cleaned.split()) >= 3 and cleaned.lower() not in [q.lower() for q in cleaned_queries]:
                cleaned_queries.append(cleaned)
                if len(cleaned_queries) >= count + 1:
                    break

        return cleaned_queries[: max(count, 3)]

    @classmethod
    def _generate_semantic_variations(cls, query: str, count: int = 3) -> List[str]:
        """
        Deterministic, offline query expansion that extracts intents, entities,
        and constructs 3-5 comprehensive search variations.
        """
        clean_q = query.strip()
        q_norm = clean_q.lower()
        results: List[str] = [clean_q]

        # Extract core entity / subject
        clean_term = VietnameseNLP.clean_extracted_term(clean_q)
        subject_match = re.search(
            r"(?:có bao nhiêu|các|danh sách|những|phân loại|tìm hiểu về|cho biết về|giải thích|là gì)\s+([^?.,]+)",
            clean_q,
            re.IGNORECASE,
        )
        subject = subject_match.group(1).strip() if subject_match else clean_term
        if not subject or len(subject) < 3:
            subject = clean_term or clean_q

        # Remove question suffixes
        subject = re.sub(
            r"\s+(?:là gì|như thế nào|thế nào|ở đâu|khi nào|bao nhiêu|bao gồm những gì)\s*$",
            "",
            subject,
            flags=re.IGNORECASE,
        ).strip()

        # Intent 1: "Có bao nhiêu / Phân loại / Các loại"
        if any(w in q_norm for w in ["bao nhieu", "bao nhiêu", "cac loai", "các loại", "phan loai", "phân loại"]):
            results.append(f"Các loại {subject} là gì và được phân loại như thế nào?")
            results.append(f"Danh sách các hệ {subject} và đặc điểm từng loại")
            if "ma hoa" in q_norm or "mã hóa" in q_norm or "mã hoá" in q_norm:
                results.append(f"Phân loại mã hóa đối xứng và bất đối xứng")
            else:
                results.append(f"Cấu trúc và các thành phần chính của {subject}")

        # Intent 2: "Là gì / Khái niệm / Định nghĩa"
        elif any(w in q_norm for w in ["la gi", "là gì", "dinh nghia", "định nghĩa", "khai niem", "khái niệm"]):
            results.append(f"Định nghĩa và nguyên lý hoạt động của {subject}")
            results.append(f"Cơ chế và ứng dụng thực tế của {subject}")
            results.append(f"Đặc điểm, ưu điểm và hạn chế của {subject}")

        # Intent 3: "Như thế nào / Hoạt động / Cơ chế"
        elif any(w in q_norm for w in ["nhu the nao", "như thế nào", "hoat dong", "hoạt động", "co che", "cơ chế", "nguyen ly", "nguyên lý"]):
            results.append(f"Nguyên lý hoạt động và quy trình các bước của {subject}")
            results.append(f"Kiến trúc kỹ thuật và thuật toán của {subject}")
            results.append(f"Các ví dụ minh họa và công thức của {subject}")

        # Intent 4: "So sánh / Khác nhau"
        elif any(w in q_norm for w in ["so sanh", "so sánh", "khac nhau", "khác nhau", "khac biet", "khác biệt"]):
            entities = re.split(r"\s+và\s+|\s+voi\s+|\s+với\s+", subject, flags=re.IGNORECASE)
            if len(entities) >= 2:
                results.append(f"Đặc điểm và cơ chế của {entities[0].strip()}")
                results.append(f"Đặc điểm và cơ chế của {entities[1].strip()}")
                results.append(f"Điểm khác biệt và so sánh hiệu năng giữa {entities[0].strip()} và {entities[1].strip()}")
            else:
                results.append(f"Điểm tương đồng và khác biệt của {subject}")
                results.append(f"Đánh giá ưu nhược điểm của {subject}")

        # Intent 5: General fallback expansions
        if len(results) < count:
            results.append(f"Thông tin chi tiết và cơ chế của {subject}")
        if len(results) < count:
            results.append(f"Nội dung cốt lõi và số liệu liên quan đến {subject}")
        if len(results) < count:
            results.append(f"Tổng quan và các phương pháp giải quyết {subject}")

        # Deduplicate while preserving order
        dedup_results: List[str] = []
        seen = set()
        for q in results:
            q_clean = q.strip()
            if q_clean and q_clean.lower() not in seen:
                seen.add(q_clean.lower())
                dedup_results.append(q_clean)

        return dedup_results[: max(count, 3)]


class MultiQueryRetriever:
    """
    Executes Multi-Query Retrieval across vector/lexical engines:
    1. Generates 3-5 query variants via MultiQueryExpander.
    2. Runs retrieval on each variant.
    3. Fuses and ranks all retrieved passages using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        base_retriever: Any,
        expander: Optional[MultiQueryExpander] = None,
        num_queries: int = 3,
        rrf_k: int = 60,
    ):
        self.base_retriever = base_retriever
        self.expander = expander or MultiQueryExpander()
        self.num_queries = num_queries
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieves evidence by firing multi-queries and fusing with RRF.
        """
        # 1. Expand query into 3-5 variations
        queries = self.expander.generate_queries(
            query=query,
            count=self.num_queries,
            model=model,
            tokenizer=tokenizer,
        )

        if not queries:
            queries = [query]

        # 2. Execute retrieval per query
        rrf_scores: Dict[str, float] = {}
        passage_map: Dict[str, RetrievalResult] = {}
        query_coverage: Dict[str, Set[str]] = {}

        for sub_q in queries:
            sub_results: List[RetrievalResult] = []
            try:
                if hasattr(self.base_retriever, "retrieve"):
                    # Check signature
                    import inspect
                    sig = inspect.signature(self.base_retriever.retrieve)
                    kwargs: Dict[str, Any] = {"top_k": top_k * 2}
                    if "document_id" in sig.parameters and document_id is not None:
                        kwargs["document_id"] = document_id
                    elif "allowed_document_ids" in sig.parameters and document_id is not None:
                        kwargs["allowed_document_ids"] = [document_id]
                    sub_results = self.base_retriever.retrieve(sub_q, **kwargs)
            except Exception as e:
                logger.debug(f"Multi-query sub-retrieval note for '{sub_q}': {e}")
                continue

            for rank, res in enumerate(sub_results, start=1):
                pid = res.passage_id
                passage_map[pid] = res
                query_coverage.setdefault(pid, set()).add(sub_q)

                # RRF Formula: 1 / (k + rank)
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (self.rrf_k + rank))

        # 3. Sort by aggregated RRF score
        sorted_pids = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        merged_results: List[RetrievalResult] = []
        for pid, score in sorted_pids[:top_k]:
            orig_res = passage_map[pid]
            meta = dict(getattr(orig_res, "metadata", {}) or {})
            meta["multi_query_score"] = round(float(score), 5)
            meta["multi_queries_matched"] = list(query_coverage.get(pid, []))
            meta["retrieval_strategy"] = "multi_query_rrf"

            merged_results.append(
                RetrievalResult(
                    passage_id=orig_res.passage_id,
                    document_id=orig_res.document_id,
                    text=orig_res.text,
                    score=float(score),
                    start_char=orig_res.start_char,
                    end_char=orig_res.end_char,
                    section_title=orig_res.section_title,
                    metadata=meta,
                )
            )

        return merged_results

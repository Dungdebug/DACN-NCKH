"""
HyDE (Hypothetical Document Embeddings) Query Transformation Engine.

Mechanism:
1. When a user asks a short, ambiguous, or incomplete query (e.g. "RSA là gì?"),
   standard vector search matches short query embeddings against long declarative passages poorly.
2. HyDE uses the SLM generator or linguistic assertion synthesis to generate a
   "Hypothetical Document" (a plausible textbook answer passage).
3. The embedding of this hypothetical document captures the dense declarative style
   and vocabulary of genuine source documents.
4. Searches the vector space using the hypothetical document embedding (or convex blend with query).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.hybrid_qa.embedding import BiEncoderEmbedder
from src.hybrid_qa.retriever import RetrievalResult

logger = logging.getLogger(__name__)


class HyDEEngine:
    """
    Hypothetical Document Embeddings (HyDE) Transformation Engine.
    """

    def __init__(
        self,
        embedder: Optional[BiEncoderEmbedder] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        blend_weight: float = 0.65,  # 65% hypothetical document, 35% original query
    ):
        self.embedder = embedder or BiEncoderEmbedder()
        self.model = model
        self.tokenizer = tokenizer
        self.blend_weight = blend_weight

    def _generate_synthetic_passage_template(self, query: str, language: str = "vi") -> str:
        """
        Synthesizes a declarative textbook assertion from an interrogative query
        when running fast-path or offline without model forward overhead.
        """
        q = query.strip()
        q_clean = re.sub(r"[?!.,]", "", q).strip()

        if language in ("vi", "vietnamese"):
            # Turn question into textbook definition
            if any(w in q.lower() for w in ["là gì", "la gi", "khái niệm", "định nghĩa"]):
                subject = re.sub(r"(?i)(?:cho biết|hãy giải thích|là gì|khái niệm|định nghĩa|thế nào là|về)\s*", "", q_clean).strip()
                return f"{subject} là một khái niệm, thuật toán hoặc hệ thống được nghiên cứu trong tài liệu, bao gồm các thành phần cốt lõi, đặc tính kỹ thuật và nguyên lý hoạt động chuẩn hóa."
            elif any(w in q.lower() for w in ["bao nhiêu", "mấy", "số lượng", "kích thước"]):
                return f"Thông số kỹ thuật, số lượng và kích thước đo lường cụ thể được ghi nhận chi tiết với các giá trị định lượng, tham số và đơn vị đo lường trong văn bản."
            elif any(w in q.lower() for w in ["phương pháp", "các loại", "phân loại", "gồm những gì"]):
                return f"Các phương pháp và phân loại bao gồm các hệ thống chính, quy trình triển khai từng bước và các trường hợp áp dụng tiêu biểu trong lĩnh vực chuyên môn."
            else:
                return f"{q_clean} liên quan đến các nguyên lý cấu trúc, dữ liệu thực nghiệm và giải pháp kỹ thuật được trình bày trong tài liệu."
        else:
            if "what is" in q.lower() or "define" in q.lower():
                subject = re.sub(r"(?i)(?:what is|define|explain)\s*", "", q_clean).strip()
                return f"{subject} is defined as a foundational technical system and algorithm with structured properties, core principles, and operational mechanisms."
            else:
                return f"The topic {q_clean} encompasses structural architecture, formal specifications, and documented methodologies."

    def generate_hypothetical_document(self, query: str, language: str = "vi") -> str:
        """
        Generates a hypothetical document either via Neural SLM or Declarative Synthesizer.
        """
        # If neural model available, attempt short autoregressive expansion
        if self.model is not None and self.tokenizer is not None:
            try:
                prompt = (
                    f"Viết một đoạn văn bản học thuật ngắn (2 câu) giải thích câu hỏi sau:\n"
                    f"Câu hỏi: {query}\n"
                    f"Đoạn văn giải thích:"
                ) if language in ("vi", "vietnamese") else (
                    f"Write a short 2-sentence textbook excerpt answering this question:\n"
                    f"Question: {query}\n"
                    f"Excerpt:"
                )
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=128)
                if hasattr(self.model, "device"):
                    input_ids = input_ids.to(self.model.device)

                with torch.no_grad():
                    self.model.eval()
                    # Generate up to 32 tokens
                    gen_ids = input_ids
                    for _ in range(32):
                        logits = self.model(gen_ids)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        if self.tokenizer.eos_token_id and next_tok.item() == self.tokenizer.eos_token_id:
                            break
                        gen_ids = torch.cat([gen_ids, next_tok], dim=1)

                    hypo_text = self.tokenizer.decode(gen_ids[0][input_ids.size(1):], skip_special_tokens=True).strip()
                    if len(hypo_text) >= 15:
                        return hypo_text
            except Exception as e:
                logger.debug(f"Neural HyDE generation fallback note: {e}")

        # Fallback to structural declarative template
        return self._generate_synthetic_passage_template(query, language=language)

    def encode_transformed_query(self, query: str, language: str = "vi") -> Tuple[np.ndarray, str]:
        """
        Produces a blended dense vector from both the query and its hypothetical document.
        Returns: (dense_vector, hypothetical_document_text).
        """
        hypo_doc = self.generate_hypothetical_document(query, language=language)

        vec_query = self.embedder.encode(query)
        vec_hypo = self.embedder.encode(hypo_doc)

        # Convex combination of embeddings
        blended = self.blend_weight * vec_hypo + (1.0 - self.blend_weight) * vec_query
        norm = np.linalg.norm(blended)
        if norm > 1e-9:
            blended = blended / norm

        return blended, hypo_doc

"""
Retriever — BM25 baseline for document passage retrieval.

Design decisions:
- BM25 via rank_bm25 (pip install rank-bm25) as stable, reproducible baseline.
- Falls back to TF-IDF if rank_bm25 is unavailable.
- Retriever is independent of SA-CMS — they are separate branches.
- Returns structured results with passage_id, score, source document info.
- Pure Python stdlib implementation, 100% offline.
"""

import math
import re
import unicodedata
from collections import Counter
from typing import List, Dict, Any, Optional, Collection, Tuple, Set

from src.hybrid_qa.document_store import Passage


TOPIC_ALIAS_PATTERN = re.compile(
    r"\b(?:topic|chủ đề|chu de|bài|bai|mục|muc|chương|chuong|phần|phan|section|chapter|lesson|unit)\s*(\d+)\b",
    re.IGNORECASE,
)

NUMBERED_HEADING_PIN_PATTERN = re.compile(
    r"(?:^\s*|(?<=['\"“«\(]))(\d{1,3})\s*[\.\:\-\–]\s+(?=[A-Za-zÀ-ỹ])"
    r"|(?<=[\s])(\d{1,2})\s*[\.\:\-\–]\s+([^\W\d_]{2,})"
)

LEADING_SECTION_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:topic|chủ đề|chu de|bài|bai|mục|muc|chương|chuong|phần|phan|section|chapter|lesson|unit)\s*(\d+)\b"
    r"|(\d{1,3})\s*[\.\:\-\–]\s+(?=[A-Za-zÀ-ỹ])"
    r")",
    re.IGNORECASE,
)


def extract_topic_and_section_numbers(text: str) -> Set[str]:
    """Extract explicit topic alias numbers and numbered section digits (e.g. 'Topic 1', '9. HỆ MÃ...')."""
    if not text:
        return set()
    matches = set(TOPIC_ALIAS_PATTERN.findall(text))
    for g1, g2, g2_word in NUMBERED_HEADING_PIN_PATTERN.findall(text):
        if g1:
            matches.add(g1)
        if g2 and g2_word.isupper():
            matches.add(g2)
    return matches


def extract_leading_section_number(text: str) -> Optional[str]:
    """Extract leading topic/section number if text starts with a section heading."""
    if not text:
        return None
    m = LEADING_SECTION_PATTERN.match(text)
    if not m:
        return None
    return m.group(1) or m.group(2)


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive matching."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


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

    def __init__(self, k1: float = 1.5, b: float = 0.75, proximity_weight: float = 0.0):
        self.k1 = k1
        self.b = b
        self.proximity_weight = proximity_weight
        self.passages: List[Passage] = []
        self.tokenized_corpus: List[List[str]] = []
        self._term_freqs: List[Counter] = []
        self._folded_term_freqs: List[Counter] = []
        self._doc_term_positions: List[Dict[str, List[int]]] = []
        self._folded_doc_term_positions: List[Dict[str, List[int]]] = []
        self.doc_freqs: Dict[str, int] = {}
        self._folded_df: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_dl: float = 0.0
        self.corpus_size: int = 0
        self._indexed = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercasing tokenizer preserving digits."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1 or t.isdigit()]

    def build_index(self, passages: List[Passage]) -> None:
        """Build BM25 index from a list of passages."""
        self.passages = list(passages)
        self.tokenized_corpus = []
        self._term_freqs = []
        self._folded_term_freqs = []
        self._doc_term_positions = []
        self._folded_doc_term_positions = []
        self.doc_freqs = {}
        self._folded_df = {}
        self.doc_lengths = []

        for passage in self.passages:
            tokens = self._tokenize(passage.text)
            self.tokenized_corpus.append(tokens)
            self.doc_lengths.append(len(tokens))

            # Raw term frequencies
            tf = Counter(tokens)
            self._term_freqs.append(tf)
            for term in tf.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

            # Raw & folded token positions
            doc_pos: Dict[str, List[int]] = {}
            folded_pos: Dict[str, List[int]] = {}
            folded_tokens: List[str] = []
            for pos, token in enumerate(tokens):
                doc_pos.setdefault(token, []).append(pos)
                f_token = strip_accents(token)
                folded_tokens.append(f_token)
                folded_pos.setdefault(f_token, []).append(pos)

            self._doc_term_positions.append(doc_pos)
            self._folded_doc_term_positions.append(folded_pos)

            # Folded term frequencies and doc frequencies
            folded_tf = Counter(folded_tokens)
            self._folded_term_freqs.append(folded_tf)
            for f_term in folded_tf.keys():
                self._folded_df[f_term] = self._folded_df.get(f_term, 0) + 1

        self.corpus_size = len(self.passages)
        self.avg_dl = sum(self.doc_lengths) / max(1, self.corpus_size)
        self._indexed = True

    def _compute_proximity_bonus(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        Compute proximity bonus for doc_idx.
        Finds minimal window covering >= 2 distinct query terms and calculates:
        (m - 1) / (1 + log(1 + span))
        """
        # Deduplicate query tokens preserving order
        seen_terms: Set[str] = set()
        unique_terms: List[str] = []
        for t in query_tokens:
            if t not in seen_terms:
                seen_terms.add(t)
                unique_terms.append(t)

        if len(unique_terms) < 2:
            return 0.0

        doc_positions = self._doc_term_positions[doc_idx]
        folded_positions = self._folded_doc_term_positions[doc_idx]

        occurrences: List[Tuple[int, str]] = []
        matched_term_ids: Set[str] = set()

        for term in unique_terms:
            if term in self.doc_freqs:
                term_id = term
                pos_list = doc_positions.get(term, [])
            else:
                f_term = strip_accents(term)
                if f_term in self._folded_df:
                    term_id = f_term
                    pos_list = folded_positions.get(f_term, [])
                else:
                    pos_list = []

            if pos_list:
                matched_term_ids.add(term_id)
                for p in pos_list:
                    occurrences.append((p, term_id))

        if len(matched_term_ids) < 2 or len(occurrences) < 2:
            return 0.0

        occurrences.sort(key=lambda x: x[0])

        best_bonus = 0.0
        n = len(occurrences)
        total_matched = len(matched_term_ids)

        for i in range(n):
            terms_in_window = {occurrences[i][1]}
            for j in range(i + 1, n):
                terms_in_window.add(occurrences[j][1])
                m = len(terms_in_window)
                if m >= 2:
                    span = occurrences[j][0] - occurrences[i][0]
                    bonus = (m - 1) / (1.0 + math.log(1.0 + span))
                    if bonus > best_bonus:
                        best_bonus = bonus
                if len(terms_in_window) == total_matched:
                    break

        return best_bonus

    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        dl = self.doc_lengths[doc_idx]
        term_freqs = self._term_freqs[doc_idx]
        folded_term_freqs = self._folded_term_freqs[doc_idx]

        score = 0.0
        for term in query_tokens:
            if term in self.doc_freqs:
                tf = term_freqs.get(term, 0)
                df = self.doc_freqs[term]
            else:
                folded_term = strip_accents(term)
                if folded_term in self._folded_df:
                    tf = folded_term_freqs.get(folded_term, 0)
                    df = self._folded_df[folded_term]
                else:
                    continue

            # IDF with log smoothing
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += idf * numerator / denominator

        if self.proximity_weight > 0.0:
            score += self.proximity_weight * self._compute_proximity_bonus(query_tokens, doc_idx)

        return score

    def _mmr_rerank(
        self,
        candidate_scores: List[Tuple[int, float]],
        top_k: int,
        mmr_lambda: float = 0.7,
    ) -> List[Tuple[int, float]]:
        """
        MMR rerank candidates over top candidates using Jaccard token similarity
        plus a 0.15 penalty for repeated section_title.
        """
        if not candidate_scores:
            return []

        max_score = candidate_scores[0][1]

        # Precompute token sets and normalized section titles
        doc_token_sets = {idx: set(self.tokenized_corpus[idx]) for idx, _ in candidate_scores}
        doc_sections = {idx: (self.passages[idx].section_title or "").strip().lower() for idx, _ in candidate_scores}

        selected: List[Tuple[int, float]] = [candidate_scores[0]]
        remaining = list(candidate_scores[1:])

        while remaining and len(selected) < top_k:
            best_mmr = -float("inf")
            best_cand = None

            for cand in remaining:
                cand_idx, cand_score = cand
                cand_tokens = doc_token_sets[cand_idx]
                cand_sec = doc_sections[cand_idx]

                rel = (cand_score / max_score) if max_score > 0 else 0.0

                max_redundancy = 0.0
                for sel_idx, _ in selected:
                    sel_tokens = doc_token_sets[sel_idx]
                    sel_sec = doc_sections[sel_idx]

                    union_size = len(cand_tokens | sel_tokens)
                    jaccard = (len(cand_tokens & sel_tokens) / union_size) if union_size > 0 else 0.0

                    section_penalty = 0.15 if (cand_sec and sel_sec and cand_sec == sel_sec) else 0.0
                    redundancy = jaccard + section_penalty
                    if redundancy > max_redundancy:
                        max_redundancy = redundancy

                mmr = mmr_lambda * rel - (1.0 - mmr_lambda) * max_redundancy
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_cand = cand

            if best_cand is not None:
                selected.append(best_cand)
                remaining.remove(best_cand)
            else:
                break

        return selected

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        allowed_document_ids: Optional[Collection[str]] = None,
        diversify: bool = False,
        mmr_lambda: float = 0.7,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k passages for a query with optional document_id filter, allowed_document_ids set,
        and optional MMR diversification.
        
        Returns:
            List[RetrievalResult] sorted by descending score.
        """
        if not self._indexed or not self.passages:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        allowed_set: Optional[Set[str]] = None
        if allowed_document_ids is not None:
            allowed_set = set(allowed_document_ids)
        if document_id:
            allowed_set = (allowed_set & {document_id}) if allowed_set is not None else {document_id}
        if allowed_set is not None and not allowed_set:
            return []

        # Check topic / numbered-section pin
        q_topic_matches = extract_topic_and_section_numbers(query)

        # Check quoted phrases in query (e.g. 'PHÁ MÃ' or "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG")
        quoted_spans = [
            m.strip().lower()
            for m in re.findall(r"['\"“«]([^'\"”»]+)['\"”»]", query)
            if m.strip()
        ]

        q_lower = query.lower()
        is_concept_and_types = (
            ("là gì" in q_lower or "khái niệm" in q_lower or "định nghĩa" in q_lower)
            and ("bao nhiêu" in q_lower or "các loại" in q_lower or "phương pháp" in q_lower or "phân loại" in q_lower or "các hệ" in q_lower or "ví dụ" in q_lower)
        ) or (
            ("là gì" in q_lower or "khái niệm" in q_lower)
            and ("phương pháp" in q_lower or "loại" in q_lower)
        )

        target_passages_exist = False
        if q_topic_matches:
            for idx in range(self.corpus_size):
                if allowed_set is not None and self.passages[idx].document_id not in allowed_set:
                    continue
                p = self.passages[idx]
                sec_topics = extract_topic_and_section_numbers(p.section_title or "")
                text_topics = set(TOPIC_ALIAS_PATTERN.findall((p.text or "").lower()))
                head_sec_num = extract_leading_section_number(p.text or "")
                if head_sec_num:
                    text_topics.add(head_sec_num)
                if (sec_topics & q_topic_matches) or (text_topics & q_topic_matches):
                    target_passages_exist = True
                    break

        scores = []
        for idx in range(self.corpus_size):
            if allowed_set is not None and self.passages[idx].document_id not in allowed_set:
                continue

            p = self.passages[idx]
            sec_topics = extract_topic_and_section_numbers(p.section_title or "")
            text_topics = set(TOPIC_ALIAS_PATTERN.findall((p.text or "").lower()))
            head_sec_num = extract_leading_section_number(p.text or "")
            if head_sec_num:
                text_topics.add(head_sec_num)

            # If >= 1 passage matches target topic N, exclude passages declaring a different number
            if target_passages_exist:
                if sec_topics and not (sec_topics & q_topic_matches):
                    continue
                if head_sec_num and head_sec_num not in q_topic_matches:
                    continue

            score = self._score_document(query_tokens, idx)

            # Match on section_title first (+30.0), then text (+15.0)
            if q_topic_matches:
                if sec_topics & q_topic_matches:
                    score += 30.0
                elif text_topics & q_topic_matches:
                    score += 15.0

            # Direct match bonus (+5.0) for quoted entity or section title in query
            sec_lower = (p.section_title or "").lower()
            first_line_lower = (p.text.split("\n", 1)[0] if p.text else "").lower()

            if quoted_spans:
                if any(qs in sec_lower or qs in first_line_lower for qs in quoted_spans):
                    score += 5.0
            elif sec_lower:
                clean_sec = re.sub(r"^\s*\d+[\.:\s\-–]+", "", sec_lower).strip(" :.-–")
                if clean_sec and len(clean_sec) >= 3 and clean_sec in q_lower:
                    score += 5.0
                elif sec_lower and sec_lower in q_lower:
                    score += 5.0

            # General concept + classification bonus (+3.0)
            if is_concept_and_types:
                if any(kw in sec_lower for kw in ["giới thiệu", "khái niệm", "định nghĩa", "phân loại", "1."]):
                    score += 3.0
                p_text_lower = (p.text or "").lower()
                if any(kw in p_text_lower for kw in ["phương pháp phổ biến", "bao gồm", "có 2 phương pháp", "có hai phương pháp"]):
                    score += 3.0

            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        if not diversify or len(scores) <= 1 or top_k <= 1:
            selected_candidates = scores[:top_k]
        else:
            selected_candidates = self._mmr_rerank(
                candidate_scores=scores[: min(len(scores), 3 * top_k)],
                top_k=top_k,
                mmr_lambda=mmr_lambda,
            )

        results = []
        for idx, score in selected_candidates:
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

    def add_passages(self, new_passages: List[Passage]) -> None:
        """Incrementally add passages and rebuild index."""
        self.passages.extend(new_passages)
        self.build_index(self.passages)

    @property
    def passage_count(self) -> int:
        return len(self.passages)

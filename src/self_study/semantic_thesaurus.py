"""
Domain-Specific Semantic Thesaurus & Soft Lexical Matcher for Scientific Document QA.
Bridges the lexical gap between colloquial or varied query formulations
and authoritative technical document terminology without external LLMs.
100% Offline, Pure Python, Highly Optimized for Vietnamese & Academic Contexts.
"""

import re
import unicodedata
from typing import List, Dict, Set, Tuple, Optional


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive canonical alignment."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


# Pre-compiled high-precision synonym clusters for academic & technical document QA
_SYNONYM_CLUSTERS: Dict[str, List[str]] = {
    "advantage": [
        "ưu điểm", "ưu thế", "lợi ích", "thế mạnh", "điểm mạnh",
        "giúp ích", "tác dụng", "hiệu quả", "tiết kiệm", "tối ưu",
        "tối ưu hóa", "cắt giảm", "vượt trội", "tốt hơn", "cải thiện",
        "đạt", "đạt được", "giảm", "nâng cao", "rút ngắn", "lợi thế"
    ],
    "limitation": [
        "hạn chế", "nhược điểm", "điểm yếu", "khó khăn", "thách thức",
        "rủi ro", "đánh đổi", "chưa giải quyết", "yếu điểm", "trở ngại",
        "cảnh báo", "ràng buộc", "giới hạn", "nhược điểm chính", "khuyết điểm"
    ],
    "mechanism": [
        "cơ chế", "nguyên lý", "hoạt động", "cách thức", "làm thế nào",
        "bằng cách nào", "quy trình", "thuật toán", "kiến trúc", "phương pháp",
        "vận hành", "tích hợp", "cập nhật", "nơ-ron", "synapse", "kết nối",
        "bộ nhớ", "động cơ", "mô hình", "hệ thống"
    ],
    "cause_motive": [
        "tại sao", "vì sao", "nguyên nhân", "lý do", "động lực",
        "mục tiêu", "mục đích", "bối cảnh", "nguồn gốc", "xuất phát từ",
        "động cơ", "vấn đề tồn đọng", "thực trạng"
    ],
    "metric": [
        "kết quả", "số liệu", "thực nghiệm", "đo đạc", "bao nhiêu",
        "tỷ lệ", "phần trăm", "độ trễ", "chi phí", "token", "thời gian",
        "bộ nhớ", "vram", "ms", "f1", "chính xác", "tốc độ", "bảng"
    ],
    "definition": [
        "là gì", "khái niệm", "định nghĩa", "nghĩa là", "được hiểu là",
        "thế nào là", "bản chất", "tổng quan", "nội dung chính", "giới thiệu"
    ],
    "structure": [
        "cấu trúc", "phân tầng", "tầng", "mức", "thành phần",
        "bao gồm", "đoạn", "mục", "phần", "tài liệu", "chương",
        "phân đoạn", "cấp độ", "đa thang"
    ]
}

# Stopwords for filtering out lexical noise
_VI_STOPWORDS: Set[str] = {
    strip_accents(w)
    for w in (
        "và", "của", "các", "những", "trong", "để", "với", "cho", "tại", "là",
        "có", "được", "khi", "này", "đó", "theo", "về", "như", "một", "nhiều",
        "thì", "mà", "từ", "ở", "ra", "đã", "sẽ", "đang", "bởi", "do", "bằng",
        "gì", "nào", "sao", "thế", "đâu", "mấy", "ai", "bao", "nhiêu", "vậy",
        "cũng", "rất", "nữa", "hãy", "xin", "ơi", "ạ", "nhé", "thôi", "vào",
        "the", "is", "are", "of", "and", "in", "to", "for", "with", "on", "a",
        "an", "what", "which", "how", "when", "where", "why",
    )
}


class SemanticThesaurus:
    """
    Authoritative domain-specific semantic thesaurus and soft matcher.
    Expands conversational user intents into technical document cues.
    """

    def __init__(self, clusters: Optional[Dict[str, List[str]]] = None):
        self.clusters = clusters or _SYNONYM_CLUSTERS
        # Build inverted index for fast keyword/phrase lookup
        self._inverted_index: Dict[str, str] = {}
        for cluster_name, terms in self.clusters.items():
            for t in terms:
                norm_t = strip_accents(t.lower()).strip()
                self._inverted_index[norm_t] = cluster_name

    def get_synonyms_for_phrase(self, phrase: str) -> List[str]:
        """Finds all synonyms belonging to the same semantic cluster as the input phrase."""
        norm_p = strip_accents(phrase.lower()).strip()
        if not norm_p:
            return []

        # 1. Direct inverted index match
        cluster_name = self._inverted_index.get(norm_p)

        # 2. Substring match: check if a known cluster term is contained in the phrase
        if not cluster_name:
            for term, c_name in self._inverted_index.items():
                if term in norm_p:
                    cluster_name = c_name
                    break

        # 3. Controlled prefix/containment only for long phrases (len >= 5)
        if not cluster_name and len(norm_p) >= 5:
            for term, c_name in self._inverted_index.items():
                if norm_p in term:
                    cluster_name = c_name
                    break

        if cluster_name and cluster_name in self.clusters:
            return [t for t in self.clusters[cluster_name] if strip_accents(t.lower()) != norm_p]
        return []

    def get_intent_cues(self, intent: str) -> List[str]:
        """Returns semantic cues associated with a standard intent category."""
        intent_map = {
            "ADVANTAGE": "advantage",
            "LIMITATION": "limitation",
            "MECHANISM": "mechanism",
            "CAUSE": "cause_motive",
            "METRIC": "metric",
            "DEFINITION": "definition",
            "ENUMERATION": "structure",
            "SUMMARY": "definition",
        }
        cluster_key = intent_map.get(intent.upper(), "advantage")
        return self.clusters.get(cluster_key, [])

    def expand_query_terms(self, query: str, max_expansions: int = 4) -> List[str]:
        """
        Extracts salient query terms and enriches them with high-value technical synonyms.
        Avoids token explosion while ensuring lexical gap coverage.
        """
        words = re.findall(r"\w+", query)
        salient_terms: List[str] = []
        q_norm = strip_accents(query.lower())

        # 1. Preserve technical acronyms and hyphenated identifiers (e.g., SA-CMS, GPU, RAG)
        for w in words:
            if (w.isupper() and len(w) >= 2) or "-" in w or (len(w) > 3 and any(c.isupper() for c in w[1:])):
                if w not in salient_terms:
                    salient_terms.append(w)

        # 2. Extract domain phrases and intent cues present in the query (longer phrases first)
        sorted_terms = sorted(self._inverted_index.items(), key=lambda x: len(x[0]), reverse=True)
        for term, cluster_name in sorted_terms:
            # Match whole phrase in normalized query
            pattern = r"(?:^|\s)" + re.escape(term) + r"(?:$|\s)"
            if re.search(pattern, q_norm) or term in q_norm:
                orig_term = next((t for t in self.clusters[cluster_name] if strip_accents(t.lower()) == term), term)
                if orig_term not in salient_terms and strip_accents(orig_term.lower()) not in [strip_accents(t.lower()) for t in salient_terms]:
                    salient_terms.append(orig_term)

        # 3. Add synonym expansions prioritizing intent phrases over generic nouns
        expansions: List[str] = list(salient_terms)
        added_count = 0

        # Prioritize key clusters (advantage, limitation, cause_motive, metric) first
        priority_clusters = {"advantage", "limitation", "cause_motive", "metric"}
        salient_prioritized = sorted(
            salient_terms,
            key=lambda t: 0 if self._inverted_index.get(strip_accents(t.lower())) in priority_clusters else 1
        )

        for term in salient_prioritized:
            syns = self.get_synonyms_for_phrase(term)
            for s in syns:
                if s not in expansions:
                    expansions.append(s)
                    added_count += 1
                    if added_count >= max_expansions:
                        break
            if added_count >= max_expansions:
                break

        return expansions

    def compute_soft_match(
        self,
        query: str,
        sentence: str,
        intent: str = "GENERIC",
    ) -> Tuple[float, List[str]]:
        """
        Computes soft lexical overlap between query and candidate sentence,
        awarding points for direct matches as well as semantic synonym activations.
        Returns:
            (soft_score, matched_terms)
        """
        q_norm = strip_accents(query.lower())
        s_norm = strip_accents(sentence.lower())
        s_words = set(re.findall(r"\w+", s_norm))

        query_words = [
            w for w in re.findall(r"\w+", q_norm)
            if w not in _VI_STOPWORDS and len(w) > 1
        ]
        if not query_words:
            return 0.0, []

        direct_matches: List[str] = []
        synonym_matches: List[str] = []

        # 1. Check direct word matches
        for qw in query_words:
            if qw in s_words:
                direct_matches.append(qw)

        # 2. Check technical acronyms (e.g. SA-CMS)
        for ent in re.findall(r"[A-Za-z0-9]+-[A-Za-z0-9]+|[A-Z]{2,}", query):
            if ent.lower() in sentence.lower():
                if ent not in direct_matches:
                    direct_matches.append(ent)

        # 3. Check synonym expansions for query phrases
        for term, cluster_name in self._inverted_index.items():
            if term in q_norm:
                # Check if any synonym in this cluster appears in the sentence
                cluster_terms = self.clusters[cluster_name]
                for syn in cluster_terms:
                    syn_norm = strip_accents(syn.lower())
                    if syn_norm in s_norm:
                        if syn not in synonym_matches:
                            synonym_matches.append(syn)
                        break

        # 4. Check intent cues
        if intent != "GENERIC":
            intent_cues = self.get_intent_cues(intent)
            for cue in intent_cues:
                cue_norm = strip_accents(cue.lower())
                if cue_norm in s_norm:
                    if cue not in synonym_matches:
                        synonym_matches.append(cue)
                    break

        all_matched = list(set(direct_matches + synonym_matches))

        # Scoring formula: 70% direct coverage + 30% synonym reinforcement
        direct_coverage = len(direct_matches) / max(1, len(query_words))
        synonym_bonus = min(0.4, len(synonym_matches) * 0.15)

        soft_score = min(1.0, (direct_coverage * 0.7) + synonym_bonus)
        return round(soft_score, 4), all_matched

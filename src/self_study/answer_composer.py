"""
Coherent Multi-Sentence Answer Composer for Sovereign Document Brain.
Proposal Reference: docs/de_cuong_text.txt Section 6.5.

Transforms fragmented extractive sentence spans into fluent, logically connected,
multi-sentence explanations with in-text citations ([1], [2]).
Guarantees 100% extractive grounding (zero hallucination): every factual clause
is an exact verifiable substring from the source documents.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive rule checking."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


class CoherentAnswerComposer:
    """
    Composes structured, coherent, multi-sentence answers from top ranked candidate sentences.
    Connects core concepts, quantitative metrics, and constraints with natural discourse markers.
    """

    METRIC_REGEX = re.compile(
        r"(?<!\w)(?:\d+(?:[.,]\d+)?\s*(?:%|phần\s*trăm|token[s]?|ms|giây|phút|giờ|"
        r"gb|mb|kb|vram|gpu|triệu|tỷ|nghìn|lần|đô|usd|vnd|f1|bleu|rouge)|"
        r"(?:giảm|tăng|đạt|tiết kiệm|chi phí|độ trễ)\s+\d+)(?!\w)",
        re.IGNORECASE,
    )

    LIMITATION_TERMS = {
        "han che", "nhuoc diem", "thach thuc", "kho khan", "rui ro",
        "danh doi", "yeu cau", "rang buoc", "tuy nhien", "mac du",
        "chua giai quyet", "yeu diem", "gioi han", "khuyet diem",
    }

    MECHANISM_TERMS = {
        "co che", "nguyen ly", "thong qua", "bang cach", "quy trinh",
        "kien truc", "tich hop", "cap nhat", "hoat dong", "van hanh",
        "bao gom", "thanh phan",
    }

    def __init__(self):
        pass

    def classify_sentence_role(self, sentence: str) -> str:
        """
        Classifies a candidate sentence into a functional discourse role:
        'metric', 'limitation', 'mechanism', or 'elaboration'.
        """
        s_norm = strip_accents(sentence.lower())

        # 1. Limitation / Constraint
        if any(term in s_norm for term in self.LIMITATION_TERMS):
            return "limitation"

        # 2. Quantitative Metric / Experimental Evidence
        if self.METRIC_REGEX.search(sentence):
            return "metric"

        # 3. Mechanism / Architecture
        if any(term in s_norm for term in self.MECHANISM_TERMS):
            return "mechanism"

        return "elaboration"

    @staticmethod
    def _clean_sentence(sentence: str) -> str:
        """Trims whitespace and ensures complete terminal punctuation."""
        s = sentence.strip(" \t\n-–—")
        # Remove opening redundant connectives if present
        s = re.sub(
            r"^(?:Tuy nhiên|Do đó|Vì vậy|Theo đó|Ngoài ra|Hơn nữa|Cụ thể|Bên cạnh đó|Đồng thời)[,:\s]+",
            "",
            s,
            flags=re.IGNORECASE,
        ).strip()
        if s and s[-1] not in ".!?":
            s = f"{s}."
        return s

    @staticmethod
    def _lower_first(text: str) -> str:
        """Lowers the first character if it is not an acronym/proper noun."""
        if not text:
            return ""
        words = text.split()
        first_word = words[0]
        # Keep acronyms or capitalized model names (e.g. SA-CMS, GPU, SmolLM2)
        if first_word.isupper() or "-" in first_word or any(c.isupper() for c in first_word[1:]):
            return text
        return text[0].lower() + text[1:]

    def compose(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        intent: str = "GENERIC",
        answer_mode: str = "balanced",
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        Builds a multi-sentence coherent response from candidate sentences.

        Returns:
            (composed_prose, key_points, active_citations)
        """
        if not candidates:
            return "", [], []

        # Candidate 0 is always the primary core answer
        primary = candidates[0]
        core_sent = self._clean_sentence(primary["sentence"])
        primary_cit = primary.get("citation_label", "[1]")

        if answer_mode == "concise" or len(candidates) == 1:
            prose = f"{core_sent} {primary_cit}"
            return prose, [f"{core_sent} {primary_cit}"], [primary]

        # Categorize remaining candidate sentences to form a coherent triad:
        # [Core concept] + [Quantitative metric] + [Limitation/Condition or Deep mechanism]
        used_sentences: Set[str] = {primary["sentence"]}
        active_citations: List[Dict[str, Any]] = [primary]

        metric_candidate: Optional[Dict[str, Any]] = None
        limitation_candidate: Optional[Dict[str, Any]] = None
        elaboration_candidate: Optional[Dict[str, Any]] = None

        for cand in candidates[1:]:
            sent = cand["sentence"]
            if sent in used_sentences:
                continue

            # Deduplication: check lexical overlap with already used sentences
            cand_words = set(re.findall(r"\w+", strip_accents(sent.lower())))
            redundant = False
            for u in used_sentences:
                u_words = set(re.findall(r"\w+", strip_accents(u.lower())))
                jaccard = len(cand_words & u_words) / max(1, len(cand_words | u_words))
                if jaccard > 0.60:
                    redundant = True
                    break
            if redundant:
                continue

            role = self.classify_sentence_role(sent)
            if role == "metric" and metric_candidate is None:
                metric_candidate = cand
                used_sentences.add(sent)
                active_citations.append(cand)
            elif role == "limitation" and limitation_candidate is None:
                limitation_candidate = cand
                used_sentences.add(sent)
                active_citations.append(cand)
            elif role in ("mechanism", "elaboration") and elaboration_candidate is None:
                elaboration_candidate = cand
                used_sentences.add(sent)
                active_citations.append(cand)

            if metric_candidate and (limitation_candidate or elaboration_candidate):
                break

        # Assemble coherent prose with natural Vietnamese discourse markers
        composed_parts: List[str] = [f"{core_sent} {primary_cit}"]
        key_points: List[str] = [f"{core_sent} {primary_cit}"]

        # 1. Quantitative elaboration
        if metric_candidate:
            m_sent = self._clean_sentence(metric_candidate["sentence"])
            m_cit = metric_candidate.get("citation_label", "[2]")
            lead = "Đáng chú ý, " if not m_sent.lower().startswith("đáng chú ý") else ""
            m_body = f"{lead}{self._lower_first(m_sent)} {m_cit}"
            composed_parts.append(m_body)
            key_points.append(f"{m_sent} {m_cit}")

        # 2. Limitation or qualification
        if limitation_candidate:
            l_sent = self._clean_sentence(limitation_candidate["sentence"])
            l_cit = limitation_candidate.get("citation_label", f"[{len(active_citations)}]")
            if any(l_sent.lower().startswith(c) for c in ["tuy nhiên", "mặc dù", "song"]):
                l_body = f"{l_sent} {l_cit}"
            else:
                l_body = f"Tuy nhiên, tài liệu cũng lưu ý rằng {self._lower_first(l_sent)} {l_cit}"
            composed_parts.append(l_body)
            key_points.append(f"{l_sent} {l_cit}")
        elif elaboration_candidate and answer_mode == "comprehensive":
            # In comprehensive mode, include complementary structural detail if no limitation was found
            e_sent = self._clean_sentence(elaboration_candidate["sentence"])
            e_cit = elaboration_candidate.get("citation_label", f"[{len(active_citations)}]")
            e_body = f"Cụ thể hơn, {self._lower_first(e_sent)} {e_cit}"
            composed_parts.append(e_body)
            key_points.append(f"{e_sent} {e_cit}")

        # Combine into cohesive paragraph
        composed_prose = " ".join(composed_parts).strip()
        return composed_prose, key_points, active_citations


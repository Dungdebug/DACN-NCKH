"""
Hierarchical Section Drawer for Sovereign Document Brain.
Proposal Reference: docs/de_cuong_text.txt Section 6.4 & 6.5.

Partitions and indexes document sections into 4 canonical functional drawers:
1. DRAWER_PROBLEM: Background, motivation, problem statement, related work.
2. DRAWER_SOLUTION: Architecture, methodology, algorithms, continuum memory mechanisms.
3. DRAWER_EXPERIMENT: Quantitative evaluations, token reduction benchmarks, latency metrics.
4. DRAWER_CONCLUSION: Limitations, trade-offs, discussion, future work.

Enables instant targeted retrieval (< 1ms) and structured multi-drawer synthesis.
100% Offline, Deterministic, Zero LLM overhead.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Set


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive canonical alignment."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


DRAWER_PROBLEM = "DRAWER_PROBLEM"
DRAWER_SOLUTION = "DRAWER_SOLUTION"
DRAWER_EXPERIMENT = "DRAWER_EXPERIMENT"
DRAWER_CONCLUSION = "DRAWER_CONCLUSION"
DRAWER_GENERIC = "DRAWER_GENERIC"


class SectionDrawer:
    """
    Categorizes and indexes document sections into functional drawers.
    Provides drawer-aware evidence re-ranking and intent routing.
    """

    DRAWER_KEYWORDS: Dict[str, Set[str]] = {
        DRAWER_PROBLEM: {
            "gioi thieu", "tong quan", "boi canh", "thuc trang", "van de",
            "dong luc", "muc tieu", "nghien cuu lien quan", "dat van de",
            "ly do", "introduction", "background", "motivation", "related work",
            "problem statement", "overview"
        },
        DRAWER_SOLUTION: {
            "phuong phap", "kien truc", "co che", "thiet ke", "mo hinh",
            "thuat toan", "cau truc", "trien khai", "nguyen ly", "phan tang",
            "bo nho", "sa-cms", "giao tiep", "lien cap", "methodology",
            "architecture", "design", "model", "algorithm", "implementation",
            "mechanism", "framework"
        },
        DRAWER_EXPERIMENT: {
            "thuc nghiem", "ket qua", "danh gia", "so lieu", "hieu nang",
            "do dac", "bang", "bieu do", "chi phi token", "do tre", "rtx",
            "evaluation", "experiments", "results", "benchmark", "metrics",
            "performance", "latency"
        },
        DRAWER_CONCLUSION: {
            "ket luan", "han che", "thao luan", "huong phat trien", "danh doi",
            "khuyen nghi", "tong ket", "tuong lai", "conclusion", "limitations",
            "discussion", "future work", "trade-off"
        },
    }

    INTENT_TO_DRAWER: Dict[str, List[str]] = {
        "SUMMARY": [DRAWER_PROBLEM, DRAWER_SOLUTION, DRAWER_CONCLUSION],
        "CAUSE": [DRAWER_PROBLEM, DRAWER_CONCLUSION],
        "DEFINITION": [DRAWER_SOLUTION, DRAWER_PROBLEM],
        "MECHANISM": [DRAWER_SOLUTION],
        "ENUMERATION": [DRAWER_SOLUTION],
        "METRIC": [DRAWER_EXPERIMENT],
        "COMPARISON": [DRAWER_EXPERIMENT, DRAWER_SOLUTION],
        "LIMITATION": [DRAWER_CONCLUSION, DRAWER_EXPERIMENT],
        "ADVANTAGE": [DRAWER_EXPERIMENT, DRAWER_SOLUTION],
    }

    def __init__(self):
        pass

    def categorize_section(self, section_title: str, text_sample: str = "") -> str:
        """
        Determines the functional drawer of a section by analyzing its title
        and falling back to representative text sample tokens if title is ambiguous.
        """
        title_norm = strip_accents(section_title.lower()).strip()

        # Direct title keyword match
        for drawer, keywords in self.DRAWER_KEYWORDS.items():
            for kw in keywords:
                # Match whole words or boundary phrases
                if re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", title_norm):
                    return drawer

        # Fallback: check text sample if title is generic (e.g. "Chương 2", "Phần 3")
        if text_sample:
            text_norm = strip_accents(text_sample[:400].lower())
            scores: Dict[str, int] = {d: 0 for d in self.DRAWER_KEYWORDS}
            for drawer, keywords in self.DRAWER_KEYWORDS.items():
                for kw in keywords:
                    if kw in text_norm:
                        scores[drawer] += 1
            best_drawer = max(scores, key=scores.get)
            if scores[best_drawer] > 0:
                return best_drawer

        return DRAWER_GENERIC

    def index_document_drawers(self, passages: List[Any]) -> Dict[str, List[Any]]:
        """
        Groups document passages into the 4 functional drawers.
        """
        drawers: Dict[str, List[Any]] = {
            DRAWER_PROBLEM: [],
            DRAWER_SOLUTION: [],
            DRAWER_EXPERIMENT: [],
            DRAWER_CONCLUSION: [],
            DRAWER_GENERIC: [],
        }

        for p in passages:
            sec_title = getattr(p, "section_title", "")
            if not sec_title and isinstance(p, dict):
                sec_title = p.get("section_title", "")

            sample_text = getattr(p, "text", "")
            if not sample_text and isinstance(p, dict):
                sample_text = p.get("text", "")

            drawer = self.categorize_section(sec_title, text_sample=sample_text)
            drawers[drawer].append(p)

        return drawers

    def filter_and_rank_by_intent(
        self,
        passages: List[Any],
        intent: str,
        max_passages: int = 5,
    ) -> List[Any]:
        """
        Re-ranks and filters passages so that passages originating from the
        semantically aligned drawer appear first.
        """
        target_drawers = self.INTENT_TO_DRAWER.get(intent.upper(), [DRAWER_GENERIC])
        drawer_groups = self.index_document_drawers(passages)

        aligned_passages: List[Any] = []
        for drawer in target_drawers:
            aligned_passages.extend(drawer_groups.get(drawer, []))

        # Other passages act as secondary context
        other_passages: List[Any] = []
        for drawer, p_list in drawer_groups.items():
            if drawer not in target_drawers:
                other_passages.extend(p_list)

        combined = aligned_passages + other_passages
        return combined[:max_passages]


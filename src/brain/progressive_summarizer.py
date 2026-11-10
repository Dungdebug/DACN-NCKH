"""
Multi-Scale Progressive Digest (Ticket 7.6).
Generates multi-level progressive document summaries:
- Level 1 (L1): Micro TL;DR (<= 35 words).
- Level 2 (L2): Executive Abstract (3-5 structured sentences drawing from canonical drawers).
- Level 3 (L3): Structural Section-by-Section Digest.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
import re
from src.brain.section_drawer import (
    SectionDrawer,
    DRAWER_PROBLEM,
    DRAWER_SOLUTION,
    DRAWER_EXPERIMENT,
    DRAWER_CONCLUSION,
    DRAWER_GENERIC,
)


@dataclass
class ProgressiveDigest:
    """Consolidated 3-tier digest."""
    doc_title: str
    l1_tldr: str
    l2_executive: str
    l3_section_digest: List[Dict[str, str]] = field(default_factory=list)


class ProgressiveSummarizer:
    """
    Summarizes documents across three progressive zoom levels.
    """

    def __init__(self):
        self.drawer = SectionDrawer()

    def split_clean_sentences(self, text: str) -> List[str]:
        """Preserves numbers and decimals while splitting sentences."""
        masked = re.sub(r"(\d+)[.,](\d+)", r"\1__DEC__\2", text)
        masked = re.sub(r"\b([A-Z])\.", r"\1__DOT__", masked)
        parts = re.split(r"[.!?]+(?:\s+|\s*$)", masked)
        cleaned = []
        for p in parts:
            p_restored = p.replace("__DEC__", ",").replace("__DOT__", ".").strip()
            if len(p_restored) >= 15:
                cleaned.append(p_restored)
        return cleaned

    def parse_sections(self, text: str) -> List[Dict[str, str]]:
        """Parses markdown headers or section blocks."""
        lines = text.split("\n")
        sections = []
        current_title = "Giới thiệu"
        current_body = []

        for line in lines:
            line_s = line.strip()
            if line_s.startswith(("#", "Chương", "Phần", "Section", "1.", "2.", "3.", "4.")):
                if current_body:
                    sections.append({
                        "title": current_title,
                        "text": "\n".join(current_body).strip(),
                    })
                    current_body = []
                current_title = re.sub(r"^[#\s\d.-]+", "", line_s) or line_s
            else:
                if line_s:
                    current_body.append(line_s)

        if current_body:
            sections.append({
                "title": current_title,
                "text": "\n".join(current_body).strip(),
            })

        return sections if sections else [{"title": "Toàn văn", "text": text}]

    def summarize_l1(self, text: str, doc_title: str = "Tài liệu") -> str:
        """L1 Micro TL;DR: 1 direct sentence <= 35 words."""
        sections = self.parse_sections(text)
        # Look for solution or experiment sentence
        sol_sent = ""
        exp_sent = ""
        for s in sections:
            role = self.drawer.categorize_section(s["title"], s["text"][:150])
            sents = self.split_clean_sentences(s["text"])
            if not sents:
                continue
            if role == DRAWER_SOLUTION and not sol_sent:
                sol_sent = sents[0]
            elif role == DRAWER_EXPERIMENT and not exp_sent:
                exp_sent = sents[0]

        if sol_sent and exp_sent:
            combined = f"Tài liệu {doc_title} đề xuất kiến trúc {sol_sent.rstrip('.')} và thực nghiệm chứng minh {exp_sent.rstrip('.')}."
            words = combined.split()
            if len(words) <= 40:
                return combined

        # Fallback to first salient sentence
        all_sents = self.split_clean_sentences(text)
        if all_sents:
            first = all_sents[0].rstrip(".")
            return f"Tài liệu {doc_title} tập trung vào {first[:120]}."
        return f"Tài liệu {doc_title} trình bày nội dung nghiên cứu kỹ thuật chuyên sâu."

    def summarize_l2(self, text: str, doc_title: str = "Tài liệu") -> str:
        """L2 Executive Abstract: 3-5 structured sentences covering problem, solution, experiment, conclusion."""
        sections = self.parse_sections(text)
        drawer_sentences: Dict[str, str] = {}

        for s in sections:
            role = self.drawer.categorize_section(s["title"], s["text"][:150])
            sents = self.split_clean_sentences(s["text"])
            if sents and role not in drawer_sentences:
                drawer_sentences[role] = sents[0]

        sentences = []
        if DRAWER_PROBLEM in drawer_sentences:
            sentences.append(f"Về bài toán, {drawer_sentences[DRAWER_PROBLEM].rstrip('.')}.")
        if DRAWER_SOLUTION in drawer_sentences:
            sentences.append(f"Về giải pháp, {drawer_sentences[DRAWER_SOLUTION].rstrip('.')}.")
        if DRAWER_EXPERIMENT in drawer_sentences:
            sentences.append(f"Về thực nghiệm, {drawer_sentences[DRAWER_EXPERIMENT].rstrip('.')}.")
        if DRAWER_CONCLUSION in drawer_sentences:
            sentences.append(f"Kết luận ghi nhận {drawer_sentences[DRAWER_CONCLUSION].rstrip('.')}.")

        if len(sentences) < 3:
            all_sents = self.split_clean_sentences(text)
            for s in all_sents:
                if s not in sentences and len(sentences) < 4:
                    sentences.append(f"{s.rstrip('.')}.")

        return " ".join(sentences)

    def summarize_l3(self, text: str, doc_title: str = "Tài liệu") -> List[Dict[str, str]]:
        """L3 Deep Structural Digest: Per-section key takeaways."""
        sections = self.parse_sections(text)
        digest = []
        for s in sections:
            sents = self.split_clean_sentences(s["text"])
            key_point = sents[0] if sents else "Nội dung phần này."
            digest.append({
                "section_title": s["title"],
                "summary": key_point,
            })
        return digest

    def generate_digest(self, text: str, doc_title: str = "Tài liệu") -> ProgressiveDigest:
        """Generates all 3 levels in one pass."""
        l1 = self.summarize_l1(text, doc_title)
        l2 = self.summarize_l2(text, doc_title)
        l3 = self.summarize_l3(text, doc_title)
        return ProgressiveDigest(
            doc_title=doc_title,
            l1_tldr=l1,
            l2_executive=l2,
            l3_section_digest=l3,
        )

"""
Cross-Document Consensus & Divergence Matrix (Ticket 7.3).
Compares claims, architectures, and quantitative metrics across multiple documents
to discover points of agreement (Consensus), points of contradiction/variation (Divergence),
and complementary findings into a structured Markdown matrix.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
import re


@dataclass
class ConsensusPoint:
    """A claim or finding affirmed across multiple documents."""
    topic: str
    statement: str
    document_supports: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class DivergencePoint:
    """A claim, scope, or number that conflicts between documents."""
    topic: str
    claim_a: str
    claim_b: str
    divergence_type: str  # "numerical_difference", "antonym_conflict", "scope_variation"
    doc_id_a: str = ""
    doc_id_b: str = ""


@dataclass
class CrossDocAnalysisReport:
    """Consolidated cross-document synthesis report."""
    consensus_points: List[ConsensusPoint] = field(default_factory=list)
    divergence_points: List[DivergencePoint] = field(default_factory=list)
    complementary_insights: List[str] = field(default_factory=list)
    summary_markdown: str = ""


class CrossDocConsensusEngine:
    """
    Analyzes agreement and divergence across documents.
    """

    ANTONYM_PAIRS = [
        ("tăng", "giảm"),
        ("nhanh", "chậm"),
        ("tốt", "kém"),
        ("tiết kiệm", "tốn kém"),
        ("hiệu quả", "không hiệu quả"),
        ("cao", "thấp"),
    ]

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

    def extract_numbers_and_metrics(self, text: str) -> List[str]:
        """Extracts numerical quantities and percentages."""
        return re.findall(r"\d+(?:[.,]\d+)?\s*%?", text)

    def extract_keywords(self, text: str) -> Set[str]:
        """Extracts meaningful lower-case words of length >= 4."""
        stopwords = {"trong", "những", "chúng", "được", "nghiên", "nhiều", "phần", "tuyệt", "hoàn", "cũng"}
        words = set(re.findall(r"\b[a-zA-Zà-ỹÀ-Ỹ0-9]{4,}\b", text.lower()))
        return words - stopwords

    def compute_jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def analyze_cross_documents(self, docs: List[Dict[str, Any]]) -> CrossDocAnalysisReport:
        """Compares multiple documents to produce consensus and divergence items."""
        if len(docs) < 2:
            return CrossDocAnalysisReport(
                consensus_points=[],
                divergence_points=[],
                complementary_insights=[],
                summary_markdown="*Cần tối thiểu 2 tài liệu để thực hiện đối chiếu đồng thuận và phân kỳ liên văn bản.*",
            )

        consensus_list: List[ConsensusPoint] = []
        divergence_list: List[DivergencePoint] = []
        complementary_list: List[str] = []

        # Parse sentences per document
        doc_sentences: Dict[str, List[Tuple[str, Set[str], List[str]]]] = {}
        doc_titles: Dict[str, str] = {}
        for d in docs:
            d_id = d.get("document_id", "DOC")
            d_title = d.get("title", d_id)
            doc_titles[d_id] = d_title
            raw_text = d.get("raw_text", "")
            sents = self.split_clean_sentences(raw_text)
            doc_sentences[d_id] = [
                (s, self.extract_keywords(s), self.extract_numbers_and_metrics(s))
                for s in sents
            ]

        doc_ids = list(doc_sentences.keys())

        # Pairwise comparison between first two documents (or multiple)
        seen_pairs = set()
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                id_a, id_b = doc_ids[i], doc_ids[j]
                title_a, title_b = doc_titles[id_a], doc_titles[id_b]

                for s_a, kw_a, num_a in doc_sentences[id_a]:
                    for s_b, kw_b, num_b in doc_sentences[id_b]:
                        sim = self.compute_jaccard(kw_a, kw_b)
                        if sim < 0.28:
                            continue

                        pair_key = (min(s_a, s_b), max(s_a, s_b))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        # Determine if numerical divergence
                        if num_a and num_b and set(num_a) != set(num_b):
                            # Mismatch in numbers on shared context
                            shared_topic = " / ".join(list(kw_a & kw_b)[:3]) or "Chỉ số thực nghiệm"
                            divergence_list.append(
                                DivergencePoint(
                                    topic=f"Số liệu {shared_topic}",
                                    claim_a=f"{s_a} ({title_a})",
                                    claim_b=f"{s_b} ({title_b})",
                                    divergence_type="numerical_difference",
                                    doc_id_a=id_a,
                                    doc_id_b=id_b,
                                )
                            )
                            continue

                        # Check antonym conflict
                        has_antonym = False
                        s_a_lower, s_b_lower = s_a.lower(), s_b.lower()
                        for pos, neg in self.ANTONYM_PAIRS:
                            if (pos in s_a_lower and neg in s_b_lower) or (neg in s_a_lower and pos in s_b_lower):
                                has_antonym = True
                                break

                        if has_antonym:
                            shared_topic = " / ".join(list(kw_a & kw_b)[:3]) or "Khẳng định đối lập"
                            divergence_list.append(
                                DivergencePoint(
                                    topic=f"Mâu thuẫn {shared_topic}",
                                    claim_a=f"{s_a} ({title_a})",
                                    claim_b=f"{s_b} ({title_b})",
                                    divergence_type="antonym_conflict",
                                    doc_id_a=id_a,
                                    doc_id_b=id_b,
                                )
                            )
                            continue

                        # Otherwise, consensus
                        if sim >= 0.35:
                            shared_topic = " / ".join(list(kw_a & kw_b)[:3]) or "Luận điểm chung"
                            consensus_list.append(
                                ConsensusPoint(
                                    topic=shared_topic,
                                    statement=s_a,
                                    document_supports={
                                        id_a: s_a,
                                        id_b: s_b,
                                    },
                                    confidence=round(sim, 2),
                                )
                            )

        # Build Markdown Table
        md_lines = [
            "### Bảng đối chiếu & Tổng hợp Liên tài liệu (Cross-Document Matrix)",
            "",
            "| Điểm đối chiếu | Phân loại | Tài liệu A | Tài liệu B | Đánh giá |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for c in consensus_list[:3]:
            supports = list(c.document_supports.values())
            text_a = supports[0][:120] + "..." if len(supports[0]) > 120 else supports[0]
            text_b = supports[1][:120] + "..." if len(supports) > 1 and len(supports[1]) > 120 else (supports[1] if len(supports) > 1 else "")
            md_lines.append(f"| {c.topic} | **Đồng thuận** | {text_a} | {text_b} | Cùng xác thực luận điểm |")

        for d in divergence_list[:3]:
            claim_a = d.claim_a[:120] + "..." if len(d.claim_a) > 120 else d.claim_a
            claim_b = d.claim_b[:120] + "..." if len(d.claim_b) > 120 else d.claim_b
            reason = "Chênh lệch số liệu đo đạc" if d.divergence_type == "numerical_difference" else "Nhận định trái chiều"
            md_lines.append(f"| {d.topic} | **Phân kỳ** | {claim_a} | {claim_b} | {reason} |")

        if not consensus_list and not divergence_list:
            md_lines.append("| Nội dung tổng quát | Độc lập | Không có giao thoa lớn | Không có giao thoa lớn | Hai tài liệu trình bày hai khía cạnh độc lập |")

        summary_markdown = "\n".join(md_lines)

        return CrossDocAnalysisReport(
            consensus_points=consensus_list,
            divergence_points=divergence_list,
            complementary_insights=complementary_list,
            summary_markdown=summary_markdown,
        )

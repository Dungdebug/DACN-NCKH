"""
Counterfactual Prober & Boundary Inspector (Ticket 7.2).
Inspects document context for boundary conditions, operational constraints,
trade-offs, and prerequisite scopes (e.g. text length, hardware, assumptions).
Detects absolute/over-generalized queries and generates grounded advisories.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
import re


@dataclass
class BoundaryClause:
    """An explicit boundary condition or trade-off extracted from text."""
    condition_text: str
    category: str  # "scope_limit", "trade_off", "prerequisite", "hardware_assumption"
    citation_label: str
    passage_id: str = ""
    severity: str = "moderate"  # "high", "moderate"


@dataclass
class BoundaryAdvisory:
    """Structured advisory regarding boundaries and potential overgeneralizations."""
    has_boundary: bool = False
    challenges_absolute: bool = False
    advisory_text: str = ""
    clauses: List[BoundaryClause] = field(default_factory=list)


class BoundaryInspector:
    """
    Detects operational limits, trade-offs, and counterfactual boundaries.
    """

    _ABSOLUTE_PATTERNS = [
        r"\b(luôn luôn|mọi trường hợp|bất kỳ trường hợp nào|tất cả trường hợp)\b",
        r"\b(hoàn toàn không có|không bao giờ|tuyệt đối không|chắc chắn 100%)\b",
        r"\b(không có bất kỳ (nhược điểm|hạn chế|rủi ro|chi phí) nào)\b",
        r"\b(always|in all cases|never fails|zero downside|without any limitation)\b",
    ]

    _SCOPE_PATTERNS = [
        (r"\b(chỉ khi|chỉ phát huy|với điều kiện|trong điều kiện|áp dụng cho|giới hạn ở|dưới giả định)\b", "scope_limit"),
        (r"\b(đối với (các |những )?(đoạn )?văn bản (quá )?(ngắn|dài)|trên \d+|dưới \d+)\b", "scope_limit"),
        (r"\b(đổi lại|tuy nhiên|mặc dù vậy|thách thức là|hạn chế là|chi phí ban đầu)\b", "trade_off"),
        (r"\b(đòi hỏi|yêu cầu cao|chi phí cập nhật|thời gian nạp|latency)\b", "trade_off"),
        (r"\b(thực hiện trên gpu|phần cứng|ram|vram|smollm2|rtx)\b", "hardware_assumption"),
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

    def is_absolute_query(self, query: str) -> bool:
        """Determines if the query carries over-generalized absolute claims."""
        q_lower = query.lower()
        return any(re.search(pat, q_lower) for pat in self._ABSOLUTE_PATTERNS)

    def extract_boundaries(self, passages: List[Dict[str, Any]]) -> List[BoundaryClause]:
        """Extracts boundary clauses from evidence passages."""
        clauses = []
        for p_idx, passage in enumerate(passages):
            raw_text = passage.get("text", "")
            cit_label = passage.get("citation_label", f"[{p_idx + 1}]")
            p_id = passage.get("passage_id", f"p_{p_idx}")

            sentences = self.split_clean_sentences(raw_text)
            for sent in sentences:
                s_lower = sent.lower()
                detected_cat = None
                for pat, cat in self._SCOPE_PATTERNS:
                    if re.search(pat, s_lower):
                        detected_cat = cat
                        break
                
                if detected_cat:
                    severity = "high" if detected_cat in ("scope_limit", "prerequisite") else "moderate"
                    clauses.append(
                        BoundaryClause(
                            condition_text=sent,
                            category=detected_cat,
                            citation_label=cit_label,
                            passage_id=p_id,
                            severity=severity,
                        )
                    )
        return clauses

    def inspect_and_advise(self, passages: List[Dict[str, Any]], query: str) -> BoundaryAdvisory:
        """Evaluates whether to challenge overgeneralizations or emit boundary advisories."""
        clauses = self.extract_boundaries(passages)
        is_abs = self.is_absolute_query(query)

        if not clauses:
            return BoundaryAdvisory(has_boundary=False, challenges_absolute=False, advisory_text="", clauses=[])

        # If user asks an absolute question and boundaries exist:
        if is_abs:
            primary = clauses[0]
            advisory_text = (
                f"Không thể khẳng định trong mọi trường hợp. Tài liệu chỉ ra điều kiện ràng buộc: "
                f"{primary.condition_text} {primary.citation_label}"
            )
            return BoundaryAdvisory(
                has_boundary=True,
                challenges_absolute=True,
                advisory_text=advisory_text,
                clauses=clauses,
            )

        # Normal advisory
        primary = clauses[0]
        advisory_text = f"Lưu ý về điều kiện biên: {primary.condition_text} {primary.citation_label}"
        return BoundaryAdvisory(
            has_boundary=True,
            challenges_absolute=False,
            advisory_text=advisory_text,
            clauses=clauses,
        )

    def enhance_answer_with_boundaries(
        self,
        base_answer: str,
        passages: List[Dict[str, Any]],
        query: str,
    ) -> str:
        """Appends boundary conditions or counter-statements to the answer if applicable."""
        advisory = self.inspect_and_advise(passages, query)
        if not advisory.has_boundary:
            return base_answer

        # If already discussed in base answer, skip repeating
        if advisory.clauses and advisory.clauses[0].condition_text in base_answer:
            return base_answer

        if advisory.challenges_absolute:
            return f"{advisory.advisory_text}\n\nChi tiết cụ thể: {base_answer}"

        return f"{base_answer}\n\n👉 **{advisory.advisory_text}**"


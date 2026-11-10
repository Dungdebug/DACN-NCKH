"""
Data structures for Self-Study via Self-Generated QA (Proposal Section 6.4).
100% On-Device, Zero External API, Pure Python Data Classes.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class QARuleType(str, Enum):
    """Categories of syntactic rules used for grounded QA pair generation."""
    DEFINITION = "definition"        # X là Y -> X là gì?
    MECHANISM = "mechanism"          # X hoạt động nhờ/bằng/qua Y -> X hoạt động như thế nào?
    METRIC = "metric"                # Chỉ số / số liệu / định lượng
    ENUMERATION = "enumeration"      # Gồm / bao gồm các thành phần
    COMPARISON = "comparison"        # So sánh X hơn Y
    LIMITATION = "limitation"        # Hạn chế / nhược điểm / thách thức


@dataclass
class SelfStudyQAPair:
    """A single self-generated question-answer pair grounded in document structure."""
    question: str
    answer: str
    span_start: int
    span_end: int
    source_text: str
    document_id: str = ""
    level: int = 1                  # 1=Paragraph, 2=Section, 3=Document
    rule_type: QARuleType = QARuleType.DEFINITION
    confidence: float = 1.0
    section_title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Strict validation: answer must exist in source_text
        if self.answer not in self.source_text:
            raise ValueError(
                f"Grounding violation: answer '{self.answer[:30]}...' not found in source text"
            )
        # Ensure question ends with question mark
        if not self.question.endswith("?"):
            self.question = f"{self.question}?"

    @property
    def span_length(self) -> int:
        return max(0, self.span_end - self.span_start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "document_id": self.document_id,
            "level": self.level,
            "rule_type": self.rule_type.value if isinstance(self.rule_type, QARuleType) else str(self.rule_type),
            "confidence": self.confidence,
            "section_title": self.section_title,
            "metadata": self.metadata,
        }


@dataclass
class SelfStudyReport:
    """Consolidated report of self-study pairs extracted from a document."""
    document_id: str
    total_pairs: int
    pairs: List[SelfStudyQAPair] = field(default_factory=list)
    pairs_by_rule: Dict[str, int] = field(default_factory=dict)
    extraction_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "total_pairs": self.total_pairs,
            "pairs_by_rule": self.pairs_by_rule,
            "extraction_time_ms": self.extraction_time_ms,
            "pairs": [p.to_dict() for p in self.pairs],
            "metadata": self.metadata,
        }


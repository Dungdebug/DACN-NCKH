"""
Data types and representations for document structural parsing in SA-CMS.
Supports multi-level structural hierarchy: Chunk, Paragraph, Section, Document.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class BoundaryType(Enum):
    CHUNK = "chunk"
    PARAGRAPH = "paragraph"
    SECTION = "section"
    DOCUMENT = "document"
    RANDOM = "random"


@dataclass
class StructuralSpan:
    span_type: BoundaryType
    start_token: int
    end_token: int
    start_char: int = 0
    end_char: int = 0
    text: str = ""
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_length(self) -> int:
        return max(0, self.end_token - self.start_token)


@dataclass
class DocumentStructure:
    text: str
    total_tokens: int
    paragraphs: List[StructuralSpan] = field(default_factory=list)
    sections: List[StructuralSpan] = field(default_factory=list)
    document_span: Optional[StructuralSpan] = None
    chunks: List[StructuralSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_update_boundaries(self, level: int) -> List[int]:
        """
        Returns end token indices for update events at a given memory level in SA-CMS:
        Level 1: Paragraph boundaries
        Level 2: Section boundaries
        Level 3: Document boundaries
        """
        if level == 1:
            boundaries = [p.end_token for p in self.paragraphs]
        elif level == 2:
            boundaries = [s.end_token for s in self.sections]
        elif level == 3:
            boundaries = [self.total_tokens] if self.total_tokens > 0 else []
        else:
            boundaries = [self.total_tokens]

        # Ensure sorted, deduplicated, within range, ending with total_tokens if needed
        boundaries = sorted(list(set(b for b in boundaries if 0 < b <= self.total_tokens)))
        if not boundaries or boundaries[-1] != self.total_tokens:
            boundaries.append(self.total_tokens)
        return boundaries

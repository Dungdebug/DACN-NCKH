"""
Scope-Aware Formula & Example Block Scanner.

Detects the full boundaries (start -> body/steps -> end) of formulas and worked
examples in document passages so answers can present them cleanly and visually
instead of truncating mid-example or dumping concatenated formula paragraphs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from src.web.document_analyzer import strip_accents


@dataclass
class FormulaBlock:
    label: str
    expression: str
    condition: str = ""
    citation_label: str = "[1]"


@dataclass
class ExampleBlock:
    title: str
    problem: str = ""
    steps: List[str] = field(default_factory=list)
    citation_label: str = "[1]"


_SECTION_BOUNDARY_RE = re.compile(
    r"^\s*(?:#{1,6}\s+|\d+(?:\.\d+)*\.?\s+[A-ZÀ-Ỹ]|(?:Phần|Chương|Bài|Mục|Section|Topic)\s+\d+)",
    re.IGNORECASE,
)

_EXAMPLE_START_RE = re.compile(
    r"(?:^|\n|\.\s+)(Ví\s*dụ(?:\s*\d+(?:\.\d+)*)?(?:\s*\([^)]+\))?\s*:?|Bài\s*toán\s*(?:mẫu|minh\s*họa)?(?:\s*\d+(?:\.\d+)*)?\s*:?)",
    re.IGNORECASE,
)

_STEP_MARKER_RE = re.compile(
    r"(?=(?:Bước\s*\d+(?:\s*\([^)]+\))?\s*:|Đề\s*bài\s*:|Lời\s*giải\s*:|Hướng\s*dẫn\s*:|Kết\s*luận\s*:))",
    re.IGNORECASE,
)

# Matches "Label:" boundaries for inline concatenated formulas (Unicode-safe for all Vietnamese characters)
_INLINE_LABEL_RE = re.compile(
    r"(?:^|(?<=\s))([^\W\d_][^\n:()=→⇌⇒]{2,75}?(?:\s*\([^()\n]{1,25}\))?)\s*:\s*",
    re.UNICODE,
)


def _clean_line(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).strip()


def _split_expression_and_condition(expr_raw: str) -> Tuple[str, str]:
    """Separates trailing parenthetical condition/unit notes from a formula expression."""
    txt = _clean_line(expr_raw).rstrip(".;")
    m = re.match(r"^(.*?)\s*\(([^()]{4,100})\)\s*$", txt)
    if m:
        left = m.group(1).strip()
        cond = m.group(2).strip()
        if any(op in left for op in ("=", "→", "⇌", "⇒", "%", "^")):
            return left, cond
    return txt, ""


def extract_formula_blocks(
    text: str,
    citation_label: str = "[1]",
    max_blocks: int = 6,
) -> List[FormulaBlock]:
    """
    Extracts structured FormulaBlocks (label, expression, condition/unit) from passage text,
    splitting concatenated inline formulas as well as multi-line formula blocks.
    """
    if not text or not text.strip():
        return []

    blocks: List[FormulaBlock] = []
    seen_expr: set[str] = set()

    for raw_line in text.splitlines():
        line = _clean_line(raw_line)
        if len(line) < 6:
            continue
        line_norm = strip_accents(line.lower())

        # Skip example step lines so we don't confuse intermediate calculations with general formulas
        if re.match(r"^(?:vi\s*du|de\s*bai|buoc\s*\d|loi\s*giai)\b", line_norm):
            continue

        # 1. Try splitting multiple "Label: Expression" segments on the same line
        label_matches = [
            m for m in _INLINE_LABEL_RE.finditer(line)
            if m.group(1).strip()[0].isupper() or m.group(1).strip().startswith("∑")
        ]
        if label_matches:
            matched_any = False
            for idx_m, m in enumerate(label_matches):
                lbl = _clean_line(m.group(1)).lstrip("-*• ")
                val_start = m.end()
                val_end = label_matches[idx_m + 1].start() if idx_m + 1 < len(label_matches) else len(line)
                val_raw = line[val_start:val_end].strip()
                val_norm = strip_accents(val_raw.lower())
                if not (any(op in val_raw for op in ("=", "→", "⇌", "⇒", "%")) or "mod " in val_norm):
                    continue
                expr, cond = _split_expression_and_condition(val_raw)
                key = strip_accents(expr.lower())
                if len(expr) >= 3 and key not in seen_expr:
                    seen_expr.add(key)
                    blocks.append(FormulaBlock(
                        label=lbl,
                        expression=expr,
                        condition=cond,
                        citation_label=citation_label,
                    ))
                    matched_any = True
                    if len(blocks) >= max_blocks:
                        return blocks
            if matched_any:
                continue

        # 2. Standalone formula line without "Label:" prefix
        has_op = any(op in line for op in ("=", "→", "⇌", "⇒")) or "mod " in line_norm
        if has_op and len(line) <= 180:
            expr, cond = _split_expression_and_condition(line.lstrip("-*• "))
            key = strip_accents(expr.lower())
            if key not in seen_expr:
                seen_expr.add(key)
                blocks.append(FormulaBlock(
                    label="Biểu thức / Phương trình",
                    expression=expr,
                    condition=cond,
                    citation_label=citation_label,
                ))
                if len(blocks) >= max_blocks:
                    return blocks

    return blocks


def extract_example_blocks(
    text: str,
    citation_label: str = "[1]",
    max_blocks: int = 3,
) -> List[ExampleBlock]:
    """
    Extracts complete worked ExampleBlocks (title, problem statement, step-by-step solution)
    respecting example start/end boundaries across lines and inline steps.
    """
    if not text or not text.strip():
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    blocks: List[ExampleBlock] = []
    i = 0

    while i < len(lines) and len(blocks) < max_blocks:
        ln = lines[i]
        ln_norm = strip_accents(ln.lower())
        is_ex_start = bool(
            re.match(r"^(?:vi\s*du|bai\s*toan\s*(?:mau|minh\s*hoa)?|minh\s*hoa)\b", ln_norm)
        )
        if not is_ex_start:
            i += 1
            continue

        # Collect all consecutive lines belonging to this example until a boundary
        chunk_parts = [ln]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            nxt_norm = strip_accents(nxt.lower())
            if _SECTION_BOUNDARY_RE.match(nxt):
                break
            if re.match(r"^(?:vi\s*du|bai\s*toan\s*(?:mau|minh\s*hoa)?)\b", nxt_norm):
                break
            chunk_parts.append(nxt)
            j += 1

        full_ex = " ".join(chunk_parts)
        parsed = _parse_single_example(full_ex, citation_label=citation_label)
        if parsed:
            blocks.append(parsed)
        i = j

    return blocks


def _parse_single_example(full_ex: str, citation_label: str = "[1]") -> Optional[ExampleBlock]:
    """Parses a bounded example string into title, problem, and ordered steps."""
    txt = _clean_line(full_ex)
    if len(txt) < 12:
        return None

    # Split at "Đề bài:" / "Bước X:" / "Lời giải:" markers
    parts = [p.strip() for p in _STEP_MARKER_RE.split(txt) if p and p.strip()]
    if not parts:
        return None

    title = parts[0].rstrip(":").strip()
    problem = ""
    steps: List[str] = []

    for seg in parts[1:]:
        seg_norm = strip_accents(seg.lower())
        if seg_norm.startswith("de bai"):
            problem = re.sub(r"^Đề\s*bài\s*:\s*", "", seg, flags=re.IGNORECASE).strip()
        else:
            steps.append(seg)

    # If "Đề bài:" wasn't explicit, check if title itself contains ":" followed by problem text
    if not problem and not steps and ":" in txt:
        head, tail = txt.split(":", 1)
        title = head.strip()
        problem = tail.strip()

    return ExampleBlock(
        title=title or "Ví dụ minh họa",
        problem=problem,
        steps=steps,
        citation_label=citation_label,
    )


def scan_passages_for_formulas_and_examples(
    passages: Sequence[Any],
    passage_cit_labels: Optional[Sequence[str]] = None,
) -> Tuple[List[FormulaBlock], List[ExampleBlock]]:
    """Scans a sequence of evidence passages and returns deduplicated FormulaBlocks and ExampleBlocks."""
    all_formulas: List[FormulaBlock] = []
    all_examples: List[ExampleBlock] = []
    seen_f: set[str] = set()
    seen_e: set[str] = set()

    for idx, p in enumerate(passages):
        p_text = getattr(p, "text", "") if hasattr(p, "text") else str(p)
        lbl = (
            passage_cit_labels[idx]
            if passage_cit_labels and idx < len(passage_cit_labels)
            else f"[{idx + 1}]"
        )
        for fb in extract_formula_blocks(p_text, citation_label=lbl):
            k = strip_accents(fb.expression.lower())
            if k not in seen_f:
                seen_f.add(k)
                all_formulas.append(fb)
        for eb in extract_example_blocks(p_text, citation_label=lbl):
            k = strip_accents(eb.title.lower())
            if k not in seen_e:
                seen_e.add(k)
                all_examples.append(eb)

    return all_formulas, all_examples


def format_formula_blocks_markdown(formulas: Sequence[FormulaBlock], max_items: int = 5) -> str:
    """Renders FormulaBlocks as clean, scannable Markdown bullets with scope/conditions."""
    if not formulas:
        return ""
    lines = ["📐 **Công thức & Phạm vi áp dụng trong tài liệu:**"]
    for fb in formulas[:max_items]:
        cond_part = f" *(Phạm vi/Quy đổi: {fb.condition})*" if fb.condition else ""
        lines.append(f"- **{fb.label}:** `{fb.expression}`{cond_part} {fb.citation_label}")
    return "\n".join(lines)


def format_example_blocks_markdown(examples: Sequence[ExampleBlock], max_items: int = 2) -> str:
    """Renders ExampleBlocks with clear problem statement and step-by-step breakdown."""
    if not examples:
        return ""
    lines = ["📝 **Ví dụ minh họa chi tiết trong tài liệu:**"]
    for eb in examples[:max_items]:
        lines.append(f"- **{eb.title}** {eb.citation_label}")
        if eb.problem:
            lines.append(f"  - *Đề bài:* {eb.problem}")
        for st in eb.steps[:6]:
            lines.append(f"  - {st}")
    return "\n".join(lines)

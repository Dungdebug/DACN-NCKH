"""
Neural Query Decomposer: Decomposes compound, multi-part, comparative, or causal queries
into atomic sub-queries for multi-hop retrieval and synthesis.
Sovereign Brain 4.0: 100% offline, deterministic regex heuristics with zero heavy LLM overhead.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import re


@dataclass
class SubQuery:
    text: str
    intent: str = "general"
    dependency_index: Optional[int] = None


@dataclass
class DecompositionResult:
    is_compound: bool
    sub_queries: List[SubQuery] = field(default_factory=list)
    original_query: str = ""
    reasoning_type: str = "atomic"


class QueryDecomposer:
    """
    Decomposes complex user queries into atomic sub-questions.
    Follows Ponytail principles: deterministic, fast regex heuristics with zero heavy LLM overhead.
    """

    # Multi-entity & general comparison patterns
    COMPARISON_PREFIX_PATTERNS = [
        re.compile(
            r"(?:so sánh|sự khác (?:nhau|biệt) giữa)\s+(?:sự khác biệt giữa|điểm khác biệt giữa|giữa)?\s*(.+)",
            re.IGNORECASE,
        ),
    ]

    COMPARISON_PATTERNS = [
        re.compile(
            r"so sánh\s+(?:sự khác biệt giữa|điểm khác biệt giữa|giữa)?\s*([^,\.]+?)\s+và\s+([^,\.\?]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"sự khác (?:nhau|biệt) giữa\s+([^,\.]+?)\s+và\s+([^,\.\?]+)",
            re.IGNORECASE,
        ),
    ]

    # Causal & Multi-Hop Bridge patterns
    CAUSAL_PATTERNS = [
        # 1. Tại sao / Vì sao ... (lại) dẫn đến / dẫn tới / gây ra ...
        (
            re.compile(
                r"(?:tại sao|vì sao)\s+(.+?)\s+(?:lại\s+)?(?:dẫn đến|dẫn tới|gây ra)\s+(.+)",
                re.IGNORECASE,
            ),
            "why_leads_to",
        ),
        # 2. Mối liên hệ / quan hệ giữa ... và ...
        (
            re.compile(
                r"mối\s+(?:liên hệ|quan hệ)\s+giữa\s+(.+?)\s+và\s+(.+)",
                re.IGNORECASE,
            ),
            "relation_between",
        ),
        # 3. Ảnh hưởng / tác động của ... đối với / đến / lên ...
        (
            re.compile(
                r"(?:ảnh hưởng|tác động)\s+của\s+(.+?)\s+(?:đối với|đến|lên)\s+(.+)",
                re.IGNORECASE,
            ),
            "impact_on",
        ),
    ]

    CONJUNCTION_SPLITTERS = [
        r"\s+đồng thời\s+",
        r"\s+song song đó\s+",
        r"\s+cũng như\s+",
        r"\s+ngoài ra\s+",
        r"\s+và\s+",
    ]

    ACRONYM_STOPWORDS = {
        "VÀ", "CHO", "VỚI", "CỦA", "TRONG", "LÀ", "CÁC", "NHỮNG",
        "THE", "AND", "FOR", "WITH", "FROM", "ABOUT"
    }

    INVALID_TITLES = {
        "Tại Sao", "Vì Sao", "So Sánh", "Làm Sao", "Như Thế Nào",
        "Là Gì", "Điểm Khác Biệt", "Sự Khác Biệt", "Cho Biết",
        "Hãy Cho Biết", "Đặc Điểm", "Nội Dung"
    }

    def _clean_causal_term(self, term: str) -> str:
        t = term.strip("?. \t\n")
        suffixes = [
            " là gì", " như thế nào", " ra sao", " thế nào", " ra làm sao",
            " diễn ra thế nào", " diễn ra như thế nào", " có đúng không"
        ]
        changed = True
        while changed:
            changed = False
            t_lower = t.lower()
            for suf in suffixes:
                if t_lower.endswith(suf):
                    t = t[:-len(suf)].strip("?. \t\n")
                    changed = True
                    break
        return t.strip("?. \t\n")

    def _parse_comparison_entities(self, target_str: str) -> List[str]:
        s = target_str.strip("?. \t\n")
        if "," in s:
            raw_parts = [p.strip() for p in s.split(",") if p.strip()]
            parts = []
            for p in raw_parts:
                if re.search(r"\s+và\s+", p, re.IGNORECASE):
                    subparts = re.split(r"\s+và\s+", p, flags=re.IGNORECASE)
                    parts.extend([sp.strip() for sp in subparts if sp.strip()])
                else:
                    cleaned = re.sub(r"^(?:và|cũng như)\s+", "", p, flags=re.IGNORECASE).strip()
                    if cleaned:
                        parts.append(cleaned)
            return parts
        elif re.search(r"\s+và\s+", s, re.IGNORECASE):
            subparts = re.split(r"\s+và\s+", s, flags=re.IGNORECASE)
            return [sp.strip() for sp in subparts if sp.strip()]
        return []

    def extract_focus_entities(self, query: str) -> List[str]:
        """
        Extract technical acronyms, model names, hardware, sections, and capitalized entities
        (e.g. SA-CMS, SmolLM2-135M, BM25, RAG, RTX 3050, Topic 1, Chủ đề 2, Sovereign Brain).
        """
        if not query or not query.strip():
            return []

        q = query.strip()
        raw: List[str] = []

        # 1. Quoted terms
        for m in re.finditer(r'["“]([^"”]+)["”]', q):
            val = m.group(1).strip()
            if len(val) >= 2:
                raw.append(val)

        # 2. Labeled topics / chapters / sections
        for m in re.finditer(r"\b(?:Topic|Chủ đề|Chương|Phần|Mục|Điều|Khoản|Bảng|Hình|Figure|Table|Section)\s+\d+(?:\.\d+)*\b", q, re.IGNORECASE):
            raw.append(m.group(0).strip())

        # 3. Hardware / product specifications
        for m in re.finditer(r"\b(?:RTX|GTX|GPU|CPU|TPU|CUDA)\s+[A-Za-z0-9\.\-]+\b", q, re.IGNORECASE):
            raw.append(m.group(0).strip())
        for m in re.finditer(r"\b[A-Z]{2,}\s+\d+[A-Za-z0-9]*\b", q):
            raw.append(m.group(0).strip())

        # 4. Hyphenated / dotted technical model identifiers (with letters)
        for m in re.finditer(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b", q):
            val = m.group(0).strip()
            if re.search(r"[A-Za-z]", val) and not val.isdigit():
                raw.append(val)

        # 5. Mixed-case CamelCase (e.g. SmolLM2, ResNet50, PyTorch)
        for m in re.finditer(r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b", q):
            raw.append(m.group(0).strip())

        # 6. Multi-word Title Case phrases
        for m in re.finditer(r"\b[A-ZÀ-Ỹ][a-zà-ỹ0-9]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ0-9]+)+\b", q):
            val = m.group(0).strip()
            if val not in self.INVALID_TITLES:
                raw.append(val)

        # 7. Acronyms & codes (e.g. BM25, RAG, CMS)
        for m in re.finditer(r"\b[A-Z][A-Z0-9]{1,}\b", q):
            val = m.group(0).strip()
            if val.upper() not in self.ACRONYM_STOPWORDS:
                raw.append(val)

        # Deduplicate while preserving first appearance
        unique: List[str] = []
        for item in raw:
            if not any(item.lower() == u.lower() for u in unique):
                unique.append(item)

        # Substring suppression (e.g., if 'RTX 3050' is present, drop 'RTX')
        filtered = [
            e for e in unique
            if not any(e.lower() != other.lower() and e.lower() in other.lower() for other in unique)
        ]

        # Sort by appearance in original query
        def get_pos(ent: str) -> int:
            idx = q.lower().find(ent.lower())
            return idx if idx != -1 else 999999

        filtered.sort(key=get_pos)
        return filtered

    def decompose(self, query: str) -> DecompositionResult:
        q_clean = query.strip()
        if not q_clean:
            return DecompositionResult(
                is_compound=False,
                sub_queries=[SubQuery(query)],
                original_query=query,
                reasoning_type="atomic",
            )

        # 1. Comparison queries (3+ entities -> multi_entity_comparison, 2 entities -> comparison)
        for pat in self.COMPARISON_PREFIX_PATTERNS:
            m = pat.search(q_clean)
            if m:
                target_str = m.group(1).strip()
                entities = self._parse_comparison_entities(target_str)
                if len(entities) >= 3:
                    sub_queries = [
                        SubQuery(
                            text=f"Đặc điểm và nội dung của {ent} là gì?",
                            intent=f"comparison_{i+1}",
                        )
                        for i, ent in enumerate(entities)
                    ]
                    return DecompositionResult(
                        is_compound=True,
                        sub_queries=sub_queries,
                        original_query=query,
                        reasoning_type="multi_entity_comparison",
                    )
                elif len(entities) == 2:
                    sub_queries = [
                        SubQuery(text=f"Đặc điểm và nội dung của {entities[0]} là gì?", intent="comparison_a"),
                        SubQuery(text=f"Đặc điểm và nội dung của {entities[1]} là gì?", intent="comparison_b"),
                    ]
                    return DecompositionResult(
                        is_compound=True,
                        sub_queries=sub_queries,
                        original_query=query,
                        reasoning_type="comparison",
                    )

        # Fallback comparison pattern check
        for pat in self.COMPARISON_PATTERNS:
            m = pat.search(q_clean)
            if m:
                ent_a = m.group(1).strip()
                ent_b = m.group(2).strip()
                sub_queries = [
                    SubQuery(text=f"Đặc điểm và nội dung của {ent_a} là gì?", intent="comparison_a"),
                    SubQuery(text=f"Đặc điểm và nội dung của {ent_b} là gì?", intent="comparison_b"),
                ]
                return DecompositionResult(
                    is_compound=True,
                    sub_queries=sub_queries,
                    original_query=query,
                    reasoning_type="comparison",
                )

        # 2. Causal & Multi-Hop Bridge Decomposition
        for pat, kind in self.CAUSAL_PATTERNS:
            m = pat.search(q_clean)
            if m:
                cause = self._clean_causal_term(m.group(1))
                effect = self._clean_causal_term(m.group(2))
                if cause and effect and len(cause) >= 2 and len(effect) >= 2:
                    if kind == "why_leads_to":
                        sq_cause = SubQuery(
                            text=f"Cơ chế và nguyên nhân của {cause} là gì?",
                            intent="cause_source",
                        )
                        sq_effect = SubQuery(
                            text=f"Tác động của {cause} dẫn đến {effect} như thế nào?",
                            intent="effect_target",
                            dependency_index=0,
                        )
                    elif kind == "relation_between":
                        sq_cause = SubQuery(
                            text=f"Đặc điểm và cơ chế của {cause} là gì?",
                            intent="cause_source",
                        )
                        sq_effect = SubQuery(
                            text=f"Mối liên hệ giữa {cause} và {effect} như thế nào?",
                            intent="effect_target",
                            dependency_index=0,
                        )
                    else:  # impact_on
                        sq_cause = SubQuery(
                            text=f"Đặc điểm và cơ chế của {cause} là gì?",
                            intent="cause_source",
                        )
                        sq_effect = SubQuery(
                            text=f"Tác động của {cause} đối với {effect} như thế nào?",
                            intent="effect_target",
                            dependency_index=0,
                        )

                    return DecompositionResult(
                        is_compound=True,
                        sub_queries=[sq_cause, sq_effect],
                        original_query=query,
                        reasoning_type="causal_chain",
                    )

        # 3. Sequential question mark split: "Câu hỏi 1? Câu hỏi 2?"
        if "?" in q_clean[:-1]:
            parts = [p.strip() for p in q_clean.split("?") if len(p.strip()) >= 10]
            if len(parts) >= 2:
                sub_queries = [
                    SubQuery(text=f"{p}?" if not p.endswith("?") else p, intent="sequential")
                    for p in parts
                ]
                return DecompositionResult(
                    is_compound=True,
                    sub_queries=sub_queries,
                    original_query=query,
                    reasoning_type="sequential",
                )

        # 4. Conjunction splitting
        for pattern in self.CONJUNCTION_SPLITTERS:
            if re.search(pattern, q_clean, re.IGNORECASE):
                parts = re.split(pattern, q_clean, flags=re.IGNORECASE)
                parts = [p.strip() for p in parts if len(p.strip()) >= 10]
                if len(parts) >= 2:
                    # Enrich second clause if missing interrogative structure
                    sub_queries = []
                    for idx, part in enumerate(parts):
                        txt = part.rstrip("?. ")
                        if not any(k in txt.lower() for k in ["là gì", "thế nào", "ra sao", "bao nhiêu", "ai", "ở đâu"]):
                            txt = f"Nội dung về {txt} là gì?"
                        else:
                            txt = f"{txt}?"
                        sub_queries.append(SubQuery(text=txt, intent=f"part_{idx+1}"))

                    return DecompositionResult(
                        is_compound=True,
                        sub_queries=sub_queries,
                        original_query=query,
                        reasoning_type="compound",
                    )

        # 5. Single atomic query fallback
        return DecompositionResult(
            is_compound=False,
            sub_queries=[SubQuery(text=query, intent="atomic")],
            original_query=query,
            reasoning_type="atomic",
        )

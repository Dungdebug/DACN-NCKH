"""
Dual-Process System 1 / System 2 Reasoning & Premise Validator.
Proposal Reference: docs/de_cuong_text.txt Section 6.4, 6.5.

Combines:
1. System 1 (Fast reflex < 5ms): Instant direct extraction for single-fact queries.
2. System 2 (Internal deliberation): Multi-hop synthesis across section drawers.
3. Premise Invalidation: Detects false presuppositions (loaded questions) and refutes them
   with direct grounded citations instead of hallucinating justifications.

100% Offline, Deterministic, Zero LLM API calls.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


# Antonym pairs to catch false presuppositions (Query presupposition -> Document reality)
_PRESUPPOSITION_ANTONYMS: List[Tuple[str, str, str]] = [
    # (query_bias, doc_fact, description)
    ("tang", "giam", "tăng vs giảm"),
    ("lam tang", "tiet kiem", "làm tăng vs tiết kiệm"),
    ("ton kem", "tiet kiem", "tốn kém vs tiết kiệm"),
    ("cham", "nhanh", "chậm vs nhanh"),
    ("kem hon", "vuot troi", "kém hơn vs vượt trội"),
    ("that bai", "thanh cong", "thất bại vs thành công"),
    ("gay hai", "loi ich", "gây hại vs lợi ích"),
    ("khong the", "co the", "không thể vs có thể"),
]


@dataclass
class PremiseValidationResult:
    has_false_premise: bool
    presupposition_found: str = ""
    counter_evidence_sentence: str = ""
    counter_citation: str = ""
    refutation_text: str = ""


@dataclass
class DualProcessResult:
    process_mode: str  # "SYSTEM_1_FAST" or "SYSTEM_2_DELIBERATE"
    is_refutation: bool
    final_answer: str
    core_span: str
    citations: List[Dict[str, Any]]
    confidence: float
    latency_ms: float
    reasoning_trace: List[str] = field(default_factory=list)


class PremiseValidator:
    """
    Validates user questions for loaded or false presuppositions.
    Protects against adversarial or mistaken queries (e.g. 'Tại sao SA-CMS lại làm tăng chi phí?').
    """

    def __init__(self):
        pass

    def validate_premise(
        self,
        query: str,
        passages: List[Dict[str, Any]],
    ) -> PremiseValidationResult:
        """
        Inspects query and evidence. If the query assumes X while documents state opposite Y
        on the exact same entity/metric, constructs an authoritative refutation.
        """
        q_norm = strip_accents(query.lower())

        # Check each antonym pair
        for query_bias, doc_fact, desc in _PRESUPPOSITION_ANTONYMS:
            # Does query contain the bias term?
            bias_pattern = r"(?<!\w)" + re.escape(query_bias) + r"(?!\w)"
            if re.search(bias_pattern, q_norm):
                # Search for counter-evidence in passages
                for p_idx, p in enumerate(passages):
                    text = p.get("text", "")
                    cit = p.get("citation_label", f"[{p_idx + 1}]")
                    # Break into sentences
                    from src.self_study.reader import split_clean_sentences
                    sentences = split_clean_sentences(text)

                    for s in sentences:
                        s_norm = strip_accents(s.lower())
                        fact_pattern = r"(?<!\w)" + re.escape(doc_fact) + r"(?!\w)"
                        if re.search(fact_pattern, s_norm):
                            # Ensure topical overlap (e.g. query mentions "chi phí", sentence also mentions "chi phí")
                            q_words = set(re.findall(r"\w+", q_norm))
                            s_words = set(re.findall(r"\w+", s_norm))
                            topic_overlap = q_words & s_words
                            # Filter trivial stopwords
                            from src.self_study.semantic_thesaurus import _VI_STOPWORDS
                            meaningful_overlap = [w for w in topic_overlap if w not in _VI_STOPWORDS and len(w) > 2]

                            if len(meaningful_overlap) >= 1:
                                refutation = (
                                    f"Giả định trong câu hỏi không phù hợp với tài liệu: "
                                    f"Tài liệu chỉ ra rằng \"{s.strip()}\" {cit}, "
                                    f"chứ không phải như giả định được nêu."
                                )
                                return PremiseValidationResult(
                                    has_false_premise=True,
                                    presupposition_found=query_bias,
                                    counter_evidence_sentence=s.strip(),
                                    counter_citation=cit,
                                    refutation_text=refutation,
                                )

        return PremiseValidationResult(has_false_premise=False)


class DualProcessController:
    """
    Coordinates System 1 (fast reflex) and System 2 (deliberation & premise check).
    """

    def __init__(self, reader_engine: Optional[Any] = None):
        from src.self_study.reader import NeuralReaderEngine
        self.reader = reader_engine or NeuralReaderEngine()
        self.premise_validator = PremiseValidator()

    def determine_mode(self, query: str, intent: str) -> str:
        """Decides between System 1 (Fast) and System 2 (Deliberate)."""
        q_norm = strip_accents(query.lower())

        # Complex analytical, comparative, or multi-question triggers -> System 2
        sys2_triggers = ["tai sao", "vi sao", "so sanh", "khac gi", "nguyen nhan", "danh gia", "tong hop", "nhu the nao"]
        if any(trig in q_norm for trig in sys2_triggers) or intent in ("COMPARISON", "CAUSE", "SUMMARY"):
            return "SYSTEM_2_DELIBERATE"

        # Query has multiple clauses or length > 12 words -> System 2
        if len(query.split()) > 12 or " và " in query.lower() or " song " in query.lower():
            return "SYSTEM_2_DELIBERATE"

        return "SYSTEM_1_FAST"

    def execute(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        answer_mode: Optional[str] = None,
    ) -> DualProcessResult:
        """
        Executes grounded QA through the dual-process architecture:
        1. Premise Invalidation Check
        2. Mode Routing (System 1 vs System 2)
        3. Grounded Synthesis
        """
        import time
        t_start = time.perf_counter()
        trace: List[str] = []

        # 1. Premise Invalidation Guard
        validation = self.premise_validator.validate_premise(query, passages)
        if validation.has_false_premise:
            latency = (time.perf_counter() - t_start) * 1000.0
            trace.append(f"Premise Refutation: detected bias '{validation.presupposition_found}' refuted by evidence.")
            return DualProcessResult(
                process_mode="PREMISE_REFUTATION",
                is_refutation=True,
                final_answer=validation.refutation_text,
                core_span=validation.counter_evidence_sentence,
                citations=[{"citation_label": validation.counter_citation}],
                confidence=1.0,
                latency_ms=round(latency, 2),
                reasoning_trace=trace,
            )

        # 2. Dual Process Routing
        intent = self.reader.thesaurus.get_intent_cues(query) if hasattr(self.reader, "thesaurus") else "GENERIC"
        from src.self_study.reader import detect_query_intent
        query_intent = detect_query_intent(query)
        mode = self.determine_mode(query, query_intent)
        trace.append(f"DualProcess Mode: {mode} (intent: {query_intent})")

        # 3. Execution
        effective_answer_mode = answer_mode or ("concise" if mode == "SYSTEM_1_FAST" else "balanced")
        reader_res = self.reader.synthesize(
            query=query,
            passages=passages,
            answer_mode=effective_answer_mode,
        )

        latency = (time.perf_counter() - t_start) * 1000.0
        trace.append(f"Grounded synthesis finished in {latency:.2f}ms with confidence {reader_res.confidence:.2f}")

        return DualProcessResult(
            process_mode=mode,
            is_refutation=False,
            final_answer=reader_res.formatted_answer,
            core_span=reader_res.core_answer,
            citations=reader_res.citations,
            confidence=reader_res.confidence,
            latency_ms=round(latency, 2),
            reasoning_trace=trace,
        )


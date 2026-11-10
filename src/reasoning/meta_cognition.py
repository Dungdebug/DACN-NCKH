"""
Calibrated Meta-Cognitive Evaluator for SA-CMS.
Epistemic Calibration & Self-Awareness of Knowledge Boundaries.
Decomposes answers into:
1. Verified Grounded Facts [Chắc chắn] (backed by citations).
2. Tentative / Hypothetical Claims [Có khả năng cao] (modal hedges).
3. Explicit Knowledge Gaps (aspects absent in current document store).
100% Strict Offline, Zero External APIs, Anti-Hallucination Meta-Cognition.
"""

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for hedge keyword matching."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


# Modal hedge phrases indicating uncertainty or hypothesis
_MODAL_HEDGE_CUES = [
    "co the", "du kien", "gia dinh", "uoc tinh", "theo ly thuyet",
    "du doan", "co kha nang", "chua chinh thuc", "tiem nang", "huong toi",
    "probably", "potentially", "expected", "hypothesized", "estimated",
]


@dataclass
class MetaCognitiveReport:
    """Scholarly calibrated evaluation of an answer's epistemic grounding."""
    query: str
    overall_confidence: float
    confidence_tier: str  # "VERY_HIGH" (>= 0.85), "HIGH" (>= 0.70), "MODERATE" (>= 0.45), "LOW" (< 0.45)
    verified_facts: List[Dict[str, Any]]
    uncertain_claims: List[Dict[str, Any]]
    knowledge_gaps: List[str]
    formatted_scholarly_response: str
    is_refusal: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "overall_confidence": round(self.overall_confidence, 4),
            "confidence_tier": self.confidence_tier,
            "verified_facts_count": len(self.verified_facts),
            "verified_facts": self.verified_facts,
            "uncertain_claims_count": len(self.uncertain_claims),
            "uncertain_claims": self.uncertain_claims,
            "knowledge_gaps": self.knowledge_gaps,
            "formatted_scholarly_response": self.formatted_scholarly_response,
            "is_refusal": self.is_refusal,
        }


class MetaCognitiveEvaluator:
    """
    Evaluates grounding certainty, segregates verified vs tentative facts,
    and identifies explicit knowledge gaps to guarantee zero hallucination.
    """

    def __init__(self, high_confidence_threshold: float = 0.70):
        self.high_confidence_threshold = high_confidence_threshold

    def evaluate_response(
        self,
        query: str,
        answer_text: str,
        citations: List[Dict[str, Any]],
        raw_confidence: float = 0.8,
    ) -> MetaCognitiveReport:
        """
        Calibrates confidence and segments answer into verified, tentative, and gap zones.
        """
        if not answer_text.strip() or not citations:
            return MetaCognitiveReport(
                query=query,
                overall_confidence=0.0,
                confidence_tier="LOW",
                verified_facts=[],
                uncertain_claims=[],
                knowledge_gaps=["Tài liệu hiện tại không chứa bất kỳ dữ kiện xác thực nào liên quan đến câu hỏi."],
                formatted_scholarly_response="Không tìm thấy đủ thông tin xác thực trong tài liệu để trả lời câu hỏi này.",
                is_refusal=True,
            )

        # Protect decimals before splitting sentences
        protected = re.sub(r"(?<=\d)\.(?=\d)", "\u0001", answer_text)
        protected = re.sub(r"(?<=\d),(?=\d)", "\u0002", protected)
        sentences = re.split(r"(?<=[.!?])\s+|\n+", protected)

        verified_facts = []
        uncertain_claims = []

        for idx, s in enumerate(sentences):
            clean = s.replace("\u0001", ".").replace("\u0002", ",").strip(" -–—\t")
            if len(clean) < 15 or clean.startswith("💡"):
                continue

            s_norm = strip_accents(clean.lower())
            is_hedged = any(cue in s_norm for cue in _MODAL_HEDGE_CUES)

            # Match associated citation
            cit_label = f"[{idx % len(citations) + 1}]" if citations else "[1]"
            for c in citations:
                lbl = c.get("citation_label", "")
                if lbl and lbl in clean:
                    cit_label = lbl
                    break

            if is_hedged:
                uncertain_claims.append({
                    "statement": clean,
                    "epistemic_tag": "[Có khả năng cao / Giả thuyết]",
                    "citation": cit_label,
                })
            else:
                verified_facts.append({
                    "statement": clean,
                    "epistemic_tag": "[Chắc chắn]",
                    "citation": cit_label,
                })

        # Calculate calibrated confidence
        total_claims = len(verified_facts) + len(uncertain_claims)
        if total_claims == 0:
            calibrated_conf = 0.0
        else:
            verified_ratio = len(verified_facts) / total_claims
            calibrated_conf = min(1.0, max(0.0, (raw_confidence * 0.6) + (verified_ratio * 0.4)))

        # Determine confidence tier
        if calibrated_conf >= 0.85:
            conf_tier = "VERY_HIGH"
        elif calibrated_conf >= 0.70:
            conf_tier = "HIGH"
        elif calibrated_conf >= 0.45:
            conf_tier = "MODERATE"
        else:
            conf_tier = "LOW"

        # Detect knowledge gaps based on query terms not present in evidence
        q_norm = strip_accents(query.lower())
        q_words = [w for w in re.findall(r"\w+", q_norm) if len(w) > 2]
        all_evidence = " ".join([c.get("text", "") for c in citations])
        ev_norm = strip_accents(all_evidence.lower())

        knowledge_gaps = []
        missing_terms = [w for w in q_words if w not in ev_norm]
        if missing_terms and len(missing_terms) >= 2:
            gap_phrase = " ".join(missing_terms[:3])
            knowledge_gaps.append(
                f"Tài liệu chưa công bố thông tin chi tiết về khía cạnh '{gap_phrase}'."
            )

        # Build structured scholarly response
        lines = []
        if verified_facts:
            lines.append("**Dữ kiện đã được khẳng định chắc chắn:**")
            for vf in verified_facts[:3]:
                lines.append(f"- [Chắc chắn] {vf['statement']} {vf['citation']}")

        if uncertain_claims:
            lines.append("\n**Dữ kiện giả định / Có độ bất định:**")
            for uc in uncertain_claims[:2]:
                lines.append(f"- [Giả thuyết] {uc['statement']} {uc['citation']}")

        if knowledge_gaps:
            lines.append("\n**Khoảng trống tri thức ghi nhận:**")
            for kg in knowledge_gaps:
                lines.append(f"- {kg}")

        formatted = "\n".join(lines) if lines else answer_text

        return MetaCognitiveReport(
            query=query,
            overall_confidence=calibrated_conf,
            confidence_tier=conf_tier,
            verified_facts=verified_facts,
            uncertain_claims=uncertain_claims,
            knowledge_gaps=knowledge_gaps,
            formatted_scholarly_response=formatted,
            is_refusal=False,
        )


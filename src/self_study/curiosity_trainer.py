"""
Curiosity-Driven Active Ingestion (Ticket 7.5).
Evaluates information surprise, syntactic complexity, and numerical density
to prioritize hard QA pairs for SA-CMS Level 2 & 3 memory updates.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
import re
from src.self_study.types import SelfStudyQAPair, QARuleType


@dataclass
class CuriosityScore:
    """Evaluated curiosity and hardness metric for a single QA pair."""
    pair: Any  # SelfStudyQAPair or duck-typed QA pair
    curiosity_score: float  # [0.0, 1.0]
    difficulty_tier: str    # "hard", "medium", "simple"
    hard_factors: List[str] = field(default_factory=list)


@dataclass
class CuriosityIngestionReport:
    """Report describing curated curriculum for memory assimilation."""
    total_evaluated: int
    high_curiosity_count: int
    curated_pairs: List[Any] = field(default_factory=list)
    step_weights: List[float] = field(default_factory=list)
    difficulty_breakdown: Dict[str, int] = field(default_factory=dict)
    mean_curiosity_score: float = 0.0


class CuriosityDrivenIngestionEngine:
    """
    Curates and prioritizes self-study QA pairs based on model epistemic curiosity.
    """

    def score_curiosity(self, pairs: List[Any]) -> List[CuriosityScore]:
        """Scores each QA pair on information density and difficulty."""
        results = []
        for p in pairs:
            q_text = getattr(p, "question", "").lower()
            a_text = getattr(p, "answer", "").lower()
            rule_name = str(getattr(p, "rule_name", getattr(p, "rule_type", ""))).lower()

            factors = []
            score = 0.30  # Baseline curiosity

            # Multi-clause or conditional question
            if any(k in q_text for k in ["và trong điều kiện", "tại sao", "như thế nào", "khác gì", "so với"]):
                score += 0.25
                factors.append("multi_clause_reasoning")

            # Numerical or metric precision
            if re.search(r"\d+([.,]\d+)?\s*%", a_text) or re.search(r"o\([n\^0-9]+\)", a_text):
                score += 0.25
                factors.append("quantitative_precision")

            # Trade-off or boundary mention
            if any(k in a_text for k in ["tuy nhiên", "nhưng", "đòi hỏi", "chi phí", "trong điều kiện"]):
                score += 0.15
                factors.append("boundary_tradeoff")

            # Rule-based difficulty boost
            if "cause" in rule_name or "metric" in rule_name or "comparison" in rule_name:
                score += 0.10
                factors.append("high_order_intent")
            elif "definition" in rule_name:
                score -= 0.10  # Simple definition is easiest

            final_score = round(max(0.1, min(1.0, score)), 3)

            tier = "hard" if final_score >= 0.70 else ("medium" if final_score >= 0.45 else "simple")
            results.append(
                CuriosityScore(
                    pair=p,
                    curiosity_score=final_score,
                    difficulty_tier=tier,
                    hard_factors=factors,
                )
            )

        return results

    def curate_curriculum(self, pairs: List[Any], max_budget: int = 15) -> CuriosityIngestionReport:
        """Curates a prioritized curriculum within a maximum gradient update budget."""
        if not pairs:
            return CuriosityIngestionReport(
                total_evaluated=0,
                high_curiosity_count=0,
                curated_pairs=[],
                step_weights=[],
                difficulty_breakdown={"hard": 0, "medium": 0, "simple": 0},
                mean_curiosity_score=0.0,
            )

        scored = self.score_curiosity(pairs)
        # Sort descending by curiosity score
        scored.sort(key=lambda s: s.curiosity_score, reverse=True)

        selected = scored[:max_budget]
        curated_pairs = [s.pair for s in selected]

        # Allocate step weights: hard get 2.0x, medium 1.5x, simple 1.0x
        weights = []
        for s in selected:
            if s.difficulty_tier == "hard":
                weights.append(2.0)
            elif s.difficulty_tier == "medium":
                weights.append(1.5)
            else:
                weights.append(1.0)

        breakdown = {"hard": 0, "medium": 0, "simple": 0}
        for s in selected:
            breakdown[s.difficulty_tier] = breakdown.get(s.difficulty_tier, 0) + 1

        mean_curiosity = sum(s.curiosity_score for s in selected) / len(selected) if selected else 0.0

        return CuriosityIngestionReport(
            total_evaluated=len(pairs),
            high_curiosity_count=breakdown.get("hard", 0),
            curated_pairs=curated_pairs,
            step_weights=weights,
            difficulty_breakdown=breakdown,
            mean_curiosity_score=round(mean_curiosity, 3),
        )

    def get_gradient_step_weights(self, pairs: List[Any]) -> List[float]:
        """Returns step weights for a list of QA pairs according to hardness."""
        scored = self.score_curiosity(pairs)
        weights = []
        for s in scored:
            if s.curiosity_score >= 0.70:
                weights.append(2.0)
            elif s.curiosity_score >= 0.45:
                weights.append(1.5)
            else:
                weights.append(1.0)
        return weights

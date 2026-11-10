"""
Epistemic Uncertainty & Bayesian Confidence Calibrator (Ticket 7.4).
Computes mathematically grounded Epistemic Uncertainty U_epi using:
1. Gating Shannon Entropy H(g) from SA-CMS Level 1/2/3 activations (Eq. 74).
2. Evidence Density D_ev across retrieved passages.
3. Semantic Drift S_drift between query keywords and evidence vocabulary.
Outputs calibrated epistemic labels ([Chắc chắn] / [Có khả năng cao] / [Giả thuyết]).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
import math
import re


@dataclass
class CalibratedEpistemicReport:
    """Detailed mathematical calibration report."""
    uncertainty_score: float  # [0.0, 1.0], where 0 is absolute certainty
    gating_entropy: float     # Normalized Shannon entropy [0.0, 1.0]
    semantic_drift: float     # [0.0, 1.0]
    evidence_density: float   # [0.0, 1.0]
    epistemic_label: str      # "[Chắc chắn]", "[Có khả năng cao]", "[Giả thuyết]"
    calibration_explanation: str
    dominant_level: str = ""


class EpistemicCalibrator:
    """
    Evaluates epistemic uncertainty and calibrates confidence labels.
    """

    LOG2_3 = math.log2(3.0)  # Maximum Shannon entropy for 3 discrete states

    def calculate_gating_entropy(self, g1: float, g2: float, g3: float) -> float:
        """
        Calculates normalized Shannon entropy H(g) / log2(3).
        g_i are positive weights summing to ~1.0.
        """
        total = max(1e-9, g1 + g2 + g3)
        p1, p2, p3 = g1 / total, g2 / total, g3 / total

        eps = 1e-12
        h = -(
            (p1 * math.log2(p1 + eps)) +
            (p2 * math.log2(p2 + eps)) +
            (p3 * math.log2(p3 + eps))
        )
        normalized = max(0.0, min(1.0, h / self.LOG2_3))
        return round(normalized, 4)

    QUESTION_STOPWORDS = {
        "bao", "nhiêu", "như", "thế", "nào", "tại", "sao", "gì", "cái",
        "không", "được", "phải", "chăng", "những", "các", "cho", "biết",
        "với", "trong", "của", "và", "là", "có", "nào", "mấy",
    }

    def calculate_semantic_drift(self, query: str, evidence_texts: List[str]) -> float:
        """Calculates 1 - keyword overlap similarity between query and combined evidence."""
        if not evidence_texts:
            return 1.0

        raw_q_words = set(re.findall(r"\b[a-zA-Zà-ỹÀ-Ỹ0-9]{3,}\b", query.lower()))
        q_words = raw_q_words - self.QUESTION_STOPWORDS
        if not q_words:
            q_words = raw_q_words or {"truy_van"}

        combined_text = " ".join(evidence_texts).lower()
        ev_words = set(re.findall(r"\b[a-zA-Zà-ỹÀ-Ỹ0-9]{3,}\b", combined_text))

        overlap = len(q_words & ev_words)
        coverage = overlap / len(q_words)
        drift = max(0.0, min(1.0, 1.0 - coverage))
        return round(drift, 4)

    def calculate_evidence_density(self, evidence_texts: List[str]) -> float:
        """Calculates evidence density based on number and substance of supporting passages."""
        if not evidence_texts:
            return 0.0
        # Saturation curve: 1 text -> 0.7, 2 texts -> 0.9, 3+ texts -> 1.0
        count = len(evidence_texts)
        if count >= 3:
            return 1.0
        elif count == 2:
            return 0.90
        elif count == 1:
            return 0.70
        return 0.0

    def calibrate(
        self,
        query: str,
        evidence_texts: List[str],
        gating_weights: Tuple[float, float, float] = (0.33, 0.33, 0.34),
    ) -> CalibratedEpistemicReport:
        """
        Performs full calibration combining entropy, drift, and density.
        """
        if not evidence_texts:
            return CalibratedEpistemicReport(
                uncertainty_score=1.0,
                gating_entropy=1.0,
                semantic_drift=1.0,
                evidence_density=0.0,
                epistemic_label="[Giả thuyết]",
                calibration_explanation="Không có bằng chứng thực chứng trong tài liệu (Độ bất định tối đa).",
                dominant_level="Không xác định",
            )

        g1, g2, g3 = gating_weights
        h_norm = self.calculate_gating_entropy(g1, g2, g3)
        s_drift = self.calculate_semantic_drift(query, evidence_texts)
        d_ev = self.calculate_evidence_density(evidence_texts)

        # Total epistemic uncertainty:
        # Weighted combination: 40% semantic drift + 35% gating entropy + 25% evidence sparsity
        u_epi = (0.40 * s_drift) + (0.35 * h_norm) + (0.25 * (1.0 - d_ev))
        u_epi = round(max(0.0, min(1.0, u_epi)), 4)

        # Determine dominant level
        levels = [("Level 1 (Từ vựng)", g1), ("Level 2 (Khái niệm)", g2), ("Level 3 (Toàn văn)", g3)]
        levels.sort(key=lambda x: x[1], reverse=True)
        dominant_level = levels[0][0]

        # Epistemic labels with well-calibrated scientific thresholds
        if u_epi <= 0.40:
            label = "[Chắc chắn]"
            explanation = (
                f"Độ tin cậy chắc chắn với độ bất định thấp ({u_epi:.2f}). Bằng chứng đối chiếu trực tiếp từ tài liệu với "
                f"trọng số cổng kích hoạt tập trung tại {dominant_level}."
            )
        elif u_epi <= 0.68:
            label = "[Có khả năng cao]"
            explanation = (
                f"Độ tin cậy có khả năng cao với độ bất định trung bình ({u_epi:.2f}). Bằng chứng có độ tương đồng ngữ nghĩa tốt "
                f"nhưng phân bổ kích hoạt cổng nhớ có độ phân tán nhất định."
            )
        else:
            label = "[Giả thuyết]"
            explanation = (
                f"Độ bất định cao ({u_epi:.2f}), bằng chứng thưa thớt hoặc độ lệch ngữ nghĩa lớn, "
                f"cần thêm dữ liệu xác thực độc lập."
            )

        return CalibratedEpistemicReport(
            uncertainty_score=u_epi,
            gating_entropy=h_norm,
            semantic_drift=s_drift,
            evidence_density=d_ev,
            epistemic_label=label,
            calibration_explanation=explanation,
            dominant_level=dominant_level,
        )

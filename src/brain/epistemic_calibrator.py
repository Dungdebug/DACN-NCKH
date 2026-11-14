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
import unicodedata


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive canonical alignment."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


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
    evidence_quality: float = 0.0
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    claim_support: float = 1.0


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

    FOLDED_QUESTION_STOPWORDS = {
        "bao", "nhieu", "nhu", "the", "nao", "tai", "sao", "gi", "cai",
        "khong", "duoc", "phai", "chang", "nhung", "cac", "cho", "biet",
        "voi", "trong", "cua", "va", "la", "co", "nao", "may",
    }

    POLARITY_DIMENSIONS = [
        {
            "dim": "tang_giam",
            "name": "tăng / giảm",
            "pos_cues": ["tang", "gia tang", "tang truong", "tang toc"],
            "neg_cues": ["giam", "suy giam", "ha thap", "tut giam"],
        },
        {
            "dim": "cao_thap",
            "name": "cao / thấp",
            "pos_cues": ["cao", "toi da", "lon hon", "vuot troi"],
            "neg_cues": ["thap", "toi thieu", "nho hon", "kem"],
        },
        {
            "dim": "co_khong",
            "name": "có / không",
            "pos_cues": ["co", "ton tai", "xac nhan", "xuat hien"],
            "neg_cues": ["khong", "khong co", "chua co", "chua tung", "khong the"],
        },
        {
            "dim": "cai_thien_suy_giam",
            "name": "cải thiện / suy giảm",
            "pos_cues": ["cai thien", "nang cao", "tot hon", "toi uu hoa", "toi uu"],
            "neg_cues": ["suy giam", "kem di", "te hon", "suy thoai", "suy yeu"],
        },
        {
            "dim": "nhanh_cham",
            "name": "nhanh / chậm",
            "pos_cues": ["nhanh", "toc do cao", "tuc thi", "tuc thoi"],
            "neg_cues": ["cham", "tri hoan", "tre han", "keo dai"],
        },
    ]

    _POLARITY_CUE_WORDS = {
        "tang", "giam", "cao", "thap", "co", "khong", "chua", "cai",
        "thien", "suy", "nhanh", "cham", "ha", "nho", "lon", "tot",
        "kem", "toi", "da", "thieu", "te", "hon", "ton", "tai", "tri",
        "hoan", "tre", "keo", "dai", "nang", "uu", "hoa",
    }

    _SYNTACTIC_STOPWORDS = {
        "thi", "la", "ma", "do", "bi", "duoc", "boi", "rat", "qua", "hon",
        "nhieu", "it", "khi", "neu", "tuy", "nhien", "nhung", "dong", "thoi",
        "dac", "biet", "mot", "cac", "nhung", "nay", "do", "kia", "tai",
    }

    def calculate_evidence_quality(self, query: str, evidence_texts: List[str]) -> float:
        """
        Calculates evidence quality:
        mean per-passage accent-folded query coverage × source-diversity factor (unique passages / total), saturating.
        """
        if not evidence_texts:
            return 0.0

        clean_passages = [p.strip() for p in evidence_texts if p and p.strip()]
        if not clean_passages:
            return 0.0

        q_folded = strip_accents(query.lower())
        raw_q_words = re.findall(r"\b[a-z0-9]{2,}\b", q_folded)
        q_words = set(w for w in raw_q_words if w not in self.FOLDED_QUESTION_STOPWORDS)
        if not q_words:
            q_words = set(raw_q_words)
        if not q_words:
            return 0.0

        coverages = []
        for passage in clean_passages:
            p_folded = strip_accents(passage.lower())
            p_words = set(re.findall(r"\b[a-z0-9]{2,}\b", p_folded))
            overlap = len(q_words & p_words)
            cov = overlap / len(q_words)
            coverages.append(cov)

        mean_coverage = sum(coverages) / len(coverages)

        # Source diversity factor
        unique_passages = len(set(clean_passages))
        total_passages = len(evidence_texts)
        diversity_ratio = unique_passages / max(1, total_passages)

        # Saturation curve based on distinct supporting passages
        if unique_passages >= 3:
            sat = 1.0
        elif unique_passages == 2:
            sat = 0.90
        elif unique_passages == 1:
            sat = 0.70
        else:
            sat = 0.0

        diversity_factor = diversity_ratio * sat
        quality = mean_coverage * diversity_factor
        return round(max(0.0, min(1.0, quality)), 4)

    def _determine_sentence_polarity(self, sent_folded: str, dim_spec: Dict[str, Any]) -> Optional[str]:
        """Detects whether a sentence exhibits positive or negative polarity along a dimension."""
        dim_id = dim_spec["dim"]

        # Negation check on positive cues e.g. "khong tang", "khong nhanh"
        if dim_id != "co_khong":
            for pos in dim_spec["pos_cues"]:
                if re.search(r"\b(khong|chua)\s+" + re.escape(pos) + r"\b", sent_folded):
                    return "neg"

        # Check negative cues
        for neg in dim_spec["neg_cues"]:
            if re.search(r"\b" + re.escape(neg) + r"\b", sent_folded):
                return "neg"

        # Check positive cues
        for pos in dim_spec["pos_cues"]:
            if re.search(r"\b" + re.escape(pos) + r"\b", sent_folded):
                return "pos"

        return None

    def detect_polarity_conflicts(self, evidence_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Detects conflicting claims across evidence passages:
        pairs of sentences that share ≥2 content keywords but have opposite polarity
        (tăng/giảm, cao/thấp, có/không, cải thiện/suy giảm, nhanh/chậm).
        """
        if not evidence_texts:
            return []

        sentences: List[str] = []
        for passage in evidence_texts:
            if not passage or not passage.strip():
                continue
            raw_sents = re.split(r"[.!?;\n]+", passage)
            for s in raw_sents:
                clean_s = s.strip()
                if len(clean_s.split()) >= 3:
                    sentences.append(clean_s)

        # Deduplicate identical sentences
        unique_sents = list(dict.fromkeys(sentences))
        n = len(unique_sents)
        if n < 2:
            return []

        # Pre-process folded text, content words, and polarities
        sent_info = []
        for s in unique_sents:
            s_folded = strip_accents(s.lower())
            words = set(re.findall(r"\b[a-z0-9]{2,}\b", s_folded))
            content_words = words - self.FOLDED_QUESTION_STOPWORDS - self._POLARITY_CUE_WORDS - self._SYNTACTIC_STOPWORDS

            polarities = {}
            for dim in self.POLARITY_DIMENSIONS:
                pol = self._determine_sentence_polarity(s_folded, dim)
                if pol:
                    polarities[dim["dim"]] = pol

            sent_info.append({
                "raw": s,
                "folded": s_folded,
                "content_words": content_words,
                "polarities": polarities,
            })

        conflicts = []
        for i in range(n):
            for j in range(i + 1, n):
                info_a = sent_info[i]
                info_b = sent_info[j]

                shared_kws = info_a["content_words"] & info_b["content_words"]
                if len(shared_kws) < 2:
                    continue

                # Check if opposite polarity exists along any dimension
                for dim in self.POLARITY_DIMENSIONS:
                    dim_key = dim["dim"]
                    pol_a = info_a["polarities"].get(dim_key)
                    pol_b = info_b["polarities"].get(dim_key)

                    if pol_a and pol_b and pol_a != pol_b:
                        conflicts.append({
                            "sentence_a": info_a["raw"],
                            "sentence_b": info_b["raw"],
                            "polarity_dimension": dim["name"],
                            "polarity_a": pol_a,
                            "polarity_b": pol_b,
                            "shared_keywords": sorted(list(shared_kws)),
                        })
                        break

        return conflicts

    def calculate_folded_semantic_drift(self, query: str, evidence_texts: List[str]) -> float:
        """
        Calculates 1 - keyword overlap similarity between query and combined evidence,
        using accent-folded unigrams and bigrams.
        """
        if not evidence_texts:
            return 1.0

        q_folded = strip_accents(query.lower())
        raw_q_words = re.findall(r"\b[a-z0-9]{2,}\b", q_folded)
        q_words = [w for w in raw_q_words if w not in self.FOLDED_QUESTION_STOPWORDS]
        if not q_words:
            q_words = raw_q_words
        if not q_words:
            return 1.0

        q_items = set(q_words)
        if len(q_words) >= 2:
            for i in range(len(q_words) - 1):
                q_items.add(f"{q_words[i]} {q_words[i+1]}")

        combined_text = strip_accents(" ".join(evidence_texts).lower())
        ev_words = re.findall(r"\b[a-z0-9]{2,}\b", combined_text)
        ev_items = set(ev_words)
        if len(ev_words) >= 2:
            for i in range(len(ev_words) - 1):
                ev_items.add(f"{ev_words[i]} {ev_words[i+1]}")

        overlap = len(q_items & ev_items)
        coverage = overlap / len(q_items)
        drift = max(0.0, min(1.0, 1.0 - coverage))
        return round(drift, 4)

    def calibrate_deep(
        self,
        query: str,
        evidence_texts: List[str],
        gating_weights: Tuple[float, float, float] = (0.33, 0.33, 0.34),
        answer_text: Optional[str] = None,
    ) -> CalibratedEpistemicReport:
        """
        Performs deep epistemic calibration incorporating:
        1. Folded unigram + bigram semantic drift.
        2. Normalized gating entropy H(g).
        3. Accent-folded evidence quality with source diversity.
        4. Polarity conflict penalty across evidence passages.
        5. Answer claim support verification against evidence.
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
                evidence_quality=0.0,
                conflicts=[],
                claim_support=0.0 if answer_text else 1.0,
            )

        g1, g2, g3 = gating_weights
        h_norm = self.calculate_gating_entropy(g1, g2, g3)
        s_drift = self.calculate_folded_semantic_drift(query, evidence_texts)
        d_ev = self.calculate_evidence_density(evidence_texts)
        quality = self.calculate_evidence_quality(query, evidence_texts)
        conflicts = self.detect_polarity_conflicts(evidence_texts)
        conflict_term = min(1.0, len(conflicts) / 2.0)

        # Base uncertainty calculation:
        # u = 0.30·drift + 0.25·H + 0.25·(1−quality) + 0.20·min(1, conflicts/2)
        u_base = (0.30 * s_drift) + (0.25 * h_norm) + (0.25 * (1.0 - quality)) + (0.20 * conflict_term)
        u_epi = round(max(0.0, min(1.0, u_base)), 4)

        # Claim support verification if answer_text is provided
        claim_support = 1.0
        downgraded = False
        if answer_text is not None and answer_text.strip():
            ans_sents = [s.strip() for s in re.split(r"[.!?;\n]+", answer_text) if s.strip()]
            if ans_sents:
                ev_folded = strip_accents(" ".join(evidence_texts).lower())
                ev_tokens = set(re.findall(r"\b[a-z0-9]{2,}\b", ev_folded))

                supported_count = 0
                valid_count = 0
                for sent in ans_sents:
                    sent_folded = strip_accents(sent.lower())
                    sent_words = [w for w in re.findall(r"\b[a-z0-9]{2,}\b", sent_folded) if w not in self.FOLDED_QUESTION_STOPWORDS]
                    if not sent_words:
                        sent_words = re.findall(r"\b[a-z0-9]{2,}\b", sent_folded)
                    if not sent_words:
                        continue
                    valid_count += 1
                    sent_set = set(sent_words)
                    cov = len(sent_set & ev_tokens) / len(sent_set)
                    if cov >= 0.50:
                        supported_count += 1

                if valid_count > 0:
                    claim_support = round(supported_count / valid_count, 4)
                else:
                    claim_support = 1.0

            # If claim support < 0.50, downgrade to [Giả thuyết]
            if claim_support < 0.50:
                downgraded = True
                u_epi = max(u_epi, 0.70)

        # Determine dominant level
        levels = [("Level 1 (Từ vựng)", g1), ("Level 2 (Khái niệm)", g2), ("Level 3 (Toàn văn)", g3)]
        levels.sort(key=lambda x: x[1], reverse=True)
        dominant_level = levels[0][0]

        # Epistemic labels with well-calibrated scientific thresholds
        if u_epi <= 0.40 and not downgraded:
            label = "[Chắc chắn]"
            explanation = (
                f"Độ tin cậy chắc chắn với độ bất định thấp ({u_epi:.2f}). Bằng chứng chất lượng cao "
                f"(quality={quality:.2f}) với trọng số cổng kích hoạt tập trung tại {dominant_level}."
            )
        elif u_epi <= 0.68 and not downgraded:
            label = "[Có khả năng cao]"
            explanation = (
                f"Độ tin cậy có khả năng cao với độ bất định trung bình ({u_epi:.2f}). Bằng chứng có độ bao phủ tốt "
                f"nhưng có thể phân tán hoặc có độ bất định cổng nhớ."
            )
            if conflicts:
                explanation += f" Lưu ý: phát hiện {len(conflicts)} xung đột phân cực trong bằng chứng."
        else:
            label = "[Giả thuyết]"
            if downgraded:
                explanation = (
                    f"Độ bất định cao ({u_epi:.2f}). Câu trả lời thiếu hỗ trợ từ bằng chứng thực chứng "
                    f"(claim_support={claim_support*100:.1f}% < 50%), bị hạ bậc thành [Giả thuyết]."
                )
            elif conflicts:
                explanation = (
                    f"Độ bất định cao ({u_epi:.2f}) do phát hiện {len(conflicts)} xung đột phân cực "
                    f"trong tài liệu đối chiếu."
                )
            else:
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
            evidence_quality=quality,
            conflicts=conflicts,
            claim_support=claim_support,
        )

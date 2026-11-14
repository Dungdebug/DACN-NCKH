"""
Defense Strategist for Sovereign Document Brain.
Classifies defense archetypes, plans multi-drawer argumentative structures,
and synthesizes evidence-backed defense arguments for academic thesis defense.
100% Offline, Deterministic, Zero LLM overhead.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Union
import math
import re

from src.brain.section_drawer import (
    SectionDrawer,
    DRAWER_PROBLEM,
    DRAWER_SOLUTION,
    DRAWER_EXPERIMENT,
    DRAWER_CONCLUSION,
    DRAWER_GENERIC,
    strip_accents,
)


# Defense Archetype Constants
ARCHETYPE_NOVELTY = "NOVELTY"
ARCHETYPE_JUSTIFICATION = "JUSTIFICATION"
ARCHETYPE_VALIDITY = "VALIDITY"
ARCHETYPE_LIMITATION = "LIMITATION"
ARCHETYPE_SCALABILITY = "SCALABILITY"
ARCHETYPE_REPRODUCIBILITY = "REPRODUCIBILITY"
ARCHETYPE_FUTURE_WORK = "FUTURE_WORK"
ARCHETYPE_METRIC_DEFENSE = "METRIC_DEFENSE"
ARCHETYPE_GENERIC = "GENERIC"

ALL_ARCHETYPES = [
    ARCHETYPE_NOVELTY,
    ARCHETYPE_JUSTIFICATION,
    ARCHETYPE_VALIDITY,
    ARCHETYPE_LIMITATION,
    ARCHETYPE_SCALABILITY,
    ARCHETYPE_REPRODUCIBILITY,
    ARCHETYPE_FUTURE_WORK,
    ARCHETYPE_METRIC_DEFENSE,
    ARCHETYPE_GENERIC,
]


@dataclass
class DefensePlan:
    archetype: str
    confidence: float
    target_drawers: List[str] = field(default_factory=list)
    argument_slots: List[str] = field(default_factory=list)
    expansion_terms: List[str] = field(default_factory=list)


# Accent-folded weighted regex cues for defense classification
ARCHETYPE_CUES: Dict[str, List[Tuple[str, float]]] = {
    ARCHETYPE_NOVELTY: [
        (r"\b(?:tinh|diem)\s+moi\b", 3.5),
        (r"\bdong\s+gop\s+(?:moi|chinh|khoa\s+hoc)\b", 3.5),
        (r"\bkhac\s+biet\s+so\s+voi\s+(?:nghien\s+cuu\s+truoc|cac\s+phuong\s+phap)\b", 3.5),
        (r"\bnovelty\b", 3.0),
        (r"\bcontribution\b", 2.5),
        (r"\bsang\s+tao\b", 2.0),
        (r"\bdot\s+pha\b", 2.0),
    ],
    ARCHETYPE_JUSTIFICATION: [
        (r"\b(?:tai\s+sao|vi\s+sao)\s+(?:lai\s+)?(?:chon|dung|su\s+dung)\b", 3.5),
        (r"\bma\s+khong\s+(?:phai|dung)\b", 3.0),
        (r"\bthay\s+vi\b", 2.5),
        (r"\bly\s+do\s+(?:chon|su\s+dung)\b", 3.0),
        (r"\bjustification\b", 3.0),
        (r"\bwhy\s+(?:choose|use)\b", 2.5),
        (r"\bcan\s+cu\s+lua\s+chon\b", 2.5),
    ],
    ARCHETYPE_VALIDITY: [
        (r"\bbaseline\b", 2.5),
        (r"\btap\s+du\s+lieu\b", 2.0),
        (r"\bdu\s+lieu\s+huan\s+luyen\b", 2.0),
        (r"\bkhach\s+quan\b", 3.0),
        (r"\bdo\s+tin\s+cay\b", 3.0),
        (r"\btinh\s+hop\s+le\b", 3.0),
        (r"\bbenchmark\b", 2.0),
        (r"\bdataset\b", 2.0),
        (r"\boverfit(?:ting)?\b", 2.5),
        (r"\bvalidity\b", 3.0),
        (r"\bkiem\s+dinh\b", 2.5),
    ],
    ARCHETYPE_LIMITATION: [
        (r"\bhan\s+che\b", 3.5),
        (r"\bdiem\s+yeu\b", 3.0),
        (r"\bnhuoc\s+diem\b", 3.0),
        (r"\bthieu\s+sot\b", 2.5),
        (r"\brui\s+ro\b", 2.0),
        (r"\blimitation\b", 3.0),
        (r"\bweakness\b", 2.5),
        (r"\bdrawback\b", 2.5),
        (r"\bchua\s+lam\s+duoc\b", 2.5),
    ],
    ARCHETYPE_SCALABILITY: [
        (r"\b(?:kha\s+nang\s+)?mo\s+rong\b", 3.5),
        (r"\btrien\s+khai\s+thuc\s+te\b", 3.5),
        (r"\bscale\b", 2.0),
        (r"\bscalability\b", 3.0),
        (r"\bnang\s+tai\b", 2.5),
        (r"\bproduction\b", 2.5),
        (r"\bdeploy\b", 2.5),
        (r"\bthroughput\b", 2.0),
        (r"\bvram\b", 1.5),
        (r"\bgpu\b", 1.5),
    ],
    ARCHETYPE_REPRODUCIBILITY: [
        (r"\btai\s+(?:lap|hien)\b", 3.5),
        (r"\breproducib(?:ility|le)\b", 3.0),
        (r"\bma\s+nguon\b", 2.5),
        (r"\bsource\s+code\b", 2.5),
        (r"\bhyperparameter\b", 2.5),
        (r"\bsieu\s+tham\s+so\b", 2.5),
        (r"\bseed\b", 2.0),
        (r"\bcai\s+dat\b", 1.5),
    ],
    ARCHETYPE_FUTURE_WORK: [
        (r"\btuong\s+lai\b", 2.5),
        (r"\bhuong\s+phat\s+trien\b", 3.5),
        (r"\bphat\s+trien\s+tiep\s+theo\b", 3.5),
        (r"\bfuture\s+work\b", 3.0),
        (r"\bnghien\s+cuu\s+tiep\b", 2.5),
        (r"\bgiai\s+doan\s+tiep\s+theo\b", 2.5),
    ],
    ARCHETYPE_METRIC_DEFENSE: [
        (r"\bchi\s+so\b", 2.0),
        (r"\bso\s+lieu\b", 2.0),
        (r"\bdo\s+chinh\s+xac\b", 2.0),
        (r"\bf1(?:\s+score)?\b", 2.5),
        (r"\bdo\s+tre\b", 2.0),
        (r"\blatency\b", 2.0),
        (r"\btiet\s+kiem\s+token\b", 2.5),
        (r"\bdanh\s+doi\b", 2.5),
        (r"\btrade-?off\b", 2.5),
        (r"\bp-value\b", 2.5),
    ],
}

ARCHETYPE_DRAWERS: Dict[str, List[str]] = {
    ARCHETYPE_JUSTIFICATION: [DRAWER_SOLUTION, DRAWER_EXPERIMENT, DRAWER_CONCLUSION],
    ARCHETYPE_LIMITATION: [DRAWER_CONCLUSION, DRAWER_EXPERIMENT],
    ARCHETYPE_NOVELTY: [DRAWER_PROBLEM, DRAWER_SOLUTION],
    ARCHETYPE_VALIDITY: [DRAWER_EXPERIMENT, DRAWER_SOLUTION],
    ARCHETYPE_SCALABILITY: [DRAWER_SOLUTION, DRAWER_EXPERIMENT, DRAWER_CONCLUSION],
    ARCHETYPE_REPRODUCIBILITY: [DRAWER_SOLUTION, DRAWER_EXPERIMENT],
    ARCHETYPE_FUTURE_WORK: [DRAWER_CONCLUSION, DRAWER_SOLUTION],
    ARCHETYPE_METRIC_DEFENSE: [DRAWER_EXPERIMENT, DRAWER_CONCLUSION],
    ARCHETYPE_GENERIC: [DRAWER_PROBLEM, DRAWER_SOLUTION, DRAWER_EXPERIMENT, DRAWER_CONCLUSION],
}

ARCHETYPE_SLOTS: Dict[str, List[str]] = {
    ARCHETYPE_JUSTIFICATION: ["claim", "alternative", "evidence", "trade_off"],
    ARCHETYPE_LIMITATION: ["acknowledge", "evidence", "mitigation"],
    ARCHETYPE_NOVELTY: ["gap", "proposed_novelty", "differentiation", "evidence"],
    ARCHETYPE_VALIDITY: ["claim", "baseline_setup", "empirical_evidence", "robustness"],
    ARCHETYPE_SCALABILITY: ["resource_requirement", "bottleneck_analysis", "empirical_scaling", "deployment_feasibility"],
    ARCHETYPE_REPRODUCIBILITY: ["experimental_setup", "parameters", "availability", "validation"],
    ARCHETYPE_FUTURE_WORK: ["current_scope", "unresolved_challenge", "next_steps"],
    ARCHETYPE_METRIC_DEFENSE: ["metric_selection", "empirical_result", "statistical_significance", "trade_off"],
    ARCHETYPE_GENERIC: ["claim", "evidence", "conclusion"],
}

ARCHETYPE_EXPANSIONS: Dict[str, List[str]] = {
    ARCHETYPE_JUSTIFICATION: ["so sánh", "lý do lựa chọn", "ưu điểm", "đánh đổi", "baseline", "alternative"],
    ARCHETYPE_LIMITATION: ["hạn chế", "điểm yếu", "phạm vi", "khắc phục", "giảm thiểu", "mitigation"],
    ARCHETYPE_NOVELTY: ["tính mới", "đóng góp mới", "khác biệt", "nghiên cứu trước", "novelty", "contribution"],
    ARCHETYPE_VALIDITY: ["tính hợp lệ", "đối chứng", "baseline", "tập kiểm thử", "khách quan", "ground truth"],
    ARCHETYPE_SCALABILITY: ["khả năng mở rộng", "tài nguyên", "độ trễ", "vram", "chi phí tính toán", "throughput"],
    ARCHETYPE_REPRODUCIBILITY: ["tái lập", "siêu tham số", "mã nguồn", "môi trường", "hyperparameters", "seed"],
    ARCHETYPE_FUTURE_WORK: ["hướng phát triển", "tương lai", "nghiên cứu tiếp theo", "mở rộng"],
    ARCHETYPE_METRIC_DEFENSE: ["chỉ số", "độ chính xác", "f1-score", "độ trễ", "tiết kiệm token", "hiệu năng"],
    ARCHETYPE_GENERIC: ["luận điểm", "bằng chứng", "kết luận"],
}

SLOT_TITLES: Dict[str, str] = {
    "claim": "Claim",
    "alternative": "Alternative",
    "evidence": "Evidence",
    "trade_off": "Trade-off",
    "acknowledge": "Acknowledge",
    "mitigation": "Mitigation",
    "gap": "Gap",
    "proposed_novelty": "Novelty",
    "differentiation": "Differentiation",
    "baseline_setup": "Baseline-Setup",
    "empirical_evidence": "Empirical-Evidence",
    "robustness": "Robustness",
    "resource_requirement": "Resource-Requirement",
    "bottleneck_analysis": "Bottleneck-Analysis",
    "empirical_scaling": "Empirical-Scaling",
    "deployment_feasibility": "Deployment-Feasibility",
    "experimental_setup": "Experimental-Setup",
    "parameters": "Parameters",
    "availability": "Availability",
    "validation": "Validation",
    "current_scope": "Current-Scope",
    "unresolved_challenge": "Unresolved-Challenge",
    "next_steps": "Next-Steps",
    "metric_selection": "Metric-Selection",
    "empirical_result": "Empirical-Result",
    "statistical_significance": "Statistical-Significance",
    "conclusion": "Conclusion",
}

SLOT_KEYWORDS: Dict[str, List[str]] = {
    "claim": [
        "de xuat", "giai phap", "muc tieu", "mo hinh", "chung toi", "kien truc",
        "phuong phap", "chung minh", "dat duoc", "khang dinh", "proposal", "method"
    ],
    "alternative": [
        "thay vi", "so voi", "khac voi", "baseline", "truyen thong", "mo hinh cu",
        "tiep can khac", "transformer", "nghien cuu truoc", "alternative", "compared to"
    ],
    "evidence": [
        "ket qua", "thuc nghiem", "do dac", "dat", "tang", "giam", "%", "ms",
        "f1", "chinh xac", "hieu nang", "so lieu", "bang", "figure", "table", "benchmark"
    ],
    "trade_off": [
        "danh doi", "tuy nhien", "bu lai", "mac du", "gioi han",
        "chi phi thoi gian", "thoi gian khoi tao", "thoi gian khoi dong",
        "trade-off", "tradeoff", "downside"
    ],
    "acknowledge": [
        "thua nhan", "han che", "chua xu ly", "chua giai quyet", "diem yeu",
        "nhuoc diem", "thieu sot", "phuc tap", "limitation", "weakness"
    ],
    "mitigation": [
        "giai phap", "khac phuc", "giam thieu", "toi uu", "huong giai quyet",
        "kiem soat", "khuyen nghi", "mitigation", "workaround"
    ],
    "gap": [
        "thach thuc", "kho khan", "khoang trong", "chua duoc nghien cuu", "ton tai",
        "bat cap", "gap", "challenge"
    ],
    "proposed_novelty": [
        "diem moi", "tinh moi", "dong gop moi", "sang tao", "lan dau tien",
        "dong gop chinh", "novelty", "novel contribution"
    ],
    "differentiation": [
        "khac biet", "uu the", "khac voi", "vuot troi", "cai tien",
        "differentiation", "distinction"
    ],
    "baseline_setup": [
        "baseline", "tap du lieu", "tap kiem thu", "dataset", "moi truong",
        "cai dat", "benchmark setup"
    ],
    "empirical_evidence": [
        "ket qua thuc nghiem", "so lieu", "danh gia", "kiem chung", "%", "f1",
        "accuracy", "empirical"
    ],
    "robustness": [
        "on dinh", "ben vung", "khach quan", "tong quat hoa", "khong overfit",
        "robustness", "validity"
    ],
    "resource_requirement": [
        "tai nguyen", "vram", "gpu", "ram", "phan cung", "bo nho",
        "requirement", "memory"
    ],
    "bottleneck_analysis": [
        "diem nghen", "nghen co chai", "gioi han toc do", "do tre", "bottleneck"
    ],
    "empirical_scaling": [
        "mo rong", "tang kich thuoc", "batch size", "scaling", "scale", "throughput"
    ],
    "deployment_feasibility": [
        "trien khai", "thuc te", "production", "san pham", "ung dung", "kha thi",
        "feasibility"
    ],
    "experimental_setup": [
        "thiet lap", "cau hinh", "tham so", "moi truong", "setup", "hardware"
    ],
    "parameters": [
        "sieu tham so", "hyperparameter", "learning rate", "seed", "epoch", "batch"
    ],
    "availability": [
        "ma nguon", "source code", "cong khai", "github", "truy cap", "open source"
    ],
    "validation": [
        "kiem chung", "xac nhan", "tai lap thanh cong", "validation"
    ],
    "current_scope": [
        "pham vi", "gioi han", "hien tai", "tap trung vao", "scope"
    ],
    "unresolved_challenge": [
        "chua giai quyet", "van de con lai", "kho khan", "thach thuc"
    ],
    "next_steps": [
        "tiep theo", "tuong lai", "huong di", "phat trien", "se mo rong", "next step"
    ],
    "metric_selection": [
        "chi so", "lua chon", "do luong", "tieu chi", "metric"
    ],
    "empirical_result": [
        "ket qua", "so lieu", "bang", "bieu do", "%", "ms", "f1"
    ],
    "statistical_significance": [
        "y nghia thong ke", "p-value", "do tin cay", "khoang tin cay", "significance"
    ],
    "conclusion": [
        "ket luan", "tong ket", "khang dinh", "dong gop"
    ],
}

SLOT_DRAWER_BOOST: Dict[str, str] = {
    "claim": DRAWER_SOLUTION,
    "proposed_novelty": DRAWER_SOLUTION,
    "differentiation": DRAWER_SOLUTION,
    "gap": DRAWER_PROBLEM,
    "current_scope": DRAWER_PROBLEM,
    "alternative": DRAWER_SOLUTION,
    "baseline_setup": DRAWER_SOLUTION,
    "experimental_setup": DRAWER_SOLUTION,
    "parameters": DRAWER_SOLUTION,
    "evidence": DRAWER_EXPERIMENT,
    "empirical_evidence": DRAWER_EXPERIMENT,
    "empirical_result": DRAWER_EXPERIMENT,
    "empirical_scaling": DRAWER_EXPERIMENT,
    "statistical_significance": DRAWER_EXPERIMENT,
    "metric_selection": DRAWER_EXPERIMENT,
    "trade_off": DRAWER_CONCLUSION,
    "acknowledge": DRAWER_CONCLUSION,
    "mitigation": DRAWER_CONCLUSION,
    "unresolved_challenge": DRAWER_CONCLUSION,
    "next_steps": DRAWER_CONCLUSION,
    "conclusion": DRAWER_CONCLUSION,
    "deployment_feasibility": DRAWER_SOLUTION,
    "resource_requirement": DRAWER_EXPERIMENT,
    "bottleneck_analysis": DRAWER_EXPERIMENT,
    "robustness": DRAWER_EXPERIMENT,
    "availability": DRAWER_SOLUTION,
    "validation": DRAWER_EXPERIMENT,
}


def classify_defense_archetype(query: str) -> Tuple[str, float]:
    """
    Classifies a defense query into one of 9 defense archetypes using accent-folded
    weighted regex cues, returning (label, confidence) where confidence is the softmax
    margin between the top-1 and top-2 probabilities.
    """
    if not query or not query.strip():
        return ARCHETYPE_GENERIC, 0.0

    q_norm = strip_accents(query.lower()).strip()
    raw_scores: Dict[str, float] = {arch: 0.0 for arch in ALL_ARCHETYPES}

    for arch, cues in ARCHETYPE_CUES.items():
        for pat, weight in cues:
            if re.search(pat, q_norm):
                raw_scores[arch] += weight

    max_v = max(raw_scores.values())
    if max_v <= 0.0:
        return ARCHETYPE_GENERIC, 0.0

    # Softmax calculation
    exp_vals = {arch: math.exp(score - max_v) for arch, score in raw_scores.items()}
    sum_exp = sum(exp_vals.values())
    probs = {arch: val / sum_exp for arch, val in exp_vals.items()}

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top1_label, top1_prob = sorted_probs[0]
    top2_label, top2_prob = sorted_probs[1]

    confidence = round(max(0.0, top1_prob - top2_prob), 4)
    return top1_label, confidence


class DefenseStrategist:
    """
    Synthesizes structured, evidence-backed defense arguments for academic defense panels.
    Partitions evidence across functional drawers and argument slots.
    """

    def __init__(self):
        self.section_drawer = SectionDrawer()

    def classify_defense_archetype(self, query: str) -> Tuple[str, float]:
        """Convenience method delegating to module-level classification."""
        return classify_defense_archetype(query)

    def plan(self, query: str) -> DefensePlan:
        """
        Creates a structured DefensePlan for the user query, mapping the defense
        archetype to target drawers, argument slots, and expansion terms.
        """
        archetype, confidence = classify_defense_archetype(query)
        target_drawers = ARCHETYPE_DRAWERS.get(archetype, ARCHETYPE_DRAWERS[ARCHETYPE_GENERIC])
        argument_slots = ARCHETYPE_SLOTS.get(archetype, ARCHETYPE_SLOTS[ARCHETYPE_GENERIC])
        expansion_terms = ARCHETYPE_EXPANSIONS.get(archetype, ARCHETYPE_EXPANSIONS[ARCHETYPE_GENERIC])

        return DefensePlan(
            archetype=archetype,
            confidence=confidence,
            target_drawers=target_drawers,
            argument_slots=argument_slots,
            expansion_terms=expansion_terms,
        )

    def _format_citation(self, label: Any) -> str:
        lbl = str(label).strip() if label is not None else ""
        if not lbl:
            return ""
        if lbl.startswith("[") and lbl.endswith("]"):
            return lbl
        return f"[{lbl}]"

    def assemble_argument(
        self,
        plan: DefensePlan,
        candidates: List[Union[Dict[str, Any], Any]],
    ) -> str:
        """
        Assigns candidate sentences to their best argument slot using slot keyword cues
        and SectionDrawer drawer categorization, formatting as bold bullet points.
        Never fabricates; skips slots without evidence.
        """
        header = "🎓 **Lập luận bảo vệ:**"
        if not candidates or not plan.argument_slots:
            return header

        # Normalize candidates
        parsed_candidates: List[Dict[str, Any]] = []
        for idx, c in enumerate(candidates):
            if isinstance(c, dict):
                sentence = str(c.get("sentence", "")).strip()
                citation_label = c.get("citation_label", f"{idx + 1}")
                section_title = str(c.get("section_title", "")).strip()
            else:
                sentence = str(getattr(c, "sentence", "")).strip()
                citation_label = getattr(c, "citation_label", f"{idx + 1}")
                section_title = str(getattr(c, "section_title", "")).strip()

            if sentence:
                drawer = self.section_drawer.categorize_section(section_title, text_sample=sentence)
                parsed_candidates.append({
                    "sentence": sentence,
                    "citation_label": self._format_citation(citation_label),
                    "section_title": section_title,
                    "drawer": drawer,
                    "norm": strip_accents(sentence.lower()),
                })

        if not parsed_candidates:
            return header

        # Score candidate-to-slot matches
        assigned_slots: Dict[str, Dict[str, Any]] = {}
        used_candidate_indices: set = set()

        for slot in plan.argument_slots:
            keywords = SLOT_KEYWORDS.get(slot, [])
            preferred_drawer = SLOT_DRAWER_BOOST.get(slot)
            best_cand_idx = -1
            best_score = 0.0

            for c_idx, cand in enumerate(parsed_candidates):
                kw_score = 0.0
                for kw in keywords:
                    if kw in cand["norm"]:
                        kw_score += 1.5

                # Candidate MUST match at least one slot keyword cue to be considered
                if kw_score == 0.0:
                    continue

                drawer_score = 0.0
                if preferred_drawer and cand["drawer"] == preferred_drawer:
                    drawer_score = 1.0

                score = kw_score + drawer_score

                # Penalty if candidate was already assigned to a previous slot
                if c_idx in used_candidate_indices:
                    score *= 0.3

                if score > best_score:
                    best_score = score
                    best_cand_idx = c_idx

            if best_cand_idx != -1 and best_score > 0.0:
                assigned_slots[slot] = parsed_candidates[best_cand_idx]
                used_candidate_indices.add(best_cand_idx)

        # Build output bullets
        bullets: List[str] = []
        for slot in plan.argument_slots:
            if slot in assigned_slots:
                cand = assigned_slots[slot]
                slot_title = SLOT_TITLES.get(slot, slot.replace("_", "-").capitalize())
                cit = cand["citation_label"]
                bullet = f"- **{slot_title}:** {cand['sentence']}"
                if cit:
                    bullet += f" {cit}"
                bullets.append(bullet)

        if not bullets:
            return header

        return header + "\n" + "\n".join(bullets)

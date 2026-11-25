"""
Self-Built Lightweight PyTorch Discourse Role & Salience Scorer.

100% Local Offline, Zero External API, Pure PyTorch + NumPy + stdlib.
Classifies each candidate clause into one of 6 discourse roles:
  0: DEFINITION
  1: MECHANISM
  2: CAUSE_EFFECT
  3: COMPARISON_CONTRAST
  4: METRIC_EXAMPLE
  5: LIMITATION_CONDITION
and predicts calibrated salience score in [0, 1] to filter out keyword-matched
noise/fillers/headings before GroundedExplainer synthesis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.brain.textrank_salience import (
    VI_STOPWORDS,
    rank_sentences,
    strip_accents,
    tokenize as tr_tokenize,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINT_PATH = ROOT_DIR / "local_runtime" / "checkpoints" / "discourse_scorer.pt"

FEATURE_DIM = 24

DISCOURSE_ROLES: Tuple[str, ...] = (
    "DEFINITION",
    "MECHANISM",
    "CAUSE_EFFECT",
    "COMPARISON_CONTRAST",
    "METRIC_EXAMPLE",
    "LIMITATION_CONDITION",
)
ROLE_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(DISCOURSE_ROLES)}


@dataclass
class ClauseScore:
    """Scored candidate clause produced by NeuralDiscourseScorer."""

    text: str
    role: str
    role_confidence: float
    salience: float
    query_relevance: float
    is_filler: bool
    role_probs: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Accent-Folded Linguistic Patterns (Vietnamese + English)
# ---------------------------------------------------------------------------

_DEF_CUES = (
    " la mot ", " la ", " duoc dinh nghia", " duoc hieu la", " nghia la",
    " goi la ", " khai niem", " dung de ", " la giao thuc", " la he thong",
    " la mo hinh", " la phuong phap", " la qua trinh", " la dai luong",
    " la chat", " la thuat toan", " is a ", " is an ", " refers to ",
    " defined as ", " termed ", " known as ", " means ",
)
_MECH_CUES = (
    "co che", "nguyen ly", "hoat dong", "van hanh", "thong qua", "bang cach",
    "quy trinh", "thuat toan", "kien truc", "tich hop", "cap nhat", "xu ly",
    "phan ra", "dinh tuyen", "ma hoa", "giai ma", "su dung", "ap dung",
    " buoc ", "chia thanh", " gom ", "bao gom", "trien khai", "architecture",
    "mechanism", "operates", "works by", "replaces", "maintains", "decomposes",
    "computes", "stores", "maps", "converts",
)
_CAUSE_CUES = (
    " vi ", "boi vi", " do ", "nho vao", "nho co", "nho do", " nen ", "vi vay", "do do", "vi the",
    "dan den", "gay ra", "tao ra", "he qua", "tac dung", " giup ", " nham ",
    "lam cho", "ngan chan", "toi uu", "cai thien", "giam thieu", "nang cao",
    "bao ve", "because", "therefore", "leads to", " helps ", "prevents",
    "due to", "mitigate", "as a result", "so that",
)
_COMPARE_CUES = (
    "so voi", "khac voi", "khac nhau", "giong nhau", "doi chieu", "so sanh",
    "nguoc lai", "trai lai", "trong khi", "uu diem", "nhuoc diem", "vuot troi",
    "cao hon", "thap hon", "nhanh hon", "tot hon", "kem hon", "on dinh hon",
    "compared to", "whereas", "unlike", "versus", "vs.", "in contrast",
    "outperforms", "better than", "faster than", "higher than", "lower than",
    "alternatively",
)
_METRIC_UNITS_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|phan tram|tokens?|ms|mili[- ]?giay|giay|phut|gio|"
    r"gb|mb|kb|tb|mw|kw|wh/kg|byte|pixel|chu ky|nam|lan|trieu|ty|nghin|tang|lop|"
    r"benh nhan|yeu cau|giao dich|diem|f1|auc|bleu|rouge|epoch|bit)\b",
    re.IGNORECASE,
)
_FORMULA_RE = re.compile(
    r"(?:[a-z0-9_)\]]\s*=\s*[^=]|\bmod\b|\bxor\b|\bgcd\b|\bo\s*\([^)]+\)|<=|>=|\^)",
    re.IGNORECASE,
)
_EXAMPLE_CUES = (
    "vi du", "chang han", "minh hoa", "gia su", "ta co", "cho p =", "cho p=",
    "truong hop", "bai toan", "de bai", "for example", "e.g.", "suppose",
)
_LIMIT_CUES = (
    "han che", "nhuoc diem", "thach thuc", "kho khan", "rui ro", "danh doi",
    "rang buoc", "dieu kien", "chua giai quyet", "yeu diem", "khuyet diem",
    "tuy nhien", "mac du", " song ", "chi khi", " neu ", "can luu y", "yeu cau",
    "doi hoi", "bat buoc", "khong the", "nguy co", "de doa", "suy giam",
    "limitation", "trade-off", "however", "requires", "constraint", "suffer from",
    "limits", "bottleneck", "degradation", "finite",
)
_FILLER_CUES = (
    "nhu da noi", "nhu da trinh bay", "nhu tren", "duoi day la", "sau day la",
    "xem them", "tham khao muc", "bang duoi day", "hinh ben duoi", "hinh tren",
    "chung ta cung xem", "o muc nay ta se", "o phan nay ta se",
    "noi dung chi tiet nhu sau", "chi tiet xem tai", "trang tiep theo",
)
_VERB_PREDICATE_CUES = (
    " la ", " giup ", " tao ", " dung ", " su dung ", " dat ", " giam ", " tang ",
    " hoat dong", " van hanh", " gom ", " bao gom", " cho phep", " ngan ",
    " chuyen ", " phat hien", " xac thuc", " duy tri", " ap dung", " thu ",
    " nhuong ", " nhan ", " bien ", " thay the", " ma hoa", " giai ma",
    " phan tich", " xu ly", " kiem soat", " phan hoi", " is ", " are ", " uses ",
    " helps ", " provides ", " achieves ", " = ",
)

_QUERY_STOPWORDS = VI_STOPWORDS | {
    "hay", "cho", "biet", "giai", "thich", "ngan", "gon", "de", "hieu",
    "trinh", "bay", "tom", "tat", "luoc", "phan", "tich", "muc", "chuong",
    "bai", "what", "how", "why", "explain", "briefly", "describe",
}


def _is_one_edit_away(a: str, b: str) -> bool:
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1 or la < 3 or lb < 3:
        return False
    if la == lb:
        return sum(1 for ca, cb in zip(a, b) if ca != cb) == 1
    if la < lb:
        a, b = b, a
    i = j = diffs = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            diffs += 1
            if diffs > 1:
                return False
            i += 1
        else:
            i += 1
            j += 1
    return True


def _detect_heading_or_marker(clause: str) -> bool:
    s = (clause or "").strip()
    if not s:
        return True
    if "=== Page" in s or re.match(r"^===.*===$", s):
        return True
    if s.startswith("#"):
        return True
    if re.match(
        r"^(?:phan|chuong|muc|bai|topic|chu\s*de|section|chapter)\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[:.\-–]?\s*[^\n]{0,70}$",
        strip_accents(s.lower()),
    ) and ":" not in s[35:] and "." not in s.rstrip(".")[-15:]:
        # Pure section title without explanatory predicate
        s_norm = f" {strip_accents(s.lower())} "
        if not any(v in s_norm for v in _VERB_PREDICATE_CUES):
            return True
    if re.match(r"^\d+(?:\.\d+)*[\.)]\s+[A-ZÀ-Ỹ]", s) and len(s.split()) <= 10 and "=" not in s:
        s_norm = f" {strip_accents(s.lower())} "
        if not any(v in s_norm for v in _VERB_PREDICATE_CUES):
            return True
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters) and "=" not in s:
        return True
    if s.endswith(":") and len(s.split()) <= 12:
        return True
    return False


def _detect_filler(clause: str) -> bool:
    s = (clause or "").strip()
    if len(s) < 12:
        return True
    if _detect_heading_or_marker(s):
        return True
    s_norm = strip_accents(s.lower())
    if any(fc in s_norm for fc in _FILLER_CUES) and len(s.split()) <= 18:
        return True
    return False


def _query_intent_flags(q_norm: str) -> Dict[str, bool]:
    return {
        "definition": any(k in q_norm for k in ("la gi", "khai niem", "dinh nghia", "nghia la", "what is", "define")),
        "mechanism": any(k in q_norm for k in ("hoat dong", "co che", "nguyen ly", "cach ", "quy trinh", "cac buoc", "nhu the nao", "the nao", "how ")),
        "cause": any(k in q_norm for k in ("tai sao", "vi sao", "nguyen nhan", "ly do", "muc dich", "vai tro", "tac dung", "why ")),
        "compare": any(k in q_norm for k in ("so sanh", "phan biet", "khac nhau", "giong nhau", "uu diem", "nhuoc diem", "uu nhuoc", "vs", "compare")),
        "metric": any(k in q_norm for k in ("bao nhieu", "%", "ty le", "so lieu", "ket qua", "hieu nang", "toc do", "do tre", "cong thuc")),
        "example": any(k in q_norm for k in ("vi du", "minh hoa", "bai toan", "truong hop", "ap dung", "example")),
        "limitation": any(k in q_norm for k in ("han che", "nhuoc diem", "thach thuc", "kho khan", "rui ro", "dieu kien", "can luu y", "limitation")),
    }


def extract_clause_features(
    clause: str,
    query: str,
    passage_pos_ratio: float = 0.0,
    textrank_score: float = 0.5,
    redundancy_with_selected: float = 0.0,
) -> np.ndarray:
    """
    Extracts deterministic 24-dim feature vector for a (query, clause) pair.
    All features are bounded floats suitable for LayerNorm + MLP inference.
    """
    c_raw = (clause or "").strip()
    q_raw = (query or "").strip()
    c_norm = strip_accents(c_raw.lower())
    q_norm = strip_accents(q_raw.lower())
    c_padded = f" {c_norm} "

    q_all_toks = [t for t in re.findall(r"[a-z0-9_]+", q_norm) if len(t) >= 2 and t not in _QUERY_STOPWORDS]
    q_set = set(q_all_toks)
    c_toks = [t for t in re.findall(r"[a-z0-9_]+", c_norm) if len(t) >= 2]
    c_set = set(c_toks)

    # 0-1: exact & fuzzy overlap
    if q_set:
        exact_hits = len(q_set & c_set)
        fuzzy_hits = 0
        if exact_hits > 0 or len(q_set) <= 2:
            for qt in q_set - c_set:
                if len(qt) >= 3 and any(_is_one_edit_away(qt, ct) for ct in c_set):
                    fuzzy_hits += 1
        exact_overlap = min(1.0, exact_hits / len(q_set))
        fuzzy_overlap = min(1.0, (exact_hits + 0.85 * fuzzy_hits) / len(q_set))
    else:
        exact_overlap = 0.25 if c_set else 0.0
        fuzzy_overlap = exact_overlap

    # 2: subject-head match
    subj_split = re.split(r"\b(?:la|duoc|is|are)\b|:", c_norm, maxsplit=1)
    subj_toks = set(re.findall(r"[a-z0-9_]+", subj_split[0])) if subj_split else set()
    subject_head_match = min(1.0, len(q_set & subj_toks) / max(1, len(q_set))) if q_set else 0.0

    # 4-10: Role cue signals (matched against space-padded normalized text; 'nhờ' checked with tone mark to avoid 'bộ nhớ')
    def_score = min(1.0, sum(1 for c in _DEF_CUES if c in c_padded) * 0.5)
    mech_score = min(1.0, sum(1 for c in _MECH_CUES if c in c_padded) * 0.35)
    cause_hits = sum(1 for c in _CAUSE_CUES if c in c_padded) + (1 if re.search(r"\bnhờ\b", c_raw.lower()) else 0)
    cause_score = min(1.0, cause_hits * 0.4)
    compare_score = min(1.0, sum(1 for c in _COMPARE_CUES if c in c_norm) * 0.45)
    has_metric = bool(_METRIC_UNITS_RE.search(c_norm))
    has_formula = bool(_FORMULA_RE.search(c_norm))
    metric_formula_score = min(1.0, (0.65 if has_metric else 0.0) + (0.65 if has_formula else 0.0))
    example_score = min(1.0, sum(1 for c in _EXAMPLE_CUES if c in c_norm) * 0.55)
    limit_score = min(1.0, sum(1 for c in _LIMIT_CUES if c in c_padded) * 0.4)

    # 3: query-intent alignment
    q_flags = _query_intent_flags(q_norm)
    intent_match = 0.0
    if q_flags["definition"] and def_score > 0:
        intent_match = max(intent_match, 1.0)
    if q_flags["mechanism"] and mech_score > 0:
        intent_match = max(intent_match, 0.9)
    if q_flags["cause"] and (cause_score > 0 or mech_score > 0):
        intent_match = max(intent_match, 0.95)
    if q_flags["compare"] and compare_score > 0:
        intent_match = max(intent_match, 1.0)
    if q_flags["metric"] and metric_formula_score > 0:
        intent_match = max(intent_match, 1.0)
    if q_flags["example"] and (example_score > 0 or metric_formula_score > 0):
        intent_match = max(intent_match, 0.95)
    if q_flags["limitation"] and limit_score > 0:
        intent_match = max(intent_match, 1.0)

    # 11-16: Quality & noise rejection
    words = c_raw.split()
    wc = len(words)
    len_norm = min(1.0, wc / 40.0)
    if 8 <= wc <= 32:
        optimal_len = 1.0
    elif 5 <= wc < 8 or 32 < wc <= 45:
        optimal_len = 0.6
    else:
        optimal_len = 0.15
    too_short_or_long = 1.0 if (wc < 5 or wc > 55) else 0.0
    is_heading = 1.0 if _detect_heading_or_marker(c_raw) else 0.0
    is_filler = 1.0 if _detect_filler(c_raw) else 0.0
    ex_penalty = 1.0 if (example_score >= 0.5 and not q_flags["example"] and not q_flags["metric"]) else 0.0

    # 17-19: Context graph & positional priors
    lead_pos = 1.0 - float(min(1.0, max(0.0, passage_pos_ratio)))
    tr = float(min(1.0, max(0.0, textrank_score)))
    redund = float(min(1.0, max(0.0, redundancy_with_selected)))

    # 20-23: Syntactic & topical completeness
    content_tokens = [t for t in c_toks if t not in VI_STOPWORDS]
    info_density = min(1.0, len(set(content_tokens)) / max(1, wc)) if wc else 0.0
    has_verb = 1.0 if any(v in c_padded for v in _VERB_PREDICATE_CUES) else 0.0
    has_any_discourse = max(
        def_score, mech_score, cause_score, compare_score, metric_formula_score, limit_score
    ) > 0.0
    no_discourse_signal = 1.0 if (not has_any_discourse and exact_overlap < 0.2) else 0.0
    sec_intro_lead = 1.0 if (passage_pos_ratio <= 0.20 and is_heading == 0.0 and wc >= 7) else 0.0

    feats = np.array(
        [
            exact_overlap,          # 0
            fuzzy_overlap,          # 1
            subject_head_match,     # 2
            intent_match,           # 3
            def_score,              # 4
            mech_score,             # 5
            cause_score,            # 6
            compare_score,          # 7
            metric_formula_score,   # 8
            example_score,          # 9
            limit_score,            # 10
            len_norm,               # 11
            optimal_len,            # 12
            too_short_or_long,      # 13
            is_heading,             # 14
            is_filler,              # 15
            ex_penalty,             # 16
            lead_pos,               # 17
            tr,                     # 18
            redund,                 # 19
            info_density,           # 20
            has_verb,               # 21
            no_discourse_signal,    # 22
            sec_intro_lead,         # 23
        ],
        dtype=np.float32,
    )
    return feats


ROLE_FEATURE_INDICES: Tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21)


class NeuralDiscourseHead(nn.Module):
    """
    Lightweight Self-Built Dual-Head Neural Scorer (~4.5K params):
      - Role head consumes strictly clause-intrinsic semantic features (immune to position shortcuts)
      - Salience head consumes full 24-dim contextual + quality features via residual MLP
    """

    def __init__(self, in_dim: int = FEATURE_DIM, hidden_dim: int = 48, num_roles: int = len(DISCOURSE_ROLES)) -> None:
        super().__init__()
        self.register_buffer(
            "role_idx",
            torch.tensor(ROLE_FEATURE_INDICES, dtype=torch.long),
            persistent=False,
        )
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.res_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.salience_head = nn.Linear(hidden_dim, 1)

        self.role_proj = nn.Linear(len(ROLE_FEATURE_INDICES), 32)
        self.role_norm = nn.LayerNorm(32)
        self.role_head = nn.Linear(32, num_roles)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.norm1(self.in_proj(x)))
        h = h + self.act(self.norm2(self.res_proj(h)))
        salience_logit = self.salience_head(h).squeeze(-1)

        x_role = x.index_select(-1, self.role_idx)
        hr = self.act(self.role_norm(self.role_proj(x_role)))
        role_logits = self.role_head(hr)
        return role_logits, salience_logit

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _analytic_role_and_salience(feats: np.ndarray) -> Tuple[int, float, float]:
    """Deterministic fallback when trained checkpoint file is absent."""
    # Map feature cues to DISCOURSE_ROLES indices:
    # 0:DEF(4), 1:MECH(5), 2:CAUSE(6), 3:COMPARE(7), 4:METRIC_EX(8,9), 5:LIMIT(10)
    role_scores = np.array(
        [
            feats[4] * 1.2 + feats[23] * 0.25,
            feats[5] * 1.15,
            feats[6] * 1.25,
            feats[7] * 1.30,
            max(feats[8], feats[9]) * 1.30,
            feats[10] * 1.25,
        ],
        dtype=np.float32,
    )
    if float(role_scores.max()) <= 0.05:
        role_idx = 1 if feats[21] > 0.5 else 0
        conf = 0.50
    else:
        exp_s = np.exp(role_scores * 2.5 - float(role_scores.max()) * 2.5)
        probs = exp_s / max(1e-6, float(exp_s.sum()))
        role_idx = int(np.argmax(probs))
        conf = float(probs[role_idx])

    raw_sal = (
        1.25 * feats[0]
        + 0.70 * feats[2]
        + 0.85 * feats[3]
        + 0.55 * feats[12]
        + 0.45 * feats[18]
        + 0.35 * feats[21]
        + 0.25 * feats[23]
        - 1.80 * feats[14]
        - 1.60 * feats[15]
        - 0.75 * feats[16]
        - 1.10 * feats[19]
        - 0.90 * feats[22]
        - 0.40
    )
    salience = float(1.0 / (1.0 + np.exp(-raw_sal)))
    return role_idx, conf, salience


class NeuralDiscourseScorer:
    """
    Singleton Offline Scorer wrapping NeuralDiscourseHead.
    Uses PyTorch CPU inference (~0.5ms per batch) if checkpoint exists,
    falling back seamlessly to deterministic analytic weights if checkpoint is missing.
    """

    _instance: Optional["NeuralDiscourseScorer"] = None

    def __init__(self, checkpoint_path: Optional[Path] = None) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        self.model: Optional[NeuralDiscourseHead] = None
        self.has_trained_weights: bool = False
        self.metrics: Dict[str, Any] = {}
        self._load_checkpoint()

    @classmethod
    def get_instance(cls) -> "NeuralDiscourseScorer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reload(self, checkpoint_path: Optional[Path] = None) -> None:
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            self.model = None
            self.has_trained_weights = False
            return
        try:
            payload = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(str(self.checkpoint_path), map_location="cpu")
        except Exception as exc:
            logger.warning("Failed to load discourse scorer checkpoint %s: %s", self.checkpoint_path, exc)
            self.model = None
            self.has_trained_weights = False
            return

        try:
            state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            hidden_dim = int(payload.get("hidden_dim", 48)) if isinstance(payload, dict) else 48
            net = NeuralDiscourseHead(in_dim=FEATURE_DIM, hidden_dim=hidden_dim, num_roles=len(DISCOURSE_ROLES))
            net.load_state_dict(state, strict=True)
            net.eval()
            self.model = net
            self.has_trained_weights = True
            self.metrics = dict(payload.get("metrics", {})) if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.warning("Corrupt discourse scorer state_dict %s: %s", self.checkpoint_path, exc)
            self.model = None
            self.has_trained_weights = False

    def score_clauses(
        self,
        query: str,
        clauses: Sequence[str],
        passage_positions: Optional[Sequence[float]] = None,
        redundancies: Optional[Sequence[float]] = None,
    ) -> List[ClauseScore]:
        """Scores a sequence of candidate clauses for salience and discourse role."""
        clean_clauses = [str(c or "").strip() for c in clauses]
        n = len(clean_clauses)
        if n == 0:
            return []

        if passage_positions is not None and len(passage_positions) == n:
            pos_list = [float(p) for p in passage_positions]
        else:
            pos_list = [i / max(1, n - 1) if n > 1 else 0.0 for i in range(n)]

        if redundancies is not None and len(redundancies) == n:
            red_list = [float(r) for r in redundancies]
        else:
            red_list = [0.0] * n

        tr_raw = rank_sentences(clean_clauses) if n > 1 else [1.0]
        tr_max = max(tr_raw) if tr_raw else 1.0
        tr_norm = [v / tr_max if tr_max > 0 else 0.5 for v in tr_raw]

        feat_rows = [
            extract_clause_features(
                clause=clean_clauses[i],
                query=query,
                passage_pos_ratio=pos_list[i],
                textrank_score=tr_norm[i],
                redundancy_with_selected=red_list[i],
            )
            for i in range(n)
        ]
        feat_mat = np.stack(feat_rows, axis=0)

        results: List[ClauseScore] = []
        if self.model is not None and self.has_trained_weights:
            with torch.no_grad():
                x = torch.from_numpy(feat_mat)
                role_logits, sal_logits = self.model(x)
                role_probs_np = torch.softmax(role_logits, dim=-1).cpu().numpy()
                sal_probs_np = torch.sigmoid(sal_logits).cpu().numpy()
            for i in range(n):
                r_idx = int(np.argmax(role_probs_np[i]))
                r_conf = float(role_probs_np[i, r_idx])
                sal = float(sal_probs_np[i])
                feats = feat_rows[i]
                filler = bool(feats[14] > 0.5 or feats[15] > 0.5 or feats[13] > 0.5 and len(clean_clauses[i].split()) < 5)
                if filler:
                    sal = min(sal, 0.25)
                results.append(
                    ClauseScore(
                        text=clean_clauses[i],
                        role=DISCOURSE_ROLES[r_idx],
                        role_confidence=round(r_conf, 4),
                        salience=round(sal, 4),
                        query_relevance=round(float(feats[1]), 4),
                        is_filler=filler,
                        role_probs={r: round(float(role_probs_np[i, j]), 4) for j, r in enumerate(DISCOURSE_ROLES)},
                    )
                )
        else:
            for i in range(n):
                feats = feat_rows[i]
                r_idx, r_conf, sal = _analytic_role_and_salience(feats)
                filler = bool(feats[14] > 0.5 or feats[15] > 0.5 or feats[13] > 0.5 and len(clean_clauses[i].split()) < 5)
                if filler:
                    sal = min(sal, 0.25)
                results.append(
                    ClauseScore(
                        text=clean_clauses[i],
                        role=DISCOURSE_ROLES[r_idx],
                        role_confidence=round(r_conf, 4),
                        salience=round(sal, 4),
                        query_relevance=round(float(feats[1]), 4),
                        is_filler=filler,
                    )
                )

        return results


__all__ = [
    "DISCOURSE_ROLES",
    "FEATURE_DIM",
    "DEFAULT_CHECKPOINT_PATH",
    "ClauseScore",
    "NeuralDiscourseHead",
    "NeuralDiscourseScorer",
    "extract_clause_features",
]

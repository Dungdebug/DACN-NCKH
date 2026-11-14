"""
TextRank Salience & Key Point Extractor for Sovereign Document Brain.
Pure-Python, graph-based TextRank salience scoring, MMR diversity selection,
and claim-evidence association.
Zero external API, zero network egress, fully deterministic.
"""

import math
import re
import unicodedata
from typing import List, Tuple, Set, Optional


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive canonical alignment."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


VI_STOPWORDS: Set[str] = {
    "va", "cua", "cac", "nhung", "trong", "de", "voi", "cho", "tai", "la",
    "co", "duoc", "khi", "nay", "do", "theo", "ve", "nhu", "mot", "nhieu",
    "thi", "ma", "tu", "o", "ra", "da", "se", "dang", "boi", "bang",
    "hoac", "hay", "vi", "neu", "tuy", "nhien", "chung_toi", "chung", "toi",
    "ho", "no", "rat", "qua", "lam", "lai", "len", "xuong", "di", "den",
    "cung", "luon", "chi", "moi", "con", "nao", "sao", "gi",
}

CLAIM_CUES: List[str] = [
    "đề xuất", "cho thấy", "chứng minh", "khẳng định", "phát hiện", "kết luận", "chỉ ra", "đạt được",
    "de xuat", "cho thay", "chung minh", "khang dinh", "phat hien", "ket luan", "chi ra", "dat duoc",
]

METRIC_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|tokens?|ms|s|gb|mb|kb|triệu|nghìn|tỷ|f1|accuracy|bleu|tầng|lớp|param|tham số)\b",
    re.IGNORECASE,
)


def tokenize(text: str) -> List[str]:
    """Extracts accent-folded content tokens, filtering out stop words and single chars."""
    norm = strip_accents(text.lower())
    words = re.findall(r"\b[a-z0-9_]+\b", norm)
    return [w for w in words if len(w) > 1 and w not in VI_STOPWORDS]


def sentence_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """
    Classic TextRank sentence similarity:
    Sim(Si, Sj) = |Si ∩ Sj| / (log(|Si|) + log(|Sj|))
    """
    if not tokens1 or not tokens2:
        return 0.0
    set1 = set(tokens1)
    set2 = set(tokens2)
    overlap = len(set1 & set2)
    if overlap == 0:
        return 0.0
    denom = math.log(max(len(tokens1), 2)) + math.log(max(len(tokens2), 2))
    return overlap / denom if denom > 0 else 0.0


def rank_sentences(
    sentences: List[str],
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> List[float]:
    """
    Computes graph-based TextRank salience scores for sentences via pure-Python power iteration.
    Returns a list of float scores parallel to the input sentences.
    """
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    tokenized = [tokenize(s) for s in sentences]

    # Build symmetric adjacency matrix
    weights = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            w = sentence_similarity(tokenized[i], tokenized[j])
            weights[i][j] = w
            weights[j][i] = w

    out_sums = [sum(weights[i]) for i in range(n)]
    scores = [1.0] * n

    for _ in range(max_iter):
        new_scores = [0.0] * n
        for i in range(n):
            incoming = 0.0
            for j in range(n):
                if i != j and out_sums[j] > 0:
                    incoming += (weights[j][i] / out_sums[j]) * scores[j]
            new_scores[i] = (1.0 - damping) + damping * incoming

        diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if diff < tol:
            break

    return scores


def heuristic_score(sentence: str) -> float:
    """
    Scores informative markers: contribution, results, limitations, metrics, and length sanity.
    """
    score = 0.0
    s_lower = sentence.lower()
    s_folded = strip_accents(s_lower)

    # Contribution / model cues
    cues_contrib = [
        "mục tiêu", "nghiên cứu", "đề xuất", "hệ thống", "kiến trúc", "mô hình",
        "phương pháp", "giải pháp",
        "muc tieu", "nghien cuu", "de xuat", "he thong", "kien truc", "mo hinh",
        "phuong phap", "giai phap",
    ]
    if any(k in s_lower or k in s_folded for k in cues_contrib):
        score += 3.0

    # Result / evaluation cues
    cues_result = [
        "kết quả", "hiệu năng", "đạt", "đánh giá", "tối ưu", "cải thiện", "chính xác",
        "ket qua", "hieu nang", "dat", "danh gia", "toi uu", "cai thien", "chinh xac",
    ]
    if any(k in s_lower or k in s_folded for k in cues_result):
        score += 2.5

    # Limitation / challenge cues
    cues_limit = [
        "hạn chế", "thách thức", "vấn đề", "khó khăn", "tuy nhiên", "nhược điểm",
        "han che", "thach thuc", "van de", "kho khan", "tuy nhien", "nhuoc diem",
    ]
    if any(k in s_lower or k in s_folded for k in cues_limit):
        score += 2.0

    # Numeric metrics / benchmark units
    if METRIC_PATTERN.search(sentence):
        score += 2.0

    # Length sanity check (prefer 40 - 220 chars)
    if 40 <= len(sentence) <= 220:
        score += 1.0

    return score


def _normalize_scores(values: List[float]) -> List[float]:
    """Scales scores to [0, 1] relative to max. Returns [1.0]*n if all zero."""
    if not values:
        return []
    max_v = max(values)
    if max_v > 1e-9:
        return [max(0.0, v / max_v) for v in values]
    return [1.0] * len(values)


def select_key_points(
    sentences: List[str],
    k: int = 4,
    mmr_lambda: float = 0.7,
) -> List[str]:
    """
    Selects top-k salient key points combining:
      Final score = 0.6·TextRank (normalized) + 0.4·heuristic markers
    followed by Maximal Marginal Relevance (MMR) deduplication.
    """
    if not sentences or k <= 0:
        return []

    target_k = min(k, len(sentences))

    tr_scores = rank_sentences(sentences)
    heur_scores = [heuristic_score(s) for s in sentences]

    norm_tr = _normalize_scores(tr_scores)
    norm_heur = _normalize_scores(heur_scores)

    relevance = [0.6 * tr + 0.4 * heur for tr, heur in zip(norm_tr, norm_heur)]
    token_sets = [set(tokenize(s)) for s in sentences]

    selected_indices: List[int] = []
    candidates = list(range(len(sentences)))

    while len(selected_indices) < target_k and candidates:
        if not selected_indices:
            best_idx = max(candidates, key=lambda idx: relevance[idx])
        else:
            def mmr_score(cand_idx: int) -> float:
                cand_tokens = token_sets[cand_idx]
                cand_s = sentences[cand_idx]
                cand_prefix = " ".join(cand_s.split()[:4]).lower()

                max_sim = 0.0
                has_identical_prefix = False

                for sel_idx in selected_indices:
                    sel_tokens = token_sets[sel_idx]
                    sel_s = sentences[sel_idx]
                    sel_prefix = " ".join(sel_s.split()[:4]).lower()

                    if cand_prefix == sel_prefix:
                        has_identical_prefix = True

                    if cand_tokens and sel_tokens:
                        sim = len(cand_tokens & sel_tokens) / len(cand_tokens | sel_tokens)
                    elif not cand_tokens and not sel_tokens:
                        sim = 1.0
                    else:
                        sim = 0.0
                    if sim > max_sim:
                        max_sim = sim

                # Apply penalty for redundancy / near-duplicates
                if has_identical_prefix or max_sim >= 0.45:
                    penalty = 1.0 + max_sim
                else:
                    penalty = max_sim

                return mmr_lambda * relevance[cand_idx] - (1.0 - mmr_lambda) * penalty

            best_idx = max(candidates, key=mmr_score)

        selected_indices.append(best_idx)
        candidates.remove(best_idx)

    return [sentences[idx] for idx in selected_indices]


def extract_claim_evidence_pairs(sentences: List[str]) -> List[Tuple[str, str]]:
    """
    Links claim sentences (containing đề xuất/cho thấy/chứng minh/...)
    to the most similar sentence containing an empirical metric.
    """
    if not sentences:
        return []

    metric_indices = [idx for idx, s in enumerate(sentences) if METRIC_PATTERN.search(s)]
    if not metric_indices:
        return []

    tokenized_sets = [set(tokenize(s)) for s in sentences]
    pairs: List[Tuple[str, str]] = []

    for i, s in enumerate(sentences):
        s_lower = s.lower()
        s_folded = strip_accents(s_lower)
        is_claim = any(cue in s_lower or cue in s_folded for cue in CLAIM_CUES)
        if not is_claim:
            continue

        cands = [j for j in metric_indices if j != i]
        if not cands:
            cands = [j for j in metric_indices]

        tokens_c = tokenized_sets[i]

        def pair_score(j: int) -> float:
            tokens_e = tokenized_sets[j]
            if tokens_c and tokens_e:
                sim = len(tokens_c & tokens_e) / len(tokens_c | tokens_e)
            else:
                sim = 0.0
            proximity = 0.05 / (1.0 + abs(i - j))
            return sim + proximity

        best_j = max(cands, key=pair_score)
        pairs.append((s, sentences[best_j]))

    return pairs

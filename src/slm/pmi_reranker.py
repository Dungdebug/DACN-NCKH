"""
PMI (Pointwise Mutual Information) Neural Reranker for WaveCMS-R^2 (Ticket T3).

100% offline, PyTorch + transformers local.
Scores passages for a query using length-normalized PMI:
    s_PMI(q, e_i) = (1 / |q|) * [log P_theta(q | e_i^(window)) - log P_theta(q | empty)]

Supports:
- Base SmolLM2-135M (enable_cms=False, Ablation W4a -> mode_label="base_smollm2")
- SA-CMS Adapter on SmolLM2-135M (enable_cms=True, W4b -> mode_label="sa_cms_adapter")
- Graceful unavailable fallback (mode_label="unavailable")
- Single null-context forward pass per query + single batched passage forward pass
- CUDA float16 autocast and adaptive latency budget guard
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Enforce offline transformers behavior
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("SA_CMS_OFFLINE_MODE", "1")

DEFAULT_CHECKPOINT_CANDIDATES: Tuple[str, ...] = (
    "checkpoints/train_api_expanded/cms_20m_tokens_final.pt",
    "local_runtime/checkpoints/cms_3lvl_full_dataset.pt",
    "checkpoints/full_dataset_training/cms_3lvl_full_seed_42.pt",
    "checkpoints/sa_cms_level_3_sa_cms.pt",
    "local_runtime/model",
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+|\n+")
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def should_run_pmi(
    lse_margin: float,
    cross_agree: float = 0.0,
    margin_threshold: float = 0.35,
    agree_threshold: float = 0.65,
) -> bool:
    """Returns False when sparse_lse routing is already decisive, skipping GPU PMI."""
    return not (float(lse_margin) >= float(margin_threshold) or float(cross_agree) >= float(agree_threshold))


class PMIReranker:
    """
    Offline PMI reranker over a local causal LM (SmolLM2-135M or StructureAlignedHopeLM).
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: str = "auto",
        window: int = 64,
        k: int = 5,
        budget_ms: float = 60.0,
        checkpoint_path: Optional[str] = None,
        mode_label: Optional[str] = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.window = max(1, int(window))
        self.k = max(1, int(k))
        self.budget_ms = float(budget_ms)

        self.model = model
        self.tokenizer = tokenizer
        self._checkpoint_path = checkpoint_path
        self._last_latency_ms: float = 0.0
        self._disabled_by_budget: bool = False

        if mode_label is not None:
            self._mode_label = mode_label
        elif self.model is None or self.tokenizer is None:
            self._mode_label = "unavailable"
        elif getattr(self.model, "cms", None) is not None:
            self._mode_label = "sa_cms_adapter"
        else:
            self._mode_label = "base_smollm2"

        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()

    @property
    def available(self) -> bool:
        return (
            self.model is not None
            and self.tokenizer is not None
            and not self._disabled_by_budget
        )

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def checkpoint_path(self) -> Optional[str]:
        return self._checkpoint_path

    @property
    def mode_label(self) -> str:
        if self.model is None or self.tokenizer is None:
            return "unavailable"
        return self._mode_label

    def _encode(self, text: str) -> List[int]:
        if hasattr(self.tokenizer, "encode"):
            try:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                ids = self.tokenizer.encode(text)
            return list(ids)
        out = self.tokenizer(text, add_special_tokens=False)
        return list(out["input_ids"])

    def _decode(self, token_ids: Sequence[int]) -> str:
        if hasattr(self.tokenizer, "decode"):
            try:
                return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)
            except TypeError:
                return self.tokenizer.decode(list(token_ids))
        return " ".join(str(t) for t in token_ids)

    def _get_bos_id(self) -> int:
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is None:
            bos = getattr(self.tokenizer, "eos_token_id", None)
        if bos is None:
            bos = getattr(self.tokenizer, "pad_token_id", None)
        return 0 if bos is None else int(bos)

    def _get_pad_id(self) -> int:
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer, "eos_token_id", None)
        return 0 if pad is None else int(pad)

    def _extract_best_window(self, query: str, passage: str) -> str:
        """
        Extracts a window of at most `self.window` tokens centered around the
        sentence in `passage` with the highest keyword overlap with `query`.
        """
        text = (passage or "").strip()
        if not text:
            return ""

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            sentences = [text]

        query_terms = {w.lower() for w in _WORD_RE.findall(query or "") if len(w) >= 2}
        if not query_terms:
            query_terms = {w.lower() for w in _WORD_RE.findall(query or "")}

        best_idx = 0
        best_hits = -1
        for idx, sent in enumerate(sentences):
            sent_terms = {w.lower() for w in _WORD_RE.findall(sent)}
            hits = len(query_terms & sent_terms)
            if hits > best_hits:
                best_hits = hits
                best_idx = idx

        # Encode each sentence to locate token span of the best sentence
        sent_token_lists: List[List[int]] = [self._encode(s) for s in sentences]
        all_tokens: List[int] = []
        best_start = 0
        best_end = 0
        for idx, toks in enumerate(sent_token_lists):
            if idx == best_idx:
                best_start = len(all_tokens)
                best_end = best_start + len(toks)
            all_tokens.extend(toks)

        if len(all_tokens) <= self.window:
            return text

        center = (best_start + best_end) // 2
        half = self.window // 2
        win_start = max(0, center - half)
        win_end = win_start + self.window
        if win_end > len(all_tokens):
            win_end = len(all_tokens)
            win_start = max(0, win_end - self.window)

        return self._decode(all_tokens[win_start:win_end]).strip()

    def _forward_logits(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(out, tuple):
            return out[0]
        if hasattr(out, "logits"):
            return out.logits
        return out

    def _score_query_given_sequences(
        self,
        sequences: List[List[int]],
        query_ids: List[int],
    ) -> torch.Tensor:
        """
        Computes sum_t log P_theta(q_t | context_i, q_<t) for a batch of
        sequences where each sequence ends with `query_ids`.
        """
        pad_id = self._get_pad_id()
        max_len = max(len(seq) for seq in sequences)
        batch_size = len(sequences)
        q_len = len(query_ids)

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        # Left-pad so query tokens align at the right edge [-q_len:]
        for i, seq in enumerate(sequences):
            pad_len = max_len - len(seq)
            input_ids[i, pad_len:] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, pad_len:] = 1

        dev = torch.device(self.device)
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)
        target_ids = torch.tensor(query_ids, dtype=torch.long, device=dev)

        is_cuda = dev.type == "cuda"
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=is_cuda, dtype=torch.float16):
                logits = self._forward_logits(input_ids, attention_mask=attention_mask)
                # Position max_len - q_len - 1 + j predicts token max_len - q_len + j
                pred_logits = logits[:, -q_len - 1 : -1, :].float()
                log_probs = F.log_softmax(pred_logits, dim=-1)
                gathered = log_probs.gather(
                    dim=-1,
                    index=target_ids.view(1, q_len, 1).expand(batch_size, q_len, 1),
                ).squeeze(-1)
                return gathered.sum(dim=-1)

    def score(
        self,
        query: str,
        passages: List[str],
        use_budget_guard: bool = True,
    ) -> List[float]:
        """
        Computes length-normalized PMI scores for `passages` given `query`.
        Only the top `self.k` passages are evaluated by the LM in one batch;
        remaining passages receive a fallback score below the scored minimum.
        """
        if not passages:
            return []

        if self.model is None or self.tokenizer is None or not (query or "").strip():
            self._last_latency_ms = 0.0
            return [0.0] * len(passages)

        if use_budget_guard and self._disabled_by_budget:
            self._last_latency_ms = 0.0
            return [0.0] * len(passages)

        q_ids = self._encode(query.strip())
        if not q_ids:
            self._last_latency_ms = 0.0
            return [0.0] * len(passages)

        t0 = time.perf_counter()

        bos_id = self._get_bos_id()
        sep_ids = self._encode("\n\n")

        # 1. Single-batch forward: [null_seq] + top-k passage windows in 1 LM call
        null_seq = [bos_id] + q_ids
        k_eff = min(self.k, len(passages))
        batch_seqs: List[List[int]] = []
        for p in passages[:k_eff]:
            win_text = self._extract_best_window(query, p)
            win_ids = self._encode(win_text)
            if len(win_ids) > self.window:
                win_ids = win_ids[: self.window]
            seq = [bos_id] + win_ids + sep_ids + q_ids
            batch_seqs.append(seq)

        all_ll = self._score_query_given_sequences([null_seq] + batch_seqs, q_ids).tolist()
        log_p_null = float(all_ll[0])
        log_p_cond = all_ll[1:]

        q_len = float(len(q_ids))
        scored = [(lp - log_p_null) / q_len for lp in log_p_cond]

        if len(passages) > k_eff:
            fallback = (min(scored) - 1.0) if scored else -1e9
            scores = scored + [fallback] * (len(passages) - k_eff)
        else:
            scores = scored

        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Adaptive budget guard
        if use_budget_guard:
            if self.k > 2 and self._last_latency_ms > self.budget_ms:
                logger.warning(
                    "PMIReranker latency %.2fms exceeded budget %.2fms; reducing k from %d to 2.",
                    self._last_latency_ms,
                    self.budget_ms,
                    self.k,
                )
                self.k = 2
            elif self.k <= 2 and self._last_latency_ms > 2.5 * self.budget_ms:
                logger.warning(
                    "PMIReranker latency %.2fms still exceeded 2.5x budget (%.2fms); disabling reranker.",
                    self._last_latency_ms,
                    2.5 * self.budget_ms,
                )
                self._disabled_by_budget = True

        return scores

    @staticmethod
    def should_run_pmi(
        lse_margin: float,
        cross_agree: float = 0.0,
        margin_threshold: float = 0.35,
        agree_threshold: float = 0.65,
    ) -> bool:
        """Margin-gated check: returns False when routing is decisive."""
        return should_run_pmi(
            lse_margin=lse_margin,
            cross_agree=cross_agree,
            margin_threshold=margin_threshold,
            agree_threshold=agree_threshold,
        )

    def score_passages(
        self,
        query: str,
        passages: Sequence[str],
        k: Optional[int] = None,
        use_budget_guard: bool = True,
    ) -> List[float]:
        """Alias for `score` accepting any Sequence[str] and optional per-call `k`."""
        passage_list = list(passages)
        if k is None:
            return self.score(query, passage_list, use_budget_guard=use_budget_guard)
        prev_k = self.k
        self.k = max(1, int(k))
        try:
            return self.score(query, passage_list, use_budget_guard=use_budget_guard)
        finally:
            if not use_budget_guard or self.k == max(1, int(k)):
                self.k = prev_k

    def score_passages_gated(
        self,
        query: str,
        passages: Sequence[str],
        lse_margin: float,
        cross_agree: float = 0.0,
        margin_threshold: float = 0.35,
        agree_threshold: float = 0.65,
        k: Optional[int] = None,
        use_budget_guard: bool = True,
    ) -> Tuple[Optional[List[float]], bool]:
        """Runs PMI only when `should_run_pmi(...)` is True; otherwise returns (None, False) in 0ms."""
        if not should_run_pmi(
            lse_margin=lse_margin,
            cross_agree=cross_agree,
            margin_threshold=margin_threshold,
            agree_threshold=agree_threshold,
        ):
            self._last_latency_ms = 0.0
            return None, False
        scores = self.score_passages(query, passages, k=k, use_budget_guard=use_budget_guard)
        return scores, True

    @classmethod
    def from_checkpoint(
        cls,
        path: Optional[str] = None,
        enable_cms: bool = True,
        **kwargs: Any,
    ) -> "PMIReranker":
        """
        Loads PMIReranker strictly offline from local checkpoints or local_runtime/model.
        Supports both base_smollm2 (enable_cms=False) and sa_cms_adapter (enable_cms=True).
        """
        device_arg = kwargs.pop("device", "auto")
        resolved_device = _resolve_device(device_arg)
        dtype = torch.float16 if resolved_device.startswith("cuda") else torch.float32

        candidates: List[Path] = []
        if path:
            candidates.append(Path(path))
        for rel in DEFAULT_CHECKPOINT_CANDIDATES:
            p = Path(rel)
            if p not in candidates:
                candidates.append(p)

        chosen: Optional[Path] = None
        for cand in candidates:
            if cand.exists():
                chosen = cand
                break

        backbone_dir = Path("local_runtime/model")
        if chosen is not None and chosen.is_dir() and (chosen / "config.json").exists():
            backbone_dir = chosen

        if not backbone_dir.exists() or not (backbone_dir / "config.json").exists():
            logger.warning(
                "Local SmolLM2 backbone directory not found at %s; returning unavailable PMIReranker.",
                backbone_dir,
            )
            return cls(model=None, tokenizer=None, device=resolved_device, **kwargs)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer_dir = (
                Path("local_runtime/tokenizer")
                if (Path("local_runtime/tokenizer") / "tokenizer.json").exists()
                else backbone_dir
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_dir),
                local_files_only=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if not enable_cms or chosen is None or chosen.is_dir():
                model = AutoModelForCausalLM.from_pretrained(
                    str(backbone_dir),
                    torch_dtype=dtype,
                    local_files_only=True,
                ).to(resolved_device)
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                loaded_path = str(chosen or backbone_dir)
                logger.info(
                    "PMIReranker loaded base_smollm2 strictly offline from: %s (device=%s)",
                    loaded_path,
                    resolved_device,
                )
                return cls(
                    model=model,
                    tokenizer=tokenizer,
                    device=resolved_device,
                    checkpoint_path=loaded_path,
                    mode_label="base_smollm2",
                    **kwargs,
                )

            # enable_cms=True and chosen is a .pt checkpoint file
            from src.hope_attention.sa_cms import StructureAlignedHopeLM

            ckpt = torch.load(str(chosen), map_location="cpu")
            num_levels = int(ckpt.get("num_levels", 3)) if isinstance(ckpt, dict) else 3

            model = StructureAlignedHopeLM(
                model_name_or_path=str(backbone_dir),
                num_levels=num_levels,
                device=resolved_device,
                torch_dtype=dtype,
                enable_cms=True,
            )
            model.tokenizer = tokenizer

            if isinstance(ckpt, dict):
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
                else:
                    if "cms_state_dict" in ckpt and model.cms is not None:
                        model.cms.load_state_dict(ckpt["cms_state_dict"], strict=False)
                    elif "state_dict" in ckpt and model.cms is not None:
                        model.cms.load_state_dict(ckpt["state_dict"], strict=False)
                    if (
                        "cms_norm_state_dict" in ckpt
                        and ckpt["cms_norm_state_dict"] is not None
                        and getattr(model, "cms_norm", None) is not None
                    ):
                        model.cms_norm.load_state_dict(ckpt["cms_norm_state_dict"], strict=False)

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            logger.info(
                "PMIReranker loaded sa_cms_adapter strictly offline from: %s (backbone=%s, device=%s)",
                chosen,
                backbone_dir,
                resolved_device,
            )
            return cls(
                model=model,
                tokenizer=tokenizer,
                device=resolved_device,
                checkpoint_path=str(chosen),
                mode_label="sa_cms_adapter",
                **kwargs,
            )
        except Exception as exc:
            logger.exception("PMIReranker.from_checkpoint failed: %s", exc)
            return cls(model=None, tokenizer=None, device=resolved_device, **kwargs)

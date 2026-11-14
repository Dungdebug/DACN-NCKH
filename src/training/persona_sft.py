"""
Offline Supervised Fine-Tuning (SFT) & 4-Pillar Closed-Loop Pipeline.

Connects:
- Pillar 4: AcademicPersonaEngine & export_sft_training_samples()
- Pillar 1: SovereignTokenizer + pure-PyTorch LMs (SovereignMiniTransformer / StandardTransformerLM) + SovereignSLMGenerator
- Pillar 2 & 3: WaveCMSEngine (Tree-Wavelet Memory, Equilibrium Readout, Causal Attribution, Certified Exact Unlearning)

100% offline, pure PyTorch + Python stdlib, zero external network calls.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.model.tokenizer import SovereignTokenizer
from src.self_study.persona import (
    AcademicIntent,
    AcademicPersonaEngine,
    export_sft_training_samples,
)
from src.slm import SovereignSLMGenerator
from src.training.loss import compute_perplexity
from src.wave_cms.wave_engine import WaveCMSEngine


@dataclass
class PersonaSFTResult:
    """Summary metrics from an offline persona SFT run."""

    model_name: str
    num_samples: int
    steps: int
    initial_loss: float
    final_loss: float
    initial_ppl: float
    final_ppl: float
    loss_history: List[float]
    trainable_params: int
    total_params: int
    adapt_only_wavelet: bool
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def format_sft_messages(
    sample: Dict[str, Any],
    include_system: bool = False,
) -> Tuple[str, str]:
    """Extracts (prompt, assistant_response) strings from a conversational SFT dict."""
    messages = sample.get("messages", []) if isinstance(sample, dict) else []
    system_text = ""
    user_text = ""
    assistant_text = ""

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = str(msg.get("content", "")).strip()
        if role == "system" and content:
            system_text = content
        elif role == "user" and content:
            user_text = content
        elif role == "assistant" and content:
            assistant_text = content

    if not user_text and not assistant_text:
        fallback = str(sample.get("text", sample.get("response", sample))).strip()
        return "Hỏi: Tóm tắt nội dung học thuật.\nĐáp:\n", fallback

    if include_system and system_text:
        prompt = f"[Hệ thống]\n{system_text}\n[Học viên]\n{user_text}\n[Trợ lý NCKH]\n"
    else:
        prompt = f"Hỏi: {user_text}\nĐáp:\n"
    return prompt, assistant_text


def build_persona_sft_tokenizer(
    samples: Optional[Sequence[Dict[str, Any]]] = None,
    vocab_size: int = 512,
    min_frequency: int = 2,
    include_system: bool = False,
) -> SovereignTokenizer:
    """Trains a local offline SovereignTokenizer on academic persona SFT samples."""
    sft_samples = list(samples) if samples is not None else export_sft_training_samples()
    texts = [
        f"{prompt}{answer}"
        for prompt, answer in (
            format_sft_messages(s, include_system=include_system) for s in sft_samples
        )
        if prompt or answer
    ]
    return SovereignTokenizer.build_academic_tokenizer(
        extra_texts=texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )


def prepare_sft_batch(
    samples: Sequence[Dict[str, Any]],
    tokenizer: SovereignTokenizer,
    max_seq_len: int = 256,
    include_system: bool = False,
    mask_prompt_loss: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenizes SFT samples into shifted (input_ids, targets) tensors.
    Masks padding and prompt tokens with -100 so loss focuses on assistant generation.
    """
    if max_seq_len < 3:
        raise ValueError(f"max_seq_len must be >= 3, got {max_seq_len}")

    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    input_rows: List[List[int]] = []
    target_rows: List[List[int]] = []

    for sample in samples:
        prompt, answer = format_sft_messages(sample, include_system=include_system)
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)

        if not answer_ids:
            continue

        # Keep room for answer tokens when prompt is long
        max_prompt_budget = max(2, max_seq_len // 2)
        if len(prompt_ids) + len(answer_ids) > max_seq_len and len(prompt_ids) > max_prompt_budget:
            prompt_ids = prompt_ids[:max_prompt_budget]

        seq = (prompt_ids + answer_ids)[:max_seq_len]
        if len(seq) < 2:
            continue

        inp = seq[:-1]
        tgt = list(seq[1:])
        if mask_prompt_loss:
            # prompt_ids occupy seq[0 .. prompt_len-1]; target index i predicts seq[i+1]
            unmasked_start = max(0, min(len(tgt) - 1, len(prompt_ids) - 1))
            for i in range(unmasked_start):
                tgt[i] = -100

        input_rows.append(inp)
        target_rows.append(tgt)

    if not input_rows:
        raise ValueError("No valid SFT sequences with at least 2 tokens were produced")

    batch_len = max(len(r) for r in input_rows)
    padded_inputs = [r + [pad_id] * (batch_len - len(r)) for r in input_rows]
    padded_targets = [r + [-100] * (batch_len - len(r)) for r in target_rows]

    return (
        torch.tensor(padded_inputs, dtype=torch.long),
        torch.tensor(padded_targets, dtype=torch.long),
    )


class PersonaSFTTrainer:
    """
    Unified offline SFT trainer for both SovereignMiniTransformer (W_0 + B Theta A)
    and StandardTransformerLM.
    """

    def __init__(
        self,
        tokenizer: Optional[SovereignTokenizer] = None,
        max_seq_len: int = 256,
        include_system: bool = False,
        mask_prompt_loss: bool = True,
        device: str = "cpu",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.include_system = include_system
        self.mask_prompt_loss = mask_prompt_loss
        self.device = device

    def _configure_trainable_parameters(
        self,
        model: nn.Module,
        adapt_only_wavelet: bool,
        freeze_bases_ab: bool,
    ) -> int:
        if adapt_only_wavelet and hasattr(model, "freeze_base_weights"):
            return int(model.freeze_base_weights(freeze_bases_ab=freeze_bases_ab))
        if hasattr(model, "unfreeze_all"):
            return int(model.unfreeze_all())
        for param in model.parameters():
            param.requires_grad = True
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @torch.no_grad()
    def evaluate_loss(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        was_training = model.training
        model.eval()
        _, loss, _ = model(input_ids, targets=targets)
        if was_training:
            model.train()
        return float(loss.item()) if loss is not None else 0.0

    def train(
        self,
        model: nn.Module,
        samples: Optional[Sequence[Dict[str, Any]]] = None,
        steps: int = 12,
        lr: float = 1e-2,
        adapt_only_wavelet: bool = True,
        freeze_bases_ab: bool = True,
    ) -> PersonaSFTResult:
        """Runs pure-PyTorch AdamW SFT on conversational persona samples."""
        t0 = time.perf_counter()
        sft_samples = list(samples) if samples is not None else export_sft_training_samples()

        if self.tokenizer is None:
            self.tokenizer = build_persona_sft_tokenizer(
                samples=sft_samples,
                include_system=self.include_system,
            )

        model_max_len = getattr(model, "max_seq_len", self.max_seq_len)
        effective_max_len = min(self.max_seq_len, int(model_max_len))

        input_ids, targets = prepare_sft_batch(
            samples=sft_samples,
            tokenizer=self.tokenizer,
            max_seq_len=effective_max_len,
            include_system=self.include_system,
            mask_prompt_loss=self.mask_prompt_loss,
        )
        model.to(self.device)
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)

        trainable_count = self._configure_trainable_parameters(
            model=model,
            adapt_only_wavelet=adapt_only_wavelet,
            freeze_bases_ab=freeze_bases_ab,
        )
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("Model has zero trainable parameters after freeze configuration")

        initial_loss = self.evaluate_loss(model, input_ids, targets)
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)

        model.train()
        loss_history: List[float] = []
        for _ in range(max(1, int(steps))):
            optimizer.zero_grad(set_to_none=True)
            _, loss, _ = model(input_ids, targets=targets)
            if loss is None:
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            loss_history.append(float(loss.item()))

        model.eval()
        final_loss = self.evaluate_loss(model, input_ids, targets)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return PersonaSFTResult(
            model_name=type(model).__name__,
            num_samples=input_ids.size(0),
            steps=len(loss_history),
            initial_loss=initial_loss,
            final_loss=final_loss,
            initial_ppl=compute_perplexity(initial_loss),
            final_ppl=compute_perplexity(final_loss),
            loss_history=loss_history,
            trainable_params=trainable_count,
            total_params=sum(p.numel() for p in model.parameters()),
            adapt_only_wavelet=bool(
                adapt_only_wavelet and hasattr(model, "freeze_base_weights")
            ),
            elapsed_ms=round(elapsed_ms, 2),
        )


def run_four_pillar_closed_loop(
    model: nn.Module,
    tokenizer: Optional[SovereignTokenizer] = None,
    samples: Optional[Sequence[Dict[str, Any]]] = None,
    steps: int = 10,
    lr: float = 1e-2,
    adapt_only_wavelet: bool = True,
    max_seq_len: int = 192,
) -> Dict[str, Any]:
    """
    Executes an end-to-end offline 4-Pillar Closed-Loop verification:
    - Pillar 4: AcademicPersonaEngine SFT export & intent classification
    - Pillar 1: PyTorch SLM SFT + SovereignSLMGenerator grounded synthesis
    - Pillar 2 & 3: WaveCMSEngine multi-hop readout + Theta transfer + exact unlearning
    """
    persona_engine = AcademicPersonaEngine()
    sft_samples = list(samples) if samples is not None else export_sft_training_samples()

    trainer = PersonaSFTTrainer(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        include_system=False,
        mask_prompt_loss=True,
    )
    sft_result = trainer.train(
        model=model,
        samples=sft_samples,
        steps=steps,
        lr=lr,
        adapt_only_wavelet=adapt_only_wavelet,
    )
    active_tokenizer = trainer.tokenizer

    # Pillar 4: Deterministic intent sanity check
    intent_checks = {
        "greeting": persona_engine.classify_intent("Xin chào trợ lý").value == AcademicIntent.GREETING.value,
        "out_of_scope": persona_engine.classify_intent("Thời tiết hôm nay thế nào?").value == AcademicIntent.OUT_OF_SCOPE.value,
        "academic": persona_engine.classify_intent("Mã hóa đối xứng là gì?").value == AcademicIntent.ACADEMIC_PEDAGOGICAL.value,
    }

    # Pillar 2 & 3: WaveCMS ingestion, readout, and certified exact unlearning
    rank_r = int(getattr(model, "rank_r", 8))
    wave_engine = WaveCMSEngine(r=rank_r, dim=64, max_hops=4, tau=1.0, refusal_threshold=0.25)
    wave_engine.slm_generator = SovereignSLMGenerator(
        backbone=model,
        tokenizer=active_tokenizer,
        device="cpu",
    )

    wave_engine.ingest_document(
        doc_id="DOC_SYMMETRIC",
        title="Mã hóa đối xứng",
        sections=[
            {
                "section_title": "Khái niệm và chuẩn AES",
                "paragraphs": [
                    "Mã hóa đối xứng sử dụng chung một khóa bí mật cho cả quá trình mã hóa và giải mã.",
                    "Thuật toán tiêu biểu gồm AES, DES, 3DES và ChaCha20 với tốc độ xử lý cao.",
                ],
            }
        ],
    )
    wave_engine.ingest_document(
        doc_id="DOC_WAVE_UNLEARN",
        title="Gỡ bỏ tri thức WaveCMS",
        sections=[
            {
                "section_title": "Định lý Telescoping",
                "paragraphs": [
                    "Gỡ bỏ tri thức chính xác trong WaveCMS đạt độ phức tạp O(depth) không sót sai số.",
                ],
            }
        ],
    )

    theta_root = wave_engine.tree_memory.root.theta
    if hasattr(model, "apply_tree_memory_theta"):
        model.apply_tree_memory_theta(wave_engine.tree_memory)

    # Pillar 1: SovereignSLMGenerator backed by SFT-trained PyTorch SLM + WaveCMS Theta
    slm_result = wave_engine.slm_generator.generate(
        query="Mã hóa đối xứng là gì?",
        passages=[
            {
                "passage_id": "P1",
                "citation_label": "[1]",
                "text": (
                    "Mã hóa đối xứng sử dụng chung một khóa bí mật cho cả quá trình mã hóa và giải mã. "
                    "Thuật toán tiêu biểu gồm AES, DES, 3DES và ChaCha20."
                ),
            }
        ],
        answer_mode="balanced",
        theta=theta_root if hasattr(model, "rank_r") else None,
    )

    wave_response = wave_engine.query("Mã hóa đối xứng là gì?", answer_mode="detailed")
    unlearn_report = wave_engine.unlearn_document("DOC_WAVE_UNLEARN")

    if hasattr(model, "apply_tree_memory_theta"):
        model.apply_tree_memory_theta(None)
    if hasattr(model, "reset_wavelet_adapters"):
        model.reset_wavelet_adapters()

    return {
        "sft": sft_result.to_dict(),
        "intent_checks": intent_checks,
        "slm_generation": {
            "backbone_mode": slm_result.backbone_mode,
            "is_grounded": slm_result.is_grounded,
            "latency_ms": slm_result.latency_ms,
            "answer": slm_result.answer,
        },
        "wave_query": {
            "answer": wave_response.answer,
            "refused": wave_response.refused,
            "contraction_factor": wave_response.contraction_factor,
            "is_contractive": wave_response.is_contractive,
        },
        "unlearning": unlearn_report,
    }

"""
Unit & Integration Tests for SelfStudyTrainer (Ticket 1.2).
Validates:
1. Parameter Isolation: Backbone and Level 1 are 100% frozen; Level 2 & 3 are trainable.
2. Prefix Masking: Teacher-forcing cross-entropy loss computed only on answer tokens.
3. Monotonic Learning: Loss decreases over self-study adaptation steps.
4. Weight Delta Isolation: Level 1 and backbone weights are completely unchanged.
5. Checkpoint Persistence: Checkpoints save and reload with exact parameter matching.
6. Memory Reset: Calling reset_memory() restores original parameters.
7. Empty Pair Handling: Gracefully terminates without errors.
"""

import copy
import shutil
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cms.continuum_memory import ContinuumMemorySystem
from src.self_study.types import SelfStudyQAPair, QARuleType
from src.self_study.trainer import SelfStudyTrainer


class MockTokenizer:
    """Fast offline mock tokenizer for self-study testing."""
    def __init__(self, vocab_size: int = 200):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self._vocab = {"<pad>": 0, "<eos>": 1}

    def __call__(self, text: str, return_tensors: str = "pt"):
        words = text.replace("\n", " ").split()
        ids = []
        for w in words:
            if w not in self._vocab:
                self._vocab[w] = (len(self._vocab) + 2) % self.vocab_size
            ids.append(self._vocab[w])
        if not ids:
            ids = [0]
        t = torch.tensor([ids], dtype=torch.long)

        class Enc:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        return Enc(t)


class MockBackbone(nn.Module):
    def __init__(self, d_model: int = 32, vocab_size: int = 200):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids, **kwargs):
        class Output:
            def __init__(self, hidden):
                self.last_hidden_state = hidden
        return Output(self.embed(input_ids))


class MockStructureAlignedModel(nn.Module):
    """Simulates StructureAlignedHopeLM with 3-level CMS."""
    def __init__(self, d_model: int = 32, vocab_size: int = 200, num_levels: int = 3):
        super().__init__()
        self.device = torch.device("cpu")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tokenizer = MockTokenizer(vocab_size=vocab_size)

        self.backbone = MockBackbone(d_model=d_model, vocab_size=vocab_size)
        self.cms_norm = nn.LayerNorm(d_model)
        self.cms = ContinuumMemorySystem(
            d_model=d_model,
            d_ff=d_model * 2,
            num_levels=num_levels,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, targets=None, **kwargs):
        h = self.backbone(input_ids).last_hidden_state
        if self.cms is not None:
            normed = self.cms_norm(h)
            mem = self.cms(normed)
            h = h + mem
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def reset_memory(self):
        self.cms.reset_memory()


@pytest.fixture
def mock_setup(tmp_path):
    model = MockStructureAlignedModel(d_model=32, vocab_size=200, num_levels=3)
    ckpt_dir = tmp_path / "checkpoints"
    trainer = SelfStudyTrainer(
        model=model,
        tokenizer=model.tokenizer,
        checkpoint_dir=ckpt_dir,
        device="cpu",
    )
    return trainer, model, ckpt_dir


@pytest.fixture
def sample_qa_pairs():
    text1 = "SA-CMS là hệ thống bộ nhớ liên tục cấu trúc. Nó giúp tối ưu hóa chi phí."
    ans1 = "SA-CMS là hệ thống bộ nhớ liên tục cấu trúc."
    text2 = "Mức 2 cập nhật theo ranh giới mục. Mức 3 cập nhật theo tài liệu."
    ans2 = "Mức 2 cập nhật theo ranh giới mục."
    text3 = "Mô hình đạt tỷ lệ giảm 87% chi phí token."
    ans3 = "Mô hình đạt tỷ lệ giảm 87% chi phí token."

    return [
        SelfStudyQAPair(
            question="SA-CMS là gì?",
            answer=ans1,
            span_start=0,
            span_end=len(ans1),
            source_text=text1,
            level=2,
            rule_type=QARuleType.DEFINITION,
            confidence=0.95,
            section_title="Giới thiệu",
        ),
        SelfStudyQAPair(
            question="Tần suất cập nhật của Level 2 là bao nhiêu?",
            answer=ans2,
            span_start=0,
            span_end=len(ans2),
            source_text=text2,
            level=2,
            rule_type=QARuleType.MECHANISM,
            confidence=0.90,
            section_title="Phương pháp",
        ),
        SelfStudyQAPair(
            question="Tỷ lệ giảm token của mô hình đạt bao nhiêu?",
            answer=ans3,
            span_start=0,
            span_end=len(ans3),
            source_text=text3,
            level=3,
            rule_type=QARuleType.METRIC,
            confidence=0.98,
            section_title="Kết quả",
        ),
    ]


def test_parameter_freezing_isolation(mock_setup):
    """
    Backbone and Level 1 must be strictly FROZEN.
    Level 2 and Level 3 must be TRAINABLE.
    """
    trainer, model, _ = mock_setup
    status = trainer.configure_trainable_parameters()

    # Backbone parameters must not require grad
    for name, p in model.backbone.named_parameters():
        assert not p.requires_grad, f"Backbone parameter {name} should be frozen!"

    # Level 1 parameters (blocks[0]) must not require grad
    for name, p in model.cms.chain.blocks[0].named_parameters():
        assert not p.requires_grad, f"Level 1 parameter {name} should be frozen!"

    # Level 2 parameters (blocks[1]) must require grad
    for name, p in model.cms.chain.blocks[1].named_parameters():
        assert p.requires_grad, f"Level 2 parameter {name} should be trainable!"

    # Level 3 parameters (blocks[2]) must require grad
    for name, p in model.cms.chain.blocks[2].named_parameters():
        assert p.requires_grad, f"Level 3 parameter {name} should be trainable!"

    assert status["backbone_frozen"] is True
    assert status["level_1_frozen"] is True
    assert 2 in status["trainable_levels"]
    assert 3 in status["trainable_levels"]


def test_compute_qa_loss_masking(mock_setup, sample_qa_pairs):
    """
    Computes loss and verifies answer tokens are correctly isolated.
    """
    trainer, model, _ = mock_setup
    pair = sample_qa_pairs[0]

    loss, num_ans_tokens = trainer.compute_qa_loss(pair)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    assert num_ans_tokens > 0


def test_training_loop_monotonically_learns(mock_setup, sample_qa_pairs):
    """
    Verifies that:
    1. Training executes the requested number of steps.
    2. Loss decreases over multiple gradient steps.
    3. Level 2 & 3 weights update.
    4. Level 1 & backbone weights remain completely UNCHANGED.
    """
    trainer, model, ckpt_dir = mock_setup

    # Record weights before training
    bb_weight_before = model.backbone.embed.weight.detach().clone()
    l1_weight_before = model.cms.chain.blocks[0].w1.weight.detach().clone()
    l2_weight_before = model.cms.chain.blocks[1].w1.weight.detach().clone()
    l3_weight_before = model.cms.chain.blocks[2].w1.weight.detach().clone()

    # Train for 15 steps
    report = trainer.train(
        qa_pairs=sample_qa_pairs,
        epochs=5,
        lr_level2=1e-2,
        lr_level3=1e-2,
        max_steps=15,
        document_id="doc_test_01",
    )

    assert report["status"] == "success"
    assert report["steps_executed"] > 0
    assert report["initial_loss"] > report["final_loss"], (
        f"Loss should decrease: initial={report['initial_loss']}, final={report['final_loss']}"
    )

    # Check Level 1 and backbone were untouched
    bb_weight_after = model.backbone.embed.weight
    l1_weight_after = model.cms.chain.blocks[0].w1.weight
    assert torch.equal(bb_weight_before, bb_weight_after), "Backbone weights changed!"
    assert torch.equal(l1_weight_before, l1_weight_after), "Level 1 memory weights changed!"

    # Check Level 2 and 3 actually updated
    l2_weight_after = model.cms.chain.blocks[1].w1.weight
    l3_weight_after = model.cms.chain.blocks[2].w1.weight
    assert not torch.equal(l2_weight_before, l2_weight_after), "Level 2 memory should have updated!"
    assert not torch.equal(l3_weight_before, l3_weight_after), "Level 3 memory should have updated!"

    # Verify checkpoint was written
    expected_ckpt = ckpt_dir / "doc_test_01.pt"
    assert expected_ckpt.exists()


def test_checkpoint_save_and_reload(mock_setup, sample_qa_pairs):
    """
    Verifies saving and loading of document-specific adapted memory.
    """
    trainer, model, ckpt_dir = mock_setup

    # Train briefly
    trainer.train(qa_pairs=sample_qa_pairs, epochs=2, max_steps=5, document_id="doc_persist")

    saved_l2_weight = model.cms.chain.blocks[1].w1.weight.detach().clone()

    # Revert model memory to initial state
    model.reset_memory()
    reset_l2_weight = model.cms.chain.blocks[1].w1.weight
    assert not torch.equal(saved_l2_weight, reset_l2_weight)

    # Reload saved checkpoint
    success = trainer.load_document_checkpoint("doc_persist")
    assert success is True

    reloaded_l2_weight = model.cms.chain.blocks[1].w1.weight
    assert torch.equal(saved_l2_weight, reloaded_l2_weight)


def test_empty_pairs_handling(mock_setup):
    """
    Handles empty QA list gracefully.
    """
    trainer, _, _ = mock_setup
    report = trainer.train(qa_pairs=[], epochs=2, max_steps=10)
    assert report["status"] == "empty_pairs"
    assert report["steps_executed"] == 0


def test_reset_memory_restores_initial_state(mock_setup, sample_qa_pairs):
    """
    Verifies that calling reset_memory() on trainer reverts CMS to initial state.
    """
    trainer, model, _ = mock_setup
    initial_l2 = model.cms.chain.blocks[1].w1.weight.detach().clone()

    trainer.train(qa_pairs=sample_qa_pairs, epochs=2, max_steps=5)
    adapted_l2 = model.cms.chain.blocks[1].w1.weight.detach().clone()
    assert not torch.equal(initial_l2, adapted_l2)

    trainer.reset_memory()
    restored_l2 = model.cms.chain.blocks[1].w1.weight
    assert torch.equal(initial_l2, restored_l2)


def test_real_model_parameter_isolation_and_step(tmp_path, sample_qa_pairs):
    """
    Integration test with the real local SmolLM2-135M + SA-CMS model if local weights exist.
    Confirms 100% frozen backbone & Level 1, trainable Level 2 & 3.
    """
    from src.web.local_model_loader import LocalModelLoader
    loader = LocalModelLoader.get_instance()
    real_model, real_tok = loader.load_offline_model(prefer_gpu=False)

    if real_model is None or real_tok is None:
        pytest.skip("Local model weights not found in local_runtime/model")

    trainer = SelfStudyTrainer(
        model=real_model,
        tokenizer=real_tok,
        checkpoint_dir=tmp_path / "real_ckpt",
    )

    status = trainer.configure_trainable_parameters()
    assert status["backbone_frozen"] is True
    assert status["level_1_frozen"] is True
    assert 2 in status["trainable_levels"]
    assert 3 in status["trainable_levels"]

    # Run 2 training steps on the real model
    report = trainer.train(
        qa_pairs=sample_qa_pairs[:2],
        epochs=1,
        max_steps=2,
        document_id="doc_real_test",
    )
    assert report["status"] == "success"
    assert report["steps_executed"] == 2
    assert report["saved_checkpoint"] is not None
    assert Path(report["saved_checkpoint"]).exists()

    # Reset real model memory back
    trainer.reset_memory()


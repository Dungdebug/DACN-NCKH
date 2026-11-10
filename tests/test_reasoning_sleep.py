"""
Unit Tests for SleepReplayEngine & Synaptic Consolidation (Ticket 3.2).
Validates:
1. Fisher Information diagonal estimation for Level 2 & 3 CMS blocks.
2. Quadratic EWC penalty computation.
3. Sleep consolidation rehearsal loop.
4. Synaptic delta threshold pruning.
5. Empty pairs edge cases.
"""

import pytest
import torch

from src.self_study.types import SelfStudyQAPair, QARuleType
from src.self_study.trainer import SelfStudyTrainer
from src.reasoning.sleep_replay import SleepReplayEngine, ConsolidationReport
from tests.test_self_study_trainer import MockStructureAlignedModel


@pytest.fixture
def mock_setup():
    model = MockStructureAlignedModel(d_model=32, vocab_size=200, num_levels=3)
    trainer = SelfStudyTrainer(
        model=model,
        tokenizer=model.tokenizer,
        device="cpu",
    )
    engine = SleepReplayEngine(model=model, device="cpu", ewc_lambda=100.0)
    return engine, model, trainer


@pytest.fixture
def sample_pairs():
    text1 = "SA-CMS là hệ thống bộ nhớ liên tục đa tầng."
    ans1 = "SA-CMS là hệ thống bộ nhớ liên tục đa tầng."
    text2 = "Mức 2 cập nhật theo ranh giới mục."
    ans2 = "Mức 2 cập nhật theo ranh giới mục."

    return [
        SelfStudyQAPair(
            question="SA-CMS là gì?",
            answer=ans1,
            span_start=0,
            span_end=len(ans1),
            source_text=text1,
            document_id="doc_mem_1",
            level=2,
            rule_type=QARuleType.DEFINITION,
        ),
        SelfStudyQAPair(
            question="Mức 2 cập nhật thế nào?",
            answer=ans2,
            span_start=0,
            span_end=len(ans2),
            source_text=text2,
            document_id="doc_mem_2",
            level=2,
            rule_type=QARuleType.MECHANISM,
        ),
    ]


def test_fisher_information_computation(mock_setup, sample_pairs):
    engine, model, trainer = mock_setup

    fisher = engine.calculate_fisher_information(sample_pairs, trainer=trainer)
    assert len(fisher) > 0

    for name, f_tensor in fisher.items():
        assert isinstance(f_tensor, torch.Tensor)
        assert torch.all(f_tensor >= 0.0), f"Negative Fisher entry in {name}"
        assert f_tensor.shape == engine.optimal_params[name].shape


def test_ewc_loss_computation(mock_setup, sample_pairs):
    engine, model, trainer = mock_setup

    # 1. Compute Fisher info
    engine.calculate_fisher_information(sample_pairs, trainer=trainer)

    # Initially at optimal params theta^*, EWC loss must be 0.0
    ewc_init = engine.compute_ewc_loss()
    assert torch.isclose(ewc_init, torch.tensor(0.0), atol=1e-6)

    # Perturb Level 2 weights away from theta^*
    with torch.no_grad():
        model.cms.chain.blocks[1].w1.weight.add_(0.05)

    # EWC loss must now be positive
    ewc_perturbed = engine.compute_ewc_loss()
    assert ewc_perturbed.item() > 0.0


def test_sleep_consolidation_cycle(mock_setup, sample_pairs):
    engine, model, trainer = mock_setup

    report = engine.consolidate(
        sample_pairs=sample_pairs,
        trainer=trainer,
        steps=3,
        lr=1e-3,
    )

    assert isinstance(report, ConsolidationReport)
    assert report.status == "success"
    assert report.sample_pairs_count == len(sample_pairs)
    assert report.fisher_parameters_count > 0
    assert report.fisher_mean_importance > 0.0
    assert "doc_mem_1" in report.documents_replayed
    assert "doc_mem_2" in report.documents_replayed
    assert report.duration_seconds > 0.0


def test_synaptic_delta_pruning(mock_setup, sample_pairs):
    engine, model, trainer = mock_setup

    engine.calculate_fisher_information(sample_pairs, trainer=trainer)

    # Introduce a tiny delta (1e-6 < 1e-4) into Level 2 weights
    with torch.no_grad():
        model.cms.chain.blocks[1].w1.weight[0, 0] += 1e-6

    # Prune deltas
    pruned_count = engine.prune_synaptic_deltas(threshold=1e-4)
    assert pruned_count >= 1

    # Verify weight restored to exact optimal value
    opt_val = engine.optimal_params["level_2.w1.weight"][0, 0]
    curr_val = model.cms.chain.blocks[1].w1.weight[0, 0]
    assert torch.equal(opt_val, curr_val)


def test_empty_pairs_handling(mock_setup):
    engine, _, _ = mock_setup
    report = engine.consolidate([], steps=3)
    assert report.status == "no_pairs_to_replay"
    assert report.sample_pairs_count == 0


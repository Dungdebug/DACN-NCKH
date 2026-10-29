"""
Unit tests for MemoryMechanisticInspector.
Verifies calculation of residual norm, hidden state shift, logits shift, and level contributions.
"""

import pytest
import torch
import torch.nn as nn

from src.memory.mechanistic_inspector import MemoryMechanisticInspector
from src.cms.continuum_memory import ContinuumMemorySystem


def test_mechanistic_inspector_mlp_chain():
    inspector = MemoryMechanisticInspector(d_model=64, num_levels=3)
    cms = ContinuumMemorySystem(d_model=64, d_ff=128, num_levels=3)

    x = torch.randn(1, 10, 64)
    res = inspector.inspect_mlp_chain(cms, x)

    assert "total_residual_norm" in res
    assert "level_norms" in res
    assert len(res["level_norms"]) == 3
    assert "level_contributions_pct" in res
    assert "level_1_paragraph" in res["level_contributions_pct"]
    assert "level_2_section" in res["level_contributions_pct"]
    assert "level_3_document" in res["level_contributions_pct"]
    assert len(res["heatmap_tokens_x_levels"]) == 10


def test_mechanistic_inspector_hidden_shift():
    inspector = MemoryMechanisticInspector(d_model=64)
    h_base = torch.randn(1, 8, 64)
    h_mem = h_base + torch.randn(1, 8, 64) * 0.1

    shift = inspector.inspect_hidden_state_shift(h_base, h_mem)
    assert shift["absolute_hidden_difference_norm"] > 0
    assert 0.0 <= shift["cosine_alignment"] <= 1.0


def test_mechanistic_inspector_logits_shift():
    inspector = MemoryMechanisticInspector(d_model=64)
    logits_a = torch.randn(1, 1, 100)
    logits_b = logits_a + torch.randn(1, 1, 100) * 0.5

    l_res = inspector.inspect_logits_shift(logits_a, logits_b)
    assert "kl_divergence" in l_res
    assert "top_k_rank_overlap" in l_res
    assert isinstance(l_res["top_1_token_changed"], bool)


def test_synthesize_mechanistic_profile():
    inspector = MemoryMechanisticInspector(d_model=576)
    evicted_prof = inspector.synthesize_mechanistic_profile(context_status="evicted")
    present_prof = inspector.synthesize_mechanistic_profile(context_status="present")

    assert evicted_prof["memory_residual_norm"] > present_prof["memory_residual_norm"]
    assert len(evicted_prof["heatmap"]) == 8

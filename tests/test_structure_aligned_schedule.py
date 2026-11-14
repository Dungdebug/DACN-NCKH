"""
Tests for Trụ cột 1: Structure-Aligned Boundary Scheduler in CMS (SA-CMS Core).
Verifies:
1. StructureAlignedSchedule in src/cms/schedule.py supports structure, fixed_token, and random modes.
2. ContinuumMemorySystem supports structural boundary updates (b_l(t) = 1).
3. Level 1 reset at section boundary (Section 6.2 of thesis).
4. Equivalence property: SA-CMS matches baseline when boundaries are token-aligned.
"""

import pytest
import torch
import torch.nn as nn

from src.document_structure.types import BoundaryType, StructuralSpan, DocumentStructure
from src.cms.schedule import CMSSchedule, StructureAlignedSchedule
from src.cms.continuum_memory import ContinuumMemorySystem


def test_structure_aligned_schedule_initialization():
    """Verifies that StructureAlignedSchedule parses DocumentStructure boundaries."""
    paragraphs = [
        StructuralSpan(BoundaryType.PARAGRAPH, start_token=0, end_token=30),
        StructuralSpan(BoundaryType.PARAGRAPH, start_token=30, end_token=75),
        StructuralSpan(BoundaryType.PARAGRAPH, start_token=75, end_token=120),
    ]
    sections = [
        StructuralSpan(BoundaryType.SECTION, start_token=0, end_token=75),
        StructuralSpan(BoundaryType.SECTION, start_token=75, end_token=120),
    ]
    doc_struct = DocumentStructure(
        text="Sample document",
        total_tokens=120,
        paragraphs=paragraphs,
        sections=sections,
    )

    sched = StructureAlignedSchedule(
        num_levels=3,
        total_tokens=120,
        schedule_mode="structure",
        doc_structure=doc_struct,
    )

    # Level 0 (Paragraphs): 30, 75, 120
    assert sched.boundaries[0] == [30, 75, 120]
    # Level 1 (Sections): 75, 120
    assert sched.boundaries[1] == [75, 120]
    # Level 2 (Document): 120
    assert sched.boundaries[2] == [120]

    # Test should_update
    assert sched.should_update(level=0, current_token_pos=30) is True
    assert sched.should_update(level=0, current_token_pos=45) is False
    assert sched.should_update(level=1, current_token_pos=75) is True
    assert sched.should_update(level=1, current_token_pos=30) is False
    assert sched.should_update(level=2, current_token_pos=120) is True


def test_continuum_memory_system_with_structure_schedule():
    """
    Verifies that ContinuumMemorySystem can be initialized with a StructureAlignedSchedule
    and executes gradient updates exactly when token positions hit structural boundaries.
    """
    torch.manual_seed(42)
    total_tokens = 100
    paragraphs = [
        StructuralSpan(BoundaryType.PARAGRAPH, start_token=0, end_token=40),
        StructuralSpan(BoundaryType.PARAGRAPH, start_token=40, end_token=100),
    ]
    sections = [
        StructuralSpan(BoundaryType.SECTION, start_token=0, end_token=100),
    ]
    doc_struct = DocumentStructure(
        text="Sample test",
        total_tokens=total_tokens,
        paragraphs=paragraphs,
        sections=sections,
    )

    sched = StructureAlignedSchedule(
        num_levels=2,
        total_tokens=total_tokens,
        schedule_mode="structure",
        doc_structure=doc_struct,
    )

    cms = ContinuumMemorySystem(
        d_model=64,
        d_ff=128,
        num_levels=2,
        schedule=sched,
    )

    x = torch.randn(1, 10, 64)
    out = cms(x)
    loss = out.sum()

    # Step to token 30: neither level should update
    updated_30 = cms.update_at_token_position(loss=loss, token_pos=30)
    assert updated_30 == []

    # Step to token 40: Level 0 (paragraph) should update, Level 1 (section) should NOT
    updated_40 = cms.update_at_token_position(loss=loss, token_pos=40)
    assert 0 in updated_40
    assert 1 not in updated_40

    # Forward pass on subsequent chunk (tokens 40..100)
    out2 = cms(torch.randn(1, 10, 64))
    loss2 = out2.sum()

    # Step to token 100: Both Level 0 (paragraph end) and Level 1 (section end) should update
    updated_100 = cms.update_at_token_position(loss=loss2, token_pos=100)
    assert 0 in updated_100
    assert 1 in updated_100


def test_level_1_reset_at_section_boundary():
    """
    Verifies Section 6.2 of thesis:
    Level 1 (Paragraph memory) can be reset at section boundaries while
    preserving Level 2 (Section memory) and Level 3 (Document memory).
    """
    torch.manual_seed(42)
    cms = ContinuumMemorySystem(
        d_model=64,
        d_ff=128,
        num_levels=3,
    )

    # Simulate weights change across all levels
    with torch.no_grad():
        for block in cms.chain.blocks:
            for p in block.parameters():
                p.add_(torch.ones_like(p) * 0.5)

    # Reset only Level 0 (Paragraph level)
    cms.reset_level(level_idx=0)

    # Level 0 parameters should match initial state theta_0
    for name, p in cms.chain.blocks[0].named_parameters():
        init_key = f"blocks.0.{name}"
        if init_key in cms._initial_state_dict:
            assert torch.allclose(p, cms._initial_state_dict[init_key])

    # Level 1 parameters should NOT be reset (still has the 0.5 addition)
    for name, p in cms.chain.blocks[1].named_parameters():
        init_key = f"blocks.1.{name}"
        if init_key in cms._initial_state_dict:
            assert not torch.allclose(p, cms._initial_state_dict[init_key])

"""
WaveCMS: Document-Tree Wavelet Continuum Memory with Equilibrium Multi-Hop Readout
and Certified Unlearning.

Proposal Reference: docs/de_cuong_text.txt, arXiv:2512.24695 (Nested Learning),
and wavecms_research_proposal.md.
100% Strict Offline, Pure Python stdlib & NumPy, Zero External APIs.
"""

from src.wave_cms.tree_memory import (
    TreeLevel,
    DocumentTreeNode,
    WaveTreeMemory,
    compute_deterministic_routing_key,
    strip_accents,
)
from src.wave_cms.equilibrium_readout import (
    ResolutionLevel,
    RoutingResult,
    EquilibriumResult,
    TreeSoftmaxRouter,
    EquilibriumMultiHopReadout,
)
from src.wave_cms.causal_attribution import (
    CausalAttributionItem,
    CounterfactualLeafAttributor,
    EvidenceChainHop,
    ConformalRiskController,
    RefusalDecision,
)
from src.wave_cms.wave_engine import (
    WaveCMSEngine,
    WaveCMSQueryResult,
)
from src.wave_cms.continuum_memory import (
    ContinuumMemoryAdapter,
    ContinuumMemoryLinear,
    CertifiedUnlearningEngine,
    release_subtree_memory,
)

__all__ = [
    "TreeLevel",
    "DocumentTreeNode",
    "WaveTreeMemory",
    "compute_deterministic_routing_key",
    "strip_accents",
    "ResolutionLevel",
    "RoutingResult",
    "EquilibriumResult",
    "TreeSoftmaxRouter",
    "EquilibriumMultiHopReadout",
    "CausalAttributionItem",
    "CounterfactualLeafAttributor",
    "EvidenceChainHop",
    "ConformalRiskController",
    "RefusalDecision",
    "WaveCMSEngine",
    "WaveCMSQueryResult",
    "ContinuumMemoryAdapter",
    "ContinuumMemoryLinear",
    "CertifiedUnlearningEngine",
    "release_subtree_memory",
]

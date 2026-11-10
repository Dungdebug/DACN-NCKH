"""
Neuro-Symbolic Reasoning Module for SA-CMS.
Implements Concept Hypergraph, Multi-Hop Path Reasoning, and Cross-Document Contradiction Detection.
100% Strict Offline, Local Pure Python & PyTorch Data Structures.
"""

from src.reasoning.hypergraph import (
    RelationType,
    ConceptNode,
    RelationEdge,
    MultiHopChain,
    ConceptHypergraph,
)
from src.reasoning.cross_doc_scanner import (
    ContradictionType,
    ContradictionAlert,
    ContradictionScanReport,
    CrossDocumentContradictionScanner,
)
from src.reasoning.entropy_gate import (
    EntropySurprisalGate,
    SurprisalResult,
)
from src.reasoning.sleep_replay import (
    SleepReplayEngine,
    ConsolidationReport,
)
from src.reasoning.dialogue_tracker import (
    DialogueTurn,
    DialogueSessionState,
    ParametricDialogueTracker,
)
from src.reasoning.meta_cognition import (
    MetaCognitiveReport,
    MetaCognitiveEvaluator,
)

__all__ = [
    "RelationType",
    "ConceptNode",
    "RelationEdge",
    "MultiHopChain",
    "ConceptHypergraph",
    "ContradictionType",
    "ContradictionAlert",
    "ContradictionScanReport",
    "CrossDocumentContradictionScanner",
    "EntropySurprisalGate",
    "SurprisalResult",
    "SleepReplayEngine",
    "ConsolidationReport",
    "DialogueTurn",
    "DialogueSessionState",
    "ParametricDialogueTracker",
    "MetaCognitiveReport",
    "MetaCognitiveEvaluator",
]

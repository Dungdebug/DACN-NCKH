"""
Self-Study & Neural Reader Module (Sections 6.4 and 6.5 of Research Proposal).
100% Strict Offline, Zero External API, Local PyTorch & Rule-Based Operations.
"""

from src.self_study.types import (
    QARuleType,
    SelfStudyQAPair,
    SelfStudyReport,
)
from src.self_study.extractor import QAPairExtractor
from src.self_study.trainer import SelfStudyTrainer
from src.self_study.reader import (
    NeuralReaderEngine,
    NeuralReaderResult,
    detect_query_intent,
    split_clean_sentences,
)

__all__ = [
    "QARuleType",
    "SelfStudyQAPair",
    "SelfStudyReport",
    "QAPairExtractor",
    "SelfStudyTrainer",
    "NeuralReaderEngine",
    "NeuralReaderResult",
    "detect_query_intent",
    "split_clean_sentences",
]


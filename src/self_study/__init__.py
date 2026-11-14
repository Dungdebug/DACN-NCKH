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
from src.self_study.knowledge_bank import (
    ANSWER_INSTRUCTION_BLUEPRINTS,
    SelfStudyKnowledgeBank,
    get_knowledge_bank,
)
from src.self_study.datasets import (
    expand_domain_query_terms,
    extract_taxonomy_and_types,
    extract_examples_and_formulas,
    normalize_student_question,
    QuestionParaphraseNormalizer,
    SelfCurriculumGenerator,
    CurriculumLevel,
    ExtractiveTextStitcher,
    # Wave 5.5 Multi-Domain
    AcademicDomain,
    AcademicDomainRegistry,
    expand_all_academic_synonyms,
    parse_vietnamese_academic_blocks,
    synthesize_pros_cons_and_comparison,
    synthesize_step_by_step_process,
)
from src.self_study.persona import (
    AcademicIntent,
    PedagogicalTier,
    PedagogicalResponse,
    AcademicPersonaEngine,
    ACADEMIC_PERSONA_DATASET,
    get_academic_persona_dataset,
    export_sft_training_samples,
    get_academic_persona_engine,
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
    "ANSWER_INSTRUCTION_BLUEPRINTS",
    "SelfStudyKnowledgeBank",
    "get_knowledge_bank",
    "expand_domain_query_terms",
    "extract_taxonomy_and_types",
    "extract_examples_and_formulas",
    "normalize_student_question",
    "QuestionParaphraseNormalizer",
    "SelfCurriculumGenerator",
    "CurriculumLevel",
    "ExtractiveTextStitcher",
    "AcademicDomain",
    "AcademicDomainRegistry",
    "expand_all_academic_synonyms",
    "parse_vietnamese_academic_blocks",
    "synthesize_pros_cons_and_comparison",
    "synthesize_step_by_step_process",
    # Pillar 4 Academic Persona
    "AcademicIntent",
    "PedagogicalTier",
    "PedagogicalResponse",
    "AcademicPersonaEngine",
    "ACADEMIC_PERSONA_DATASET",
    "get_academic_persona_dataset",
    "export_sft_training_samples",
    "get_academic_persona_engine",
]

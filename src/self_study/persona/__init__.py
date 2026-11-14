"""
Academic Persona Package (Trụ cột 4 - Sovereign Brain v5.5).
100% Offline, Pure Python stdlib, Zero Cloud API.
"""

from src.self_study.persona.academic_persona import (
    AcademicIntent,
    PedagogicalTier,
    PedagogicalResponse,
    AcademicPersonaEngine,
    ACADEMIC_PERSONA_DATASET,
    get_academic_persona_dataset,
    export_sft_training_samples,
    export_few_shot_prompt,
    get_academic_persona_engine,
    strip_accents,
    clean_whitespace,
)

__all__ = [
    "AcademicIntent",
    "PedagogicalTier",
    "PedagogicalResponse",
    "AcademicPersonaEngine",
    "ACADEMIC_PERSONA_DATASET",
    "get_academic_persona_dataset",
    "export_sft_training_samples",
    "export_few_shot_prompt",
    "get_academic_persona_engine",
    "strip_accents",
    "clean_whitespace",
]

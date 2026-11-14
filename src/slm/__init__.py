"""
Sovereign Offline SLM/LLM Module (Pillar 1).
Exposes the SovereignSLMGenerator and generation telemetry.
"""

from src.slm.sovereign_generator import (
    SovereignGenerationResult,
    SovereignSLMGenerator,
)

__all__ = [
    "SovereignSLMGenerator",
    "SovereignGenerationResult",
]

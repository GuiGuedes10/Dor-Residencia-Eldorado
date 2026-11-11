"""
REIA Voice Navigation Pipeline Subsystem (Milestone 4).

Provides:
- VoiceNavigationPipeline: End-to-end orchestrator connecting audio validation,
  sanitization, neural classification, and web navigation.
- PipelineCoordinator: Interface-compatible alias for VoiceNavigationPipeline.
- PipelineResult: Structured result dataclass with metadata and dual access pattern.
- Messages and user feedback routines.
"""

from reia.pipeline.coordinator import (
    PipelineCoordinator,
    PipelineResult,
    SYNTHETIC_SCENARIOS,
    VoiceNavigationPipeline,
)
from reia.pipeline.messages import (
    FEEDBACK_MAP,
    MSG_AUDIO_CORRUPTED,
    MSG_AUDIO_EMPTY,
    MSG_AUDIO_NOISE,
    MSG_AUDIO_SILENCE,
    MSG_AUDIO_TOO_LONG,
    MSG_AUDIO_TOO_SHORT,
    MSG_AUDIO_VALID,
    MSG_NAVIGATION_ERROR,
    MSG_NAVIGATION_SKIPPED,
    MSG_NAVIGATION_SUCCESS_FULL,
    MSG_NAVIGATION_SUCCESS_SHORT,
    MSG_NEURAL_AMBIGUOUS_DOMAIN,
    MSG_NEURAL_AMBIGUOUS_INTENT,
    MSG_NEURAL_EMPTY_TEXT,
    MSG_NEURAL_LOW_CONFIDENCE,
    MSG_NEURAL_OOD,
    get_audio_feedback_message,
    get_navigation_feedback_message,
    get_neural_feedback_message,
)

__all__ = [
    "VoiceNavigationPipeline",
    "PipelineCoordinator",
    "PipelineResult",
    "SYNTHETIC_SCENARIOS",
    "FEEDBACK_MAP",
    "MSG_AUDIO_VALID",
    "MSG_AUDIO_SILENCE",
    "MSG_AUDIO_NOISE",
    "MSG_AUDIO_TOO_SHORT",
    "MSG_AUDIO_TOO_LONG",
    "MSG_AUDIO_CORRUPTED",
    "MSG_AUDIO_EMPTY",
    "MSG_NEURAL_OOD",
    "MSG_NEURAL_AMBIGUOUS_INTENT",
    "MSG_NEURAL_AMBIGUOUS_DOMAIN",
    "MSG_NEURAL_LOW_CONFIDENCE",
    "MSG_NEURAL_EMPTY_TEXT",
    "MSG_NAVIGATION_SUCCESS_SHORT",
    "MSG_NAVIGATION_SUCCESS_FULL",
    "MSG_NAVIGATION_ERROR",
    "MSG_NAVIGATION_SKIPPED",
    "get_audio_feedback_message",
    "get_neural_feedback_message",
    "get_navigation_feedback_message",
]

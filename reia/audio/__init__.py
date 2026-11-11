"""
Audio ingestion, validation, sanitization, and synthetic generator subsystem for REIA.
"""

from reia.audio.sources import (
    AudioSource,
    FileAudioSource,
    BytesAudioSource,
    MicrophoneAudioSource,
)
from reia.audio.validation import (
    AudioStatus,
    AudioValidationResult,
    AudioMetrics,
    validate_and_process_audio,
    validate_audio,
    validate_or_raise,
)
from reia.audio.sanitization import sanitize_transcription
from reia.audio.synthetic import SyntheticAudioGenerator

__all__ = [
    "AudioSource",
    "FileAudioSource",
    "BytesAudioSource",
    "MicrophoneAudioSource",
    "AudioStatus",
    "AudioValidationResult",
    "AudioMetrics",
    "validate_and_process_audio",
    "validate_audio",
    "validate_or_raise",
    "sanitize_transcription",
    "SyntheticAudioGenerator",
]

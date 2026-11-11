"""
Core configuration and exception hierarchy for REIA.
"""

from reia.core.config import AudioConfig, TargetUrls, TARGET_URLS, config
from reia.core.exceptions import (
    AudioValidationError,
    SilenceDetectedError,
    ExcessiveNoiseError,
    AudioDurationError,
    AudioTooShortError,
    AudioTooLongError,
    CorruptedAudioError,
    EmptyAudioError,
)

__all__ = [
    "AudioConfig",
    "TargetUrls",
    "TARGET_URLS",
    "config",
    "AudioValidationError",
    "SilenceDetectedError",
    "ExcessiveNoiseError",
    "AudioDurationError",
    "AudioTooShortError",
    "AudioTooLongError",
    "CorruptedAudioError",
    "EmptyAudioError",
]

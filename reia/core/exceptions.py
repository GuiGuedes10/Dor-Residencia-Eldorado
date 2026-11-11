"""
Custom exception hierarchy for REIA audio validation and pipeline errors.
"""

from typing import Optional, Any


class AudioValidationError(Exception):
    """Base exception for all audio validation errors."""
    def __init__(self, message: str, result: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.result = result


class SilenceDetectedError(AudioValidationError):
    """Raised when audio input contains predominant silence or is inaudible."""
    pass


class ExcessiveNoiseError(AudioValidationError):
    """Raised when background noise floor obscures the speech signal (low SNR)."""
    pass


class AudioDurationError(AudioValidationError):
    """Raised when audio duration falls outside allowed thresholds."""
    pass


class AudioTooShortError(AudioDurationError):
    """Raised when audio is shorter than the minimum duration threshold."""
    pass


class AudioTooLongError(AudioDurationError):
    """Raised when audio is longer than the maximum duration threshold."""
    pass


class CorruptedAudioError(AudioValidationError):
    """Raised when audio stream or file header cannot be decoded."""
    pass


class EmptyAudioError(AudioValidationError):
    """Raised when received audio buffer contains 0 bytes."""
    pass

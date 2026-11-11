"""
Synthetic Audio Signal Generator for REIA.

Generates deterministic 16kHz 16-bit mono WAV byte streams for unit and
integration tests without requiring external audio hardware or audioop.
"""

import io
import math
import random
import struct
from typing import List, Optional
import wave


class SyntheticAudioGenerator:
    """
    Deterministic procedural generator of audio byte streams for test automation.
    Produces valid and invalid WAV streams conforming to REIA constraints.
    """

    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    SAMPWIDTH: int = 2  # 16-bit PCM

    @classmethod
    def _create_wav(cls, samples: List[float], sample_rate: int = 16000) -> bytes:
        """
        Converts a list of float samples in [-1.0, 1.0] to a 16-bit PCM WAV byte stream.
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(cls.CHANNELS)
            wf.setsampwidth(cls.SAMPWIDTH)
            wf.setframerate(sample_rate)
            raw = bytearray()
            for s in samples:
                # Clamp between -1.0 and 1.0
                clamped = max(min(s, 1.0), -1.0)
                int_val = int(clamped * 32767)
                raw.extend(struct.pack("<h", int_val))
            wf.writeframes(raw)
        return buf.getvalue()

    @classmethod
    def generate_silence(cls, duration_sec: float = 2.0) -> bytes:
        """
        Generates pure digital silence (all zero amplitudes).
        """
        num_samples = int(duration_sec * cls.SAMPLE_RATE)
        return cls._create_wav([0.0] * num_samples, cls.SAMPLE_RATE)

    @classmethod
    def generate_white_noise(
        cls,
        duration_sec: float = 2.0,
        amplitude: float = 0.7,
        seed: int = 42,
    ) -> bytes:
        """
        Generates stationary white noise with uniform distribution and low SNR.
        """
        rng = random.Random(seed)
        num_samples = int(duration_sec * cls.SAMPLE_RATE)
        samples = [rng.uniform(-amplitude, amplitude) for _ in range(num_samples)]
        return cls._create_wav(samples, cls.SAMPLE_RATE)

    @classmethod
    def generate_mock_speech(
        cls,
        duration_sec: float = 3.0,
        pitch_hz: float = 160.0,
    ) -> bytes:
        """
        Generates deterministic synthetic speech-like acoustic pattern with
        harmonic formant resonances, amplitude envelope, and natural pauses.
        Meets all valid audio criteria: active voice > 0.4s, silence ratio < 0.85,
        and high SNR (> 15 dB).
        """
        num_samples = int(duration_sec * cls.SAMPLE_RATE)
        samples: List[float] = []

        # Active speech envelope between 0.3s and duration - 0.3s
        start_active = 0.3
        end_active = max(start_active + 0.5, duration_sec - 0.3)
        active_window = end_active - start_active

        for i in range(num_samples):
            t = i / cls.SAMPLE_RATE
            if start_active <= t <= end_active:
                # Sinusoidal syllabic envelope modulation (~3.5 Hz speech syllable rhythm)
                syllable_mod = 0.5 * (1.0 + math.sin(2 * math.pi * 3.5 * t))
                window_env = math.sin(math.pi * ((t - start_active) / active_window))
                envelope = window_env * (0.3 + 0.7 * syllable_mod)

                # Harmonic multi-formant synthetic vowels
                tone = (
                    0.50 * math.sin(2 * math.pi * pitch_hz * t)
                    + 0.30 * math.sin(2 * math.pi * (2 * pitch_hz) * t)
                    + 0.15 * math.sin(2 * math.pi * (4 * pitch_hz) * t)
                    + 0.05 * math.sin(2 * math.pi * 2200 * t)
                )
                sample = envelope * tone * 0.75
            else:
                # Very low quiet ambient room floor
                sample = 0.0005 * math.sin(2 * math.pi * 60.0 * t)

            samples.append(sample)

        return cls._create_wav(samples, cls.SAMPLE_RATE)

    @classmethod
    def generate_short_audio(cls, duration_sec: float = 0.3) -> bytes:
        """
        Generates audio shorter than the minimum duration threshold (0.8s).
        """
        return cls.generate_mock_speech(duration_sec=duration_sec)

    @classmethod
    def generate_long_audio(cls, duration_sec: float = 15.0) -> bytes:
        """
        Generates audio longer than the maximum duration threshold (12.0s).
        """
        return cls.generate_mock_speech(duration_sec=duration_sec)

    @classmethod
    def generate_corrupted_audio(cls, corrupt_type: str = "truncated_header") -> bytes:
        """
        Generates corrupted or non-standard audio byte sequences.

        Args:
            corrupt_type: 'truncated_header', 'random_bytes', or 'empty'.
        """
        if corrupt_type == "truncated_header":
            return b"RIFF\x24\x00\x00\x00WAVEfmt "
        elif corrupt_type == "random_bytes":
            return b"THIS_IS_DEFINITELY_NOT_A_VALID_RIFF_WAV_HEADER_1234567890"
        elif corrupt_type == "truncated_frames":
            # Generate valid WAV but chop off 90% of data bytes
            valid = cls.generate_mock_speech(duration_sec=1.0)
            return valid[: len(valid) // 4]
        elif corrupt_type == "empty":
            return b""
        return b"INVALID_CORRUPTED_BYTES"

    # Aliases for compatibility
    generate_speech_mock = generate_mock_speech
    generate_too_short = generate_short_audio
    generate_too_long = generate_long_audio
    generate_corrupted = generate_corrupted_audio

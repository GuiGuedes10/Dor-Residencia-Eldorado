"""
Audio ingestion abstractions for REIA.

Provides unified AudioSource interface supporting local files, raw byte buffers,
and local microphone capture with clean fallbacks for headless/CI environments.
"""

from abc import ABC, abstractmethod
import io
import os
import wave
from typing import Optional, Union


class AudioSource(ABC):
    """Abstract base class for all REIA audio input sources."""

    @abstractmethod
    def read_bytes(self) -> bytes:
        """
        Reads and returns raw audio file bytes (e.g. WAV format).

        Returns:
            bytes: The binary audio data.
        """
        pass


class FileAudioSource(AudioSource):
    """
    Audio source reading from a local filesystem file.
    """

    def __init__(self, file_path: Union[str, os.PathLike]):
        self.file_path = str(file_path)

    def read_bytes(self) -> bytes:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {self.file_path}")
        with open(self.file_path, "rb") as f:
            return f.read()


class BytesAudioSource(AudioSource):
    """
    Audio source reading directly from an in-memory byte buffer.
    """

    def __init__(self, raw_bytes: Union[bytes, bytearray, io.BytesIO]):
        if isinstance(raw_bytes, io.BytesIO):
            self._raw_bytes = raw_bytes.getvalue()
        elif isinstance(raw_bytes, (bytes, bytearray)):
            self._raw_bytes = bytes(raw_bytes)
        else:
            raise TypeError(f"Formato inválido de buffer de bytes: {type(raw_bytes)}")

    def read_bytes(self) -> bytes:
        return self._raw_bytes


class MicrophoneAudioSource(AudioSource):
    """
    Audio source capturing live audio from a local microphone using sounddevice.
    Includes robust fallback for headless/CI environments where microphone
    hardware or sounddevice is not available.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration_seconds: float = 4.0,
        device: Optional[Union[int, str]] = None,
        fallback_bytes: Optional[bytes] = None,
    ):
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.device = device
        self.fallback_bytes = fallback_bytes

    def read_bytes(self) -> bytes:
        """
        Records audio from the local microphone.

        If sounddevice is unavailable or fails (e.g. in headless CI),
        returns fallback_bytes if provided, otherwise raises RuntimeError.
        """
        try:
            import sounddevice as sd
            import numpy as np

            num_samples = int(self.duration_seconds * self.sample_rate)
            recording = sd.rec(
                num_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                device=self.device,
            )
            sd.wait()

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(recording.tobytes())
            return buf.getvalue()
        except Exception as exc:
            if self.fallback_bytes is not None:
                return self.fallback_bytes
            raise RuntimeError(
                f"Microfone local indisponível no ambiente atual ({exc}). "
                "Para testes ou ambientes CI/headless, utilize FileAudioSource ou BytesAudioSource."
            ) from exc

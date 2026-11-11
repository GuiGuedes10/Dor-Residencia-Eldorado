"""
Pure Python Audio Validation Engine for REIA.

Evaluates audio constraints: windowed RMS energy (50ms frames), silence ratio,
active voice duration, energy-quantile SNR estimation, and duration bounds.
Fully compliant with Python 3.10+ without the removed `audioop` module.
"""

from dataclasses import dataclass, field
from enum import Enum
import io
import math
import struct
from typing import Optional, Any
import wave

from reia.core.config import AudioConfig, config as default_app_config
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


class AudioStatus(str, Enum):
    """Enumeration of possible validation states for audio input."""
    VALID = "VALID"
    EMPTY_INPUT = "EMPTY_INPUT"
    SILENCE_DETECTED = "SILENCE_DETECTED"
    EXCESSIVE_NOISE = "EXCESSIVE_NOISE"
    TOO_SHORT = "TOO_SHORT"
    TOO_LONG = "TOO_LONG"
    CORRUPTED_FORMAT = "CORRUPTED_FORMAT"
    UNSUPPORTED_SAMPLE_RATE = "UNSUPPORTED_SAMPLE_RATE"


@dataclass(frozen=True)
class AudioMetrics:
    """Detailed acoustic metrics calculated from audio signal."""
    duration_sec: float
    sample_rate: int
    channels: int
    rms_dbfs: float
    peak_amplitude: float
    silence_ratio: float
    active_duration_sec: float
    snr_db: float
    is_clipped: bool


@dataclass
class AudioValidationResult:
    """
    Standard result returned by audio validation.
    Conforms to REIA interface contract between reia.audio and reia.neural.
    """
    is_valid: bool
    status: AudioStatus
    message: str
    duration_seconds: float
    rms_dbfs: float
    snr_db: float
    silence_ratio: float
    transcription: Optional[str] = None
    sanitized_text: Optional[str] = None
    error_code: Optional[str] = None
    metrics: Optional[AudioMetrics] = None
    processed_wav_bytes: Optional[bytes] = None


def validate_and_process_audio(
    audio_bytes: bytes,
    cfg: Optional[AudioConfig] = None,
) -> AudioValidationResult:
    """
    Validates audio byte stream against acoustic constraints.

    Evaluates:
    - Byte integrity and WAV 16-bit PCM header format.
    - Duration bounds (min_duration <= duration <= max_duration).
    - Windowed RMS energy across 50ms frames and silence ratio.
    - Energy-quantile based Signal-to-Noise Ratio (SNR).

    Args:
        audio_bytes: Raw binary audio bytes (expected WAV format).
        cfg: Optional AudioConfig with custom thresholds.

    Returns:
        AudioValidationResult with validation flag, status, and acoustic metrics.
    """
    if cfg is None:
        cfg = default_app_config.audio

    # 1. Empty buffer check
    if not audio_bytes or len(audio_bytes) == 0:
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.EMPTY_INPUT,
            message="Nenhum dado de áudio foi recebido. Verifique o microfone ou o arquivo enviado.",
            duration_seconds=0.0,
            rms_dbfs=-100.0,
            snr_db=0.0,
            silence_ratio=1.0,
            error_code="ERR_AUDIO_EMPTY",
        )

    # 2. WAV structure validation
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()

            if sampwidth != 2:
                return AudioValidationResult(
                    is_valid=False,
                    status=AudioStatus.CORRUPTED_FORMAT,
                    message="Formato de áudio não suportado. Requer PCM de 16 bits.",
                    duration_seconds=0.0,
                    rms_dbfs=-100.0,
                    snr_db=0.0,
                    silence_ratio=1.0,
                    error_code="ERR_UNSUPPORTED_SAMPLE_WIDTH",
                )

            duration_seconds = n_frames / sample_rate if sample_rate > 0 else 0.0
            raw_frames = wf.readframes(n_frames)
    except Exception as exc:
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.CORRUPTED_FORMAT,
            message="O formato de áudio é inválido ou o arquivo está corrompido.",
            duration_seconds=0.0,
            rms_dbfs=-100.0,
            snr_db=0.0,
            silence_ratio=1.0,
            error_code="ERR_AUDIO_CORRUPTED",
        )

    # 3. Duration Limits Check
    if duration_seconds < cfg.min_duration:
        metrics = AudioMetrics(
            duration_sec=duration_seconds,
            sample_rate=sample_rate,
            channels=n_channels,
            rms_dbfs=-100.0,
            peak_amplitude=0.0,
            silence_ratio=1.0,
            active_duration_sec=duration_seconds,
            snr_db=0.0,
            is_clipped=False,
        )
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.TOO_SHORT,
            message=(
                f"Comando de voz muito curto ({duration_seconds:.2f}s). "
                f"Por favor, fale a frase completa (mínimo de {cfg.min_duration}s)."
            ),
            duration_seconds=duration_seconds,
            rms_dbfs=-100.0,
            snr_db=0.0,
            silence_ratio=1.0,
            error_code="ERR_AUDIO_TOO_SHORT",
            metrics=metrics,
        )

    if duration_seconds > cfg.max_duration:
        metrics = AudioMetrics(
            duration_sec=duration_seconds,
            sample_rate=sample_rate,
            channels=n_channels,
            rms_dbfs=-100.0,
            peak_amplitude=0.0,
            silence_ratio=1.0,
            active_duration_sec=duration_seconds,
            snr_db=0.0,
            is_clipped=False,
        )
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.TOO_LONG,
            message=(
                f"Comando de voz muito longo ({duration_seconds:.2f}s). "
                f"Por favor, seja mais sucinto (máximo de {cfg.max_duration}s)."
            ),
            duration_seconds=duration_seconds,
            rms_dbfs=-100.0,
            snr_db=0.0,
            silence_ratio=1.0,
            error_code="ERR_AUDIO_TOO_LONG",
            metrics=metrics,
        )

    # 4. Unpack 16-bit PCM and convert to mono float in [-1.0, 1.0]
    expected_samples = n_frames * n_channels
    try:
        unpacked = struct.unpack(f"<{expected_samples}h", raw_frames)
    except struct.error:
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.CORRUPTED_FORMAT,
            message="O arquivo de áudio possui dados incompletos ou corrompidos.",
            duration_seconds=duration_seconds,
            rms_dbfs=-100.0,
            snr_db=0.0,
            silence_ratio=1.0,
            error_code="ERR_AUDIO_CORRUPTED",
        )

    if n_channels == 1:
        samples = [s / 32768.0 for s in unpacked]
    else:
        # Downmix multi-channel to mono
        samples = [
            (sum(unpacked[i * n_channels + c] for c in range(n_channels)) / n_channels) / 32768.0
            for i in range(n_frames)
        ]

    # 5. Windowed RMS analysis (default 50ms frames)
    samples_per_frame = max(1, int(sample_rate * (cfg.frame_ms / 1000.0)))
    n_chunks = len(samples) // samples_per_frame
    if n_chunks == 0:
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.TOO_SHORT,
            message="Áudio com quadros insuficientes para análise acústica.",
            duration_seconds=duration_seconds,
            rms_dbfs=-100.0,
            snr_db=0.0,
            silence_ratio=1.0,
            error_code="ERR_AUDIO_TOO_SHORT",
        )

    frame_rms_list = []
    silent_frames = 0
    peak_val = 0.0

    for c in range(n_chunks):
        chunk = samples[c * samples_per_frame : (c + 1) * samples_per_frame]
        chunk_sq_sum = sum(s * s for s in chunk)
        c_rms = math.sqrt(chunk_sq_sum / len(chunk))
        c_db = 20.0 * math.log10(c_rms + 1e-9)
        frame_rms_list.append(c_rms)
        if c_db < cfg.silence_threshold_dbfs:
            silent_frames += 1
        chunk_peak = max(abs(s) for s in chunk)
        if chunk_peak > peak_val:
            peak_val = chunk_peak

    silence_ratio = silent_frames / n_chunks
    active_sec = (n_chunks - silent_frames) * (cfg.frame_ms / 1000.0)

    # Overall RMS energy across all samples
    overall_sq_sum = sum(s * s for s in samples)
    overall_rms = math.sqrt(overall_sq_sum / len(samples))
    overall_dbfs = 20.0 * math.log10(overall_rms + 1e-9)

    # 6. Energy-Quantile SNR Estimation
    sorted_rms = sorted(frame_rms_list)
    p20_idx = max(1, int(0.20 * len(sorted_rms)))
    p80_idx = max(1, int(0.80 * len(sorted_rms)))
    noise_floor_rms = math.sqrt(sum(r * r for r in sorted_rms[:p20_idx]) / p20_idx)
    speech_frames = sorted_rms[p80_idx:]
    speech_rms = (
        math.sqrt(sum(r * r for r in speech_frames) / len(speech_frames))
        if speech_frames
        else overall_rms
    )
    snr_db = 20.0 * math.log10((speech_rms + 1e-9) / (noise_floor_rms + 1e-9))

    metrics = AudioMetrics(
        duration_sec=duration_seconds,
        sample_rate=sample_rate,
        channels=1,
        rms_dbfs=overall_dbfs,
        peak_amplitude=peak_val,
        silence_ratio=silence_ratio,
        active_duration_sec=active_sec,
        snr_db=snr_db,
        is_clipped=(peak_val >= 0.99),
    )

    # 7. Silence & inaudible speech checks
    if (
        overall_dbfs < cfg.overall_min_rms_dbfs
        or silence_ratio > cfg.max_silence_ratio
        or active_sec < cfg.min_active_duration
    ):
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.SILENCE_DETECTED,
            message="Não detectamos sua voz. O áudio está muito baixo ou em silêncio. Tente novamente.",
            duration_seconds=duration_seconds,
            rms_dbfs=overall_dbfs,
            snr_db=snr_db,
            silence_ratio=silence_ratio,
            error_code="ERR_AUDIO_SILENCE",
            metrics=metrics,
        )

    # 8. Noise floor check
    if snr_db < cfg.min_snr_db:
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.EXCESSIVE_NOISE,
            message="Ruído excessivo detectado no ambiente. Por favor, fale em um local mais silencioso.",
            duration_seconds=duration_seconds,
            rms_dbfs=overall_dbfs,
            snr_db=snr_db,
            silence_ratio=silence_ratio,
            error_code="ERR_AUDIO_NOISE",
            metrics=metrics,
        )

    # 9. Clean audio processing (DC offset removal + peak normalization to -1.5 dBFS)
    dc_offset = sum(samples) / len(samples)
    samples_clean = [s - dc_offset for s in samples]
    norm_factor = 0.85 / max(peak_val, 1e-4) if peak_val < 0.60 else 1.0
    normalized_samples = [max(min(s * norm_factor, 1.0), -1.0) for s in samples_clean]

    processed_buf = io.BytesIO()
    with wave.open(processed_buf, "wb") as pwf:
        pwf.setnchannels(1)
        pwf.setsampwidth(2)
        pwf.setframerate(sample_rate)
        packed = bytearray()
        for s in normalized_samples:
            packed.extend(struct.pack("<h", int(s * 32767)))
        pwf.writeframes(packed)

    return AudioValidationResult(
        is_valid=True,
        status=AudioStatus.VALID,
        message="Áudio válido e pronto para processamento.",
        duration_seconds=duration_seconds,
        rms_dbfs=overall_dbfs,
        snr_db=snr_db,
        silence_ratio=silence_ratio,
        error_code=None,
        metrics=metrics,
        processed_wav_bytes=processed_buf.getvalue(),
    )


def validate_audio(
    audio_bytes: bytes,
    cfg: Optional[AudioConfig] = None,
) -> AudioValidationResult:
    """Convenience alias for validate_and_process_audio."""
    return validate_and_process_audio(audio_bytes, cfg)


def validate_or_raise(
    audio_bytes: bytes,
    cfg: Optional[AudioConfig] = None,
) -> AudioValidationResult:
    """
    Validates audio and raises explicit AudioValidationError subclass if invalid.

    Raises:
        EmptyAudioError: If audio buffer is empty.
        CorruptedAudioError: If audio header/stream is corrupted.
        AudioTooShortError: If duration is below threshold.
        AudioTooLongError: If duration is above threshold.
        SilenceDetectedError: If audio contains silence or is inaudible.
        ExcessiveNoiseError: If audio has low SNR.
    """
    res = validate_and_process_audio(audio_bytes, cfg)
    if res.is_valid:
        return res

    if res.error_code == "ERR_AUDIO_EMPTY":
        raise EmptyAudioError(res.message, result=res)
    if res.status == AudioStatus.CORRUPTED_FORMAT:
        raise CorruptedAudioError(res.message, result=res)
    if res.status == AudioStatus.TOO_SHORT:
        raise AudioTooShortError(res.message, result=res)
    if res.status == AudioStatus.TOO_LONG:
        raise AudioTooLongError(res.message, result=res)
    if res.status == AudioStatus.SILENCE_DETECTED:
        raise SilenceDetectedError(res.message, result=res)
    if res.status == AudioStatus.EXCESSIVE_NOISE:
        raise ExcessiveNoiseError(res.message, result=res)

    raise AudioValidationError(res.message, result=res)

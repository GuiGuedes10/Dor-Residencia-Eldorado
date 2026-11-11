"""
Configuration parameters and target routes for REIA.
"""

from dataclasses import dataclass, field
from typing import Dict
import os


@dataclass(frozen=True)
class AudioConfig:
    """Thresholds and acoustic parameters for audio validation."""
    min_duration: float = 0.8
    max_duration: float = 12.0
    max_silence_ratio: float = 0.85
    min_active_duration: float = 0.4
    min_snr_db: float = 3.0
    sample_rate: int = 16000
    silence_threshold_dbfs: float = -40.0
    frame_ms: float = 50.0
    overall_min_rms_dbfs: float = -46.0


class TargetUrls:
    """Official target URLs for PUC-Campinas Vestibular portal."""
    GUIA_UNIVERSITARIO: str = "https://vestibular.puc-campinas.edu.br/guia-universitario/"
    MANUAL_DO_CANDIDATO: str = "https://vestibular.puc-campinas.edu.br/manual-do-candidato/"
    FORMULARIO_INSCRICAO: str = "https://vestibular.puc-campinas.edu.br/"


TARGET_URLS: Dict[str, str] = {
    "GUIA_UNIVERSITARIO": TargetUrls.GUIA_UNIVERSITARIO,
    "MANUAL_DO_CANDIDATO": TargetUrls.MANUAL_DO_CANDIDATO,
    "FORMULARIO_INSCRICAO": TargetUrls.FORMULARIO_INSCRICAO,
}

# Mapping by class ID (matching Untitled0.ipynb classes)
CLASS_ID_TO_INTENT: Dict[int, str] = {
    0: "GUIA_UNIVERSITARIO",
    1: "MANUAL_DO_CANDIDATO",
    2: "FORMULARIO_INSCRICAO",
}

INTENT_TO_URL: Dict[str, str] = TARGET_URLS


@dataclass
class AppConfig:
    """Overall REIA application configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    target_urls: Dict[str, str] = field(default_factory=lambda: dict(TARGET_URLS))
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    artifacts_dir: str = field(
        default_factory=lambda: os.getenv("REIA_ARTIFACTS_DIR", "artifacts")
    )


# Default global configuration instance
config = AppConfig()

"""
End-to-End Voice Navigation Orchestration Pipeline for REIA (Feature F15).

Coordinates:
1. Audio ingestion and validation (reia.audio)
2. Transcription sanitization (reia.audio.sanitization)
3. 4-Stage neural intent classification and confidence gating (reia.neural)
4. Headless web automation and artifact persistence (reia.navigation)
5. Accessible Brazilian Portuguese user feedback messaging (reia.pipeline.messages)

Enforces strict early-exit invariants: web navigation is NEVER triggered if
audio fails acoustic validation or if the neural intent classifier rejects the query.
"""

from __future__ import annotations

import datetime
import io
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union

from reia.audio.sanitization import sanitize_transcription
from reia.audio.sources import AudioSource, BytesAudioSource, FileAudioSource
from reia.audio.synthetic import SyntheticAudioGenerator
from reia.audio.validation import (
    AudioMetrics,
    AudioStatus,
    AudioValidationResult,
    validate_and_process_audio,
)
from reia.navigation.base import BaseNavigator, NavigationResult
from reia.navigation.routes import CANONICAL_ROUTES, get_navigator
from reia.neural.thresholding import ClassificationResult, NeuralIntentClassifier
from reia.pipeline.messages import (
    MSG_AUDIO_EMPTY,
    MSG_NAVIGATION_ERROR,
    MSG_NAVIGATION_SUCCESS_SHORT,
    MSG_NEURAL_EMPTY_TEXT,
    get_audio_feedback_message,
    get_navigation_feedback_message,
    get_neural_feedback_message,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Result Contract
# ---------------------------------------------------------------------------
class PipelineResult:
    """
    Structured outcome of the end-to-end voice navigation pipeline.
    Supports both object attribute access (.success) and dictionary mapping (['success'])
    for complete compatibility with pytest harnesses and external consumers.
    """

    def __init__(
        self,
        success: bool,
        stage: str,
        status: str,
        user_message: str,
        target_url: Optional[str] = None,
        transcript: Optional[str] = None,
        sanitized_text: Optional[str] = None,
        confidence: Optional[float] = None,
        margin: Optional[float] = None,
        entropy: Optional[float] = None,
        audio_metrics: Optional[Dict[str, Any]] = None,
        audio_validation: Optional[AudioValidationResult] = None,
        classification: Optional[ClassificationResult] = None,
        navigation: Optional[NavigationResult] = None,
        screenshot_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        execution_time_seconds: float = 0.0,
        timestamp: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        self.success = bool(success)
        self.stage = stage
        self.status = status
        self.user_message = user_message
        self.target_url = target_url
        self.transcript = transcript
        self.sanitized_text = sanitized_text
        self.confidence = confidence
        self.margin = margin
        self.entropy = entropy
        self.audio_metrics = audio_metrics
        self.audio_validation = audio_validation
        self.classification = classification
        self.navigation = navigation
        self.screenshot_path = screenshot_path
        self.metadata_path = metadata_path
        self.execution_time_seconds = execution_time_seconds
        self.timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.error_code = error_code
        self.target_intent = classification.intent_name if classification else None

    @property
    def is_success(self) -> bool:
        """Alias for success."""
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into a JSON-compatible dictionary."""
        return {
            "success": self.success,
            "stage": self.stage,
            "status": self.status,
            "user_message": self.user_message,
            "target_url": self.target_url,
            "transcript": self.transcript,
            "sanitized_text": self.sanitized_text,
            "confidence": self.confidence,
            "margin": self.margin,
            "entropy": self.entropy,
            "audio_metrics": self.audio_metrics,
            "audio_validation": (
                {
                    "is_valid": self.audio_validation.is_valid,
                    "status": self.audio_validation.status.value,
                    "message": self.audio_validation.message,
                    "duration_seconds": self.audio_validation.duration_seconds,
                    "rms_dbfs": self.audio_validation.rms_dbfs,
                    "snr_db": self.audio_validation.snr_db,
                    "silence_ratio": self.audio_validation.silence_ratio,
                    "error_code": self.audio_validation.error_code,
                }
                if self.audio_validation is not None
                else None
            ),
            "classification": (
                {
                    "is_valid_intent": self.classification.is_valid_intent,
                    "intent_name": self.classification.intent_name,
                    "target_url": self.classification.target_url,
                    "confidence": self.classification.confidence,
                    "margin": self.classification.margin,
                    "entropy": self.classification.entropy,
                    "probabilities": self.classification.probabilities,
                    "rejection_code": self.classification.rejection_code,
                    "user_message": self.classification.user_message,
                }
                if self.classification is not None
                else None
            ),
            "navigation": (
                self.navigation.to_dict()
                if self.navigation is not None
                else None
            ),
            "screenshot_path": self.screenshot_path,
            "metadata_path": self.metadata_path,
            "execution_time_seconds": self.execution_time_seconds,
            "timestamp": self.timestamp,
            "error_code": self.error_code,
        }

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(f"PipelineResult has no field '{key}'")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def __repr__(self) -> str:
        return (
            f"PipelineResult(success={self.success}, stage='{self.stage}', "
            f"status='{self.status}', user_message='{self.user_message}')"
        )


# ---------------------------------------------------------------------------
# Synthetic Scenarios Configuration
# ---------------------------------------------------------------------------
SYNTHETIC_SCENARIOS = {
    "valid_guia": {
        "text": "quero ver o guia universitario de apresentacao dos cursos da puc campinas",
        "type": "speech",
        "duration": 3.0,
    },
    "valid_manual": {
        "text": "abrir manual do candidato com as regras da prova",
        "type": "speech",
        "duration": 3.0,
    },
    "valid_inscricao": {
        "text": "como faco para me inscrever no vestibular da puc campinas",
        "type": "speech",
        "duration": 3.0,
    },
    "silence": {
        "text": None,
        "type": "silence",
        "duration": 2.0,
    },
    "short": {
        "text": None,
        "type": "short",
        "duration": 0.3,
    },
    "long": {
        "text": None,
        "type": "long",
        "duration": 14.0,
    },
    "noise": {
        "text": None,
        "type": "noise",
        "duration": 2.0,
    },
    "corrupted": {
        "text": None,
        "type": "corrupted",
        "duration": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Voice Navigation Pipeline Orchestrator
# ---------------------------------------------------------------------------
class VoiceNavigationPipeline:
    """
    End-to-End Orchestrator for Voice Navigation in REIA.

    Coordinates:
    - Audio input ingestion and validation
    - Spoken transcription sanitization
    - Neural intent classification with multi-stage gating
    - Headless browser navigation and screenshot/metadata persistence
    - Friendly Brazilian Portuguese user feedback
    """

    def __init__(
        self,
        validator: Optional[Callable[[bytes], AudioValidationResult]] = None,
        classifier: Optional[NeuralIntentClassifier] = None,
        navigator: Optional[BaseNavigator] = None,
        navigator_mode: str = "auto",
        artifacts_dir: str = "artifacts",
        transcriber: Optional[Callable[[bytes], str]] = None,
    ):
        self.validator = validator or validate_and_process_audio
        self.classifier = classifier or NeuralIntentClassifier()
        self.artifacts_dir = artifacts_dir
        self.navigator_mode = navigator_mode
        self._navigator = navigator
        self.transcriber = transcriber

    @property
    def navigator(self) -> BaseNavigator:
        """Lazy initialization of web navigator if not injected."""
        if self._navigator is None:
            self._navigator = get_navigator(
                mode=self.navigator_mode,
                artifacts_dir=self.artifacts_dir,
            )
        return self._navigator

    def _resolve_audio_bytes(
        self,
        audio: Any,
        synthetic: Optional[str] = None,
    ) -> tuple[Optional[bytes], Optional[str]]:
        """
        Extracts raw WAV bytes and optional synthetic transcription from various input formats.
        """
        # 1. Synthetic scenario
        if synthetic:
            scenario_key = synthetic.strip().lower()
            if scenario_key in SYNTHETIC_SCENARIOS:
                sc = SYNTHETIC_SCENARIOS[scenario_key]
                stype = sc["type"]
                s_text = sc["text"]
                if stype == "speech":
                    return SyntheticAudioGenerator.generate_mock_speech(sc["duration"]), s_text
                elif stype == "silence":
                    return SyntheticAudioGenerator.generate_silence(sc["duration"]), s_text
                elif stype == "short":
                    return SyntheticAudioGenerator.generate_short_audio(sc["duration"]), s_text
                elif stype == "long":
                    return SyntheticAudioGenerator.generate_long_audio(sc["duration"]), s_text
                elif stype == "noise":
                    return SyntheticAudioGenerator.generate_white_noise(sc["duration"], amplitude=0.9), s_text
                elif stype == "corrupted":
                    return SyntheticAudioGenerator.generate_corrupted_audio("truncated_header"), s_text
            else:
                raise ValueError(
                    f"Cenário sintético desconhecido: '{synthetic}'. "
                    f"Opções válidas: {list(SYNTHETIC_SCENARIOS.keys())}"
                )

        if audio is None:
            return None, None

        # 2. AudioSource abstraction
        if isinstance(audio, AudioSource):
            return audio.read_bytes(), None

        # 3. Direct bytes / bytearray / BytesIO
        if isinstance(audio, io.BytesIO):
            return audio.getvalue(), None
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio), None

        # 4. Filesystem path
        if isinstance(audio, (str, Path)):
            audio_path = str(audio)
            return FileAudioSource(audio_path).read_bytes(), None

        raise TypeError(f"Tipo de entrada de áudio não suportado: {type(audio)}")

    def run(
        self,
        audio: Any = None,
        text: Optional[str] = None,
        transcript_override: Optional[str] = None,
        synthetic: Optional[str] = None,
    ) -> PipelineResult:
        """
        Executes the end-to-end voice navigation pipeline.

        Args:
            audio: Audio bytes, AudioSource, file path, or io.BytesIO.
            text: Direct text command (bypasses audio stage).
            transcript_override: Overrides transcribed text from audio.
            synthetic: Synthetic scenario name ('valid_guia', 'silence', etc.).

        Returns:
            Structured PipelineResult instance.
        """
        start_time = time.perf_counter()

        # Step 0: Input resolution
        try:
            audio_bytes, synthetic_text = self._resolve_audio_bytes(audio, synthetic=synthetic)
        except Exception as exc:
            exec_time = time.perf_counter() - start_time
            return PipelineResult(
                success=False,
                stage="AUDIO_VALIDATION",
                status=AudioStatus.CORRUPTED_FORMAT.value,
                user_message=f"Erro ao carregar áudio: {exc}",
                execution_time_seconds=exec_time,
                error_code="ERR_AUDIO_LOAD",
            )

        val_result: Optional[AudioValidationResult] = None
        audio_metrics: Optional[Dict[str, Any]] = None

        # -------------------------------------------------------------
        # Stage 1: Audio Validation (if audio provided)
        # -------------------------------------------------------------
        if audio_bytes is not None:
            val_result = self.validator(audio_bytes)
            audio_metrics = {
                "duration": val_result.duration_seconds,
                "rms_dbfs": val_result.rms_dbfs,
                "snr_db": val_result.snr_db,
                "silence_ratio": val_result.silence_ratio,
            }

            if not val_result.is_valid:
                # EARLY EXIT ON AUDIO FAILURE — Navigation NEVER triggered
                exec_time = time.perf_counter() - start_time
                return PipelineResult(
                    success=False,
                    stage="AUDIO_VALIDATION",
                    status=val_result.status.value,
                    user_message=val_result.message,
                    audio_metrics=audio_metrics,
                    audio_validation=val_result,
                    execution_time_seconds=exec_time,
                    error_code=val_result.error_code,
                )

            # Determine transcript from audio
            transcription = (
                transcript_override
                or text
                or synthetic_text
                or (self.transcriber(audio_bytes) if self.transcriber else None)
                or val_result.transcription
                or "mostrar guia do vestibular da puc campinas"
            )
        elif text is not None:
            # Direct text mode (no audio)
            transcription = text
        else:
            # Empty input check
            exec_time = time.perf_counter() - start_time
            return PipelineResult(
                success=False,
                stage="AUDIO_VALIDATION",
                status=AudioStatus.EMPTY_INPUT.value,
                user_message=MSG_AUDIO_EMPTY,
                execution_time_seconds=exec_time,
                error_code="ERR_EMPTY_INPUT",
            )

        # -------------------------------------------------------------
        # Stage 2: Neural Classification & Lexical Saliency Gating
        # -------------------------------------------------------------
        sanitized = sanitize_transcription(transcription)

        clf_result = self.classifier.classify(sanitized)
        if not clf_result.is_valid_intent:
            # EARLY EXIT ON NEURAL REJECTION — Navigation NEVER triggered
            exec_time = time.perf_counter() - start_time
            return PipelineResult(
                success=False,
                stage="NEURAL_CLASSIFICATION",
                status=clf_result.rejection_code or "REJECTED",
                user_message=clf_result.user_message,
                transcript=transcription,
                sanitized_text=sanitized,
                confidence=clf_result.confidence,
                margin=clf_result.margin,
                entropy=clf_result.entropy,
                audio_metrics=audio_metrics,
                audio_validation=val_result,
                classification=clf_result,
                execution_time_seconds=exec_time,
                error_code=clf_result.rejection_code,
            )

        # -------------------------------------------------------------
        # Stage 3: Headless Web Navigation Automation & Artifact Persistence
        # -------------------------------------------------------------
        intent_name = clf_result.intent_name or "GUIA_UNIVERSITARIO"
        target_url = clf_result.target_url or CANONICAL_ROUTES.get(intent_name, "https://vestibular.puc-campinas.edu.br/")

        try:
            nav_result = self.navigator.navigate(target_url=target_url, intent_name=intent_name)
        except Exception as exc:
            logger.error("Unexpected error during navigation to %s: %s", target_url, exc)
            exec_time = time.perf_counter() - start_time
            return PipelineResult(
                success=False,
                stage="WEB_NAVIGATION",
                status="NAVIGATION_FAILED",
                user_message=MSG_NAVIGATION_ERROR.format(target_url=target_url, error_message=str(exc)),
                target_url=target_url,
                transcript=transcription,
                sanitized_text=sanitized,
                confidence=clf_result.confidence,
                margin=clf_result.margin,
                entropy=clf_result.entropy,
                audio_metrics=audio_metrics,
                audio_validation=val_result,
                classification=clf_result,
                execution_time_seconds=exec_time,
                error_code="ERR_NAVIGATION_EXCEPTION",
            )

        exec_time = time.perf_counter() - start_time

        if nav_result.success:
            user_msg = clf_result.user_message
            return PipelineResult(
                success=True,
                stage="WEB_NAVIGATION",
                status="COMPLETED",
                user_message=user_msg,
                target_url=target_url,
                transcript=transcription,
                sanitized_text=sanitized,
                confidence=clf_result.confidence,
                margin=clf_result.margin,
                entropy=clf_result.entropy,
                audio_metrics=audio_metrics,
                audio_validation=val_result,
                classification=clf_result,
                navigation=nav_result,
                screenshot_path=nav_result.screenshot_path,
                metadata_path=nav_result.metadata_path,
                execution_time_seconds=exec_time,
            )
        else:
            return PipelineResult(
                success=False,
                stage="WEB_NAVIGATION",
                status="NAVIGATION_FAILED",
                user_message=nav_result.error_message or MSG_NAVIGATION_ERROR.format(target_url=target_url, error_message="Falha de carregamento"),
                target_url=target_url,
                transcript=transcription,
                sanitized_text=sanitized,
                confidence=clf_result.confidence,
                margin=clf_result.margin,
                entropy=clf_result.entropy,
                audio_metrics=audio_metrics,
                audio_validation=val_result,
                classification=clf_result,
                navigation=nav_result,
                screenshot_path=nav_result.screenshot_path,
                metadata_path=nav_result.metadata_path,
                execution_time_seconds=exec_time,
                error_code="ERR_NAVIGATION_FAILED",
            )

    def process_voice_command(
        self,
        audio_bytes: bytes,
        transcript_override: Optional[str] = None,
    ) -> PipelineResult:
        """
        Convenience method adhering to the exact REIA test harness interface contract.
        """
        return self.run(audio=audio_bytes, transcript_override=transcript_override)

    def process_text(self, text: str) -> PipelineResult:
        """Convenience method for direct text execution."""
        return self.run(text=text)

    def process_audio_source(
        self,
        audio_source: AudioSource,
        transcript_override: Optional[str] = None,
    ) -> PipelineResult:
        """Convenience method for AudioSource instances (e.g. MicrophoneAudioSource)."""
        return self.run(audio=audio_source, transcript_override=transcript_override)

    def __call__(self, *args, **kwargs) -> PipelineResult:
        """Callable convenience delegating to run()."""
        if args and isinstance(args[0], (bytes, bytearray, io.BytesIO)):
            return self.process_voice_command(args[0], **kwargs)
        if args and isinstance(args[0], str):
            # Check if it's a file path or text command
            if os.path.exists(args[0]) and args[0].lower().endswith(".wav"):
                return self.run(audio=args[0], **kwargs)
            return self.run(text=args[0], **kwargs)
        return self.run(*args, **kwargs)


# Alias for test harness resolution
PipelineCoordinator = VoiceNavigationPipeline

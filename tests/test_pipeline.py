"""
Comprehensive Test Suite for REIA Voice Navigation Pipeline and CLI Subsystem (Milestone 4).

Covers:
- End-to-end voice navigation flows for all 3 canonical destinations:
  - Guia Universitário
  - Manual do Candidato
  - Formulário de Inscrição
- Invariant enforcement: early exit on invalid audio and neural intent gating rejections
  (web navigation is NEVER triggered on invalid inputs)
- Structured PipelineResult contract and dual object/dictionary access patterns
- Audio constraint rejection handling (silence, excessive noise, too short, too long, corrupted)
- Neural intent gating rejection handling (out-of-domain, ambiguous, low confidence)
- Headless web navigation error handling and custom artifact persistence
- CLI application execution via subprocess (--audio, --text, --synthetic, --navigator, --json, --quiet, --verbose)
- Brazilian Portuguese accessible user feedback messaging across all stages
"""

from __future__ import annotations

import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reia.audio.sources import BytesAudioSource, FileAudioSource
from reia.audio.synthetic import SyntheticAudioGenerator
from reia.audio.validation import AudioStatus, AudioValidationResult
from reia.navigation.base import BaseNavigator, NavigationResult
from reia.navigation.mock_driver import MockNavigator
from reia.navigation.routes import CANONICAL_ROUTES
from reia.neural.thresholding import ClassificationResult, NeuralIntentClassifier
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
    MSG_NAVIGATION_ERROR,
    MSG_NAVIGATION_SUCCESS_SHORT,
    MSG_NEURAL_AMBIGUOUS_INTENT,
    MSG_NEURAL_LOW_CONFIDENCE,
    MSG_NEURAL_OOD,
    get_audio_feedback_message,
    get_navigation_feedback_message,
    get_neural_feedback_message,
)


# ===========================================================================
# 1. Successful End-to-End Navigation Flows
# ===========================================================================
class TestPipelineSuccessfulFlows:
    """Verifies complete, genuine end-to-end flows producing real visual and metadata artifacts."""

    def test_guia_universitario_audio_flow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts_dir = str(Path(tmp_dir) / "artifacts")
            nav = MockNavigator(artifacts_dir=artifacts_dir)
            pipeline = VoiceNavigationPipeline(navigator=nav, artifacts_dir=artifacts_dir)

            audio_bytes = SyntheticAudioGenerator.generate_mock_speech(duration_sec=3.0)
            transcript = "quero ver o guia universitario de apresentacao dos cursos da puc campinas"

            result = pipeline.run(audio=audio_bytes, transcript_override=transcript)

            assert result.success is True
            assert result.stage == "WEB_NAVIGATION"
            assert result.status == "COMPLETED"
            assert result.target_url == CANONICAL_ROUTES["GUIA_UNIVERSITARIO"]
            assert result.classification is not None
            assert result.classification.intent_name == "GUIA_UNIVERSITARIO"
            assert result.classification.confidence >= 0.65
            assert result.navigation is not None
            assert result.navigation.success is True
            assert "Guia" in result.navigation.page_title
            assert "navegando" in result.user_message.lower()

            # Verify physical artifacts
            assert result.screenshot_path is not None
            assert os.path.isfile(result.screenshot_path)
            png_bytes = Path(result.screenshot_path).read_bytes()
            assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

            assert result.metadata_path is not None
            assert os.path.isfile(result.metadata_path)
            meta = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
            assert meta["intent_name"] == "GUIA_UNIVERSITARIO"
            assert len(meta["headings"]) > 0

    def test_manual_do_candidato_audio_flow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts_dir = str(Path(tmp_dir) / "artifacts")
            nav = MockNavigator(artifacts_dir=artifacts_dir)
            pipeline = VoiceNavigationPipeline(navigator=nav, artifacts_dir=artifacts_dir)

            audio_bytes = SyntheticAudioGenerator.generate_mock_speech(duration_sec=3.0)
            transcript = "abrir manual do candidato com as regras da prova e edital da puc"

            result = pipeline.run(audio=audio_bytes, transcript_override=transcript)

            assert result.success is True
            assert result.stage == "WEB_NAVIGATION"
            assert result.status == "COMPLETED"
            assert result.target_url == CANONICAL_ROUTES["MANUAL_DO_CANDIDATO"]
            assert result.classification.intent_name == "MANUAL_DO_CANDIDATO"
            assert result.navigation.success is True
            assert os.path.isfile(result.screenshot_path)
            assert os.path.isfile(result.metadata_path)

    def test_formulario_inscricao_audio_flow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts_dir = str(Path(tmp_dir) / "artifacts")
            nav = MockNavigator(artifacts_dir=artifacts_dir)
            pipeline = VoiceNavigationPipeline(navigator=nav, artifacts_dir=artifacts_dir)

            audio_bytes = SyntheticAudioGenerator.generate_mock_speech(duration_sec=3.0)
            transcript = "como faco para me inscrever no vestibular da puc campinas"

            result = pipeline.run(audio=audio_bytes, transcript_override=transcript)

            assert result.success is True
            assert result.stage == "WEB_NAVIGATION"
            assert result.status == "COMPLETED"
            assert result.target_url == CANONICAL_ROUTES["FORMULARIO_INSCRICAO"]
            assert result.classification.intent_name == "FORMULARIO_INSCRICAO"
            assert result.navigation.success is True
            assert os.path.isfile(result.screenshot_path)
            assert os.path.isfile(result.metadata_path)

    def test_direct_text_input_all_destinations(self):
        """Verifies text queries without audio bypass the audio validation stage cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts_dir = str(Path(tmp_dir) / "artifacts")
            nav = MockNavigator(artifacts_dir=artifacts_dir)
            pipeline = VoiceNavigationPipeline(navigator=nav, artifacts_dir=artifacts_dir)

            cases = [
                ("quero ver o guia de cursos da puc", "GUIA_UNIVERSITARIO"),
                ("abrir edital e manual de regras do vestibular", "MANUAL_DO_CANDIDATO"),
                ("fazer cadastro e inscricao no vestibular agora", "FORMULARIO_INSCRICAO"),
            ]

            for query, expected_intent in cases:
                result = pipeline.process_text(query)
                assert result.success is True
                assert result.stage == "WEB_NAVIGATION"
                assert result.classification.intent_name == expected_intent
                assert result.audio_validation is None
                assert result.audio_metrics is None
                assert os.path.isfile(result.screenshot_path)

    def test_pipeline_result_dual_access_pattern(self):
        """Verifies PipelineResult is accessible as both object attributes and dictionary keys."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nav = MockNavigator(artifacts_dir=tmp_dir)
            pipeline = VoiceNavigationPipeline(navigator=nav, artifacts_dir=tmp_dir)

            result = pipeline.process_text("mostrar guia da puc campinas")

            # Attribute access
            assert result.success is True
            assert result.stage == "WEB_NAVIGATION"
            assert result.status == "COMPLETED"
            assert result.is_success is True

            # Dictionary access
            assert result["success"] is True
            assert result["stage"] == "WEB_NAVIGATION"
            assert result["status"] == "COMPLETED"
            assert "GUIA_UNIVERSITARIO" in result["target_url"] or "guia" in result["target_url"]
            assert result.get("success") is True
            assert result.get("nonexistent_key", "default_val") == "default_val"
            assert "user_message" in result

            # Dictionary conversion and JSON serialization
            as_dict = result.to_dict()
            assert isinstance(as_dict, dict)
            assert as_dict["success"] is True
            json_str = json.dumps(as_dict)
            assert "GUIA_UNIVERSITARIO" in json_str

    def test_pipeline_callable_and_coordinator_alias(self):
        """Verifies PipelineCoordinator alias and __call__ syntactic shortcuts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nav = MockNavigator(artifacts_dir=tmp_dir)
            coordinator = PipelineCoordinator(navigator=nav, artifacts_dir=tmp_dir)

            # Process voice command via legacy method
            audio_bytes = SyntheticAudioGenerator.generate_mock_speech(duration_sec=2.5)
            res1 = coordinator.process_voice_command(
                audio_bytes,
                transcript_override="mostrar guia do vestibular da puc campinas",
            )
            assert res1.success is True
            assert res1["classification"].intent_name == "GUIA_UNIVERSITARIO"

            # Process via direct call with text
            res2 = coordinator("abrir edital do vestibular da puc campinas")
            assert res2.success is True
            assert res2.classification.intent_name == "MANUAL_DO_CANDIDATO"


# ===========================================================================
# 2. Audio Validation Rejection Handling
# ===========================================================================
class TestPipelineAudioRejections:
    """Verifies acoustic validation stops the pipeline immediately before neural/web stages."""

    def test_silence_audio_rejection(self):
        nav = MockNavigator()
        pipeline = VoiceNavigationPipeline(navigator=nav)
        silence_bytes = SyntheticAudioGenerator.generate_silence(duration_sec=2.0)

        result = pipeline.run(audio=silence_bytes, transcript_override="mostrar guia")

        assert result.success is False
        assert result.stage == "AUDIO_VALIDATION"
        assert result.status == AudioStatus.SILENCE_DETECTED.value
        assert result.classification is None
        assert result.navigation is None  # Web navigation NEVER triggered
        assert "voz" in result.user_message.lower()

    def test_excessive_noise_rejection(self):
        nav = MockNavigator()
        pipeline = VoiceNavigationPipeline(navigator=nav)
        noise_bytes = SyntheticAudioGenerator.generate_white_noise(duration_sec=2.0, amplitude=0.9)

        result = pipeline.run(audio=noise_bytes, transcript_override="mostrar guia")

        assert result.success is False
        assert result.stage == "AUDIO_VALIDATION"
        assert result.status == AudioStatus.EXCESSIVE_NOISE.value
        assert result.classification is None
        assert result.navigation is None  # Web navigation NEVER triggered
        assert "ruído" in result.user_message.lower() or "ruido" in result.user_message.lower()

    def test_too_short_audio_rejection(self):
        nav = MockNavigator()
        pipeline = VoiceNavigationPipeline(navigator=nav)
        short_bytes = SyntheticAudioGenerator.generate_short_audio(duration_sec=0.3)

        result = pipeline.run(audio=short_bytes, transcript_override="mostrar guia")

        assert result.success is False
        assert result.stage == "AUDIO_VALIDATION"
        assert result.status == AudioStatus.TOO_SHORT.value
        assert result.classification is None
        assert result.navigation is None  # Web navigation NEVER triggered
        assert "curto" in result.user_message.lower()

    def test_too_long_audio_rejection(self):
        nav = MockNavigator()
        pipeline = VoiceNavigationPipeline(navigator=nav)
        long_bytes = SyntheticAudioGenerator.generate_long_audio(duration_sec=14.0)

        result = pipeline.run(audio=long_bytes, transcript_override="mostrar guia")

        assert result.success is False
        assert result.stage == "AUDIO_VALIDATION"
        assert result.status == AudioStatus.TOO_LONG.value
        assert result.classification is None
        assert result.navigation is None  # Web navigation NEVER triggered
        assert "longo" in result.user_message.lower()

    def test_corrupted_audio_rejection(self):
        nav = MockNavigator()
        pipeline = VoiceNavigationPipeline(navigator=nav)
        corrupted_bytes = SyntheticAudioGenerator.generate_corrupted_audio("truncated_header")

        result = pipeline.run(audio=corrupted_bytes)

        assert result.success is False
        assert result.stage == "AUDIO_VALIDATION"
        assert result.status == AudioStatus.CORRUPTED_FORMAT.value
        assert result.classification is None
        assert result.navigation is None  # Web navigation NEVER triggered
        assert "inválido" in result.user_message.lower() or "corrompido" in result.user_message.lower()

    def test_empty_audio_rejection(self):
        nav = MockNavigator()
        pipeline = VoiceNavigationPipeline(navigator=nav)

        result = pipeline.run(audio=b"")

        assert result.success is False
        assert result.stage == "AUDIO_VALIDATION"
        assert result.status == AudioStatus.EMPTY_INPUT.value
        assert result.navigation is None

    def test_audio_source_abstractions(self):
        """Verifies FileAudioSource and BytesAudioSource inputs to the pipeline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "test_cmd.wav"
            speech = SyntheticAudioGenerator.generate_mock_speech(duration_sec=2.5)
            wav_path.write_bytes(speech)

            nav = MockNavigator(artifacts_dir=tmp_dir)
            pipeline = VoiceNavigationPipeline(navigator=nav, artifacts_dir=tmp_dir)

            # Test using FileAudioSource
            file_src = FileAudioSource(wav_path)
            res_file = pipeline.run(
                audio=file_src,
                transcript_override="mostrar guia do vestibular da puc campinas",
            )
            assert res_file.success is True
            assert res_file.classification.intent_name == "GUIA_UNIVERSITARIO"

            # Test using BytesAudioSource
            bytes_src = BytesAudioSource(speech)
            res_bytes = pipeline.run(
                audio=bytes_src,
                transcript_override="abrir manual do candidato da puc campinas",
            )
            assert res_bytes.success is True
            assert res_bytes.classification.intent_name == "MANUAL_DO_CANDIDATO"


# ===========================================================================
# 3. Neural Gating Rejection Handling
# ===========================================================================
class TestPipelineNeuralGatingRejections:
    """Verifies lexical gating and confidence thresholding stop the pipeline before web navigation."""

    def test_ood_query_rejection(self):
        nav = MockNavigator()
        pipeline = VoiceNavigationPipeline(navigator=nav)
        speech_bytes = SyntheticAudioGenerator.generate_mock_speech(duration_sec=3.0)

        # Off-topic phrase without any vestibular domain keywords
        ood_command = "quero pedir uma pizza de quatro queijos para entrega"
        result = pipeline.run(audio=speech_bytes, transcript_override=ood_command)

        assert result.success is False
        assert result.stage == "NEURAL_CLASSIFICATION"
        assert result.status == "OOD_NO_DOMAIN_KEYWORDS"
        assert result.navigation is None  # Web navigation NEVER triggered
        assert "vestibular" in result.user_message.lower()

    def test_ambiguous_domain_only_query_rejection(self):
        nav = MockNavigator()
        pipeline = VoiceNavigationPipeline(navigator=nav)
        speech_bytes = SyntheticAudioGenerator.generate_mock_speech(duration_sec=3.0)

        # Query mentions vestibular anchor but does not specify which destination
        ambiguous_command = "vestibular da puc campinas"
        result = pipeline.run(audio=speech_bytes, transcript_override=ambiguous_command)

        assert result.success is False
        assert result.stage == "NEURAL_CLASSIFICATION"
        assert result.status in ("AMBIGUOUS_DOMAIN_ONLY", "AMBIGUOUS_INTENT")
        assert result.navigation is None  # Web navigation NEVER triggered

    def test_ambiguous_conflicting_intents_rejection(self):
        nav = MockNavigator()
        pipeline = VoiceNavigationPipeline(navigator=nav)
        speech_bytes = SyntheticAudioGenerator.generate_mock_speech(duration_sec=3.0)

        # Query contains conflicting action keywords across all classes
        conflicting_cmd = (
            "quero ver o guia dos cursos e tambem abrir o manual do candidato e fazer a inscricao"
        )
        result = pipeline.run(audio=speech_bytes, transcript_override=conflicting_cmd)

        assert result.success is False
        assert result.stage == "NEURAL_CLASSIFICATION"
        assert result.status == "AMBIGUOUS_INTENT"
        assert result.navigation is None  # Web navigation NEVER triggered

    def test_low_confidence_threshold_rejection(self):
        """Forces low confidence rejection using a strict classifier."""
        # Classifier with impossibly high confidence requirement
        strict_classifier = NeuralIntentClassifier(confidence_threshold=0.999999999)
        nav = MockNavigator()
        pipeline = VoiceNavigationPipeline(classifier=strict_classifier, navigator=nav)

        result = pipeline.process_text("mostrar guia vestibular puc")

        assert result.success is False
        assert result.stage == "NEURAL_CLASSIFICATION"
        assert result.status == "LOW_CONFIDENCE"
        assert result.navigation is None  # Web navigation NEVER triggered
        assert "confiança" in result.user_message.lower() or "confianca" in result.user_message.lower()


# ===========================================================================
# 4. Navigation Errors and Fallback Handling
# ===========================================================================
class TestPipelineNavigationErrorAndFallback:
    """Verifies graceful handling of browser errors, driver exceptions, and custom output paths."""

    class FailingNavigator(BaseNavigator):
        def navigate(self, target_url: str, intent_name: str) -> NavigationResult:
            return NavigationResult(
                success=False,
                intent_name=intent_name,
                target_url=target_url,
                page_title="",
                screenshot_path="",
                metadata_path="",
                headings=[],
                error_message="Erro de comunicação com o navegador (DNS Timeout)",
            )

    class CrashingNavigator(BaseNavigator):
        def navigate(self, target_url: str, intent_name: str) -> NavigationResult:
            raise RuntimeError("Fatal Chrome process crash: WebDriver server closed abruptly.")

    def test_navigation_failure_result_handling(self):
        failing_nav = self.FailingNavigator()
        pipeline = VoiceNavigationPipeline(navigator=failing_nav)

        result = pipeline.process_text("quero ver o guia de cursos da puc campinas")

        assert result.success is False
        assert result.stage == "WEB_NAVIGATION"
        assert result.status == "NAVIGATION_FAILED"
        assert "DNS Timeout" in result.user_message

    def test_navigation_exception_graceful_recovery(self):
        crashing_nav = self.CrashingNavigator()
        pipeline = VoiceNavigationPipeline(navigator=crashing_nav)

        result = pipeline.process_text("abrir manual do candidato da puc campinas")

        assert result.success is False
        assert result.stage == "WEB_NAVIGATION"
        assert result.status == "NAVIGATION_FAILED"
        assert "Fatal Chrome process crash" in result.user_message

    def test_custom_artifacts_directory_creation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_artifacts = str(Path(tmp_dir) / "custom_output_artifacts")
            pipeline = VoiceNavigationPipeline(
                navigator_mode="mock",
                artifacts_dir=custom_artifacts,
            )

            result = pipeline.process_text("como faco para me inscrever no vestibular")

            assert result.success is True
            assert custom_artifacts in result.screenshot_path
            assert custom_artifacts in result.metadata_path
            assert os.path.isfile(result.screenshot_path)
            assert os.path.isfile(result.metadata_path)


# ===========================================================================
# 5. CLI Execution Tests (via subprocess.run)
# ===========================================================================
class TestPipelineCLI:
    """Tests the command-line application directly via subprocess execution."""

    def test_cli_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "reia.pipeline.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "REIA" in proc.stdout
        assert "--audio" in proc.stdout
        assert "--text" in proc.stdout
        assert "--synthetic" in proc.stdout
        assert "--navigator" in proc.stdout
        assert "--json" in proc.stdout

    def test_cli_text_execution(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "reia.pipeline.cli",
                "--text",
                "quero ver o guia universitario de apresentacao dos cursos",
                "--navigator",
                "mock",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "SUCESSO" in proc.stdout
        assert "GUIA_UNIVERSITARIO" in proc.stdout

    def test_cli_synthetic_valid_guia_json(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "reia.pipeline.cli",
                "--synthetic",
                "valid_guia",
                "--navigator",
                "mock",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["success"] is True
        assert data["stage"] == "WEB_NAVIGATION"
        assert data["status"] == "COMPLETED"
        assert data["classification"]["intent_name"] == "GUIA_UNIVERSITARIO"
        assert os.path.isfile(data["screenshot_path"])
        assert os.path.isfile(data["metadata_path"])

    def test_cli_synthetic_valid_manual_json(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "reia.pipeline.cli",
                "--synthetic",
                "valid_manual",
                "--navigator",
                "mock",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["success"] is True
        assert data["classification"]["intent_name"] == "MANUAL_DO_CANDIDATO"

    def test_cli_synthetic_valid_inscricao_json(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "reia.pipeline.cli",
                "--synthetic",
                "valid_inscricao",
                "--navigator",
                "mock",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["success"] is True
        assert data["classification"]["intent_name"] == "FORMULARIO_INSCRICAO"

    def test_cli_synthetic_silence_rejection(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "reia.pipeline.cli",
                "--synthetic",
                "silence",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1
        data = json.loads(proc.stdout)
        assert data["success"] is False
        assert data["stage"] == "AUDIO_VALIDATION"
        assert data["status"] == "SILENCE_DETECTED"

    def test_cli_synthetic_noise_quiet(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "reia.pipeline.cli",
                "--synthetic",
                "noise",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1
        assert "ruído" in proc.stdout.lower() or "ruido" in proc.stdout.lower()

    def test_cli_top_level_module_execution(self):
        """Verifies executing `python3 -m reia` works identically."""
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "reia",
                "--text",
                "como faco para me inscrever no vestibular",
                "--navigator",
                "mock",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "Navegando para FORMULARIO_INSCRICAO" in proc.stdout

    def test_cli_missing_input_json(self):
        proc = subprocess.run(
            [sys.executable, "-m", "reia.pipeline.cli", "--json"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1
        data = json.loads(proc.stdout)
        assert data["success"] is False
        assert data["status"] == "MISSING_INPUT"

    def test_cli_exit_zero_flag(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "reia.pipeline.cli",
                "--synthetic",
                "silence",
                "--exit-zero",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["success"] is False
        assert data["status"] == "SILENCE_DETECTED"


# ===========================================================================
# 6. User Feedback Messaging Formatting
# ===========================================================================
class TestPipelineMessages:
    """Verifies accessible Brazilian Portuguese messaging templates."""

    def test_audio_message_formatting(self):
        msg_short = get_audio_feedback_message("TOO_SHORT", duration=0.4, min_duration=0.8)
        assert "0.40s" in msg_short
        assert "0.8" in msg_short

        msg_long = get_audio_feedback_message("TOO_LONG", duration=15.0, max_duration=12.0)
        assert "15.00s" in msg_long
        assert "12.0" in msg_long

        msg_silence = get_audio_feedback_message("SILENCE_DETECTED")
        assert "voz" in msg_silence.lower()

    def test_neural_message_formatting(self):
        msg_ood = get_neural_feedback_message("OOD_NO_DOMAIN_KEYWORDS")
        assert "vestibular" in msg_ood.lower()

        msg_amb = get_neural_feedback_message("AMBIGUOUS_INTENT")
        assert "ambíguo" in msg_amb or "ambiguo" in msg_amb

        msg_success = get_neural_feedback_message(
            None,
            intent_name="GUIA_UNIVERSITARIO",
            target_url="https://vestibular.puc-campinas.edu.br/guia-universitario/",
        )
        assert "Navegando" in msg_success

    def test_navigation_message_formatting(self):
        msg_ok = get_navigation_feedback_message(
            success=True,
            target_url="https://vestibular.puc-campinas.edu.br/",
            intent_name="FORMULARIO_INSCRICAO",
        )
        assert "Navegando" in msg_ok

        msg_err = get_navigation_feedback_message(
            success=False,
            target_url="https://vestibular.puc-campinas.edu.br/",
            error_message="Connection timed out",
        )
        assert "Falha" in msg_err
        assert "Connection timed out" in msg_err


# ===========================================================================
# Standalone Test Suite Runner
# ===========================================================================
if __name__ == "__main__":
    test_classes = [
        TestPipelineSuccessfulFlows,
        TestPipelineAudioRejections,
        TestPipelineNeuralGatingRejections,
        TestPipelineNavigationErrorAndFallback,
        TestPipelineCLI,
        TestPipelineMessages,
    ]

    start = time.time()
    total_passed = 0
    total_failed = 0

    print("======================================================================")
    print("REIA Voice Navigation Pipeline Subsystem (M4) Unit & Integration Test Suite")
    print("======================================================================")

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_") and callable(getattr(instance, m))]
        print(f"\nRunning {cls.__name__} ({len(methods)} tests)...")
        passed = 0
        failed = 0
        for method_name in methods:
            test_fn = getattr(instance, method_name)
            try:
                test_fn()
                passed += 1
            except Exception as e:
                failed += 1
                print(f"  FAIL: {method_name}: {e}")
                import traceback
                traceback.print_exc()
        print(f"  Result: {passed} passed, {failed} failed (total {len(methods)})")
        total_passed += passed
        total_failed += failed

    elapsed = time.time() - start
    print("\n======================================================================")
    print(f"Pipeline Test Suite Summary: {total_passed} passed, {total_failed} failed in {elapsed:.2f}s")
    print("======================================================================")

    if total_failed > 0:
        sys.exit(1)
    sys.exit(0)

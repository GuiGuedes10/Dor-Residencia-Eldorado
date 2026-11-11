"""Tier 4: Real-World End-to-End Scenarios & Full User Journeys.

Tests authentic end-to-end voice workflows from ingestion to verified browser
navigation and artifact persistence:
- Scenario 1: Prospective Student — Exploring Courses via Guia Universitário
- Scenario 2: Candidate — Checking Exam Rules via Manual do Candidato
- Scenario 3: Applicant — Directly Accessing Registration Form (Inscrição)
- Scenario 4: Silent Microphone Recovery Journey
- Scenario 5: High Ambient Noise / Street Environment Recovery
- Scenario 6: Accidental Quick Tap (Too Short Duration)
- Scenario 7: Off-Topic Query Guidance (Out of Domain)
- Scenario 8: Conflicting Multi-Intent Disambiguation
- Scenario 9: Web Navigation Network Timeout Graceful Degradation
- Scenario 10: Sequential Multi-Command Session Isolation
- Scenario 11: Corrupted Audio File Upload Handling
- Scenario 12: Long Monologue Duration Boundary Rejection

Total Tier 4 Test Cases: 12
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

try:
    import pytest
except ImportError:
    from conftest import pytest

from conftest import (
    AudioStatus,
    ClassificationResult,
    MockNavigator,
    ReferenceNeuralClassifier,
    ReferencePipelineCoordinator,
    SyntheticAudioGenerator,
    get_audio_validator,
    get_neural_classifier,
    get_synthetic_audio_gen,
    get_text_sanitizer,
    get_web_navigator,
)

TARGET_URLS = {
    "GUIA_UNIVERSITARIO": "https://vestibular.puc-campinas.edu.br/guia-universitario/",
    "MANUAL_DO_CANDIDATO": "https://vestibular.puc-campinas.edu.br/manual-do-candidato/",
    "FORMULARIO_INSCRICAO": "https://vestibular.puc-campinas.edu.br/",
}


# ===========================================================================
# Scenario 1: Prospective Student — Exploring Courses via Guia Universitário
# ===========================================================================
def test_scenario_guia_universitario_full_journey(tmp_path):
    gen = get_synthetic_audio_gen()
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    # 1. User records voice query
    audio_bytes = gen.generate_speech_mock(duration_sec=3.0)
    # 2. Whisper transcription with natural spoken Portuguese
    user_speech = "Quero ver a apresentação do guia universitário de cursos da PUC-Campinas!"

    # 3. Pipeline execution
    result = coordinator.process_voice_command(audio_bytes, transcript_override=user_speech)

    assert result["success"] is True
    assert result["stage"] == "WEB_NAVIGATION"
    assert result["status"] == "COMPLETED"

    # Verify classification
    clf = result["classification"]
    assert clf.intent_name == "GUIA_UNIVERSITARIO"
    assert clf.target_url == TARGET_URLS["GUIA_UNIVERSITARIO"]
    assert clf.confidence >= 0.65

    # Verify navigation & artifacts
    nav_res = result["navigation"]
    assert nav_res.success is True
    assert "Guia Universitário" in nav_res.page_title
    assert os.path.exists(nav_res.screenshot_path)
    assert os.path.exists(nav_res.metadata_path)

    # Verify screenshot is non-empty valid PNG
    screenshot_bytes = Path(nav_res.screenshot_path).read_bytes()
    assert screenshot_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    # Verify metadata JSON
    meta = json.loads(Path(nav_res.metadata_path).read_text(encoding="utf-8"))
    assert meta["intent_name"] == "GUIA_UNIVERSITARIO"
    assert meta["links_count"] > 0


# ===========================================================================
# Scenario 2: Candidate — Checking Exam Rules via Manual do Candidato
# ===========================================================================
def test_scenario_manual_do_candidato_full_journey(tmp_path):
    gen = get_synthetic_audio_gen()
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    audio_bytes = gen.generate_speech_mock(duration_sec=3.5)
    user_speech = "Onde consulto o manual do candidato com as regras da prova e o edital da PUC?"

    result = coordinator.process_voice_command(audio_bytes, transcript_override=user_speech)

    assert result["success"] is True
    assert result["stage"] == "WEB_NAVIGATION"
    assert result["classification"].intent_name == "MANUAL_DO_CANDIDATO"

    nav_res = result["navigation"]
    assert nav_res.success is True
    assert "Manual" in nav_res.page_title
    assert any("edital" in link["href"].lower() for link in nav_res.links)
    assert os.path.isfile(nav_res.screenshot_path)
    assert os.path.isfile(nav_res.metadata_path)


# ===========================================================================
# Scenario 3: Applicant — Directly Accessing Registration Form (Inscrição)
# ===========================================================================
def test_scenario_formulario_inscricao_full_journey(tmp_path):
    gen = get_synthetic_audio_gen()
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    audio_bytes = gen.generate_speech_mock(duration_sec=3.2)
    user_speech = "Como faço para abrir a página de inscrição e preencher meu cadastro no vestibular da PUC?"

    result = coordinator.process_voice_command(audio_bytes, transcript_override=user_speech)

    assert result["success"] is True
    assert result["stage"] == "WEB_NAVIGATION"
    assert result["classification"].intent_name == "FORMULARIO_INSCRICAO"

    nav_res = result["navigation"]
    assert nav_res.success is True
    assert "Inscrição" in nav_res.page_title
    assert os.path.isfile(nav_res.screenshot_path)


# ===========================================================================
# Scenario 4: Silent Microphone Recovery Journey
# ===========================================================================
def test_scenario_microphone_muted_or_silent(tmp_path):
    gen = get_synthetic_audio_gen()
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    # User voice input is pure silence (microphone muted)
    silent_audio = gen.generate_silence(duration_sec=2.0)
    result = coordinator.process_voice_command(silent_audio, transcript_override="mostrar guia")

    # Pipeline gracefully intercepts at audio stage
    assert result["success"] is False
    assert result["stage"] == "AUDIO_VALIDATION"
    assert result["status"] == AudioStatus.SILENCE_DETECTED.value
    assert "não detectamos sua voz" in result["user_message"].lower()

    # Neither classification nor navigation was invoked
    assert result["classification"] is None
    assert result["navigation"] is None


# ===========================================================================
# Scenario 5: High Ambient Noise / Street Environment Recovery
# ===========================================================================
def test_scenario_noisy_traffic_environment(tmp_path):
    gen = get_synthetic_audio_gen()
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    # High ambient noise with low SNR (< 3.0 dB)
    noisy_audio = gen.generate_white_noise(duration_sec=2.5, amplitude=0.9)
    result = coordinator.process_voice_command(noisy_audio, transcript_override="abrir manual")

    assert result["success"] is False
    assert result["stage"] == "AUDIO_VALIDATION"
    assert result["status"] == AudioStatus.EXCESSIVE_NOISE.value
    assert "ruído excessivo" in result["user_message"].lower()
    assert result["navigation"] is None


# ===========================================================================
# Scenario 6: Accidental Quick Tap (Too Short Duration)
# ===========================================================================
def test_scenario_accidental_quick_tap(tmp_path):
    gen = get_synthetic_audio_gen()
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    # 0.25s duration
    too_short_audio = gen.generate_duration(duration_sec=0.25, speech=False)
    result = coordinator.process_voice_command(too_short_audio, transcript_override="guia")

    assert result["success"] is False
    assert result["stage"] == "AUDIO_VALIDATION"
    assert result["status"] == AudioStatus.TOO_SHORT.value
    assert "muito curto" in result["user_message"].lower()


# ===========================================================================
# Scenario 7: Off-Topic Query Guidance (Out of Domain)
# ===========================================================================
def test_scenario_off_topic_pizza_order(tmp_path):
    gen = get_synthetic_audio_gen()
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    valid_audio = gen.generate_speech_mock(duration_sec=3.0)
    pizza_query = "Quero pedir uma pizza de calabresa com refrigerante"
    result = coordinator.process_voice_command(valid_audio, transcript_override=pizza_query)

    assert result["success"] is False
    assert result["stage"] == "NEURAL_CLASSIFICATION"
    assert result["status"] == "OOD_NO_DOMAIN_KEYWORDS"
    assert "fora de domínio" in result["user_message"].lower()
    assert result["navigation"] is None


# ===========================================================================
# Scenario 8: Conflicting Multi-Intent Disambiguation
# ===========================================================================
def test_scenario_conflicting_ambiguous_request(tmp_path):
    gen = get_synthetic_audio_gen()
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    valid_audio = gen.generate_speech_mock(duration_sec=3.0)
    ambiguous_query = "Quero o guia e o manual do candidato da PUC"
    result = coordinator.process_voice_command(valid_audio, transcript_override=ambiguous_query)

    assert result["success"] is False
    assert result["stage"] == "NEURAL_CLASSIFICATION"
    assert result["status"] == "AMBIGUOUS_INTENT"
    assert "ambíguo" in result["user_message"].lower() or "uma opção" in result["user_message"].lower()
    assert result["navigation"] is None


# ===========================================================================
# Scenario 9: Web Navigation Network Timeout Graceful Degradation
# ===========================================================================
def test_scenario_web_server_timeout_resilience(tmp_path):
    gen = get_synthetic_audio_gen()
    artifacts_dir = tmp_path / "artifacts"
    # Navigator configured to simulate network timeout
    nav = MockNavigator(artifacts_dir=str(artifacts_dir), simulate_timeout=True)
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    valid_audio = gen.generate_speech_mock(duration_sec=3.0)
    valid_query = "Quero ver o guia universitário da PUC-Campinas"
    result = coordinator.process_voice_command(valid_audio, transcript_override=valid_query)

    assert result["success"] is False
    assert result["stage"] == "WEB_NAVIGATION"
    assert result["status"] == "NAVIGATION_FAILED"
    assert "timeout" in result["user_message"].lower() or "tempo limite" in result["user_message"].lower()


# ===========================================================================
# Scenario 10: Sequential Multi-Command Session Isolation
# ===========================================================================
def test_scenario_sequential_multi_command_sessions(tmp_path):
    gen = get_synthetic_audio_gen()
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    audio = gen.generate_speech_mock(duration_sec=3.0)

    # 1. First command: Guia
    res1 = coordinator.process_voice_command(audio, transcript_override="mostrar guia da puc")
    assert res1["success"] is True
    assert res1["classification"].intent_name == "GUIA_UNIVERSITARIO"

    # 2. Second command: Manual
    res2 = coordinator.process_voice_command(audio, transcript_override="abrir manual de regras da prova puc")
    assert res2["success"] is True
    assert res2["classification"].intent_name == "MANUAL_DO_CANDIDATO"

    # 3. Third command: Inscrição
    res3 = coordinator.process_voice_command(audio, transcript_override="como faco para me inscrever no vestibular puc")
    assert res3["success"] is True
    assert res3["classification"].intent_name == "FORMULARIO_INSCRICAO"

    # Verify screenshots are all distinct files
    screenshots = [
        res1["navigation"].screenshot_path,
        res2["navigation"].screenshot_path,
        res3["navigation"].screenshot_path,
    ]
    assert len(set(screenshots)) == 3
    for path in screenshots:
        assert os.path.isfile(path)


# ===========================================================================
# Scenario 11: Corrupted Audio File Upload Handling
# ===========================================================================
def test_scenario_corrupted_audio_file_upload(tmp_path):
    gen = get_synthetic_audio_gen()
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    corrupted_wav = gen.generate_corrupted("truncated_header")
    result = coordinator.process_voice_command(corrupted_wav, transcript_override="mostrar guia")

    assert result["success"] is False
    assert result["stage"] == "AUDIO_VALIDATION"
    assert result["status"] == AudioStatus.CORRUPTED_FORMAT.value
    assert "falha" in result["user_message"].lower() or "formato" in result["user_message"].lower()


# ===========================================================================
# Scenario 12: Long Monologue Duration Boundary Rejection
# ===========================================================================
def test_scenario_excessive_duration_monologue(tmp_path):
    gen = get_synthetic_audio_gen()
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    # 16.0s duration monologue (exceeding 12.0s limit)
    long_audio = gen.generate_duration(duration_sec=16.0, speech=True)
    result = coordinator.process_voice_command(long_audio, transcript_override="mostrar guia")

    assert result["success"] is False
    assert result["stage"] == "AUDIO_VALIDATION"
    assert result["status"] == AudioStatus.TOO_LONG.value
    assert "muito longo" in result["user_message"].lower()


# ===========================================================================
# Standalone execution runner
# ===========================================================================
if __name__ == "__main__":
    import inspect
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = 0
    failed = 0
    print(f"Running {len(tests)} Tier 4 scenario tests...")

    for test in tests:
        try:
            sig = inspect.signature(test)
            if "tmp_path" in sig.parameters:
                with tempfile.TemporaryDirectory() as td:
                    test(Path(td))
            else:
                test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")

    print(f"\nTier 4 Summary: {passed} passed, {failed} failed out of {len(tests)} tests.")
    if failed > 0:
        raise SystemExit(1)

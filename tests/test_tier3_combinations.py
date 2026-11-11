"""Tier 3: Pairwise Combinations & Cross-Feature Interactions.

Tests cross-subsystem interface contracts and cascading flows:
- Combo 1: Audio Ingestion x Audio Validation
- Combo 2: Audio Validation x Transcription Sanitizer
- Combo 3: Text Sanitizer x Neural Vectorization & Inference
- Combo 4: Audio Gate Short-Circuiting (Audio failure halts pipeline)
- Combo 5: Neural Gate Short-Circuiting (OOD/ambiguity halts navigation)
- Combo 6: Intent Classification x URL Route Dispatching
- Combo 7: Navigation Execution x Dual Artifact Persistence
- Combo 8: Complex Multi-Feature Robustness Flows

Total Tier 3 Test Cases: 18
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
    reference_sanitize_transcription,
    reference_validate_audio,
)

TARGET_URLS = {
    "GUIA_UNIVERSITARIO": "https://vestibular.puc-campinas.edu.br/guia-universitario/",
    "MANUAL_DO_CANDIDATO": "https://vestibular.puc-campinas.edu.br/manual-do-candidato/",
    "FORMULARIO_INSCRICAO": "https://vestibular.puc-campinas.edu.br/",
}


# ===========================================================================
# Combo 1: Audio Ingestion x Audio Validation
# ===========================================================================
def test_combo_valid_audio_ingestion_to_validation():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    wav_bytes = gen.generate_speech_mock(duration_sec=3.5)
    res = val(wav_bytes)

    assert res.is_valid is True
    assert res.status == AudioStatus.VALID
    assert res.processed_wav_bytes is not None
    assert len(res.processed_wav_bytes) > 0


def test_combo_corrupt_stream_ingestion_to_validation():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    corrupt_stream = gen.generate_corrupted("truncated_header")
    res = val(corrupt_stream)

    assert res.is_valid is False
    assert res.status == AudioStatus.CORRUPTED_FORMAT
    assert res.processed_wav_bytes is None


# ===========================================================================
# Combo 2: Audio Validation x Transcription Sanitizer
# ===========================================================================
def test_combo_valid_audio_to_sanitizer_flow():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    sanitizer = get_text_sanitizer()

    wav_bytes = gen.generate_speech_mock(duration_sec=2.5)
    res = val(wav_bytes)
    assert res.is_valid is True

    # Raw speech recognition output simulating Whisper transcription
    whisper_raw = "[Música] Olá, quero ver o GUIA de Cursos da PUC-Campinas!"
    sanitized = sanitizer(whisper_raw)

    assert "música" not in sanitized
    assert "guia de cursos da puc campinas" in sanitized
    assert "-" not in sanitized
    assert "!" not in sanitized


def test_combo_silent_audio_produces_empty_sanitization():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    sanitizer = get_text_sanitizer()

    silent_wav = gen.generate_silence(duration_sec=2.0)
    res = val(silent_wav)
    assert res.is_valid is False

    # Whisper output on silence is either empty or acoustic artifacts
    silence_transcript = "[Ruído de fundo] ... (silêncio) ..."
    sanitized = sanitizer(silence_transcript)
    assert sanitized == ""


# ===========================================================================
# Combo 3: Text Sanitizer x Neural Vectorization & Inference
# ===========================================================================
def test_combo_sanitizer_to_neural_guia():
    sanitizer = get_text_sanitizer()
    clf = get_neural_classifier()

    raw_voice = "Quero ver a apresentação do Guia Universitário da PUC-Campinas!"
    clean = sanitizer(raw_voice)
    res = clf.classify(clean)

    assert res.is_valid_intent is True
    assert res.intent_name == "GUIA_UNIVERSITARIO"
    assert res.target_url == TARGET_URLS["GUIA_UNIVERSITARIO"]


def test_combo_sanitizer_to_neural_manual():
    sanitizer = get_text_sanitizer()
    clf = get_neural_classifier()

    raw_voice = "Onde consulto o Manual do Candidato com o Edital e as Normas da prova???"
    clean = sanitizer(raw_voice)
    res = clf.classify(clean)

    assert res.is_valid_intent is True
    assert res.intent_name == "MANUAL_DO_CANDIDATO"
    assert res.target_url == TARGET_URLS["MANUAL_DO_CANDIDATO"]


def test_combo_sanitizer_to_neural_inscricao():
    sanitizer = get_text_sanitizer()
    clf = get_neural_classifier()

    raw_voice = "Como faço para preencher a minha inscrição no vestibular da PUC?"
    clean = sanitizer(raw_voice)
    res = clf.classify(clean)

    assert res.is_valid_intent is True
    assert res.intent_name == "FORMULARIO_INSCRICAO"
    assert res.target_url == TARGET_URLS["FORMULARIO_INSCRICAO"]


# ===========================================================================
# Combo 4: Audio Gate Short-Circuiting (Pipeline stops at audio failure)
# ===========================================================================
def test_combo_audio_silence_short_circuits_pipeline(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    silence_bytes = gen.generate_silence(duration_sec=2.0)
    result = coordinator.process_voice_command(silence_bytes, transcript_override="mostrar guia")

    assert result["success"] is False
    assert result["stage"] == "AUDIO_VALIDATION"
    assert result["classification"] is None
    assert result["navigation"] is None


def test_combo_audio_noise_short_circuits_pipeline(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    noise_bytes = gen.generate_white_noise(duration_sec=2.0, amplitude=0.9)
    result = coordinator.process_voice_command(noise_bytes, transcript_override="mostrar guia")

    assert result["success"] is False
    assert result["stage"] == "AUDIO_VALIDATION"
    assert result["status"] == AudioStatus.EXCESSIVE_NOISE.value
    assert result["classification"] is None
    assert result["navigation"] is None


def test_combo_audio_short_duration_short_circuits_pipeline(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    short_bytes = gen.generate_duration(duration_sec=0.3, speech=False)
    result = coordinator.process_voice_command(short_bytes, transcript_override="mostrar guia")

    assert result["success"] is False
    assert result["stage"] == "AUDIO_VALIDATION"
    assert result["status"] == AudioStatus.TOO_SHORT.value
    assert result["classification"] is None


# ===========================================================================
# Combo 5: Neural Gate Short-Circuiting (Pipeline stops at neural failure)
# ===========================================================================
def test_combo_neural_ood_short_circuits_navigation(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    audio = gen.generate_speech_mock(duration_sec=3.0)
    ood_phrase = "qual a receita de bolo de chocolate"
    result = coordinator.process_voice_command(audio, transcript_override=ood_phrase)

    assert result["success"] is False
    assert result["stage"] == "NEURAL_CLASSIFICATION"
    assert result["status"] == "OOD_NO_DOMAIN_KEYWORDS"
    assert result["navigation"] is None


def test_combo_neural_ambiguous_short_circuits_navigation(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    audio = gen.generate_speech_mock(duration_sec=3.0)
    ambiguous = "quero o guia e o manual do candidato da puc"
    result = coordinator.process_voice_command(audio, transcript_override=ambiguous)

    assert result["success"] is False
    assert result["stage"] == "NEURAL_CLASSIFICATION"
    assert result["status"] == "AMBIGUOUS_INTENT"
    assert result["navigation"] is None


def test_combo_neural_anchor_only_short_circuits_navigation(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    audio = gen.generate_speech_mock(duration_sec=3.0)
    anchor_only = "vestibular puc campinas"
    result = coordinator.process_voice_command(audio, transcript_override=anchor_only)

    assert result["success"] is False
    assert result["stage"] == "NEURAL_CLASSIFICATION"
    assert result["status"] == "AMBIGUOUS_DOMAIN_ONLY"
    assert result["navigation"] is None


# ===========================================================================
# Combo 6: Intent Classification x URL Route Dispatching
# ===========================================================================
def test_combo_intent_to_route_guia(tmp_path):
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    res = nav.navigate_intent("GUIA_UNIVERSITARIO")

    assert res.success is True
    assert res.target_url == "https://vestibular.puc-campinas.edu.br/guia-universitario/"
    assert "Guia Universitário" in res.page_title


def test_combo_intent_to_route_manual(tmp_path):
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    res = nav.navigate_intent("MANUAL_DO_CANDIDATO")

    assert res.success is True
    assert res.target_url == "https://vestibular.puc-campinas.edu.br/manual-do-candidato/"
    assert "Manual" in res.page_title


def test_combo_intent_to_route_inscricao(tmp_path):
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    res = nav.navigate_intent("FORMULARIO_INSCRICAO")

    assert res.success is True
    assert res.target_url == "https://vestibular.puc-campinas.edu.br/"
    assert "Inscrição" in res.page_title


# ===========================================================================
# Combo 7: Navigation Execution x Dual Artifact Persistence
# ===========================================================================
def test_combo_navigation_persists_screenshot_and_metadata(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    res = nav.navigate_intent("GUIA_UNIVERSITARIO")

    assert res.success is True
    assert os.path.isfile(res.screenshot_path)
    assert os.path.isfile(res.metadata_path)

    # Verify screenshot integrity
    png_bytes = Path(res.screenshot_path).read_bytes()
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    # Verify JSON metadata content
    meta = json.loads(Path(res.metadata_path).read_text(encoding="utf-8"))
    assert meta["intent_name"] == "GUIA_UNIVERSITARIO"
    assert meta["target_url"] == TARGET_URLS["GUIA_UNIVERSITARIO"]
    assert len(meta["headings"]) > 0


# ===========================================================================
# Combo 8: Multi-Feature Robustness Flows
# ===========================================================================
def test_combo_noisy_speech_with_accents_complete_pipeline(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    # Valid speech audio with realistic voice envelope
    audio = gen.generate_speech_mock(duration_sec=3.2)
    # Transcribed command with accents, casing, and punctuation
    raw_command = "Como faço para acessar o Guia de Cursos do Vestibular PUC-Campinas?"
    result = coordinator.process_voice_command(audio, transcript_override=raw_command)

    assert result["success"] is True
    assert result["stage"] == "WEB_NAVIGATION"
    assert result["classification"].intent_name == "GUIA_UNIVERSITARIO"
    assert result["navigation"].success is True
    assert os.path.exists(result["navigation"].screenshot_path)


# ===========================================================================
# Standalone execution runner
# ===========================================================================
if __name__ == "__main__":
    import inspect
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = 0
    failed = 0
    print(f"Running {len(tests)} Tier 3 combination tests...")

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

    print(f"\nTier 3 Summary: {passed} passed, {failed} failed out of {len(tests)} tests.")
    if failed > 0:
        raise SystemExit(1)

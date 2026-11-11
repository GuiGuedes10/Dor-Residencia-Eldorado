"""Tier 1: Feature Coverage (Category-Partition Testing).

Validates each core feature against its functional specification:
- Feature 1: Audio Validation Engine (>=5 tests)
- Feature 2: Text Sanitization & Normalization (>=5 tests)
- Feature 3: Neural Intent Classification (>=5 tests)
- Feature 4: Confidence & Multi-Metric Thresholding (>=5 tests)
- Feature 5: Web Navigation Automation (>=5 tests)
- Feature 6: Orchestration Pipeline & User Messaging (>=5 tests)

Total Tier 1 Test Cases: 36
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# Support running directly or via pytest
try:
    import pytest
except ImportError:
    from conftest import pytest  # fallback stub

from conftest import (
    AudioStatus,
    ClassificationResult,
    MockNavigator,
    ReferenceNeuralClassifier,
    ReferencePipelineCoordinator,
    SyntheticAudioGenerator,
    generate_valid_png_bytes,
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
# Feature 1: Audio Validation Engine
# ===========================================================================
def test_audio_validation_valid_speech_accepted():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    wav_bytes = gen.generate_speech_mock(duration_sec=3.0)
    result = val(wav_bytes)

    assert result.is_valid is True
    assert result.status == AudioStatus.VALID
    assert "válido" in result.message.lower()
    assert 2.9 <= result.duration_seconds <= 3.1


def test_audio_validation_rms_energy_calculation():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    wav_bytes = gen.generate_speech_mock(duration_sec=2.5)
    result = val(wav_bytes)

    assert result.is_valid is True
    assert result.rms_dbfs > -46.0, f"RMS {result.rms_dbfs} dBFS should exceed -46 dBFS silence limit"


def test_audio_validation_silence_ratio_detection():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    wav_bytes = gen.generate_silence(duration_sec=2.0)
    result = val(wav_bytes)

    assert result.is_valid is False
    assert result.status == AudioStatus.SILENCE_DETECTED
    assert result.silence_ratio >= 0.85


def test_audio_validation_snr_estimation():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    # High noise signal
    noisy_wav = gen.generate_white_noise(duration_sec=2.0, amplitude=0.9)
    result = val(noisy_wav)

    assert result.is_valid is False
    assert result.status == AudioStatus.EXCESSIVE_NOISE
    assert result.snr_db < 3.0, f"Expected SNR < 3.0 dB, got {result.snr_db}"


def test_audio_validation_active_duration_metric():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    wav_bytes = gen.generate_speech_mock(duration_sec=4.0)
    result = val(wav_bytes)

    assert result.is_valid is True
    assert result.duration_seconds >= 0.8
    assert result.silence_ratio < 0.85


def test_audio_validation_rejection_status_structure():
    val = get_audio_validator()
    # Test with empty bytes
    result = val(b"")

    assert result.is_valid is False
    assert result.status == AudioStatus.EMPTY_INPUT
    assert len(result.message) > 0
    assert result.duration_seconds == 0.0


# ===========================================================================
# Feature 2: Text Sanitization & Normalization
# ===========================================================================
def test_sanitization_lowercasing():
    sanitizer = get_text_sanitizer()
    raw = "QUERO VER O GUIA DO VESTIBULAR DA PUC CAMPINAS"
    cleaned = sanitizer(raw)
    assert cleaned == cleaned.lower()
    assert "puc" in cleaned
    assert "campinas" in cleaned


def test_sanitization_diacritic_removal():
    sanitizer = get_text_sanitizer()
    raw = "Como faço para ver a página de inscrição e horários?"
    cleaned = sanitizer(raw)

    assert "faço" not in cleaned
    assert "faco" in cleaned
    assert "página" not in cleaned
    assert "pagina" in cleaned
    assert "inscrição" not in cleaned
    assert "inscricao" in cleaned
    assert "horários" not in cleaned
    assert "horarios" in cleaned


def test_sanitization_punctuation_removal():
    sanitizer = get_text_sanitizer()
    raw = "puc-campinas! onde encontro o edital, as regras e datas???"
    cleaned = sanitizer(raw)

    assert "-" not in cleaned
    assert "!" not in cleaned
    assert "?" not in cleaned
    assert "," not in cleaned
    assert "puc campinas" in cleaned
    assert "edital" in cleaned


def test_sanitization_bracketed_tag_removal():
    sanitizer = get_text_sanitizer()
    raw = "[Música] Olá, quero ver o manual do candidato (aplausos)"
    cleaned = sanitizer(raw)

    assert "música" not in cleaned
    assert "musica" not in cleaned
    assert "aplausos" not in cleaned
    assert "manual do candidato" in cleaned


def test_sanitization_whitespace_collapsing():
    sanitizer = get_text_sanitizer()
    raw = "   mostrar    o   guia   universitario   da   puc    "
    cleaned = sanitizer(raw)

    assert not cleaned.startswith(" ")
    assert not cleaned.endswith(" ")
    assert "  " not in cleaned
    assert cleaned == "mostrar o guia universitario da puc"


def test_sanitization_empty_or_none_handling():
    sanitizer = get_text_sanitizer()
    assert sanitizer("") == ""
    assert sanitizer("   ") == ""
    assert sanitizer(None) == ""
    assert sanitizer("[ruido]") == ""


# ===========================================================================
# Feature 3: Neural Intent Classification
# ===========================================================================
def test_neural_classification_guia_intent():
    clf = get_neural_classifier()
    phrase = "quero ver a apresentacao do guia da puc campinas"
    res = clf.classify(phrase)

    assert res.is_valid_intent is True
    assert res.intent_name == "GUIA_UNIVERSITARIO"
    assert res.intent_id == 0
    assert res.confidence >= 0.65


def test_neural_classification_manual_intent():
    clf = get_neural_classifier()
    phrase = "abrir manual do candidato com as regras da prova e edital"
    res = clf.classify(phrase)

    assert res.is_valid_intent is True
    assert res.intent_name == "MANUAL_DO_CANDIDATO"
    assert res.intent_id == 1
    assert res.confidence >= 0.65


def test_neural_classification_inscricao_intent():
    clf = get_neural_classifier()
    phrase = "como faco para me inscrever no vestibular da puc campinas"
    res = clf.classify(phrase)

    assert res.is_valid_intent is True
    assert res.intent_name == "FORMULARIO_INSCRICAO"
    assert res.intent_id == 2
    assert res.confidence >= 0.65


def test_neural_classification_target_url_mapping():
    clf = get_neural_classifier()
    for intent, expected_url in TARGET_URLS.items():
        if intent == "GUIA_UNIVERSITARIO":
            phrase = "mostrar guia de cursos vestibular puc"
        elif intent == "MANUAL_DO_CANDIDATO":
            phrase = "onde encontro o edital e manual de regras da prova puc"
        else:
            phrase = "abrir formulario para cadastrar minha inscricao no vestibular puc"

        res = clf.classify(phrase)
        assert res.is_valid_intent is True
        assert res.target_url == expected_url, f"Expected {expected_url}, got {res.target_url}"


def test_neural_classification_probability_distribution():
    clf = get_neural_classifier()
    res = clf.classify("quero conhecer os cursos da puc campinas no guia")

    assert res.is_valid_intent is True
    assert isinstance(res.probabilities, dict)
    assert len(res.probabilities) == 3
    total_prob = sum(res.probabilities.values())
    assert 0.99 <= total_prob <= 1.01, f"Probabilities sum {total_prob} not normalized"


def test_neural_classification_matched_tokens():
    clf = get_neural_classifier()
    res = clf.classify("abrir manual oficial do candidato vestibular puc campinas")

    assert res.is_valid_intent is True
    assert len(res.matched_domain_tokens) >= 2
    assert "puc" in res.matched_domain_tokens
    assert "manual" in res.matched_domain_tokens


# ===========================================================================
# Feature 4: Confidence & Multi-Metric Thresholding
# ===========================================================================
def test_thresholding_high_confidence_pass():
    clf = get_neural_classifier()
    res = clf.classify("quero ver o guia universitario de cursos da puc campinas")

    assert res.is_valid_intent is True
    assert res.confidence >= 0.65
    assert res.margin >= 0.20
    assert res.entropy <= 0.65
    assert res.rejection_code is None


def test_thresholding_low_confidence_rejection():
    clf = ReferenceNeuralClassifier(confidence_threshold=0.99)
    res = clf.classify("abrir guia da puc")

    assert res.is_valid_intent is False
    assert res.rejection_code == "LOW_CONFIDENCE"
    assert "repita o comando" in res.user_message.lower()


def test_thresholding_narrow_margin_rejection():
    clf = get_neural_classifier()
    # Ambiguous input requesting both manual rules and course guide
    ambiguous = "quero ver o guia e o manual do candidato no vestibular da puc"
    res = clf.classify(ambiguous)

    assert res.is_valid_intent is False
    assert res.rejection_code == "AMBIGUOUS_INTENT"
    assert "ambíguo" in res.user_message.lower() or "escolha" in res.user_message.lower()


def test_thresholding_high_entropy_rejection():
    # If entropy threshold is extremely strict (e.g. 0.01), normal distributions are filtered
    clf = ReferenceNeuralClassifier(entropy_threshold=0.01)
    res = clf.classify("vestibular puc guia cursos")

    assert res.is_valid_intent is False
    assert res.rejection_code == "AMBIGUOUS_INTENT"


def test_thresholding_lexical_gating_ood():
    clf = get_neural_classifier()
    ood_phrase = "qual a previsao do tempo para amanha em sao paulo"
    res = clf.classify(ood_phrase)

    assert res.is_valid_intent is False
    assert res.rejection_code == "OOD_NO_DOMAIN_KEYWORDS"
    assert "fora de domínio" in res.user_message.lower() or "vestibular" in res.user_message.lower()


def test_thresholding_lexical_gating_anchor_only():
    clf = get_neural_classifier()
    anchor_phrase = "puc campinas vestibular"
    res = clf.classify(anchor_phrase)

    assert res.is_valid_intent is False
    assert res.rejection_code == "AMBIGUOUS_DOMAIN_ONLY"
    assert "especifique" in res.user_message.lower()


# ===========================================================================
# Feature 5: Web Navigation Automation
# ===========================================================================
def test_navigation_guia_route_success(tmp_path):
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    res = nav.navigate_intent("GUIA_UNIVERSITARIO")

    assert res.success is True
    assert res.target_url == TARGET_URLS["GUIA_UNIVERSITARIO"]
    assert "Guia Universitário" in res.page_title
    assert any("Cursos" in h for h in res.headings)
    assert len(res.links) > 0


def test_navigation_manual_route_success(tmp_path):
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    res = nav.navigate_intent("MANUAL_DO_CANDIDATO")

    assert res.success is True
    assert res.target_url == TARGET_URLS["MANUAL_DO_CANDIDATO"]
    assert "Manual" in res.page_title
    assert any("edital" in link["href"].lower() for link in res.links)


def test_navigation_inscricao_route_success(tmp_path):
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    res = nav.navigate_intent("FORMULARIO_INSCRICAO")

    assert res.success is True
    assert res.target_url == TARGET_URLS["FORMULARIO_INSCRICAO"]
    assert "Inscrição" in res.page_title


def test_navigation_screenshot_creation(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    res = nav.navigate_intent("GUIA_UNIVERSITARIO")

    assert os.path.exists(res.screenshot_path)
    file_bytes = Path(res.screenshot_path).read_bytes()
    # PNG signature: 89 50 4E 47 0D 0A 1A 0A
    assert file_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(file_bytes) > 100


def test_navigation_metadata_json_creation(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    nav = MockNavigator(artifacts_dir=str(artifacts_dir))
    res = nav.navigate_intent("MANUAL_DO_CANDIDATO")

    assert os.path.exists(res.metadata_path)
    data = json.loads(Path(res.metadata_path).read_text(encoding="utf-8"))
    assert data["intent_name"] == "MANUAL_DO_CANDIDATO"
    assert data["target_url"] == TARGET_URLS["MANUAL_DO_CANDIDATO"]
    assert "screenshot_file" in data


def test_navigation_error_state_handling(tmp_path):
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"), simulate_error=True)
    res = nav.navigate_intent("GUIA_UNIVERSITARIO")

    assert res.success is False
    assert res.error_message is not None
    assert "Falha" in res.error_message


# ===========================================================================
# Feature 6: CLI & Orchestration Pipeline
# ===========================================================================
def test_pipeline_e2e_successful_guia_dispatch(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    audio = gen.generate_speech_mock(duration_sec=3.0)
    transcript = "quero ver o guia universitario de apresentacao dos cursos da puc"
    result = coordinator.process_voice_command(audio, transcript_override=transcript)

    assert result["success"] is True
    assert result["stage"] == "WEB_NAVIGATION"
    assert result["status"] == "COMPLETED"
    assert result["classification"].intent_name == "GUIA_UNIVERSITARIO"
    assert result["navigation"].success is True


def test_pipeline_e2e_successful_manual_dispatch(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    audio = gen.generate_speech_mock(duration_sec=3.0)
    transcript = "onde vejo o edital e as normas do vestibular da puc campinas"
    result = coordinator.process_voice_command(audio, transcript_override=transcript)

    assert result["success"] is True
    assert result["stage"] == "WEB_NAVIGATION"
    assert result["classification"].intent_name == "MANUAL_DO_CANDIDATO"


def test_pipeline_e2e_successful_inscricao_dispatch(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    audio = gen.generate_speech_mock(duration_sec=3.0)
    transcript = "como faco para me inscrever no vestibular da puc campinas"
    result = coordinator.process_voice_command(audio, transcript_override=transcript)

    assert result["success"] is True
    assert result["stage"] == "WEB_NAVIGATION"
    assert result["classification"].intent_name == "FORMULARIO_INSCRICAO"


def test_pipeline_audio_rejection_halts_pipeline(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    silent_audio = gen.generate_silence(duration_sec=2.0)
    result = coordinator.process_voice_command(silent_audio, transcript_override="mostrar guia")

    assert result["success"] is False
    assert result["stage"] == "AUDIO_VALIDATION"
    assert result["status"] == AudioStatus.SILENCE_DETECTED.value
    assert result["classification"] is None
    assert result["navigation"] is None


def test_pipeline_neural_rejection_halts_pipeline(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    audio = gen.generate_speech_mock(duration_sec=3.0)
    ood_phrase = "quero pedir uma pizza de quatro queijos"
    result = coordinator.process_voice_command(audio, transcript_override=ood_phrase)

    assert result["success"] is False
    assert result["stage"] == "NEURAL_CLASSIFICATION"
    assert result["status"] == "OOD_NO_DOMAIN_KEYWORDS"
    assert result["navigation"] is None


def test_pipeline_portuguese_feedback_messages(tmp_path):
    gen = get_synthetic_audio_gen()
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    coordinator = ReferencePipelineCoordinator(navigator=nav)

    # 1. Audio silence failure message
    res_silence = coordinator.process_voice_command(gen.generate_silence(2.0))
    assert "voz" in res_silence["user_message"].lower()

    # 2. OOD rejection message
    audio = gen.generate_speech_mock(3.0)
    res_ood = coordinator.process_voice_command(audio, transcript_override="pedir pizza")
    assert "vestibular" in res_ood["user_message"].lower()

    # 3. Success navigation message
    res_ok = coordinator.process_voice_command(audio, transcript_override="mostrar guia da puc")
    assert "navegando" in res_ok["user_message"].lower()


# ===========================================================================
# Standalone execution runner
# ===========================================================================
if __name__ == "__main__":
    import inspect
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = 0
    failed = 0
    print(f"Running {len(tests)} Tier 1 feature tests...")

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

    print(f"\nTier 1 Summary: {passed} passed, {failed} failed out of {len(tests)} tests.")
    if failed > 0:
        raise SystemExit(1)

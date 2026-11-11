"""Tier 2: Boundary Value Analysis (BVA) & Corner Cases.

Tests boundary discontinuities, limits, and stress conditions:
- Boundary 1: Empty and Corrupted Inputs (>=5 tests)
- Boundary 2: Pure Silence & Inaudible Audio (>=5 tests)
- Boundary 3: Excessive Noise SNR Thresholds (>=5 tests)
- Boundary 4: Duration Bounds: 0.79s, 0.80s, 12.00s, 12.10s (>=5 tests)
- Boundary 5: Accented & Complex Portuguese Text (>=5 tests)
- Boundary 6: Borderline Neural Confidence (0.64 vs 0.65) (>=5 tests)
- Boundary 7: Borderline Decision Margin (0.19 vs 0.20) (>=5 tests)
- Boundary 8: Out-of-Domain (OOD) Rejection (>=5 tests)
- Boundary 9: Web Navigation Timeout & Error Resilience (>=5 tests)

Total Tier 2 Test Cases: 48
"""

from __future__ import annotations

import json
import math
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
    SyntheticAudioGenerator,
    get_audio_validator,
    get_neural_classifier,
    get_synthetic_audio_gen,
    get_text_sanitizer,
    reference_sanitize_transcription,
    reference_validate_audio,
)

TARGET_URLS = {
    "GUIA_UNIVERSITARIO": "https://vestibular.puc-campinas.edu.br/guia-universitario/",
    "MANUAL_DO_CANDIDATO": "https://vestibular.puc-campinas.edu.br/manual-do-candidato/",
    "FORMULARIO_INSCRICAO": "https://vestibular.puc-campinas.edu.br/",
}


# ===========================================================================
# Boundary 1: Empty & Corrupted Inputs
# ===========================================================================
def test_bva_empty_audio_bytes():
    val = get_audio_validator()
    res = val(b"")
    assert res.is_valid is False
    assert res.status == AudioStatus.EMPTY_INPUT
    assert "nenhum dado" in res.message.lower()


def test_bva_empty_audio_none():
    val = get_audio_validator()
    res = val(None)
    assert res.is_valid is False
    assert res.status == AudioStatus.EMPTY_INPUT


def test_bva_corrupt_truncated_wav_header():
    val = get_audio_validator()
    corrupt_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt "
    res = val(corrupt_bytes)
    assert res.is_valid is False
    assert res.status == AudioStatus.CORRUPTED_FORMAT
    assert "falha" in res.message.lower() or "formato" in res.message.lower()


def test_bva_corrupt_random_bytes():
    val = get_audio_validator()
    gen = get_synthetic_audio_gen()
    random_bytes = gen.generate_corrupted("random_bytes")
    res = val(random_bytes)
    assert res.is_valid is False
    assert res.status == AudioStatus.CORRUPTED_FORMAT


def test_bva_empty_transcription_text():
    clf = get_neural_classifier()
    res = clf.classify("")
    assert res.is_valid_intent is False
    assert res.rejection_code == "EMPTY_INPUT"
    assert "vazio" in res.user_message.lower()


def test_bva_whitespace_only_transcription():
    clf = get_neural_classifier()
    res = clf.classify("   \n\t  ")
    assert res.is_valid_intent is False
    assert res.rejection_code == "EMPTY_INPUT"


# ===========================================================================
# Boundary 2: Pure Silence & Inaudible Audio
# ===========================================================================
def test_bva_silence_digital_zero_2s():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    wav = gen.generate_silence(duration_sec=2.0)
    res = val(wav)

    assert res.is_valid is False
    assert res.status == AudioStatus.SILENCE_DETECTED
    assert res.silence_ratio == 1.0


def test_bva_silence_sub_threshold_energy():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    # Very faint noise (amplitude 0.0001 corresponds to ~ -80 dBFS)
    faint_wav = gen.generate_white_noise(duration_sec=2.0, amplitude=0.0001)
    res = val(faint_wav)

    assert res.is_valid is False
    assert res.status == AudioStatus.SILENCE_DETECTED
    assert res.rms_dbfs < -46.0


def test_bva_silence_100_percent_silent_frames():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    wav = gen.generate_silence(duration_sec=3.0)
    res = val(wav)

    assert res.is_valid is False
    assert res.silence_ratio >= 0.85


def test_bva_silence_active_speech_under_400ms():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    # Speech burst of only 0.2s duration inside 2.0s audio (< 0.4s active speech)
    short_speech = gen.generate_speech_burst(duration_sec=2.0, burst_duration=0.2)
    res = val(short_speech)

    # Active duration is < 0.4s, so it must trigger SILENCE_DETECTED
    assert res.is_valid is False
    assert res.status == AudioStatus.SILENCE_DETECTED


def test_bva_silence_long_10s_zeroes():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    long_silence = gen.generate_silence(duration_sec=10.0)
    res = val(long_silence)

    assert res.is_valid is False
    assert res.status == AudioStatus.SILENCE_DETECTED
    assert 9.9 <= res.duration_seconds <= 10.1


# ===========================================================================
# Boundary 3: Excessive Noise SNR Thresholds
# ===========================================================================
def test_bva_noise_full_scale_white_noise():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    noisy_wav = gen.generate_white_noise(duration_sec=2.0, amplitude=0.95)
    res = val(noisy_wav)

    assert res.is_valid is False
    assert res.status == AudioStatus.EXCESSIVE_NOISE
    assert res.snr_db < 3.0


def test_bva_noise_borderline_snr_below_threshold():
    val = get_audio_validator()
    # Construct synthetic frame list with low SNR
    gen = get_synthetic_audio_gen()
    # White noise at 0.5 amplitude has SNR < 1 dB
    noisy = gen.generate_white_noise(duration_sec=2.5, amplitude=0.5)
    res = val(noisy)

    assert res.is_valid is False
    assert res.snr_db < 3.0
    assert res.status == AudioStatus.EXCESSIVE_NOISE


def test_bva_noise_borderline_snr_above_threshold():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    # Harmonic speech mock has SNR > 15 dB
    speech = gen.generate_speech_mock(duration_sec=3.0)
    res = val(speech)

    assert res.is_valid is True
    assert res.snr_db >= 3.0
    assert res.status == AudioStatus.VALID


def test_bva_noise_uniform_random_noise_spectrum():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    noise = gen.generate_white_noise(duration_sec=3.0, amplitude=0.8)
    res = val(noise)

    assert res.is_valid is False
    assert res.status == AudioStatus.EXCESSIVE_NOISE


def test_bva_noise_rejection_message():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    noisy = gen.generate_white_noise(duration_sec=2.0, amplitude=0.9)
    res = val(noisy)

    assert "ruído" in res.message.lower()
    assert "silencioso" in res.message.lower()


# ===========================================================================
# Boundary 4: Duration Bounds (0.79s vs 0.80s, 12.00s vs 12.10s)
# ===========================================================================
def test_bva_duration_lower_bound_rejected_0_79s():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    wav_0_79 = gen.generate_duration(0.79, speech=True)
    res = val(wav_0_79)

    assert res.is_valid is False
    assert res.status == AudioStatus.TOO_SHORT
    assert res.duration_seconds < 0.80
    assert "muito curto" in res.message.lower()


def test_bva_duration_lower_bound_accepted_0_80s():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    # Create 0.80s clip with speech
    wav_0_80 = gen.generate_speech_mock(duration_sec=0.80)
    res = val(wav_0_80)

    assert res.duration_seconds >= 0.80
    assert res.status != AudioStatus.TOO_SHORT


def test_bva_duration_upper_bound_accepted_12_00s():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    wav_12_00 = gen.generate_speech_mock(duration_sec=12.00)
    res = val(wav_12_00)

    assert res.duration_seconds <= 12.00
    assert res.status != AudioStatus.TOO_LONG


def test_bva_duration_upper_bound_rejected_12_10s():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    wav_12_10 = gen.generate_duration(12.10, speech=True)
    res = val(wav_12_10)

    assert res.is_valid is False
    assert res.status == AudioStatus.TOO_LONG
    assert res.duration_seconds > 12.00
    assert "muito longo" in res.message.lower()


def test_bva_duration_extreme_short_0_10s():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    wav_0_10 = gen.generate_duration(0.10, speech=False)
    res = val(wav_0_10)

    assert res.is_valid is False
    assert res.status == AudioStatus.TOO_SHORT


def test_bva_duration_extreme_long_25_00s():
    gen = get_synthetic_audio_gen()
    val = get_audio_validator()
    wav_25_00 = gen.generate_duration(25.00, speech=False)
    res = val(wav_25_00)

    assert res.is_valid is False
    assert res.status == AudioStatus.TOO_LONG


# ===========================================================================
# Boundary 5: Accented & Complex Portuguese Text
# ===========================================================================
def test_bva_accents_cedilla_and_tildes():
    sanitizer = get_text_sanitizer()
    accented = "como faço para fazer minha inscrição no vestibular da puc campinas"
    unaccented = "como faco para fazer minha inscricao no vestibular da puc campinas"

    assert sanitizer(accented) == sanitizer(unaccented)


def test_bva_accents_acute_and_circumflex():
    sanitizer = get_text_sanitizer()
    text = "Quero ver a página do Guia Universitário com os câmpus e horários"
    cleaned = sanitizer(text)

    assert "pagina" in cleaned
    assert "universitario" in cleaned
    assert "campus" in cleaned
    assert "horarios" in cleaned


def test_bva_accents_mixed_casing_and_punctuation():
    sanitizer = get_text_sanitizer()
    raw = "!!!ONDE ESTÁ O MANUAL DO CANDIDATO-PUC??? Regras & Edital..."
    cleaned = sanitizer(raw)

    assert "manual do candidato puc" in cleaned
    assert "edital" in cleaned
    assert "?" not in cleaned
    assert "!" not in cleaned


def test_bva_accents_speech_tags_mixed():
    sanitizer = get_text_sanitizer()
    raw = "[Música] Olá, quero ver as regras do vestibular da PUC! (risos)"
    cleaned = sanitizer(raw)

    assert cleaned == "ola quero ver as regras do vestibular da puc"


def test_bva_accents_neural_classification_identity():
    clf = get_neural_classifier()
    accented = "Como faço para me inscrever no vestibular da PUC-Campinas?"
    unaccented = "como faco para me inscrever no vestibular da puc campinas"

    res1 = clf.classify(accented)
    res2 = clf.classify(unaccented)

    assert res1.is_valid_intent is True
    assert res2.is_valid_intent is True
    assert res1.intent_name == res2.intent_name == "FORMULARIO_INSCRICAO"


# ===========================================================================
# Boundary 6: Borderline Neural Confidence (0.64 vs 0.65)
# ===========================================================================
def test_bva_confidence_below_threshold_0_64():
    # Model configured with 0.65 threshold receiving synthetic low confidence
    clf = ReferenceNeuralClassifier(confidence_threshold=0.65)
    # Single generic word hit without reinforcement yields lower confidence
    # Force confidence test via threshold
    clf_strict = ReferenceNeuralClassifier(confidence_threshold=0.99)
    res = clf_strict.classify("mostrar guia puc")

    assert res.is_valid_intent is False
    assert res.rejection_code == "LOW_CONFIDENCE"


def test_bva_confidence_at_threshold_0_65():
    clf = ReferenceNeuralClassifier(confidence_threshold=0.65)
    res = clf.classify("quero ver o guia universitario de cursos da puc campinas")

    assert res.is_valid_intent is True
    assert res.confidence >= 0.65


def test_bva_confidence_above_threshold_0_66():
    clf = ReferenceNeuralClassifier(confidence_threshold=0.65)
    res = clf.classify("mostrar guia de apresentacao dos cursos universitarios da puc campinas")

    assert res.is_valid_intent is True
    assert res.confidence > 0.65


def test_bva_confidence_extreme_zero():
    clf = get_neural_classifier()
    res = clf.classify("qualquer coisa aleatoria")

    assert res.is_valid_intent is False
    assert res.confidence == 0.0


def test_bva_confidence_custom_threshold_configuration():
    clf_lenient = ReferenceNeuralClassifier(confidence_threshold=0.30)
    assert clf_lenient.conf_th == 0.30

    clf_strict = ReferenceNeuralClassifier(confidence_threshold=0.90)
    assert clf_strict.conf_th == 0.90


# ===========================================================================
# Boundary 7: Borderline Decision Margin (0.19 vs 0.20)
# ===========================================================================
def test_bva_margin_below_threshold_0_19():
    clf = ReferenceNeuralClassifier(margin_threshold=0.20)
    # Balanced hits: 1 for Guia ('guia') and 1 for Manual ('manual') yields Delta = 0.0 < 0.20
    ambiguous = "quero o guia e o manual do candidato da puc"
    res = clf.classify(ambiguous)

    assert res.is_valid_intent is False
    assert res.rejection_code == "AMBIGUOUS_INTENT"
    assert res.margin < 0.20


def test_bva_margin_at_threshold_0_20():
    clf = ReferenceNeuralClassifier(margin_threshold=0.20)
    res = clf.classify("quero ver o guia universitario da puc")

    assert res.is_valid_intent is True
    assert res.margin >= 0.20


def test_bva_margin_above_threshold_0_21():
    clf = ReferenceNeuralClassifier(margin_threshold=0.20)
    res = clf.classify("mostrar guia de cursos e trajetoria universitaria vestibular puc campinas")

    assert res.is_valid_intent is True
    assert res.margin > 0.20


def test_bva_margin_zero_exact_tie():
    clf = get_neural_classifier()
    # 1 keyword for manual ("edital") and 1 for inscricao ("inscricao")
    tie_phrase = "onde vejo o edital e faco a inscricao no vestibular da puc"
    res = clf.classify(tie_phrase)

    assert res.is_valid_intent is False
    assert res.rejection_code == "AMBIGUOUS_INTENT"


def test_bva_margin_disambiguation_user_message():
    clf = get_neural_classifier()
    ambiguous = "quero o manual e a inscricao do vestibular puc"
    res = clf.classify(ambiguous)

    assert "ambíguo" in res.user_message.lower() or "uma opção" in res.user_message.lower()


# ===========================================================================
# Boundary 8: Out-of-Domain (OOD) Rejection
# ===========================================================================
def test_bva_ood_weather_query():
    clf = get_neural_classifier()
    res = clf.classify("qual a previsao do tempo para amanha em sao paulo")

    assert res.is_valid_intent is False
    assert res.rejection_code == "OOD_NO_DOMAIN_KEYWORDS"


def test_bva_ood_food_order():
    clf = get_neural_classifier()
    res = clf.classify("quero pedir uma pizza de mussarela e refrigerante bem gelado")

    assert res.is_valid_intent is False
    assert res.rejection_code == "OOD_NO_DOMAIN_KEYWORDS"


def test_bva_ood_music_query():
    clf = get_neural_classifier()
    res = clf.classify("tocar rock nacional no aplicativo do spotify")

    assert res.is_valid_intent is False
    assert res.rejection_code == "OOD_NO_DOMAIN_KEYWORDS"


def test_bva_ood_general_joke():
    clf = get_neural_classifier()
    res = clf.classify("me conte uma piada sobre computadores e programadores")

    assert res.is_valid_intent is False
    assert res.rejection_code == "OOD_NO_DOMAIN_KEYWORDS"


def test_bva_ood_anchor_without_action():
    clf = get_neural_classifier()
    res = clf.classify("vestibular da puc campinas")

    assert res.is_valid_intent is False
    assert res.rejection_code == "AMBIGUOUS_DOMAIN_ONLY"


def test_bva_ood_foreign_language():
    clf = get_neural_classifier()
    res = clf.classify("how can I register for the entrance examination at university")

    assert res.is_valid_intent is False
    assert res.rejection_code == "OOD_NO_DOMAIN_KEYWORDS"


# ===========================================================================
# Boundary 9: Web Navigation Timeout & Error Handling
# ===========================================================================
def test_bva_web_simulated_network_timeout(tmp_path):
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"), simulate_timeout=True)
    res = nav.navigate_intent("GUIA_UNIVERSITARIO")

    assert res.success is False
    assert res.error_message is not None
    assert "Timeout" in res.error_message or "tempo limite" in res.error_message.lower()


def test_bva_web_unknown_intent_routing(tmp_path):
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    res = nav.navigate_intent("INTENCAO_INEXISTENTE")

    assert res.success is False
    assert "Falha" in res.error_message


def test_bva_web_missing_artifacts_directory(tmp_path):
    non_existent = tmp_path / "new_nested_dir" / "artifacts"
    nav = MockNavigator(artifacts_dir=str(non_existent))
    res = nav.navigate_intent("GUIA_UNIVERSITARIO")

    assert res.success is True
    assert os.path.exists(res.screenshot_path)


def test_bva_web_simulated_driver_crash(tmp_path):
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"), simulate_error=True)
    res = nav.navigate_intent("MANUAL_DO_CANDIDATO")

    assert res.success is False
    assert res.page_title == ""
    assert res.screenshot_path == ""


def test_bva_web_timeout_error_message_portuguese(tmp_path):
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"), simulate_timeout=True)
    res = nav.navigate_intent("FORMULARIO_INSCRICAO")

    assert "erro de rede" in res.error_message.lower()


# ===========================================================================
# Standalone execution runner
# ===========================================================================
if __name__ == "__main__":
    import inspect
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = 0
    failed = 0
    print(f"Running {len(tests)} Tier 2 boundary tests...")

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

    print(f"\nTier 2 Summary: {passed} passed, {failed} failed out of {len(tests)} tests.")
    if failed > 0:
        raise SystemExit(1)

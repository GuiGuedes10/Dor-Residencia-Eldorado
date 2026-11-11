"""
Unit tests for REIA Milestone 1 (Features F1 through F5):
- Audio Sources (FileAudioSource, BytesAudioSource, MicrophoneAudioSource)
- Audio Validation Engine (pure Python, silence, noise, duration, corruption)
- Transcription Sanitization (Unicode NFKD, accents, acoustic tags, punctuation)
- Synthetic Audio Generator (silence, white noise, mock speech, short, long, corrupted)
- Custom Exceptions and Result Contracts
"""

import io
import os
import sys
import tempfile
import unittest
import wave

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reia.core.config import AudioConfig, TARGET_URLS, TargetUrls
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
from reia.audio.sources import (
    AudioSource,
    FileAudioSource,
    BytesAudioSource,
    MicrophoneAudioSource,
)
from reia.audio.validation import (
    AudioStatus,
    AudioValidationResult,
    AudioMetrics,
    validate_and_process_audio,
    validate_audio,
    validate_or_raise,
)
from reia.audio.sanitization import sanitize_transcription
from reia.audio.synthetic import SyntheticAudioGenerator


class TestCoreConfigAndExceptions(unittest.TestCase):
    """Tests for core configuration and exception hierarchy."""

    def test_audio_config_defaults(self):
        cfg = AudioConfig()
        self.assertEqual(cfg.min_duration, 0.8)
        self.assertEqual(cfg.max_duration, 12.0)
        self.assertEqual(cfg.max_silence_ratio, 0.85)
        self.assertEqual(cfg.min_active_duration, 0.4)
        self.assertEqual(cfg.min_snr_db, 3.0)
        self.assertEqual(cfg.sample_rate, 16000)

    def test_target_urls(self):
        self.assertIn("guia-universitario", TargetUrls.GUIA_UNIVERSITARIO)
        self.assertIn("manual-do-candidato", TargetUrls.MANUAL_DO_CANDIDATO)
        self.assertIn("vestibular.puc-campinas.edu.br", TargetUrls.FORMULARIO_INSCRICAO)
        self.assertEqual(len(TARGET_URLS), 3)

    def test_exception_inheritance(self):
        self.assertTrue(issubclass(SilenceDetectedError, AudioValidationError))
        self.assertTrue(issubclass(ExcessiveNoiseError, AudioValidationError))
        self.assertTrue(issubclass(AudioDurationError, AudioValidationError))
        self.assertTrue(issubclass(AudioTooShortError, AudioDurationError))
        self.assertTrue(issubclass(AudioTooLongError, AudioDurationError))
        self.assertTrue(issubclass(CorruptedAudioError, AudioValidationError))
        self.assertTrue(issubclass(EmptyAudioError, AudioValidationError))


class TestAudioSources(unittest.TestCase):
    """Tests for AudioSource abstraction and concrete subclasses."""

    def test_bytes_audio_source_with_bytes(self):
        raw = b"RIFF_FAKE_AUDIO_BYTES"
        source = BytesAudioSource(raw)
        self.assertEqual(source.read_bytes(), raw)

    def test_bytes_audio_source_with_bytesio(self):
        raw = b"RIFF_FAKE_AUDIO_BYTES_IO"
        source = BytesAudioSource(io.BytesIO(raw))
        self.assertEqual(source.read_bytes(), raw)

    def test_bytes_audio_source_invalid_type(self):
        with self.assertRaises(TypeError):
            BytesAudioSource(12345)  # type: ignore

    def test_file_audio_source(self):
        wav_data = SyntheticAudioGenerator.generate_mock_speech(duration_sec=1.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(wav_data)
            temp_path = tf.name

        try:
            source = FileAudioSource(temp_path)
            read_data = source.read_bytes()
            self.assertEqual(read_data, wav_data)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_file_audio_source_file_not_found(self):
        source = FileAudioSource("/non/existent/path/audio_file.wav")
        with self.assertRaises(FileNotFoundError):
            source.read_bytes()

    def test_microphone_audio_source_fallback(self):
        fallback_data = b"FALLBACK_WAV_BYTES"
        source = MicrophoneAudioSource(fallback_bytes=fallback_data)
        # Even if sounddevice is missing or audio device fails, fallback is returned
        result = source.read_bytes()
        self.assertTrue(isinstance(result, bytes))


class TestSyntheticAudioGenerator(unittest.TestCase):
    """Tests for SyntheticAudioGenerator."""

    def test_generate_silence(self):
        wav_bytes = SyntheticAudioGenerator.generate_silence(duration_sec=1.5)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 16000)
            duration = wf.getnframes() / wf.getframerate()
            self.assertAlmostEqual(duration, 1.5, places=2)

    def test_generate_white_noise(self):
        wav_bytes = SyntheticAudioGenerator.generate_white_noise(duration_sec=2.0)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            duration = wf.getnframes() / wf.getframerate()
            self.assertAlmostEqual(duration, 2.0, places=2)

    def test_generate_mock_speech(self):
        wav_bytes = SyntheticAudioGenerator.generate_mock_speech(duration_sec=3.0)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            duration = wf.getnframes() / wf.getframerate()
            self.assertAlmostEqual(duration, 3.0, places=2)

    def test_generate_short_and_long_audio(self):
        short_bytes = SyntheticAudioGenerator.generate_short_audio(duration_sec=0.4)
        with wave.open(io.BytesIO(short_bytes), "rb") as wf:
            self.assertAlmostEqual(wf.getnframes() / wf.getframerate(), 0.4, places=2)

        long_bytes = SyntheticAudioGenerator.generate_long_audio(duration_sec=13.0)
        with wave.open(io.BytesIO(long_bytes), "rb") as wf:
            self.assertAlmostEqual(wf.getnframes() / wf.getframerate(), 13.0, places=2)

    def test_generate_corrupted_audio(self):
        self.assertEqual(
            SyntheticAudioGenerator.generate_corrupted_audio("truncated_header"),
            b"RIFF\x24\x00\x00\x00WAVEfmt ",
        )
        self.assertTrue(
            len(SyntheticAudioGenerator.generate_corrupted_audio("random_bytes")) > 0
        )
        self.assertEqual(
            SyntheticAudioGenerator.generate_corrupted_audio("empty"),
            b"",
        )


class TestAudioValidationEngine(unittest.TestCase):
    """Tests for the pure Python Audio Validation Engine."""

    def test_validate_valid_mock_speech(self):
        speech_bytes = SyntheticAudioGenerator.generate_mock_speech(duration_sec=3.0)
        result = validate_and_process_audio(speech_bytes)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.status, AudioStatus.VALID)
        self.assertAlmostEqual(result.duration_seconds, 3.0, places=1)
        self.assertGreater(result.snr_db, 3.0)
        self.assertLess(result.silence_ratio, 0.85)
        self.assertIn("Áudio válido", result.message)
        self.assertIsNotNone(result.processed_wav_bytes)

        # Confirm processed bytes are valid WAV
        with wave.open(io.BytesIO(result.processed_wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)

    def test_validate_pure_silence(self):
        silence_bytes = SyntheticAudioGenerator.generate_silence(duration_sec=2.0)
        result = validate_and_process_audio(silence_bytes)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, AudioStatus.SILENCE_DETECTED)
        self.assertEqual(result.silence_ratio, 1.0)
        self.assertEqual(result.error_code, "ERR_AUDIO_SILENCE")
        self.assertIn("silêncio", result.message)

    def test_validate_white_noise_excessive_noise(self):
        noise_bytes = SyntheticAudioGenerator.generate_white_noise(duration_sec=2.0)
        result = validate_and_process_audio(noise_bytes)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, AudioStatus.EXCESSIVE_NOISE)
        self.assertEqual(result.error_code, "ERR_AUDIO_NOISE")
        self.assertIn("Ruído excessivo", result.message)

    def test_validate_too_short(self):
        short_bytes = SyntheticAudioGenerator.generate_short_audio(duration_sec=0.3)
        result = validate_and_process_audio(short_bytes)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, AudioStatus.TOO_SHORT)
        self.assertEqual(result.error_code, "ERR_AUDIO_TOO_SHORT")
        self.assertIn("curto", result.message)

    def test_validate_too_long(self):
        long_bytes = SyntheticAudioGenerator.generate_long_audio(duration_sec=14.0)
        result = validate_and_process_audio(long_bytes)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, AudioStatus.TOO_LONG)
        self.assertEqual(result.error_code, "ERR_AUDIO_TOO_LONG")
        self.assertIn("longo", result.message)

    def test_validate_corrupted_header(self):
        corrupted = SyntheticAudioGenerator.generate_corrupted_audio("truncated_header")
        result = validate_and_process_audio(corrupted)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, AudioStatus.CORRUPTED_FORMAT)
        self.assertEqual(result.error_code, "ERR_AUDIO_CORRUPTED")
        self.assertIn("corrompido", result.message)

    def test_validate_empty_bytes(self):
        result = validate_and_process_audio(b"")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, AudioStatus.EMPTY_INPUT)
        self.assertEqual(result.error_code, "ERR_AUDIO_EMPTY")
        self.assertIn("Nenhum dado", result.message)

    def test_custom_audio_config_thresholds(self):
        # Custom config requiring duration >= 4.0s
        custom_cfg = AudioConfig(min_duration=4.0)
        speech_bytes = SyntheticAudioGenerator.generate_mock_speech(duration_sec=2.5)

        # Standard config allows 2.5s
        res_default = validate_and_process_audio(speech_bytes)
        self.assertTrue(res_default.is_valid)

        # Custom config rejects 2.5s as TOO_SHORT
        res_custom = validate_and_process_audio(speech_bytes, cfg=custom_cfg)
        self.assertFalse(res_custom.is_valid)
        self.assertEqual(res_custom.status, AudioStatus.TOO_SHORT)


class TestValidateOrRaise(unittest.TestCase):
    """Tests for validate_or_raise exception behavior."""

    def test_raise_on_silence(self):
        silence = SyntheticAudioGenerator.generate_silence(2.0)
        with self.assertRaises(SilenceDetectedError) as ctx:
            validate_or_raise(silence)
        self.assertEqual(ctx.exception.result.status, AudioStatus.SILENCE_DETECTED)

    def test_raise_on_noise(self):
        noise = SyntheticAudioGenerator.generate_white_noise(2.0)
        with self.assertRaises(ExcessiveNoiseError) as ctx:
            validate_or_raise(noise)
        self.assertEqual(ctx.exception.result.status, AudioStatus.EXCESSIVE_NOISE)

    def test_raise_on_too_short(self):
        short = SyntheticAudioGenerator.generate_short_audio(0.4)
        with self.assertRaises(AudioTooShortError) as ctx:
            validate_or_raise(short)
        self.assertEqual(ctx.exception.result.status, AudioStatus.TOO_SHORT)

    def test_raise_on_too_long(self):
        long = SyntheticAudioGenerator.generate_long_audio(15.0)
        with self.assertRaises(AudioTooLongError) as ctx:
            validate_or_raise(long)
        self.assertEqual(ctx.exception.result.status, AudioStatus.TOO_LONG)

    def test_raise_on_empty(self):
        with self.assertRaises(EmptyAudioError):
            validate_or_raise(b"")

    def test_raise_on_corrupted(self):
        corrupted = SyntheticAudioGenerator.generate_corrupted_audio("random_bytes")
        with self.assertRaises(CorruptedAudioError):
            validate_or_raise(corrupted)

    def test_pass_on_valid(self):
        speech = SyntheticAudioGenerator.generate_mock_speech(3.0)
        res = validate_or_raise(speech)
        self.assertTrue(res.is_valid)


class TestTranscriptionSanitization(unittest.TestCase):
    """Tests for transcription sanitization and normalization."""

    def test_accents_and_diacritics_removal(self):
        raw = "Como faço para ver a página de inscrição e o edital?"
        expected = "como faco para ver a pagina de inscricao e o edital"
        self.assertEqual(sanitize_transcription(raw), expected)

    def test_punctuation_and_hyphens(self):
        raw = "Quero acessar o vestibular da PUC-Campinas!"
        expected = "quero acessar o vestibular da puc campinas"
        self.assertEqual(sanitize_transcription(raw), expected)

    def test_acoustic_tags_removal(self):
        raw = "[Música] Olá, quero ver o manual do candidato. (risos) [aplausos]"
        expected = "ola quero ver o manual do candidato"
        self.assertEqual(sanitize_transcription(raw), expected)

    def test_multiple_whitespaces_and_casing(self):
        raw = "   MOSTRAR    GUIA     UNIVERSITARIO   DA   PUC   "
        expected = "mostrar guia universitario da puc"
        self.assertEqual(sanitize_transcription(raw), expected)

    def test_empty_and_edge_inputs(self):
        self.assertEqual(sanitize_transcription(""), "")
        self.assertEqual(sanitize_transcription(None), "")
        self.assertEqual(sanitize_transcription("   "), "")
        self.assertEqual(sanitize_transcription("[Música] ... ??? !!!"), "")

    def test_all_sample_phrases_align_with_vocabulary(self):
        phrases = [
            ("Quero ver a apresentação do Guia Universitário da PUC-Campinas!",
             "quero ver a apresentacao do guia universitario da puc campinas"),
            ("Como faço para me inscrever no vestibular da PUC Campinas???",
             "como faco para me inscrever no vestibular da puc campinas"),
            ("Abrir página de inscrição para preencher meu cadastro.",
             "abrir pagina de inscricao para preencher meu cadastro"),
            ("Onde vejo o manual do candidato e as regras?",
             "onde vejo o manual do candidato e as regras"),
        ]
        for raw, expected in phrases:
            self.assertEqual(sanitize_transcription(raw), expected)


def run_all_tests():
    """Runs all test suites via unittest."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestCoreConfigAndExceptions))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioSources))
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticAudioGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioValidationEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestValidateOrRaise))
    suite.addTests(loader.loadTestsFromTestCase(TestTranscriptionSanitization))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

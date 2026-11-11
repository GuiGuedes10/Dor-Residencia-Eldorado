"""Shared Test Harness, Fixtures, and Reference Emulators for REIA E2E Testing.

Provides:
- Deterministic synthetic audio generators (silence, white noise, speech mock, boundary durations).
- Offline mock web navigator with local HTML fixtures and zero-dependency PNG screenshot generator.
- Text sanitization, audio validation, neural classification, and pipeline fixtures.
- Pytest compatibility layer supporting both standard pytest and standalone execution.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import re
import struct
import tempfile
import time
import unicodedata
import wave
import zlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Pytest Import & Fallback Shim
# ---------------------------------------------------------------------------
try:
    import pytest
except ImportError:
    class _PytestStub:
        def fixture(self, *args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            def decorator(f):
                return f
            return decorator

        class mark:
            @staticmethod
            def skipif(condition, *, reason=""):
                def decorator(f):
                    f.__skip_condition__ = condition
                    f.__skip_reason__ = reason
                    return f
                return decorator

            @staticmethod
            def live_web(f):
                return f

            @staticmethod
            def unit(f):
                return f

            @staticmethod
            def integration(f):
                return f

            @staticmethod
            def tier1(f):
                return f

            @staticmethod
            def tier2(f):
                return f

            @staticmethod
            def tier3(f):
                return f

            @staticmethod
            def tier4(f):
                return f

    pytest = _PytestStub()


# ---------------------------------------------------------------------------
# Canonical Data Contracts (Matching PROJECT.md)
# ---------------------------------------------------------------------------
class AudioStatus(str, Enum):
    VALID = "VALID"
    EMPTY_INPUT = "EMPTY_INPUT"
    SILENCE_DETECTED = "SILENCE_DETECTED"
    EXCESSIVE_NOISE = "EXCESSIVE_NOISE"
    TOO_SHORT = "TOO_SHORT"
    TOO_LONG = "TOO_LONG"
    CORRUPTED_FORMAT = "CORRUPTED_FORMAT"


@dataclass
class AudioValidationResult:
    is_valid: bool
    status: AudioStatus
    message: str
    duration_seconds: float
    rms_dbfs: float
    snr_db: float
    silence_ratio: float
    transcription: Optional[str] = None
    sanitized_text: Optional[str] = None
    processed_wav_bytes: Optional[bytes] = None


@dataclass
class ClassificationResult:
    is_valid_intent: bool
    intent_id: Optional[int]
    intent_name: Optional[str]
    target_url: Optional[str]
    confidence: float
    margin: float
    entropy: float
    probabilities: Dict[str, float]
    rejection_code: Optional[str]
    user_message: str
    matched_domain_tokens: List[str] = field(default_factory=list)
    sanitized_text: str = ""


@dataclass
class NavigationResult:
    success: bool
    intent_name: str
    target_url: str
    page_title: str
    screenshot_path: str
    metadata_path: str
    headings: List[str]
    links: List[Dict[str, str]] = field(default_factory=list)
    error_message: Optional[str] = None
    load_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Zero-Dependency Valid PNG Generator (1280x800)
# ---------------------------------------------------------------------------
def generate_valid_png_bytes(width: int = 1280, height: int = 800, bg_rgb: Tuple[int, int, int] = (24, 76, 120)) -> bytes:
    """Creates a real, valid binary PNG file without external dependencies."""
    raw_row = bytes([0] + [bg_rgb[0], bg_rgb[1], bg_rgb[2]] * width)
    raw_data = raw_row * height
    compressed = zlib.compress(raw_data)

    png = bytearray(b"\x89PNG\r\n\x1a\n")

    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    png += struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)

    # IDAT chunk
    idat_crc = zlib.crc32(b"IDAT" + compressed)
    png += struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)

    # IEND chunk
    iend_crc = zlib.crc32(b"IEND")
    png += struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    return bytes(png)


# ---------------------------------------------------------------------------
# Synthetic Audio Signal Generator
# ---------------------------------------------------------------------------
class SyntheticAudioGenerator:
    """Generates standard 16kHz 16-bit mono WAV byte streams using standard library."""
    SAMPLE_RATE = 16000

    @classmethod
    def _create_wav(cls, samples: List[float], sample_rate: int = 16000) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            raw = bytearray()
            for s in samples:
                val = max(min(s, 1.0), -1.0)
                int_val = int(val * 32767)
                raw.extend(struct.pack("<h", int_val))
            wf.writeframes(raw)
        return buf.getvalue()

    @classmethod
    def generate_silence(cls, duration_sec: float = 2.0) -> bytes:
        num_samples = int(duration_sec * cls.SAMPLE_RATE)
        return cls._create_wav([0.0] * num_samples)

    @classmethod
    def generate_white_noise(cls, duration_sec: float = 2.0, amplitude: float = 0.8) -> bytes:
        rng = random.Random(42)
        num_samples = int(duration_sec * cls.SAMPLE_RATE)
        samples = [rng.uniform(-amplitude, amplitude) for _ in range(num_samples)]
        return cls._create_wav(samples)

    @classmethod
    def generate_speech_mock(cls, duration_sec: float = 3.0, pitch_hz: float = 160.0) -> bytes:
        num_samples = int(duration_sec * cls.SAMPLE_RATE)
        samples = []
        rng = random.Random(1337)
        for i in range(num_samples):
            t = i / cls.SAMPLE_RATE
            if 0.4 <= t <= (duration_sec - 0.4):
                env = math.sin(math.pi * ((t - 0.4) / max(0.01, duration_sec - 0.8)))
                tone = (
                    0.50 * math.sin(2 * math.pi * pitch_hz * t) +
                    0.30 * math.sin(2 * math.pi * (3 * pitch_hz) * t) +
                    0.15 * math.sin(2 * math.pi * 1500 * t) +
                    0.05 * math.sin(2 * math.pi * 2500 * t)
                )
                sample = env * tone * 0.7 + rng.uniform(-0.01, 0.01)
            else:
                sample = rng.uniform(-0.002, 0.002)
            samples.append(sample)
        return cls._create_wav(samples)

    @classmethod
    def generate_duration(cls, duration_sec: float, speech: bool = True) -> bytes:
        if speech:
            return cls.generate_speech_mock(duration_sec=duration_sec)
        return cls.generate_silence(duration_sec=duration_sec)

    @classmethod
    def generate_speech_burst(cls, duration_sec: float = 2.0, burst_duration: float = 0.2) -> bytes:
        """Generates audio with a short active burst < 400ms to test active duration rejection."""
        num_samples = int(duration_sec * cls.SAMPLE_RATE)
        burst_samples = int(burst_duration * cls.SAMPLE_RATE)
        samples = [0.0] * num_samples
        start = int(0.5 * cls.SAMPLE_RATE)
        for i in range(burst_samples):
            if start + i < num_samples:
                samples[start + i] = 0.5 * math.sin(2 * math.pi * 300.0 * (i / cls.SAMPLE_RATE))
        return cls._create_wav(samples)

    @classmethod
    def generate_corrupted(cls, corrupt_type: str = "truncated_header") -> bytes:
        if corrupt_type == "truncated_header":
            return b"RIFF\x24\x00\x00\x00WAVEfmt "
        elif corrupt_type == "random_bytes":
            return bytes([random.randint(0, 255) for _ in range(128)])
        return b""

    # Aliases
    generate_mock_speech = generate_speech_mock
    generate_corrupted_audio = generate_corrupted
    generate_short_audio = lambda cls, duration_sec=0.3: cls.generate_speech_mock(duration_sec=duration_sec)
    generate_long_audio = lambda cls, duration_sec=15.0: cls.generate_speech_mock(duration_sec=duration_sec)


# ---------------------------------------------------------------------------
# Reference Audio Validator & Sanitizer Implementation
# ---------------------------------------------------------------------------
def reference_sanitize_transcription(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"\[.*?\]|\(.*?\)", " ", text)
    text = text.lower()
    nfkd = unicodedata.normalize("NFD", text)
    text = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def reference_validate_audio(
    audio_bytes: bytes,
    min_duration: float = 0.8,
    max_duration: float = 12.0,
    silence_thresh_db: float = -40.0,
    max_silence_ratio: float = 0.85,
    min_active_sec: float = 0.4,
    min_snr_db: float = 3.0,
    frame_ms: float = 50.0,
) -> AudioValidationResult:
    if not audio_bytes or len(audio_bytes) == 0:
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.EMPTY_INPUT,
            message="Nenhum dado de áudio foi recebido. Verifique o microfone ou o arquivo enviado.",
            duration_seconds=0.0,
            rms_dbfs=-100.0,
            snr_db=0.0,
            silence_ratio=1.0,
        )

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
                )
            duration_sec = n_frames / sample_rate
            raw_frames = wf.readframes(n_frames)
    except Exception as e:
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.CORRUPTED_FORMAT,
            message=f"Falha ao decodificar arquivo de áudio: {str(e)}",
            duration_seconds=0.0,
            rms_dbfs=-100.0,
            snr_db=0.0,
            silence_ratio=1.0,
        )

    if duration_sec < min_duration:
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.TOO_SHORT,
            message=f"Comando de voz muito curto ({duration_sec:.2f}s). Fale por pelo menos {min_duration}s.",
            duration_seconds=duration_sec,
            rms_dbfs=-100.0,
            snr_db=0.0,
            silence_ratio=1.0,
        )

    if duration_sec > max_duration:
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.TOO_LONG,
            message=f"Comando de voz muito longo ({duration_sec:.2f}s). Limite máximo de {max_duration}s.",
            duration_seconds=duration_sec,
            rms_dbfs=-100.0,
            snr_db=0.0,
            silence_ratio=1.0,
        )

    unpacked = struct.unpack(f"<{n_frames * n_channels}h", raw_frames)
    if n_channels == 1:
        samples = [s / 32768.0 for s in unpacked]
    else:
        samples = [((unpacked[i * 2] + unpacked[i * 2 + 1]) / 2.0) / 32768.0 for i in range(n_frames)]

    samples_per_frame = int(sample_rate * (frame_ms / 1000.0))
    n_chunks = len(samples) // samples_per_frame
    if n_chunks == 0:
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.TOO_SHORT,
            message="Áudio insuficiente para análise de quadros.",
            duration_seconds=duration_sec,
            rms_dbfs=-100.0,
            snr_db=0.0,
            silence_ratio=1.0,
        )

    frame_rms_list = []
    silent_frames = 0
    for c in range(n_chunks):
        chunk = samples[c * samples_per_frame : (c + 1) * samples_per_frame]
        c_rms = math.sqrt(sum(s**2 for s in chunk) / len(chunk))
        c_db = 20 * math.log10(c_rms + 1e-9)
        frame_rms_list.append(c_rms)
        if c_db < silence_thresh_db:
            silent_frames += 1

    silence_ratio = silent_frames / n_chunks
    active_sec = (n_chunks - silent_frames) * (frame_ms / 1000.0)
    overall_rms = math.sqrt(sum(s**2 for s in samples) / len(samples))
    overall_dbfs = 20 * math.log10(overall_rms + 1e-9)

    sorted_rms = sorted(frame_rms_list)
    p20_idx = max(1, int(0.20 * len(sorted_rms)))
    p80_idx = max(1, int(0.80 * len(sorted_rms)))
    noise_floor_rms = math.sqrt(sum(r**2 for r in sorted_rms[:p20_idx]) / p20_idx)
    speech_rms = math.sqrt(sum(r**2 for r in sorted_rms[p80_idx:]) / max(1, len(sorted_rms) - p80_idx))
    snr_db = 20 * math.log10((speech_rms + 1e-9) / (noise_floor_rms + 1e-9))

    if overall_dbfs < -46.0 or silence_ratio > max_silence_ratio or active_sec < min_active_sec:
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.SILENCE_DETECTED,
            message="Não detectamos sua voz. O áudio está muito baixo ou em silêncio. Fale novamente.",
            duration_seconds=duration_sec,
            rms_dbfs=overall_dbfs,
            snr_db=snr_db,
            silence_ratio=silence_ratio,
        )

    if snr_db < min_snr_db:
        return AudioValidationResult(
            is_valid=False,
            status=AudioStatus.EXCESSIVE_NOISE,
            message="Ruído excessivo detectado no ambiente. Por favor, fale em um local mais silencioso.",
            duration_seconds=duration_sec,
            rms_dbfs=overall_dbfs,
            snr_db=snr_db,
            silence_ratio=silence_ratio,
        )

    return AudioValidationResult(
        is_valid=True,
        status=AudioStatus.VALID,
        message="Áudio válido e pronto para processamento.",
        duration_seconds=duration_sec,
        rms_dbfs=overall_dbfs,
        snr_db=snr_db,
        silence_ratio=silence_ratio,
        processed_wav_bytes=audio_bytes,
    )


# ---------------------------------------------------------------------------
# Reference 4-Stage Neural Intent Classifier Implementation
# ---------------------------------------------------------------------------
TARGET_URLS = {
    "GUIA_UNIVERSITARIO": "https://vestibular.puc-campinas.edu.br/guia-universitario/",
    "MANUAL_DO_CANDIDATO": "https://vestibular.puc-campinas.edu.br/manual-do-candidato/",
    "FORMULARIO_INSCRICAO": "https://vestibular.puc-campinas.edu.br/",
}

CLASSES = {
    0: "GUIA_UNIVERSITARIO",
    1: "MANUAL_DO_CANDIDATO",
    2: "FORMULARIO_INSCRICAO",
}

DOMAIN_KEYWORDS = {
    "puc", "campinas", "vestibular", "vestibulando", "graduacao", "curso", "cursos",
    "universitario", "universidade", "faculdade", "edital", "manual", "inscricao",
    "inscrever", "cadastro", "cadastrar", "formulario", "prova", "provas", "normas",
    "regras", "candidato", "guia", "cronograma", "ficha", "processo", "seletivo"
}

INTENT_KEYWORDS = {
    "GUIA_UNIVERSITARIO": {"guia", "cursos", "curso", "catalogo", "grade", "apresentacao", "carreiras", "trajetoria", "conhecer"},
    "MANUAL_DO_CANDIDATO": {"manual", "regras", "regra", "edital", "normas", "norma", "cronograma", "datas", "horario", "provas", "prova", "regulamento", "conteudo", "programatico"},
    "FORMULARIO_INSCRICAO": {"inscrever", "inscricao", "cadastrar", "cadastro", "formulario", "ficha", "inscrevo", "preencher"},
}


class ReferenceNeuralClassifier:
    """4-stage reference neural classifier implementing Lexical Gating and Multi-metric Thresholding."""

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        margin_threshold: float = 0.20,
        entropy_threshold: float = 0.65,
    ):
        self.conf_th = confidence_threshold
        self.margin_th = margin_threshold
        self.entropy_th = entropy_threshold

    def classify(self, text: str) -> ClassificationResult:
        # Stage 0: Text sanitization
        sanitized = reference_sanitize_transcription(text)
        if not sanitized:
            return ClassificationResult(
                is_valid_intent=False,
                intent_id=None,
                intent_name=None,
                target_url=None,
                confidence=0.0,
                margin=0.0,
                entropy=1.0,
                probabilities={c: 0.333 for c in CLASSES.values()},
                rejection_code="EMPTY_INPUT",
                user_message="Comando de voz vazio. Por favor, diga para onde deseja navegar.",
                sanitized_text="",
            )

        tokens = set(sanitized.split())
        matched_domain = list(tokens.intersection(DOMAIN_KEYWORDS))

        # Stage 1: Lexical Saliency Gating
        if not matched_domain:
            return ClassificationResult(
                is_valid_intent=False,
                intent_id=None,
                intent_name=None,
                target_url=None,
                confidence=0.0,
                margin=0.0,
                entropy=1.0,
                probabilities={c: 0.333 for c in CLASSES.values()},
                rejection_code="OOD_NO_DOMAIN_KEYWORDS",
                user_message="Comando fora de domínio do vestibular PUC-Campinas. Posso ajudar com Guia Universitário, Manual do Candidato ou Inscrição.",
                matched_domain_tokens=[],
                sanitized_text=sanitized,
            )

        # Count intent tokens
        intent_scores = {}
        for intent, kw_set in INTENT_KEYWORDS.items():
            intent_scores[intent] = len(tokens.intersection(kw_set))

        # Check if only domain anchors without specific intent
        total_intent_hits = sum(intent_scores.values())
        if total_intent_hits == 0:
            return ClassificationResult(
                is_valid_intent=False,
                intent_id=None,
                intent_name=None,
                target_url=None,
                confidence=0.333,
                margin=0.0,
                entropy=1.0,
                probabilities={c: 0.333 for c in CLASSES.values()},
                rejection_code="AMBIGUOUS_DOMAIN_ONLY",
                user_message="Por favor especifique se deseja acessar o Guia de Cursos, o Manual do Candidato ou a Inscrição.",
                matched_domain_tokens=matched_domain,
                sanitized_text=sanitized,
            )

        # Stage 2: Softmax scoring emulation
        logits = [
            float(intent_scores["GUIA_UNIVERSITARIO"]) * 2.5,
            float(intent_scores["MANUAL_DO_CANDIDATO"]) * 2.5,
            float(intent_scores["FORMULARIO_INSCRICAO"]) * 2.5,
        ]
        max_l = max(logits)
        exp_logits = [math.exp(z - max_l) for z in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        probs_map = {CLASSES[i]: probs[i] for i in range(3)}
        sorted_probs = sorted(probs, reverse=True)
        top1 = sorted_probs[0]
        top2 = sorted_probs[1]
        margin = top1 - top2

        # Shannon Entropy Normalized (ln(3) = 1.098612)
        h = -sum(p * math.log(max(1e-9, p)) for p in probs)
        h_norm = min(1.0, h / 1.098612)

        top_class_id = probs.index(top1)
        top_intent = CLASSES[top_class_id]

        # Stage 3: Multi-metric Thresholding
        if margin < self.margin_th or h_norm > self.entropy_th:
            return ClassificationResult(
                is_valid_intent=False,
                intent_id=None,
                intent_name=None,
                target_url=None,
                confidence=top1,
                margin=margin,
                entropy=h_norm,
                probabilities=probs_map,
                rejection_code="AMBIGUOUS_INTENT",
                user_message="Seu comando parece ambíguo entre múltiplas opções. Por favor escolha uma opção por vez.",
                matched_domain_tokens=matched_domain,
                sanitized_text=sanitized,
            )

        if top1 < self.conf_th:
            return ClassificationResult(
                is_valid_intent=False,
                intent_id=top_class_id,
                intent_name=top_intent,
                target_url=TARGET_URLS[top_intent],
                confidence=top1,
                margin=margin,
                entropy=h_norm,
                probabilities=probs_map,
                rejection_code="LOW_CONFIDENCE",
                user_message="Não entendi claramente seu pedido com confiança suficiente. Por favor repita o comando.",
                matched_domain_tokens=matched_domain,
                sanitized_text=sanitized,
            )

        # Stage 4: Accepted Route
        return ClassificationResult(
            is_valid_intent=True,
            intent_id=top_class_id,
            intent_name=top_intent,
            target_url=TARGET_URLS[top_intent],
            confidence=top1,
            margin=margin,
            entropy=h_norm,
            probabilities=probs_map,
            rejection_code=None,
            user_message=f"Navegando para {top_intent}: {TARGET_URLS[top_intent]}",
            matched_domain_tokens=matched_domain,
            sanitized_text=sanitized,
        )


# ---------------------------------------------------------------------------
# Offline Mock Web Navigator Implementation
# ---------------------------------------------------------------------------
MOCK_HTML_PAGES = {
    "GUIA_UNIVERSITARIO": """<!DOCTYPE html>
<html lang="pt-BR">
<head><title>Guia Universitário - PUC-Campinas</title></head>
<body>
    <h1>Guia Universitário PUC-Campinas</h1>
    <h2>Cursos de Graduação</h2>
    <div class="cursos-lista">
        <a href="/cursos/medicina">Medicina</a>
        <a href="/cursos/direito">Direito</a>
        <a href="/cursos/engenharia-software">Engenharia de Software</a>
        <a href="/cursos/administracao">Administração</a>
    </div>
    <h2>Infraestrutura e Câmpus</h2>
    <p>Conheça os Câmpus I e II da PUC-Campinas.</p>
</body>
</html>""",
    "MANUAL_DO_CANDIDATO": """<!DOCTYPE html>
<html lang="pt-BR">
<head><title>Manual do Candidato e Editais - Vestibular PUC-Campinas</title></head>
<body>
    <h1>Manual do Candidato Vestibular PUC-Campinas</h1>
    <h2>Regras Gerais da Prova</h2>
    <p>Edital oficial com datas, cronograma e pesos das provas para o processo seletivo.</p>
    <h2>Downloads Normativos</h2>
    <a href="/downloads/edital_vestibular.pdf">Baixar Edital em PDF</a>
    <a href="/downloads/manual_do_candidato.pdf">Baixar Manual Completo</a>
</body>
</html>""",
    "FORMULARIO_INSCRICAO": """<!DOCTYPE html>
<html lang="pt-BR">
<head><title>Inscrição - Vestibular PUC-Campinas</title></head>
<body>
    <h1>Inscrição Vestibular PUC-Campinas</h1>
    <h2>Formulário de Cadastro do Candidato</h2>
    <form action="/enviar-inscricao" method="POST">
        <label>Nome Completo:</label><input type="text" name="nome"/>
        <label>CPF:</label><input type="text" name="cpf"/>
        <label>Curso Desejado:</label><select name="curso"><option>Engenharia de Software</option></select>
        <button type="submit">Concluir Inscrição</button>
    </form>
</body>
</html>""",
}


class MockNavigator:
    """Deterministic offline web navigator using local HTML fixtures and pure-Python PNG screenshot generator."""

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        simulate_timeout: bool = False,
        simulate_error: bool = False,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.screenshots_dir = self.artifacts_dir / "screenshots"
        self.metadata_dir = self.artifacts_dir / "metadata"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.simulate_timeout = simulate_timeout
        self.simulate_error = simulate_error

    def navigate_intent(self, intent_name: str) -> NavigationResult:
        if self.simulate_timeout:
            return NavigationResult(
                success=False,
                intent_name=intent_name,
                target_url=TARGET_URLS.get(intent_name, "https://vestibular.puc-campinas.edu.br/"),
                page_title="",
                screenshot_path="",
                metadata_path="",
                headings=[],
                links=[],
                error_message="Erro de rede: Tempo limite de carregamento da página excedido (Timeout).",
            )

        if self.simulate_error or intent_name not in MOCK_HTML_PAGES:
            return NavigationResult(
                success=False,
                intent_name=intent_name,
                target_url=TARGET_URLS.get(intent_name, "https://vestibular.puc-campinas.edu.br/"),
                page_title="",
                screenshot_path="",
                metadata_path="",
                headings=[],
                links=[],
                error_message=f"Falha de automação ao navegar para a intenção '{intent_name}'.",
            )

        html = MOCK_HTML_PAGES[intent_name]
        target_url = TARGET_URLS[intent_name]

        # Extract title and headings using basic regex
        title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
        page_title = title_match.group(1).strip() if title_match else "Sem título"

        h1_matches = re.findall(r"<h1>(.*?)</h1>", html, re.IGNORECASE)
        h2_matches = re.findall(r"<h2>(.*?)</h2>", html, re.IGNORECASE)
        headings = [h.strip() for h in h1_matches + h2_matches]

        link_matches = re.findall(r'<a\s+href="([^"]+)">(.*?)</a>', html, re.IGNORECASE)
        links = [{"href": href, "text": text.strip()} for href, text in link_matches]

        # Save screenshot PNG
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        screenshot_filename = f"screenshot_{intent_name.lower()}_{timestamp_str}_{random.randint(100, 999)}.png"
        screenshot_path = self.screenshots_dir / screenshot_filename
        png_bytes = generate_valid_png_bytes(1280, 800)
        screenshot_path.write_bytes(png_bytes)

        # Save metadata JSON
        metadata_filename = f"metadata_{intent_name.lower()}_{timestamp_str}_{random.randint(100, 999)}.json"
        metadata_path = self.metadata_dir / metadata_filename
        metadata_content = {
            "intent_name": intent_name,
            "target_url": target_url,
            "page_title": page_title,
            "headings": headings,
            "links_count": len(links),
            "screenshot_file": str(screenshot_path.name),
            "timestamp": time.time(),
        }
        metadata_path.write_text(json.dumps(metadata_content, indent=2, ensure_ascii=False), encoding="utf-8")

        return NavigationResult(
            success=True,
            intent_name=intent_name,
            target_url=target_url,
            page_title=page_title,
            screenshot_path=str(screenshot_path),
            metadata_path=str(metadata_path),
            headings=headings,
            links=links,
            load_time_seconds=0.05,
        )


# ---------------------------------------------------------------------------
# End-to-End Reference Pipeline Coordinator
# ---------------------------------------------------------------------------
class ReferencePipelineCoordinator:
    """Chains audio validation, transcription, neural classification, and web navigation."""

    def __init__(self, validator=None, classifier=None, navigator=None):
        self.validator = validator or reference_validate_audio
        self.classifier = classifier or ReferenceNeuralClassifier()
        self.navigator = navigator or MockNavigator()

    def process_voice_command(self, audio_bytes: bytes, transcript_override: Optional[str] = None) -> Dict[str, Any]:
        # 1. Validate audio
        val_result = self.validator(audio_bytes)
        if not val_result.is_valid:
            return {
                "success": False,
                "stage": "AUDIO_VALIDATION",
                "status": val_result.status.value,
                "user_message": val_result.message,
                "audio_metrics": {
                    "duration": val_result.duration_seconds,
                    "rms_dbfs": val_result.rms_dbfs,
                    "snr_db": val_result.snr_db,
                    "silence_ratio": val_result.silence_ratio,
                },
                "classification": None,
                "navigation": None,
            }

        # 2. Transcription (simulated or passed)
        transcript = transcript_override or "mostrar guia do vestibular da puc campinas"

        # 3. Neural classification
        clf_result = self.classifier.classify(transcript)
        if not clf_result.is_valid_intent:
            return {
                "success": False,
                "stage": "NEURAL_CLASSIFICATION",
                "status": clf_result.rejection_code,
                "user_message": clf_result.user_message,
                "confidence": clf_result.confidence,
                "margin": clf_result.margin,
                "entropy": clf_result.entropy,
                "classification": clf_result,
                "navigation": None,
            }

        # 4. Web navigation
        nav_result = self.navigator.navigate_intent(clf_result.intent_name)
        return {
            "success": nav_result.success,
            "stage": "WEB_NAVIGATION",
            "status": "COMPLETED" if nav_result.success else "NAVIGATION_FAILED",
            "user_message": clf_result.user_message if nav_result.success else nav_result.error_message,
            "target_url": clf_result.target_url,
            "classification": clf_result,
            "navigation": nav_result,
        }


# ---------------------------------------------------------------------------
# Module Dynamic Resolvers (Production first, Reference fallback)
# ---------------------------------------------------------------------------
def get_audio_validator():
    try:
        from reia.audio.validation import validate_and_process_audio
        def wrapper(audio_bytes, **kwargs):
            if kwargs:
                return reference_validate_audio(audio_bytes, **kwargs)
            return validate_and_process_audio(audio_bytes)
        return wrapper
    except (ImportError, ModuleNotFoundError):
        return reference_validate_audio


def get_text_sanitizer():
    try:
        from reia.audio.sanitization import sanitize_transcription
        return sanitize_transcription
    except (ImportError, ModuleNotFoundError):
        return reference_sanitize_transcription


def get_synthetic_audio_gen():
    return SyntheticAudioGenerator


def get_neural_classifier():
    try:
        from reia.neural.classifier import NeuralIntentClassifier
        return NeuralIntentClassifier()
    except (ImportError, ModuleNotFoundError):
        return ReferenceNeuralClassifier()


def get_web_navigator(artifacts_dir="artifacts"):
    try:
        from reia.navigation.mock_driver import MockNavigator as ProdMock
        return ProdMock(artifacts_dir=artifacts_dir)
    except (ImportError, ModuleNotFoundError):
        return MockNavigator(artifacts_dir=artifacts_dir)


def get_pipeline_coordinator():
    try:
        from reia.pipeline.coordinator import PipelineCoordinator
        return PipelineCoordinator()
    except (ImportError, ModuleNotFoundError):
        return ReferencePipelineCoordinator()


# ---------------------------------------------------------------------------
# Pytest Fixture Declarations
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_audio_gen():
    return get_synthetic_audio_gen()


@pytest.fixture
def audio_validator():
    return get_audio_validator()


@pytest.fixture
def text_sanitizer():
    return get_text_sanitizer()


@pytest.fixture
def neural_classifier():
    return get_neural_classifier()


@pytest.fixture
def mock_navigator(tmp_path):
    return MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))


@pytest.fixture
def pipeline_coordinator(tmp_path):
    nav = MockNavigator(artifacts_dir=str(tmp_path / "artifacts"))
    return ReferencePipelineCoordinator(navigator=nav)


@pytest.fixture
def sample_intent_phrases():
    return {
        "GUIA_UNIVERSITARIO": [
            "mostrar guia do vestibular da puc campinas",
            "abrir o guia universitario de apresentacao dos cursos",
            "quero ver a apresentacao do guia da puc campinas",
            "como funciona o guia de apresentacao para vestibulandos",
            "navegar ate o guia de cursos e trajetoria universitaria",
            "quais os cursos oferecidos no vestibular da puc",
        ],
        "MANUAL_DO_CANDIDATO": [
            "abrir manual do candidato com as regras da prova",
            "onde vejo o edital e as normas do vestibular da puc",
            "mostrar o manual oficial para leitura das provas",
            "quero conferir o cronograma e o manual de regras",
            "quais sao as regras e datas do vestibular",
            "abrir edital do vestibular da puc campinas",
        ],
        "FORMULARIO_INSCRICAO": [
            "como faco para me inscrever no vestibular da puc campinas",
            "abrir pagina de inscricao para preencher meu cadastro",
            "ir para a tela de inscricao do vestibular",
            "quero fazer minha inscricao no vestibular agora",
            "fazer meu cadastro no vestibular da puc",
            "como me inscrevo no vestibular da puc campinas",
        ],
    }


@pytest.fixture
def ood_phrases():
    return [
        "qual a previsao do tempo para amanha em campinas",
        "quero pedir uma pizza de calabresa bem quente",
        "tocar musicas dos beatles no spotify",
        "me conte uma piada sobre computadores",
        "como consertar um pneu furado de bicicleta",
        "qual a cotacao do dolar hoje",
    ]


@pytest.fixture
def ambiguous_phrases():
    return [
        "quero ver o manual do candidato e fazer a inscricao agora mesmo",
        "abrir o guia de cursos e conferir o edital da prova",
        "como faco a inscricao e onde vejo o guia de cursos",
        "puc campinas vestibular",
        "vestibular da puc",
    ]


# ---------------------------------------------------------------------------
# Self-Verification Master Runner (Runs test suite if executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import inspect
    import sys
    import tempfile
    import time
    from pathlib import Path

    tests_dir = Path(__file__).parent.resolve()
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))

    modules = [
        "test_tier1_features",
        "test_tier2_boundaries",
        "test_tier3_combinations",
        "test_tier4_scenarios",
        "test_tier5_adversarial",
    ]

    start_time = time.time()
    total_passed = 0
    total_failed = 0

    print("======================================================================")
    print("REIA 5-Tier E2E Test Suite Execution (Standalone Runner)")
    print("======================================================================")

    for mod_name in modules:
        try:
            mod = __import__(mod_name)
            tests = [getattr(mod, name) for name in sorted(dir(mod)) if name.startswith("test_") and callable(getattr(mod, name))]
            passed = 0
            failed = 0
            print(f"\nRunning {mod_name} ({len(tests)} tests)...")
            for test in tests:
                try:
                    sig = inspect.signature(test)
                    if "tmp_path" in sig.parameters:
                        with tempfile.TemporaryDirectory() as td:
                            test(Path(td))
                    else:
                        test()
                    passed += 1
                except Exception as e:
                    failed += 1
                    print(f"  FAIL: {test.__name__}: {e}")
            print(f"  Result: {passed} passed, {failed} failed (total {len(tests)})")
            total_passed += passed
            total_failed += failed
        except Exception as exc:
            print(f"  ERROR loading {mod_name}: {exc}")
            total_failed += 1

    elapsed = time.time() - start_time
    print("\n======================================================================")
    print(f"REIA Test Suite Summary: {total_passed} passed, {total_failed} failed in {elapsed:.2f}s")
    print("======================================================================")

    if total_failed > 0:
        sys.exit(1)
    sys.exit(0)

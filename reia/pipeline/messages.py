"""
User Feedback and Accessible Messaging Subsystem for REIA.

Provides clear, friendly, and accessible Brazilian Portuguese feedback messages
for CLI output, screen readers, and vocal feedback (TTS) across all pipeline stages:
- Audio constraint validation (silence, noise/SNR, duration limits, corruption)
- Lexical gating and neural intent classification (OOD, ambiguity, low confidence)
- Headless web navigation (success confirmation, connection error, timeout)
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Audio Feedback Messages
# ---------------------------------------------------------------------------
MSG_AUDIO_VALID = "Áudio válido e pronto para processamento."
MSG_AUDIO_SILENCE = "Não detectamos sua voz. O áudio está muito baixo ou em silêncio. Tente novamente."
MSG_AUDIO_NOISE = "Ruído excessivo detectado no ambiente. Por favor, fale em um local mais silencioso."
MSG_AUDIO_TOO_SHORT = "Comando de voz muito curto ({duration:.2f}s). Por favor, fale a frase completa (mínimo de {min_duration:.1f}s)."
MSG_AUDIO_TOO_LONG = "Comando de voz muito longo ({duration:.2f}s). Por favor, seja mais sucinto (máximo de {max_duration:.1f}s)."
MSG_AUDIO_CORRUPTED = "O formato de áudio é inválido ou o arquivo está corrompido. Utilize formato WAV PCM de 16 bits."
MSG_AUDIO_EMPTY = "Nenhum dado de áudio foi recebido. Verifique o microfone ou o arquivo enviado."


# ---------------------------------------------------------------------------
# Neural Classification Feedback Messages
# ---------------------------------------------------------------------------
MSG_NEURAL_OOD = "Comando fora do domínio do vestibular. Por favor, pergunte sobre cursos, manual ou inscrição do vestibular da PUC-Campinas."
MSG_NEURAL_AMBIGUOUS_INTENT = "Seu comando parece ambíguo entre múltiplas opções do vestibular. Por favor escolha uma opção por vez."
MSG_NEURAL_AMBIGUOUS_DOMAIN = "Você mencionou o vestibular da PUC-Campinas, mas não especificou a ação desejada (guia de cursos, manual do candidato ou inscrição)."
MSG_NEURAL_LOW_CONFIDENCE = "Não entendi claramente seu pedido com confiança suficiente. Por favor repita o comando."
MSG_NEURAL_EMPTY_TEXT = "Nenhum comando de texto foi fornecido para classificação."


# ---------------------------------------------------------------------------
# Web Navigation Feedback Messages
# ---------------------------------------------------------------------------
MSG_NAVIGATION_SUCCESS_SHORT = "Navegando para {intent_name}: {target_url}"
MSG_NAVIGATION_SUCCESS_FULL = "Navegação concluída com sucesso para '{page_title}' ({target_url}). Captura de tela salva em {screenshot_path}."
MSG_NAVIGATION_ERROR = "Falha durante a navegação web para '{target_url}': {error_message}."
MSG_NAVIGATION_SKIPPED = "Navegação não executada devido a rejeições em etapas anteriores."


# ---------------------------------------------------------------------------
# Status Mapping Dictionary
# ---------------------------------------------------------------------------
FEEDBACK_MAP: Dict[str, str] = {
    "VALID": MSG_AUDIO_VALID,
    "SILENCE_DETECTED": MSG_AUDIO_SILENCE,
    "EXCESSIVE_NOISE": MSG_AUDIO_NOISE,
    "TOO_SHORT": MSG_AUDIO_TOO_SHORT,
    "TOO_LONG": MSG_AUDIO_TOO_LONG,
    "CORRUPTED_FORMAT": MSG_AUDIO_CORRUPTED,
    "EMPTY_INPUT": MSG_AUDIO_EMPTY,
    "OOD_NO_DOMAIN_KEYWORDS": MSG_NEURAL_OOD,
    "AMBIGUOUS_INTENT": MSG_NEURAL_AMBIGUOUS_INTENT,
    "AMBIGUOUS_DOMAIN_ONLY": MSG_NEURAL_AMBIGUOUS_DOMAIN,
    "LOW_CONFIDENCE": MSG_NEURAL_LOW_CONFIDENCE,
}


def get_audio_feedback_message(
    status: str,
    duration: float = 0.0,
    min_duration: float = 0.8,
    max_duration: float = 12.0,
    fallback: Optional[str] = None,
) -> str:
    """Returns accessible Portuguese message for an audio validation status."""
    key = str(status).upper().strip()
    if key == "TOO_SHORT":
        return MSG_AUDIO_TOO_SHORT.format(duration=duration, min_duration=min_duration)
    if key == "TOO_LONG":
        return MSG_AUDIO_TOO_LONG.format(duration=duration, max_duration=max_duration)
    return FEEDBACK_MAP.get(key, fallback or MSG_AUDIO_SILENCE)


def get_neural_feedback_message(
    rejection_code: Optional[str],
    intent_name: Optional[str] = None,
    target_url: Optional[str] = None,
    fallback: Optional[str] = None,
) -> str:
    """Returns accessible Portuguese message for a neural classification result."""
    if not rejection_code and intent_name and target_url:
        return MSG_NAVIGATION_SUCCESS_SHORT.format(intent_name=intent_name, target_url=target_url)
    key = str(rejection_code or "").upper().strip()
    return FEEDBACK_MAP.get(key, fallback or MSG_NEURAL_LOW_CONFIDENCE)


def get_navigation_feedback_message(
    success: bool,
    target_url: str,
    page_title: str = "",
    screenshot_path: str = "",
    error_message: Optional[str] = None,
    intent_name: Optional[str] = None,
) -> str:
    """Returns accessible Portuguese message for navigation outcome."""
    if success:
        return MSG_NAVIGATION_SUCCESS_SHORT.format(
            intent_name=intent_name or page_title or "PUC-Campinas",
            target_url=target_url,
        )
    return MSG_NAVIGATION_ERROR.format(
        target_url=target_url,
        error_message=error_message or "Erro desconhecido ao carregar página",
    )

"""
Target Route Dispatcher and Navigator Factory for REIA.

Provides:
- Canonical routing for PUC-Campinas Vestibular portal intents:
    - GUIA_UNIVERSITARIO -> https://vestibular.puc-campinas.edu.br/guia-universitario/
    - MANUAL_DO_CANDIDATO -> https://vestibular.puc-campinas.edu.br/manual-do-candidato/
    - FORMULARIO_INSCRICAO -> https://vestibular.puc-campinas.edu.br/
- Bidirectional intent <-> URL resolution.
- Dynamic get_navigator() factory with environment-aware auto-fallback.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, Optional

from reia.core.config import TARGET_URLS, TargetUrls

if TYPE_CHECKING:
    from reia.navigation.base import BaseNavigator

logger = logging.getLogger(__name__)

# Canonical map of intents to target URLs
CANONICAL_ROUTES: Dict[str, str] = {
    "GUIA_UNIVERSITARIO": TargetUrls.GUIA_UNIVERSITARIO,
    "MANUAL_DO_CANDIDATO": TargetUrls.MANUAL_DO_CANDIDATO,
    "FORMULARIO_INSCRICAO": TargetUrls.FORMULARIO_INSCRICAO,
}

# Inverted mapping for URL -> Intent lookup
URL_TO_INTENT: Dict[str, str] = {
    TargetUrls.GUIA_UNIVERSITARIO.rstrip("/"): "GUIA_UNIVERSITARIO",
    TargetUrls.MANUAL_DO_CANDIDATO.rstrip("/"): "MANUAL_DO_CANDIDATO",
    TargetUrls.FORMULARIO_INSCRICAO.rstrip("/"): "FORMULARIO_INSCRICAO",
}


def get_target_url(intent_name: str) -> str:
    """Return the official target URL corresponding to an intent name.

    Args:
        intent_name: Identifier such as 'GUIA_UNIVERSITARIO', 'MANUAL_DO_CANDIDATO',
            or 'FORMULARIO_INSCRICAO'. Case-insensitive.

    Returns:
        Canonical URL string. Defaults to root vestibular portal for unknown intents.
    """
    normalized = (intent_name or "").upper().strip()
    if normalized in CANONICAL_ROUTES:
        return CANONICAL_ROUTES[normalized]
    # Check if defined in reia.core.config.TARGET_URLS
    if normalized in TARGET_URLS:
        return TARGET_URLS[normalized]
    return TargetUrls.FORMULARIO_INSCRICAO


def resolve_intent_from_url(url: str) -> Optional[str]:
    """Resolve the canonical intent name corresponding to a URL.

    Args:
        url: URL string to inspect.

    Returns:
        Canonical intent name if recognized, or None.
    """
    if not url:
        return None

    clean_url = url.strip().rstrip("/").lower()

    # Exact matches
    for base_url, intent in URL_TO_INTENT.items():
        if clean_url == base_url.lower():
            return intent

    # Substring heuristics
    if "guia-universitario" in clean_url or "guia" in clean_url:
        return "GUIA_UNIVERSITARIO"
    if "manual-do-candidato" in clean_url or "manual" in clean_url or "edital" in clean_url:
        return "MANUAL_DO_CANDIDATO"
    if "vestibular.puc-campinas.edu.br" in clean_url or "inscricao" in clean_url:
        return "FORMULARIO_INSCRICAO"

    return None


def is_valid_intent(intent_name: str) -> bool:
    """Check whether an intent name belongs to the canonical vestibular classes."""
    if not intent_name:
        return False
    return intent_name.upper().strip() in CANONICAL_ROUTES


def is_chrome_and_selenium_available() -> bool:
    """Check whether Selenium and Chrome webdriver are available in the local environment."""
    try:
        import selenium  # noqa: F401
        from selenium import webdriver  # noqa: F401
        from selenium.webdriver.chrome.options import Options  # noqa: F401
        return True
    except (ImportError, Exception):
        return False


def get_navigator(
    mode: str = "auto",
    artifacts_dir: str = "artifacts",
    **kwargs,
) -> BaseNavigator:
    """Factory creating the appropriate web navigator based on configuration and environment.

    Args:
        mode: Navigation mode:
            - 'auto': Use Selenium if available and functional, otherwise gracefully fallback to Mock.
            - 'mock': Always use deterministic offline MockNavigator with local fixtures.
            - 'selenium' | 'live': Use standard Selenium 4 headless Chrome navigator.
        artifacts_dir: Path to directory where screenshots and metadata will be stored.
        **kwargs: Extra parameters passed to the navigator constructor.

    Returns:
        Instance of BaseNavigator (SeleniumNavigator or MockNavigator).

    Raises:
        ValueError: If mode is not one of ('auto', 'mock', 'selenium', 'live').
    """
    env_mode = os.getenv("REIA_NAVIGATOR_MODE", "").strip().lower()
    selected_mode = (env_mode or mode).strip().lower()

    if selected_mode in ("mock", "offline"):
        from reia.navigation.mock_driver import MockNavigator
        return MockNavigator(artifacts_dir=artifacts_dir, **kwargs)

    elif selected_mode in ("selenium", "live", "chrome"):
        from reia.navigation.selenium_driver import SeleniumNavigator
        return SeleniumNavigator(artifacts_dir=artifacts_dir, **kwargs)

    elif selected_mode == "auto":
        # Environment detection
        if is_chrome_and_selenium_available():
            try:
                from reia.navigation.selenium_driver import SeleniumNavigator
                # Attempt light instantiation without starting browser yet
                return SeleniumNavigator(artifacts_dir=artifacts_dir, **kwargs)
            except Exception as e:
                logger.warning("Failed to initialize SeleniumNavigator in auto mode (%s); falling back to MockNavigator.", e)

        from reia.navigation.mock_driver import MockNavigator
        return MockNavigator(artifacts_dir=artifacts_dir, **kwargs)

    else:
        raise ValueError(
            f"Modo de navegação desconhecido: '{mode}'. Opções válidas: 'auto', 'mock', 'selenium'."
        )

"""
Headless Selenium 4 Automation Driver for REIA.

Standard Selenium Chrome webdriver completely free of Google Colab packages.
Provides:
- Resilient headless options (--headless=new, --no-sandbox, --disable-gpu).
- Explicit WebDriverWait replacing hardcoded sleeps.
- DOM heading and link extraction.
- Automatic capture and persistence of 1280x800 screenshots and metadata JSON.
- Clean error handling catching missing binaries, sandbox restrictions, and timeouts.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from reia.navigation.artifacts import ArtifactEngine
from reia.navigation.base import BaseNavigator, NavigationResult
from reia.navigation.routes import get_target_url, resolve_intent_from_url

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def build_chrome_options(
    headless: bool = True,
    custom_user_agent: Optional[str] = None,
    window_size: str = "1280,800",
) -> Any:
    """Build production Chrome Options with headless and anti-detection flags.

    Args:
        headless: Whether to run Chrome in headless mode.
        custom_user_agent: Custom User-Agent string. Defaults to realistic macOS Chrome.
        window_size: Browser dimensions formatted as 'width,height'.

    Returns:
        Configured selenium.webdriver.chrome.options.Options instance.

    Raises:
        RuntimeError: If selenium package is not installed.
    """
    try:
        from selenium.webdriver.chrome.options import Options
    except ImportError as e:
        raise RuntimeError(
            "A biblioteca 'selenium' não está instalada no ambiente. "
            "Instale com 'pip install selenium' para utilizar o driver de produção."
        ) from e

    options = Options()

    if headless:
        # Chrome 109+ modern headless mode
        options.add_argument("--headless=new")

    # Essential sandbox & memory stability flags
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument(f"--window-size={window_size}")

    # Anti-bot / WAF detection mitigation
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    ua = custom_user_agent or DEFAULT_USER_AGENT
    options.add_argument(f"user-agent={ua}")

    return options


try:
    import selenium
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    By = None
    EC = None
    WebDriverWait = None


class SeleniumNavigator(BaseNavigator):
    """Production web navigator automating local headless Chrome via standard Selenium 4."""

    def __init__(
        self,
        artifacts_dir: str | Path = "artifacts",
        artifacts_engine: Optional[ArtifactEngine] = None,
        headless: bool = True,
        timeout: int = 15,
        custom_user_agent: Optional[str] = None,
        driver: Optional[Any] = None,
    ):
        """Initialize SeleniumNavigator.

        Args:
            artifacts_dir: Directory where screenshots and metadata will be persisted.
            artifacts_engine: Optional ArtifactEngine instance.
            headless: Whether to run in headless mode.
            timeout: Maximum wait time in seconds for DOM elements.
            custom_user_agent: Optional custom User-Agent string.
            driver: Optional pre-configured WebDriver instance (useful for dependency injection).
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_engine = artifacts_engine or ArtifactEngine(artifacts_dir=self.artifacts_dir)
        self.headless = headless
        self.timeout = timeout
        self.custom_user_agent = custom_user_agent
        self._driver = driver
        self._owns_driver = driver is None

    def _get_or_create_driver(self) -> Any:
        """Obtain existing WebDriver or instantiate standard Chrome driver."""
        if self._driver is not None:
            return self._driver

        try:
            from selenium import webdriver
        except ImportError as e:
            raise RuntimeError(
                f"Selenium não está instalado no ambiente: {e}. "
                "Execute 'pip install selenium' ou utilize o MockNavigator."
            ) from e

        options = build_chrome_options(
            headless=self.headless,
            custom_user_agent=self.custom_user_agent,
        )

        try:
            self._driver = webdriver.Chrome(options=options)
            return self._driver
        except Exception as e:
            logger.error("Failed to initialize Chrome webdriver: %s", e)
            raise RuntimeError(f"Falha ao inicializar o navegador Chrome: {e}") from e

    def navigate(self, target_url: str, intent_name: str) -> NavigationResult:
        """Navigate to target URL, wait dynamically for page readiness, and capture artifacts.

        Args:
            target_url: URL to navigate to.
            intent_name: Canonical intent identifier.

        Returns:
            NavigationResult populated with captured data, or error details on failure.
        """
        clean_intent = (intent_name or "").upper().strip()
        if not clean_intent and target_url:
            clean_intent = resolve_intent_from_url(target_url) or "UNKNOWN"

        effective_url = target_url or get_target_url(clean_intent)

        start_time = time.time()

        try:
            driver = self._get_or_create_driver()
        except Exception as e:
            return NavigationResult(
                success=False,
                intent_name=clean_intent,
                target_url=effective_url,
                page_title="",
                screenshot_path="",
                metadata_path="",
                headings=[],
                links=[],
                error_message=f"Falha ao inicializar navegador web: {str(e)}",
                load_time_seconds=0.0,
            )

        try:
            driver.get(effective_url)

            # Dynamic explicit synchronization
            if HAS_SELENIUM and WebDriverWait is not None and EC is not None and By is not None:
                wait = WebDriverWait(driver, self.timeout)
                # 1. Wait for document.readyState == 'complete'
                wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
                # 2. Wait for <body> tag presence
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            else:
                # Wait loop for document.readyState when testing with injected mock driver
                end_time = time.time() + self.timeout
                ready = False
                while time.time() < end_time:
                    try:
                        state = driver.execute_script("return document.readyState")
                        if state == "complete":
                            ready = True
                            break
                    except Exception:
                        pass
                    time.sleep(0.02)
                if not ready:
                    raise TimeoutError("Tempo limite de carregamento da página excedido (readyState != complete).")

            # Wait for any entrance animations or banner overlay transitions
            time.sleep(1.0)

            # Interactive Guidance: Highlight & Auto-focus interactive elements (e.g. COMECE AGORA)
            try:
                highlight_script = """
                // Locate call-to-action buttons/links (e.g., 'COMECE AGORA', 'INSCREVA-SE', or main CTA)
                const ctas = Array.from(document.querySelectorAll('a, button')).filter(el => {
                    const txt = (el.innerText || '').toUpperCase();
                    const href = (el.getAttribute('href') || '').toLowerCase();
                    return txt.includes('COMECE AGORA') || txt.includes('INSCREVA') || href.includes('/guia') || href.includes('inscricao');
                });

                if (ctas.length > 0) {
                    const target = ctas[0];
                    // Scroll target into view
                    target.scrollIntoView({ behavior: 'instant', block: 'center' });
                    // Visual assistance: apply accessibility pulse border highlight
                    target.style.outline = '4px solid #FFCC00';
                    target.style.boxShadow = '0 0 15px rgba(255, 204, 0, 0.8)';
                    target.style.transition = 'all 0.3s ease';
                } else {
                    // Fallback: smooth scroll past hero banner
                    window.scrollBy({ top: 450, behavior: 'instant' });
                }
                """
                driver.execute_script(highlight_script)
                time.sleep(0.6)
            except Exception:
                pass

            # DOM Extraction
            page_title = driver.title.strip() if driver.title else "Sem título"

            tag_by = By.TAG_NAME if By is not None else "tag name"
            h1_elements = [
                el.text.strip()
                for el in driver.find_elements(tag_by, "h1")
                if getattr(el, "text", None) and el.text.strip()
            ]
            h2_elements = [
                el.text.strip()
                for el in driver.find_elements(tag_by, "h2")
                if getattr(el, "text", None) and el.text.strip()
            ][:10]
            headings = h1_elements + h2_elements

            links: List[Dict[str, str]] = []
            for a in driver.find_elements(tag_by, "a")[:30]:
                href = a.get_attribute("href")
                text = a.text.strip() if getattr(a, "text", None) else ""
                if href:
                    links.append({"href": href, "text": text or href})

            load_duration = max(0.01, time.time() - start_time)

            # Screenshot capture
            png_bytes = driver.get_screenshot_as_png()
            screenshot_path = self.artifacts_engine.save_screenshot(
                intent_name=clean_intent,
                image_bytes=png_bytes,
            )

            # Metadata JSON persistence
            metadata_path = self.artifacts_engine.save_metadata(
                intent_name=clean_intent,
                target_url=effective_url,
                page_title=page_title,
                headings=headings,
                screenshot_path=screenshot_path,
                duration=load_duration,
                links=links,
            )

            return NavigationResult(
                success=True,
                intent_name=clean_intent,
                target_url=effective_url,
                page_title=page_title,
                screenshot_path=screenshot_path,
                metadata_path=metadata_path,
                headings=headings,
                links=links,
                error_message=None,
                load_time_seconds=load_duration,
            )

        except Exception as exc:
            # Check for specific selenium exceptions without hard dependency at import time
            exc_type = type(exc).__name__
            logger.error("Navigation error (%s) on %s: %s", exc_type, effective_url, exc)

            if "TimeoutException" in exc_type:
                error_msg = f"Tempo limite de carregamento da página excedido (Timeout): {exc}"
            elif "WebDriverException" in exc_type:
                error_msg = f"Erro de automação do Selenium WebDriver: {exc}"
            else:
                error_msg = f"Falha durante a navegação web: {exc}"

            return NavigationResult(
                success=False,
                intent_name=clean_intent,
                target_url=effective_url,
                page_title="",
                screenshot_path="",
                metadata_path="",
                headings=[],
                links=[],
                error_message=error_msg,
                load_time_seconds=max(0.01, time.time() - start_time),
            )

    def close(self) -> None:
        """Close browser driver session if managed by this instance."""
        if self._driver is not None and self._owns_driver:
            try:
                self._driver.quit()
            except Exception as e:
                logger.warning("Error closing Chrome webdriver: %s", e)
            finally:
                self._driver = None

    def quit(self) -> None:
        """Alias for close()."""
        self.close()

    def __enter__(self) -> SeleniumNavigator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# Backward-compatible alias matching survey report
LocalSeleniumNavigator = SeleniumNavigator

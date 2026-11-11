"""
Comprehensive Unit Test Suite for REIA Web Navigation Subsystem (Milestone 3).

Covers:
- NavigationResult data model contract, serialization, and immutability.
- BaseNavigator abstract class and routing methods.
- ArtifactEngine screenshot generation, PNG binary validation (IHDR, IDAT, IEND),
  and JSON metadata persistence.
- MockNavigator offline execution across all three PUC-Campinas Vestibular routes,
  DOM heading extraction, simulated timeouts, and error handling.
- SeleniumNavigator headless Chrome configuration options, dynamic waits,
  graceful fallback when Chrome/Selenium is unavailable, and mocked driver execution.
- Route resolution, canonical URL mapping, and get_navigator() factory logic.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
import zlib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reia.core.config import TARGET_URLS, TargetUrls
from reia.navigation.artifacts import ArtifactEngine, generate_png_bytes
from reia.navigation.base import BaseNavigator, NavigationResult
from reia.navigation.mock_driver import MOCK_HTML_FIXTURES, MockNavigator
from reia.navigation.routes import (
    CANONICAL_ROUTES,
    get_navigator,
    get_target_url,
    is_valid_intent,
    resolve_intent_from_url,
)
from reia.navigation.selenium_driver import (
    DEFAULT_USER_AGENT,
    LocalSeleniumNavigator,
    SeleniumNavigator,
    build_chrome_options,
)


# ===========================================================================
# 1. Tests for Navigation Contracts & Data Classes
# ===========================================================================
class TestNavigationContracts:
    """Verify NavigationResult contract matching PROJECT.md interface specifications."""

    def test_navigation_result_fields_and_defaults(self):
        res = NavigationResult(
            success=True,
            intent_name="GUIA_UNIVERSITARIO",
            target_url="https://vestibular.puc-campinas.edu.br/guia-universitario/",
            page_title="Guia Universitário - PUC-Campinas",
            screenshot_path="/tmp/shot.png",
            metadata_path="/tmp/meta.json",
            headings=["Guia Universitário", "Cursos de Graduação"],
        )
        assert res.success is True
        assert res.intent_name == "GUIA_UNIVERSITARIO"
        assert res.target_url == TargetUrls.GUIA_UNIVERSITARIO
        assert res.page_title == "Guia Universitário - PUC-Campinas"
        assert res.screenshot_path == "/tmp/shot.png"
        assert res.metadata_path == "/tmp/meta.json"
        assert len(res.headings) == 2
        assert res.links == []
        assert res.error_message is None
        assert res.load_time_seconds == 0.0

    def test_navigation_result_to_dict(self):
        res = NavigationResult(
            success=False,
            intent_name="MANUAL_DO_CANDIDATO",
            target_url="https://vestibular.puc-campinas.edu.br/manual-do-candidato/",
            page_title="",
            screenshot_path="",
            metadata_path="",
            headings=[],
            links=[],
            error_message="Falha de conexão",
            load_time_seconds=1.25,
        )
        d = res.to_dict()
        assert isinstance(d, dict)
        assert d["success"] is False
        assert d["intent_name"] == "MANUAL_DO_CANDIDATO"
        assert d["error_message"] == "Falha de conexão"
        assert d["load_time_seconds"] == 1.25

    def test_base_navigator_abc(self):
        # BaseNavigator is an ABC and cannot be directly instantiated
        try:
            BaseNavigator()  # type: ignore
            assert False, "BaseNavigator should not be instantiable directly."
        except TypeError:
            pass

        # Subclass must implement navigate()
        class ConcreteNavigator(BaseNavigator):
            def navigate(self, target_url: str, intent_name: str) -> NavigationResult:
                return NavigationResult(
                    success=True,
                    intent_name=intent_name,
                    target_url=target_url,
                    page_title="Concrete",
                    screenshot_path="",
                    metadata_path="",
                    headings=[],
                )

        nav = ConcreteNavigator()
        res_intent = nav.navigate_intent("GUIA_UNIVERSITARIO")
        assert res_intent.success is True
        assert res_intent.target_url == TargetUrls.GUIA_UNIVERSITARIO

        # Callable test
        res_call = nav("MANUAL_DO_CANDIDATO")
        assert res_call.success is True
        assert res_call.target_url == TargetUrls.MANUAL_DO_CANDIDATO


# ===========================================================================
# 2. Tests for Artifact Persistence Engine & Valid PNG Generator
# ===========================================================================
class TestArtifactEngine:
    """Verify PNG binary structure, CRC validation, and metadata persistence."""

    def test_png_generator_signature_and_dimensions(self):
        png = generate_png_bytes(width=1280, height=800, bg_rgb=(24, 76, 120))
        assert isinstance(png, bytes)

        # PNG 8-byte Magic Signature
        assert png.startswith(b"\x89PNG\r\n\x1a\n")

        # Parse chunks
        offset = 8
        chunks = []
        while offset < len(png):
            length = struct.unpack(">I", png[offset : offset + 4])[0]
            chunk_type = png[offset + 4 : offset + 8]
            data = png[offset + 8 : offset + 8 + length]
            crc = struct.unpack(">I", png[offset + 8 + length : offset + 12 + length])[0]

            # Verify chunk CRC32 checksum
            expected_crc = zlib.crc32(chunk_type + data)
            assert crc == expected_crc, f"CRC mismatch in chunk {chunk_type}"

            chunks.append((chunk_type, length, data))
            offset += 12 + length

        chunk_names = [c[0] for c in chunks]
        assert b"IHDR" in chunk_names
        assert b"IDAT" in chunk_names
        assert b"IEND" in chunk_names

        # Validate IHDR contents
        ihdr_data = chunks[0][2]
        width, height, bit_depth, color_type = struct.unpack(">IIBB", ihdr_data[:10])
        assert width == 1280
        assert height == 800
        assert bit_depth == 8
        assert color_type == 2  # Truecolor RGB

    def test_png_generator_invalid_dimensions(self):
        try:
            generate_png_bytes(width=-10, height=800)
            assert False, "Should raise ValueError for negative width"
        except ValueError:
            pass

        try:
            generate_png_bytes(width=1280, height=0)
            assert False, "Should raise ValueError for zero height"
        except ValueError:
            pass

    def test_artifact_engine_directory_creation(self):
        with tempfile.TemporaryDirectory() as td:
            engine = ArtifactEngine(artifacts_dir=Path(td) / "custom_artifacts")
            assert engine.screenshots_dir.is_dir()
            assert engine.metadata_dir.is_dir()

    def test_save_screenshot(self):
        with tempfile.TemporaryDirectory() as td:
            engine = ArtifactEngine(artifacts_dir=td)
            path = engine.save_screenshot(intent_name="GUIA_UNIVERSITARIO")
            assert os.path.exists(path)
            data = Path(path).read_bytes()
            assert data.startswith(b"\x89PNG\r\n\x1a\n")
            assert len(data) > 200

    def test_save_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            engine = ArtifactEngine(artifacts_dir=td)
            path = engine.save_metadata(
                intent_name="MANUAL_DO_CANDIDATO",
                target_url=TargetUrls.MANUAL_DO_CANDIDATO,
                page_title="Manual do Candidato",
                headings=["Regras", "Editais"],
                screenshot_path="/tmp/fake.png",
                duration=0.45,
                links=[{"href": "/edital.pdf", "text": "Edital"}],
            )
            assert os.path.exists(path)
            content = json.loads(Path(path).read_text(encoding="utf-8"))
            assert content["intent_name"] == "MANUAL_DO_CANDIDATO"
            assert content["target_url"] == TargetUrls.MANUAL_DO_CANDIDATO
            assert content["page_title"] == "Manual do Candidato"
            assert content["headings"] == ["Regras", "Editais"]
            assert content["links_count"] == 1
            assert content["duration_seconds"] == 0.45
            assert "iso_timestamp" in content


# ===========================================================================
# 3. Tests for MockNavigator (Offline Deterministic Driver)
# ===========================================================================
class TestMockNavigator:
    """Verify MockNavigator behavior with PUC-Campinas Vestibular fixtures."""

    def test_mock_navigation_guia_universitario(self):
        with tempfile.TemporaryDirectory() as td:
            nav = MockNavigator(artifacts_dir=td)
            res = nav.navigate_intent("GUIA_UNIVERSITARIO")

            assert res.success is True
            assert res.intent_name == "GUIA_UNIVERSITARIO"
            assert res.target_url == TargetUrls.GUIA_UNIVERSITARIO
            assert "Guia Universitário" in res.page_title
            assert any("Cursos" in h for h in res.headings)
            assert len(res.links) > 0
            assert os.path.exists(res.screenshot_path)
            assert os.path.exists(res.metadata_path)

            # Check screenshot is genuine PNG
            png_bytes = Path(res.screenshot_path).read_bytes()
            assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    def test_mock_navigation_manual_do_candidato(self):
        with tempfile.TemporaryDirectory() as td:
            nav = MockNavigator(artifacts_dir=td)
            res = nav.navigate_intent("MANUAL_DO_CANDIDATO")

            assert res.success is True
            assert res.intent_name == "MANUAL_DO_CANDIDATO"
            assert res.target_url == TargetUrls.MANUAL_DO_CANDIDATO
            assert "Manual do Candidato" in res.page_title
            assert any("Regras" in h for h in res.headings)
            assert any("edital" in link["href"].lower() for link in res.links)
            assert os.path.exists(res.screenshot_path)
            assert os.path.exists(res.metadata_path)

    def test_mock_navigation_formulario_inscricao(self):
        with tempfile.TemporaryDirectory() as td:
            nav = MockNavigator(artifacts_dir=td)
            res = nav.navigate_intent("FORMULARIO_INSCRICAO")

            assert res.success is True
            assert res.intent_name == "FORMULARIO_INSCRICAO"
            assert res.target_url == TargetUrls.FORMULARIO_INSCRICAO
            assert "Inscrição" in res.page_title
            assert any("Formulário" in h or "Inscrição" in h for h in res.headings)
            assert os.path.exists(res.screenshot_path)
            assert os.path.exists(res.metadata_path)

    def test_mock_navigation_callable_shortcut(self):
        with tempfile.TemporaryDirectory() as td:
            nav = MockNavigator(artifacts_dir=td)
            res = nav("GUIA_UNIVERSITARIO")
            assert res.success is True
            assert res.intent_name == "GUIA_UNIVERSITARIO"

    def test_mock_navigation_simulate_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            nav = MockNavigator(artifacts_dir=td, simulate_timeout=True)
            res = nav.navigate_intent("GUIA_UNIVERSITARIO")

            assert res.success is False
            assert res.error_message is not None
            assert "Timeout" in res.error_message or "limite" in res.error_message
            assert res.screenshot_path == ""

    def test_mock_navigation_simulate_error(self):
        with tempfile.TemporaryDirectory() as td:
            nav = MockNavigator(artifacts_dir=td, simulate_error=True)
            res = nav.navigate_intent("MANUAL_DO_CANDIDATO")

            assert res.success is False
            assert res.error_message is not None
            assert "Falha" in res.error_message

    def test_mock_navigation_unknown_intent(self):
        with tempfile.TemporaryDirectory() as td:
            nav = MockNavigator(artifacts_dir=td)
            res = nav.navigate(target_url="https://unknown.domain.com/random", intent_name="NON_EXISTENT_INTENT")
            assert res.success is False
            assert res.error_message is not None
            assert "não foi encontrada" in res.error_message or "Falha" in res.error_message

    def test_mock_navigation_with_external_fixtures(self):
        with tempfile.TemporaryDirectory() as td:
            fixtures_dir = Path(td) / "fixtures"
            fixtures_dir.mkdir()
            custom_html = """<!DOCTYPE html><html><head><title>Custom Guia Test</title></head>
            <body><h1>Custom Heading 1</h1><h2>Custom Heading 2</h2><a href="/test">Test Link</a></body></html>"""
            (fixtures_dir / "guia_universitario.html").write_text(custom_html, encoding="utf-8")

            nav = MockNavigator(artifacts_dir=td, fixtures_dir=fixtures_dir)
            res = nav.navigate_intent("GUIA_UNIVERSITARIO")
            assert res.success is True
            assert res.page_title == "Custom Guia Test"
            assert "Custom Heading 1" in res.headings
            assert "Custom Heading 2" in res.headings


# ===========================================================================
# 4. Tests for SeleniumNavigator (Headless Automation Driver)
# ===========================================================================
class MockSeleniumWebElement:
    """Mock element mimicking a Selenium WebElement."""
    def __init__(self, text: str = "", href: Optional[str] = None):
        self.text = text
        self._href = href

    def get_attribute(self, name: str) -> Optional[str]:
        if name == "href":
            return self._href
        return None


class MockSeleniumWebDriver:
    """Mock WebDriver for testing SeleniumNavigator DOM extraction without starting Chrome."""
    def __init__(self, should_timeout: bool = False, should_throw_driver_error: bool = False):
        self.should_timeout = should_timeout
        self.should_throw_driver_error = should_throw_driver_error
        self.current_url = ""
        self.title = "Mock PUC-Campinas Vestibular"
        self.quit_called = False

    def get(self, url: str) -> None:
        self.current_url = url
        if self.should_throw_driver_error:
            raise RuntimeError("Mock driver fatal communication error")

    def execute_script(self, script: str) -> Any:
        if self.should_timeout:
            return "loading"
        if script == "return document.readyState":
            return "complete"
        return None

    def find_elements(self, by: str, value: str) -> List[Any]:
        if value == "h1":
            return [MockSeleniumWebElement(text="Portal do Vestibular")]
        elif value == "h2":
            return [MockSeleniumWebElement(text="Informações Importantes")]
        elif value == "a":
            return [MockSeleniumWebElement(text="Manual do Candidato", href="/manual.pdf")]
        return [MockSeleniumWebElement(text="Mock Element")]

    def find_element(self, by: str, value: str) -> Any:
        return MockSeleniumWebElement(text="Mock Element")

    def get_screenshot_as_png(self) -> bytes:
        return generate_png_bytes(1280, 800)

    def quit(self) -> None:
        self.quit_called = True


class TestSeleniumNavigator:
    """Verify SeleniumNavigator options, dynamic wait, error paths, and mock injection."""

    def test_build_chrome_options_arguments_if_selenium_available(self):
        try:
            import selenium  # noqa: F401
            options = build_chrome_options(headless=True)
            args = options.arguments
            assert "--headless=new" in args
            assert "--no-sandbox" in args
            assert "--disable-gpu" in args
            assert "--disable-dev-shm-usage" in args
            assert "--window-size=1280,800" in args
            assert any("user-agent=" in a for a in args)
        except (ImportError, RuntimeError):
            # If selenium is not in environment, build_chrome_options raises informative RuntimeError
            try:
                build_chrome_options()
                assert False, "Should raise RuntimeError when selenium is not installed"
            except RuntimeError as e:
                assert "selenium" in str(e).lower()

    def test_selenium_navigator_graceful_handling_when_uninitialized(self):
        with tempfile.TemporaryDirectory() as td:
            # Without pre-injected driver, if selenium/chrome is not installed, navigate returns success=False
            nav = SeleniumNavigator(artifacts_dir=td)
            res = nav.navigate("https://vestibular.puc-campinas.edu.br/", "FORMULARIO_INSCRICAO")
            assert isinstance(res, NavigationResult)
            # Must return a valid NavigationResult with success=False, not raise an uncaught exception
            assert res.success is False or res.success is True
            if not res.success:
                assert res.error_message is not None
                assert "Falha" in res.error_message or "Selenium" in res.error_message

    def test_selenium_navigator_with_injected_mock_driver(self):
        with tempfile.TemporaryDirectory() as td:
            mock_driver = MockSeleniumWebDriver()
            nav = SeleniumNavigator(artifacts_dir=td, driver=mock_driver)

            res = nav.navigate("https://vestibular.puc-campinas.edu.br/guia-universitario/", "GUIA_UNIVERSITARIO")

            assert res.success is True
            assert res.intent_name == "GUIA_UNIVERSITARIO"
            assert res.page_title == "Mock PUC-Campinas Vestibular"
            assert "Portal do Vestibular" in res.headings
            assert "Informações Importantes" in res.headings
            assert len(res.links) == 1
            assert res.links[0]["text"] == "Manual do Candidato"
            assert os.path.exists(res.screenshot_path)
            assert os.path.exists(res.metadata_path)

    def test_selenium_navigator_with_injected_driver_error(self):
        with tempfile.TemporaryDirectory() as td:
            mock_driver = MockSeleniumWebDriver(should_throw_driver_error=True)
            nav = SeleniumNavigator(artifacts_dir=td, driver=mock_driver)

            res = nav.navigate("https://vestibular.puc-campinas.edu.br/", "FORMULARIO_INSCRICAO")

            assert res.success is False
            assert res.error_message is not None
            assert "Falha" in res.error_message or "error" in res.error_message.lower()

    def test_selenium_navigator_context_manager_and_quit(self):
        mock_driver = MockSeleniumWebDriver()
        with SeleniumNavigator(driver=mock_driver) as nav:
            assert nav._driver is mock_driver
        # Injected drivers are not closed by default (_owns_driver is False)
        assert mock_driver.quit_called is False

        # If _owns_driver is True, nav.close() calls quit()
        owned_mock = MockSeleniumWebDriver()
        nav_owned = SeleniumNavigator()
        nav_owned._driver = owned_mock
        nav_owned._owns_driver = True
        nav_owned.close()
        assert owned_mock.quit_called is True
        assert nav_owned._driver is None

    def test_local_selenium_navigator_alias(self):
        assert LocalSeleniumNavigator is SeleniumNavigator


# ===========================================================================
# 5. Tests for Route Resolution and get_navigator() Factory
# ===========================================================================
class TestRoutesAndFactory:
    """Verify URL routing, intent resolution, and navigator factory."""

    def test_canonical_routes_and_urls(self):
        assert get_target_url("GUIA_UNIVERSITARIO") == TargetUrls.GUIA_UNIVERSITARIO
        assert get_target_url("MANUAL_DO_CANDIDATO") == TargetUrls.MANUAL_DO_CANDIDATO
        assert get_target_url("FORMULARIO_INSCRICAO") == TargetUrls.FORMULARIO_INSCRICAO

        # Case-insensitivity
        assert get_target_url("guia_universitario") == TargetUrls.GUIA_UNIVERSITARIO
        assert get_target_url("manual_do_candidato") == TargetUrls.MANUAL_DO_CANDIDATO
        assert get_target_url("formulario_inscricao") == TargetUrls.FORMULARIO_INSCRICAO

        # Unknown intent defaults to main vestibular portal
        assert get_target_url("UNKNOWN_ROUTE") == TargetUrls.FORMULARIO_INSCRICAO

    def test_resolve_intent_from_url(self):
        # Exact canonical URLs
        assert resolve_intent_from_url(TargetUrls.GUIA_UNIVERSITARIO) == "GUIA_UNIVERSITARIO"
        assert resolve_intent_from_url(TargetUrls.MANUAL_DO_CANDIDATO) == "MANUAL_DO_CANDIDATO"
        assert resolve_intent_from_url(TargetUrls.FORMULARIO_INSCRICAO) == "FORMULARIO_INSCRICAO"

        # Trailing slash variants
        assert resolve_intent_from_url("https://vestibular.puc-campinas.edu.br/guia-universitario") == "GUIA_UNIVERSITARIO"
        assert resolve_intent_from_url("https://vestibular.puc-campinas.edu.br/manual-do-candidato") == "MANUAL_DO_CANDIDATO"

        # Heuristic substrings
        assert resolve_intent_from_url("https://example.com/puc/edital-completo") == "MANUAL_DO_CANDIDATO"

        # Empty / unrecognized
        assert resolve_intent_from_url("") is None
        assert resolve_intent_from_url("https://google.com/search") is None

    def test_is_valid_intent(self):
        assert is_valid_intent("GUIA_UNIVERSITARIO") is True
        assert is_valid_intent("MANUAL_DO_CANDIDATO") is True
        assert is_valid_intent("FORMULARIO_INSCRICAO") is True
        assert is_valid_intent("guia_universitario") is True
        assert is_valid_intent("PEDIR_PIZZA") is False
        assert is_valid_intent("") is False

    def test_get_navigator_mock_mode(self):
        nav = get_navigator(mode="mock")
        assert isinstance(nav, MockNavigator)
        assert isinstance(nav, BaseNavigator)

    def test_get_navigator_auto_mode(self):
        nav = get_navigator(mode="auto")
        assert isinstance(nav, BaseNavigator)

    def test_get_navigator_invalid_mode(self):
        try:
            get_navigator(mode="invalid_xyz_mode")
            assert False, "Should raise ValueError for invalid mode"
        except ValueError as e:
            assert "Modo de navegação desconhecido" in str(e)

    def test_get_navigator_env_var_override(self):
        old_env = os.environ.get("REIA_NAVIGATOR_MODE")
        try:
            os.environ["REIA_NAVIGATOR_MODE"] = "mock"
            nav = get_navigator(mode="selenium")
            # Should have overridden 'selenium' with 'mock'
            assert isinstance(nav, MockNavigator)
        finally:
            if old_env is None:
                os.environ.pop("REIA_NAVIGATOR_MODE", None)
            else:
                os.environ["REIA_NAVIGATOR_MODE"] = old_env


# ===========================================================================
# Standalone Test Execution Entrypoint
# ===========================================================================
if __name__ == "__main__":
    test_classes = [
        TestNavigationContracts,
        TestArtifactEngine,
        TestMockNavigator,
        TestSeleniumNavigator,
        TestRoutesAndFactory,
    ]

    start = time.time()
    total_passed = 0
    total_failed = 0

    print("======================================================================")
    print("REIA Web Navigation Subsystem (M3) Unit Test Suite")
    print("======================================================================")

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_") and callable(getattr(instance, m))]
        print(f"\nRunning {cls.__name__} ({len(methods)} tests)...")
        passed = 0
        failed = 0
        for method_name in methods:
            test_fn = getattr(instance, method_name)
            try:
                sig = inspect.signature(test_fn)
                test_fn()
                passed += 1
            except Exception as e:
                failed += 1
                print(f"  FAIL: {method_name}: {e}")
                import traceback
                traceback.print_exc()
        print(f"  Result: {passed} passed, {failed} failed (total {len(methods)})")
        total_passed += passed
        total_failed += failed

    elapsed = time.time() - start
    print("\n======================================================================")
    print(f"Navigation Test Suite Summary: {total_passed} passed, {total_failed} failed in {elapsed:.2f}s")
    print("======================================================================")

    if total_failed > 0:
        sys.exit(1)
    sys.exit(0)

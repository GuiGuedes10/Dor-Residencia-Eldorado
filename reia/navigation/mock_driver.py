"""
Offline Mock Navigation Driver for REIA.

Provides deterministic, offline execution with local HTML fixtures matching
PUC-Campinas Vestibular pages. Generates genuine 1280x800 PNG screenshots
using pure Python (struct and zlib) with zero external dependencies.
"""

from __future__ import annotations

from html.parser import HTMLParser
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple

from reia.navigation.artifacts import ArtifactEngine, generate_png_bytes
from reia.navigation.base import BaseNavigator, NavigationResult
from reia.navigation.routes import CANONICAL_ROUTES, get_target_url, resolve_intent_from_url

# Embedded canonical HTML fixtures for PUC-Campinas Vestibular pages
MOCK_HTML_FIXTURES: Dict[str, str] = {
    "GUIA_UNIVERSITARIO": """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Guia Universitário - PUC-Campinas</title>
</head>
<body>
    <header>
        <h1>Guia Universitário PUC-Campinas</h1>
    </header>
    <main>
        <h2>Cursos de Graduação</h2>
        <p>Explore a relação completa de cursos oferecidos no processo seletivo:</p>
        <div class="cursos-grid">
            <a href="/cursos/medicina">Medicina</a>
            <a href="/cursos/direito">Direito</a>
            <a href="/cursos/engenharia-software">Engenharia de Software</a>
            <a href="/cursos/administracao">Administração</a>
            <a href="/cursos/psicologia">Psicologia</a>
            <a href="/cursos/arquitetura">Arquitetura e Urbanismo</a>
        </div>
        <h2>Infraestrutura e Câmpus</h2>
        <p>Conheça os laboratórios, bibliotecas e centros de convivência dos Câmpus I e II.</p>
        <a href="/campus/tour-virtual">Tour Virtual do Campus</a>
    </main>
</body>
</html>""",
    "MANUAL_DO_CANDIDATO": """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Manual do Candidato e Editais - Vestibular PUC-Campinas</title>
</head>
<body>
    <header>
        <h1>Manual do Candidato Vestibular PUC-Campinas</h1>
    </header>
    <main>
        <h2>Regras Gerais da Prova</h2>
        <p>Edital oficial com normas, critérios de convocação e pesos das matérias.</p>
        <h2>Downloads Normativos</h2>
        <div class="downloads-lista">
            <a href="/downloads/edital_vestibular.pdf">Baixar Edital em PDF</a>
            <a href="/downloads/manual_do_candidato.pdf">Baixar Manual do Candidato</a>
            <a href="/downloads/cronograma_oficial.pdf">Cronograma Completo das Provas</a>
        </div>
    </main>
</body>
</html>""",
    "FORMULARIO_INSCRICAO": """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Inscrição - Vestibular PUC-Campinas</title>
</head>
<body>
    <header>
        <h1>Inscrição Vestibular PUC-Campinas</h1>
    </header>
    <main>
        <h2>Formulário de Cadastro do Candidato</h2>
        <form action="/enviar-inscricao" method="POST">
            <label for="nome">Nome Completo:</label>
            <input type="text" id="nome" name="nome" placeholder="Digite seu nome"/>
            <label for="cpf">CPF:</label>
            <input type="text" id="cpf" name="cpf" placeholder="000.000.000-00"/>
            <label for="curso">Curso Escolhido:</label>
            <select id="curso" name="curso">
                <option value="eng-software">Engenharia de Software</option>
                <option value="medicina">Medicina</option>
                <option value="direito">Direito</option>
            </select>
            <button type="submit">Concluir Inscrição</button>
        </form>
        <a href="/vestibular/ajuda-inscricao">Dúvidas sobre a inscrição</a>
    </main>
</body>
</html>""",
}


class _SimpleHTMLDataExtractor(HTMLParser):
    """Native standard-library HTML parser extracting title, headings, and links."""

    def __init__(self):
        super().__init__()
        self.title = ""
        self.headings: List[str] = []
        self.links: List[Dict[str, str]] = []

        self._in_title = False
        self._in_heading = False
        self._current_heading_text = ""
        self._current_link_href: Optional[str] = None
        self._current_link_text = ""

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attrs_dict = dict(attrs)
        if tag.lower() == "title":
            self._in_title = True
        elif tag.lower() in ("h1", "h2", "h3"):
            self._in_heading = True
            self._current_heading_text = ""
        elif tag.lower() == "a":
            self._current_link_href = attrs_dict.get("href", "")
            self._current_link_text = ""

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._in_heading:
            self._current_heading_text += data
        if self._current_link_href is not None:
            self._current_link_text += data

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
        elif tag.lower() in ("h1", "h2", "h3"):
            self._in_heading = False
            clean_heading = self._current_heading_text.strip()
            if clean_heading:
                self.headings.append(clean_heading)
            self._current_heading_text = ""
        elif tag.lower() == "a":
            if self._current_link_href is not None:
                clean_text = self._current_link_text.strip()
                self.links.append({
                    "href": self._current_link_href,
                    "text": clean_text or self._current_link_href,
                })
            self._current_link_href = None
            self._current_link_text = ""


class MockNavigator(BaseNavigator):
    """Deterministic offline web navigator using local HTML fixtures and pure-Python PNG generation."""

    def __init__(
        self,
        artifacts_dir: str | Path = "artifacts",
        artifacts_engine: Optional[ArtifactEngine] = None,
        simulate_timeout: bool = False,
        simulate_error: bool = False,
        fixtures_dir: Optional[str | Path] = None,
    ):
        """Initialize MockNavigator.

        Args:
            artifacts_dir: Directory where screenshots and metadata will be persisted.
            artifacts_engine: Optional existing ArtifactEngine instance.
            simulate_timeout: When True, simulates a network timeout failure.
            simulate_error: When True, simulates a generic navigation error.
            fixtures_dir: Optional directory to load external .html fixture files from.
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_engine = artifacts_engine or ArtifactEngine(artifacts_dir=self.artifacts_dir)
        self.simulate_timeout = simulate_timeout
        self.simulate_error = simulate_error
        self.fixtures_dir = Path(fixtures_dir) if fixtures_dir else None

    def _load_html(self, intent_name: str) -> Optional[str]:
        """Load HTML content for an intent, checking fixtures_dir first then embedded fixtures."""
        clean_intent = (intent_name or "").upper().strip()

        # Check external fixtures directory if provided
        if self.fixtures_dir and self.fixtures_dir.exists():
            candidate_files = [
                self.fixtures_dir / f"{clean_intent.lower()}.html",
                self.fixtures_dir / f"puc_{clean_intent.lower()}.html",
            ]
            for file_path in candidate_files:
                if file_path.exists():
                    return file_path.read_text(encoding="utf-8")

        # Fallback to embedded canonical fixtures
        return MOCK_HTML_FIXTURES.get(clean_intent)

    def navigate(self, target_url: str, intent_name: str) -> NavigationResult:
        """Execute deterministic mock navigation and generate authentic visual/textual artifacts.

        Args:
            target_url: Intended target URL.
            intent_name: Canonical intent identifier.

        Returns:
            NavigationResult populated with extracted DOM data and artifact paths.
        """
        clean_intent = (intent_name or "").upper().strip()

        # Resolve intent from target_url if not explicitly provided
        if not clean_intent and target_url:
            resolved = resolve_intent_from_url(target_url)
            clean_intent = resolved or "FORMULARIO_INSCRICAO"

        effective_url = target_url or get_target_url(clean_intent)

        # Simulated timeout condition
        if self.simulate_timeout:
            return NavigationResult(
                success=False,
                intent_name=clean_intent,
                target_url=effective_url,
                page_title="",
                screenshot_path="",
                metadata_path="",
                headings=[],
                links=[],
                error_message="Erro de rede: Tempo limite de carregamento da página excedido (Timeout).",
                load_time_seconds=0.0,
            )

        # Simulated generic error condition
        if self.simulate_error:
            return NavigationResult(
                success=False,
                intent_name=clean_intent,
                target_url=effective_url,
                page_title="",
                screenshot_path="",
                metadata_path="",
                headings=[],
                links=[],
                error_message=f"Falha de automação ao navegar para a intenção '{clean_intent}'.",
                load_time_seconds=0.0,
            )

        html_content = self._load_html(clean_intent)
        if not html_content:
            return NavigationResult(
                success=False,
                intent_name=clean_intent,
                target_url=effective_url,
                page_title="",
                screenshot_path="",
                metadata_path="",
                headings=[],
                links=[],
                error_message=f"Falha: Fixture HTML correspondente à intenção '{clean_intent}' não foi encontrada.",
                load_time_seconds=0.0,
            )

        start_time = time.time()

        # Parse DOM structure with native HTMLParser
        parser = _SimpleHTMLDataExtractor()
        try:
            parser.feed(html_content)
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
                error_message=f"Falha ao realizar parse do conteúdo HTML: {e}",
                load_time_seconds=0.0,
            )

        page_title = parser.title.strip() if parser.title else f"PUC-Campinas - {clean_intent}"
        headings = parser.headings
        links = parser.links

        # Generate genuine 1280x800 PNG bytes
        png_bytes = generate_png_bytes(width=1280, height=800, bg_rgb=(24, 76, 120))
        screenshot_path = self.artifacts_engine.save_screenshot(
            intent_name=clean_intent,
            image_bytes=png_bytes,
        )

        load_duration = max(0.01, time.time() - start_time)

        # Save metadata JSON file
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

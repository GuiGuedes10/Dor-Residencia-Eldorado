"""
Command-Line Interface (CLI) Entrypoint for REIA (Feature F17).

Provides CLI execution supporting:
- Audio files (--audio <path.wav>)
- Direct text queries (--text "<query>")
- Procedural synthetic audio scenarios (--synthetic <scenario>)
- Navigator selection (--navigator auto|mock|selenium)
- Custom artifact directories (--artifacts-dir <path>)
- Headless JSON output (--json)
- Quiet and verbose terminal logging (--quiet, --verbose)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import List, Optional

from reia.pipeline.coordinator import SYNTHETIC_SCENARIOS, VoiceNavigationPipeline


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for REIA CLI."""
    parser = argparse.ArgumentParser(
        prog="reia",
        description="REIA — Assistente de Navegação por Voz para o Vestibular PUC-Campinas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Executar com comando de texto direto:
  python3 -m reia --text "quero ver o guia universitario de cursos"

  # Executar com arquivo de áudio WAV:
  python3 -m reia --audio comando.wav

  # Executar cenário sintético para teste:
  python3 -m reia --synthetic valid_guia --navigator mock --json

  # Cenários sintéticos disponíveis:
  valid_guia, valid_manual, valid_inscricao, silence, short, long, noise, corrupted
        """,
    )

    input_group = parser.add_argument_group("Entrada")
    input_group.add_argument(
        "-a",
        "--audio",
        type=str,
        default=None,
        help="Caminho para arquivo de áudio (.wav) para processamento.",
    )
    input_group.add_argument(
        "-t",
        "--text",
        type=str,
        default=None,
        help="Comando de texto direto (ignora etapa de áudio).",
    )
    input_group.add_argument(
        "-s",
        "--synthetic",
        type=str,
        choices=list(SYNTHETIC_SCENARIOS.keys()),
        default=None,
        help="Gera e processa cenário de áudio sintético.",
    )

    nav_group = parser.add_argument_group("Navegação e Saída")
    nav_group.add_argument(
        "-m",
        "--navigator",
        type=str,
        choices=["auto", "mock", "selenium"],
        default="auto",
        help="Modo de navegação web (padrão: auto).",
    )
    nav_group.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Diretório onde capturas de tela e metadados serão gravados (padrão: artifacts).",
    )
    nav_group.add_argument(
        "-j",
        "--json",
        action="store_true",
        default=False,
        help="Exibe resultado formatado em JSON estruturado no stdout.",
    )
    nav_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Modo silencioso: exibe apenas a mensagem final ao usuário.",
    )
    nav_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Modo detalhado: exibe métricas acústicas, probabilidades e logs.",
    )
    nav_group.add_argument(
        "--exit-zero",
        action="store_true",
        default=False,
        help="Sempre encerra com código 0 caso a execução não tenha travado fatalmente.",
    )

    return parser


def format_human_output(result, verbose: bool = False) -> str:
    """Formats PipelineResult into an accessible, human-friendly terminal report."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("REIA — Assistente de Navegação por Voz (Vestibular PUC-Campinas)")
    lines.append("=" * 70)

    status_tag = "SUCESSO" if result.success else "REJEITADO / FALHA"
    lines.append(f"Status geral: {status_tag} (Etapa: {result.stage}, Código: {result.status})")
    lines.append(f"Mensagem:     {result.user_message}")

    if result.transcript:
        lines.append(f"Transcrição:  \"{result.transcript}\"")
    if result.sanitized_text:
        lines.append(f"Texto limpo:  \"{result.sanitized_text}\"")

    if result.classification and result.classification.is_valid_intent:
        lines.append("-" * 70)
        lines.append(f"Destino:      {result.classification.intent_name}")
        lines.append(f"URL Alvo:     {result.target_url}")
        lines.append(
            f"Confiança:    {result.confidence:.1%} (Margem: {result.margin:.2f}, Entropia: {result.entropy:.2f})"
        )

    if result.navigation and result.navigation.success:
        lines.append("-" * 70)
        lines.append(f"Título da Página: {result.navigation.page_title}")
        if result.screenshot_path:
            lines.append(f"Captura de Tela:  {result.screenshot_path}")
        if result.metadata_path:
            lines.append(f"Metadados JSON:   {result.metadata_path}")

    if verbose and result.audio_metrics:
        lines.append("-" * 70)
        lines.append("Métricas Acústicas:")
        for k, v in result.audio_metrics.items():
            if isinstance(v, float):
                lines.append(f"  - {k}: {v:.2f}")
            else:
                lines.append(f"  - {k}: {v}")

    lines.append("-" * 70)
    lines.append(f"Tempo de Execução: {result.execution_time_seconds:.2f}s")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else (logging.ERROR if args.quiet else logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Validate input presence
    if not (args.audio or args.text or args.synthetic):
        if args.json:
            error_dict = {
                "success": False,
                "stage": "INPUT_VALIDATION",
                "status": "MISSING_INPUT",
                "user_message": "Nenhum parâmetro de entrada fornecido. Utilize --audio, --text ou --synthetic.",
            }
            print(json.dumps(error_dict, indent=2, ensure_ascii=False))
            return 0 if args.exit_zero else 1
        else:
            parser.print_help()
            print("\n[ERRO] É necessário fornecer pelo menos uma entrada: --audio, --text ou --synthetic.")
            return 0 if args.exit_zero else 1

    # Instantiate pipeline
    pipeline = VoiceNavigationPipeline(
        navigator_mode=args.navigator,
        artifacts_dir=args.artifacts_dir,
    )

    # Execute
    result = pipeline.run(
        audio=args.audio,
        text=args.text,
        synthetic=args.synthetic,
    )

    # Output formatting
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    elif args.quiet:
        print(result.user_message)
    else:
        print(format_human_output(result, verbose=args.verbose))

    if result.success or args.exit_zero:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

# Contribuindo com o REIA / Dør

Obrigado por contribuir! Este guia resume o fluxo de trabalho de engenharia do projeto.

## Visão rápida do monorepo

| Componente | Pasta | Stack | Como rodar |
|---|---|---|---|
| Pipeline de referência | [`reia/`](reia/) | Python 3.10+ | `python3 -m reia --text "…"` |
| CLI de terminal | [`voice-nav-cli/`](voice-nav-cli/) | Node ≥ 18 | `npm start` |
| Extensão de navegador | [`extension/`](extension/) | JS (MV3) | carregar sem empacotar no Chrome |

Veja [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) para o desenho completo.

## Pré-requisitos

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Para a CLI: `cd voice-nav-cli && npm install` (requer `sox`/`ffmpeg` e `whisper.cpp` para voz; veja o README da CLI).

## Fluxo de contribuição

1. **Abra uma issue** descrevendo o bug ou a proposta antes de um PR grande.
2. **Crie um branch** a partir de `main`: `git checkout -b feat/descricao-curta`.
3. **Faça commits pequenos** no padrão [Conventional Commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
4. **Rode os testes** (ver abaixo) — o CI precisa passar.
5. **Abra o Pull Request** usando o template; descreva o que mudou e como testar.

## Testes

```bash
# Suíte completa (unit + tiers), via pytest
pytest -q

# Runner standalone sem dependências externas (só os 5 tiers)
python3 tests/conftest.py
```

Toda mudança em `reia/` deve vir acompanhada de teste. Mudanças de UX na extensão
devem preservar acessibilidade (ver [`docs/ACESSIBILIDADE.md`](docs/ACESSIBILIDADE.md)).

## Estilo de código

- **Python:** PEP 8, linha ≤ 100 colunas, type hints onde fizer sentido.
- **JavaScript:** 2 espaços, `const`/`let`, sem dependências de build na extensão.
- O [`.editorconfig`](.editorconfig) cobre indentação e fim de linha.

## Dados e modelos

- **Não** versione binários grandes (`*.bin`, `*.pt`, `node_modules/`) — já estão no `.gitignore`.
- O classificador de tópicos (`models/intent.json`) **é** versionado e é regenerado com
  `cd voice-nav-cli && npm run train` a partir de `training/intents.tsv`.
- Mapas de site (`voice-nav-cli/data/*.json`) são gerados por `npm run map` e não são versionados.

## Acessibilidade é requisito, não extra

O produto existe para **indicar e orientar** pessoas — com foco em baixa visão e
mobilidade reduzida. Qualquer mudança visual deve manter: alvos de toque ≥ 48px,
foco de teclado visível, contraste AA e pistas que não dependam só de cor.

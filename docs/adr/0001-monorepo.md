# ADR 0001 — Monorepo com três componentes

- **Status:** aceito
- **Data:** 2026-11-11

## Contexto

O projeto nasceu como protótipo em Colab e evoluiu em três frentes que compartilham a mesma
ideia: uma **extensão** de navegador (o produto), um **pipeline Python** robusto e testado
(a referência de IA) e uma **CLI** de terminal (laboratório de voz e crawling). Essas frentes
compartilham conceitos (tópicos, rotas da PUC, o classificador `intent.json`).

## Decisão

Manter **um único repositório** (monorepo) com pastas de primeiro nível por componente
(`extension/`, `reia/`, `voice-nav-cli/`), docs compartilhadas em `docs/` e testes em `tests/`.
O classificador de tópicos (`intent.json`) é treinado em um lugar (`voice-nav-cli/training/`) e
copiado para a extensão.

## Consequências

**Positivas**
- Uma fonte de verdade para contexto, rotas e vocabulário; mudanças atômicas entre componentes.
- CI único; documentação e ADRs centralizados.
- Facilita a avaliação da Residência (tudo em um lugar).

**Negativas / mitigação**
- Stacks diferentes (Python + Node + JS de extensão) no mesmo repo → mitigado com `.editorconfig`,
  READMEs por componente e jobs de CI separados.
- Risco de versionar binários grandes → mitigado por regras de `.gitignore` (modelos `*.bin`,
  `node_modules/`, mapas de site gerados).

## Alternativas consideradas
- **Repos separados:** mais isolamento, porém sincronizar `intent.json`, rotas e docs entre
  repositórios seria custoso para uma equipe pequena de residência.

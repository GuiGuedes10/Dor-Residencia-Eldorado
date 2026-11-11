# Changelog

Todas as mudanças notáveis deste projeto são documentadas aqui.
O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o projeto adota [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Não lançado]

### Adicionado
- **Monorepo organizado** com três componentes claros: `reia/` (pipeline Python de
  referência), `voice-nav-cli/` (CLI de terminal) e `extension/` (extensão de navegador).
- Documentação de engenharia: `ARQUITETURA.md`, `ACESSIBILIDADE.md`, ADRs em `docs/adr/`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` e `LICENSE` (MIT).
- Integração contínua (GitHub Actions): testes Python e verificação de sintaxe JS.
- Medidor de confiança no modal da extensão (barra + texto), colorido pela mesma escala.
- Prévia visual de UX em `docs/ux-preview.html`.

### Alterado — UX da extensão (acessibilidade)
- **Modal maior e mais legível:** largura até 460px, mais respiro, título visível,
  botões de opção com altura mínima de 56px e fonte de 19px.
- **Alvos de toque ≥ 48px**, foco de teclado sempre visível (`:focus-visible`),
  contraste reforçado e suporte a `prefers-reduced-motion` e `forced-colors`.
- **Nova escala de cor divergente** do halo-guia, agora semântica de acerto:
  **vermelho = dando errado (longe) → âmbar = quase lá → azul = dando certo (no alvo)**.
  Baseada no ColorBrewer RdYlBu, segura para daltonismo (evita vermelho↔verde).
- A cor deixou de ser a única pista: o halo também **encolhe** ao se aproximar do alvo.

### Corrigido
- Textos e metadados (`manifest.json`, tela de treino) alinhados à nova semântica de cor.
- Números de teste no README corrigidos para refletir a suíte real.

## [0.1.0] — Inicial
- Pipeline modular `reia` (áudio → neural → navegação) com suíte de testes multi-tier.
- Protótipo de CLI e primeira versão da extensão.

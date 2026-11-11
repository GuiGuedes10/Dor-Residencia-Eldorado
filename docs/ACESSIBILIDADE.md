# Acessibilidade e UX — Modal e Guia Visual

Acessibilidade é **requisito**, não enfeite: o REIA existe para ajudar pessoas com baixa visão,
baixa mobilidade ou pouca familiaridade digital. Este documento registra as decisões de UX da
extensão e os critérios WCAG correspondentes.

Prévia visual (abra no navegador): **[ux-preview.html](ux-preview.html)**.

## 1. Modal maior e mais legível

| Item | Antes | Depois | Por quê |
|---|---|---|---|
| Largura | `min(380px, 92vw)` | `min(460px, 94vw)` | Mais espaço para texto e botões |
| Padding | 16px | 22px | Respiro; menos toque acidental |
| Título | só `aria-label` | `<h2>` visível + `aria-labelledby` | Orientação e leitor de tela |
| Texto de status | 15px | 17px | Legibilidade |
| Opções | 18px, 14px de padding | 19px, 16px, **altura ≥ 56px**, numeradas | Alvo de toque e varredura visual |
| Campo de texto | 16px | 17px, **altura ≥ 52px** | Alvo de toque |
| Botões (Falar/OK) | 16px | 17px, **altura ≥ 52px**, com cor | Alvo de toque e affordance |
| Botão flutuante | 64px, opacidade 0.55 | 72px, opacidade 0.92 | Visível e fácil de acertar |

**WCAG atendidos:** 1.4.4 (redimensionar texto), 1.4.10 (reflow até 320px via `vw`),
2.5.5 / 2.5.8 (tamanho de alvo — todos ≥ 44–48px), 1.4.3 (contraste AA do texto sobre os fundos).

## 2. Escala de cor do guia: azul ↔ vermelho

O halo que acompanha o mouse comunica **se você está dando certo**. A escala é **divergente**:

```
 w → 0        w ≈ 0.5         w → 1
🔴 vermelho   🟠 âmbar        🔵 azul
dando errado  quase lá        dando certo
(longe)       (intermediário) (no alvo)
```

Implementação em [`extension/nn.js`](../extension/nn.js) (`color(w)`), interpolando em RGB entre
cinco paradas no espírito do RdYlBu —
`[[211,47,47],[238,138,66],[245,194,66],[120,170,226],[37,109,235]]`
(vermelho → laranja → âmbar → azul-claro → azul). As cinco paradas evitam um meio "oliva"
sujo e fazem a metade superior já tender ao azul (dando certo).

### Por que azul↔vermelho (e não verde↔vermelho)
- **Daltonismo:** vermelho↔verde é indistinguível para ~8% dos homens (deuteranopia/protanopia).
  A dupla **azul↔vermelho com âmbar no meio** segue o padrão **ColorBrewer RdYlBu**, recomendado
  como *colorblind-safe*. As três cores diferem em **matiz e luminância**, não só em matiz.
- **Semântica:** azul lê como "ok/estável" e vermelho como "atenção" em interfaces ocidentais,
  casando com "dando certo / dando errado".

### A cor nunca é a única pista (WCAG 1.4.1)
- O **halo encolhe** conforme você se aproxima (`size = 40 + (1-w)·36`px) — pista redundante de
  tamanho, independente de cor.
- Uma **legenda textual** aparece na faixa de instrução (vermelho=errado · âmbar=quase · azul=certo).
- O **contorno do alvo** acompanha a mesma cor, reforçando qual elemento é o destino.

### Medidor de confiança no modal
Quando a rede de tópicos reconhece a intenção, o modal mostra uma **barra de confiança**
colorida pela **mesma escala** (azul = alta confiança, vermelho = baixa) **com o valor em texto**
("confiança 72%"). Assim o significado da cor é consistente em todo o produto e sempre
acompanhado de número.

## 3. Teclado

- **Alt+V** abre o modal e começa a ouvir.
- **1–4** escolhem uma das opções sugeridas.
- No modo guia: **Enter** abre o link, **Esc** cancela.
- **Foco sempre visível** via `:focus-visible` (contorno de 4px), em botões, campo e opções.
- Ao abrir o modal, o foco vai para o campo de texto; ao listar opções, para a primeira opção.

**WCAG atendidos:** 2.1.1 (tudo operável por teclado), 2.4.7 (foco visível), 2.4.3 (ordem de foco).

## 4. Leitor de tela

- Modal com `role="dialog"`, `aria-modal`, `aria-labelledby` apontando para o título.
- Status com `role="status"` + `aria-live="polite"`: transcrição e sugestões são anunciadas.
- Cada opção tem `aria-label` ("Opção 2: Manual do Candidato").
- Halo e faixa visual são auxiliares de mouse → `aria-hidden` no halo; a faixa é `aria-live`.

## 5. Preferências do sistema

- `prefers-reduced-motion: reduce` → desativa transições e animações.
- `forced-colors: active` (modo de alto contraste do Windows) → bordas e foco redesenhados com
  `CanvasText`/`Highlight`.

## 6. Checklist para novas mudanças de UI

- [ ] Alvo de toque ≥ 48px
- [ ] Foco de teclado visível
- [ ] Contraste de texto ≥ 4.5:1 (AA)
- [ ] Informação não transmitida só por cor
- [ ] Respeita `prefers-reduced-motion`
- [ ] Anunciado a leitor de tela quando relevante

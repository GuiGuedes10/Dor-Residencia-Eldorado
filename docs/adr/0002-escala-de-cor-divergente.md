# ADR 0002 — Escala de cor divergente azul ↔ vermelho para o guia

- **Status:** aceito
- **Data:** 2026-11-11

## Contexto

O modo guia pinta um halo que segue o mouse para indicar se a pessoa está se aproximando do
link-alvo. A versão anterior usava a metáfora "frio→quente" (`hsl` de azul a vermelho, onde
**perto = vermelho**). Isso tinha dois problemas: (1) a semântica era de temperatura, não de
acerto; e (2) variar só o matiz tende a passar por verde, prejudicando daltônicos.

Requisito do produto: **azul = dando certo**, **vermelho = dando errado**, com um **tom
intermediário**, e com acessibilidade em primeiro lugar.

## Decisão

Adotar uma **escala divergente** de cinco paradas interpolada em RGB:

```
vermelho → laranja → âmbar → azul-claro → azul
 w=0        w=0.25    w=0.5    w=0.75       w=1
 errado                quase                certo/no alvo
(211,47,47) (238,138,66) (245,194,66) (120,170,226) (37,109,235)
```

Baseada no **ColorBrewer RdYlBu**, reconhecida como *colorblind-safe*. Cinco paradas (em vez de
três) evitam um ponto médio "oliva" sujo e fazem a metade superior já tender ao azul. A mesma função
(`ReiaNN.color`) passa a colorir também o **medidor de confiança** do modal (azul = alta
confiança), unificando o significado da cor em todo o produto.

Como **cor não pode ser a única pista** (WCAG 1.4.1), adicionamos pistas redundantes:
o halo **encolhe** ao se aproximar e há **legenda textual** na faixa de instrução.

## Consequências

- Significado consistente e intuitivo ("certo/errado") em halo, contorno do alvo e medidor.
- Seguro para as formas comuns de daltonismo (evita vermelho↔verde; cores diferem em
  matiz **e** luminância).
- Exigiu inverter a semântica anterior e atualizar textos (`manifest.json`, tela de treino,
  comentários) — feito.

## Alternativas consideradas
- **Verde↔vermelho:** descartada por não ser acessível a daltônicos.
- **Matiz único (azul→vermelho via HSL):** descartada por passar por verde e ter luminância
  irregular.

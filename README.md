# REIA / Dør — Navegação Web Assistida por Voz e IA

[![CI](https://github.com/GuiGuedes10/Dor-Residencia-Eldorado/actions/workflows/ci.yml/badge.svg)](https://github.com/GuiGuedes10/Dor-Residencia-Eldorado/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/testes-297%20pytest%20·%20160%20standalone-brightgreen.svg)](#-testes)
[![Acessibilidade](https://img.shields.io/badge/foco-acessibilidade%20(WCAG)-8a63d2.svg)](docs/ACESSIBILIDADE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **REIA** (*Reconhecimento de Entradas por Inteligência Artificial*) é uma solução de
> **navegação web assistida por voz** com foco em **acessibilidade**. O sistema **indica e
> orienta** — ele destaca o próximo passo e o elemento-alvo, mas **quem decide e clica é a
> pessoa usuária**. Estudo de caso principal: o fluxo de vestibular da **PUC-Campinas**.
>
> Projeto do **Grupo Dør** — Residência em IA (CPS · PUC-Campinas · **Instituto Eldorado**).

---

## ✨ O que é

Sites institucionais (universidades, IFs, fundações) têm centenas de links — uma barreira
real para quem tem baixa visão, baixa mobilidade ou pouca familiaridade digital. O REIA
reduz esse ruído a uma pergunta simples: **"o que você procura?"** — e devolve **no máximo 4
opções** ranqueadas, além de um **guia visual** que acompanha o mouse até o link certo.

A classificação do que o usuário quer é feita por **redes neurais** (duas, pequenas e locais);
nada é enviado para servidores de terceiros (ver [SECURITY.md](SECURITY.md)).

---

## 🧩 Monorepo: três componentes

Este repositório reúne três implementações complementares da mesma ideia, do protótipo de
pesquisa ao produto final:

| Componente | Pasta | Papel | Stack | Status |
|---|---|---|---|---|
| **Extensão de navegador** | [`extension/`](extension/) | **Produto.** Controle flutuante acessível que lê a página, classifica a intenção e guia o mouse. | JavaScript (Manifest V3), rede neural no navegador | 🟢 Funcional |
| **Pipeline de referência** | [`reia/`](reia/) | **Engenharia/pesquisa.** Pipeline robusto áudio → neural → navegação, orientado a testes. | Python 3.10+, PyTorch (c/ fallback puro), Selenium | 🟢 297 testes |
| **CLI de terminal** | [`voice-nav-cli/`](voice-nav-cli/) | **Protótipo/laboratório.** Mesma lógica no terminal: voz local, crawler de sites e ranking. | Node ≥ 18, `whisper.cpp`, Ollama (opcional) | 🟡 Experimental |

```
                           A pessoa diz / digita o que procura
                                          │
         ┌────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                  ▼
  extension/  (produto)            reia/  (referência)              voice-nav-cli/ (lab)
  Web Speech API  ·  topic net     Validação de áudio  ·  MLP       whisper.cpp  ·  bridge
  halo-guia (nn.js)                4 estágios de rejeição           crawler + ranking
         │                                 │                                  │
         ▼                                 ▼                                  ▼
  destaca ≤4 links + guia        navega (Selenium/mock)            abre a URL / top-4
  a cor do halo mostra            + screenshot/metadados            no terminal
  se está "dando certo"
```

Veja o desenho completo em **[docs/ARQUITETURA.md](docs/ARQUITETURA.md)**.

---

## 🚀 Começando

### 1) Extensão de navegador (o produto)

```bash
# Chrome/Edge/Brave → chrome://extensions → ativar "Modo do desenvolvedor"
#   → "Carregar sem compactação" → selecionar a pasta extension/
```

Depois, em qualquer site: clique no 🎤 (ou **Alt+V**), fale/digite, escolha uma das ≤4 opções
e siga o **halo** com o mouse. Detalhes e treino da rede do halo em
[`extension/README.md`](extension/README.md).

### 2) Pipeline Python (referência, testável)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Navegar a partir de texto (classificado pela rede neural)
python3 -m reia --text "quero ver os cursos oferecidos e o campus"

# A partir de um WAV, ou via microfone, ou em modo simulado (offline)
python3 -m reia --audio-file caminho/audio.wav
python3 -m reia --text "qual a data da prova e o edital" --mock
```

### 3) CLI de terminal (laboratório)

```bash
cd voice-nav-cli && npm install
npm start                 # fluxo guiado (instituição → voz → opções)
npm run doctor            # diagnóstico de microfone/transcrição
npm run map -- https://www.exemplo.edu.br   # mapeia um site novo
```

Requisitos de voz da CLI (`sox`/`ffmpeg`, `whisper.cpp`) estão em
[`voice-nav-cli/README.md`](voice-nav-cli/README.md).

---

## 🧠 As redes neurais

O projeto usa **duas redes pequenas, treináveis e executadas localmente** — mais a rede de
referência do pacote Python:

1. **Classificador de tópicos** (`extension/models/intent.json`, também usado pela CLI):
   MLP *bag-of-words* que mapeia a frase do usuário para um tópico
   (`vestibular`, `cursos`, `extensão`, `concursos`, `bolsas`, `pós`, `contato`).
   Treinado por `voice-nav-cli/training/train.py` a partir de `training/intents.tsv`
   (`npm run train`). **Só biblioteca padrão.**

2. **Rede do "halo"** (`extension/nn.js`): MLP minúsculo (5→8→1) que estima, a cada movimento
   do mouse, o quanto você está **"dando certo"** (perto do alvo). É **treinável pelo próprio
   usuário** em `train.html` e aprende o estilo de movimento da pessoa — útil para quem tem
   tremor ou mira imprecisa.

3. **Rede de intenção de referência** (`reia/neural/`): MLP PyTorch (58→16→3) com
   **4 estágios de rejeição** (saliência lexical, softmax, confiança ≥ 0.65, margem ≥ 0.20 e
   entropia ≤ 0.65) e **fallback em Python puro** via `weights.json`. É a base matemática
   que a CLI consulta para a PUC-Campinas.

> **Por que isso importa para acessibilidade:** o sistema prefere **não decidir** a decidir
> errado. Um comando ambíguo é rejeitado com uma mensagem clara em vez de levar a pessoa para
> a página errada.

---

## 🎨 UX e acessibilidade

O modal foi redesenhado para ser **maior e mais legível** e o guia visual ganhou uma **escala
de cor divergente e semântica**:

- 🔴 **vermelho = dando errado** (longe do alvo)
- 🟠 **âmbar = quase lá** (tom intermediário)
- 🔵 **azul = dando certo** (em cima do alvo)

A escala segue o padrão **ColorBrewer RdYlBu**, **segura para daltonismo** (evita a dupla
vermelho↔verde). E a cor **nunca é a única pista**: o halo também **encolhe** ao se aproximar.
Decisões, critérios WCAG e a prévia visual estão em
**[docs/ACESSIBILIDADE.md](docs/ACESSIBILIDADE.md)** · **[docs/ux-preview.html](docs/ux-preview.html)**.

---

## 🧪 Testes

```bash
pytest -q                     # suíte completa: 297 testes
python3 tests/conftest.py     # runner standalone (sem dependências): 160 testes de tier
```

| Camada | Arquivos | Testes |
|---|---|---|
| Unitários | `test_audio`, `test_neural`, `test_navigation`, `test_pipeline` | inclusos no pytest |
| Tier 1 — funcionalidades | `test_tier1_features.py` | 36 |
| Tier 2 — limites | `test_tier2_boundaries.py` | 48 |
| Tier 3 — combinações | `test_tier3_combinations.py` | 18 |
| Tier 4 — cenários E2E | `test_tier4_scenarios.py` | 12 |
| Tier 5 — adversarial | `test_tier5_adversarial.py` | 46 |

O runner standalone roda em Python puro (sem PyTorch/Selenium), provando o caminho de
*fallback*. O CI roda ambos + checagem de JS ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

---

## 📁 Estrutura do repositório

```
.
├── extension/            # Extensão de navegador (produto) — ver extension/README.md
│   ├── manifest.json     # Manifest V3
│   ├── content.js        # modal acessível + modo guia (halo)
│   ├── nn.js             # rede do halo + escala de cor divergente
│   ├── train.html/.js    # treino da rede do halo pelo usuário
│   └── models/intent.json# classificador de tópicos (versionado)
├── reia/                 # Pipeline Python de referência
│   ├── audio/  neural/  navigation/  pipeline/  core/
├── voice-nav-cli/        # CLI de terminal (Node)
│   ├── src/  training/  bridge/
├── tests/                # Suíte multi-tier (pytest + runner standalone)
├── docs/                 # ARQUITETURA, ACESSIBILIDADE, ADRs, apresentação, prévia de UX
├── notebooks/            # Protótipo original (Colab)
├── .github/              # CI e templates de issue/PR
├── LICENSE · CONTRIBUTING.md · CODE_OF_CONDUCT.md · SECURITY.md · CHANGELOG.md
└── pyproject.toml · requirements.txt
```

---

## 🗺️ Roadmap

**✅ Concluído**
- [x] Pipeline Python modular e testado (áudio → neural → navegação)
- [x] Extensão funcional com modal acessível e guia visual (escala de cor divergente)
- [x] Organização de engenharia (CI, licença, docs, ADRs)

**🎯 Curto prazo — avaliação e dados** _(o que hoje falta para medir se o modelo melhora)_
- [ ] **Métrica de acerto do link (success@1 / success@4):** o link certo apareceu? foi o 1º? É a métrica que mede o produto, não só o classificador.
- [ ] **Avaliação formal do classificador:** matriz de confusão e precisão/recall/F1 por classe — além da acurácia.
- [ ] Dataset de voz real e retreino do classificador de tópicos (hoje ~52 frases)
- [ ] **Loop de correção do usuário → novos exemplos** (aprendizado ativo): registrar quando a pessoa escolhe diferente do sugerido e reusar no treino.
- [ ] **Testes automatizados da extensão** (lógica JS: ranking, cor, `topicOf`) — hoje só o Python tem testes.

**🔧 Médio prazo — robustez e produto**
- [ ] **Robustez a SPAs / conteúdo dinâmico:** `MutationObserver` + re-ranqueamento quando a página muda.
- [ ] **Embeddings de elementos do DOM** (sentence-transformers) no lugar do ranking lexical — fecha a visão original do projeto.
- [ ] Onboarding de primeiro uso + personalização de acessibilidade (tamanho de fonte, velocidade da voz, alto contraste)
- [ ] Mapa verificado por instituição/estado (URLs e seções oficiais)
- [ ] Novos tópicos: cursos de extensão/FIC e concursos (docente/técnico)

**🌐 Longo prazo — distribuição, validação e ética**
- [ ] Publicar na Chrome Web Store / Edge Add-ons + empacotamento e CI de release
- [ ] Testes de acessibilidade com usuários reais (WCAG 2.2 AA), incluindo leitores de tela (NVDA / VoiceOver / TalkBack)
- [ ] **Equidade de fala:** validar sotaques variados, idosos e fala atípica/disfluente
- [ ] **Privacidade / LGPD:** consentimento explícito (microfone) e política de dados
- [ ] Relatório / pôster final da residência com métricas (confusion matrix, success@k)

---

## 🤝 Como contribuir

Leia o [CONTRIBUTING.md](CONTRIBUTING.md) e o [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
Em resumo: issue → branch → commits pequenos ([Conventional Commits](https://www.conventionalcommits.org/)) →
testes passando → PR usando o template.

## 📄 Licença

[MIT](LICENSE) © 2026 Grupo Dør — Residência em IA (CPS · PUC-Campinas · Instituto Eldorado).

# REIA — Voice-Guided Web Navigation Assistant (PUC-Campinas Vestibular)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/architecture-modular%20pipeline-green.svg)](#arquitetura-do-sistema)
[![Tests](https://img.shields.io/badge/tests-343%20passed%20(100%25)-brightgreen.svg)](#executando-os-testes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**REIA** (**R**econhecimento de **E**ntradas por **I**nteligência **A**rtificial) é uma solução em Python estruturada para navegação assistida por voz, projetada para auxiliar candidatos e estudantes no acesso às páginas de vestibular da **PUC-Campinas** como estudo de caso.

O projeto evoluiu um protótipo experimental inicialmente concebido no Google Colab para uma arquitetura modular, desacoplada, orientada a testes e pronta para produção em ambiente local (macOS/Linux/Windows).

---

## Destaques da Solução

- 🎙️ **Tratamento Rigoroso de Restrições de Voz (`reia.audio`)**:
  - Janelas de energia RMS de 50ms para detecção de silêncio preponderante ($R_{\text{silence}} \le 0.85$, $T_{\text{active}} \ge 0.4\text{s}$).
  - Estimativa de relação sinal-ruído por quantis de energia ($\text{SNR} \ge 3.0\text{ dB}$).
  - Limites rígidos de duração de fala útil ($0.8\text{s} \le T \le 12.0\text{s}$).
  - Sanitização de transcrição com decomposição Unicode NFKD, remoção de diacríticos e eliminação de tags acústicas do Whisper (ex: `[Música]`, `(risos)`).

- 🧠 **Classificação Neural com Portão de Rejeição (`reia.neural`)**:
  - Modelo MLP (PyTorch com fallback matricial em NumPy/Python puro) balanceado para as rotas do vestibular da PUC-Campinas.
  - Vetorizador multi-hot Bag-of-Words.
  - Calibrador de confiança dinâmico com **duplo portão**: rejeição de comandos ambíguos com múltiplas intenções concorrentes, filtro de termos fora de domínio e limiares de probabilidade ($\ge 0.65$), margem ($\ge 0.20$) e entropia de Shannon ($\le 0.65$).

- 🌐 **Navegação Web e Evidências Visuais (`reia.navigation`)**:
  - Driver Selenium 4 em modo *headless* local e driver mock para testes herméticos offline.
  - Roteamento determinístico para as páginas canônicas:
    - `GUIA_UNIVERSITARIO` (cursos, campus, infraestrutura)
    - `MANUAL_DO_CANDIDATO` (edital, datas, regras e provas)
    - `FORMULARIO_INSCRICAO` (prazos, taxas e inscrição)
  - Persistência automática de capturas de tela (PNG 1280x800) e metadados estruturados (JSON) no diretório `artifacts/`.

- ⚡ **Pipeline Unificado com Curto-Circuito (`reia.pipeline`)**:
  - Arquitetura *fail-fast*: erros de áudio ou baixa confiança neural abortam o fluxo antes de qualquer navegação externa indevida, emitindo mensagens explicativas em português.
  - Interface CLI completa (`python3 -m reia`).

---

## Arquitetura do Sistema

```
                      [Entrada de Voz: Microfone / Arquivo WAV / Bytes]
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │       reia.audio.validation     │
                             │  • Duração útil: 0.8s a 12.0s   │
                             │  • Razão de silêncio <= 0.85    │
                             │  • SNR estimado >= 3.0 dB       │
                             └────────────────┬────────────────┘
                                              │ (Áudio Válido)
                                              ▼
                             ┌─────────────────────────────────┐
                             │      Transcrição / Sanitização  │
                             │  • Unicode NFKD (sem acentos)   │
                             │  • Remoção de tags acústicas    │
                             └────────────────┬────────────────┘
                                              │ (Texto Sanitizado)
                                              ▼
                             ┌─────────────────────────────────┐
                             │          reia.neural            │
                             │  • Vetorização BoW Multi-hot    │
                             │  • MLP (PyTorch / Fallback)     │
                             │  • Portão de Ambiguidade        │
                             │  • Threshold de Confiança >=0.65│
                             └────────────────┬────────────────┘
                                              │ (Intenção Aprovada)
                                              ▼
                             ┌─────────────────────────────────┐
                             │         reia.navigation         │
                             │  • Selenium Headless Local      │
                             │  • Resolução de Rotas PUC       │
                             │  • Screenshots PNG & JSON Meta  │
                             └─────────────────────────────────┘
```

---

## Estrutura do Repositório

```
reia/
├── .gitignore                   # Regras abrangentes para Python, caches e artefatos
├── pyproject.toml               # Metadados de empacotamento e configuração de ferramentas
├── requirements.txt             # Dependências Python locais
├── README.md                    # Documentação principal e showcase
├── reia/                        # Pacote modular da aplicação
│   ├── __init__.py
│   ├── __main__.py              # Ponto de entrada CLI (python3 -m reia)
│   ├── core/                    # Configurações, limiares e exceções tipadas
│   ├── audio/                   # Validação acústica, fontes de som e sanitização
│   ├── neural/                  # Rede neural MLP, vetorizador e pesos treinados
│   ├── navigation/              # Drivers Selenium headless, mock e artefatos
│   └── pipeline/                # Orquestrador central e interface de comando
├── tests/                       # Suíte completa de testes automatizados
│   ├── conftest.py              # Test harness e runner mestre standalone
│   ├── test_audio.py            # Testes unitários do subsistema de áudio
│   ├── test_neural.py           # Testes unitários da rede neural
│   ├── test_navigation.py       # Testes unitários da navegação web
│   ├── test_pipeline.py         # Testes unitários do pipeline integrado
│   ├── test_tier1_features.py   # Testes de funcionalidades base
│   ├── test_tier2_boundaries.py # Análise de valores limite e fronteiras
│   ├── test_tier3_combinations.py # Combinações pairwise e curto-circuito
│   ├── test_tier4_scenarios.py  # Cenários E2E de jornadas completas
│   └── test_tier5_adversarial.py# Testes de estresse e resiliência adversarial
├── notebooks/                   # Notebooks originais de exploração e prototipagem
│   └── vestibular_puc_poc_original.ipynb
├── docs/                        # Especificações técnicas e registros de arquitetura
│   ├── PROJECT.md               # Especificação detalhada de requisitos e arquitetura
│   ├── TEST_INFRA.md            # Metodologia e infraestrutura dos testes
│   ├── TEST_READY.md            # Certificado e inventário de testes homologados
│   └── contexto_residencia_ia.md# Notas de contexto do Grupo Dør / Residência em IA
└── artifacts/                   # Diretório de persistência de evidências visuais
    └── .gitkeep
```

---

## Instalação e Configuração

### Requisitos
- Python 3.10 ou superior
- Google Chrome instalado (para execução do Selenium em modo headless real)

### 1. Criar e Ativar Ambiente Virtual
```bash
python3 -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
```

### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```

Ou instalar o pacote em modo editável:
```bash
pip install -e .
```

---

## Como Usar a CLI

O REIA fornece uma interface de linha de comando flexível através de `python3 -m reia`:

```bash
# 1. Navegação a partir de texto direto (processado pela rede neural)
python3 -m reia --text "quero ver os cursos oferecidos e informacoes do campus"

# 2. Navegação a partir de arquivo de áudio WAV local
python3 -m reia --audio-file caminho/do/audio.wav

# 3. Navegação via microfone interativo (se disponível no sistema)
python3 -m reia --mic

# 4. Modo headless explícito ou modo simulado para testes rápidos
python3 -m reia --text "qual a data da prova e edital do vestibular" --mock
```

---

## Executando os Testes

O projeto conta com mais de 340 testes automatizados cobrindo testes unitários, testes de limites, combinações e cenários adversariais.

### Execução via Runner Standalone (Sem dependências externas)
```bash
python3 tests/conftest.py
```

### Execução via Unittest
```bash
python3 -m unittest discover tests
```

### Execução via Pytest
```bash
pytest -v
```

---

## Licença

Distribuído sob a licença MIT. Consulte `pyproject.toml` para mais informações.

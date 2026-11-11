# Arquitetura — REIA / Dør

Este documento descreve a arquitetura dos três componentes do monorepo, como eles se
relacionam e os contratos entre as camadas.

## 1. Visão geral

O produto é uma **extensão de navegador**. O pacote **Python (`reia/`)** é a implementação de
referência da inteligência — robusta, testada e usada como "fonte da verdade" matemática. A
**CLI (`voice-nav-cli/`)** é um laboratório de terminal que reaproveita a mesma lógica e serve
para mapear sites e experimentar voz local.

```
            ┌──────────────────────────── MESMA IDEIA, 3 EXECUÇÕES ────────────────────────────┐
            │                                                                                    │
   extension/  (produto, roda no navegador)        reia/  (referência, Python, testada)
   ───────────────────────────────────────         ──────────────────────────────────────
   voz: Web Speech API (pt-BR)                      áudio: validação acústica (RMS/SNR/silêncio)
   intenção: topic net (intent.json)                intenção: MLP 58→16→3 + 4 portões de rejeição
   guia: halo com rede "calor" (nn.js)              navegação: Selenium headless / mock
   saída: ≤4 links + halo colorido                  saída: página + screenshot + metadados JSON
            │                                                         ▲
            │   models/intent.json  ◀── treinado por ──  voice-nav-cli/training/train.py
            │                                                         │
            └────────────── voice-nav-cli/ (CLI) ── bridge/intent.py ─┘  (consulta reia/neural)
```

## 2. Componente `extension/` (produto)

Extensão **Manifest V3**, sem build e sem dependências externas. Toda a UI vive num
**Shadow DOM** para não colidir com o CSS do site.

### Fluxo
1. **Ativação** — botão flutuante (FAB) ou atalho **Alt+V**.
2. **Entrada** — voz (`webkitSpeechRecognition`, pt-BR) ou texto.
3. **Classificação de tópico** — `topicOf()` roda o MLP de `models/intent.json` no navegador
   e devolve `{ topic, confidence }`.
4. **Ranking de links** — `rankLinks()` coleta `<a href>` visíveis da página, expande a
   consulta com sinônimos do tópico e pontua por interseção de tokens (texto + caminho da URL),
   com leve bônus para links em `nav`/`header`. Retorna o **top-4**.
5. **Guia visual** — ao escolher uma opção, entra o **modo guia**: um *halo* segue o mouse e a
   cada movimento a rede de "calor" (`nn.js`) estima `w ∈ [0,1]` (perto do alvo). A cor do halo
   e do contorno do alvo vem de `ReiaNN.color(w)` (ver [ACESSIBILIDADE.md](ACESSIBILIDADE.md)).
6. **Ação** — **Enter** abre o link, **Esc** cancela. O sistema nunca clica sozinho.

### Arquivos
| Arquivo | Responsabilidade |
|---|---|
| `manifest.json` | Permissões (`storage` apenas), content scripts, página de opções |
| `content.js` | UI (modal acessível), classificação, ranking, modo guia |
| `nn.js` | Rede do halo: `features`, `predict`, `train`, `accuracy`, **`color`** |
| `train.html` / `train.js` | Coleta de dados e treino da rede do halo, no navegador |
| `models/intent.json` | Pesos do classificador de tópicos (versionado) |

### Rede do "calor" (`nn.js`)
- **Features (5):** distância normalizada à borda do alvo, alinhamento da direção do movimento,
  velocidade de aproximação, velocidade e "dentro do alvo?".
- **Modelo:** MLP 5→8→1, ReLU + sigmoide, treino por SGD com entropia cruzada binária.
- **Sem treino:** heurística "perto = quente". **Com treino:** aprende o movimento da pessoa.
- Pesos ficam em `chrome.storage.local` (`warmthModel`); dados de treino em `warmthData`.

## 3. Componente `reia/` (pipeline de referência)

Pacote Python modular. Arquitetura *fail-fast*: erros de áudio ou baixa confiança abortam o
fluxo **antes** de qualquer navegação.

```
[Voz: microfone / WAV / bytes]
        │
        ▼
reia.audio.validation ──► duração 0.8–12.0s · silêncio ≤ 0.85 · SNR ≥ 3.0 dB
        │ (válido)
        ▼
transcrição + reia.audio.sanitization ──► NFKD, sem acentos, sem tags acústicas
        │
        ▼
reia.neural ──► vetor multi-hot → MLP 58→16→3 → softmax
        │        portões: saliência lexical · confiança ≥ 0.65 · margem ≥ 0.20 · entropia ≤ 0.65
        │ (intenção aprovada)
        ▼
reia.navigation ──► Selenium headless / MockNavigator → screenshot PNG + metadados JSON
        │
        ▼
reia.pipeline.coordinator ──► orquestra tudo, mensagens em pt-BR, nunca "crasha"
```

### Módulos
| Pacote | Conteúdo |
|---|---|
| `reia.core` | `config` (limiares, URLs), `exceptions` tipadas |
| `reia.audio` | `validation`, `sources` (arquivo/bytes/mic), `sanitization`, `synthetic` (testes) |
| `reia.neural` | `model` (MLP + fallback), `vectorizer` (BoW), `dataset`, `thresholding`, `persistence`, `semantic_search` |
| `reia.navigation` | `base` (adapter), `selenium_driver`, `mock_driver`, `routes`, `artifacts` |
| `reia.pipeline` | `coordinator` (E2E), `cli`, `messages` |

### Rotas (PUC-Campinas)
`GUIA_UNIVERSITARIO` · `MANUAL_DO_CANDIDATO` · `FORMULARIO_INSCRICAO` — mapeadas em
`reia/navigation/routes.py`. A navegação só ocorre se `is_valid_intent == True` **e** houver
`target_url`.

### Persistência dupla do modelo
PyTorch (`checkpoint.pt`, opcional) **e** `weights.json` (versionado) para *forward pass* em
Python puro — é isso que faz o runner standalone e o CI rodarem sem instalar PyTorch.

## 4. Componente `voice-nav-cli/` (CLI)

Fluxo de terminal, acessível (todo menu tem ≤ 4 linhas; "Mais opções" pagina o resto):

1. **Instituição** → PUC (campus) ou UF/IF (região → estado → instituição).
2. **Voz** com confirmação: grava com `sox`, transcreve com `whisper.cpp`, pergunta
   *"Entendi: … Está certo? Sim / Repetir / Digitar"*.
3. **PUC-Campinas:** consulta a rede de referência via `bridge/intent.py` (`neuralIntent`).
4. **Outros sites:** `topicIntent` (mesmo `intent.json`) + `sitemap.js` (crawler 2 níveis,
   cache em `data/`) + `rank.js` (embeddings via Ollama com *fallback* por palavras-chave).
5. Abre a URL (`open`) ou mostra o top-4.

Comandos: `npm start`, `npm run doctor` (microfone), `npm run map -- <url>`, `npm run train`.

## 5. Contratos entre camadas (Python)

```python
# audio → neural
sanitize_transcription(text: str) -> str      # NFKD, sem diacríticos/pontuação

# neural → navigation
@dataclass
class ClassificationResult:
    is_valid_intent: bool
    intent_name: Optional[str]   # GUIA_UNIVERSITARIO | MANUAL_DO_CANDIDATO | FORMULARIO_INSCRICAO
    target_url: Optional[str]
    confidence: float
    rejection_code: Optional[str]  # OOD_NO_DOMAIN_KEYWORDS | LOW_CONFIDENCE | AMBIGUOUS_INTENT
    user_message: str

# navigation → pipeline
@dataclass
class NavigationResult:
    success: bool
    intent_name: str
    target_url: str
    screenshot_path: str   # sempre em artifacts/screenshots/
    metadata_path: str     # sempre em artifacts/metadata/
```

## 6. Dados e modelos

| Artefato | Origem | Versionado? |
|---|---|---|
| `extension/models/intent.json` | `npm run train` (BoW) | ✅ sim (pequeno) |
| `reia/neural/weights.json` | treino do MLP de referência | ✅ sim (fallback) |
| `voice-nav-cli/models/ggml-*.bin` | download do `whisper.cpp` | ❌ não (141 MB) |
| `voice-nav-cli/data/*.json` | `npm run map` (crawler) | ❌ não (regenerável) |
| `warmthModel` / `warmthData` | treino da rede do halo | em `chrome.storage.local` |

## 7. Decisões de arquitetura
Registradas como ADRs em [`docs/adr/`](adr/):
[0001 — Monorepo](adr/0001-monorepo.md) ·
[0002 — Escala de cor divergente](adr/0002-escala-de-cor-divergente.md) ·
[0003 — IA no dispositivo](adr/0003-ia-no-dispositivo.md).

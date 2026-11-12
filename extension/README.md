# Extensão Dør — navegação por voz acessível

Controle flutuante que lê a página, entende o que você procura e **guia o mouse** até o link
certo. Roda inteiramente no navegador (Manifest V3), **sem servidor** e pedindo apenas a
permissão `storage`.

## Instalar (modo desenvolvedor)

1. Abra `chrome://extensions` (Chrome, Edge ou Brave).
2. Ative **Modo do desenvolvedor**.
3. Clique em **Carregar sem compactação** e selecione esta pasta `extension/`.
4. (Opcional) fixe a extensão na barra.

## Usar

- Clique no **🎤** (canto inferior direito) ou pressione **Alt+V**.
- **Fale** ou **digite** o que procura (ex.: *"quero me inscrever no vestibular"*).
- Escolha uma das **até 4 opções** (mouse, toque ou teclas **1–4**).
- Siga o **halo** com o mouse até o alvo:
  🔴 vermelho = longe/errado · 🟠 âmbar = quase · 🔵 azul = no alvo/certo.
  O halo **encolhe** ao chegar perto. **Enter** abre · **Esc** cancela.

> A extensão **indica e orienta** — ela nunca clica nem envia nada por você.

## Como ela escolhe o link (ranqueamento semântico)

Quando você diz o que procura, a extensão ordena os links da página por **similaridade de
significado**, não por palavras iguais — com um ranqueador **TF-IDF de n-gramas de caractere**
([`tfidf.js`](tfidf.js)) que roda **no navegador, sem modelo nem download**. Cada link é
representado por `texto + aria-label + title + heading da seção + caminho da URL`, o que resolve
botões com rótulo pobre ("Saiba mais").

Validado com links reais da PUC ([`../prototipos/classificador-elementos/`](../prototipos/classificador-elementos/)):
**56% de acerto no topo‑1 vs 36% da interseção de palavras** (68% no top‑3). Demo interativa:
[`../docs/extension-ranker-demo.html`](../docs/extension-ranker-demo.html).

> Sinônimos puros (*advogado* → Direito) são o limite do ranqueador leve. A **fase 2** (*experimental*)
> refina com **embeddings neurais** (`sentence-transformers` via **Transformers.js** num documento offscreen,
> 68% no protótipo): o TF-IDF aparece na hora e, quando o modelo responde, as opções são **reordenadas**.
> Para habilitar: `sh vendor/baixar.sh` (baixa a lib + ONNX Runtime, ~22 MB, não versionados) e recarregue a
> extensão; o modelo (~100 MB) baixa do HF na 1ª consulta. Sem isso, fica no TF-IDF — nada quebra.
> **Confirmado funcionando no Chrome.** Para a loja, versione os arquivos de `vendor/` (lojas não aceitam código remoto).

## As duas redes neurais

### 1. Classificador de tópicos — `models/intent.json`
MLP *bag-of-words* que mapeia a frase para um tópico (`vestibular`, `cursos`, `extensão`,
`concursos`, `bolsas`, `pós`, `contato`). É treinado fora da extensão:

```bash
cd ../voice-nav-cli
# edite training/intents.tsv  (uma linha por exemplo:  rótulo <TAB> frase)
npm run train                 # gera models/intent.json
cp models/intent.json ../extension/models/intent.json
```

### 2. Rede do "halo" — `nn.js` (treinável por você)
Estima, a cada movimento, o quanto você está "dando certo". Abra a **página de opções** da
extensão (ou `train.html`) e:

1. Clique **no alvo amarelo** de forma natural, várias vezes (30+ tentativas).
2. Clique em **Treinar com os dados coletados**.
3. O modelo fica salvo em `chrome.storage.local` e passa a colorir o halo nos sites.

Isso personaliza o guia para o seu jeito de mover o mouse — útil para tremor ou mira imprecisa.
Nada sai do navegador; dá para **Exportar dados (JSON)** e **Apagar dados** quando quiser.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `manifest.json` | Manifest V3; permissões, content scripts, página de opções |
| `content.js` | Modal acessível + ranqueamento semântico dos links + modo guia (halo) |
| `tfidf.js` | Ranqueador semântico on-device (TF-IDF char-ngram, `ReiaRank.rank`) |
| `background.js` · `offscreen.html` · `offscreen.js` | Fase 2: modelo de embeddings neurais (Transformers.js) + relé |
| `nn.js` | Rede do halo (`features`/`predict`/`train`) e a **escala de cor** `color(w)` |
| `train.html` · `train.js` | Coleta e treino da rede do halo |
| `models/intent.json` | Pesos do classificador de tópicos |

## Acessibilidade

Modal grande, alvos de toque ≥ 48px, foco de teclado visível, contraste AA, escala de cor
segura para daltonismo e pistas além da cor. Detalhes e critérios WCAG em
[`../docs/ACESSIBILIDADE.md`](../docs/ACESSIBILIDADE.md).

## Privacidade

Voz pela Web Speech API do navegador; dados de treino do halo só em `chrome.storage.local`.
Suas consultas e os links da página são processados **localmente** — nada é enviado a servidores
nossos, sem telemetria. Exceção (Fase 2): o **modelo** de embeddings é baixado uma vez do Hugging
Face e fica em cache no navegador; a inferência roda on-device. Ver [`../SECURITY.md`](../SECURITY.md).

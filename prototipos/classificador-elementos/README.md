# Protótipo — Classificador de Elementos (por embeddings)

Ataca o **núcleo da visão da residência** ([contexto](../../docs/contexto_residencia_ia.md)):

> _"A rede neural classifica os elementos HTML extraídos → qual botão/link é o próximo passo."_

Hoje a extensão escolhe o link por **interseção de palavras** (`rankLinks`), que quebra em
sinônimos ("quero ser **advogado**" ≠ "**Direito**"). Aqui estão o **contrato**, os
**ranqueadores** (lexical vs. embeddings), um **corpus de elementos reais da PUC** e um
**harness de avaliação por `success@k`**.

## Contrato (entrada → saída)

```
Entrada:
  intenção : str                                   # ex.: "quero ser advogado"
  elementos: [{texto, url, contexto}]              # elementos da página (DOM)
Saída:
  ranking  : [(elemento, score)]                   # ordenado: o 1º é o alvo mais provável
```

`contexto` é o **enriquecimento** recomendado pela pesquisa (ver REFERENCIAS): `aria-label`,
`title`, heading da seção e palavras do `href`. Resolve o caso real de botões com texto pobre
("Saiba mais", "clique aqui"), comum em portais universitários.

## Arquivos

| Arquivo | Papel |
|---|---|
| [`ranqueadores.py`](ranqueadores.py) | Baseline lexical + `TfidfEmbedder` (palavras [+ char n-gram]) + adaptador `SentenceTransformerEmbedder` (opcional). Usa `texto + contexto + href`. |
| [`avaliar.py`](avaliar.py) | Harness `success@1/@3` + MRR; aceita um dataset por argumento; pula alvos ausentes do corpus. |
| [`elementos_cursos.json`](elementos_cursos.json) | 64 links reais de cursos (crawl do portal do vestibular). |
| [`elementos_portal.json`](elementos_portal.json) | 15 elementos do **fluxo transacional** (inscrição, cartão, LIBRAS, bolsas, FIES, crédito, atendimento…), lidos das páginas públicas em 2026-11-11, com `contexto`. |
| [`dataset_pesquisa.json`](dataset_pesquisa.json) | 32 pares reais `intenção→alvo` (da pesquisa web, verificada). |
| [`dataset.json`](dataset.json) | 20 casos curados à mão (sanity-check). |
| [`REFERENCIAS.md`](REFERENCIAS.md) | Datasets/papers da pesquisa **verificados na web**, com status e licenças. |
| [`fine_tune.py`](fine_tune.py) | Experimento de fine-tune (`MultipleNegativesRankingLoss`, loop manual): zero-shot vs adaptado. |
| [`montar_dataset.py`](montar_dataset.py) · [`normalizar_coleta.py`](normalizar_coleta.py) · [`coleta.csv`](coleta.csv) | Pipeline de coleta: normaliza a pesquisa web → corpus PUC, valida e gera o split treino/teste. |
| [`exportar_onnx.py`](exportar_onnx.py) | Treina, salva e exporta o modelo para **ONNX quantizado** (formato da extensão) + verifica. |

## Dados — reais, da web

- **Cursos + portal** foram lidos das páginas públicas da PUC-Campinas.
- **Descoberta honesta:** parte do fluxo transacional é pública (cartão, LIBRAS, bolsa social,
  FIES, crédito, CAA, acessibilidade), mas **isenção de taxa, boleto/2ª via, cotas, convocados e
  matrícula estão atrás de login** — por isso 7 pares do dataset ficam `presente_no_corpus=false`
  (um limite real do crawl público, não uma falha do método).
- Os pares são **paráfrases ancoradas em fontes** (Quero Bolsa, Guia do Estudante, FAQ PUC), não
  logs de usuários — validar depois com voz/paráfrases reais.

## Como rodar

```bash
python3 prototipos/classificador-elementos/avaliar.py dataset_pesquisa.json   # pares reais
python3 prototipos/classificador-elementos/avaliar.py                         # dataset.json (sanity)
pip install sentence-transformers   # adiciona a coluna de embeddings neurais (~0,5 GB)
```

## Resultados

**`dataset_pesquisa.json` — 25 pares reais com alvo no corpus (7 pulados, atrás de login); todos semânticos:**

| Estratégia | succ@1 | succ@3 | MRR |
|---|---|---|---|
| baseline lexical (`rankLinks`) | 36% | 44% | 0.42 |
| TF-IDF (palavras) | 36% | 36% | 0.40 |
| TF-IDF (palavras + char n-gram) | 56% | 68% | 0.63 |
| **sentence-transformers** (MiniLM multilíngue) | **68%** | **84%** | **0.77** |

Leitura: os **embeddings neurais vencem** — quase **dobram** o baseline no succ@1 e acertam
**84% no top-3** (MRR 0.77). A maioria dos 8 erros de top-1 são **quase-acertos** (o alvo caiu em 2º)
ou rótulos discutíveis — ex.: para _"baixar provas antigas e gabaritos"_ o modelo escolheu
**"Provas Anteriores"**, tão ou mais correta que o alvo anotado. Conclusão: **vale adotar embeddings**
no lugar do ranking lexical (idealmente um encoder pt-BR fine-tunado — ver [REFERENCIAS.md](REFERENCIAS.md)).

## Fine-tune (experimento)

[`fine_tune.py`](fine_tune.py) adapta o encoder ao domínio com `MultipleNegativesRankingLoss`
(loop manual, sem depender de `datasets`).

**① Validação LIMPA (a que vale)** — treino e teste são **100 pares reais** da coleta
(`montar_dataset.py`), disjuntos, **sem overlap**:

| | succ@1 | succ@3 |
|---|---|---|
| zero-shot | 40% | 45% |
| **fine-tunado** | **65%** | **80%** |

**+25 pontos de succ@1** (e +35 de succ@3) num teste real held-out — com só 80 pares de treino.
É a evidência sólida: **adaptar o encoder ao domínio funciona**, sem vazamento.

**② Augmentation (exploratório)** — treino com *data augmentation ancorada* e split do próprio
conjunto aumentado: zero-shot 81% → fine-tunado 97% succ@1. Nos pares reais esse número fica
otimista (overlap de mapeamento), por isso o experimento ① é o que conta.

Próximo: mais pares reais ainda (a coleta é repetível) e **exportar o modelo para ONNX** para
plugar na extensão (Fase 2), fechando o ciclo pesquisa → dado → modelo → produto.

## Deploy do modelo na extensão (ONNX)

[`exportar_onnx.py`](exportar_onnx.py) treina nos pares reais, salva o modelo torch e exporta um
ONNX **quantizado int8** (`onnx/model_qint8_arm64.onnx`, ~113 MB) — o formato que a extensão carrega.
A verificação roda em **CPUExecutionProvider** (no macOS o CoreML EP falha em nós do MiniLM; é só
artefato local — a extensão usa ORT **Web/WASM**, não CoreML):

| modelo | succ@1 | succ@3 |
|---|---|---|
| torch fine-tunado | 65% | 75% |
| **ONNX int8** (o que vai p/ a extensão) | **60%** | **75%** |

A quantização int8 custou **−5 pp no succ@1** (1 caso em 20) e manteve o **succ@3 idêntico** — perda
esperada, aceitável para rodar on-device.

**Passos:**
1. **Exporte:** `python3 exportar_onnx.py` (requer `pip install "optimum[onnxruntime]" onnx httpx`).
2. **Vendorize:** copie `modelo-finetunado/` → `extension/vendor/modelo/reia-finetunado/` e
   **renomeie** `onnx/model_qint8_arm64.onnx` → `onnx/model_quantized.onnx` (nome que o
   Transformers.js procura com `{ quantized: true }`). ~113 MB, não versionado (gitignored).
3. **Recarregue** a extensão. O [`offscreen.js`](../../extension/offscreen.js) já **tenta o modelo
   local primeiro** (`vendor/modelo/reia-finetunado/`) e, se ausente, cai no modelo genérico do HF.

> ⚠️ O carregamento do modelo custom na extensão **precisa de teste no Chrome** — o export e o
> success@k são validados em Python, mas o browser é o seu.

## Coletar mais dados reais (pipeline)

O ganho do fine-tune precisa ser validado com **pares reais**, sem overlap com a augmentation.
Pipeline repetível:

1. **Colete:** copie [`coleta_template.csv`](coleta_template.csv) → `coleta.csv` e preencha 1 linha
   por par real (`intencao,alvo,fonte,tipo`). Fontes: equipe, usuários de teste e a pesquisa web.
2. **Monte:** `python3 montar_dataset.py` — valida cada alvo contra o corpus, deduplica e gera
   `dataset_treino.json` / `dataset_teste.json` (split 80/20, **teste só com pares reais**).
3. **Meça:** `python3 avaliar.py dataset_teste.json` e re-rode o `fine_tune.py`.

Mire **100+ pares** variados por abordagem — quanto mais real e diverso, mais confiável o número.
**Já aplicado:** [`coleta.csv`](coleta.csv) = 75 pares da pesquisa web (Quero Bolsa, Guia do Estudante,
Reddit r/enem) normalizados ao corpus PUC + 25 da pesquisa anterior = **100 pares reais**. O
[`normalizar_coleta.py`](normalizar_coleta.py) mapeia os alvos genéricos (ENEM) para os nossos e
descarta o que não tem elemento (isenção, SISU, matrícula — fora do portal PUC).

### Prompt para o Gemini (colher frases reais)

```text
Você é um pesquisador. Pesquise na web, com FONTES (URLs), e devolva um CSV com colunas
intencao,alvo,fonte,tipo contendo 80+ pares REAIS de "como a pessoa pede" → curso/seção do
vestibular, em pt-BR coloquial. Foque em: profissão→curso (dentista→odontologia), dúvidas
operacionais→seção (isenção de taxa, boleto, cartão de confirmação, bolsas, matrícula) e
paráfrases de fóruns (Reddit r/enem, Brainly) e guias (Quero Bolsa, Guia do Estudante).
alvo = trecho do slug do elemento (ex.: curso/direito, odontologia, manual-do-candidato,
vestibular-social, central-de-atendimento). Cite a fonte de cada par e NÃO invente frases.
```

Traga o CSV de volta ao Claude para verificar as fontes e normalizar antes de virar `coleta.csv`.

## Próximos passos

1. **Medir com `sentence-transformers`** (bge-m3 / Serafim / BERTimbau fine-tunado) no mesmo harness.
2. **Crawlear o fluxo logado** (ou obter o mapa oficial) para fechar os 7 pares restantes.
3. **Ampliar o dataset** com FaQuAD-NLI (pt-BR) + paráfrases/voz reais.
4. **Integrar no produto**: trocar o `rankLinks` por este ranqueamento, com `contexto` do DOM
   (o crawler [`sitemap.js`](../../voice-nav-cli/src/sitemap.js) já captura `aria-label`/`title`/href).
5. **Arquitetura de 2 estágios** (ver [REFERENCIAS.md](REFERENCIAS.md): Mind2Web, PageGuide, WebNavigator),
   com bi-encoder quantizado on-device na extensão.

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
| [`montar_dataset.py`](montar_dataset.py) · [`coleta_template.csv`](coleta_template.csv) | Pipeline de coleta: valida pares reais contra o corpus e gera o split treino/teste. |

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

[`fine_tune.py`](fine_tune.py) adapta o encoder ao domínio com `MultipleNegativesRankingLoss`,
usando *data augmentation ancorada* (nome do curso + profissão/ação → elemento) e split 80/20:

| | succ@1 | succ@3 |
|---|---|---|
| zero-shot (teste aumentado, limpo) | 81% | 94% |
| **fine-tunado (teste aumentado)** | **97%** | **100%** |
| zero-shot (pares reais, difícil) | 68% | 84% |
| fine-tunado (pares reais) | 84%\* | 92%\* |

**+16 pontos de succ@1** no teste limpo (held-out 80/20). \*No teste de **pares reais** o ganho é
**otimista**: há overlap de mapeamento entre a augmentation e o teste (ex.: *advogado→Direito* em
ambos), então serve para **direção**, não como medida limpa. Conclusão: **adaptar o encoder ao
domínio vale muito** — o passo seguinte é um **conjunto rotulado real maior** (voz/paráfrases) para
validar sem overlap, e aí exportar o modelo para ONNX e plugar na extensão (Fase 2).

## Coletar mais dados reais (pipeline)

O ganho do fine-tune precisa ser validado com **pares reais**, sem overlap com a augmentation.
Pipeline repetível:

1. **Colete:** copie [`coleta_template.csv`](coleta_template.csv) → `coleta.csv` e preencha 1 linha
   por par real (`intencao,alvo,fonte,tipo`). Fontes: equipe, usuários de teste e a pesquisa web.
2. **Monte:** `python3 montar_dataset.py` — valida cada alvo contra o corpus, deduplica e gera
   `dataset_treino.json` / `dataset_teste.json` (split 80/20, **teste só com pares reais**).
3. **Meça:** `python3 avaliar.py dataset_teste.json` e re-rode o `fine_tune.py`.

Mire **100+ pares** variados por abordagem — quanto mais real e diverso, mais confiável o número.

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

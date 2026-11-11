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
| **TF-IDF (palavras + char n-gram)** | **56%** | **68%** | **0.63** |
| sentence-transformers | _(em medição — instale a lib)_ | | |

Leitura: char n-gram quase **dobra** o baseline e acerta **68% no top-3**. Os ~11 erros de top-1
são sinônimos puros (advogado→Direito, dentista→Odontologia) — o trabalho do `sentence-transformers`.

## Próximos passos

1. **Medir com `sentence-transformers`** (bge-m3 / Serafim / BERTimbau fine-tunado) no mesmo harness.
2. **Crawlear o fluxo logado** (ou obter o mapa oficial) para fechar os 7 pares restantes.
3. **Ampliar o dataset** com FaQuAD-NLI (pt-BR) + paráfrases/voz reais.
4. **Integrar no produto**: trocar o `rankLinks` por este ranqueamento, com `contexto` do DOM
   (o crawler [`sitemap.js`](../../voice-nav-cli/src/sitemap.js) já captura `aria-label`/`title`/href).
5. **Arquitetura de 2 estágios** (ver [REFERENCIAS.md](REFERENCIAS.md): Mind2Web, PageGuide, WebNavigator),
   com bi-encoder quantizado on-device na extensão.

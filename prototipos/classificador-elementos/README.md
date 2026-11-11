# Protótipo — Classificador de Elementos (por embeddings)

Este protótipo ataca o **núcleo da visão da residência** ([contexto](../../docs/contexto_residencia_ia.md)):

> _"A rede neural classifica os elementos HTML extraídos → qual botão/link é o próximo passo."_

Hoje a extensão **não** faz isso: quem escolhe o link é a função `rankLinks` por **interseção de
palavras** (lexical). Isso quebra em sinônimos ("quero ser **advogado**" não casa com "**Direito**").
Aqui montamos o **contrato**, um **dataset rotulado com links reais da PUC-Campinas** e uma
**avaliação por `success@k`** comparando o baseline lexical com rankeamento por **embeddings**.

## Contrato (entrada → saída)

```
Entrada:
  intenção : str                       # ex.: "quero ser advogado"
  elementos: [{texto, url}]            # links/botões extraídos da página (DOM)
Saída:
  ranking  : [(elemento, score)]       # ordenado: o 1º é o alvo mais provável
```

No futuro, `elementos` inclui também `aria-label`, `title`, `classes` e o seletor DOM (`dom_target_selector`),
e a saída alimenta o destaque visual na extensão (substituindo o `rankLinks`).

## Arquivos

| Arquivo | Papel |
|---|---|
| [`ranqueadores.py`](ranqueadores.py) | Baseline lexical + `TfidfEmbedder` (palavras [+ n-gramas de caractere]) + adaptador `SentenceTransformerEmbedder` (opcional) |
| [`gatilhos.json`](gatilhos.json) | Curadoria por curso: sinônimos, profissões e objetivos (dá a riqueza semântica ao dataset) |
| [`gerar_dataset.py`](gerar_dataset.py) | Expande nomes reais dos cursos + gatilhos × modelos de frase → dataset grande, com split treino/teste |
| [`dataset_treino.json`](dataset_treino.json) · [`dataset_teste.json`](dataset_teste.json) | **1265 exemplos** gerados (1008 treino / 257 teste), rotulados `lexical`/`semantico` |
| [`dataset.json`](dataset.json) | Conjunto pequeno curado à mão (20 casos), para sanity-check |
| [`avaliar.py`](avaliar.py) | Resolve o alvo, roda cada estratégia e calcula `success@1`, `success@3` e MRR (usa o teste gerado por padrão) |

## Como rodar

```bash
# 1) (re)gerar o dataset grande a partir dos links reais + gatilhos curados
python3 prototipos/classificador-elementos/gerar_dataset.py

# 2) avaliar (usa dataset_teste.json automaticamente)
python3 prototipos/classificador-elementos/avaliar.py
```

Para o salto semântico real (embeddings neurais multilíngues):

```bash
pip install sentence-transformers   # baixa ~0,5 GB na 1ª vez
python3 prototipos/classificador-elementos/avaliar.py   # agora inclui a coluna sentence-transformers
```

## Dataset (gerado)

A partir dos **64 links reais** e da curadoria em `gatilhos.json`, o `gerar_dataset.py`
produz **1265 exemplos únicos** (568 lexicais + 697 semânticos) sobre 59 cursos/páginas,
com split estratificado **1008 treino / 257 teste**. Exemplos:

- lexical: _"curso de segurança da informação"_ → `ciberseguranca`
- semântico: _"sonho em ser advogado criminalista"_ → `direito`

Para ampliar, acrescente gatilhos em `gatilhos.json` e rode o gerador de novo.

## Resultados (conjunto de teste: 257 casos — 128 lexicais / 129 semânticos)

| Estratégia | succ@1 | succ@3 | MRR | lex@1 | sem@1 |
|---|---|---|---|---|---|
| baseline lexical (`rankLinks` de hoje) | 49% | 53% | 0.54 | 109/128 | 18/129 |
| TF-IDF (palavras) | 53% | 57% | 0.57 | 114/128 | 21/129 |
| **TF-IDF (palavras + n-gramas de caractere)** | **65%** | **69%** | **0.69** | 118/128 | **50/129** |
| sentence-transformers | _(instale a lib)_ | | | | |

**Leitura:**
- Em casos **lexicais**, todas vão bem (a palavra aparece no link).
- Em **semânticos**, o baseline acerta só **18/129 (14%)**; os **n-gramas de caractere** sobem para **50/129 (39%)** e **superam o baseline em +16pp de succ@1**.
- Os ~89 casos que ainda erram o top-1 são sinônimos/paráfrases puros (_"pretendo cozinhar profissionalmente"_ → gastronomia; _"fazer remédios"_ → farmácia) — a **motivação quantificada** para o `sentence-transformers`.

> Dados sintéticos fixam **metodologia** e **baseline**; o próximo passo é validar com paráfrases/voz reais e medir de novo.

## Próximos passos

1. **Instalar `sentence-transformers`** e medir o ganho nos casos semânticos (esperado: resolver os sinônimos).
2. **Ampliar o dataset** para 100+ casos e incluir vozes/paráfrases reais.
3. **Enriquecer os elementos** com `aria-label`, `title` e `classes` (não só o texto visível).
4. **Integrar no produto**: trocar o `rankLinks` lexical ([extension/content.js](../../extension/content.js))
   por este ranqueamento semântico, mantendo o fallback lexical offline.
5. Reaproveitar a infra de [`reia/neural/semantic_search.py`](../../reia/neural/semantic_search.py)
   (embedders + store vetorial com ChromaDB) para indexar o portal inteiro.

Relaciona-se ao roadmap (Médio prazo → _"Embeddings de elementos do DOM"_) e à métrica `success@k`
(Curto prazo → _"Avaliação"_).

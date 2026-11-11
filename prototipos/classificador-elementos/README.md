# Protótipo — Classificador de Elementos (por embeddings)

Este protótipo ataca o **núcleo da visão da residência** ([contexto](../../docs/contexto_residencia_ia.md)):

> _"A rede neural classifica os elementos HTML extraídos → qual botão/link é o próximo passo."_

Hoje a extensão **não** faz isso: quem escolhe o link é a função `rankLinks` por **interseção de
palavras** (lexical). Isso quebra em sinônimos ("quero ser **advogado**" não casa com "**Direito**").
Aqui ficam o **contrato**, os **ranqueadores** (lexical vs. embeddings) e um **harness de avaliação
por `success@k`** — prontos para rodar sobre **qualquer dataset** no formato padrão.

## Contrato (entrada → saída)

```
Entrada:
  intenção : str                       # ex.: "quero ser advogado"
  elementos: [{texto, url}]            # links/botões extraídos da página (DOM)
Saída:
  ranking  : [(elemento, score)]       # ordenado: o 1º é o alvo mais provável
```

No futuro, `elementos` inclui também `aria-label`, `title`, `classes` e o seletor DOM,
e a saída alimenta o destaque visual na extensão (substituindo o `rankLinks`).

## Arquivos

| Arquivo | Papel |
|---|---|
| [`ranqueadores.py`](ranqueadores.py) | Baseline lexical + `TfidfEmbedder` (palavras [+ n-gramas de caractere]) + adaptador `SentenceTransformerEmbedder` (opcional) |
| [`avaliar.py`](avaliar.py) | Roda cada estratégia sobre um dataset e calcula `success@1`, `success@3` e MRR |
| [`dataset.json`](dataset.json) | Conjunto **de exemplo**, pequeno e curado à mão (20 casos), só para sanity-check do harness |

## Dados — vêm de pesquisa na web, não de geração sintética

O dataset de verdade será montado a partir de **pesquisa na internet** (guias de profissões,
FAQs de vestibular, benchmarks e artigos), com **fontes citadas**, avaliado e normalizado para
o formato abaixo. O `avaliar.py` aceita qualquer arquivo nesse formato:

```json
{ "casos": [ {"intencao": "quero ser advogado", "alvo": "curso/direito", "tipo": "semantico"} ] }
```

- `alvo` = trecho distintivo do slug do link correto (o avaliador casa pelo slug mais curto que o contém).
- `tipo` = `lexical` (a palavra aparece no link) ou `semantico` (sinônimo/paráfrase).

## Como rodar

```bash
# sanity-check com o conjunto de exemplo
python3 prototipos/classificador-elementos/avaliar.py

# avaliar um dataset próprio (ex.: o montado a partir da pesquisa)
python3 prototipos/classificador-elementos/avaliar.py meu_dataset.json
```

Para o salto semântico real (embeddings neurais multilíngues):

```bash
pip install sentence-transformers   # baixa ~0,5 GB na 1ª vez
```

A coluna `sentence-transformers` passa a aparecer automaticamente na avaliação.

## Resultados (exemplo — 64 links reais da PUC, 20 casos de sanity-check)

| Estratégia | succ@1 | succ@3 | MRR |
|---|---|---|---|
| baseline lexical (`rankLinks` de hoje) | 45% | 50% | 0.49 |
| TF-IDF (palavras) | 40% | 50% | 0.47 |
| **TF-IDF (palavras + n-gramas de caractere)** | **55%** | **60%** | **0.60** |
| sentence-transformers | _(instale a lib)_ | | |

> Conjunto minúsculo (20 casos): serve só para **validar o harness e a metodologia**.
> Os números que valem virão quando o **dataset real** (da pesquisa na web) estiver montado.

## Próximos passos

1. **Montar o dataset real** a partir de pesquisa na web (fontes citadas), normalizado para
   `{intencao, alvo, fonte, tipo}` — avaliado criticamente antes de entrar.
2. **Instalar `sentence-transformers`** e medir o ganho nos casos semânticos.
3. **Enriquecer os elementos** com `aria-label`, `title` e `classes` (não só o texto visível).
4. **Integrar no produto**: trocar o `rankLinks` lexical ([extension/content.js](../../extension/content.js))
   por este ranqueamento semântico, mantendo o fallback lexical offline.
5. Reaproveitar a infra de [`reia/neural/semantic_search.py`](../../reia/neural/semantic_search.py)
   (embedders + store vetorial com ChromaDB).

Relaciona-se ao roadmap (Médio prazo → _"Embeddings de elementos do DOM"_) e à métrica `success@k`
(Curto prazo → _"Avaliação"_).

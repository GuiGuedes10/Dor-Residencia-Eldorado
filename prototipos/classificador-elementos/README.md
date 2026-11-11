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
| [`dataset.json`](dataset.json) | 20 intenções rotuladas → link-alvo, marcadas como `lexical` ou `semantico` |
| [`avaliar.py`](avaliar.py) | Resolve o alvo, roda cada estratégia e calcula `success@1`, `success@3` e MRR |

## Como rodar

```bash
python3 prototipos/classificador-elementos/avaliar.py
```

Para o salto semântico real (embeddings neurais multilíngues):

```bash
pip install sentence-transformers   # baixa ~0,5 GB na 1ª vez
python3 prototipos/classificador-elementos/avaliar.py   # agora inclui a coluna sentence-transformers
```

## Resultados (64 links reais da PUC, 20 casos)

| Estratégia | succ@1 | succ@3 | MRR | lexicais | semânticos |
|---|---|---|---|---|---|
| baseline lexical (`rankLinks` de hoje) | 45% | 50% | 0.49 | 6/6 | 3/14 |
| TF-IDF (palavras) | 40% | 50% | 0.47 | 6/6 | 2/14 |
| **TF-IDF (palavras + n-gramas de caractere)** | **55%** | **60%** | **0.60** | 6/6 | **5/14** |
| sentence-transformers | _(instale a lib)_ | | | | |

**Leitura:**
- Em casos **lexicais** (a palavra aparece no link), todas acertam — o baseline já basta.
- Em casos **semânticos** (sinônimo/paráfrase), o baseline acerta só **3/14**.
- Os **n-gramas de caractere** já sobem para **5/14** (pegam morfologia: médico↔medicina, jornalista↔jornalismo) e **superam o baseline** no geral.
- Sinônimos puros (advogado→direito, dentista→odontologia, chef→gastronomia, saúde mental→psicologia)
  **continuam falhando** sem embeddings neurais — é a **motivação quantificada** para o `sentence-transformers`.

> Números são de um dataset pequeno (20 casos): servem para fixar a **metodologia** e um **baseline**,
> não como resultado final. O próximo passo é ampliar o dataset e medir de novo.

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

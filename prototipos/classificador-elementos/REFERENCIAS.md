# Referências verificadas — dados e estado da arte

Curadoria da pesquisa (Gemini Pro) **verificada pelo Claude**. Legenda de status:
✅ **web** = conferido na web nesta sessão · ✅ **conhecido** = referência consolidada (pré-2026) ·
⚠️ = existe, mas com ressalva/correção.

## Datasets e benchmarks

| Dataset | Link | Licença | Status | Uso no projeto |
|---|---|---|---|---|
| **FaQuAD-NLI** (pt-BR) | huggingface.co/datasets/ruanchaves/faquad-nli | CC BY 4.0 | ✅ web | **Melhor match pt-BR**: perguntas reais sobre ensino superior BR. Fine-tuning no domínio. |
| **Mind2Web** | huggingface.co/datasets/osunlp/Mind2Web | CC BY 4.0 (dados) / MIT (código) | ✅ web | Ranqueamento intenção→elemento DOM em sites reais. Treino = 1.009 ex.; teste em zip à parte. |
| **WebLINX** | huggingface.co/datasets/McGill-NLP/WebLINX | Apache 2.0 (código) / fair-use pesquisa (dados) | ✅ conhecido | Navegação multi-turno + reranking de elementos HTML. |
| **WebSRC** | github.com/X-LANCE/WebSRC-Baseline | Apache 2.0 (código) | ✅ conhecido | QA → `element_id` no HTML; ~400k pares. |
| **MiniWoB++** | github.com/Farama-Foundation/miniwob-plusplus | MIT | ✅ conhecido | Ambientes sintéticos p/ extração de features do DOM. |
| **ASSIN 2** (pt-BR) | huggingface.co/datasets/nilc-nlp/assin2 | não declarada (uso acadêmico) | ✅ conhecido | Calibração de similaridade semântica (STS) em pt-BR. |
| **Amazon MASSIVE** | huggingface.co/datasets/AmazonScience/massive | CC BY 4.0 | ✅ conhecido | ⚠️ o "pt" é **pt-PT**, não pt-BR — só pré-treino de intenções. |
| **AccessGuru** | github.com/NadeenAhmad/AccessGuruLLM | DOI DARUS-5177 (checar termos) | ✅ web ⚠️ | ⚠️ paper real = **ASSETS 2025** (arXiv:2507.19549); ~**3.500** violações / **448** URLs (Gemini disse 3.524/588). |
| **Widget Captioning / Screen2Words** | huggingface.co/datasets (RICO) | CC BY 4.0 | ✅ conhecido | ⚠️ UI **mobile** (RICO), não web — transferência parcial p/ ícones sem texto. |

## Artigos (estado da arte)

| Título | Ano | arXiv | Status | Relevância |
|---|---|---|---|---|
| **PageGuide: Browser extension to assist users in navigating a webpage** | 2026 | 2604.23772 | ✅ web ⚠️ | **Análogo direto ao projeto** (Auburn Univ.): destaca o elemento DOM, modo guia passo-a-passo. Estudo **N=94** (Gemini disse 92). É a prior art a citar. |
| **WebNavigator: Global Web Navigation via Interaction Graph Retrieval** | 2026 | 2603.20366 | ✅ web | Crawl offline → grafo de interação (DOM+acessibilidade) em banco vetorial → recuperação. NeurIPS 2026. |
| **Mind2Web: Towards a Generalist Agent for the Web** | 2023 | 2306.06070 | ✅ conhecido | Arquitetura de 2 estágios (bi-encoder p/ filtrar nós + predição de ação). |
| **WebLINX** | 2024 | 2402.05930 | ✅ conhecido | Modelos pequenos de reranking superam LLMs genéricos na seleção do elemento. |
| **ScreenAI** | 2024 | 2402.04615 | ✅ conhecido | VLM p/ UI; árvore de acessibilidade + visão. |
| **Ferret-UI** | 2024 | 2404.05719 | ✅ conhecido | *referring* (função do widget) + *grounding* (achar elemento). |
| **Global vs. Local Sentence Embeddings for BR Portuguese** (Peixoto et al.) | 2026 | PROPOR 2026 (ACL 2026.propor-1.52) | ✅ web ⚠️ | ⚠️ conclusão **mais nuançada** que o Gemini pintou: monolíngues adaptam melhor no fine-tuning, **mas tokenizadores multilíngues modernos já lidam bem com morfologia**; "o melhor depende da tarefa". Números 0.847/0.834 **não confirmados**. |

## Modelos de embedding pt-BR (para similaridade intenção ↔ elemento)

- **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) — ✅ conhecido. ⚠️ o base só fez MLM: **não** gera bons embeddings de sentença sem fine-tuning.
- **Serafim** (`PORTULAN/serafim-100m/335m`), **Albertina-900M** — monolíngues pt, bons p/ fine-tuning.
- **BAAI/bge-m3** — multilíngue forte, bom p/ zero-shot.
- Técnica recomendada: **MultipleNegativesRankingLoss (InfoNCE)** sobre pares *(intenção coloquial, elemento serializado)* — treino contrastivo assimétrico.

## Correções/ressalvas encontradas na verificação

1. **PageGuide**: estudo com **N=94** (não 92); sem validação independente/código público (números são dos autores).
2. **AccessGuru**: é de **2025 (ASSETS)**, não 2026; **~3.500** violações / **448** URLs.
3. **Peixoto/PROPOR 2026**: monolíngue **não** vence sempre — depende da tarefa; multilíngue moderno lida bem com morfologia. Não citar os números exatos sem ler o PDF.
4. **MASSIVE** é **pt-PT**. **Yahoo Respostas** encerrado (2021, sem dump); **Brainly/Google PAA** sem API pública — o Gemini acertou ao não prometer dumps dessas fontes.

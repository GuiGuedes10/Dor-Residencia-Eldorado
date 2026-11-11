# Comando para Antigravity — Agentes para o Projeto Dør

## Prompt para Antigravity (Pesquisa com Agentes)

```
Você é um agente de pesquisa técnica para o projeto "Classificação de Tags HTML para Navegação Assistida" do Grupo Dør (Residência em IA — CPS / PUC-Campinas / Instituto Eldorado).

## Contexto do Projeto

Estamos construindo uma extensão de navegador que ajuda usuários a navegar o site da PUC-Campinas (foco: fluxo de inscrição no vestibular). O pipeline tem 5 etapas:

1. Usuário acessa o site da PUC-Campinas
2. Extensão de navegador lê o DOM e extrai botões/elementos interativos
3. Rede neural (TensorFlow) classifica os elementos extraídos
4. Agente de IA usa a classificação para construir orientações
5. Usuário recebe orientação visual (próximo passo + tag alvo destacada)

A rede neural recebe como entrada:
- Texto dos botões/elementos interativos extraídos do HTML (transformados em embeddings via sentence-transformers)
- Intenção do usuário (ex: "quero me inscrever no vestibular")

A saída é: próximo passo + tag HTML alvo a ser visualmente modificada.

O agente INDICA e ORIENTA, não decide — foco em acessibilidade.

## Stack Técnica
- TensorFlow para a rede neural
- sentence-transformers para embeddings
- ChromaDB para armazenamento vetorial
- Google Colab para notebooks
- Extensão Chrome (JavaScript) para leitura do DOM
- Protótipo existente: Qwen 2.5-7B (4-bit), Redis, Whisper-small STT, RAG com 25 PDFs, Gradio

## Suas Tarefas como Agente

1. **Pesquisa de artigos**: Encontre artigos científicos sobre uso de redes neurais para classificação de elementos HTML e/ou acessibilidade web. Priorize artigos que combinem deep learning + web accessibility + browser automation.

2. **Referência técnica para embeddings de HTML**: Pesquise as melhores práticas para gerar embeddings de elementos HTML (texto de botões, atributos aria, classes CSS). Compare sentence-transformers vs. alternativas para esse caso específico.

3. **Arquitetura da rede neural**: Sugira uma arquitetura de rede neural adequada para classificação multi-classe de tags HTML, considerando:
   - Input: embedding do texto do botão + embedding da intenção do usuário
   - Output: classe da tag alvo (qual botão clicar)
   - Dataset pequeno inicialmente (~50-200 exemplos)
   - Deve rodar leve (será integrado com LLM + embedding model)

4. **Extensão Chrome — DOM extraction**: Pesquise exemplos e boas práticas para extrair elementos interativos do DOM via extensão Chrome. Como filtrar apenas botões, links e elementos clicáveis? Como lidar com SPAs e conteúdo dinâmico?

5. **ChromaDB + RAG pipeline**: Como integrar o ChromaDB com o pipeline de embeddings para armazenar e buscar os elementos HTML classificados?

Para cada tópico, forneça: fontes, código de exemplo quando possível, e recomendações específicas para o nosso caso.
```

---

## Estrutura de Pastas — Google Drive

```
📁 Grupo Dør — Residência IA/
│
├── 📁 01_Documentação/
│   ├── 📄 README.md (visão geral do projeto)
│   ├── 📄 Ata de Reuniões/
│   │   ├── 2026-11-04_reuniao.md
│   │   └── 2026-11-05_reuniao.md
│   ├── 📄 Decisões Técnicas.md
│   └── 📄 Competências e Entregas.md
│
├── 📁 02_Apresentações/
│   ├── 📄 Desafio 1 — Semanas 1-2.pdf
│   └── 📄 Desafio 2 — Semanas 3-4.pdf
│
├── 📁 03_Dados/
│   ├── 📁 PDFs_Institucionais/ (os 25 PDFs do RAG)
│   ├── 📁 HTML_Extraido/ (páginas HTML salvas do site PUC)
│   ├── 📁 Labels/ (dados anotados: botão → ação correta)
│   └── 📄 dataset_botoes_vestibular.csv
│
├── 📁 04_Notebooks/
│   ├── 📄 01_extracao_botoes_html.ipynb
│   ├── 📄 02_geracao_embeddings.ipynb
│   ├── 📄 03_treino_rede_neural.ipynb
│   ├── 📄 04_avaliacao_metricas.ipynb
│   └── 📄 05_integracao_agente.ipynb
│
├── 📁 05_Modelos/
│   ├── 📁 Checkpoints/ (pesos salvos durante treino)
│   ├── 📁 Final/ (modelo treinado pronto para uso)
│   └── 📄 config_modelo.json
│
├── 📁 06_Extensão_Chrome/
│   ├── 📄 manifest.json
│   ├── 📄 content.js (extração do DOM)
│   ├── 📄 popup.html
│   └── 📄 background.js
│
├── 📁 07_Agente/
│   ├── 📄 agente_orquestrador.py
│   ├── 📄 rag_pipeline.py
│   └── 📄 config.yaml
│
├── 📁 08_Referências/
│   ├── 📄 Artigos/ (papers relevantes)
│   ├── 📄 Tutoriais/ (links e anotações)
│   └── 📄 tutorial_navegacao_vestibular.md
│
└── 📁 09_Resultados/
    ├── 📄 metricas_treino.png
    ├── 📄 confusion_matrix.png
    └── 📄 relatorio_final.md
```

### Notas sobre a estrutura

- **03_Dados/Labels/**: Aqui vão os dados anotados manualmente — cada par (botão, intenção) → ação correta. Comece com o fluxo de vestibular.
- **04_Notebooks/**: Numerados na ordem do pipeline. Cada notebook é autocontido e pode ser rodado no Colab.
- **05_Modelos/**: Separe checkpoints (durante treino) de modelos finais. O `config_modelo.json` registra hiperparâmetros.
- **08_Referências/tutorial_navegacao_vestibular.md**: Se os 25 PDFs não cobrirem vestibular, crie este tutorial como ground truth para treino.

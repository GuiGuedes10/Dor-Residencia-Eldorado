# ADR 0003 — Inteligência no dispositivo (on-device)

- **Status:** aceito
- **Data:** 2026-11-11

## Contexto

O público-alvo inclui pessoas em situação de vulnerabilidade digital. Enviar voz e navegação
para serviços externos levantaria questões de privacidade, custo e dependência de conexão.

## Decisão

Processar **tudo localmente**:

- **Extensão:** classificador de tópicos e rede do halo rodam no navegador (JS puro). Voz pela
  Web Speech API do próprio navegador. Dados/modelos só em `chrome.storage.local`.
- **CLI:** STT com `whisper.cpp` e LLM/embeddings via Ollama — tudo local, sem chaves de API.
- **Pipeline Python:** execução local (Selenium headless local ou driver mock) e **fallback em
  Python puro** (`weights.json`) para não exigir PyTorch onde não for possível.

## Consequências

**Positivas**
- Privacidade por padrão (ver [SECURITY.md](../../SECURITY.md)); funciona offline; sem custo por uso.
- O *fallback* puro permite rodar testes e CI sem instalar PyTorch/Selenium.

**Negativas / mitigação**
- Modelos pequenos têm acurácia limitada → mitigado priorizando **rejeição segura** (melhor não
  decidir do que decidir errado) e coleta de mais dados no roadmap.
- Qualidade de voz depende do navegador/sistema → a CLI oferece `npm run doctor` e a extensão
  sempre permite digitar.

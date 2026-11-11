# Política de Segurança e Privacidade

## Princípio: processamento no dispositivo

O REIA/Dør foi desenhado para **não enviar dados do usuário para servidores de terceiros**:

- A **extensão** roda a rede neural de tópicos e a rede do "halo" inteiramente no
  navegador. A voz usa a Web Speech API do próprio navegador; a coleta de treino do
  halo fica em `chrome.storage.local` e **nada é enviado**.
- A **CLI** usa STT/LLM **locais** (ex.: `whisper.cpp`, Ollama). Nenhuma chave de API
  é necessária.
- O **pipeline Python** roda localmente (Selenium headless local ou driver mock).

A extensão pede apenas a permissão `storage`. Ela **não** lê formulários, senhas ou
histórico, e **indica/orienta** — não clica nem submete nada pelo usuário.

## Reportando uma vulnerabilidade

Encontrou um problema de segurança ou privacidade? **Não** abra uma issue pública.
Entre em contato de forma privada com a equipe mantenedora (Grupo Dør — Residência em IA)
descrevendo:

1. O componente afetado (`reia/`, `voice-nav-cli/` ou `extension/`).
2. Passos para reproduzir.
3. Impacto potencial.

Faremos o possível para responder em até 7 dias.

## Escopo

Como ferramenta de acessibilidade, a regra é clara: o agente **sugere e destaca**,
mas a ação final (abrir o link, preencher, enviar) é sempre do usuário.

/* Documento offscreen — Fase 2: embeddings neurais on-device.
   Carrega o Transformers.js e um modelo multilíngue (pt-BR) e responde a pedidos de "embed".
   O modelo (~100 MB quantizado) é baixado do HF Hub na 1ª vez e fica em cache no navegador.

   NOTA (dev): a lib é importada da CDN (exige a CSP relaxada em manifest.json, válida para
   extensão "sem compactação"). Para publicar na loja, empacote o Transformers.js localmente,
   pois lojas não aceitam código remoto. Se qualquer passo falhar, o content.js mantém o
   ranqueamento TF-IDF — a extensão nunca depende disto para funcionar. */

const MODELO = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2';
let extractorPromise = null;

async function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3');
      env.allowLocalModels = false; // busca o modelo no HF Hub (cache no navegador)
      return pipeline('feature-extraction', MODELO, { quantized: true });
    })();
  }
  return extractorPromise;
}

async function embed(texts) {
  const extractor = await getExtractor();
  const out = await extractor(texts, { pooling: 'mean', normalize: true });
  return out.tolist(); // vetores já normalizados (L2) → cosseno = produto interno
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.type === 'reia-embed') {
    embed(msg.texts)
      .then((vectors) => sendResponse({ ok: true, vectors }))
      .catch((e) => sendResponse({ ok: false, error: String(e && e.message || e) }));
    return true; // resposta assíncrona
  }
});

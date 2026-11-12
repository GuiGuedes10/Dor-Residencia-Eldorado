/* Documento offscreen — Fase 2: embeddings neurais on-device.
   Carrega o Transformers.js + ONNX Runtime a partir de arquivos LOCAIS em `vendor/`
   (o Chrome MV3 proíbe código remoto no script-src), e um modelo multilíngue (pt-BR).

   Os arquivos de `vendor/` (~22 MB) NÃO são versionados — rode `extension/vendor/baixar.sh`
   para obtê-los. Sem eles, o import falha e o content.js mantém o ranqueamento TF-IDF:
   a extensão nunca depende disto para funcionar.

   O modelo (~100 MB quantizado) é baixado do HF Hub na 1ª consulta e fica em cache. */

const MODELO = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2';
let extractorPromise = null;

async function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline, env } = await import(chrome.runtime.getURL('vendor/transformers.min.js'));
      env.backends.onnx.wasm.numThreads = 1;                            // 1 thread: dispensa cross-origin isolation
      env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL('vendor/'); // ORT WASM local (self)
      // 1) tenta o modelo fine-tunado no domínio (se vendorado em vendor/modelo/reia-finetunado/)
      try {
        env.allowLocalModels = true;
        env.localModelPath = chrome.runtime.getURL('vendor/modelo/');
        return await pipeline('feature-extraction', 'reia-finetunado', { quantized: true });
      } catch (_) {
        // 2) fallback: modelo multilíngue genérico do HF Hub (Fase 2 padrão, já validada)
        env.allowLocalModels = false;
        return pipeline('feature-extraction', MODELO, { quantized: true });
      }
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
      .catch((e) => sendResponse({ ok: false, error: String((e && e.message) || e) }));
    return true; // resposta assíncrona
  }
});

/* Service worker — gerencia o documento offscreen que roda o modelo de embeddings
   e relaia os pedidos do content script. Tudo on-device; nada é enviado a servidores
   nossos (o modelo é baixado do HF Hub e fica em cache no navegador). */

async function temOffscreen() {
  if (chrome.offscreen && chrome.offscreen.hasDocument) return chrome.offscreen.hasDocument();
  try {
    const ctxs = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] });
    return Array.isArray(ctxs) && ctxs.length > 0;
  } catch { return false; }
}

let criando = null;
async function garantirOffscreen() {
  if (await temOffscreen()) return;
  if (!criando) {
    criando = chrome.offscreen.createDocument({
      url: 'offscreen.html',
      reasons: ['WORKERS'],
      justification: 'Rodar o modelo de embeddings (Transformers.js) para ranquear links por significado.',
    }).finally(() => { criando = null; });
  }
  return criando;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.type === 'reia-rank-embed') {
    (async () => {
      try {
        await garantirOffscreen();
        sendResponse(await chrome.runtime.sendMessage({ type: 'reia-embed', texts: msg.texts }));
      } catch (e) {
        sendResponse({ ok: false, error: String(e && e.message || e) });
      }
    })();
    return true; // resposta assíncrona
  }
});

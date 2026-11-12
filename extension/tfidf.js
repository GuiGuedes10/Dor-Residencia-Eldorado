/* Ranqueador semântico leve, on-device: TF-IDF de palavras + n-gramas de caractere.
   Sem modelo e sem download — supera a interseção de palavras em sinônimos/morfologia
   (validado no protótipo: 56% vs 36% succ@1 em pares reais da PUC).
   Ver prototipos/classificador-elementos/. Fase 2: embeddings neurais via Transformers.js. */
var ReiaRank = (() => {
  const norm = (s) => (s || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
  const tokens = (s) => norm(s).match(/[a-z0-9]+/g) || [];

  // n-gramas de caractere (3–4) com marcadores de borda <palavra> — pegam médico↔medicina
  function charNgrams(w) {
    const p = `<${w}>`, out = [];
    for (let n = 3; n <= 4; n++) for (let i = 0; i + n <= p.length; i++) out.push('#' + p.slice(i, i + n));
    return out;
  }
  function features(text) {
    const ws = tokens(text).filter((w) => w.length >= 3);
    const f = ws.slice();
    for (let i = 0; i < ws.length - 1; i++) f.push(ws[i] + '_' + ws[i + 1]); // bigramas de palavra
    for (const w of ws) for (const g of charNgrams(w)) f.push(g);
    return f;
  }
  // Ajusta IDF no conjunto de documentos e devolve: vetorizador de query + vetores dos docs.
  function fit(docs) {
    const df = new Map(), N = docs.length || 1;
    const featsList = docs.map(features);
    for (const feats of featsList) for (const f of new Set(feats)) df.set(f, (df.get(f) || 0) + 1);
    const idf = (f) => Math.log((N + 1) / ((df.get(f) || 0) + 1)) + 1;
    const vec = (feats) => {
      const v = new Map();
      for (const f of feats) v.set(f, (v.get(f) || 0) + idf(f));
      let n = 0; for (const x of v.values()) n += x * x;
      n = Math.sqrt(n) || 1;
      for (const [k, x] of v) v.set(k, x / n);
      return v;
    };
    return { vec, docVecs: featsList.map(vec) };
  }
  function cosine(a, b) {
    const [s, l] = a.size <= b.size ? [a, b] : [b, a];
    let d = 0; for (const [k, v] of s) d += v * (l.get(k) || 0);
    return d;
  }
  /** Ordena os índices de `docs` do mais ao menos similar à `query`. Retorna [{i, score}]. */
  function rank(query, docs) {
    const { vec, docVecs } = fit(docs);
    const qv = vec(features(query));
    return docVecs.map((v, i) => ({ i, score: cosine(qv, v) })).sort((a, b) => b.score - a.score);
  }
  return { tokens, features, fit, cosine, rank };
})();
if (typeof module !== 'undefined') module.exports = ReiaRank;

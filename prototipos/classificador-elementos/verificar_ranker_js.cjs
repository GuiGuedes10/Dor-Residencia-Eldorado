// Confere que o port JS (extension/tfidf.js) reproduz o resultado do ranqueador Python
// (char n-gram TF-IDF) sobre os mesmos dados. Uso: node prototipos/.../verificar_ranker_js.cjs
const fs = require('fs');
const path = require('path');
const ReiaRank = require('../../extension/tfidf.js');

const DIR = __dirname;
const norm = (s) => (s || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
const flat = (s) => (norm(s).match(/[a-z0-9]+/g) || []).join(' ');
const pathWords = (url) => { try { return decodeURIComponent(new URL(url).pathname).replace(/[^a-z0-9]+/gi, ' '); } catch { return ''; } };

function loadElementos() {
  const out = [], seen = new Set();
  for (const f of ['elementos_cursos.json', 'elementos_portal.json']) {
    const data = JSON.parse(fs.readFileSync(path.join(DIR, f), 'utf8'));
    for (const l of data.links || []) {
      const text = (l.text || '').replace(/\s+/g, ' ').trim();
      const url = l.url || '';
      if (text.length < 3 || seen.has(url)) continue;
      seen.add(url);
      const busca = [text, l.contexto || '', pathWords(url)].join(' ').replace(/\s+/g, ' ').trim();
      out.push({ text, url, busca });
    }
  }
  return out;
}
function resolver(alvo, els) {
  const a = flat(alvo);
  const cands = els.filter((e) => a && flat(e.busca).includes(a));
  return cands.length ? cands.reduce((m, e) => (e.busca.length < m.busca.length ? e : m)) : null;
}

const els = loadElementos();
const casos = JSON.parse(fs.readFileSync(path.join(DIR, 'dataset_pesquisa.json'), 'utf8')).casos;
let s1 = 0, s3 = 0, n = 0, pul = 0;
for (const c of casos) {
  const alvoEl = resolver(c.alvo, els);
  if (!alvoEl) { pul++; continue; }
  n++;
  const ranked = ReiaRank.rank(c.intencao, els.map((e) => e.busca));
  const pos = ranked.findIndex((r) => els[r.i].url === alvoEl.url) + 1;
  if (pos === 1) s1++;
  if (pos >= 1 && pos <= 3) s3++;
}
console.log(`Elementos: ${els.length} | pares avaliados: ${n} | pulados: ${pul}`);
console.log(`Port JS (char n-gram TF-IDF):  success@1 = ${(100 * s1 / n).toFixed(0)}%  success@3 = ${(100 * s3 / n).toFixed(0)}%`);
console.log('(esperado ~56% / ~68%, igual ao ranqueador Python)');

import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'data');
const fileFor = (host) => join(DIR, `${host.replace(/[^a-z0-9.-]/gi, '_')}.json`);

const strip = (s) => s.replace(/<[^>]*>/g, ' ').replace(/&nbsp;|&amp;/g, ' ').replace(/\s+/g, ' ').trim();

async function fetchHtml(url) {
  const res = await fetch(url, { headers: { 'User-Agent': 'voice-nav-cli' }, signal: AbortSignal.timeout(20000) });
  const html = await res.text();
  if (res.status !== 200 || html.length < 500) {
    throw new Error(`HTTP ${res.status} com ${html.length} bytes (possível bloqueio anti-bot; não contornado)`);
  }
  return html;
}

// Lê um atributo do <a> (aria-label, title, href...) a partir da string de atributos da tag.
const attr = (tag, name) => {
  const m = tag.match(new RegExp(`${name}\\s*=\\s*["']([^"']*)["']`, 'i'));
  return m ? strip(m[1]) : '';
};

function extractLinks(html, pageUrl, host) {
  const out = [];
  const title = strip((html.match(/<title[^>]*>([\s\S]*?)<\/title>/i) || [])[1] || '');
  for (const m of html.matchAll(/<a\s([^>]*)>([\s\S]*?)<\/a>/gi)) {
    const tag = m[1];
    const href = attr(tag, 'href');
    if (!href || href.startsWith('#')) continue;
    let u;
    try { u = new URL(href, pageUrl); } catch { continue; }
    if (!/^https?:$/.test(u.protocol) || u.hostname !== host) continue;
    if (/\.(pdf|jpe?g|png|zip|docx?|xlsx?)$/i.test(u.pathname)) continue;
    const text = strip(m[2]);
    const aria = attr(tag, 'aria-label');
    const titulo = attr(tag, 'title');
    // Rótulo: usa o aria-label/title quando o texto visível é vazio ("Saiba mais"/ícone).
    const rotulo = text || aria || titulo;
    if (!rotulo || rotulo.length < 2 || rotulo.length > 100) continue;
    // Contexto (enriquecimento p/ embeddings): aria-label + title + palavras do caminho da URL.
    const contexto = [aria, titulo, decodeURIComponent(u.pathname).replace(/[^a-z0-9]+/gi, ' ')]
      .map((s) => s.trim()).filter(Boolean).join(' ').trim();
    u.hash = '';
    out.push({ text: rotulo, url: u.href, contexto, from: title });
  }
  return out;
}

/**
 * Mapeia o site em largura: página inicial (profundidade 1) e, com depth=2, as
 * páginas linkadas (até maxPages). Salva em data/<host>.json.
 */
export async function mapSite(url, { depth = 2, maxPages = 25, maxLinks = 600 } = {}) {
  const base = new URL(url);
  const links = new Map();
  const visited = new Set();
  let frontier = [base.href];
  for (let d = 1; d <= depth && frontier.length; d++) {
    const next = [];
    for (const pageUrl of frontier) {
      if (visited.size >= maxPages || visited.has(pageUrl)) continue;
      visited.add(pageUrl);
      let html;
      try { html = await fetchHtml(pageUrl); } catch (e) { if (d === 1) throw e; continue; }
      for (const l of extractLinks(html, pageUrl, base.hostname)) {
        if (!links.has(l.url)) { links.set(l.url, l); next.push(l.url); }
      }
    }
    frontier = next;
  }
  const list = [...links.values()].slice(0, maxLinks);
  mkdirSync(DIR, { recursive: true });
  writeFileSync(fileFor(base.hostname), JSON.stringify({ site: base.origin, mapeadoEm: new Date().toISOString(), paginas: visited.size, links: list }, null, 2));
  return list;
}

export function loadMap(url) {
  const f = fileFor(new URL(url).hostname);
  return existsSync(f) ? JSON.parse(readFileSync(f, 'utf8')) : null;
}

const norm = (s) => s.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
const OLLAMA = 'http://localhost:11434';
const EMBED_MODEL = process.env.EMBED_MODEL || 'nomic-embed-text';

// Temas além do vestibular: expande o pedido com sinônimos para casar com o texto dos links.
const SINONIMOS = {
  extensao: 'extensao cursos livres capacitacao formacao continuada fic oficinas',
  concurso: 'concurso concursos edital selecao servidor docente professor tecnico administrativo',
  vestibular: 'vestibular ingresso inscricao sisu enem selecao graduacao',
  bolsa: 'bolsa bolsas auxilio assistencia estudantil financiamento',
  pos: 'pos graduacao mestrado doutorado especializacao residencia',
};
function expandir(query) {
  const q = norm(query);
  const extra = Object.entries(SINONIMOS).filter(([k]) => q.includes(k.slice(0, 5))).map(([, v]) => v);
  return extra.length ? `${query} ${extra.join(' ')}` : query;
}

async function embed(texts) {
  const res = await fetch(`${OLLAMA}/api/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EMBED_MODEL, input: texts }),
    signal: AbortSignal.timeout(60000),
  });
  if (!res.ok) throw new Error('embed indisponível');
  return (await res.json()).embeddings;
}

const cos = (a, b) => {
  let d = 0, x = 0, y = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] ** 2; y += b[i] ** 2; }
  return d / (Math.sqrt(x * y) || 1);
};

const RUIDO = /login|entrar|acessibilidade|mapa-do-site|contato|cookies|privacidade|facebook|instagram|twitter|youtube/i;

function keywordScore(query, link) {
  const q = norm(query).split(/\W+/).filter((w) => w.length > 2);
  const alvo = norm(`${link.text} ${decodeURIComponent(new URL(link.url).pathname).replace(/[-_/]/g, ' ')}`);
  return q.reduce((s, w) => s + (alvo.includes(w) ? 1 : 0), 0) / (q.length || 1);
}

/**
 * Ranqueia links pelo comando falado. Usa rede neural (embeddings via Ollama,
 * modelo nomic-embed-text) e cai para palavras-chave (texto + caminho da URL) se o Ollama não responder.
 * Remove links de ruído e itens repetidos. Retorna até `n` opções (poucas, por acessibilidade).
 */
export async function rank(query, links, n = 4) {
  const q = expandir(query);
  const cand = links.filter((l) => !RUIDO.test(l.url) && !RUIDO.test(l.text));
  let scores, metodo = 'palavras-chave';
  try {
    const vecs = await embed([q, ...cand.map((l) => `${l.text} ${new URL(l.url).pathname.replace(/[-_/]/g, ' ')}`)]);
    scores = cand.map((_, i) => cos(vecs[0], vecs[i + 1]));
    metodo = `rede neural (${EMBED_MODEL})`;
  } catch {
    scores = cand.map((l) => keywordScore(q, l));
  }
  const vistos = new Set();
  const top = [];
  for (const l of cand.map((l, i) => ({ ...l, score: scores[i] })).sort((a, b) => b.score - a.score)) {
    const k = norm(l.text);
    if (vistos.has(k)) continue;
    vistos.add(k);
    top.push(l);
    if (top.length === n) break;
  }
  return { top, metodo };
}

/* Coleta de dados e treino da rede de "calor" (nn.js). Tudo roda no navegador; nada é enviado. */
const arena = document.getElementById('arena');
const log = document.getElementById('log');
let data = [];           // [{f:[...], y:0|1}]
let trials = 0, targets = [], prev = null, buf = [];

chrome.storage.local.get('warmthData').then((s) => { data = s.warmthData || []; status(); });

const status = () => { log.textContent = `${trials} tentativas nesta sessão · ${data.length} exemplos salvos`; };

function newTrial() {
  arena.innerHTML = '';
  targets = [];
  const n = 3 + ((Math.random() * 3) | 0);
  for (let i = 0; i < n; i++) {
    let r, tries = 0;
    do {
      r = { left: 20 + Math.random() * 700, top: 20 + Math.random() * 440, w: 100 + Math.random() * 60, h: 40 + Math.random() * 30 };
      tries++;
    } while (tries < 50 && targets.some((t) => Math.abs(t.left - r.left) < 170 && Math.abs(t.top - r.top) < 80));
    const el = document.createElement('div');
    el.className = 't' + (i === 0 ? ' goal' : '');
    el.textContent = i === 0 ? 'ALVO' : 'outro';
    Object.assign(el.style, { left: r.left + 'px', top: r.top + 'px', width: r.w + 'px', height: r.h + 'px' });
    arena.appendChild(el);
    targets.push({ el, goal: i === 0 });
  }
  prev = null; buf = [];
}

arena.addEventListener('mousemove', (e) => {
  const b = arena.getBoundingClientRect();
  const cur = { x: e.clientX, y: e.clientY, t: performance.now() };
  const diag = Math.hypot(b.width, b.height);
  for (const t of targets) {
    const r = t.el.getBoundingClientRect();
    buf.push({ f: ReiaNN.features(cur, prev, r, diag), goal: t.goal });
  }
  prev = cur;
});

arena.addEventListener('click', (e) => {
  const hit = targets.find((t) => t.el.contains(e.target));
  if (hit && hit.goal) {
    // alvo escolhido (o amarelo) = 1; qualquer outro bloco = 0 para o mesmo movimento
    for (const s of buf) data.push({ f: s.f, y: s.goal ? 1 : 0 });
    // amostra subsample para limitar o tamanho
    if (data.length > 40000) data = data.slice(-40000);
    trials++;
    chrome.storage.local.set({ warmthData: data });
    status();
  }
  newTrial();
});

document.getElementById('train').onclick = async () => {
  if (data.length < 500) { log.textContent = 'Poucos dados: faça mais tentativas (mínimo ~30).'; return; }
  const shuffled = [...data].sort(() => Math.random() - 0.5);
  const cut = Math.floor(shuffled.length * 0.8);
  const tr = shuffled.slice(0, cut), va = shuffled.slice(cut);
  log.textContent = 'Treinando…';
  await new Promise((r) => setTimeout(r, 30));
  const m = ReiaNN.train(tr, { hidden: 8, epochs: 60, lr: 0.05 });
  await chrome.storage.local.set({ warmthModel: m });
  log.textContent = `✅ Treinado. Acurácia treino ${(ReiaNN.accuracy(tr, m) * 100).toFixed(0)}% · validação ${(ReiaNN.accuracy(va, m) * 100).toFixed(0)}%. Já vale nos sites.`;
};
document.getElementById('export').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' }));
  a.download = 'warmth_data.json'; a.click();
};
document.getElementById('clear').onclick = async () => { data = []; await chrome.storage.local.remove(['warmthData', 'warmthModel']); trials = 0; status(); };

newTrial();

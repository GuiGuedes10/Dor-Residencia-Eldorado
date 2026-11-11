/* REIA Voice-Nav — controle flutuante acessível. Fale (ou digite), escolha 1 de até 4 links;
   um halo muda de cor enquanto você leva o mouse ao link escolhido:
     vermelho = dando errado (longe) · âmbar = quase lá · azul = dando certo (em cima do alvo).
   A cor nunca é a única pista: o halo também encolhe ao se aproximar e o alvo fica contornado. */
(() => {
  if (window.__reiaVoiceNav) return;
  window.__reiaVoiceNav = true;

  const norm = (s) => (s || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
  const tokens = (s) => norm(s).match(/[a-z0-9]+/g) || [];
  const esc = (s) => (s || '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  const SINONIMOS = {
    vestibular: 'vestibular ingresso inscricao sisu enem selecao graduacao',
    cursos: 'cursos graduacao curso',
    extensao: 'extensao cursos livres capacitacao formacao continuada fic oficinas',
    concursos: 'concurso concursos edital selecao servidor docente professor tecnico administrativo',
    bolsas: 'bolsa bolsas auxilio assistencia estudantil financiamento',
    pos: 'pos graduacao mestrado doutorado especializacao residencia',
    contato: 'contato fale conosco telefone endereco secretaria localizacao',
  };

  // ---------- rede de tópicos (treinada com training/train.py) ----------
  let topicNet = null;
  async function loadTopicNet() {
    try {
      const r = await fetch(chrome.runtime.getURL('models/intent.json'));
      topicNet = await r.json();
      topicNet.idx = new Map(topicNet.vocab.map((w, i) => [w, i]));
    } catch { topicNet = null; }
  }
  function topicOf(text) {
    if (!topicNet) return null;
    const act = [...new Set(tokens(text).map((w) => topicNet.idx.get(w)).filter((i) => i !== undefined))];
    if (!act.length) return null;
    const h = topicNet.w1.map((row, j) => Math.max(0, act.reduce((s, i) => s + row[i], topicNet.b1[j])));
    const z = topicNet.w2.map((row, k) => row.reduce((s, w, j) => s + w * h[j], topicNet.b2[k]));
    const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), sum = e.reduce((a, b) => a + b, 0);
    const p = e.map((v) => v / sum), b = p.indexOf(Math.max(...p));
    return { topic: topicNet.classes[b], confidence: p[b] };
  }

  // ---------- links da página atual ----------
  function pageLinks() {
    const seen = new Set(), out = [];
    for (const a of document.querySelectorAll('a[href]')) {
      const text = (a.innerText || a.getAttribute('aria-label') || a.title || '').replace(/\s+/g, ' ').trim();
      if (text.length < 3 || text.length > 80) continue;
      if (/^(javascript:|mailto:|tel:|#)/i.test(a.getAttribute('href'))) continue;
      const r = a.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      const k = norm(text);
      if (seen.has(k)) continue;
      seen.add(k);
      out.push({ el: a, text, href: a.href, inNav: !!a.closest('nav, header') });
    }
    return out;
  }
  function rankLinks(query) {
    const t = topicOf(query);
    const q = new Set(tokens(query + ' ' + (t && t.confidence > 0.5 ? SINONIMOS[t.topic] || '' : '')));
    const scored = pageLinks().map((l) => {
      const alvo = tokens(l.text + ' ' + decodeURIComponent(new URL(l.href).pathname));
      const hit = alvo.filter((w) => q.has(w)).length;
      return { ...l, score: hit / Math.sqrt(alvo.length || 1) + (l.inNav ? 0.05 : 0) };
    });
    scored.sort((a, b) => b.score - a.score);
    return { top: scored.slice(0, 4), topic: t };
  }

  // ---------- interface (Shadow DOM) ----------
  const host = document.createElement('div');
  host.style.cssText = 'all:initial;position:fixed;z-index:2147483647;inset:0;pointer-events:none';
  const root = host.attachShadow({ mode: 'open' });
  root.innerHTML = `
  <style>
    :host{ --reia-accent:#8fd0ff; --reia-blue:rgb(37,109,235); --reia-amber:rgb(245,183,24); --reia-red:rgb(211,47,47); }
    *{box-sizing:border-box;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}

    /* Botão flutuante — alvo de toque grande (≥48px) e sempre visível. */
    #fab{position:fixed;right:24px;bottom:24px;width:72px;height:72px;border-radius:50%;border:0;font-size:32px;
         background:rgba(18,20,28,.92);color:#fff;cursor:pointer;pointer-events:auto;backdrop-filter:blur(6px);
         box-shadow:0 4px 16px rgba(0,0,0,.45);transition:transform .15s,box-shadow .15s}
    #fab:hover{transform:translateY(-2px);box-shadow:0 6px 22px rgba(0,0,0,.55)}
    #fab:focus-visible{outline:4px solid var(--reia-accent);outline-offset:3px}

    /* Modal maior e com mais respiro (acessibilidade). */
    #modal{position:fixed;right:24px;bottom:112px;width:min(460px,94vw);max-height:80vh;overflow:auto;
           padding:22px 22px 20px;border-radius:20px;display:none;pointer-events:auto;
           background:rgba(13,15,22,.95);color:#f4f7fb;backdrop-filter:blur(10px);
           box-shadow:0 12px 40px rgba(0,0,0,.6);border:1px solid rgba(255,255,255,.08)}
    #modal.open{display:block}
    #title{font-size:22px;line-height:1.25;margin:0 0 4px;font-weight:700}
    #status{font-size:17px;line-height:1.45;margin:0 0 12px;min-height:22px;color:#d9e4f2}

    /* Medidor de confiança: barra + texto (nunca só cor). */
    #meter{height:10px;border-radius:999px;background:rgba(255,255,255,.14);margin:0 0 14px;overflow:hidden;display:none}
    #meter.on{display:block}
    #meterfill{height:100%;width:0;border-radius:999px;transition:width .3s ease,background-color .3s ease}

    /* Opções: botões grandes, numerados, com foco visível. */
    .opt{display:flex;align-items:center;gap:10px;width:100%;margin:10px 0;padding:16px;font-size:19px;line-height:1.3;
         text-align:left;border-radius:12px;border:2px solid transparent;min-height:56px;
         background:#f3f6fb;color:#10141c;cursor:pointer;font-weight:600}
    .opt .num{flex:0 0 28px;height:28px;border-radius:7px;background:#10141c;color:#fff;
              font-size:15px;display:flex;align-items:center;justify-content:center}
    .opt:hover{background:#fff}
    .opt:focus-visible{outline:4px solid var(--reia-accent);outline-offset:2px;border-color:#10141c}

    /* Linha de entrada: campo e botões altos. */
    .row{display:flex;gap:10px;flex-wrap:wrap}
    .row input{flex:1 1 160px;min-height:52px;padding:12px 14px;font-size:17px;border-radius:12px;
               border:2px solid rgba(255,255,255,.18);background:rgba(255,255,255,.96);color:#10141c}
    .row input:focus-visible{outline:4px solid var(--reia-accent);outline-offset:1px;border-color:#10141c}
    .row button{min-height:52px;padding:12px 18px;font-size:17px;font-weight:600;border-radius:12px;border:0;cursor:pointer}
    #mic{background:var(--reia-blue);color:#fff}
    #go{background:#f3f6fb;color:#10141c}
    .row button:focus-visible{outline:4px solid var(--reia-accent);outline-offset:2px}

    /* Halo que segue o mouse. Cor = ReiaNN.color(w); tamanho encolhe ao acertar. */
    #halo{position:fixed;width:56px;height:56px;margin:-28px 0 0 -28px;border-radius:50%;pointer-events:none;display:none;
          border:6px solid var(--reia-red);box-shadow:0 0 22px currentColor;opacity:.9;
          transition:border-color .08s,width .08s,height .08s,margin .08s}

    /* Faixa de instrução no topo, com legenda de cores. */
    #hint{position:fixed;left:50%;top:18px;transform:translateX(-50%);max-width:92vw;padding:12px 18px;border-radius:12px;display:none;
          background:rgba(13,15,22,.95);color:#f4f7fb;font-size:17px;line-height:1.5;
          box-shadow:0 6px 24px rgba(0,0,0,.5);border:1px solid rgba(255,255,255,.08)}
    #hint .lg{display:inline-flex;align-items:center;gap:5px;margin:0 6px}
    #hint .sw{display:inline-block;width:13px;height:13px;border-radius:50%;box-shadow:0 0 0 1px rgba(255,255,255,.4)}
    kbd{background:#2a2f3a;border-radius:5px;padding:1px 7px;font-size:14px}

    /* Respeita preferências do sistema. */
    @media (prefers-reduced-motion: reduce){ *{transition:none !important} }
    @media (forced-colors: active){
      .opt,.row input,#modal,#hint{border:1px solid CanvasText}
      #fab:focus-visible,.opt:focus-visible,.row *:focus-visible{outline:3px solid Highlight}
    }
  </style>
  <button id="fab" aria-label="Ativar navegação por voz" title="Navegação por voz (Alt+V)">🎤</button>
  <div id="modal" role="dialog" aria-modal="true" aria-labelledby="title">
    <h2 id="title">Navegação por voz</h2>
    <p id="status" role="status" aria-live="polite">Toque no microfone e diga o que procura.</p>
    <div id="meter" aria-hidden="true"><div id="meterfill"></div></div>
    <div class="row">
      <button id="mic" aria-label="Falar agora">🎤 Falar</button>
      <input id="txt" placeholder="ou digite o que procura…" aria-label="Digitar o que procura" autocomplete="off">
      <button id="go" aria-label="Buscar">OK</button>
    </div>
    <div id="opts" role="group" aria-label="Opções sugeridas"></div>
  </div>
  <div id="halo" aria-hidden="true"></div>
  <div id="hint" role="status" aria-live="polite"></div>`;
  document.documentElement.appendChild(host);
  const $ = (id) => root.getElementById(id);
  const modal = $('modal'), status = $('status'), opts = $('opts'), halo = $('halo'), hint = $('hint');
  const meter = $('meter'), meterfill = $('meterfill');

  const say = (m) => { status.textContent = m; };
  const toggle = (open) => { modal.classList.toggle('open', open ?? !modal.classList.contains('open')); if (modal.classList.contains('open')) $('txt').focus(); };

  // ---------- voz (Web Speech API, pt-BR) ----------
  let rec = null;
  function listen() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { say('Voz não suportada neste navegador. Digite abaixo.'); return; }
    rec = new SR();
    rec.lang = 'pt-BR'; rec.interimResults = true; rec.maxAlternatives = 1;
    say('🎤 Ouvindo…');
    rec.onresult = (e) => {
      const r = e.results[e.results.length - 1];
      $('txt').value = r[0].transcript;
      if (r.isFinal) { say(`Entendi: "${r[0].transcript}"`); suggest(r[0].transcript); }
    };
    rec.onerror = (e) => say(e.error === 'not-allowed' ? 'Permita o microfone para este site ou digite.' : `Erro de voz: ${e.error}. Digite abaixo.`);
    rec.onend = () => { if (status.textContent.startsWith('🎤')) say('Não ouvi nada. Tente de novo ou digite.'); };
    rec.start();
  }

  // ---------- sugestões (≤ 4) ----------
  function suggest(query) {
    const { top, topic } = rankLinks(query);
    opts.innerHTML = '';
    if (topic) {
      const pct = Math.round(topic.confidence * 100);
      say(`Tópico: ${topic.topic} — confiança ${pct}%. Escolha uma opção:`);
      meter.classList.add('on');
      meterfill.style.width = `${pct}%`;
      meterfill.style.background = ReiaNN.color(topic.confidence); // azul = alta confiança, vermelho = baixa
    } else {
      meter.classList.remove('on');
      say('Escolha uma opção:');
    }
    top.forEach((l, i) => {
      const b = document.createElement('button');
      b.className = 'opt';
      b.innerHTML = `<span class="num">${i + 1}</span><span>${esc(l.text)}</span>`;
      b.setAttribute('aria-label', `Opção ${i + 1}: ${l.text}`);
      b.onclick = () => guide(l);
      opts.appendChild(b);
    });
    if (!top.length) { meter.classList.remove('on'); say('Nenhum link encontrado nesta página. Tente outras palavras.'); }
    opts.firstChild?.focus();
  }

  // ---------- modo guia: halo muda de cor conforme o mouse se aproxima ----------
  let guide_ = null, nnModel = null, prev = null;
  chrome.storage.local.get('warmthModel').then((s) => { nnModel = s.warmthModel || null; }).catch(() => {});

  const LEGEND =
    '<span class="lg"><i class="sw" style="background:var(--reia-red)"></i>errado</span>' +
    '<span class="lg"><i class="sw" style="background:var(--reia-amber)"></i>quase</span>' +
    '<span class="lg"><i class="sw" style="background:var(--reia-blue)"></i>certo</span>';

  function guide(link) {
    toggle(false);
    clearGuide();
    link.el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    const old = link.el.style.outline;
    link.el.style.outline = '5px solid var(--reia-red)';
    guide_ = { link, old, samples: [] };
    halo.style.display = 'block';
    hint.style.display = 'block';
    hint.innerHTML = `Leve o mouse até <b>"${esc(link.text)}"</b> &nbsp; ${LEGEND} &nbsp;·&nbsp; <kbd>Enter</kbd> abre · <kbd>Esc</kbd> cancela`;
    window.addEventListener('mousemove', onMove, true);
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('click', onClick, true);
  }
  function clearGuide() {
    if (!guide_) return;
    guide_.link.el.style.outline = guide_.old;
    guide_ = null; prev = null;
    halo.style.display = 'none'; hint.style.display = 'none';
    window.removeEventListener('mousemove', onMove, true);
    window.removeEventListener('keydown', onKey, true);
    window.removeEventListener('click', onClick, true);
  }
  function onMove(e) {
    if (!guide_) return;
    const cur = { x: e.clientX, y: e.clientY, t: performance.now() };
    const rect = guide_.link.el.getBoundingClientRect();
    const diag = Math.hypot(innerWidth, innerHeight);
    const f = ReiaNN.features(cur, prev, rect, diag);
    prev = cur;
    const w = ReiaNN.predict(f, nnModel);
    const c = ReiaNN.color(w);
    const size = Math.round(40 + (1 - w) * 36); // pista redundante: perto = halo menor/mais preciso
    halo.style.width = halo.style.height = `${size}px`;
    halo.style.margin = `${-size / 2}px 0 0 ${-size / 2}px`;
    halo.style.left = `${cur.x}px`; halo.style.top = `${cur.y}px`;
    halo.style.borderColor = c; halo.style.color = c;
    guide_.link.el.style.outlineColor = c;
  }
  function onKey(e) {
    if (e.key === 'Escape') clearGuide();
    else if (e.key === 'Enter') { const u = guide_.link.href; clearGuide(); location.href = u; }
  }
  function onClick(e) {
    if (guide_ && guide_.link.el.contains(e.target)) clearGuide(); // deixa o clique seguir para o link
  }

  // ---------- eventos ----------
  $('fab').onclick = () => toggle();
  $('mic').onclick = listen;
  $('go').onclick = () => $('txt').value.trim() && suggest($('txt').value.trim());
  $('txt').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('go').click(); e.stopPropagation(); });
  modal.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') toggle(false);
    if (/^[1-4]$/.test(e.key) && e.target.tagName !== 'INPUT') opts.children[Number(e.key) - 1]?.click();
  });
  window.addEventListener('keydown', (e) => { if (e.altKey && e.key.toLowerCase() === 'v') { toggle(true); listen(); } });
  chrome.storage.onChanged.addListener((c) => { if (c.warmthModel) nnModel = c.warmthModel.newValue; });

  loadTopicNet();
})();

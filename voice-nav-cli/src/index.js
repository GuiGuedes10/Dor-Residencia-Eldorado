#!/usr/bin/env node
import readline from 'node:readline/promises';
import { spawn } from 'node:child_process';
import { INSTITUTIONS, detectInstitution, detectCampus } from './institutions.js';
import { REGIOES, FEDERAIS } from './regioes.js';
import { mapSite, loadMap } from './sitemap.js';
import { rank } from './rank.js';
import { neuralIntent } from './neural.js';
import { topicIntent } from './intent.js';
import { recordAudio } from './record.js';
import { transcribeAudio, audioLevel } from './transcribe.js';

const argv = process.argv.slice(2);
const flag = (n) => argv.includes(`--${n}`);
const opt = (n) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : undefined; };
const positional = argv.filter((a, i) => !a.startsWith('--') && !/^--(site|audio|depth)$/.test(argv[i - 1] || ''));

const rl = readline.createInterface({ input: process.stdin, output: process.stdout, terminal: process.stdin.isTTY });
const lines = rl[Symbol.asyncIterator]();
async function ask(q) {
  process.stdout.write(q);
  const { value, done } = await lines.next();
  if (done) { console.log('\n(entrada encerrada)'); process.exit(1); }
  return value.trim();
}

/** Menu com no máximo 4 linhas (acessibilidade): com mais itens, a 4ª opção é "Mais opções". */
async function choose(titulo, items, label) {
  let page = 0;
  for (;;) {
    const paged = items.length > 4;
    const per = paged ? 3 : items.length;
    const slice = items.slice(page * per, page * per + per);
    const more = paged && (page + 1) * per < items.length;
    console.log(titulo);
    slice.forEach((it, i) => console.log(`  ${i + 1}) ${label(it)}`));
    if (paged) console.log(`  ${slice.length + 1}) ${more ? 'Mais opções' : 'Voltar ao início'}`);
    const n = Number(await ask(`Escolha 1-${slice.length + (paged ? 1 : 0)}: `));
    if (paged && n === slice.length + 1) { page = more ? page + 1 : 0; continue; }
    return slice[n - 1];
  }
}

// ---- comando: diagnóstico de microfone/transcrição ----
if (positional[0] === 'doctor') {
  console.log('Gravando — fale algo e faça uma pausa...');
  try {
    const wav = recordAudio(8);
    console.log(`Amplitude máxima: ${audioLevel(wav)} (abaixo de 0.02 = microfone mudo/sem permissão)`);
    const r = transcribeAudio(wav);
    console.log(r.text ? `✅ Transcrição: "${r.text}"` : `❌ ${r.error}`);
  } catch (e) { console.log(`❌ ${e.message}`); }
  process.exit(0);
}

// ---- comando: mapear site ----
if (positional[0] === 'map') {
  const url = positional[1];
  if (!url) { console.log('Uso: npm run map -- https://www.fia.com.br [--depth 2]'); process.exit(1); }
  console.log(`Instituição: ${detectInstitution(url)?.nome ?? 'não reconhecida'}`);
  try {
    const links = await mapSite(url, { depth: Number(opt('depth')) || 2 });
    console.log(`✅ ${links.length} links mapeados em data/`);
  } catch (e) { console.log(`❌ ${e.message}`); process.exit(1); }
  process.exit(0);
}

// ---- 1) instituição; UF/IF: região e estado; PUC: campus ----
async function escolherSite() {
  const inst = await choose('Qual instituição?', INSTITUTIONS, (i) => i.nome);
  if (!inst) return null;
  if (inst.id === 'uf' || inst.id === 'if') {
    const regiao = await choose('Qual região?', Object.keys(REGIOES), (r) => r);
    if (!regiao) return null;
    const estado = await choose(`Qual estado (${regiao})?`, REGIOES[regiao], (e) => e);
    if (!estado) return null;
    const lista = FEDERAIS[inst.id][estado];
    if (!lista) return (await ask(`Sem cadastro para ${estado}. Cole a URL do site: `)) || null;
    const escolhida = lista.length === 1 ? lista[0] : await choose(`Qual ${inst.nome} em ${estado}?`, lista, (x) => x.nome);
    return escolhida?.url ?? null;
  }
  const campus = inst.campi.length > 1 ? await choose(`Qual campus da ${inst.nome}?`, inst.campi, (c) => c.nome) : inst.campi[0];
  return campus?.url ?? null;
}
const site = opt('site') ?? (await escolherSite());
if (!site) { console.log('Opção inválida.'); process.exit(1); }
const inst = detectInstitution(site);
const campus = detectCampus(site);
console.log(`🏫 ${campus?.nome ?? inst?.nome ?? 'Instituição não reconhecida'} — ${site}`);

// ---- 2) comando por voz, com confirmação do que foi entendido ----
async function getText() {
  if (positional.length) return positional.join(' ');
  if (!flag('text')) {
    for (let tentativa = 1; tentativa <= 3; tentativa++) {
      try {
        let wav = opt('audio');
        if (!wav) { console.log('🎤 Fale agora (para sozinho após uma pausa)...'); wav = recordAudio(8); }
        const lvl = audioLevel(wav);
        if (lvl !== null && lvl < 0.02) console.log(`⚠️  Áudio quase mudo (amplitude ${lvl}). Rode: npm run doctor`);
        const r = transcribeAudio(wav);
        if (!r.text) { console.log(`⚠️  ${r.error}`); if (opt('audio')) break; continue; }
        if (opt('audio') || flag('yes')) return r.text;
        const ok = await choose(`Entendi: "${r.text}". Está certo?`, ['Sim', 'Repetir', 'Digitar'], (x) => x);
        if (ok === 'Sim') return r.text;
        if (ok === 'Digitar') break;
      } catch (e) { console.log(`⚠️  ${e.message}`); break; }
    }
  }
  return ask('Digite o comando: ');
}
const text = await getText();
if (!text) { console.log('Nenhum comando.'); process.exit(1); }
console.log(`📝 "${text}"`);

// ---- 3a) PUC-Campinas: rede neural treinada do projeto (reia/neural) ----
if (campus?.neural) {
  const r = neuralIntent(text);
  if (r?.accepted) {
    console.log(`🧠 Rede neural REIA: ${r.intent} (${(r.confidence * 100).toFixed(0)}%)`);
    console.log(`➡️  ${r.url}`);
    if (!flag('no-open')) spawn('open', [r.url], { stdio: 'ignore', detached: true }).unref();
    process.exit(0);
  }
  if (r) console.log(`🧠 Rede neural REIA não decidiu: ${r.message}`);
}

// ---- 3b) sua rede de tópicos (training/train.py) + mapa do site + ranking ----
const t = topicIntent(text);
if (t) console.log(`🧠 Tópico: ${t.topic} (${(t.confidence * 100).toFixed(0)}%)`);
const consulta = t && t.confidence > 0.5 ? `${text} ${t.topic}` : text;

let map = loadMap(site);
if (!map) {
  console.log('Mapeando o site (primeira vez, até 2 níveis)...');
  try { map = { links: await mapSite(site) }; } catch (e) { console.log(`❌ Não foi possível mapear: ${e.message}`); process.exit(1); }
}
const { top, metodo } = await rank(consulta, map.links, 4);
console.log(`🔎 Ranking por ${metodo}`);
const pick = await choose('Opções:', top, (l) => `${l.text}  [${l.score.toFixed(2)}]`);
if (!pick) { console.log('Opção inválida.'); process.exit(1); }
console.log(`➡️  ${pick.url}`);
if (!flag('no-open')) spawn('open', [pick.url], { stdio: 'ignore', detached: true }).unref();
rl.close();

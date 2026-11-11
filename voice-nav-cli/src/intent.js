import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const MODEL = join(dirname(fileURLToPath(import.meta.url)), '..', 'models', 'intent.json');
let net = null;

const tokens = (t) => t.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').match(/[a-z0-9]+/g) ?? [];

/** Inferência da rede treinada por você (training/train.py). Retorna { topic, confidence } ou null. */
export function topicIntent(text) {
  if (!net) {
    if (!existsSync(MODEL)) return null;
    net = JSON.parse(readFileSync(MODEL, 'utf8'));
    net.idx = new Map(net.vocab.map((w, i) => [w, i]));
  }
  const active = [...new Set(tokens(text).map((w) => net.idx.get(w)).filter((i) => i !== undefined))];
  if (!active.length) return null;
  const h = net.w1.map((row, j) => Math.max(0, active.reduce((s, i) => s + row[i], net.b1[j])));
  const z = net.w2.map((row, k) => row.reduce((s, w, j) => s + w * h[j], net.b2[k]));
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const sum = e.reduce((a, b) => a + b, 0);
  const p = e.map((v) => v / sum);
  const best = p.indexOf(Math.max(...p));
  return { topic: net.classes[best], confidence: p[best] };
}

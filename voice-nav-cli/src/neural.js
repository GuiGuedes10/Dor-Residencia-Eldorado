import { spawnSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT = join(dirname(fileURLToPath(import.meta.url)), '..', 'bridge', 'intent.py');

/**
 * Consulta a rede neural treinada (reia/neural, PyTorch/MLP em Python).
 * Hoje ela cobre só PUC-Campinas (3 intenções). Retorna null se indisponível.
 */
export function neuralIntent(text) {
  const r = spawnSync(process.env.PYTHON || 'python3', [SCRIPT], { input: text, encoding: 'utf8' });
  if (r.error || r.status !== 0) return null;
  try { return JSON.parse(r.stdout); } catch { return null; }
}

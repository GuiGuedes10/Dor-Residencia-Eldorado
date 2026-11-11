import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

/**
 * Grava com sox (brew install sox) em WAV 16 kHz mono.
 * Para sozinho após ~1,5s de silêncio (máx. `maxSeconds`), sem esperar tempo fixo.
 */
export function recordAudio(maxSeconds = 8) {
  const file = join(tmpdir(), `voice-nav-${Date.now()}.wav`);
  const r = spawnSync('rec', [
    '-q', '-c', '1', file, 'rate', '16000',
    'silence', '1', '0.1', '2%', '1', '1.5', '2%',
    'trim', '0', String(maxSeconds),
  ], { stdio: 'inherit' });
  if (r.error || r.status !== 0) throw new Error('Falha ao gravar. Instale sox e permita o microfone ao Terminal.');
  return file;
}

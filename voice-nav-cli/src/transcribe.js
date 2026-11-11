import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const DEFAULT_MODEL = join(dirname(fileURLToPath(import.meta.url)), '..', 'models', 'ggml-base.bin');

// Vocabulário do domínio: ajuda o Whisper a acertar termos de vestibular/campus.
const PROMPT = 'Vestibular, inscrição, manual do candidato, guia universitário, cursos, graduação, campus, PUC Campinas, PUC Rio, PUC São Paulo, bolsas, calendário, edital, matrícula, mensalidade.';

/** Transcreve com whisper.cpp. Retorna { text, error }. Nunca engole o erro. */
export function transcribeAudio(wavPath) {
  // WHISPER_MODEL só vale se o arquivo existir; senão usa o modelo baixado em models/.
  const envModel = process.env.WHISPER_MODEL;
  const model = envModel && existsSync(envModel) ? envModel : DEFAULT_MODEL;
  if (!existsSync(model)) return { text: null, error: `modelo não encontrado: ${model}` };
  const args = ['-m', model, '-l', 'pt', '-nt', '-np', '--prompt', PROMPT, '-f', wavPath];
  // WHISPER_NO_GPU=1 força CPU (útil se o Metal travar).
  if (process.env.WHISPER_NO_GPU) args.push('-ng');
  const r = spawnSync('whisper-cli', args, { encoding: 'utf8', timeout: 180000 });
  if (r.error) return { text: null, error: `whisper-cli indisponível (${r.error.code}). brew install whisper-cpp` };
  if (r.status !== 0) return { text: null, error: `whisper-cli falhou: ${(r.stderr || '').split('\n').slice(-3).join(' ')}` };
  const text = r.stdout.replace(/\[.*?\]/g, '').trim();
  return text ? { text, error: null } : { text: null, error: 'áudio sem fala detectada (volume baixo ou microfone errado)' };
}

/** Nível do áudio via sox stat; ajuda a diagnosticar microfone mudo. */
export function audioLevel(wavPath) {
  const r = spawnSync('sox', [wavPath, '-n', 'stat'], { encoding: 'utf8' });
  const m = (r.stderr || '').match(/Maximum amplitude:\s+([\d.]+)/);
  return m ? Number(m[1]) : null;
}

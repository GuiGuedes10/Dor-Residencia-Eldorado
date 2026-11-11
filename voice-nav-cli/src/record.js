import record from 'node-record-lpcm16';
import { Writable } from 'stream';

/**
 * Record a short audio snippet from the default microphone.
 * Returns a Buffer containing raw 16‑bit PCM data (signed little‑endian, 16kHz mono).
 */
export function recordAudio(durationMs = 5000) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    const writable = new Writable({
      write(chunk, _, cb) {
        chunks.push(chunk);
        cb();
      }
    });

    const mic = record.record({
      sampleRate: 16000,
      channels: 1,
      audioType: 'wav', // node‑record‑lpcm16 will output wav container
    });

    mic.stream().pipe(writable);
    setTimeout(() => {
      mic.stop();
      const buffer = Buffer.concat(chunks);
      resolve(buffer);
    }, durationMs);
  });
}

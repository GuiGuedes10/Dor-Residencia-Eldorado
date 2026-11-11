/* Rede neural minúscula (MLP 5→8→1) usada para medir "calor": quão perto o mouse está do recurso desejado. */
var ReiaNN = (() => {
  const sig = (x) => 1 / (1 + Math.exp(-x));
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const F = 5;

  /** Features: [distância à borda, alinhamento da direção, velocidade de aproximação, velocidade, dentro?] */
  function features(cur, prev, rect, diag) {
    const dx = Math.max(rect.left - cur.x, 0, cur.x - rect.right);
    const dy = Math.max(rect.top - cur.y, 0, cur.y - rect.bottom);
    const dEdge = Math.hypot(dx, dy) / diag;
    const cx = (rect.left + rect.right) / 2, cy = (rect.top + rect.bottom) / 2;
    const tx = cx - cur.x, ty = cy - cur.y, tl = Math.hypot(tx, ty) || 1;
    let align = 0, closing = 0, speed = 0;
    if (prev && cur.t > prev.t) {
      const dt = cur.t - prev.t;
      const vx = (cur.x - prev.x) / dt, vy = (cur.y - prev.y) / dt; // px/ms
      const vl = Math.hypot(vx, vy);
      speed = clamp(vl / 2, 0, 1);
      if (vl > 0.01) { align = (vx * tx + vy * ty) / (vl * tl); closing = clamp((vx * tx + vy * ty) / tl, -1, 1); }
    }
    const inside = cur.x >= rect.left && cur.x <= rect.right && cur.y >= rect.top && cur.y <= rect.bottom ? 1 : 0;
    return [clamp(dEdge, 0, 1), align, closing, speed, inside];
  }

  function init(hidden = 8, seed = 7) {
    let s = seed;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5) * 0.8;
    return {
      w1: Array.from({ length: hidden }, () => Array.from({ length: F }, rnd)),
      b1: Array(hidden).fill(0),
      w2: Array.from({ length: hidden }, rnd),
      b2: 0,
    };
  }

  function predict(f, m) {
    if (!m) { // sem treino: heurística simples (perto = quente)
      return clamp(1 - f[0] * 3, 0, 1) * 0.75 + Math.max(0, f[1]) * 0.25 * (f[3] > 0 ? 1 : 0) + f[4] * 0.3;
    }
    const h = m.w1.map((row, j) => Math.max(0, row.reduce((s, w, i) => s + w * f[i], m.b1[j])));
    return sig(m.w2.reduce((s, w, j) => s + w * h[j], m.b2));
  }

  /** samples: [{f:[...], y:0|1}] — treina com SGD e entropia cruzada binária. */
  function train(samples, { hidden = 8, epochs = 300, lr = 0.1 } = {}, onEpoch) {
    const m = init(hidden);
    const idx = samples.map((_, i) => i);
    for (let e = 0; e < epochs; e++) {
      for (let i = idx.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [idx[i], idx[j]] = [idx[j], idx[i]]; }
      for (const k of idx) {
        const { f, y } = samples[k];
        const z1 = m.w1.map((row, j) => row.reduce((s, w, i) => s + w * f[i], m.b1[j]));
        const h = z1.map((v) => Math.max(0, v));
        const p = sig(m.w2.reduce((s, w, j) => s + w * h[j], m.b2));
        const dz = p - y;
        for (let j = 0; j < hidden; j++) {
          const dh = z1[j] > 0 ? dz * m.w2[j] : 0;
          m.w2[j] -= lr * dz * h[j];
          if (dh) { for (let i = 0; i < F; i++) m.w1[j][i] -= lr * dh * f[i]; m.b1[j] -= lr * dh; }
        }
        m.b2 -= lr * dz;
      }
      if (onEpoch && e % 25 === 0) onEpoch(e);
    }
    return m;
  }

  function accuracy(samples, m) {
    return samples.filter((s) => (predict(s.f, m) > 0.5 ? 1 : 0) === s.y).length / Math.max(1, samples.length);
  }

  /**
   * Cor do halo — escala DIVERGENTE e acessível (segura para daltonismo,
   * padrão ColorBrewer RdYlBu). `w` ∈ [0,1] é o "calor" previsto pela rede
   * (1 = em cima do alvo; 0 = longe). O significado é de acerto, não de
   * temperatura:
   *   w → 0    vermelho  (dando errado / longe do alvo)
   *   w ≈ 0.5  âmbar     (tom intermediário / quase lá)
   *   w → 1    azul      (dando certo / em cima do alvo)
   * Nunca usamos a dupla vermelho↔verde (confunde ~8% dos homens).
   */
  const RAMP = [
    [211, 47, 47],    // 0.00  vermelho    — dando errado
    [238, 138, 66],   // 0.25  laranja
    [245, 194, 66],   // 0.50  âmbar       — intermediário
    [120, 170, 226],  // 0.75  azul-claro
    [37, 109, 235],   // 1.00  azul        — dando certo
  ];
  function color(w) {
    const t = clamp(w, 0, 1) * (RAMP.length - 1);
    const i = Math.min(Math.floor(t), RAMP.length - 2);
    const u = t - i;
    const a = RAMP[i], b = RAMP[i + 1];
    const mix = (k) => Math.round(a[k] + (b[k] - a[k]) * u);
    return `rgb(${mix(0)} ${mix(1)} ${mix(2)})`;
  }

  return { features, init, predict, train, accuracy, color, RAMP, F };
})();
if (typeof module !== 'undefined') module.exports = ReiaNN;

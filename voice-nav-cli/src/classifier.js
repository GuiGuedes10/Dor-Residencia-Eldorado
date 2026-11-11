import path from 'path';
import * as tf from '@tensorflow/tfjs-node';

let model = null;
const classes = ['vestibular','cultura','ciencias','idiomas'];

export async function loadModel() {
  if (!model) {
    const modelPath = path.resolve('model/page_classifier/model.json');
    model = await tf.loadLayersModel('file://' + modelPath);
  }
  return model;
}

function urlToSequence(url) {
  const pathname = new URL(url).pathname.replace(/^\//,'');
  const tokens = pathname.split('/');
  const map = {vestibular:1,cultura:2,ciencias:3,idiomas:4};
  return tokens.map(t=>map[t]||0);
}

/**
 * Classifica a URL atual e devolve a classe.
 */
export async function classifyUrl(url) {
  const m = await loadModel();
  const seq = urlToSequence(url);
  const padded = tf.pad1d(tf.tensor1d(seq,'int32'), [0, Math.max(0,10-seq.length)], 0); // pad to length 10
  const input = tf.expandDims(padded,0);
  const pred = m.predict(input);
  const idx = pred.argMax(-1).dataSync()[0];
  return classes[idx] || 'unknown';
}

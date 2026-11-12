#!/usr/bin/env sh
# Baixa o Transformers.js + ONNX Runtime (~22 MB) para HABILITAR a Fase 2 (embeddings neurais).
# Esses binários NÃO são versionados. Rode este script e recarregue a extensão.
#   sh extension/vendor/baixar.sh
set -e
cd "$(dirname "$0")"
base="https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist"
for f in transformers.min.js ort.bundle.min.mjs ort-wasm-simd-threaded.jsep.mjs ort-wasm-simd-threaded.jsep.wasm; do
  printf "baixando %s... " "$f"
  curl -sL "$base/$f" -o "$f"
  echo "ok ($(wc -c < "$f") bytes)"
done
echo "✅ pronto. Recarregue a extensão em chrome://extensions (o modelo ~100 MB baixa na 1ª consulta)."

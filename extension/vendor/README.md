# vendor/ — Fase 2 (embeddings neurais)

O Chrome MV3 **proíbe carregar código remoto**, então o Transformers.js + ONNX Runtime
precisam ficar **locais aqui**. Esses arquivos (~22 MB) **não são versionados**.

## Habilitar o refino neural (68%)

```sh
sh extension/vendor/baixar.sh
```

Isso baixa, para esta pasta:

| Arquivo | Tamanho | O que é |
|---|---|---|
| `transformers.min.js` | ~0,9 MB | biblioteca Transformers.js |
| `ort.bundle.min.mjs` | ~0,4 MB | ONNX Runtime Web (JS) |
| `ort-wasm-simd-threaded.jsep.mjs` | ~45 KB | glue do WASM |
| `ort-wasm-simd-threaded.jsep.wasm` | ~21,6 MB | runtime WASM |

Depois **recarregue** a extensão em `chrome://extensions`. Na 1ª consulta, o **modelo**
(`paraphrase-multilingual-MiniLM-L12-v2`, ~100 MB) é baixado do Hugging Face e fica em cache.

## Sem esses arquivos?

A extensão funciona normalmente com o ranqueador **TF-IDF** (`../tfidf.js`, 56%); o refino
neural simplesmente não entra. Nada quebra.

> Para publicar na Chrome Web Store, versione estes arquivos (as lojas não aceitam baixá-los
> em runtime) e mantenha o modelo sendo buscado do HF ou também empacotado.

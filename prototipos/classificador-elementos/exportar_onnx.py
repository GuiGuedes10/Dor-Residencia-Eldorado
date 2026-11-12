#!/usr/bin/env python3
"""
Treina o encoder nos pares reais, salva e EXPORTA para ONNX quantizado (int8) — o formato que
a extensão (Transformers.js) carrega. Verifica que o ONNX reproduz o success@k do modelo torch.

Pré-requisito:  pip install "optimum[onnxruntime]" onnx
Uso:            python3 prototipos/classificador-elementos/exportar_onnx.py
Saída:          modelo-finetunado/ (config + tokenizer + onnx/) — NÃO versionado (~100 MB).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from fine_tune import MODELO_BASE, carregar_elementos, carregar_split, success_at_k  # noqa: E402

random.seed(42)
QUANT = "arm64"  # arquitetura de quantização (dev); o int8 resultante roda no WASM do navegador


def treinar(model, treino, els):
    import torch
    from sentence_transformers import losses
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    dev = model.device
    pares = [(intent, els[idx]["rich"]) for intent, idx in treino]
    model.train()
    for _ in range(3):
        random.shuffle(pares)
        for b in range(0, len(pares), 32):
            lote = pares[b:b + 32]
            if len(lote) < 2:
                continue
            fa = {k: v.to(dev) for k, v in model.tokenize([a for a, _ in lote]).items() if hasattr(v, "to")}
            fp = {k: v.to(dev) for k, v in model.tokenize([p for _, p in lote]).items() if hasattr(v, "to")}
            opt.zero_grad()
            loss_fn([fa, fp], torch.zeros(len(lote), device=dev)).backward()
            opt.step()
    model.eval()


def main():
    from sentence_transformers import SentenceTransformer
    els = carregar_elementos()
    el_texts = [e["rich"] for e in els]
    treino = carregar_split("dataset_treino.json", els)
    teste = carregar_split("dataset_teste.json", els)
    if not treino or not teste:
        print("Rode antes: python3 montar_dataset.py")
        return

    print(f"Treino: {len(treino)} | teste: {len(teste)}  — treinando {MODELO_BASE} ...")
    model = SentenceTransformer(MODELO_BASE)
    treinar(model, treino, els)
    f1, f3 = success_at_k(model, teste, el_texts)
    print(f"torch fine-tunado : succ@1={f1:.0%}  succ@3={f3:.0%}")

    dest = AQUI / "modelo-finetunado"
    model.save(str(dest))
    print(f"modelo torch salvo em {dest}")

    # --- export ONNX quantizado (precisa de optimum + onnxruntime) ---
    try:
        from sentence_transformers import export_dynamic_quantized_onnx_model
        CPU = {"provider": "CPUExecutionProvider"}  # macOS: o CoreML EP falha em nós do MiniLM
        print("Exportando para ONNX (backend onnx) + quantização int8 ...")
        m_onnx = SentenceTransformer(str(dest), backend="onnx", model_kwargs=CPU)
        export_dynamic_quantized_onnx_model(m_onnx, QUANT, str(dest))
        onnx_dir = dest / "onnx"
        arqs = sorted(p.name for p in onnx_dir.glob("*.onnx")) if onnx_dir.exists() else []
        quant = next((a for a in arqs if "qint8" in a or "quant" in a), None)
        if quant:  # verifica o MESMO arquivo que vai para a extensão (int8), em CPU
            m_q = SentenceTransformer(str(dest), backend="onnx",
                                      model_kwargs={**CPU, "file_name": f"onnx/{quant}"})
            o1, o3 = success_at_k(m_q, teste, el_texts)
            print(f"ONNX int8 ({quant}) : succ@1={o1:.0%}  succ@3={o3:.0%}  (deve ~bater com o torch)")
        print(f"ONNX em {onnx_dir}  →  {arqs}")
        print(f"\nDeploy: copie modelo-finetunado/ → extension/vendor/modelo/reia-finetunado/ e renomeie")
        print(f"        onnx/{quant or 'model_qint8_arm64.onnx'} → onnx/model_quantized.onnx (Transformers.js).")
    except Exception as e:
        print(f"\n⚠️  Export ONNX indisponível: {type(e).__name__}: {e}")
        print("Instale:  pip install \"optimum[onnxruntime]\" onnx   e rode de novo.")


if __name__ == "__main__":
    main()

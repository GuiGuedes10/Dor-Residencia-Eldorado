#!/usr/bin/env python3
"""Treina a SUA rede neural de intenção (MLP bag-of-words, só biblioteca padrão).

Dados: training/intents.tsv  (intenção<TAB>frase). Adicione linhas e rode `npm run train`.
Saída: models/intent.json (usado pelo Node em src/intent.js).
"""
import json
import math
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
random.seed(42)


def tokens(text: str):
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", t)


rows = []
for line in (ROOT / "training" / "intents.tsv").read_text(encoding="utf-8").splitlines():
    if "\t" in line:
        label, phrase = line.split("\t", 1)
        rows.append((label.strip(), phrase.strip()))

classes = sorted({l for l, _ in rows})
vocab = sorted({w for _, p in rows for w in tokens(p)})
idx = {w: i for i, w in enumerate(vocab)}
V, H, C = len(vocab), 24, len(classes)


def vec(text):
    x = [0.0] * V
    for w in tokens(text):
        if w in idx:
            x[idx[w]] = 1.0
    return x


# validação: 1 de cada 5 frases por classe
random.shuffle(rows)
val = [r for i, r in enumerate(rows) if i % 5 == 0]
train = [r for i, r in enumerate(rows) if i % 5 != 0]

w1 = [[random.gauss(0, 0.3) for _ in range(V)] for _ in range(H)]
b1 = [0.0] * H
w2 = [[random.gauss(0, 0.3) for _ in range(H)] for _ in range(C)]
b2 = [0.0] * C


def forward(x):
    h = [max(0.0, sum(w1[j][i] * x[i] for i in range(V) if x[i]) + b1[j]) for j in range(H)]
    z = [sum(w2[k][j] * h[j] for j in range(H)) + b2[k] for k in range(C)]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return h, [v / s for v in e]


def acc(data):
    ok = sum(1 for l, p in data if classes[max(range(C), key=lambda k: forward(vec(p))[1][k])] == l)
    return ok / max(1, len(data))


lr = 0.1
for epoch in range(400):
    random.shuffle(train)
    for label, phrase in train:
        x = vec(phrase)
        h, p = forward(x)
        y = classes.index(label)
        dz = [p[k] - (1.0 if k == y else 0.0) for k in range(C)]
        dh = [sum(w2[k][j] * dz[k] for k in range(C)) if h[j] > 0 else 0.0 for j in range(H)]
        for k in range(C):
            for j in range(H):
                w2[k][j] -= lr * dz[k] * h[j]
            b2[k] -= lr * dz[k]
        for j in range(H):
            if dh[j]:
                for i in range(V):
                    if x[i]:
                        w1[j][i] -= lr * dh[j]
                b1[j] -= lr * dh[j]

print(f"classes={classes} vocab={V} treino={len(train)} validação={len(val)}")
print(f"acurácia treino={acc(train):.2f} validação={acc(val):.2f}")

out = ROOT / "models" / "intent.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"classes": classes, "vocab": vocab, "w1": w1, "b1": b1, "w2": w2, "b2": b2}), encoding="utf-8")
print(f"salvo em {out}")

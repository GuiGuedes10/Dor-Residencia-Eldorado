#!/usr/bin/env python3
"""
Monta o dataset REAL a partir de pares coletados à mão/da web, validando cada alvo contra o
corpus e produzindo um split treino/teste. Evita o vazamento do experimento anterior: aqui
treino e teste são ambos pares REAIS (novas frases para alvos conhecidos), sem augmentation.

Entradas (pares reais):
  - coleta.csv        (opcional): colunas intencao,alvo,fonte,tipo  (copie coleta_template.csv)
  - dataset_pesquisa.json       : os pares da pesquisa (presente_no_corpus=true)

Saídas:
  - dataset_treino.json / dataset_teste.json  (split 80/20, formato do avaliar.py)

Uso:  python3 prototipos/classificador-elementos/montar_dataset.py
Depois: python3 prototipos/classificador-elementos/avaliar.py dataset_teste.json
"""
from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
from pathlib import Path

AQUI = Path(__file__).resolve().parent
random.seed(42)


def flat(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", s))


def carregar_elementos() -> list[dict]:
    els, seen = [], set()
    for f in ("elementos_cursos.json", "elementos_portal.json"):
        p = AQUI / f
        if not p.exists():
            continue
        for l in json.loads(p.read_text(encoding="utf-8")).get("links", []):
            texto = " ".join((l.get("text") or "").split())
            url = l.get("url") or ""
            if len(texto) < 3 or url in seen:
                continue
            seen.add(url)
            slug = re.sub(r"^https?://[^/]+", "", url)
            busca = f"{texto} {l.get('contexto', '')} {re.sub(r'[^a-z0-9]+', ' ', slug, flags=re.I)}"
            els.append({"slug": slug, "busca": busca})
    return els


def resolve(alvo: str, els: list[dict]):
    a = flat(alvo)
    cands = [e for e in els if a and a in flat(e["busca"])]
    return min(cands, key=lambda e: len(e["busca"])) if cands else None


def carregar_pares() -> list[dict]:
    pares = []
    csvp = AQUI / "coleta.csv"
    if csvp.exists():
        with open(csvp, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                intent = (row.get("intencao") or "").strip()
                alvo = (row.get("alvo") or "").strip()
                if intent and alvo and not intent.startswith("#"):
                    pares.append({"intencao": intent, "alvo": alvo,
                                  "fonte": (row.get("fonte") or "coleta").strip(),
                                  "tipo": (row.get("tipo") or "semantico").strip()})
    pj = AQUI / "dataset_pesquisa.json"
    if pj.exists():
        for c in json.loads(pj.read_text(encoding="utf-8")).get("casos", []):
            if c.get("presente_no_corpus", True):
                pares.append({"intencao": c["intencao"], "alvo": c["alvo"],
                              "fonte": c.get("fonte", "pesquisa"), "tipo": c.get("tipo", "semantico")})
    return pares


def main():
    els = carregar_elementos()
    if not els:
        print("Sem corpus (elementos_*.json). Rode a partir da pasta do protótipo.")
        return
    pares = carregar_pares()

    vistos, validos, invalidos = set(), [], []
    for p in pares:
        k = flat(p["intencao"])
        if not k or k in vistos:
            continue
        if resolve(p["alvo"], els) is None:
            invalidos.append(p)
            continue
        vistos.add(k)
        validos.append(p)

    random.shuffle(validos)
    corte = max(1, round(len(validos) * 0.2)) if len(validos) >= 2 else 0
    teste, treino = validos[:corte], validos[corte:]

    (AQUI / "dataset_treino.json").write_text(json.dumps({"casos": treino}, ensure_ascii=False, indent=1), encoding="utf-8")
    (AQUI / "dataset_teste.json").write_text(json.dumps({"casos": teste}, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Pares reais válidos: {len(validos)}  (inválidos/alvo ausente no corpus: {len(invalidos)})")
    print(f"Split: treino {len(treino)} | teste {len(teste)}")
    if invalidos:
        print("\n⚠️  Alvos não encontrados no corpus (revisar o 'alvo' ou crawlear o elemento):")
        for p in invalidos[:12]:
            print(f"   \"{p['intencao']}\"  →  {p['alvo']}")
    print("\nSaídas: dataset_treino.json, dataset_teste.json")
    print("Avalie o teste:  python3 avaliar.py dataset_teste.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Gera um dataset GRANDE (centenas de exemplos) de intenção → link-alvo, a partir dos
links reais da PUC + a curadoria de gatilhos (gatilhos.json), via modelos de frase.

Saídas:
  dataset_treino.json  (~80%)   e   dataset_teste.json (~20%, estratificado por alvo)

Uso:
    python3 prototipos/classificador-elementos/gerar_dataset.py
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
random.seed(42)

# Modelos de frase por tipo de gatilho
T_CURSO = ["{}", "quero fazer {}", "curso de {}", "quero cursar {}", "quero estudar {}",
           "informações sobre {}", "quero me formar em {}", "tenho interesse em {}"]
T_PROFISSAO = ["quero ser {}", "quero trabalhar como {}", "pretendo ser {}", "sonho em ser {}"]
T_ACAO = ["quero {}", "gostaria de {}", "tenho vontade de {}", "meu objetivo é {}", "pretendo {}"]


def limpar_nome(texto: str) -> str:
    t = re.sub(r"\(.*?\)", "", texto)                                   # remove parênteses
    t = re.sub(r"\s*[–-]\s*(Bacharelado|Licenciatura|Ênfase.*)$", "", t, flags=re.I)
    t = re.sub(r"\s*[–-]\s*(Bacharelado|Licenciatura)\b", "", t, flags=re.I)
    t = re.sub(r"^Novo\s+", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" -–:").lower()
    return t


def ultimo_segmento(url: str) -> str:
    slug = re.sub(r"^https?://[^/]+", "", url)
    partes = [p for p in slug.split("/") if p]
    return partes[-1] if partes else ""


def main():
    links = json.loads((RAIZ / "voice-nav-cli/data/vestibular.puc-campinas.edu.br.json").read_text(encoding="utf-8"))["links"]
    gat = json.loads((AQUI / "gatilhos.json").read_text(encoding="utf-8"))["alvos"]

    exemplos = []  # (intencao, alvo, tipo)
    vistos = set()

    def add(intencao: str, alvo: str, tipo: str):
        intencao = re.sub(r"\s+", " ", intencao).strip()
        chave = intencao.lower()
        if chave in vistos or len(intencao) < 3:
            return
        vistos.add(chave)
        exemplos.append({"intencao": intencao, "alvo": alvo, "tipo": tipo})

    for l in links:
        texto = " ".join((l.get("text") or "").split())
        url = l.get("url") or ""
        if len(texto) < 3 or not url:
            continue
        alvo = ultimo_segmento(url)
        nome = limpar_nome(texto)
        curado = gat.get(alvo, {})

        # 1) gatilhos de CURSO: nome do link + extras curados -> templates de curso (lexical)
        cursos = [nome] + curado.get("curso", [])
        for g in cursos:
            for tmpl in T_CURSO:
                add(tmpl.format(g), alvo, "lexical")

        # 2) PROFISSÃO (semântico)
        for g in curado.get("profissao", []):
            for tmpl in T_PROFISSAO:
                add(tmpl.format(g), alvo, "semantico")

        # 3) AÇÃO/objetivo (semântico)
        for g in curado.get("acao", []):
            for tmpl in T_ACAO:
                add(tmpl.format(g), alvo, "semantico")

    # Split estratificado por alvo (80/20)
    por_alvo: dict[str, list] = {}
    for e in exemplos:
        por_alvo.setdefault(e["alvo"], []).append(e)
    treino, teste = [], []
    for alvo, lst in por_alvo.items():
        random.shuffle(lst)
        corte = max(1, round(len(lst) * 0.2)) if len(lst) >= 2 else 0
        teste.extend(lst[:corte])
        treino.extend(lst[corte:])
    random.shuffle(treino)
    random.shuffle(teste)

    (AQUI / "dataset_treino.json").write_text(
        json.dumps({"casos": treino}, ensure_ascii=False, indent=0), encoding="utf-8")
    (AQUI / "dataset_teste.json").write_text(
        json.dumps({"casos": teste}, ensure_ascii=False, indent=0), encoding="utf-8")

    n_lex = sum(e["tipo"] == "lexical" for e in exemplos)
    n_sem = sum(e["tipo"] == "semantico" for e in exemplos)
    print(f"Total de exemplos únicos: {len(exemplos)}")
    print(f"  lexicais:  {n_lex}")
    print(f"  semânticos:{n_sem}")
    print(f"  alvos distintos: {len(por_alvo)}")
    print(f"Split -> treino: {len(treino)}  |  teste: {len(teste)}")
    print("\nTop 5 alvos por nº de exemplos:")
    for alvo, lst in sorted(por_alvo.items(), key=lambda kv: -len(kv[1]))[:5]:
        print(f"  {alvo:32s} {len(lst)}")
    print("\nAmostra aleatória:")
    for e in random.sample(exemplos, 8):
        print(f"  [{e['tipo'][:3]}] \"{e['intencao']}\"  -> {e['alvo']}")


if __name__ == "__main__":
    main()

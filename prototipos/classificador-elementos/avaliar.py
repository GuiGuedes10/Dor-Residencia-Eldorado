#!/usr/bin/env python3
"""
Avalia o protótipo do classificador de elementos sobre links REAIS da PUC-Campinas.

Compara, pela mesma métrica (success@k / MRR):
  - baseline lexical (o que o rankLinks da extensão faz hoje)
  - embeddings TF-IDF (palavras)
  - embeddings TF-IDF (palavras + n-gramas de caractere)
  - sentence-transformers (se instalado)

Uso:
    python3 prototipos/classificador-elementos/avaliar.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(AQUI))

from ranqueadores import (  # noqa: E402
    Elemento,
    TfidfEmbedder,
    carregar_sentence_transformer,
    ranquear_embeddings,
    ranquear_lexico,
)


def carregar_elementos() -> list[Elemento]:
    dados = json.loads((RAIZ / "voice-nav-cli/data/vestibular.puc-campinas.edu.br.json").read_text(encoding="utf-8"))
    out, visto = [], set()
    for l in dados["links"]:
        texto = " ".join((l.get("text") or "").split())
        url = l.get("url") or ""
        if len(texto) < 3 or url in visto:
            continue
        visto.add(url)
        out.append(Elemento(texto=texto, url=url))
    return out


def resolver_alvo(alvo: str, elementos: list[Elemento]) -> Elemento | None:
    """Elemento cujo slug contém `alvo`, escolhendo o slug mais curto (mais específico/base)."""
    cands = [e for e in elementos if alvo in e.slug]
    return min(cands, key=lambda e: len(e.slug)) if cands else None


def posicao_do_alvo(ranking, alvo_el: Elemento) -> int:
    for i, (el, _score) in enumerate(ranking, start=1):
        if el.url == alvo_el.url:
            return i
    return 10**9


def avaliar(nome, funcao_rank, casos, elementos):
    s1 = s3 = 0
    rr = 0.0
    por_tipo = {"lexical": [0, 0], "semantico": [0, 0]}  # [acertos@1, total]
    falhas = []
    for c in casos:
        alvo_el = resolver_alvo(c["alvo"], elementos)
        if alvo_el is None:
            continue
        ranking = funcao_rank(c["intencao"], elementos)
        pos = posicao_do_alvo(ranking, alvo_el)
        s1 += pos == 1
        s3 += pos <= 3
        rr += 1.0 / pos
        t = por_tipo[c["tipo"]]
        t[0] += pos == 1
        t[1] += 1
        if pos != 1:
            falhas.append((c["intencao"], pos, ranking[0][0].texto))
    n = len(casos)
    return {
        "nome": nome,
        "s1": s1 / n, "s3": s3 / n, "mrr": rr / n,
        "lex": por_tipo["lexical"], "sem": por_tipo["semantico"],
        "falhas": falhas,
    }


def main():
    elementos = carregar_elementos()
    # Avalia o dataset passado por argumento; senão, o conjunto de exemplo (dataset.json).
    nome_ds = sys.argv[1] if len(sys.argv) > 1 else "dataset.json"
    casos = json.loads((AQUI / nome_ds).read_text(encoding="utf-8"))["casos"]
    print(f"Elementos (links reais da PUC): {len(elementos)}  |  dataset: {nome_ds}  |  casos: {len(casos)}\n")

    estrategias = [
        ("baseline lexical (rankLinks)", lambda q, els: ranquear_lexico(q, els)),
        ("TF-IDF palavras",              lambda q, els: ranquear_embeddings(q, els, TfidfEmbedder(usar_char=False))),
        ("TF-IDF palavras + char n-gram", lambda q, els: ranquear_embeddings(q, els, TfidfEmbedder(usar_char=True))),
    ]
    st = carregar_sentence_transformer()
    if st is not None:
        estrategias.append(("sentence-transformers", lambda q, els: ranquear_embeddings(q, els, st)))

    resultados = [avaliar(nome, fn, casos, elementos) for nome, fn in estrategias]

    print(f"{'estratégia':32s} {'succ@1':>7s} {'succ@3':>7s} {'MRR':>6s} {'lex@1':>7s} {'sem@1':>7s}")
    print("-" * 72)
    for r in resultados:
        lex = f"{r['lex'][0]}/{r['lex'][1]}"
        sem = f"{r['sem'][0]}/{r['sem'][1]}"
        print(f"{r['nome']:32s} {r['s1']:7.0%} {r['s3']:7.0%} {r['mrr']:6.2f} {lex:>7s} {sem:>7s}")

    if st is None:
        print("\n(sentence-transformers não instalado — `pip install sentence-transformers` para o salto semântico)")

    # Casos onde o melhor ranqueador disponível ainda erra o top-1 (motivam o embedder neural)
    melhor = resultados[-1]
    if melhor["falhas"]:
        print(f"\nAinda erram o top-1 em «{melhor['nome']}» ({len(melhor['falhas'])}):")
        for intent, pos, topo in melhor["falhas"][:8]:
            print(f"  pos {pos:>2}: \"{intent}\"  → topo veio: {topo[:40]}")


if __name__ == "__main__":
    main()

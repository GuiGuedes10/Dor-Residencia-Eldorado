#!/usr/bin/env python3
"""
Experimento de fine-tune de um encoder pt-BR/multilíngue para o nosso domínio
(intenção → elemento da página), com MultipleNegativesRankingLoss.

Honestidade metodológica:
- Dados rotulados REAIS são poucos. Para ter volume de treino, usamos *data augmentation
  ancorada* em mapeamentos reais (nome do curso + profissão/ação → elemento). NÃO são logs
  de usuários; servem para adaptar o modelo ao vocabulário do domínio.
- Split 80/20 por par: treino e teste são pares DISJUNTOS (mesma distribuição).
- Comparamos o MESMO teste em: zero-shot (modelo base) vs fine-tunado. Se não melhorar,
  reportamos assim mesmo — é um resultado válido.

Uso:  python3 prototipos/classificador-elementos/fine_tune.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
random.seed(42)

MODELO_BASE = "paraphrase-multilingual-MiniLM-L12-v2"

# Gatilhos curados (profissão/ação) por alvo — ancorados em guias reais (Quero Bolsa etc.).
GATILHOS = {
    "direito": {"prof": ["advogado", "juiz", "promotor", "delegado"], "acao": ["trabalhar com leis", "atuar na justiça"]},
    "medicina": {"prof": ["médico", "clínico geral"], "acao": ["curar pessoas", "trabalhar num hospital"]},
    "medicina-veterinaria": {"prof": ["veterinário"], "acao": ["cuidar de animais", "tratar bichos de estimação"]},
    "odontologia": {"prof": ["dentista"], "acao": ["cuidar dos dentes", "tratar sorrisos"]},
    "psicologia": {"prof": ["psicólogo"], "acao": ["cuidar da saúde mental", "fazer terapia com pacientes"]},
    "enfermagem": {"prof": ["enfermeiro"], "acao": ["cuidar de pacientes"]},
    "nutricao": {"prof": ["nutricionista"], "acao": ["montar dietas", "cuidar da alimentação"]},
    "fisioterapia": {"prof": ["fisioterapeuta"], "acao": ["reabilitar pacientes", "tratar lesões"]},
    "farmacia": {"prof": ["farmacêutico"], "acao": ["fazer remédios", "trabalhar numa farmácia"]},
    "biomedicina": {"prof": ["biomédico"], "acao": ["fazer análises clínicas", "trabalhar em laboratório"]},
    "gastronomia": {"prof": ["chef de cozinha", "cozinheiro"], "acao": ["cozinhar profissionalmente", "abrir um restaurante"]},
    "jornalismo": {"prof": ["jornalista", "repórter"], "acao": ["escrever notícias", "trabalhar na imprensa"]},
    "publicidade-e-propaganda": {"prof": ["publicitário"], "acao": ["criar campanhas de propaganda", "trabalhar com marketing"]},
    "design-de-games": {"prof": ["designer de games"], "acao": ["criar jogos", "desenvolver videogames"]},
    "design-de-moda": {"prof": ["estilista"], "acao": ["criar roupas", "desenhar coleções de moda"]},
    "arquitetura-e-urbanismo": {"prof": ["arquiteto"], "acao": ["projetar casas e prédios", "planejar cidades"]},
    "engenharia-de-software": {"prof": ["programador", "desenvolvedor de software"], "acao": ["programar sistemas", "desenvolver aplicativos"]},
    "ciberseguranca": {"prof": ["analista de segurança"], "acao": ["proteger sistemas contra hackers", "trabalhar com segurança digital"]},
    "ciencia-de-dados": {"prof": ["cientista de dados"], "acao": ["trabalhar com inteligência artificial", "analisar dados"]},
    "engenharia-agronomica": {"prof": ["agrônomo"], "acao": ["cuidar de plantações", "trabalhar com agricultura"]},
    "pedagogia": {"prof": ["pedagogo", "professor"], "acao": ["ensinar crianças", "trabalhar com educação"]},
    "ciencias-contabeis": {"prof": ["contador"], "acao": ["cuidar de impostos e balanços"]},
    "relacoes-internacionais": {"prof": ["diplomata"], "acao": ["trabalhar com diplomacia"]},
    "servico-social": {"prof": ["assistente social"], "acao": ["ajudar comunidades carentes"]},
}
T_CURSO = ["{}", "quero fazer {}", "curso de {}", "quero estudar {}", "quero me formar em {}"]
T_PROF = ["quero ser {}", "quero trabalhar como {}", "pretendo ser {}", "sonho em ser {}"]
T_ACAO = ["quero {}", "gostaria de {}", "pretendo {}", "meu objetivo é {}"]


def tokens(s):
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    import unicodedata
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.split()


def flat(s):
    return " ".join(tokens(s))


def limpar_nome(texto):
    t = re.sub(r"\(.*?\)", "", texto)
    t = re.sub(r"\s*[–-]\s*(Bacharelado|Licenciatura|Ênfase.*)$", "", t, flags=re.I)
    t = re.sub(r"^Novo\s+", "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip(" -–:").lower()


def carregar_elementos():
    els = []
    seen = set()
    for f in ["elementos_cursos.json", "elementos_portal.json"]:
        for l in json.loads((AQUI / f).read_text(encoding="utf-8")).get("links", []):
            texto = " ".join((l.get("text") or "").split())
            url = l.get("url") or ""
            if len(texto) < 3 or url in seen:
                continue
            seen.add(url)
            slug = re.sub(r"^https?://[^/]+", "", url)
            rich = f"{texto} {l.get('contexto','')} {re.sub(r'[^a-z0-9]+',' ', slug, flags=re.I)}".strip()
            els.append({"texto": texto, "slug": slug, "rich": rich})
    return els


def idx_por_slug(els, alvo):
    a = flat(alvo)
    cands = [i for i, e in enumerate(els) if a and a in flat(e["rich"])]
    return min(cands, key=lambda i: len(els[i]["rich"])) if cands else None


def gerar_pares(els):
    """Pares (intenção, índice_do_elemento_alvo)."""
    pares = []
    # 1) nome do curso -> elemento (todos os cursos)
    for i, e in enumerate(els):
        if "/curso/" in e["slug"]:
            nome = limpar_nome(e["texto"])
            for t in T_CURSO:
                pares.append((t.format(nome), i))
    # 2) profissão/ação -> elemento (alvos curados)
    for alvo, g in GATILHOS.items():
        i = idx_por_slug(els, alvo)
        if i is None:
            continue
        for p in g.get("prof", []):
            for t in T_PROF:
                pares.append((t.format(p), i))
        for a in g.get("acao", []):
            for t in T_ACAO:
                pares.append((t.format(a), i))
    # dedup por intenção
    vistos, out = set(), []
    for intent, i in pares:
        k = flat(intent)
        if k and k not in vistos:
            vistos.add(k)
            out.append((intent, i))
    return out


def carregar_split(nome, els):
    """Carrega um split real (dataset_treino.json / dataset_teste.json) como (intenção, índice)."""
    casos = json.loads((AQUI / nome).read_text(encoding="utf-8")).get("casos", [])
    out = []
    for c in casos:
        i = idx_por_slug(els, c["alvo"])
        if i is not None:
            out.append((c["intencao"], i))
    return out


def carregar_reais(els):
    """Pares REAIS (dataset_pesquisa.json) como teste externo — difícil (sinônimos puros).
    Ressalva: há overlap de mapeamento com a augmentation, então o número fine-tunado
    pode estar otimista; serve para ver direção, não como medida limpa."""
    try:
        casos = json.loads((AQUI / "dataset_pesquisa.json").read_text(encoding="utf-8"))["casos"]
    except Exception:
        return []
    out = []
    for c in casos:
        i = idx_por_slug(els, c["alvo"])
        if i is not None:
            out.append((c["intencao"], i))
    return out


def success_at_k(model, test, el_texts):
    import numpy as np
    emb = model.encode(el_texts, normalize_embeddings=True)
    s1 = s3 = 0
    for intent, tgt in test:
        q = model.encode([intent], normalize_embeddings=True)[0]
        sims = emb @ q
        order = list(np.argsort(sims)[::-1])
        pos = order.index(tgt) + 1
        s1 += pos == 1
        s3 += pos <= 3
    n = len(test)
    return s1 / n, s3 / n


def main():
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except Exception as e:
        print("sentence-transformers não disponível:", e)
        sys.exit(1)

    els = carregar_elementos()
    el_texts = [e["rich"] for e in els]
    # Experimento LIMPO: treino e teste são pares REAIS disjuntos (montar_dataset.py).
    treino = carregar_split("dataset_treino.json", els)
    teste = carregar_split("dataset_teste.json", els)
    if not treino or not teste:
        print("Rode antes: python3 montar_dataset.py  (gera dataset_treino/teste.json)")
        return
    print(f"Elementos: {len(els)} | treino REAL: {len(treino)} | teste REAL: {len(teste)}\n")

    print(f"Carregando modelo base: {MODELO_BASE} ...")
    model = SentenceTransformer(MODELO_BASE)

    def report(tag):
        a1, a3 = success_at_k(model, teste, el_texts)
        print(f"{tag:12s}: succ@1={a1:.0%}  succ@3={a3:.0%}")
        return a1

    z = report("ZERO-SHOT")

    # Loop de treino manual (sem depender da lib `datasets`), com a mesma loss contrastiva.
    import torch
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    device = model.device
    pares_tr = [(intent, els[i]["rich"]) for intent, i in treino]
    print("\nTreinando (MultipleNegativesRankingLoss, 3 épocas, loop manual)...")
    model.train()
    for ep in range(3):
        random.shuffle(pares_tr)
        for b in range(0, len(pares_tr), 32):
            lote = pares_tr[b:b + 32]
            if len(lote) < 2:
                continue
            fa = {k: v.to(device) for k, v in model.tokenize([a for a, _ in lote]).items() if hasattr(v, "to")}
            fp = {k: v.to(device) for k, v in model.tokenize([p for _, p in lote]).items() if hasattr(v, "to")}
            optimizer.zero_grad()
            lv = loss_fn([fa, fp], torch.zeros(len(lote), device=device))
            lv.backward()
            optimizer.step()
    model.eval()
    print()
    f = report("FINE-TUNADO")
    print(f"\nΔ succ@1 (teste real): {f - z:+.0%}")


if __name__ == "__main__":
    main()

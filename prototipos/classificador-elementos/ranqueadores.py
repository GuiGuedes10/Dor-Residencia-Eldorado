"""
Ranqueadores de elementos da página (protótipo do classificador de elementos).

Contrato do protótipo
---------------------
Entrada:
  - `consulta`: str  — a intenção do usuário em linguagem natural
  - `elementos`: list[Elemento] — os elementos interativos extraídos da página
      (texto do link/botão + URL/slug; no futuro: aria-label, title, classes…)
Saída:
  - lista de (Elemento, score) ordenada do mais provável alvo para o menos provável

Três estratégias, com a MESMA interface, para comparar de forma justa:
  1. `ranquear_lexico`      — baseline: interseção de palavras (o que o rankLinks da extensão faz hoje).
  2. `TfidfEmbedder`        — embeddings TF-IDF (palavras [+ n-gramas de caractere]), roda sem dependências.
  3. `SentenceTransformerEmbedder` — embeddings neurais reais (opcional; exige `sentence-transformers`).

Só a estratégia 3 captura sinônimos de verdade (advogado→direito). As 1 e 2 são
limitadas à forma das palavras — por isso o salto de qualidade depende dela.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple


@dataclass
class Elemento:
    texto: str
    url: str = ""

    @property
    def slug(self) -> str:
        return re.sub(r"^https?://[^/]+", "", self.url)


# ---------------------------------------------------------------------------
# Tokenização (igual à da extensão: NFD, minúsculas, alfanumérico)
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _norm(s))


def _char_ngrams(palavra: str, n_min: int = 3, n_max: int = 4) -> List[str]:
    p = f"<{palavra}>"
    out = []
    for n in range(n_min, n_max + 1):
        for i in range(len(p) - n + 1):
            out.append(f"#{p[i:i+n]}")
    return out


# ---------------------------------------------------------------------------
# 1. Baseline lexical — porta do rankLinks (interseção de palavras normalizada)
# ---------------------------------------------------------------------------
def ranquear_lexico(consulta: str, elementos: List[Elemento]) -> List[Tuple[Elemento, float]]:
    q = set(tokens(consulta))
    scored = []
    for el in elementos:
        alvo = tokens(el.texto + " " + el.slug)
        hit = sum(1 for w in alvo if w in q)
        score = hit / math.sqrt(len(alvo) or 1)
        scored.append((el, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# 2. Embeddings TF-IDF esparso (palavras + bigramas [+ n-gramas de caractere])
# ---------------------------------------------------------------------------
class Embedder(Protocol):
    def fit(self, documentos: List[str]) -> "Embedder": ...
    def vetor(self, texto: str) -> Dict[str, float]: ...


class TfidfEmbedder:
    """TF-IDF esparso e determinístico, sem dependências externas.

    features = palavras (≥3) + bigramas de palavra [+ n-gramas de caractere, se `usar_char`].
    Os n-gramas de caractere aproximam variações morfológicas (médico↔medicina,
    jornalista↔jornalismo) que a interseção de palavras não pega.
    """

    def __init__(self, usar_char: bool = True):
        self.usar_char = usar_char
        self.idf: Dict[str, float] = {}
        self._fitted = False

    def _features(self, texto: str) -> List[str]:
        ws = [w for w in tokens(texto) if len(w) >= 3]
        feats = list(ws)
        feats += [f"{ws[i]}_{ws[i+1]}" for i in range(len(ws) - 1)]  # bigramas
        if self.usar_char:
            for w in ws:
                feats += _char_ngrams(w)
        return feats

    def fit(self, documentos: List[str]) -> "TfidfEmbedder":
        n = len(documentos)
        df: Dict[str, int] = {}
        for doc in documentos:
            for f in set(self._features(doc)):
                df[f] = df.get(f, 0) + 1
        self.idf = {f: math.log((n + 1.0) / (c + 1.0)) + 1.0 for f, c in df.items()}
        self._fitted = True
        return self

    def vetor(self, texto: str) -> Dict[str, float]:
        vec: Dict[str, float] = {}
        for f in self._features(texto):
            # termos fora do vocabulário de treino recebem idf neutro (1.0)
            vec[f] = vec.get(f, 0.0) + self.idf.get(f, 1.0 if self._fitted else 1.0)
        norma = math.sqrt(sum(v * v for v in vec.values()))
        if norma > 1e-9:
            vec = {k: v / norma for k, v in vec.items()}
        return vec


def cosseno_esparso(a: Dict[str, float], b: Dict[str, float]) -> float:
    menor, maior = (a, b) if len(a) <= len(b) else (b, a)
    return sum(v * maior.get(k, 0.0) for k, v in menor.items())


# ---------------------------------------------------------------------------
# 3. Embeddings neurais reais (opcional) — sentence-transformers
# ---------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    """Adaptador para sentence-transformers (modelo multilíngue p/ pt-BR).

    Só é usado se a lib estiver instalada:
        pip install sentence-transformers
    """

    def __init__(self, modelo: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer  # erro claro se ausente
        self._m = SentenceTransformer(modelo)

    def fit(self, documentos: List[str]) -> "SentenceTransformerEmbedder":
        return self  # modelo pré-treinado, não precisa de fit

    def vetor(self, texto: str) -> Dict[str, float]:
        emb = self._m.encode([texto], normalize_embeddings=True)[0]
        return {f"d{i}": float(x) for i, x in enumerate(emb)}


# ---------------------------------------------------------------------------
# Ranqueador genérico por embeddings (qualquer Embedder acima)
# ---------------------------------------------------------------------------
def ranquear_embeddings(
    consulta: str, elementos: List[Elemento], embedder: Embedder
) -> List[Tuple[Elemento, float]]:
    embedder.fit([el.texto + " " + el.slug.replace("/", " ") for el in elementos])
    qv = embedder.vetor(consulta)
    scored = []
    for el in elementos:
        ev = embedder.vetor(el.texto + " " + el.slug.replace("/", " "))
        scored.append((el, cosseno_esparso(qv, ev)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def carregar_sentence_transformer() -> Optional[SentenceTransformerEmbedder]:
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return None

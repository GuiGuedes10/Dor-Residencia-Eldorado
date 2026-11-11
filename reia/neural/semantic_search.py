"""
Semantic Search & Vector Database Subsystem for REIA.

Implements:
1. Embedding Models:
   - Hugging Face / Sentence-Transformers (e.g. 'neuralmind/bert-base-portuguese-cased'
     or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').
   - High-performance, zero-dependency pure Python/TF-IDF Semantic Vectorizer fallback.
2. Vector Store:
   - ChromaDB Persistent Client (when chromadb is installed).
   - In-memory persistent cosine-similarity vector store fallback.
3. Portal Indexing:
   - Knowledge base of PUC-Campinas portal sections, courses, links, and DOM targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class PortalDocument:
    """Document representation stored in the semantic vector index."""
    id: str
    title: str
    description: str
    target_url: str
    intent_category: str
    keywords: List[str] = field(default_factory=list)
    dom_target_selector: Optional[str] = None
    target_action: Optional[str] = None  # e.g. "click", "navigate", "highlight"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_embed_text(self) -> str:
        """Constructs rich semantic representation for embedding."""
        kw = ", ".join(self.keywords)
        return f"{self.title}. {self.description}. Palavras-chave: {kw}."


# Canonical institutional knowledge base for PUC-Campinas (Vestibular & Whole Portal)
PUC_PORTAL_KNOWLEDGE_BASE: List[PortalDocument] = [
    # --- Vestibular & Inscrições ---
    PortalDocument(
        id="puc_guia_cursos",
        title="Guia Universitário de Cursos PUC-Campinas",
        description="Apresentação detalhada e interativa de todos os cursos de graduação, bacharelados, licenciaturas e catálogo de carreiras da universidade.",
        target_url="https://vestibular.puc-campinas.edu.br/guia-universitario/",
        intent_category="GUIA_UNIVERSITARIO",
        keywords=["cursos", "graduacao", "guia", "carreiras", "catalogo", "bacharelado", "licenciatura", "campus", "faculdades"],
        dom_target_selector="a[href*='/guia'], a:contains('COMECE AGORA')",
        target_action="highlight_and_click",
    ),
    PortalDocument(
        id="puc_manual_candidato",
        title="Manual do Candidato e Editais do Vestibular",
        description="Regulamento oficial, normas do vestibular, critérios de avaliação, locais de prova, datas do exame e informações para o candidato.",
        target_url="https://vestibular.puc-campinas.edu.br/manual-do-candidato/",
        intent_category="MANUAL_DO_CANDIDATO",
        keywords=["edital", "manual", "regras", "candidato", "normas", "prova", "exame", "gabarito", "convocacao", "datas"],
        dom_target_selector="a[href*='.pdf'], button:contains('Manual')",
        target_action="highlight",
    ),
    PortalDocument(
        id="puc_formulario_inscricao",
        title="Inscrição no Vestibular e Processo Seletivo",
        description="Ficha de inscrição online, preenchimento de dados do candidato, escolha de curso, emissão de boleto e prazos de inscrição.",
        target_url="https://vestibular.puc-campinas.edu.br/",
        intent_category="FORMULARIO_INSCRICAO",
        keywords=["inscricao", "inscrever", "cadastro", "cadastrar", "ficha", "formulario", "taxa", "boleto", "vestibular"],
        dom_target_selector="a:contains('INSCREVA-SE'), button:contains('Inscreva-se'), a[href*='inscricao']",
        target_action="highlight_and_click",
    ),
    # --- Portal Institucional (Expansão do Site Todo) ---
    PortalDocument(
        id="puc_graduacao",
        title="Portal de Cursos de Graduação Presencial e EAD",
        description="Todos os cursos de graduação oferecidos pela PUC-Campinas, corpo docente, matriz curricular e infraestrutura dos campi.",
        target_url="https://www.puc-campinas.edu.br/graduacao/",
        intent_category="PORTAL_GRADUACAO",
        keywords=["graduacao", "presencial", "ead", "matriz curricular", "grade", "faculdade", "professores"],
    ),
    PortalDocument(
        id="puc_pos_graduacao",
        title="Pós-Graduação, Mestrado e Doutorado",
        description="Cursos de especialização, MBA, mestrado acadêmico e profissional, doutorado e programas de pesquisa científica.",
        target_url="https://www.puc-campinas.edu.br/pos-graduacao/",
        intent_category="POS_GRADUACAO",
        keywords=["pos", "mestrado", "doutorado", "especializacao", "mba", "pesquisa", "stricto sensu", "lato sensu"],
    ),
    PortalDocument(
        id="puc_biblioteca",
        title="Sistema de Bibliotecas PUC-Campinas (SIB)",
        description="Consulta ao acervo de livros físicos e digitais, renovação de empréstimos, periódicos e bases de dados acadêmicas.",
        target_url="https://www.puc-campinas.edu.br/biblioteca/",
        intent_category="BIBLIOTECA",
        keywords=["biblioteca", "livro", "livros", "acervo", "consulta", "emprestimo", "periodicos", "sib"],
    ),
    PortalDocument(
        id="puc_hospital",
        title="Hospital Universitário PUC-Campinas",
        description="Atendimento hospitalar, agendamento de consultas, pronto socorro, residência médica e especialidades de saúde.",
        target_url="https://www.hospitalpuc-campinas.com.br/",
        intent_category="HOSPITAL_PUC",
        keywords=["hospital", "medico", "medicina", "consultas", "saude", "pronto socorro", "residencia medica"],
    ),
    PortalDocument(
        id="puc_portal_aluno",
        title="Portal do Aluno e Secretaria Virtual",
        description="Acesso dos estudantes para notas, faltas, horários de aulas, rematrícula, solicitações de documentos e financeiro.",
        target_url="https://www.puc-campinas.edu.br/portal-do-aluno/",
        intent_category="PORTAL_ALUNO",
        keywords=["aluno", "portal do aluno", "secretaria", "notas", "faltas", "rematricula", "boleto", "declaracao", "historico"],
    ),
    PortalDocument(
        id="puc_bolsas_financiamento",
        title="Bolsas de Estudo, ProUni e Financiamento Estudantil",
        description="Critérios para concessão de bolsas de estudo, ProUni, FIES, financiamento estudantil próprio e incentivos.",
        target_url="https://www.puc-campinas.edu.br/bolsas-e-financiamentos/",
        intent_category="BOLSAS_FINANCIAMENTO",
        keywords=["bolsa", "bolsas", "desconto", "prouni", "fies", "financiamento", "gratuito", "carencia", "social"],
    ),
]


class BaseEmbedder:
    """Base interface for text embedding models."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]


class SentenceTransformersEmbedder(BaseEmbedder):
    """
    Neural sentence embedding using Sentence-Transformers / HuggingFace.
    Supports Portuguese models like 'neuralmind/bert-base-portuguese-cased'
    or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'.
    """

    def __init__(self, model_name: str = "neuralmind/bert-base-portuguese-cased"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Falha ao inicializar sentence-transformers com modelo '{self.model_name}': {e}"
                ) from e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]


class SemanticTFIDFEmbedder(BaseEmbedder):
    """
    Zero-dependency, deterministic semantic TF-IDF + Char n-gram vectorizer.
    Guarantees fast, robust local inference without requiring GPU or huge weights.
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.dim = 0
        self._fitted = False

    def fit(self, documents: List[str]):
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        doc_count = len(tokenized_docs)
        df: Dict[str, int] = {}

        for doc_tokens in tokenized_docs:
            for token in set(doc_tokens):
                df[token] = df.get(token, 0) + 1

        # Vocabulary restricted to salient tokens
        sorted_vocab = sorted(df.keys())
        self.vocabulary = {term: idx for idx, term in enumerate(sorted_vocab)}
        self.dim = len(self.vocabulary)
        self.idf = {
            term: math.log((doc_count + 1.0) / (freq + 1.0)) + 1.0
            for term, freq in df.items()
        }
        self._fitted = True

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        words = [w for w in cleaned.split() if len(w) >= 3]
        # Include word bi-grams
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return words + bigrams

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self.fit(texts)

        results = []
        for text in texts:
            tokens = self._tokenize(text)
            vec = [0.0] * self.dim
            for t in tokens:
                idx = self.vocabulary.get(t)
                if idx is not None:
                    vec[idx] += self.idf.get(t, 1.0)

            # L2 Normalize
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 1e-9:
                vec = [x / norm for x in vec]
            results.append(vec)
        return results


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two unit vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(0.0, min(1.0, dot))


class SemanticVectorSearchEngine:
    """
    Hybrid Vector Store with ChromaDB backend and fallback cosine similarity.
    Matches natural voice commands to target pages, descriptions, and DOM elements.
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        chroma_persist_dir: Optional[str] = None,
    ):
        self.persist_dir = chroma_persist_dir or "artifacts/chroma_db"
        self.documents: Dict[str, PortalDocument] = {}
        self.doc_vectors: Dict[str, List[float]] = {}
        self.chroma_collection = None
        self.has_chroma = False

        # Attempt to initialize SentenceTransformers if available, otherwise TFIDF
        if embedder is not None:
            self.embedder = embedder
        else:
            try:
                import sentence_transformers  # noqa
                self.embedder = SentenceTransformersEmbedder()
            except ImportError:
                self.embedder = SemanticTFIDFEmbedder()

        # Initialize knowledge base
        self._init_knowledge_base()
        self._init_backend()

    def _init_knowledge_base(self):
        for doc in PUC_PORTAL_KNOWLEDGE_BASE:
            self.documents[doc.id] = doc

        # Fit embedder if it's TFIDF
        if isinstance(self.embedder, SemanticTFIDFEmbedder):
            texts = [doc.to_embed_text() for doc in self.documents.values()]
            self.embedder.fit(texts)

        # Compute document embeddings
        for doc_id, doc in self.documents.items():
            emb = self.embedder.embed_query(doc.to_embed_text())
            self.doc_vectors[doc_id] = emb

    def _init_backend(self):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            self.chroma_collection = client.get_or_create_collection(
                name="puc_campinas_portal",
                metadata={"description": "PUC-Campinas Portal Semantic Navigation Index"}
            )
            # Upsert knowledge base into ChromaDB
            ids = list(self.documents.keys())
            docs_text = [self.documents[k].to_embed_text() for k in ids]
            metadatas = [
                {
                    "intent": self.documents[k].intent_category,
                    "target_url": self.documents[k].target_url,
                    "title": self.documents[k].title,
                }
                for k in ids
            ]
            self.chroma_collection.upsert(
                ids=ids,
                documents=docs_text,
                metadatas=metadatas,
            )
            self.has_chroma = True
        except (ImportError, Exception):
            self.has_chroma = False

    def search(self, query: str, top_k: int = 3) -> List[Tuple[PortalDocument, float]]:
        """
        Search for matching portal destinations given natural query.
        Returns ranked list of (PortalDocument, similarity_score).
        """
        query_clean = query.strip()
        if not query_clean:
            return []

        # ChromaDB search if available
        if self.has_chroma and self.chroma_collection is not None:
            try:
                results = self.chroma_collection.query(
                    query_texts=[query_clean],
                    n_results=top_k,
                )
                hits = []
                for doc_id, dist in zip(results["ids"][0], results["distances"][0]):
                    doc = self.documents.get(doc_id)
                    if doc:
                        # Convert distance to similarity score
                        sim = max(0.0, 1.0 - (dist / 2.0))
                        hits.append((doc, sim))
                return hits
            except Exception:
                pass  # Fallback to in-memory cosine search

        # Fallback in-memory cosine search
        query_vec = self.embedder.embed_query(query_clean)
        scores = []
        for doc_id, doc_vec in self.doc_vectors.items():
            sim = cosine_similarity(query_vec, doc_vec)
            # Bonus for exact keyword matches
            query_lower = query_clean.lower()
            doc = self.documents[doc_id]
            for kw in doc.keywords:
                if kw in query_lower:
                    sim = min(1.0, sim + 0.15)
            scores.append((doc, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

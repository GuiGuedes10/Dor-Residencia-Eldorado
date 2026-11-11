"""
Dataset definitions, class mappings, and reference evaluation sets for REIA.

Defines:
- Canonical destination classes and URLs for the PUC-Campinas Vestibular portal.
- Balanced training dataset (30 samples, 10 per class) with sanitized phrases.
- Out-of-domain (OOD) and ambiguous evaluation test sets.
"""

from typing import Dict, List, Tuple

try:
    from reia.core.config import TARGET_URLS, CLASS_ID_TO_INTENT
except ImportError:
    TARGET_URLS = {
        "GUIA_UNIVERSITARIO": "https://vestibular.puc-campinas.edu.br/guia-universitario/",
        "MANUAL_DO_CANDIDATO": "https://vestibular.puc-campinas.edu.br/manual-do-candidato/",
        "FORMULARIO_INSCRICAO": "https://vestibular.puc-campinas.edu.br/",
    }
    CLASS_ID_TO_INTENT = {
        0: "GUIA_UNIVERSITARIO",
        1: "MANUAL_DO_CANDIDATO",
        2: "FORMULARIO_INSCRICAO",
    }

CLASSES_DESTINO: Dict[int, str] = {
    0: "GUIA_UNIVERSITARIO",
    1: "MANUAL_DO_CANDIDATO",
    2: "FORMULARIO_INSCRICAO",
}

INTENT_TO_ID: Dict[str, int] = {v: k for k, v in CLASSES_DESTINO.items()}

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "GUIA_UNIVERSITARIO": "Guia Universitário com apresentação interativa dos cursos e áreas de formação da PUC-Campinas.",
    "MANUAL_DO_CANDIDATO": "Manual Oficial com regras do vestibular, editais e critérios de convocação.",
    "FORMULARIO_INSCRICAO": "Página inicial com acesso direto ao portal de inscrição.",
}

# Domain & intent keywords for lexical saliency checking
DOMAIN_KEYWORDS: set = {
    "puc", "campinas", "vestibular", "vestibulando", "graduacao", "curso", "cursos",
    "universitario", "universidade", "faculdade", "edital", "manual", "inscricao",
    "inscrever", "inscrevo", "cadastro", "cadastrar", "formulario", "ficha", "preencher",
    "prova", "provas", "normas", "norma", "regras", "regra", "regulamento", "candidato",
    "guia", "catalogo", "grade", "apresentacao", "carreiras", "trajetoria", "cronograma",
    "processo", "seletivo"
}

INTENT_KEYWORDS: Dict[str, set] = {
    "GUIA_UNIVERSITARIO": {
        "guia", "catalogo", "grade", "apresentacao", "carreiras", "trajetoria"
    },
    "MANUAL_DO_CANDIDATO": {
        "manual", "edital", "regulamento", "normas", "norma", "cronograma", "regras", "regra"
    },
    "FORMULARIO_INSCRICAO": {
        "inscrever", "inscricao", "inscrevo", "cadastrar", "cadastro", "formulario", "ficha", "preencher"
    },
}

# Domain anchors that do not specify a concrete intent
DOMAIN_ANCHORS: set = {"puc", "campinas", "vestibular", "vestibulando"}

# Expanded balanced training dataset: 30 samples (10 per class)
# All phrases are pre-sanitized (lowercase, no diacritics, alphanumeric and spaces only).
TRAINING_DATA: List[Tuple[str, int]] = [
    # CLASSE 0: GUIA_UNIVERSITARIO (10 samples)
    ("mostrar guia do vestibular da puc campinas", 0),
    ("abrir o guia universitario de apresentacao dos cursos", 0),
    ("quero ver a apresentacao do guia da puc campinas", 0),
    ("como funciona o guia de apresentacao para vestibulandos", 0),
    ("navegar ate o guia de cursos e trajetoria universitaria", 0),
    ("mostre a tela do guia interativo de apresentacao", 0),
    ("quais os cursos oferecidos no vestibular da puc", 0),
    ("abrir catalogo de cursos de graduacao", 0),
    ("quero conhecer os cursos da puc campinas", 0),
    ("ver detalhes e grade dos cursos universitarios", 0),

    # CLASSE 1: MANUAL_DO_CANDIDATO (10 samples)
    ("abrir manual do candidato com as regras da prova", 1),
    ("onde vejo o edital e as normas do vestibular da puc", 1),
    ("mostrar o manual oficial para leitura das provas", 1),
    ("quero conferir o cronograma e o manual de regras", 1),
    ("quais sao as regras e datas do vestibular", 1),
    ("abrir edital do vestibular da puc campinas", 1),
    ("cronograma completo e horario das provas", 1),
    ("manual do vestibular com conteudo programatico", 1),
    ("onde encontro o regulamento e informacoes da prova", 1),
    ("consultar edital oficial do candidato", 1),

    # CLASSE 2: FORMULARIO_INSCRICAO (10 samples)
    ("como faco para me inscrever no vestibular da puc campinas", 2),
    ("abrir pagina de inscricao para preencher meu cadastro", 2),
    ("ir para a tela de inscricao do vestibular", 2),
    ("quero fazer minha inscricao no vestibular agora", 2),
    ("fazer meu cadastro no vestibular da puc", 2),
    ("como me inscrevo no vestibular da puc campinas", 2),
    ("abrir formulario para cadastrar minha inscricao", 2),
    ("link para inscricao direta no vestibular", 2),
    ("preencher dados para vestibular da puc", 2),
    ("quero me cadastrar agora no processo seletivo", 2),
]

# Representative Out-of-Domain (OOD) test set
OOD_TEST_SET: List[str] = [
    "qual a previsao do tempo para amanha em sao paulo",
    "quero pedir uma pizza de calabresa bem quente",
    "tocar musicas dos beatles no spotify",
    "me conte uma piada sobre computadores",
    "como consertar um pneu furado de bicicleta",
    "qual a cotacao do dolar hoje",
    "onde fica a farmacia mais proxima",
    "abrir receita de bolo de cenoura",
]

# Representative Ambiguous / Anchor-Only phrases
AMBIGUOUS_TEST_SET: List[str] = [
    "quero ver o manual do candidato e fazer a inscricao agora mesmo",
    "abrir o guia de cursos e conferir o edital da prova",
    "como faco a inscricao e onde vejo o guia de cursos",
    "puc campinas vestibular",
    "vestibular da puc",
]


def get_balanced_training_data() -> List[Tuple[str, int]]:
    """Returns the balanced training dataset."""
    return list(TRAINING_DATA)


def get_ood_samples() -> List[str]:
    """Returns the representative out-of-domain evaluation sample list."""
    return list(OOD_TEST_SET)


def get_ambiguous_samples() -> List[str]:
    """Returns the ambiguous evaluation sample list."""
    return list(AMBIGUOUS_TEST_SET)

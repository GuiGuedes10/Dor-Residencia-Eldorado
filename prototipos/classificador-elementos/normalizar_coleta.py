#!/usr/bin/env python3
"""
Normaliza o CSV bruto da pesquisa (Gemini) → coleta.csv: mapeia os alvos genéricos
(ENEM/SISU) para os slugs do NOSSO corpus (PUC-Campinas), valida contra o corpus e
DESCARTA o que não tem elemento correspondente (isenção, boleto, matrícula-docs, SISU…).

Uso:  python3 normalizar_coleta.py <csv_bruto>   (padrão: coleta_bruta.csv nesta pasta)
Saída: coleta.csv  (pronto para o montar_dataset.py)
"""
import csv
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from montar_dataset import carregar_elementos, resolve  # noqa: E402


def mapear_alvo(alvo: str):
    """Mapeia o alvo genérico para um slug do corpus PUC, ou None para descartar."""
    a = (alvo or "").lower()
    if a.startswith("curso/"):
        return alvo  # resolve() decide (mantém os nossos cursos, descarta os que não temos)
    if any(x in a for x in ("cartao-confirmacao", "local-de-prova", "local-da-prova",
                            "consulta-sala", "documentos-dia-da-prova")):
        return "cartao de confirmacao"
    if "atendimento-especial" in a:               # atendimento especializado → acessibilidade
        return "programa de acessibilidade"
    if "fies" in a:
        return "fies"
    if "credito-estudantil" in a:
        return "credito estudantil"
    if "bolsa" in a or "prouni" in a:
        return "vestibular social"
    if "central-de-atendimento" in a or "fale-conosco" in a:
        return "central de atendimento"
    if "gabarito" in a:
        return "vestibular-2027"
    if "libras" in a:
        return "libras"
    if "manual" in a:
        return "manual-do-candidato"
    return None  # isenção, boleto, matrícula, sisu, orientação vocacional, recurso… (sem elemento PUC)


def main():
    bruto = sys.argv[1] if len(sys.argv) > 1 else str(AQUI / "coleta_bruta.csv")
    els = carregar_elementos()
    out, descartados, vistos = [], 0, set()
    with open(bruto, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            intent = (row.get("intencao") or "").strip()
            alvo0 = (row.get("alvo") or "").strip()
            fonte = (row.get("fonte") or "").strip()
            if not intent or not alvo0:
                continue
            k = intent.lower()
            if k in vistos:
                continue
            alvo = mapear_alvo(alvo0)
            if alvo is None or resolve(alvo, els) is None:
                descartados += 1
                continue
            vistos.add(k)
            out.append({"intencao": intent, "alvo": alvo, "fonte": fonte, "tipo": "semantico"})

    with open(AQUI / "coleta.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["intencao", "alvo", "fonte", "tipo"])
        w.writeheader()
        w.writerows(out)
    print(f"Pares mapeados ao corpus PUC: {len(out)}  |  descartados (sem elemento): {descartados}")
    print(f"Escrito: coleta.csv ({len(out)} linhas)")


if __name__ == "__main__":
    main()

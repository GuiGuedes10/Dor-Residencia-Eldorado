"""Ponte CLI Node -> rede neural treinada do projeto REIA (reia/neural).

Uso: echo "texto" | python3 bridge/intent.py   -> imprime JSON
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reia.neural import NeuralIntentClassifier  # noqa: E402

text = sys.stdin.read().strip()
r = NeuralIntentClassifier().classify(text)
print(json.dumps({
    "accepted": bool(r.is_accepted),
    "intent": r.intent_name,
    "url": r.target_url,
    "confidence": r.confidence,
    "probabilities": r.probabilities,
    "reason": r.rejection_code,
    "message": r.user_message,
}, ensure_ascii=False))

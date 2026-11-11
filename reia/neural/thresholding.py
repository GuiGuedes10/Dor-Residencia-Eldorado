"""
4-Stage Gating, Multi-Metric Confidence Thresholding, and Structured Intent Classification for REIA.

Pipeline:
- Stage 0: Text sanitization via reia.audio.sanitization.sanitize_transcription.
- Stage 1: Lexical Saliency Gating (checks domain keywords, intent action words, anchors).
- Stage 2: Forward pass computing logits and Softmax probabilities.
- Stage 3: Multi-metric thresholding: Top-1 Confidence (>= 0.65), Decision Margin (>= 0.20),
           Normalized Shannon Entropy (<= 0.65).
- Stage 4: Structured, strongly-typed ClassificationResult.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from reia.audio.sanitization import sanitize_transcription
except ImportError:
    def sanitize_transcription(text: Optional[str]) -> str:
        if not text:
            return ""
        return " ".join(text.lower().split())

from reia.neural.dataset import (
    CLASSES_DESTINO,
    DOMAIN_ANCHORS,
    DOMAIN_KEYWORDS,
    INTENT_KEYWORDS,
    TARGET_URLS,
)
from reia.neural.model import PurePythonMLP
from reia.neural.persistence import load_model_from_json, DEFAULT_WEIGHTS_PATH
from reia.neural.vectorizer import VocabularyVectorizer


@dataclass
class ClassificationResult:
    """Structured and strongly-typed outcome of the neural intent classification pipeline."""
    is_valid_intent: bool
    intent_id: Optional[int]
    intent_name: Optional[str]
    target_url: Optional[str]
    confidence: float
    margin: float
    entropy: float
    probabilities: Dict[str, float]
    rejection_code: Optional[str]
    user_message: str
    matched_domain_tokens: List[str] = field(default_factory=list)
    sanitized_text: str = ""
    is_accepted: bool = False

    def __post_init__(self):
        # Synchronize is_accepted with is_valid_intent
        self.is_accepted = self.is_valid_intent

    @property
    def intent(self) -> Optional[str]:
        """Alias for intent_name."""
        return self.intent_name

    @property
    def rejection_reason(self) -> Optional[str]:
        """Alias for rejection_code."""
        return self.rejection_code

    @property
    def user_feedback_message(self) -> str:
        """Alias for user_message."""
        return self.user_message


class LexicalSaliencyGate:
    """
    Stage 1 Gating: Analyzes vocabulary tokens to intercept out-of-domain
    queries, empty inputs, or ambiguous anchor-only utterances before neural routing.
    """

    def __init__(
        self,
        domain_keywords: Optional[Set[str]] = None,
        intent_keywords: Optional[Dict[str, Set[str]]] = None,
        domain_anchors: Optional[Set[str]] = None,
    ):
        self.domain_keywords = domain_keywords or DOMAIN_KEYWORDS
        self.intent_keywords = intent_keywords or INTENT_KEYWORDS
        self.domain_anchors = domain_anchors or DOMAIN_ANCHORS

    def evaluate(self, sanitized_text: str) -> Tuple[bool, Optional[str], Optional[str], List[str], Dict[str, int]]:
        """
        Evaluates lexical saliency of sanitized text.
        
        Returns:
            Tuple of:
            - is_passed: bool
            - rejection_code: Optional[str] ("EMPTY_INPUT", "OOD_NO_DOMAIN_KEYWORDS", "AMBIGUOUS_DOMAIN_ONLY", "AMBIGUOUS_INTENT")
            - rejection_message: Optional[str]
            - matched_domain_tokens: List[str]
            - intent_scores: Dict[str, int]
        """
        if not sanitized_text or not sanitized_text.strip():
            return (
                False,
                "EMPTY_INPUT",
                "Comando de voz vazio. Por favor, diga para onde deseja navegar.",
                [],
                {c: 0 for c in CLASSES_DESTINO.values()},
            )

        tokens = set(sanitized_text.split())
        matched_domain = sorted(list(tokens.intersection(self.domain_keywords)))

        # Rule 1: No domain keywords present -> pure Out-of-Domain
        if not matched_domain:
            return (
                False,
                "OOD_NO_DOMAIN_KEYWORDS",
                "Comando fora de domínio do vestibular PUC-Campinas. Posso ajudar com Guia Universitário, Manual do Candidato ou Inscrição.",
                [],
                {c: 0 for c in CLASSES_DESTINO.values()},
            )

        # Count keyword hits per intent
        intent_scores = {
            intent: len(tokens.intersection(kw_set))
            for intent, kw_set in self.intent_keywords.items()
        }
        total_intent_hits = sum(intent_scores.values())

        # Rule 2: Only domain anchors (e.g. "puc campinas vestibular") without intent words
        if total_intent_hits == 0:
            return (
                False,
                "AMBIGUOUS_DOMAIN_ONLY",
                "Por favor especifique se deseja acessar o Guia de Cursos, o Manual do Candidato ou a Inscrição.",
                matched_domain,
                intent_scores,
            )

        # Rule 3: Conflicting primary action intents present
        active_intents = [intent for intent, score in intent_scores.items() if score > 0]
        if len(active_intents) > 1:
            return (
                False,
                "AMBIGUOUS_INTENT",
                "Seu comando parece ambíguo entre múltiplas opções. Por favor escolha uma opção por vez.",
                matched_domain,
                intent_scores,
            )

        return True, None, None, matched_domain, intent_scores


class NeuralIntentClassifier:
    """
    4-Stage Intent Classifier with Confidence Thresholding and OOD Gating.
    
    Stages:
      0. Text Sanitization (NFKD Unicode normalization, diacritics removal, lowercasing)
      1. Lexical Saliency Gate (domain keywords, anchor-only check)
      2. Neural Forward Pass (MLP logits & Softmax probabilities)
      3. Multi-Metric Thresholding (Top-1 confidence >= 0.65, margin >= 0.20, entropy <= 0.65)
      4. Safe Action Dispatch / Helpful Portuguese User Feedback
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.65,
        margin_threshold: float = 0.20,
        entropy_threshold: float = 0.65,
        temperature: float = 1.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.entropy_threshold = entropy_threshold
        if temperature <= 0.0:
            raise ValueError("Temperature must be strictly positive.")
        self.temperature = temperature

        # Aliases for test compatibility
        self.conf_th = confidence_threshold
        self.margin_th = margin_threshold
        self.entropy_th = entropy_threshold

        self.gate = LexicalSaliencyGate()
        self.model: Optional[PurePythonMLP] = None
        self.vectorizer: Optional[VocabularyVectorizer] = None

        self._load_model_and_vectorizer(weights_path)

    def _load_model_and_vectorizer(self, weights_path: Optional[str]) -> None:
        """Loads model and vectorizer from JSON weights."""
        resolved_path = weights_path or str(DEFAULT_WEIGHTS_PATH)
        if Path(resolved_path).is_file():
            self.model, self.vectorizer = load_model_from_json(resolved_path)
        else:
            # Fallback initialization from training data
            from reia.neural.dataset import TRAINING_DATA
            self.vectorizer = VocabularyVectorizer.from_dataset(TRAINING_DATA)
            self.model = PurePythonMLP.train_pure_python(
                TRAINING_DATA, self.vectorizer, hidden_dim=16, num_classes=3, epochs=200, lr=0.05, seed=42
            )

    @staticmethod
    def _compute_entropy_norm(probs: List[float]) -> float:
        """Computes Shannon entropy normalized by ln(len(probs)) into [0.0, 1.0]."""
        num_classes = len(probs)
        norm_factor = math.log(num_classes) if num_classes > 1 else 1.0
        h = -sum(p * math.log(max(1e-12, p)) for p in probs)
        return min(1.0, max(0.0, h / norm_factor))

    def classify(self, text: Optional[str]) -> ClassificationResult:
        """
        Executes the 4-stage classification pipeline on natural language text.
        """
        # ---------------------------------------------------------
        # Stage 0: Text Sanitization & Normalization
        # ---------------------------------------------------------
        sanitized = sanitize_transcription(text)

        # ---------------------------------------------------------
        # Stage 1: Lexical Saliency Gating
        # ---------------------------------------------------------
        is_passed, rej_code, rej_msg, matched_tokens, intent_scores = self.gate.evaluate(sanitized)

        if not is_passed:
            confidence_val = 0.333 if rej_code in ("AMBIGUOUS_DOMAIN_ONLY", "AMBIGUOUS_INTENT") else 0.0
            margin_val = 0.0
            entropy_val = 1.0
            prob_dist = {CLASSES_DESTINO[i]: 1.0 / 3.0 for i in range(3)}

            return ClassificationResult(
                is_valid_intent=False,
                intent_id=None,
                intent_name=None,
                target_url=None,
                confidence=confidence_val,
                margin=margin_val,
                entropy=entropy_val,
                probabilities=prob_dist,
                rejection_code=rej_code,
                user_message=rej_msg or "Comando não reconhecido.",
                matched_domain_tokens=matched_tokens,
                sanitized_text=sanitized,
            )

        # ---------------------------------------------------------
        # Stage 2: Forward Pass & Softmax Probability Calculation
        # ---------------------------------------------------------
        x = self.vectorizer.transform(sanitized)
        z, probs = self.model.forward(x, temperature=self.temperature)

        prob_map = {CLASSES_DESTINO[i]: probs[i] for i in range(3)}
        sorted_probs = sorted(probs, reverse=True)
        top1 = sorted_probs[0]
        top2 = sorted_probs[1]
        margin = top1 - top2
        entropy_norm = self._compute_entropy_norm(probs)

        top_class_id = probs.index(top1)
        top_intent_name = CLASSES_DESTINO[top_class_id]
        target_url = TARGET_URLS[top_intent_name]

        # ---------------------------------------------------------
        # Stage 3: Multi-Metric Confidence Thresholding
        # ---------------------------------------------------------
        # Check Decision Margin and Shannon Entropy for ambiguity
        if margin < self.margin_threshold or entropy_norm > self.entropy_threshold:
            return ClassificationResult(
                is_valid_intent=False,
                intent_id=None,
                intent_name=None,
                target_url=None,
                confidence=top1,
                margin=margin,
                entropy=entropy_norm,
                probabilities=prob_map,
                rejection_code="AMBIGUOUS_INTENT",
                user_message="Seu comando parece ambíguo entre múltiplas opções. Por favor escolha uma opção por vez.",
                matched_domain_tokens=matched_tokens,
                sanitized_text=sanitized,
            )

        # Check Top-1 Confidence
        if top1 < self.confidence_threshold:
            return ClassificationResult(
                is_valid_intent=False,
                intent_id=top_class_id,
                intent_name=top_intent_name,
                target_url=target_url,
                confidence=top1,
                margin=margin,
                entropy=entropy_norm,
                probabilities=prob_map,
                rejection_code="LOW_CONFIDENCE",
                user_message="Não entendi claramente seu pedido com confiança suficiente. Por favor repita o comando.",
                matched_domain_tokens=matched_tokens,
                sanitized_text=sanitized,
            )

        # ---------------------------------------------------------
        # Stage 4: Valid Routed Intent
        # ---------------------------------------------------------
        return ClassificationResult(
            is_valid_intent=True,
            intent_id=top_class_id,
            intent_name=top_intent_name,
            target_url=target_url,
            confidence=top1,
            margin=margin,
            entropy=entropy_norm,
            probabilities=prob_map,
            rejection_code=None,
            user_message=f"Navegando para {top_intent_name}: {target_url}",
            matched_domain_tokens=matched_tokens,
            sanitized_text=sanitized,
        )

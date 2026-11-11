"""
REIA Neural Intent Classification and Confidence Gating Subsystem.

Provides:
- NeuralIntentClassifier: 4-stage pipeline with lexical gating and confidence thresholding.
- ClassificationResult: Strongly-typed structured prediction result contract.
- PurePythonMLP: Pure Python forward pass and SGD trainer.
- VocabularyVectorizer: Multi-hot Bag-of-Words encoder.
- Persistence routines for JSON weights and PyTorch checkpoints.
"""

import sys

from reia.neural.dataset import (
    CLASSES_DESTINO,
    DOMAIN_ANCHORS,
    DOMAIN_KEYWORDS,
    INTENT_KEYWORDS,
    INTENT_TO_ID,
    OOD_TEST_SET,
    AMBIGUOUS_TEST_SET,
    TARGET_URLS,
    TRAINING_DATA,
    get_ambiguous_samples,
    get_balanced_training_data,
    get_ood_samples,
)
from reia.neural.model import PurePythonMLP, set_deterministic_seed
from reia.neural.persistence import (
    DEFAULT_WEIGHTS_PATH,
    export_model_to_json,
    load_checkpoint,
    load_model_from_json,
    load_weights_json,
    save_checkpoint,
    save_weights_json,
)
from reia.neural.thresholding import (
    ClassificationResult,
    LexicalSaliencyGate,
    NeuralIntentClassifier,
)
from reia.neural.vectorizer import VocabularyVectorizer

# Compatibility alias for resolvers expecting reia.neural.classifier
sys.modules["reia.neural.classifier"] = sys.modules[__name__]

__all__ = [
    "ClassificationResult",
    "LexicalSaliencyGate",
    "NeuralIntentClassifier",
    "PurePythonMLP",
    "VocabularyVectorizer",
    "CLASSES_DESTINO",
    "INTENT_TO_ID",
    "TARGET_URLS",
    "TRAINING_DATA",
    "OOD_TEST_SET",
    "AMBIGUOUS_TEST_SET",
    "DOMAIN_KEYWORDS",
    "INTENT_KEYWORDS",
    "DOMAIN_ANCHORS",
    "set_deterministic_seed",
    "export_model_to_json",
    "load_model_from_json",
    "save_weights_json",
    "load_weights_json",
    "save_checkpoint",
    "load_checkpoint",
    "DEFAULT_WEIGHTS_PATH",
    "get_balanced_training_data",
    "get_ood_samples",
    "get_ambiguous_samples",
]

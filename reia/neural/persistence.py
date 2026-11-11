"""
Model persistence, serialization, and JSON weight export/load for REIA.

Supports:
- Dual persistence: PyTorch checkpoint (`checkpoint.pt`) and zero-dependency JSON (`weights.json`).
- Roundtrip serialization for PurePythonMLP and ClassificadorVozPUC.
- Co-located default weights resolution.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from reia.neural.dataset import CLASSES_DESTINO
from reia.neural.model import PurePythonMLP
from reia.neural.vectorizer import VocabularyVectorizer

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "weights.json"


def save_weights_json(filepath: str, data: Dict[str, Any]) -> None:
    """Saves serialized neural model weights and metadata to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_weights_json(filepath: Optional[str] = None) -> Dict[str, Any]:
    """Loads weights and metadata from a JSON file. Defaults to bundled weights.json."""
    path = Path(filepath) if filepath else DEFAULT_WEIGHTS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Model weights file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_model_to_json(
    model: Any,
    vectorizer: VocabularyVectorizer,
    filepath: Optional[str] = None,
) -> str:
    """
    Exports a trained model (PyTorch or PurePythonMLP) and vectorizer to JSON.
    Returns the resolved output filepath.
    """
    target_path = Path(filepath) if filepath else DEFAULT_WEIGHTS_PATH

    if isinstance(model, PurePythonMLP):
        w1 = model.w1
        b1 = model.b1
        w2 = model.w2
        b2 = model.b2
        hidden_dim = model.hidden_dim
        num_classes = model.num_classes
    elif HAS_TORCH and hasattr(model, "rede"):
        # PyTorch ClassificadorVozPUC
        # rede[0] is Linear(vocab_size, hidden_dim)
        # rede[3] is Linear(hidden_dim, num_classes)
        w1 = model.rede[0].weight.detach().cpu().tolist()
        b1 = model.rede[0].bias.detach().cpu().tolist()
        w2 = model.rede[3].weight.detach().cpu().tolist()
        b2 = model.rede[3].bias.detach().cpu().tolist()
        hidden_dim = model.hidden_dim
        num_classes = model.num_classes
    else:
        raise TypeError(f"Unsupported model type for JSON export: {type(model)}")

    data = {
        "vocab": vectorizer.vocab,
        "vocab_size": vectorizer.vocab_size,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "classes": {str(k): v for k, v in CLASSES_DESTINO.items()},
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
    }

    save_weights_json(str(target_path), data)
    return str(target_path)


def load_model_from_json(
    filepath: Optional[str] = None,
) -> Tuple[PurePythonMLP, VocabularyVectorizer]:
    """
    Loads a PurePythonMLP model and VocabularyVectorizer from a weights JSON file.
    """
    data = load_weights_json(filepath)
    vectorizer = VocabularyVectorizer(vocab=data["vocab"])
    model = PurePythonMLP(
        vocab_size=int(data["vocab_size"]),
        hidden_dim=int(data["hidden_dim"]),
        num_classes=int(data["num_classes"]),
        w1=data["w1"],
        b1=data["b1"],
        w2=data["w2"],
        b2=data["b2"],
    )
    return model, vectorizer


def save_checkpoint(filepath: str, model: Any, vectorizer: VocabularyVectorizer) -> None:
    """Saves a full PyTorch model checkpoint to disk."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to save .pt checkpoints. Use export_model_to_json instead.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "vocab": vectorizer.vocab,
        "vocab_size": vectorizer.vocab_size,
        "hidden_dim": getattr(model, "hidden_dim", 16),
        "num_classes": getattr(model, "num_classes", 3),
        "classes": CLASSES_DESTINO,
    }
    torch.save(checkpoint, str(path))


def load_checkpoint(filepath: str) -> Tuple[Any, VocabularyVectorizer]:
    """Loads a PyTorch model and vectorizer from a .pt checkpoint."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to load .pt checkpoints. Use load_model_from_json instead.")

    from reia.neural.model import ClassificadorVozPUC

    checkpoint = torch.load(filepath, map_location="cpu")
    vectorizer = VocabularyVectorizer(vocab=checkpoint["vocab"])
    model = ClassificadorVozPUC(
        vocab_size=checkpoint["vocab_size"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vectorizer

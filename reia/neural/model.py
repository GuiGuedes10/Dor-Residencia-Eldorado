"""
Neural Network Models for REIA Intent Classification.

Provides:
- PyTorch MLP (`ClassificadorVozPUC`) matching the architecture from Untitled0.ipynb:
  Linear(vocab_size, 16) -> ReLU -> Dropout(0.2) -> Linear(16, 3).
- Deterministic training routines with cross-entropy loss and Adam optimizer.
- Pure Python forward-pass fallback (`PurePythonMLP`) operating on layer weights
  and biases with zero external dependencies.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    optim = None
    HAS_TORCH = False


def set_deterministic_seed(seed: int = 42) -> None:
    """Fixes random seeds for deterministic model initialization and training."""
    random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


if HAS_TORCH:
    class ClassificadorVozPUC(nn.Module):
        """
        PyTorch Multi-Layer Perceptron for voice command intent classification.
        Matches the exact architecture from Untitled0.ipynb:
        Linear(vocab_size, hidden_dim) -> ReLU -> Dropout(0.2) -> Linear(hidden_dim, num_classes).
        """

        def __init__(self, vocab_size: int, hidden_dim: int = 16, num_classes: int = 3):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.rede = nn.Sequential(
                nn.Linear(vocab_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.rede(x)

        def predict_probabilities(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[List[float], List[float]]:
            """Returns (logits, softmax_probabilities) for a single input tensor."""
            if temperature <= 0.0:
                raise ValueError("Temperature must be strictly positive.")
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=1).squeeze(0).tolist()
                logits_list = logits.squeeze(0).tolist()
            return logits_list, probs

else:
    class ClassificadorVozPUC:  # type: ignore
        """PyTorch stub when torch is not available."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available. Use PurePythonMLP instead.")


class PurePythonMLP:
    """
    Pure Python Multi-Layer Perceptron forward-pass and training implementation.
    
    Operates without PyTorch or NumPy, using standard library math.
    Architecture:
      h_pre = W1 * x + b1       (16 units)
      h = max(0, h_pre)          (ReLU)
      z = W2 * h + b2           (3 units, logits)
      p = softmax(z / T)        (probabilities)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_classes: int,
        w1: List[List[float]],
        b1: List[float],
        w2: List[List[float]],
        b2: List[float],
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.w1 = w1  # shape: (hidden_dim, vocab_size)
        self.b1 = b1  # shape: (hidden_dim,)
        self.w2 = w2  # shape: (num_classes, hidden_dim)
        self.b2 = b2  # shape: (num_classes,)

    def forward(self, x: List[float], temperature: float = 1.0) -> Tuple[List[float], List[float]]:
        """
        Executes forward pass.
        
        Args:
            x: Multi-hot vector of length vocab_size.
            temperature: Softmax temperature parameter (default: 1.0).
            
        Returns:
            Tuple of (logits, softmax_probabilities).
        """
        if temperature <= 0.0:
            raise ValueError("Temperature must be strictly positive.")

        # Hidden layer: W1 * x + b1
        h_pre = [
            self.b1[j] + sum(self.w1[j][k] * x[k] for k in range(self.vocab_size))
            for j in range(self.hidden_dim)
        ]
        # ReLU activation
        h = [max(0.0, val) for val in h_pre]

        # Output layer (logits): W2 * h + b2
        z = [
            self.b2[c] + sum(self.w2[c][j] * h[j] for j in range(self.hidden_dim))
            for c in range(self.num_classes)
        ]

        # Softmax with temperature scaling and numerical stability
        scaled_z = [val / temperature for val in z]
        max_z = max(scaled_z)
        exp_z = [math.exp(val - max_z) for val in scaled_z]
        sum_exp = sum(exp_z)
        probs = [val / sum_exp for val in exp_z]

        return z, probs

    def predict(self, x: List[float]) -> int:
        """Returns the predicted class index (argmax)."""
        _, probs = self.forward(x)
        return probs.index(max(probs))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes weights and architecture parameters to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PurePythonMLP":
        """Reconstructs PurePythonMLP from serialized dictionary state."""
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_dim=int(data["hidden_dim"]),
            num_classes=int(data["num_classes"]),
            w1=data["w1"],
            b1=data["b1"],
            w2=data["w2"],
            b2=data["b2"],
        )

    @classmethod
    def train_pure_python(
        cls,
        dataset: List[Tuple[str, int]],
        vectorizer: Any,
        hidden_dim: int = 16,
        num_classes: int = 3,
        epochs: int = 150,
        lr: float = 0.05,
        seed: int = 42,
    ) -> "PurePythonMLP":
        """
        Trains an MLP directly in pure Python using exact SGD backpropagation
        with cross-entropy loss. Trains in < 0.1s.
        """
        rng = random.Random(seed)
        vocab_size = vectorizer.vocab_size

        # He / Xavier normal initialization
        scale1 = math.sqrt(2.0 / (vocab_size + hidden_dim))
        w1 = [[rng.gauss(0.0, scale1) for _ in range(vocab_size)] for _ in range(hidden_dim)]
        b1 = [0.0] * hidden_dim

        scale2 = math.sqrt(2.0 / (hidden_dim + num_classes))
        w2 = [[rng.gauss(0.0, scale2) for _ in range(hidden_dim)] for _ in range(num_classes)]
        b2 = [0.0] * num_classes

        # Pre-vectorize dataset
        vectorized_data = [(vectorizer.transform(phrase), label) for phrase, label in dataset]

        for _ in range(epochs):
            for x, label in vectorized_data:
                # Forward pass
                h_pre = [
                    b1[j] + sum(w1[j][k] * x[k] for k in range(vocab_size))
                    for j in range(hidden_dim)
                ]
                h = [max(0.0, v) for v in h_pre]

                z = [
                    b2[c] + sum(w2[c][j] * h[j] for j in range(hidden_dim))
                    for c in range(num_classes)
                ]
                max_z = max(z)
                exp_z = [math.exp(v - max_z) for v in z]
                sum_exp = sum(exp_z)
                p = [v / sum_exp for v in exp_z]

                # Backward pass: dLoss/dz = p - y_one_hot
                dz = [p[c] - (1.0 if c == label else 0.0) for c in range(num_classes)]

                # Backprop to hidden layer using pre-update weights: dh = W2^T * dz
                dh = [sum(w2[c][j] * dz[c] for c in range(num_classes)) for j in range(hidden_dim)]

                # Update W2 and b2
                for c in range(num_classes):
                    b2[c] -= lr * dz[c]
                    for j in range(hidden_dim):
                        w2[c][j] -= lr * dz[c] * h[j]

                # ReLU gradient: dh_pre = dh * 1(h_pre > 0)
                dh_pre = [dh[j] if h_pre[j] > 0.0 else 0.0 for j in range(hidden_dim)]

                # Update W1 and b1
                for j in range(hidden_dim):
                    b1[j] -= lr * dh_pre[j]
                    for k in range(vocab_size):
                        w1[j][k] -= lr * dh_pre[j] * x[k]

        return cls(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
        )


def train_pytorch_model(
    dataset: List[Tuple[str, int]],
    vectorizer: Any,
    hidden_dim: int = 16,
    num_classes: int = 3,
    epochs: int = 120,
    lr: float = 0.02,
    seed: int = 42,
) -> "ClassificadorVozPUC":
    """
    Trains PyTorch ClassificadorVozPUC on the provided dataset.
    Matches the loop and hyperparameters from Untitled0.ipynb.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed in the current environment.")

    set_deterministic_seed(seed)
    vocab_size = vectorizer.vocab_size
    model = ClassificadorVozPUC(vocab_size=vocab_size, hidden_dim=hidden_dim, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for phrase, classe in dataset:
            optimizer.zero_grad()
            x = vectorizer.transform_tensor(phrase)
            y = torch.tensor([classe], dtype=torch.long)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model

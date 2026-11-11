"""
Vocabulary and Bag-of-Words Multi-Hot Vectorizer for REIA.

Converts sanitized natural language phrases into fixed-dimensional multi-hot
vectors matching the representation logic established in Untitled0.ipynb.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from reia.audio.sanitization import sanitize_transcription
except ImportError:
    def sanitize_transcription(text: Optional[str]) -> str:
        if not text:
            return ""
        return " ".join(text.lower().split())

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


class VocabularyVectorizer:
    """
    Multi-hot Bag-of-Words text vectorizer with vocabulary indexing.
    
    Extracts vocabulary from training corpora, maps words to indices,
    and produces deterministic multi-hot binary float vectors.
    Unknown/out-of-vocabulary tokens are safely ignored.
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        """
        Initializes the vectorizer with an optional pre-computed vocabulary list.
        """
        self._vocab_list: List[str] = []
        self._vocab_to_idx: Dict[str, int] = {}
        self._idx_to_vocab: Dict[int, str] = {}

        if vocab is not None:
            self._set_vocab(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_list)

    @property
    def vocab(self) -> List[str]:
        return list(self._vocab_list)

    @property
    def vocab_to_idx(self) -> Dict[str, int]:
        return dict(self._vocab_to_idx)

    @property
    def idx_to_vocab(self) -> Dict[int, str]:
        return dict(self._idx_to_vocab)

    def _set_vocab(self, words: Iterable[str]) -> None:
        """Sets internal vocabulary mappings from an iterable of words."""
        cleaned_unique = sorted(list({str(w).strip().lower() for w in words if str(w).strip()}))
        self._vocab_list = cleaned_unique
        self._vocab_to_idx = {word: i for i, word in enumerate(self._vocab_list)}
        self._idx_to_vocab = {i: word for i, word in enumerate(self._vocab_list)}

    def fit_from_dataset(self, dataset: List[Tuple[str, int]]) -> "VocabularyVectorizer":
        """
        Builds vocabulary from a labeled dataset of (phrase, label) pairs.
        Matches the extraction pattern from Untitled0.ipynb:
        palavras_unicas = sorted(list({p for frase, _ in dados for p in frase.lower().split()}))
        """
        tokens: set = set()
        for phrase, _ in dataset:
            clean = sanitize_transcription(phrase)
            for token in clean.split():
                if token:
                    tokens.add(token)
        self._set_vocab(tokens)
        return self

    def fit_from_texts(self, texts: List[str]) -> "VocabularyVectorizer":
        """Builds vocabulary from a list of raw or sanitized text phrases."""
        tokens: set = set()
        for text in texts:
            clean = sanitize_transcription(text)
            for token in clean.split():
                if token:
                    tokens.add(token)
        self._set_vocab(tokens)
        return self

    def transform(self, text: Optional[str]) -> List[float]:
        """
        Encodes input text as a multi-hot Bag-of-Words list of floats.
        Unknown words not in vocabulary are ignored.
        """
        vector = [0.0] * self.vocab_size
        if not text:
            return vector

        clean = sanitize_transcription(text)
        for token in clean.split():
            idx = self._vocab_to_idx.get(token)
            if idx is not None:
                vector[idx] = 1.0

        return vector

    def transform_tensor(self, text: Optional[str]):
        """
        Encodes input text as a PyTorch FloatTensor of shape (1, vocab_size).
        Raises RuntimeError if PyTorch is not installed.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed. Use transform() for pure Python vector.")
        vector = self.transform(text)
        return torch.tensor([vector], dtype=torch.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the vocabulary state to a dictionary."""
        return {
            "vocab": list(self._vocab_list),
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VocabularyVectorizer":
        """Instantiates a vectorizer from serialized dictionary state."""
        vocab = data.get("vocab", [])
        return cls(vocab=vocab)

    @classmethod
    def from_dataset(cls, dataset: List[Tuple[str, int]]) -> "VocabularyVectorizer":
        """Convenience factory: fits and returns a vectorizer from dataset."""
        vec = cls()
        vec.fit_from_dataset(dataset)
        return vec

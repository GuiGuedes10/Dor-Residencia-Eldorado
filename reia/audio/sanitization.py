"""
Transcription Sanitization Engine for REIA.

Sanitizes raw Whisper STT transcriptions to align precisely with the downstream
neural classifier's vocabulary:
- Strips acoustic tags (e.g., [Música], (risos)).
- Normalizes Unicode (NFKD) and removes Portuguese diacritics/accents.
- Lowercases all characters.
- Replaces punctuation and special symbols with spaces to preserve word boundaries.
- Collapses consecutive whitespaces.
"""

import re
import unicodedata
from typing import Optional


def sanitize_transcription(text: Optional[str], strip_accents: bool = True) -> str:
    """
    Cleans and normalizes transcribed speech for neural intent classification.

    Args:
        text: Raw transcription string from Whisper or user input.
        strip_accents: If True, strips diacritical marks (e.g., 'ç' -> 'c', 'ã' -> 'a').

    Returns:
        Cleaned, normalized, lowercased string without punctuation or acoustic tags.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Strip bracketed or parenthesized acoustic tags (e.g., [Música], (risos), [tosse])
    cleaned = re.sub(r"\[.*?\]|\(.*?\)", " ", text)

    # 2. Lowercase
    cleaned = cleaned.lower()

    # 3. Unicode normalization and diacritics stripping
    if strip_accents:
        # Decompose characters into base letters + combining marks (NFKD),
        # then strip combining mark characters (category 'Mn')
        nfkd = unicodedata.normalize("NFKD", cleaned)
        cleaned = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    else:
        cleaned = unicodedata.normalize("NFC", cleaned)

    # 4. Replace punctuation and non-alphanumeric characters with spaces
    # Keeps Portuguese alphanumeric characters a-z, 0-9 and whitespace.
    # Words connected by hyphens (e.g., 'puc-campinas') are split into distinct tokens.
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)

    # 5. Collapse consecutive spaces and trim edges
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned

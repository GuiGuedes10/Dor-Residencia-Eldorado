# Project: REIA — Voice-Guided Web Navigation Assistant for PUC-Campinas Vestibular

## Architecture
The system is structured as a modular, local Python package (`reia`) that transforms voice commands into automated web navigation actions on the PUC-Campinas vestibular portal:

1. **Audio Ingestion & Validation Layer (`reia.audio`)**:
   - Ingests audio from files, byte streams, or microphone without Colab dependencies.
   - Evaluates acoustic metrics using pure Python standard library (`wave`, `struct`, `math`): Windowed RMS energy, silence frame ratio ($R_{\text{silence}} \le 0.85$, $T_{\text{active}} \ge 0.4\text{s}$), duration bounds ($0.8\text{s} \le T \le 12.0\text{s}$), and energy-quantile SNR ($\text{SNR} \ge 3.0\text{ dB}$).
   - Sanitizes transcribed speech via Unicode NFKD normalization, diacritics stripping, and punctuation removal.
   - Generates deterministic synthetic audio signals for offline testing (`SyntheticAudioGenerator`).

2. **Neural Intent Classification & Thresholding Layer (`reia.neural`)**:
   - Multi-layer perceptron (PyTorch MLP: 58 input features $\to$ 16 hidden $\to$ 3 output classes).
   - 4-stage inference pipeline:
     - Stage 0: Text sanitization and vocabulary multi-hot vectorization.
     - Stage 1: Lexical Saliency Gating (checks for domain anchor words like "puc/vestibular" and intent action keywords).
     - Stage 2: MLP forward pass with Softmax probability calculation.
     - Stage 3: Multi-metric thresholding: Top-1 Confidence ($p_{(1)} \ge 0.65$), Decision Margin ($\Delta = p_{(1)} - p_{(2)} \ge 0.20$), and Normalized Shannon Entropy ($\tilde{H} \le 0.65$).
     - Stage 4: Structured `ClassificationResult` containing intent, confidence, probabilities, and rejection reason if filtered.
   - Dual persistence: PyTorch checkpoint (`checkpoint.pt`) and zero-dependency JSON weights (`weights.json`) for pure Python forward pass fallback.

3. **Web Navigation Automation Layer (`reia.navigation`)**:
   - Adapter pattern providing `SeleniumNavigator` (production live browser with standard Selenium 4 headless options) and `MockNavigator` (deterministic offline browser with local HTML fixtures).
   - Target URL routing for PUC-Campinas Vestibular:
     - `GUIA_UNIVERSITARIO` $\to$ `https://vestibular.puc-campinas.edu.br/guia-universitario/`
     - `MANUAL_DO_CANDIDATO` $\to$ `https://vestibular.puc-campinas.edu.br/manual-do-candidato/`
     - `FORMULARIO_INSCRICAO` $\to$ `https://vestibular.puc-campinas.edu.br/`
   - Explicit `WebDriverWait` for dynamic elements replacing hardcoded sleeps.
   - Capture and persistence of 1280x800 PNG screenshots and JSON DOM metadata under `artifacts/`.

4. **Orchestration Pipeline & CLI (`reia.pipeline`)**:
   - End-to-end coordinator chaining Audio Ingestion $\to$ Validation $\to$ Transcription $\to$ Neural Gating $\to$ Navigation $\to$ Artifact Capture.
   - Explicit Portuguese status and error messages ensuring the application never crashes on invalid inputs.

5. **Automated Testing Suite (`tests/`)**:
   - Comprehensive `pytest` synthetic test suite covering all tiers (Features, Boundaries, Combinations, Real-World Scenarios) running in $<3$ seconds with exit code 0.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Local Modular Packaging | Standard Python package layout (`pyproject.toml`/`requirements.txt`) decoupling all Colab dependencies | M1 | R1, Survey 1 |
| F2 | Audio Ingestion Abstraction | `AudioSource` interface supporting files, raw bytes, and local microphone input | M1 | R1, R2, Survey 2 |
| F3 | Pure Python Audio Validation | RMS energy, silence ratio, SNR, and duration bounds checking without deprecated `audioop` | M1 | R2, Survey 2 |
| F4 | Text Sanitization & Normalization | Unicode NFKD diacritics and punctuation stripping matching vocabulary | M1 | R2, Survey 2 |
| F5 | Synthetic Audio Generator | Deterministic generation of silence, white noise, mock speech, and invalid formats for testing | M1 | R2, R5, Survey 2 |
| F6 | Neural Intent Classifier (MLP) | PyTorch MLP model trained on balanced vestibular domain dataset (30 samples, 3 classes) | M2 | R3, Survey 1, Survey 3 |
| F7 | Vocabulary Multi-Hot Vectorizer | Multi-hot Bag-of-Words text vectorization with vocabulary mapping | M2 | R3, Survey 1, Survey 3 |
| F8 | Multi-Stage Confidence Thresholding | Softmax probabilities, confidence $\ge 0.65$, margin $\ge 0.20$, and entropy filtering | M2 | R3, Survey 3 |
| F9 | Lexical Saliency Gating | Out-of-domain and unstated intent rejection prior to neural routing | M2 | R3, Survey 3 |
| F10 | Dual Model Persistence & Fallback | PyTorch checkpoint (`checkpoint.pt`) and pure Python forward pass (`weights.json`) | M2 | R3, Survey 3 |
| F11 | Headless Selenium 4 Automation | Local headless Chrome navigator with `--headless=new` and explicit waits | M3 | R1, R4, Survey 1, Survey 3 |
| F12 | Offline Mock Navigation Adapter | Deterministic offline navigator using local HTML fixtures and zero-dependency PNG generator | M3 | R4, R5, Survey 3 |
| F13 | Target Route Dispatcher | URL mapping and DOM navigation for Guia Universitário, Manual do Candidato, and Inscrição | M3 | R4, Survey 1, Survey 3 |
| F14 | Artifact Persistence Engine | Automated saving of timestamped PNG screenshots and JSON page metadata | M3 | R4, Survey 3 |
| F15 | End-to-End Orchestration Pipeline | Coordinated pipeline connecting audio validation, neural classification, and navigation | M4 | R1, R2, R3, R4, Survey 1 |
| F16 | User Feedback & Error Messaging | Clear Portuguese status and error messages for all edge cases | M4 | R2, R3, Survey 2, Survey 3 |
| F17 | CLI Application Entrypoint | Command-line interface to execute voice navigation with files, synthetic inputs, or live audio | M4 | R1, Survey 1 |
| F18 | 4-Tier Automated Synthetic Test Suite | Comprehensive pytest suite covering unit, boundary, combination, and E2E scenarios | M5 / Test Track | R5, Survey 2, Survey 3 |
| F19 | Adversarial Coverage Hardening | White-box stress tests verifying boundary limits, corrupted inputs, and edge cases | M5 | Project Pattern |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core Package & Audio Subsystem | Project structure, `AudioSource`, audio validation engine, text sanitization, and synthetic generator (F1–F5) | none | DONE |
| M2 | Neural Intent Classification Subsystem | PyTorch MLP, vectorizer, dataset, 4-stage gating/thresholding, and dual model persistence (F6–F10) | M1 (interfaces) | DONE |
| M3 | Web Navigation Automation Subsystem | Headless Selenium 4, Mock navigator adapter, URL routing, and screenshot/metadata persistence (F11–F14) | M1 (interfaces) | DONE |
| M4 | End-to-End Voice Navigation Pipeline & CLI | Integration pipeline, error handling, Portuguese feedback messaging, and CLI entrypoint (F15–F17) | M1, M2, M3 | DONE |
| M5 | E2E Test Suite Pass & Adversarial Hardening | Execute full test suite (Tiers 1–4) until 100% pass, followed by Tier 5 adversarial hardening (F18–F19) | M4, Test Track | IN_PROGRESS |

---

## Interface Contracts

### `reia.audio` $\to$ `reia.neural`
```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

class AudioStatus(str, Enum):
    VALID = "VALID"
    SILENCE_DETECTED = "SILENCE_DETECTED"
    EXCESSIVE_NOISE = "EXCESSIVE_NOISE"
    TOO_SHORT = "TOO_SHORT"
    TOO_LONG = "TOO_LONG"
    CORRUPTED_FORMAT = "CORRUPTED_FORMAT"

@dataclass
class AudioValidationResult:
    is_valid: bool
    status: AudioStatus
    message: str
    duration_seconds: float
    rms_dbfs: float
    snr_db: float
    silence_ratio: float
    transcription: Optional[str] = None
    sanitized_text: Optional[str] = None
```
- Function: `sanitize_transcription(text: str) -> str`
- Invariant: `sanitize_transcription` removes diacritics via Unicode NFKD and strips punctuation, matching the training vocabulary.

### `reia.neural` $\to$ `reia.navigation`
```python
@dataclass
class ClassificationResult:
    is_valid_intent: bool
    intent_id: Optional[int]
    intent_name: Optional[str]   # "GUIA_UNIVERSITARIO", "MANUAL_DO_CANDIDATO", "FORMULARIO_INSCRICAO"
    target_url: Optional[str]
    confidence: float
    probabilities: Dict[str, float]
    rejection_code: Optional[str] # "OOD_NO_DOMAIN_KEYWORDS", "LOW_CONFIDENCE", "AMBIGUOUS_INTENT"
    user_message: str
```
- Invariant: Navigation is ONLY invoked if `is_valid_intent == True` and `target_url` is non-null.

### `reia.navigation` $\to$ `reia.pipeline`
```python
@dataclass
class NavigationResult:
    success: bool
    intent_name: str
    target_url: str
    page_title: str
    screenshot_path: str
    metadata_path: str
    headings: list[str]
    error_message: Optional[str] = None
```
- Invariant: `NavigationResult` always writes screenshot to `artifacts/screenshots/` and metadata to `artifacts/metadata/`, returning valid filesystem paths.

---

## Code Layout
```
reia/
├── pyproject.toml
├── requirements.txt
├── README.md
├── reia/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── exceptions.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   ├── sources.py
│   │   ├── sanitization.py
│   │   └── synthetic.py
│   ├── neural/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── dataset.py
│   │   ├── vectorizer.py
│   │   ├── thresholding.py
│   │   ├── persistence.py
│   │   └── weights.json
│   ├── navigation/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── selenium_driver.py
│   │   ├── mock_driver.py
│   │   ├── routes.py
│   │   └── artifacts.py
│   └── pipeline/
│       ├── __init__.py
│       ├── coordinator.py
│       └── cli.py
├── artifacts/
│   ├── screenshots/
│   └── metadata/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_tier1_features.py
│   ├── test_tier2_boundaries.py
│   ├── test_tier3_combinations.py
│   └── test_tier4_scenarios.py
└── .agents/
    └── teamwork/
```

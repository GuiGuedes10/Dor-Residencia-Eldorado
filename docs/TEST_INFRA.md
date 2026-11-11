# REIA — E2E Test Infrastructure & Methodology Specification

## 1. Testing Philosophy
The REIA test suite is designed as an **opaque-box, requirement-driven verification system** derived strictly from `ORIGINAL_REQUEST.md` (R1 through R5) and architectural specifications in `PROJECT.md`.

### Core Principles
1. **Opaque-Box Independence**: Tests exercise external public interfaces (`reia.audio`, `reia.neural`, `reia.navigation`, `reia.pipeline`) without coupling to private internal implementation details.
2. **Authoritative Output Derivation**: Expected outputs are derived mathematically, acoustically, and semantically from requirements:
   - Acoustic parameters: RMS $L_{\text{dBFS}} < -46$, silence frame ratio $R_{\text{silence}} \le 0.85$, $T_{\text{active}} \ge 0.4\text{s}$, SNR $\ge 3.0\text{ dB}$, duration $[0.8\text{s}, 12.0\text{s}]$.
   - Text normalization: Unicode NFKD decomposition, diacritics removal, lowercasing, punctuation stripping.
   - Neural classification thresholds: Top-1 Confidence $p_{(1)} \ge 0.65$, Decision Margin $\Delta = p_{(1)} - p_{(2)} \ge 0.20$, Shannon Entropy $\tilde{H} \le 0.65$, Lexical Gating.
   - Web navigation routing: PUC-Campinas target destinations (`GUIA_UNIVERSITARIO`, `MANUAL_DO_CANDIDATO`, `FORMULARIO_INSCRICAO`).
3. **Hermetic & Deterministic Execution**: Zero reliance on external live networks or cloud APIs during automated regression. Navigation uses local HTML fixtures and a pure-Python zero-dependency PNG generator, while audio uses deterministic mathematical signals (`SyntheticAudioGenerator`).
4. **Fast Execution**: Entire test suite executes in $< 3$ seconds with exit code 0.

---

## 2. Four-Tier Testing Methodology

The suite implements a structured 4-tier test architecture:

```
┌────────────────────────────────────────────────────────┐
│ Tier 4: Real-World Scenarios (End-to-End User Journeys)│
│ - Full voice-to-navigation flows for all 3 targets     │
│ - Resilience and error recovery journeys               │
├────────────────────────────────────────────────────────┤
│ Tier 3: Pairwise Combinations & Cross-Feature Flows   │
│ - Audio validation + text sanitization + neural routing│
│ - Cascading error propagation & gate short-circuiting  │
├────────────────────────────────────────────────────────┤
│ Tier 2: Boundary Value Analysis (BVA) & Corner Cases   │
│ - Precision duration limits (0.79s vs 0.80s, etc.)    │
│ - Borderline neural confidence (0.64 vs 0.65)          │
│ - Borderline decision margin (0.19 vs 0.20)            │
│ - Silent audio, excessive noise SNR, accented text, OOD│
├────────────────────────────────────────────────────────┤
│ Tier 1: Feature Coverage (Category-Partition)          │
│ - Audio validation engine, sanitization, MLP inference │
│ - Thresholding logic, Mock navigation, CLI pipeline    │
└────────────────────────────────────────────────────────┘
```

### Tier 1: Category-Partition Feature Coverage (`test_tier1_features.py`)
Systematically validates functional partitions of each core feature (>=5 test cases per feature):
1. **Audio Validation**: Audio format parsing, RMS calculation, active speech duration, SNR estimation, clean speech acceptance.
2. **Transcription Sanitization**: Lowercasing, diacritic removal, punctuation stripping, bracketed tag removal, whitespace collapsing.
3. **Neural Classification**: Top-1 class identification for Guia, Manual, and Inscrição, probability distribution validation, multi-hot vectorization.
4. **Confidence & Thresholding Gating**: Top-1 probability thresholding, decision margin gating, entropy filtering, lexical gating on domain keywords.
5. **Web Navigation Automation**: Route dispatching, HTML title and heading extraction, link extraction, screenshot artifact creation, metadata JSON persistence.
6. **Orchestration Pipeline & CLI**: CLI argument parsing, valid voice-to-navigation dispatch, Portuguese feedback messaging, graceful exception wrapping.

### Tier 2: Boundary Value Analysis & Edge Conditions (`test_tier2_boundaries.py`)
Focuses on boundary discontinuities, transitions, and adversarial stress (>=5 test cases per boundary):
1. **Empty & Corrupt Inputs**: Zero-length bytes, truncated WAV headers, whitespace-only transcripts, corrupted memory buffers.
2. **Pure Silence**: Total digital zero signals, quiet ambient room tone below $-46\text{ dBFS}$, 100% silence frame ratio.
3. **Excessive Noise SNR**: White noise with $\text{SNR} < 1\text{ dB}$, SNR near boundary $2.9\text{ dB}$ vs $3.0\text{ dB}$, high-power pink noise.
4. **Duration Boundaries**: Sub-millisecond boundary precision: $0.79\text{s}$ (rejected) vs $0.80\text{s}$ (accepted); $12.00\text{s}$ (accepted) vs $12.10\text{s}$ (rejected).
5. **Accented & Complex Text**: Complex Portuguese orthography with tildes, cedillas, circumflexes, mixed capitalization, and hyphens.
6. **Borderline Neural Confidence**: Exact confidence threshold evaluation ($p_{(1)} = 0.64$ rejected vs $p_{(1)} = 0.65$ accepted).
7. **Borderline Decision Margin**: Exact margin threshold evaluation ($\Delta = 0.19$ rejected vs $\Delta = 0.20$ accepted).
8. **Out-of-Domain (OOD) Rejection**: Weather queries, food orders, music requests, general conversational chatter, domain anchors without intent.
9. **Web Navigation Timeout & Error Resilience**: Network timeouts, missing DOM elements, malformed HTML pages, simulated driver crashes.

### Tier 3: Pairwise Combinations & Cross-Feature Interactions (`test_tier3_combinations.py`)
Validates interface contracts between adjacent subsystem pairs and multi-feature interaction matrices:
- **Audio Validation $\to$ Text Sanitization**: Valid audio generating transcripts passed cleanly to sanitizer.
- **Sanitization $\to$ Neural Multi-Hot Vectorization**: Transcribed phrases mapped to vectorizer matching vocabulary indices.
- **Audio Failure Short-Circuiting**: Invalid audio (silence, noise, duration) immediately halting pipeline without invoking neural classification.
- **Neural Rejection Short-Circuiting**: OOD or low-confidence classification halting pipeline without launching browser navigation.
- **Valid Intent to Navigation Dispatch**: Valid classifications correctly routing to corresponding target URLs.
- **Artifact Generation Pairwise Flow**: Navigation success generating both valid PNG screenshots and JSON metadata files.

### Tier 4: Real-World Scenarios & Full User Journeys (`test_tier4_scenarios.py`)
Simulates authentic end-to-end user journeys from voice input to verified browser navigation and artifact capture:
- **Journey 1 (Guia Universitário)**: User asks to explore courses and university guide $\to$ audio validated $\to$ intent classified $\to$ page navigated $\to$ screenshot saved.
- **Journey 2 (Manual do Candidato)**: User asks for exam rules, dates, and regulations $\to$ audio validated $\to$ intent classified $\to$ page navigated $\to$ screenshot saved.
- **Journey 3 (Formulário de Inscrição)**: User asks to register and fill out application $\to$ audio validated $\to$ intent classified $\to$ page navigated $\to$ screenshot saved.
- **Journey 4 (Silent Microphone Recovery)**: Microphone captures silence $\to$ system responds with helpful Portuguese guidance without crashing.
- **Journey 5 (Loud Background Noise Recovery)**: High background noise $\to$ system detects low SNR and advises finding a quieter room.
- **Journey 6 (Out-of-Domain Query Guidance)**: User asks off-topic question $\to$ system politely explains vestibular-only capabilities.
- **Journey 7 (Ambiguous Query Disambiguation)**: User mentions both manual and registration $\to$ system detects narrow margin and requests clarification.
- **Journey 8 (Long Monologue Handling)**: User speaks longer than 12 seconds $\to$ system explains duration limit.

---

## 3. Threshold Metrics & Tier Counts

| Test Tier | Focus Area | Minimum Required | Target Implemented |
|---|---|---|---|
| **Tier 1** | Feature Coverage (6 features $\times$ 5 tests) | 30 | 36 |
| **Tier 2** | Boundary Value Analysis (9 boundaries $\times$ 5 tests) | 45 | 48 |
| **Tier 3** | Pairwise & Combinatorial Flows | 15 | 18 |
| **Tier 4** | Real-World End-to-End Scenarios | 10 | 12 |
| **TOTAL** | Full E2E Test Suite | **100** | **114** |

---

## 4. Test Harness & Shared Fixtures (`tests/conftest.py`)

Shared fixtures provide hermetic test isolation:
- `synthetic_audio_gen`: Zero-dependency WAV audio generator for silence, white noise, mock speech, short audio, long audio, and corrupted bytes.
- `audio_validator`: Reference & production audio validation engine verifying RMS, silence ratio, SNR, and duration.
- `text_sanitizer`: Normalizer stripping diacritics via Unicode NFKD and cleaning punctuation.
- `neural_classifier`: 4-stage neural intent classifier with lexical gating and confidence/margin thresholding.
- `mock_navigator`: Deterministic offline web navigator with local HTML DOM parser and pure-Python valid PNG generator.
- `pipeline_coordinator`: Orchestrator chaining audio validation, neural routing, and navigation.
- `puc_html_fixtures`: Synthetic HTML DOM representations of PUC-Campinas Vestibular pages.
- `mock_png_generator`: Pure Python 1280x800 PNG byte generator using `struct` and `zlib`.

---

## 5. Execution Guide

Run full test suite:
```bash
pytest -v
```

Run specific tiers:
```bash
pytest tests/test_tier1_features.py -v
pytest tests/test_tier2_boundaries.py -v
pytest tests/test_tier3_combinations.py -v
pytest tests/test_tier4_scenarios.py -v
```

Execute standalone offline test runner (zero external dependencies):
```bash
python3 tests/conftest.py
```

"""
Comprehensive Unit Test Suite for REIA Neural Subsystem (Milestone 2).

Covers:
- Dataset balance, class mapping, URL routing, and OOD evaluation sets.
- VocabularyVectorizer Bag-of-Words multi-hot encoding and serialization.
- PurePythonMLP forward-pass, training routine, and PyTorch equivalence.
- LexicalSaliencyGate OOD, anchor-only, and empty input handling.
- Multi-metric confidence thresholding (Top-1, Decision Margin, Shannon Entropy).
- Model persistence (weights.json export/load, checkpointing).
- Boundary and edge case robustness.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
except ImportError:
    class _PytestRaisesContext:
        def __init__(self, expected_exc):
            self.expected_exc = expected_exc
            self.value = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.expected_exc} but no exception was raised.")
            if issubclass(exc_type, self.expected_exc):
                self.value = exc_val
                return True
            return False

    class _PytestStub:
        @staticmethod
        def raises(expected_exc):
            return _PytestRaisesContext(expected_exc)

    pytest = _PytestStub()

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
from reia.neural.model import (
    PurePythonMLP,
    ClassificadorVozPUC,
    set_deterministic_seed,
    train_pytorch_model,
    HAS_TORCH,
)
from reia.neural.persistence import (
    DEFAULT_WEIGHTS_PATH,
    export_model_to_json,
    load_model_from_json,
    load_weights_json,
    save_weights_json,
)
from reia.neural.thresholding import (
    ClassificationResult,
    LexicalSaliencyGate,
    NeuralIntentClassifier,
)
from reia.neural.vectorizer import VocabularyVectorizer


# ===========================================================================
# 1. Dataset & Canonical Mapping Tests
# ===========================================================================
class TestNeuralDataset:
    def test_classes_destino_canonical_mappings(self):
        assert len(CLASSES_DESTINO) == 3
        assert CLASSES_DESTINO[0] == "GUIA_UNIVERSITARIO"
        assert CLASSES_DESTINO[1] == "MANUAL_DO_CANDIDATO"
        assert CLASSES_DESTINO[2] == "FORMULARIO_INSCRICAO"

        assert INTENT_TO_ID["GUIA_UNIVERSITARIO"] == 0
        assert INTENT_TO_ID["MANUAL_DO_CANDIDATO"] == 1
        assert INTENT_TO_ID["FORMULARIO_INSCRICAO"] == 2

    def test_target_urls_puc_vestibular(self):
        assert "guia-universitario" in TARGET_URLS["GUIA_UNIVERSITARIO"]
        assert "manual-do-candidato" in TARGET_URLS["MANUAL_DO_CANDIDATO"]
        assert TARGET_URLS["FORMULARIO_INSCRICAO"] == "https://vestibular.puc-campinas.edu.br/"

    def test_training_dataset_balance(self):
        data = get_balanced_training_data()
        assert len(data) >= 30, f"Expected at least 30 samples, got {len(data)}"

        class_counts = {0: 0, 1: 0, 2: 0}
        for phrase, label in data:
            assert label in class_counts, f"Invalid label: {label}"
            class_counts[label] += 1
            assert isinstance(phrase, str) and len(phrase.strip()) > 0

        # Each class must have at least 10 samples for balanced training
        assert class_counts[0] >= 10
        assert class_counts[1] >= 10
        assert class_counts[2] >= 10
        assert class_counts[0] == class_counts[1] == class_counts[2]

    def test_training_phrases_sanitized_format(self):
        for phrase, _ in TRAINING_DATA:
            assert phrase == phrase.lower(), f"Phrase not lowercase: {phrase}"
            for char in phrase:
                assert char.isalnum() or char.isspace(), f"Forbidden punctuation/accent in phrase: {phrase}"

    def test_ood_and_ambiguous_evaluation_sets(self):
        ood = get_ood_samples()
        assert len(ood) >= 5
        assert any("tempo" in s for s in ood)
        assert any("pizza" in s for s in ood)

        amb = get_ambiguous_samples()
        assert len(amb) >= 3
        assert any("puc campinas vestibular" in s for s in amb)


# ===========================================================================
# 2. Vocabulary & Vectorizer Tests
# ===========================================================================
class TestVocabularyVectorizer:
    def test_fit_from_dataset_creates_sorted_vocabulary(self):
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        assert vec.vocab_size > 50
        assert vec.vocab == sorted(vec.vocab)
        assert "guia" in vec.vocab_to_idx
        assert "manual" in vec.vocab_to_idx
        assert "inscricao" in vec.vocab_to_idx
        assert "vestibular" in vec.vocab_to_idx

    def test_binary_multi_hot_encoding(self):
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        phrase = "mostrar guia vestibular puc"
        vector = vec.transform(phrase)

        assert len(vector) == vec.vocab_size
        assert all(v in (0.0, 1.0) for v in vector)

        assert vector[vec.vocab_to_idx["guia"]] == 1.0
        assert vector[vec.vocab_to_idx["vestibular"]] == 1.0
        assert vector[vec.vocab_to_idx["puc"]] == 1.0

        # Check an absent word is 0.0
        if "regras" in vec.vocab_to_idx:
            assert vector[vec.vocab_to_idx["regras"]] == 0.0

    def test_unknown_tokens_ignored(self):
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        vector = vec.transform("palavraestranhaxpto inexistente totalmente")
        assert len(vector) == vec.vocab_size
        assert sum(vector) == 0.0

    def test_empty_and_none_text_handling(self):
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        assert vec.transform("") == [0.0] * vec.vocab_size
        assert vec.transform("   ") == [0.0] * vec.vocab_size
        assert vec.transform(None) == [0.0] * vec.vocab_size

    def test_vectorizer_serialization_roundtrip(self):
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        data = vec.to_dict()
        assert "vocab" in data
        assert "vocab_size" in data

        restored = VocabularyVectorizer.from_dict(data)
        assert restored.vocab_size == vec.vocab_size
        assert restored.vocab == vec.vocab
        assert restored.transform("guia vestibular") == vec.transform("guia vestibular")

    def test_tensor_conversion_behavior(self):
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        if HAS_TORCH:
            t = vec.transform_tensor("mostrar guia")
            assert t.shape == (1, vec.vocab_size)
        else:
            with pytest.raises(RuntimeError) as exc_info:
                vec.transform_tensor("mostrar guia")
            assert "PyTorch is not installed" in str(exc_info.value)


# ===========================================================================
# 3. Neural Model & Training Tests
# ===========================================================================
class TestNeuralModels:
    def test_pure_python_mlp_forward_pass_dimensions(self):
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        model = PurePythonMLP.train_pure_python(
            TRAINING_DATA, vec, hidden_dim=16, num_classes=3, epochs=100, lr=0.05, seed=42
        )

        x = vec.transform("quero ver o guia de cursos da puc")
        logits, probs = model.forward(x)

        assert len(logits) == 3
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-6
        assert all(p >= 0.0 for p in probs)

    def test_pure_python_mlp_predict_accuracy(self):
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        model = PurePythonMLP.train_pure_python(
            TRAINING_DATA, vec, hidden_dim=16, num_classes=3, epochs=150, lr=0.05, seed=42
        )

        correct = 0
        for phrase, label in TRAINING_DATA:
            x = vec.transform(phrase)
            pred = model.predict(x)
            if pred == label:
                correct += 1

        accuracy = correct / len(TRAINING_DATA)
        assert accuracy >= 0.95, f"Training accuracy too low: {accuracy:.2%}"

    def test_pure_python_mlp_temperature_scaling(self):
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        model = PurePythonMLP.train_pure_python(
            TRAINING_DATA, vec, hidden_dim=16, num_classes=3, epochs=50, lr=0.05, seed=42
        )
        x = vec.transform("abrir manual de regras do candidato")

        _, p_norm = model.forward(x, temperature=1.0)
        _, p_high_temp = model.forward(x, temperature=10.0)

        # Higher temperature softens distribution towards uniform (1/3, 1/3, 1/3)
        assert max(p_high_temp) < max(p_norm)
        assert min(p_high_temp) > min(p_norm)

    def test_pure_python_mlp_serialization_roundtrip(self):
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        model = PurePythonMLP.train_pure_python(
            TRAINING_DATA, vec, hidden_dim=16, num_classes=3, epochs=50, lr=0.05, seed=42
        )
        data = model.to_dict()

        assert "w1" in data and "b1" in data
        assert "w2" in data and "b2" in data
        assert data["vocab_size"] == vec.vocab_size

        reconstructed = PurePythonMLP.from_dict(data)
        x = vec.transform("como faco a inscricao no vestibular puc")
        z1, p1 = model.forward(x)
        z2, p2 = reconstructed.forward(x)

        assert z1 == z2
        assert p1 == p2

    def test_pytorch_mlp_behavior(self):
        if not HAS_TORCH:
            with pytest.raises(RuntimeError):
                ClassificadorVozPUC(vocab_size=50)
            return

        set_deterministic_seed(42)
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        pt_model = train_pytorch_model(
            TRAINING_DATA, vec, hidden_dim=16, num_classes=3, epochs=100, lr=0.02, seed=42
        )
        assert pt_model.vocab_size == vec.vocab_size
        assert pt_model.hidden_dim == 16
        assert pt_model.num_classes == 3

    def test_pure_python_mlp_temperature_validation(self):
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        model = PurePythonMLP.train_pure_python(
            TRAINING_DATA, vec, hidden_dim=16, num_classes=3, epochs=10, lr=0.05, seed=42
        )
        x = vec.transform("abrir manual de regras")
        with pytest.raises(ValueError):
            model.forward(x, temperature=0.0)
        with pytest.raises(ValueError):
            model.forward(x, temperature=-1.0)

    def test_pure_python_mlp_backpropagation_exact_update_order(self):
        vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
        model = PurePythonMLP.train_pure_python(
            TRAINING_DATA, vec, hidden_dim=16, num_classes=3, epochs=150, lr=0.05, seed=42
        )
        # Verify 100% training accuracy with pre-update dh gradient
        correct = sum(1 for phrase, label in TRAINING_DATA if model.predict(vec.transform(phrase)) == label)
        assert correct == len(TRAINING_DATA)


# ===========================================================================
# 4. Lexical Saliency Gate Tests
# ===========================================================================
class TestLexicalSaliencyGate:
    def setup_method(self):
        self.gate = LexicalSaliencyGate()

    def test_empty_and_whitespace_rejection(self):
        passed, code, msg, tokens, _ = self.gate.evaluate("")
        assert passed is False
        assert code == "EMPTY_INPUT"
        assert "vazio" in msg.lower()

        passed, code, _, _, _ = self.gate.evaluate("     ")
        assert passed is False
        assert code == "EMPTY_INPUT"

    def test_pure_ood_query_rejection(self):
        ood_phrases = [
            "qual a previsao do tempo para amanha em sao paulo",
            "quero pedir uma pizza de calabresa bem quente",
            "tocar rock nacional no spotify",
            "me conte uma piada de papagaio",
        ]
        for phrase in ood_phrases:
            passed, code, msg, tokens, _ = self.gate.evaluate(phrase)
            assert passed is False, f"OOD phrase passed gate: {phrase}"
            assert code == "OOD_NO_DOMAIN_KEYWORDS"
            assert "fora de domínio" in msg.lower() or "vestibular" in msg.lower()
            assert len(tokens) == 0

    def test_domain_anchor_only_rejection(self):
        anchor_phrases = [
            "puc campinas vestibular",
            "vestibular da puc",
            "vestibulando puc campinas",
        ]
        for phrase in anchor_phrases:
            passed, code, msg, tokens, _ = self.gate.evaluate(phrase)
            assert passed is False, f"Anchor phrase passed gate: {phrase}"
            assert code == "AMBIGUOUS_DOMAIN_ONLY"
            assert "especifique" in msg.lower()
            assert len(tokens) > 0

    def test_valid_domain_phrases_pass_gate(self):
        valid_phrases = [
            "mostrar guia de cursos vestibular puc",
            "abrir manual do candidato regras da prova",
            "como faco inscricao vestibular puc campinas",
        ]
        for phrase in valid_phrases:
            passed, code, msg, tokens, intent_scores = self.gate.evaluate(phrase)
            assert passed is True, f"Valid phrase failed gate: {phrase}"
            assert code is None
            assert len(tokens) >= 2
            assert sum(intent_scores.values()) >= 1

    def test_conflicting_multi_intent_rejection_balanced_and_unbalanced(self):
        conflicts = [
            # Balanced
            "quero ver o guia e o manual do candidato",
            "abrir edital e formulario de inscricao",
            # Unbalanced
            "mostrar apresentacao do guia e edital da puc",
            "quero consultar o cronograma do edital e me inscrever",
            "abrir catalogo de graduacao da puc e edital de regras",
        ]
        for phrase in conflicts:
            passed, code, msg, tokens, intent_scores = self.gate.evaluate(phrase)
            assert passed is False, f"Conflicting phrase should have failed gate: {phrase}"
            assert code == "AMBIGUOUS_INTENT", f"Expected AMBIGUOUS_INTENT, got {code} for {phrase}"
            assert "ambíguo" in msg.lower() or "escolha" in msg.lower()


# ===========================================================================
# 5. Multi-Metric Thresholding & Pipeline Tests
# ===========================================================================
class TestMultiMetricThresholding:
    def setup_method(self):
        self.classifier = NeuralIntentClassifier()

    def test_high_confidence_guia_accepted(self):
        phrase = "quero ver a apresentacao do guia da puc campinas"
        res = self.classifier.classify(phrase)

        assert res.is_valid_intent is True
        assert res.intent_name == "GUIA_UNIVERSITARIO"
        assert res.intent_id == 0
        assert res.confidence >= 0.65
        assert res.margin >= 0.20
        assert res.entropy <= 0.65
        assert res.rejection_code is None
        assert res.target_url == TARGET_URLS["GUIA_UNIVERSITARIO"]

    def test_high_confidence_manual_accepted(self):
        phrase = "abrir manual do candidato com as regras da prova e edital"
        res = self.classifier.classify(phrase)

        assert res.is_valid_intent is True
        assert res.intent_name == "MANUAL_DO_CANDIDATO"
        assert res.intent_id == 1
        assert res.confidence >= 0.65
        assert res.margin >= 0.20
        assert res.rejection_code is None
        assert res.target_url == TARGET_URLS["MANUAL_DO_CANDIDATO"]

    def test_high_confidence_inscricao_accepted(self):
        phrase = "como faco para me inscrever no vestibular da puc campinas"
        res = self.classifier.classify(phrase)

        assert res.is_valid_intent is True
        assert res.intent_name == "FORMULARIO_INSCRICAO"
        assert res.intent_id == 2
        assert res.confidence >= 0.65
        assert res.margin >= 0.20
        assert res.rejection_code is None
        assert res.target_url == TARGET_URLS["FORMULARIO_INSCRICAO"]

    def test_low_confidence_rejection(self):
        strict_clf = NeuralIntentClassifier(confidence_threshold=0.999999)
        res = strict_clf.classify("abrir guia da puc")

        assert res.is_valid_intent is False
        assert res.rejection_code == "LOW_CONFIDENCE"
        assert "repita o comando" in res.user_message.lower()

    def test_ambiguous_intent_narrow_margin_rejection(self):
        # Asking for both manual and guide
        ambiguous = "quero ver o guia e o manual do candidato no vestibular da puc"
        res = self.classifier.classify(ambiguous)

        assert res.is_valid_intent is False
        assert res.rejection_code == "AMBIGUOUS_INTENT"
        assert "ambíguo" in res.user_message.lower() or "escolha" in res.user_message.lower()

    def test_ambiguous_intent_tie_margin_rejection(self):
        # Asking for edital (manual) and inscricao simultaneously
        tie_phrase = "onde vejo o edital e faco a inscricao no vestibular da puc"
        res = self.classifier.classify(tie_phrase)

        assert res.is_valid_intent is False
        assert res.rejection_code == "AMBIGUOUS_INTENT"
        assert res.margin < 0.20

    def test_strict_entropy_threshold_rejection(self):
        strict_entropy_clf = NeuralIntentClassifier(entropy_threshold=0.001)
        res = strict_entropy_clf.classify("vestibular puc guia cursos")

        assert res.is_valid_intent is False
        assert res.rejection_code == "AMBIGUOUS_INTENT"

    def test_probability_distribution_properties(self):
        phrase = "quero conhecer os cursos da puc campinas no guia"
        res = self.classifier.classify(phrase)

        assert isinstance(res.probabilities, dict)
        assert len(res.probabilities) == 3
        total = sum(res.probabilities.values())
        assert abs(total - 1.0) < 1e-4
        assert all(0.0 <= p <= 1.0 for p in res.probabilities.values())

    def test_accent_and_punctuation_insensitivity(self):
        accented = "Como faço para me inscrever no vestibular da PUC-Campinas?"
        unaccented = "como faco para me inscrever no vestibular da puc campinas"

        res_acc = self.classifier.classify(accented)
        res_unacc = self.classifier.classify(unaccented)

        assert res_acc.is_valid_intent is True
        assert res_unacc.is_valid_intent is True
        assert res_acc.intent_name == res_unacc.intent_name == "FORMULARIO_INSCRICAO"
        assert abs(res_acc.confidence - res_unacc.confidence) < 0.05

    def test_natural_compound_queries_with_course_and_exam_accepted(self):
        # Queries with course/exam should not collide with intent keywords
        queries = [
            ("quero me inscrever no curso de administracao da puc", "FORMULARIO_INSCRICAO"),
            ("fazer inscricao para a prova", "FORMULARIO_INSCRICAO"),
            ("onde vejo as regras do vestibular para o curso de enfermagem", "MANUAL_DO_CANDIDATO"),
        ]
        for phrase, expected_intent in queries:
            res = self.classifier.classify(phrase)
            assert res.is_valid_intent is True, f"Failed for valid query: {phrase}"
            assert res.intent_name == expected_intent, f"Wrong intent for {phrase}: {res.intent_name}"
            assert res.confidence > 0.90, f"Confidence too low for {phrase}: {res.confidence}"
            assert res.rejection_code is None
            assert res.target_url == TARGET_URLS[expected_intent]

    def test_stage2_always_executes_actual_neural_forward_pass(self):
        # Spying on model.forward to confirm execution in Stage 2
        called = []
        original_forward = self.classifier.model.forward

        def spy_forward(x, temperature=1.0):
            called.append((x, temperature))
            return original_forward(x, temperature=temperature)

        self.classifier.model.forward = spy_forward
        try:
            res = self.classifier.classify("quero me inscrever no vestibular da puc")
            assert len(called) == 1, "Stage 2 failed to execute model.forward"
            assert res.is_valid_intent is True
        finally:
            self.classifier.model.forward = original_forward

    def test_classifier_temperature_validation(self):
        with pytest.raises(ValueError):
            NeuralIntentClassifier(temperature=0.0)
        with pytest.raises(ValueError):
            NeuralIntentClassifier(temperature=-0.5)

    def test_dynamic_entropy_normalization_properties(self):
        # Test 3 classes uniform distribution (maximum entropy = 1.0)
        uniform3 = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        h_norm3 = NeuralIntentClassifier._compute_entropy_norm(uniform3)
        assert abs(h_norm3 - 1.0) < 1e-4

        # Test 2 classes uniform distribution (maximum entropy = 1.0)
        uniform2 = [0.5, 0.5]
        h_norm2 = NeuralIntentClassifier._compute_entropy_norm(uniform2)
        assert abs(h_norm2 - 1.0) < 1e-4

        # Test peaked distribution (low entropy)
        peaked = [0.98, 0.01, 0.01]
        h_peaked = NeuralIntentClassifier._compute_entropy_norm(peaked)
        assert h_peaked < 0.20


# ===========================================================================
# 6. Model Persistence Tests
# ===========================================================================
class TestModelPersistence:
    def test_load_bundled_weights_json(self):
        data = load_weights_json(str(DEFAULT_WEIGHTS_PATH))
        assert "vocab" in data
        assert "vocab_size" in data
        assert "w1" in data and "b1" in data
        assert "w2" in data and "b2" in data
        assert len(data["w1"]) == data["hidden_dim"] == 16
        assert len(data["w2"]) == data["num_classes"] == 3

    def test_export_and_load_roundtrip_json(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_weights = Path(td) / "custom_weights.json"
            vec = VocabularyVectorizer.from_dataset(TRAINING_DATA)
            model = PurePythonMLP.train_pure_python(
                TRAINING_DATA, vec, hidden_dim=16, num_classes=3, epochs=100, lr=0.05, seed=42
            )

            exported_path = export_model_to_json(model, vec, str(tmp_weights))
            assert Path(exported_path).is_file()

            loaded_model, loaded_vec = load_model_from_json(exported_path)
            assert loaded_vec.vocab == vec.vocab
            assert loaded_model.vocab_size == model.vocab_size

            x = vec.transform("mostrar guia de cursos puc")
            z1, p1 = model.forward(x)
            z2, p2 = loaded_model.forward(x)
            assert z1 == z2
            assert p1 == p2

    def test_missing_weights_file_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_weights_json("/nonexistent/path/weights.json")


# ===========================================================================
# 7. Classification Result Contract Properties
# ===========================================================================
class TestClassificationResultContract:
    def test_classification_result_field_access(self):
        res = ClassificationResult(
            is_valid_intent=True,
            intent_id=0,
            intent_name="GUIA_UNIVERSITARIO",
            target_url="https://vestibular.puc-campinas.edu.br/guia-universitario/",
            confidence=0.95,
            margin=0.90,
            entropy=0.15,
            probabilities={"GUIA_UNIVERSITARIO": 0.95, "MANUAL_DO_CANDIDATO": 0.03, "FORMULARIO_INSCRICAO": 0.02},
            rejection_code=None,
            user_message="Navegando para GUIA_UNIVERSITARIO",
            matched_domain_tokens=["guia", "puc"],
            sanitized_text="mostrar guia puc",
        )

        assert res.is_valid_intent is True
        assert res.is_accepted is True
        assert res.intent == "GUIA_UNIVERSITARIO"
        assert res.rejection_reason is None
        assert res.user_feedback_message == "Navegando para GUIA_UNIVERSITARIO"


# ===========================================================================
# Standalone Test Execution Entrypoint
# ===========================================================================
if __name__ == "__main__":
    import inspect
    import sys
    import time

    start = time.time()
    test_classes = [
        TestNeuralDataset,
        TestVocabularyVectorizer,
        TestNeuralModels,
        TestLexicalSaliencyGate,
        TestMultiMetricThresholding,
        TestModelPersistence,
        TestClassificationResultContract,
    ]

    total_passed = 0
    total_failed = 0

    print("======================================================================")
    print("REIA Neural Subsystem (M2) Unit Test Suite")
    print("======================================================================")

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_") and callable(getattr(instance, m))]
        print(f"\nRunning {cls.__name__} ({len(methods)} tests)...")
        passed = 0
        failed = 0
        for method_name in methods:
            test_fn = getattr(instance, method_name)
            try:
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                sig = inspect.signature(test_fn)
                test_fn()
                passed += 1
            except Exception as e:
                failed += 1
                print(f"  FAIL: {method_name}: {e}")
        print(f"  Result: {passed} passed, {failed} failed (total {len(methods)})")
        total_passed += passed
        total_failed += failed

    elapsed = time.time() - start
    print("\n======================================================================")
    print(f"Neural Test Suite Summary: {total_passed} passed, {total_failed} failed in {elapsed:.2f}s")
    print("======================================================================")

    if total_failed > 0:
        sys.exit(1)
    sys.exit(0)

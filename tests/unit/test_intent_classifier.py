"""Unit tests for src.agent_loop.intent_classifier and intent routing helpers."""

import pytest

from src import (
    INTENT_LABELS,
    IntentPipeline,
    IntentionClassificationPipeline,
    generate_intent_examples,
    train_intent_classifier,
    write_intent_examples,
)
from src.internal.document_index.text import Vocabulary
from src.model.intent_classifier import _IntentClassifier


def test_vocabulary_build_and_encode_support_sequence_training():
    vocab = Vocabulary()
    vocab.build([["buy", "phone"], ["buy", "laptop"]], min_freq=1)

    encoded = vocab.encode(["buy", "unknown", "phone"])

    assert vocab.token2cnt["buy"] == 2
    assert encoded[0] != 0
    assert encoded[1] == 0


def test_pipeline_predict_requires_training():
    pytest.importorskip("torch")
    pipeline = IntentPipeline()
    try:
        pipeline.predict(["buy", "phone"])
    except RuntimeError as exc:
        assert "not trained" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("predict() should require training first")


def test_intention_pipeline_alias_matches_intent_pipeline():
    assert IntentionClassificationPipeline is IntentPipeline


def test_resolve_search_settings_search_is_retrieval_heavy():
    from src.model.intent_classifier import IntentPrediction, resolve_search_settings

    t, s, r, a, meta = resolve_search_settings(
        IntentPrediction(intent="search", confidence=0.9),
        topk=5,
        max_search_limit=2,
        require_evidence=False,
        allow_internal_knowledge=True,
    )
    assert (t, s, r, a) == (8, 3, True, False)
    assert meta["intent_policy_applied"] is True


def test_low_confidence_leaves_settings_unchanged():
    from src.model.intent_classifier import IntentPrediction, resolve_search_settings

    t, s, r, a, meta = resolve_search_settings(
        IntentPrediction(intent="search", confidence=0.1),
        topk=5,
        max_search_limit=2,
        require_evidence=False,
        allow_internal_knowledge=True,
    )
    assert (t, s, r, a) == (5, 2, False, True)
    assert meta["intent_policy_applied"] is False


def test_resolve_search_settings_chat_passthrough():
    from src.model.intent_classifier import IntentPrediction, resolve_search_settings

    t, s, r, a, _meta = resolve_search_settings(
        IntentPrediction(intent="chat", confidence=0.9),
        topk=5,
        max_search_limit=2,
        require_evidence=False,
        allow_internal_knowledge=True,
    )
    assert (t, s, r, a) == (5, 2, False, True)


def test_intent_labels_are_route_strategy_values():
    from src.model.intent_classifier import INTENT_LABELS

    assert INTENT_LABELS == ["chat", "search", "tool"]


def test_intent_pipeline_save_and_load_round_trip(tmp_path):
    """A trained pipeline saved and reloaded produces the same prediction."""
    pytest.importorskip("torch")

    data = [
        (["find", "documents"], "search"),
        (["search", "results"], "search"),
        (["what", "is", "faiss"], "chat"),
        (["how", "does", "search", "work"], "chat"),
    ]
    pipeline = IntentPipeline()
    pipeline.train(data, epochs=3, min_freq=1)
    original_pred = pipeline.predict(["buy", "laptop"])

    save_path = str(tmp_path / "intent.pt")
    pipeline.save(
        save_path,
        dataset_fingerprint="test-dataset",
        promoted_min_confidence=None,
    )

    loaded = IntentPipeline.load(save_path)
    loaded_pred = loaded.predict(["buy", "laptop"])

    assert loaded.is_trained is True
    assert loaded_pred.intent == original_pred.intent
    assert abs(loaded_pred.confidence - original_pred.confidence) < 1e-4


def test_intent_pipeline_save_requires_training(tmp_path):
    pytest.importorskip("torch")
    pipeline = IntentPipeline()
    try:
        pipeline.save(
            str(tmp_path / "intent.pt"),
            dataset_fingerprint="test-dataset",
            promoted_min_confidence=None,
        )
    except RuntimeError as exc:
        assert "trained" in str(exc).lower()
    else:
        raise AssertionError("save() should require training first")


def test_generate_intent_examples_from_corpus_and_vocabulary(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '{"id": 7, "title": "FAISS", "contents": "Dense vector search indexing."}\n',
        encoding="utf-8",
    )
    vocabulary_path = tmp_path / "vocab.json"
    vocabulary_path.write_text(
        '{"vocabulary": {"token2idx": {"dense": 2, "vector": 3, "search": 4}}}',
        encoding="utf-8",
    )

    examples = generate_intent_examples(
        corpus_path=corpus_path,
        vocabulary_path=vocabulary_path,
    )

    assert len(examples) == 20
    assert {example["label"] for example in examples} == set(INTENT_LABELS)
    assert all(example["source_doc_id"] == 7 for example in examples)


def test_write_intent_examples_round_trip(tmp_path):
    output_path = tmp_path / "intent_examples.json"
    examples = [{"text": "What is FAISS?", "label": "chat"}]

    write_intent_examples(examples, output_path)

    assert '"What is FAISS?"' in output_path.read_text(encoding="utf-8")


def test_train_intent_classifier_utility_saves_pipeline(tmp_path):
    pytest.importorskip("torch")

    examples_path = tmp_path / "intent_examples.json"
    write_intent_examples(
        [
            {"text": "find documents", "label": "search"},
            {"text": "search results", "label": "search"},
            {"text": "what is faiss", "label": "chat"},
            {"text": "how does search work", "label": "chat"},
        ],
        examples_path,
    )
    output_path = tmp_path / "intent.pt"

    result = train_intent_classifier(
        examples_path=examples_path,
        output_path=output_path,
        epochs=1,
        min_freq=1,
        vocab_size=128,
        embedding_dim=16,
        hidden_dim=32,
    )

    assert output_path.exists()
    assert result.num_examples == 4
    assert result.label_counts == {"search": 2, "chat": 2}
    assert result.pipeline.is_trained is True


def test_batch_padding_does_not_change_prediction_logits():
    torch = pytest.importorskip("torch")
    model = _IntentClassifier(
        vocab_size=8, embedding_dim=4, hidden_dim=4, num_classes=3
    )
    with torch.no_grad():
        model._net.embedding.weight.copy_(
            torch.arange(32, dtype=torch.float32).reshape(8, 4)
        )
        for layer in (model._net.fc1, model._net.fc2, model._net.fc3):
            layer.weight.fill_(0.1)
            layer.bias.zero_()
    model._net.eval()

    single = model._net(model._pad_sequences([[1]]))
    batched = model._net(model._pad_sequences([[1], [1, 2, 3]]))[0:1]

    assert torch.allclose(single, batched, atol=1e-6)


def test_training_rejects_vocabulary_that_exceeds_embedding_table():
    pytest.importorskip("torch")
    pipeline = IntentPipeline(vocab_size=2, embedding_dim=4, hidden_dim=8)

    with pytest.raises(ValueError, match="vocab_size"):
        pipeline.train(
            [(["one"], "chat"), (["two"], "search"), (["three"], "tool")],
            epochs=1,
            min_freq=1,
        )


def test_training_seed_reproduces_predictions():
    pytest.importorskip("torch")
    data = [
        (["what", "is", "faiss"], "chat"),
        (["find", "documents"], "search"),
        (["run", "the", "tool"], "tool"),
    ]
    first = IntentPipeline(vocab_size=16, embedding_dim=4, hidden_dim=8)
    second = IntentPipeline(vocab_size=16, embedding_dim=4, hidden_dim=8)

    first.train(data, epochs=2, min_freq=1, seed=31)
    second.train(data, epochs=2, min_freq=1, seed=31)

    first_prediction = first.predict(["find", "documents"])
    second_prediction = second.predict(["find", "documents"])
    assert second_prediction.intent == first_prediction.intent
    assert second_prediction.confidence == pytest.approx(first_prediction.confidence)


def test_save_writes_version_two_checkpoint_contract(tmp_path):
    torch = pytest.importorskip("torch")
    pipeline = IntentPipeline(vocab_size=16, embedding_dim=4, hidden_dim=8)
    pipeline.train(
        [(["hello"], "chat"), (["find"], "search"), (["run"], "tool")],
        epochs=1,
        min_freq=1,
        seed=17,
    )
    path = tmp_path / "intent.pt"

    pipeline.save(
        str(path),
        dataset_fingerprint="sha256:abc",
        promoted_min_confidence=None,
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["version"] == 2
    assert checkpoint["intent_labels"] == ["chat", "search", "tool"]
    assert checkpoint["dataset_fingerprint"] == "sha256:abc"
    assert checkpoint["promoted_min_confidence"] is None
    assert checkpoint["preprocessing"] == {
        "tokenizer": "document_index.tokenize_text",
        "padding_id": 0,
        "pooling": "masked_mean",
    }


@pytest.mark.parametrize("labels", [["search", "chat", "tool"], ["chat", "search"]])
def test_load_rejects_checkpoint_with_incompatible_intent_labels(tmp_path, labels):
    torch = pytest.importorskip("torch")
    pipeline = IntentPipeline(vocab_size=16, embedding_dim=4, hidden_dim=8)
    pipeline.train(
        [(["hello"], "chat"), (["find"], "search"), (["run"], "tool")],
        epochs=1,
        min_freq=1,
        seed=17,
    )
    path = tmp_path / "intent.pt"
    pipeline.save(
        str(path),
        dataset_fingerprint="sha256:abc",
        promoted_min_confidence=None,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["intent_labels"] = labels
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="intent_labels"):
        IntentPipeline.load(str(path))


def test_load_rejects_version_one_checkpoint_with_retraining_message(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "legacy-intent.pt"
    torch.save({"version": 1}, path)

    with pytest.raises(ValueError, match="retrain"):
        IntentPipeline.load(str(path))


def test_load_rejects_model_state_dimensions_that_disagree_with_config(tmp_path):
    torch = pytest.importorskip("torch")
    pipeline = IntentPipeline(vocab_size=16, embedding_dim=4, hidden_dim=8)
    pipeline.train(
        [(["hello"], "chat"), (["find"], "search"), (["run"], "tool")],
        epochs=1,
        min_freq=1,
        seed=17,
    )
    path = tmp_path / "intent.pt"
    pipeline.save(
        str(path),
        dataset_fingerprint="sha256:abc",
        promoted_min_confidence=None,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["model_state"]["embedding.weight"] = checkpoint["model_state"][
        "embedding.weight"
    ][:-1]
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="config"):
        IntentPipeline.load(str(path))

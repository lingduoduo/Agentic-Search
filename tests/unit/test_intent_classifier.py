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
from src.model import intent_training
from src.model.intent_classifier import _IntentClassifier
from src.model.intent_pretrained import write_pretrained_bundle


def _bundle(dim: int = 8):
    """A tiny frozen bundle: [PAD], [UNK], and three real tokens."""
    import numpy as np

    from src.model.intent_pretrained import PretrainedBundle
    from src.model.wordpiece import WordPieceVocabulary

    tokens = (
        ["[PAD]"]
        + [f"[unused{index}]" for index in range(99)]
        + ["[UNK]", "find", "explain", "send", "the", "runbook"]
    )
    rng = np.random.default_rng(17)
    embeddings = rng.normal(size=(len(tokens), dim)).astype(np.float16)
    return PretrainedBundle(
        vocabulary=WordPieceVocabulary.from_tokens(tokens), embeddings=embeddings
    )


def test_vocabulary_build_and_encode_support_sequence_training():
    vocab = Vocabulary()
    vocab.build([["buy", "phone"], ["buy", "laptop"]], min_freq=1)

    encoded = vocab.encode(["buy", "unknown", "phone"])

    assert vocab.token2cnt["buy"] == 2
    assert encoded[0] != 0
    assert encoded[1] == 0


def test_pipeline_predict_requires_training():
    pytest.importorskip("torch")
    pipeline = IntentPipeline(_bundle(), hidden_dim=16)
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
    pipeline = IntentPipeline(_bundle(), hidden_dim=16)
    pipeline.train(data, epochs=3)
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
    pipeline = IntentPipeline(_bundle(), hidden_dim=16)
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

    assert len(examples) == len(intent_training._FRAMES)
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
    bundle_path = tmp_path / "bundle"
    bundle = _bundle()
    write_pretrained_bundle(
        bundle_path,
        tokens=bundle.vocabulary.tokens,
        embeddings=bundle.embeddings,
    )

    result = train_intent_classifier(
        examples_path=examples_path,
        output_path=output_path,
        pretrained_path=bundle_path,
        epochs=1,
        hidden_dim=32,
    )

    assert output_path.exists()
    assert result.num_examples == 4
    assert result.label_counts == {"search": 2, "chat": 2}
    assert result.pipeline.is_trained is True


def test_batch_padding_does_not_change_prediction_logits():
    torch = pytest.importorskip("torch")
    import numpy as np

    matrix = np.arange(32, dtype=np.float16).reshape(8, 4)
    model = _IntentClassifier(matrix, 4, 3)
    with torch.no_grad():
        for layer in (model._net.fc1, model._net.fc2, model._net.fc3):
            layer.weight.fill_(0.1)
            layer.bias.zero_()
    model._net.eval()

    single = model._net(model._pad_sequences([[1]]))
    batched = model._net(model._pad_sequences([[1], [1, 2, 3]]))[0:1]

    assert torch.allclose(single, batched, atol=1e-6)


def test_pipeline_derives_its_dimensions_from_the_bundle():
    pytest.importorskip("torch")
    bundle = _bundle(dim=8)

    pipeline = IntentPipeline(bundle, hidden_dim=16)

    assert pipeline._model._net.embedding.num_embeddings == bundle.size
    assert pipeline._model._net.embedding.embedding_dim == 8


def test_pretrained_embeddings_are_frozen_by_training():
    torch = pytest.importorskip("torch")
    pipeline = IntentPipeline(_bundle(), hidden_dim=16)
    before = pipeline._model._net.embedding.weight.detach().clone()

    pipeline.train(
        [
            (["find", "the", "runbook"], "search"),
            (["explain", "the", "runbook"], "chat"),
            (["send", "the", "runbook"], "tool"),
        ],
        epochs=25,
        seed=17,
    )

    assert torch.equal(pipeline._model._net.embedding.weight.detach(), before)
    assert pipeline._model._net.embedding.weight.requires_grad is False


def test_unseen_word_is_read_as_wordpieces_not_dropped():
    """The defect this change exists to fix: 'runbooks' was unreadable."""
    pytest.importorskip("torch")
    pipeline = IntentPipeline(_bundle(), hidden_dim=16)

    encoded = pipeline._encode_text("the runbook")

    assert encoded == [
        pipeline._bundle.vocabulary.encode("the")[0],
        pipeline._bundle.vocabulary.encode("runbook")[0],
    ]
    assert 100 not in encoded  # [UNK] never fires for in-vocabulary words


def test_empty_query_encodes_to_a_single_unknown():
    pytest.importorskip("torch")
    pipeline = IntentPipeline(_bundle(), hidden_dim=16)

    assert pipeline._encode_text("!!!") == [100]


def test_save_writes_version_four_checkpoint_contract(tmp_path):
    torch = pytest.importorskip("torch")
    pipeline = IntentPipeline(_bundle(), hidden_dim=16)
    pipeline.train(
        [
            (["find", "the"], "search"),
            (["explain", "the"], "chat"),
            (["send", "the"], "tool"),
        ],
        epochs=1,
        seed=17,
    )
    path = tmp_path / "intent.pt"

    pipeline.save(str(path), dataset_fingerprint="sha256:abc")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["version"] == 4
    assert checkpoint["preprocessing"] == {
        "tokenizer": "wordpiece",
        "padding_id": 0,
        "unknown_id": 100,
        "pooling": "masked_mean",
        "embeddings": "frozen_pretrained",
    }


def test_checkpoint_round_trip_preserves_predictions(tmp_path):
    pytest.importorskip("torch")
    pipeline = IntentPipeline(_bundle(), hidden_dim=16)
    pipeline.train(
        [
            (["find", "the"], "search"),
            (["explain", "the"], "chat"),
            (["send", "the"], "tool"),
        ],
        epochs=5,
        seed=17,
    )
    path = tmp_path / "intent.pt"
    pipeline.save(str(path), dataset_fingerprint="sha256:abc")

    reloaded = IntentPipeline.load(str(path))

    before = pipeline.predict_text("find the runbook")
    after = reloaded.predict_text("find the runbook")
    assert after.intent == before.intent
    assert after.confidence == pytest.approx(before.confidence, abs=1e-6)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_load_rejects_every_earlier_checkpoint_version(tmp_path, version):
    torch = pytest.importorskip("torch")
    path = tmp_path / "old-intent.pt"
    torch.save({"version": version}, path)

    with pytest.raises(ValueError, match="retrain"):
        IntentPipeline.load(str(path))


def test_training_seed_reproduces_predictions():
    pytest.importorskip("torch")
    data = [
        (["what", "is", "faiss"], "chat"),
        (["find", "documents"], "search"),
        (["run", "the", "tool"], "tool"),
    ]
    first = IntentPipeline(_bundle(), hidden_dim=16)
    second = IntentPipeline(_bundle(), hidden_dim=16)

    first.train(data, epochs=2, seed=31)
    second.train(data, epochs=2, seed=31)

    first_prediction = first.predict(["find", "documents"])
    second_prediction = second.predict(["find", "documents"])
    assert second_prediction.intent == first_prediction.intent
    assert second_prediction.confidence == pytest.approx(first_prediction.confidence)


@pytest.mark.parametrize("labels", [["search", "chat", "tool"], ["chat", "search"]])
def test_load_rejects_checkpoint_with_incompatible_intent_labels(tmp_path, labels):
    torch = pytest.importorskip("torch")
    pipeline = IntentPipeline(_bundle(), hidden_dim=16)
    pipeline.train(
        [(["hello"], "chat"), (["find"], "search"), (["run"], "tool")],
        epochs=1,
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
    pipeline = IntentPipeline(_bundle(), hidden_dim=16)
    pipeline.train(
        [(["hello"], "chat"), (["find"], "search"), (["run"], "tool")],
        epochs=1,
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

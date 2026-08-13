# Intent model: pretrained wordpiece embeddings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate out-of-vocabulary from the intent model by reading queries as pretrained wordpieces instead of a 147-token vocabulary built from its own training set.

**Architecture:** A dependency-free WordPiece tokenizer over MiniLM's 30522-token vocabulary feeds a frozen pretrained 30522 × 384 embedding matrix, mean-pooled into the existing MLP head. No transformer runs at serving time — the per-request cost stays a dict lookup plus a mean. The matrix and vocabulary are extracted once by an offline command and baked into the checkpoint, which goes to format version 4.

**Tech Stack:** Python 3, PyTorch (lazily imported), numpy, sentence-transformers (offline extraction only), pytest.

## Global Constraints

- `INTENT_LABELS = ["chat", "search", "tool"]` is unchanged.
- `src/model/wordpiece.py` must import with **no torch and no transformers** — the unit-test CI job installs neither, and this repo has twice shipped collection failures from unguarded imports (#356, re-fixed in #418).
- `Vocabulary` in `src/internal/document_index/text.py` is **not** modified — `src/internal/document_index/cli.py` shares it.
- No change to the runtime cascade, the clarification path, `intent_routing.py`, `ml_intent.py`, any dispatcher, or the frontend.
- No change to the generated dataset, the splits, the promotion gates, the calibration report, or the realistic-accuracy instrument.
- Pretrained embeddings are **frozen**. They are never a trained parameter.
- Token ids come from BERT's layout: `[PAD]` = 0, `[UNK]` = 100, `[CLS]` = 101, `[SEP]` = 102. `[CLS]`/`[SEP]` are **not** added — this is mean-pooling, not an encoder.
- WordPiece guard is `max_input_chars_per_word = 100`, the value HuggingFace's tokenizer reports.
- Checkpoint format version becomes `4`. Versions 1, 2, and 3 are rejected with a retraining message.
- The stored matrix is fp16 (23MB); it is cast to fp32 at load.
- `data/` is gitignored; anything tracked under it needs `git add -f`. The pretrained bundle is **not** tracked — it is regenerable by its command.
- Torch-importing tests need `pytest.importorskip("torch")`.
- Work happens on branch `feat/intent-pretrained-wordpiece` (already created off `db05629`). One commit per task. Open a PR after the last task; never merge it.
- Lint before each commit: `ruff check . --fix && ruff format .`

**Reference values, confirmed against the loaded tokenizer** — use these exact numbers in tests:

| text | wordpieces | ids |
|---|---|---|
| `postmortem` | `post ##mo ##rte ##m` | 2695, 5302, 19731, 2213 |
| `standup` | `stand ##up` | 3233, 6279 |
| `dashboard` | `dashboard` | 24923 |
| `reranker` | `re ##rank ##er` | 2128, 26763, 2121 |
| `p95` | `p ##9 ##5` | 1052, 2683, 2629 |
| `the summary` | `the summary` | 1996, 12654 |

Matrix: `(30522, 384)`, `torch.float32` at source. `vocab.txt` is written one token per line, ordered by id, so line *i* is token id *i*; line 0 is `[PAD]`.

## File structure

| file | responsibility |
|---|---|
| `src/model/wordpiece.py` (new) | Pure-Python WordPiece: load `vocab.txt`, encode text to ids. No torch. |
| `src/model/intent_pretrained.py` (new) | The bundle: extract from sentence-transformers, write/read `vocab.txt` + `embeddings.fp16.npy`, validate size agreement. numpy only. |
| `src/model/intent_classifier.py` | `IntentPipeline` reads wordpieces against the frozen matrix; checkpoint v4. |
| `src/model/intent_training.py` | `embeddings` subcommand, `--pretrained` flag, removal of `min_freq`/`vocab_size`/`embedding_dim`. |
| `docs/training-and-evaluation.md` | Operator workflow: the new command, the artifact size, the measured numbers. |

---

### Task 1: WordPiece tokenizer

**Files:**
- Create: `src/model/wordpiece.py`
- Test: `tests/unit/test_wordpiece.py` (new)

**Interfaces:**
- Consumes: nothing.
- Produces: `PAD_ID = 0`, `UNK_ID = 100`, `MAX_CHARS_PER_WORD = 100`; class `WordPieceVocabulary` with `from_file(path: Path) -> WordPieceVocabulary`, `from_tokens(tokens: Sequence[str]) -> WordPieceVocabulary`, `encode(text: str) -> list[int]`, and property `size: int`. Task 3 calls `encode`; Task 2 calls `from_tokens`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_wordpiece.py`:

```python
from pathlib import Path

import pytest

from src.model.wordpiece import (
    MAX_CHARS_PER_WORD,
    PAD_ID,
    UNK_ID,
    WordPieceVocabulary,
)

# A miniature vocabulary that exercises every branch. Ids are positional.
_TOKENS = (
    ["[PAD]"] + [f"[unused{index}]" for index in range(99)] + ["[UNK]"] + [
        "the",
        "post",
        "##mo",
        "##rte",
        "##m",
        "stand",
        "##up",
        "dashboard",
    ]
)
_ID = {token: index for index, token in enumerate(_TOKENS)}


def _vocab() -> WordPieceVocabulary:
    return WordPieceVocabulary.from_tokens(_TOKENS)


def test_special_ids_follow_the_bert_layout():
    assert PAD_ID == 0
    assert UNK_ID == 100
    assert _vocab().size == len(_TOKENS)


def test_whole_word_encodes_to_its_own_id():
    assert _vocab().encode("dashboard") == [_ID["dashboard"]]


def test_unseen_word_decomposes_into_continuations():
    assert _vocab().encode("postmortem") == [
        _ID["post"],
        _ID["##mo"],
        _ID["##rte"],
        _ID["##m"],
    ]


def test_greedy_match_prefers_the_longest_prefix():
    assert _vocab().encode("standup") == [_ID["stand"], _ID["##up"]]


def test_undecomposable_word_falls_back_to_unknown_once():
    # 'zzz' shares no prefix with any vocabulary entry.
    assert _vocab().encode("zzz") == [UNK_ID]


def test_unknown_fallback_is_per_word_not_per_query():
    assert _vocab().encode("zzz dashboard") == [UNK_ID, _ID["dashboard"]]


def test_word_longer_than_the_guard_is_unknown_without_scanning():
    assert _vocab().encode("a" * (MAX_CHARS_PER_WORD + 1)) == [UNK_ID]


def test_empty_and_punctuation_only_text_encode_to_nothing():
    assert _vocab().encode("") == []
    assert _vocab().encode("!!! ???") == []


def test_normalization_matches_the_shared_text_pipeline():
    # normalize_text lowercases, strips accents, and drops punctuation.
    assert _vocab().encode("Dashboard!") == _vocab().encode("dashboard")


def test_from_file_reads_one_token_per_line_in_id_order(tmp_path: Path):
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(_TOKENS) + "\n", encoding="utf-8")

    vocabulary = WordPieceVocabulary.from_file(path)

    assert vocabulary.size == len(_TOKENS)
    assert vocabulary.encode("standup") == [_ID["stand"], _ID["##up"]]


def test_from_file_rejects_a_vocabulary_without_the_required_specials(
    tmp_path: Path,
):
    path = tmp_path / "vocab.txt"
    path.write_text("the\npost\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[PAD\]"):
        WordPieceVocabulary.from_file(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_wordpiece.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.model.wordpiece'`.

- [ ] **Step 3: Implement the tokenizer**

Create `src/model/wordpiece.py`:

```python
"""Dependency-free WordPiece tokenization over a BERT-style vocabulary.

This module deliberately imports neither torch nor transformers. The intent
model reads pretrained wordpieces at serving time, and keeping that path free of
heavy dependencies is what lets the unit-test CI job — which installs neither —
actually exercise it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.internal.document_index.text import normalize_text

PAD_ID = 0
UNK_ID = 100
# HuggingFace's `max_input_chars_per_word`: a longer run of characters is
# treated as unknown rather than scanned.
MAX_CHARS_PER_WORD = 100

_PAD_TOKEN = "[PAD]"
_UNK_TOKEN = "[UNK]"
_CONTINUATION = "##"


class WordPieceVocabulary:
    """Greedy longest-match-first WordPiece over an id-ordered vocabulary."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self._token_to_id = token_to_id

    @classmethod
    def from_tokens(cls, tokens: Sequence[str]) -> "WordPieceVocabulary":
        """Build from tokens whose position in the sequence is their id."""
        token_to_id = {token: index for index, token in enumerate(tokens)}
        for token, expected in ((_PAD_TOKEN, PAD_ID), (_UNK_TOKEN, UNK_ID)):
            if token_to_id.get(token) != expected:
                raise ValueError(
                    f"Vocabulary must place {token} at id {expected}; found "
                    f"{token_to_id.get(token)}"
                )
        return cls(token_to_id)

    @classmethod
    def from_file(cls, path: Path) -> "WordPieceVocabulary":
        """Read a vocab.txt of one token per line, ordered by id."""
        tokens = path.read_text(encoding="utf-8").split("\n")
        while tokens and not tokens[-1]:
            tokens.pop()
        if not tokens:
            raise ValueError(f"WordPiece vocabulary file is empty: {path}")
        return cls.from_tokens(tokens)

    @property
    def size(self) -> int:
        return len(self._token_to_id)

    def encode(self, text: str) -> list[int]:
        """Normalize, split on whitespace, and encode each word."""
        ids: list[int] = []
        for word in normalize_text(text).split():
            ids.extend(self._encode_word(word))
        return ids

    def _encode_word(self, word: str) -> list[int]:
        if len(word) > MAX_CHARS_PER_WORD:
            return [UNK_ID]

        pieces: list[int] = []
        start = 0
        while start < len(word):
            end = len(word)
            match: int | None = None
            while start < end:
                candidate = word[start:end]
                if start > 0:
                    candidate = _CONTINUATION + candidate
                identifier = self._token_to_id.get(candidate)
                if identifier is not None:
                    match = identifier
                    break
                end -= 1
            if match is None:
                # No prefix of the remainder is in the vocabulary: the whole
                # word is unknown, not the partial match collected so far.
                return [UNK_ID]
            pieces.append(match)
            start = end
        return pieces
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_wordpiece.py -v`
Expected: PASS, all 11.

- [ ] **Step 5: Prove the module is importable without torch**

Run:

```bash
python - <<'PY'
import sys

class _Block:
    def find_module(self, name, path=None):
        if name.split(".")[0] in {"torch", "transformers", "sentence_transformers"}:
            raise ModuleNotFoundError(f"blocked: {name}")
        return None

sys.meta_path.insert(0, _Block())
from src.model.wordpiece import WordPieceVocabulary
print("imported without torch:", WordPieceVocabulary)
PY
```
Expected: prints the class. A `ModuleNotFoundError` means something in the import chain pulls torch — fix it before continuing, because CI will hit the same wall.

- [ ] **Step 6: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/wordpiece.py tests/unit/test_wordpiece.py
git commit -m "feat(intent): add a dependency-free wordpiece tokenizer"
```

---

### Task 2: Pretrained bundle — extract, write, read, validate

**Files:**
- Create: `src/model/intent_pretrained.py`
- Test: `tests/unit/test_intent_pretrained.py` (new)

**Interfaces:**
- Consumes: `WordPieceVocabulary.from_file` and `from_tokens` from Task 1.
- Produces:
  - `@dataclass(frozen=True) PretrainedBundle` with fields `vocabulary: WordPieceVocabulary`, `embeddings: numpy.ndarray` (shape `(size, dim)`, dtype `float16`), and properties `size: int`, `dim: int`.
  - `write_pretrained_bundle(directory: Path, *, tokens: Sequence[str], embeddings: numpy.ndarray) -> None`
  - `load_pretrained_bundle(directory: Path) -> PretrainedBundle`
  - `extract_pretrained_bundle(model_name: str, directory: Path) -> None` (imports `sentence_transformers` lazily, inside the function)
  - `VOCAB_FILENAME = "vocab.txt"`, `EMBEDDINGS_FILENAME = "embeddings.fp16.npy"`, `DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_intent_pretrained.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from src.model.intent_pretrained import (
    EMBEDDINGS_FILENAME,
    VOCAB_FILENAME,
    load_pretrained_bundle,
    write_pretrained_bundle,
)

_TOKENS = ["[PAD]"] + [f"[unused{i}]" for i in range(99)] + ["[UNK]", "the", "dashboard"]


def _write(directory: Path, *, tokens=None, rows: int | None = None, dim: int = 4):
    tokens = list(_TOKENS if tokens is None else tokens)
    matrix = np.arange(
        (rows if rows is not None else len(tokens)) * dim, dtype=np.float16
    ).reshape(-1, dim)
    write_pretrained_bundle(directory, tokens=tokens, embeddings=matrix)
    return matrix


def test_round_trip_preserves_vocabulary_and_matrix(tmp_path: Path):
    written = _write(tmp_path)

    bundle = load_pretrained_bundle(tmp_path)

    assert bundle.size == len(_TOKENS)
    assert bundle.dim == 4
    assert bundle.embeddings.dtype == np.float16
    np.testing.assert_array_equal(bundle.embeddings, written)
    assert bundle.vocabulary.encode("dashboard") == [_TOKENS.index("dashboard")]


def test_written_vocabulary_is_one_token_per_line_in_id_order(tmp_path: Path):
    _write(tmp_path)

    lines = (tmp_path / VOCAB_FILENAME).read_text(encoding="utf-8").splitlines()

    assert lines == _TOKENS
    assert lines[0] == "[PAD]"


def test_load_rejects_a_matrix_whose_rows_disagree_with_the_vocabulary(
    tmp_path: Path,
):
    """A shifted vocabulary gives every token the wrong vector, silently."""
    _write(tmp_path, rows=len(_TOKENS) - 1)

    with pytest.raises(ValueError, match="rows"):
        load_pretrained_bundle(tmp_path)


def test_load_reports_a_missing_bundle_by_name(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match=EMBEDDINGS_FILENAME):
        load_pretrained_bundle(tmp_path)


def test_write_rejects_a_non_float16_matrix(tmp_path: Path):
    with pytest.raises(ValueError, match="float16"):
        write_pretrained_bundle(
            tmp_path,
            tokens=_TOKENS,
            embeddings=np.zeros((len(_TOKENS), 4), dtype=np.float32),
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_pretrained.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.model.intent_pretrained'`.

- [ ] **Step 3: Implement the bundle**

Create `src/model/intent_pretrained.py`:

```python
"""The frozen pretrained embedding bundle the intent model reads.

Extraction needs sentence-transformers; loading needs only numpy. The split
matters: extraction runs once offline, while loading runs wherever the model is
trained or served.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .wordpiece import WordPieceVocabulary

VOCAB_FILENAME = "vocab.txt"
EMBEDDINGS_FILENAME = "embeddings.fp16.npy"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class PretrainedBundle:
    """A wordpiece vocabulary and the frozen matrix its ids index into."""

    vocabulary: WordPieceVocabulary
    embeddings: np.ndarray

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


def write_pretrained_bundle(
    directory: Path, *, tokens: Sequence[str], embeddings: np.ndarray
) -> None:
    """Write vocab.txt and the fp16 matrix, creating the directory if needed."""
    if embeddings.dtype != np.float16:
        raise ValueError(
            f"Pretrained embeddings must be float16, got {embeddings.dtype}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / VOCAB_FILENAME).write_text(
        "\n".join(tokens) + "\n", encoding="utf-8"
    )
    np.save(directory / EMBEDDINGS_FILENAME, embeddings)


def load_pretrained_bundle(directory: Path) -> PretrainedBundle:
    """Load and validate a bundle written by ``write_pretrained_bundle``."""
    vocabulary_path = directory / VOCAB_FILENAME
    embeddings_path = directory / EMBEDDINGS_FILENAME
    for path in (vocabulary_path, embeddings_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Pretrained bundle is missing {path.name}: {directory}. Run "
                "`python -m src.model.intent_training embeddings` to create it."
            )

    vocabulary = WordPieceVocabulary.from_file(vocabulary_path)
    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2 or embeddings.shape[0] != vocabulary.size:
        raise ValueError(
            "Pretrained matrix rows must equal the vocabulary size: "
            f"{embeddings.shape} rows against {vocabulary.size} tokens"
        )
    return PretrainedBundle(vocabulary=vocabulary, embeddings=embeddings)


def extract_pretrained_bundle(
    model_name: str = DEFAULT_MODEL, directory: Path = Path("data/intent_pretrained")
) -> None:
    """Pull the tokenizer vocabulary and input embedding matrix from a model.

    Only the embedding table is taken. The transformer itself never runs, at
    training time or at serving time.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    vocabulary = model.tokenizer.get_vocab()
    tokens = [token for token, _ in sorted(vocabulary.items(), key=lambda kv: kv[1])]
    weights = model[0].auto_model.embeddings.word_embeddings.weight
    embeddings = weights.detach().cpu().numpy().astype(np.float16)
    if embeddings.shape[0] != len(tokens):
        raise ValueError(
            "Model vocabulary and embedding matrix disagree: "
            f"{len(tokens)} tokens against {embeddings.shape[0]} rows"
        )
    write_pretrained_bundle(directory, tokens=tokens, embeddings=embeddings)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_intent_pretrained.py -v`
Expected: PASS, all 5.

- [ ] **Step 5: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_pretrained.py tests/unit/test_intent_pretrained.py
git commit -m "feat(intent): add the frozen pretrained embedding bundle"
```

---

### Task 3: Pipeline reads wordpieces against the frozen matrix

**Files:**
- Modify: `src/model/intent_classifier.py`
- Test: `tests/unit/test_intent_classifier.py`

**Interfaces:**
- Consumes: `PretrainedBundle`, `load_pretrained_bundle` from Task 2; `UNK_ID`, `PAD_ID` from Task 1.
- Produces: `IntentPipeline(bundle: PretrainedBundle, *, hidden_dim: int = 256)` — `vocab_size` and `embedding_dim` are gone, derived from the bundle. `IntentPipeline.load(path)` still returns a pipeline. Checkpoints are version `4`. `_IntentClassifier(embedding_matrix, hidden_dim, num_classes)` replaces the old four-argument form.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_intent_classifier.py`:

```python
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
```

Delete the tests that the removed configuration made meaningless, and update the ones that construct a pipeline:

- Delete `test_training_rejects_vocabulary_that_exceeds_embedding_table` — there is no built vocabulary to exceed the table.
- Delete `test_min_freq_two_trains_the_unknown_embedding` — `min_freq` is gone.
- Delete `test_unknown_token_changes_the_pooled_vector` and `test_encode_maps_an_all_unknown_query_to_the_unknown_id` — superseded by `test_unseen_word_is_read_as_wordpieces_not_dropped` and `test_empty_query_encodes_to_a_single_unknown`.
- Delete `test_save_writes_version_three_checkpoint_contract` and `test_load_rejects_version_two_checkpoint_with_retraining_message` — replaced by the version-4 contract test and the parametrized rejection above.
- **Keep** `test_vocabulary_build_and_encode_support_sequence_training`. It exercises `document_index`'s shared `Vocabulary` directly, which this plan does not touch.
- Update every remaining `IntentPipeline(vocab_size=..., embedding_dim=..., hidden_dim=...)` call to `IntentPipeline(_bundle(), hidden_dim=...)`, and drop `min_freq=` from every `train(...)` call.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_classifier.py -v`
Expected: FAIL — `TypeError: IntentPipeline.__init__() got an unexpected keyword argument 'vocab_size'` and `AttributeError: '_bundle' is not defined` style errors.

- [ ] **Step 3: Rebuild `_IntentClassifier` around a frozen matrix**

In `src/model/intent_classifier.py`, replace the `_IntentClassifier.__init__` signature and network construction:

```python
class _IntentClassifier:
    def __init__(
        self,
        embedding_matrix: "np.ndarray",
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        import torch
        import torch.nn as nn

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        embedding_dim = weights.shape[1]

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Frozen: with a few hundred training examples, fine-tuning
                # these would overwrite the pretrained semantics that make
                # unseen words readable in the first place.
                self.embedding = nn.Embedding.from_pretrained(
                    weights, freeze=True, padding_idx=PADDING_ID
                )
                self.fc1 = nn.Linear(embedding_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
                self.drop = nn.Dropout(0.3)

            def forward(self, ids: "torch.Tensor") -> "torch.Tensor":
                import torch.nn.functional as F

                mask = ids.ne(PADDING_ID).unsqueeze(-1)
                embedded = self.embedding(ids)
                x = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                x = self.drop(F.relu(self.fc1(x)))
                x = self.drop(F.relu(self.fc2(x)))
                return self.fc3(x)

        self._net = _Net().to(self._device)
```

Keep `_pad_sequences`, `train_batched`, and `predict_batch` as they are. Change only the optimizer construction inside `train_batched`, so the frozen embedding is not handed to Adam:

```python
        optimizer = self._torch.optim.Adam(
            (p for p in self._net.parameters() if p.requires_grad), lr=lr
        )
```

Add the numpy import for the type annotation at module top: `import numpy as np` — it is already an indirect dependency through scikit-learn and is present in the unit-test job.

- [ ] **Step 4: Rebuild `IntentPipeline` around the bundle**

Replace the constants and `IntentPipeline.__init__`/`train`/`predict` in `src/model/intent_classifier.py`:

```python
PADDING_ID = 0
UNKNOWN_ID = 100


class IntentPipeline:
    def __init__(self, bundle: "PretrainedBundle", *, hidden_dim: int = 256) -> None:
        self._bundle = bundle
        self._hidden_dim = hidden_dim
        self._model = self._new_model()
        self._label_to_id = {label: i for i, label in enumerate(INTENT_LABELS)}
        self.is_trained = False

    def _new_model(self) -> _IntentClassifier:
        return _IntentClassifier(
            self._bundle.embeddings, self._hidden_dim, len(INTENT_LABELS)
        )

    def _encode_text(self, text: str) -> list[int]:
        """Encode one request as wordpiece ids.

        Reading no tokens is a fact about the input, not padding, so an empty
        result becomes a single [UNK] rather than an empty sequence.
        """
        return self._bundle.vocabulary.encode(text) or [UNKNOWN_ID]

    def train(
        self,
        data: list[tuple[list[str], str]],
        *,
        epochs: int = 10,
        lr: float = 1e-3,
        seed: int = 17,
    ) -> None:
        """Train the head on (token_list, intent_label) pairs."""
        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._model = self._new_model()
        encoded = [self._encode_text(" ".join(tokens)) for tokens, _ in data]
        labels = [self._label_to_id[label] for _, label in data]
        self._model.train_batched(encoded, labels, epochs=epochs, lr=lr)
        self.is_trained = True

    def predict(self, tokens: Sequence[str]) -> IntentPrediction:
        if not self.is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")
        return self._model.predict_batch([self._encode_text(" ".join(tokens))])[0]

    def predict_text(self, text: str) -> IntentPrediction:
        if not self.is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")
        return self._model.predict_batch([self._encode_text(text)])[0]
```

Note the deliberate change: `train`/`predict` join their token list back into text so the wordpiece tokenizer sees whole words. Callers pass `tokenize_text(...)` output today, and re-joining keeps that call-site contract intact while letting WordPiece do its own splitting.

Add the import at module top:

```python
from .intent_pretrained import PretrainedBundle, load_pretrained_bundle
```

- [ ] **Step 5: Move the checkpoint to version 4**

In `save`, replace the checkpoint dict's `version`, `preprocessing`, `vocab`, and `config` entries:

```python
        checkpoint = {
            "version": 4,
            "intent_labels": list(INTENT_LABELS),
            "preprocessing": {
                "tokenizer": "wordpiece",
                "padding_id": PADDING_ID,
                "unknown_id": UNKNOWN_ID,
                "pooling": "masked_mean",
                "embeddings": "frozen_pretrained",
            },
            "dataset_fingerprint": dataset_fingerprint,
            "promoted_min_confidence": promoted_min_confidence,
            "vocab_tokens": self._bundle.vocabulary.tokens,
            "embeddings": torch.tensor(self._bundle.embeddings),
            "model_state": self._model._net.state_dict(),
            "config": {
                "vocab_size": self._bundle.size,
                "embedding_dim": self._bundle.dim,
                "hidden_dim": self._model._net.fc1.out_features,
                "num_classes": len(INTENT_LABELS),
            },
        }
```

This needs `WordPieceVocabulary` to expose its tokens in id order. Add to `src/model/wordpiece.py`:

```python
    @property
    def tokens(self) -> list[str]:
        """Vocabulary tokens in id order, as written to vocab.txt."""
        return [
            token
            for token, _ in sorted(self._token_to_id.items(), key=lambda kv: kv[1])
        ]
```

In `load`, replace the version gate and the vocabulary reconstruction:

```python
        version = checkpoint.get("version")
        if version in (1, 2, 3):
            raise ValueError(
                f"Checkpoint version {version} predates pretrained wordpiece "
                "embeddings; retrain the intent model before loading it."
            )
        if version != 4:
            raise ValueError(f"Unsupported checkpoint version: {version}")
```

```python
        expected_preprocessing = {
            "tokenizer": "wordpiece",
            "padding_id": PADDING_ID,
            "unknown_id": UNKNOWN_ID,
            "pooling": "masked_mean",
            "embeddings": "frozen_pretrained",
        }
```

and build the pipeline from the checkpoint's own bundle:

```python
        import numpy as np

        from .wordpiece import WordPieceVocabulary

        cfg = checkpoint["config"]
        cls._validate_checkpoint_dimensions(cfg, checkpoint["model_state"])
        bundle = PretrainedBundle(
            vocabulary=WordPieceVocabulary.from_tokens(checkpoint["vocab_tokens"]),
            embeddings=checkpoint["embeddings"].numpy().astype(np.float16),
        )
        pipeline = cls(bundle, hidden_dim=cfg["hidden_dim"])
        pipeline._model._net.load_state_dict(checkpoint["model_state"])
        pipeline._model._net.eval()
        pipeline.is_trained = True
```

Keep the `promoted_min_confidence` validation and assignment exactly as they are.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/unit/test_intent_classifier.py -v`
Expected: PASS. If `_validate_checkpoint_dimensions` fails on `embedding.weight`, confirm its expected shape is `(config["vocab_size"], config["embedding_dim"])` — it already is, and the bundle supplies both.

- [ ] **Step 7: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_classifier.py src/model/wordpiece.py tests/unit/test_intent_classifier.py
git commit -m "feat(intent): read wordpieces against a frozen pretrained matrix"
```

---

### Task 4: Training command wiring

**Files:**
- Modify: `src/model/intent_training.py`
- Test: `tests/unit/test_intent_training.py`

**Interfaces:**
- Consumes: `load_pretrained_bundle`, `extract_pretrained_bundle`, `DEFAULT_MODEL` from Task 2; `IntentPipeline(bundle, hidden_dim=...)` from Task 3.
- Produces: `IntentTrainingConfig.pretrained_path: Path` (required, no default); `min_freq`, `vocab_size`, and `embedding_dim` removed from the config, the CLI, and `_hyperparameters`. New CLI subcommand `embeddings --model <name> --output <dir>`; `train` gains `--pretrained <dir>`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_intent_training.py`:

```python
def _write_test_bundle(directory: Path, dim: int = 8) -> Path:
    """A bundle covering the fixture vocabulary, for fast offline training."""
    import numpy as np

    from src.model.intent_pretrained import write_pretrained_bundle

    tokens = (
        ["[PAD]"]
        + [f"[unused{index}]" for index in range(99)]
        + ["[UNK]"]
        + [
            "find",
            "explain",
            "send",
            "the",
            "a",
            "how",
            "to",
            "dense",
            "retrieval",
            "works",
            "email",
            "ticket",
            "##s",
            "##ing",
        ]
    )
    rng = np.random.default_rng(17)
    write_pretrained_bundle(
        directory,
        tokens=tokens,
        embeddings=rng.normal(size=(len(tokens), dim)).astype(np.float16),
    )
    return directory


def test_embeddings_cli_writes_a_loadable_bundle(tmp_path, monkeypatch):
    """The extraction command is the only place a model is loaded."""
    from src.model import intent_pretrained

    captured = {}

    def fake_extract(model_name, directory):
        captured["model_name"] = model_name
        _write_test_bundle(Path(directory))

    monkeypatch.setattr(intent_training, "extract_pretrained_bundle", fake_extract)
    output = tmp_path / "bundle"

    exit_code = intent_training.main(
        ["embeddings", "--output", str(output), "--model", "test/model"]
    )

    assert exit_code == 0
    assert captured["model_name"] == "test/model"
    assert intent_pretrained.load_pretrained_bundle(output).size > 100


def test_training_requires_a_pretrained_bundle(tmp_path, capsys):
    exit_code = intent_training.main(
        [
            "train",
            "--examples",
            str(FIXTURES / "intent_examples.json"),
            "--baseline",
            str(FIXTURES / "baseline_predictions.json"),
            "--pretrained",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1
    assert "embeddings" in capsys.readouterr().err
```

Then update every existing `IntentTrainingConfig(...)` construction in this file to pass `pretrained_path=_write_test_bundle(tmp_path / "bundle")`, and drop `embedding_dim=`/`hidden_dim=`-adjacent arguments that no longer exist (`embedding_dim` is gone; `hidden_dim` stays).

The two pinned-bar tests (`test_frame_trained_model_holds_the_realistic_accuracy_bar`, `test_out_of_scope_requests_score_below_in_scope_requests`) need the **real** bundle, not the fixture one. Change `_pipeline_trained_on_committed_examples` to:

```python
@functools.lru_cache(maxsize=1)
def _pipeline_trained_on_committed_examples():
    from src.model.intent_classifier import IntentPipeline
    from src.model.intent_pretrained import load_pretrained_bundle

    bundle_path = DATA / "intent_pretrained"
    if not (bundle_path / "vocab.txt").exists():
        pytest.skip(
            "run `python -m src.model.intent_training embeddings "
            f"--output {bundle_path}` to measure the pinned bars"
        )
    split = split_intent_examples(
        load_intent_examples(DATA / "intent_examples.json"), seed=17
    )
    pipeline = IntentPipeline(load_pretrained_bundle(bundle_path), hidden_dim=256)
    pipeline.train(
        [(tokenize_text(example.text), example.label) for example in split.train],
        epochs=_PINNED_EPOCHS,
        lr=_PINNED_LR,
        seed=17,
    )
    return pipeline
```

`_PINNED_EPOCHS` and `_PINNED_LR` are set in Task 5 from the sweep. Until then, use `300` and `1e-3`.

The `pytest.skip` is deliberate: the bundle is a regenerable local artifact under gitignored `data/`, so a fresh clone must not fail this test — it must say what to run.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_training.py -v`
Expected: FAIL — `TypeError: IntentTrainingConfig.__init__() got an unexpected keyword argument 'pretrained_path'`.

- [ ] **Step 3: Update the config and the training run**

In `src/model/intent_training.py`:

Extend the imports:

```python
from .intent_pretrained import (
    DEFAULT_MODEL,
    extract_pretrained_bundle,
    load_pretrained_bundle,
)
```

In `IntentTrainingConfig`, delete `min_freq`, `vocab_size`, and `embedding_dim`, and add the required bundle path as the fourth positional field, beside `baseline_path`:

```python
    pretrained_path: Path = Path("data/intent_pretrained")
```

In `run_intent_training`, replace the pipeline construction and training call:

```python
    bundle = load_pretrained_bundle(Path(config.pretrained_path))
    pipeline = IntentPipeline(bundle, hidden_dim=config.hidden_dim)
    training_data = [
        (tokenize_text(example.text), example.label) for example in split.train
    ]
    pipeline.train(
        training_data,
        epochs=config.epochs,
        lr=config.lr,
        seed=config.seed,
    )
```

In `_hyperparameters`, delete the `min_freq`, `vocab_size`, and `embedding_dim` entries and add:

```python
        "pretrained": str(config.pretrained_path),
```

In `_validate_training_config`, remove `min_freq`, `vocab_size`, and `embedding_dim` from `integer_fields`, leaving `epochs` and `hidden_dim`.

In `train_intent_classifier` (the small utility), replace its `min_freq`/`vocab_size`/`embedding_dim` parameters with `pretrained_path: Path` and build the pipeline the same way. If no caller outside tests uses it, note that in the commit message rather than deleting it — it is a public-looking helper.

- [ ] **Step 4: Add the `embeddings` subcommand**

In `_build_parser`:

```python
    embeddings = subparsers.add_parser(
        "embeddings", help="extract the frozen pretrained wordpiece bundle"
    )
    embeddings.add_argument("--model", default=DEFAULT_MODEL)
    embeddings.add_argument(
        "--output", required=True, type=Path, help="bundle directory to write"
    )
```

and on the `train` parser, delete `--min-freq`, `--vocab-size`, and `--embedding-dim`, then add:

```python
    train.add_argument(
        "--pretrained", type=Path, default=Path("data/intent_pretrained")
    )
```

In `main`, add the branch before the `train` fall-through:

```python
        if args.command == "embeddings":
            extract_pretrained_bundle(args.model, args.output)
            return 0
```

and in the `IntentTrainingConfig(...)` construction, delete `min_freq=`, `vocab_size=`, `embedding_dim=` and add `pretrained_path=args.pretrained,`.

- [ ] **Step 5: Run the intent suite**

Run: `pytest tests/unit/test_intent_training.py tests/unit/test_intent_classifier.py tests/unit/test_intent_data.py tests/unit/test_intent_evaluation.py tests/unit/test_wordpiece.py tests/unit/test_intent_pretrained.py -v`
Expected: PASS, with the two pinned-bar tests skipping until the real bundle exists.

- [ ] **Step 6: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_training.py tests/unit/test_intent_training.py
git commit -m "feat(intent): train against the pretrained bundle, drop vocabulary knobs"
```

---

### Task 5: Extract the real bundle, measure, decide

**Files:**
- Test: `tests/unit/test_intent_training.py`, `tests/unit/test_wordpiece_parity.py` (new)
- Modify: `docs/training-and-evaluation.md`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: `_PINNED_EPOCHS`, `_PINNED_LR`, `_REALISTIC_ACCURACY_FLOOR`, `_OUT_OF_SCOPE_MARGIN_FLOOR` in `tests/unit/test_intent_training.py`; the parity and no-unknown tests; operator documentation.

- [ ] **Step 1: Extract the real bundle**

```bash
python -m src.model.intent_training embeddings --output data/intent_pretrained
ls -la data/intent_pretrained/
python -c "
from pathlib import Path
from src.model.intent_pretrained import load_pretrained_bundle
b = load_pretrained_bundle(Path('data/intent_pretrained'))
print('size', b.size, 'dim', b.dim, 'dtype', b.embeddings.dtype)
"
```
Expected: `vocab.txt` (~230KB), `embeddings.fp16.npy` (~23MB), `size 30522 dim 384 dtype float16`.

- [ ] **Step 2: Write the parity and no-unknown tests**

Create `tests/unit/test_wordpiece_parity.py`:

```python
"""Parity between our WordPiece and HuggingFace's, over the real corpora.

Skipped where transformers is unavailable, which includes the unit-test CI job.
The tokenizer's own behavior is covered dependency-free in test_wordpiece.py.
"""

import json
from pathlib import Path

import pytest

from src.internal.document_index.text import normalize_text
from src.model.intent_pretrained import DEFAULT_MODEL, load_pretrained_bundle
from src.model.wordpiece import UNK_ID

DATA = Path(__file__).resolve().parents[2] / "data"


def _bundle():
    if not (DATA / "intent_pretrained" / "vocab.txt").exists():
        pytest.skip(
            "run `python -m src.model.intent_training embeddings "
            f"--output {DATA / 'intent_pretrained'}`"
        )
    return load_pretrained_bundle(DATA / "intent_pretrained")


def _corpus_texts() -> list[str]:
    texts = [
        record["text"]
        for record in json.loads(
            (DATA / "intent_eval_queries.json").read_text(encoding="utf-8")
        )
    ]
    texts += [
        record["text"]
        for record in json.loads(
            (DATA / "intent_out_of_scope.json").read_text(encoding="utf-8")
        )
    ]
    texts += [
        record["text"]
        for record in json.loads(
            (DATA / "intent_examples.json").read_text(encoding="utf-8")
        )
    ]
    return texts


def test_our_wordpiece_matches_huggingface_on_normalized_text():
    transformers = pytest.importorskip("transformers")
    bundle = _bundle()
    reference = transformers.AutoTokenizer.from_pretrained(DEFAULT_MODEL)

    for text in _corpus_texts():
        normalized = normalize_text(text)
        expected = reference(normalized, add_special_tokens=False)["input_ids"]
        assert bundle.vocabulary.encode(text) == expected, text


def test_every_evaluation_query_and_probe_decomposes_without_unknowns():
    """The direct refutation of the 47%-unread measurement from #509."""
    bundle = _bundle()

    for text in _corpus_texts():
        ids = bundle.vocabulary.encode(text)
        assert ids, text
        assert UNK_ID not in ids, text
```

Run: `pytest tests/unit/test_wordpiece_parity.py -v`

If parity fails, the difference will be punctuation or accent handling. Do **not** loosen the assertion — fix `_encode_word` or the normalization contract until our ids match the reference on normalized text, and record the divergence you found in the commit message.

- [ ] **Step 3: Sweep the hyperparameters and measure**

```bash
python - <<'PY'
from pathlib import Path

from src.internal.document_index.text import tokenize_text
from src.model.intent_classifier import IntentPipeline
from src.model.intent_data import (
    load_intent_eval_queries,
    load_intent_examples,
    load_out_of_scope_probes,
    split_intent_examples,
)
from src.model.intent_pretrained import load_pretrained_bundle

bundle = load_pretrained_bundle(Path("data/intent_pretrained"))
split = split_intent_examples(
    load_intent_examples(Path("data/intent_examples.json")), seed=17
)
training = [(tokenize_text(e.text), e.label) for e in split.train]
queries = load_intent_eval_queries(Path("data/intent_eval_queries.json"))
probes = load_out_of_scope_probes(Path("data/intent_out_of_scope.json"))

print(f"{'epochs':>7} {'lr':>7} {'acc':>6} {'in':>6} {'oos':>6} {'margin':>8}")
for lr in (1e-3, 3e-3):
    for epochs in (100, 300, 800):
        p = IntentPipeline(bundle, hidden_dim=256)
        p.train(training, epochs=epochs, lr=lr, seed=17)
        preds = [p.predict_text(q.text) for q in queries]
        acc = sum(pr.intent == q.label for pr, q in zip(preds, queries)) / len(queries)
        ins = sum(pr.confidence for pr in preds) / len(preds)
        oos = [p.predict_text(t).confidence for _, t in probes]
        moos = sum(oos) / len(oos)
        print(f"{epochs:>7} {lr:>7.4f} {acc:>6.3f} {ins:>6.3f} {moos:>6.3f} {ins-moos:>+8.3f}")
PY
```

Pick the `(epochs, lr)` with the best realistic accuracy, breaking ties toward the larger margin. Then measure per-query latency with that configuration:

```bash
python - <<'PY'
import statistics, time
from pathlib import Path

from src.internal.document_index.text import tokenize_text
from src.model.intent_classifier import IntentPipeline
from src.model.intent_data import load_intent_examples, split_intent_examples
from src.model.intent_pretrained import load_pretrained_bundle

bundle = load_pretrained_bundle(Path("data/intent_pretrained"))
split = split_intent_examples(load_intent_examples(Path("data/intent_examples.json")), seed=17)
p = IntentPipeline(bundle, hidden_dim=256)
p.train([(tokenize_text(e.text), e.label) for e in split.train], epochs=300, lr=1e-3, seed=17)
q = "where did we land on the index rebuild last week"
for _ in range(10):
    p.predict_text(q)
ts = []
for _ in range(200):
    s = time.perf_counter(); p.predict_text(q); ts.append((time.perf_counter() - s) * 1000)
print(f"p50 {statistics.median(ts):.3f}ms  p95 {sorted(ts)[190]:.3f}ms")
PY
```

- [ ] **Step 4: Apply the decision rule**

The spec fixes this in advance. Compare the best realistic accuracy against:

| realistic accuracy | verdict |
|---|---|
| ≥ 0.75 | worth promoting; continue to Step 5 and say so in the PR |
| 0.60 – 0.75 | real improvement; continue to Step 5, artifact stays dark |
| ≤ 0.60 | the representation change failed |

**If accuracy is ≤ 0.60, stop and report** — the numbers, the per-label metrics, the margin, and the latency. Do not tune further. The spec's stated fallback is the full MiniLM encoder behind the same tokenizer, and that is a new spec, not more sweeping here.

Also confirm the other criteria: p95 routing latency under 2ms, out-of-scope margin still positive, and 100% token coverage (Step 2's no-unknown test).

- [ ] **Step 5: Pin the measured values**

In `tests/unit/test_intent_training.py`, set the four constants from Step 3's chosen run:

```python
# Measured on the first pretrained-wordpiece run (seed 17). Raise the floors
# when a run beats them; never lower one without recording why.
_PINNED_EPOCHS = 300     # <- replace with the swept value
_PINNED_LR = 1e-3        # <- replace with the swept value
_REALISTIC_ACCURACY_FLOOR = 0.00   # <- replace with measured, minus ~0.02
_OUT_OF_SCOPE_MARGIN_FLOOR = 0.00  # <- replace with measured, minus ~0.02
```

Run: `pytest tests/unit/test_intent_training.py -k "realistic_accuracy_bar or out_of_scope_requests" -v`
Expected: PASS, no longer skipped.

- [ ] **Step 6: Prove the serving path took on no new dependency**

The spec requires this, and a stray `import transformers` at module scope in the
routing path would only surface in production or CI:

```bash
python - <<'PY'
import sys

class _Block:
    def find_module(self, name, path=None):
        if name.split(".")[0] in {"transformers", "sentence_transformers"}:
            raise ModuleNotFoundError(f"blocked: {name}")
        return None

sys.meta_path.insert(0, _Block())
import src.internal.servers.web.ml_intent as ml_intent
from src.model.intent_classifier import IntentPipeline
print("serving path imports without transformers:", ml_intent.predict_route, IntentPipeline)
PY
```
Expected: prints both. A `ModuleNotFoundError` means extraction leaked out of
`intent_pretrained.extract_pretrained_bundle`, where the `sentence_transformers`
import must stay function-local.

- [ ] **Step 7: Run the whole suite**

Run: `pytest` (~3-4 minutes)
Expected: PASS. Failures here are almost certainly stale `IntentPipeline(vocab_size=...)` constructions or `min_freq=` arguments in tests outside the intent files — `grep -rn "vocab_size\|min_freq" tests src | grep -i intent` finds them.

- [ ] **Step 8: Update the operator documentation**

In `docs/training-and-evaluation.md`, in the intent-model section:

- Add the extraction command as a new first step, before the `baseline` command:

```bash
python -m src.model.intent_training embeddings --output data/intent_pretrained
```

- Add a paragraph after it:

> The intent model reads requests as **pretrained wordpieces**, not as words from a vocabulary built out of its own training data. `embeddings` extracts MiniLM's tokenizer vocabulary and input embedding matrix once into `data/intent_pretrained/` (a 230KB `vocab.txt` and a 23MB fp16 matrix); the transformer itself never runs, at training time or serving time. This is what removes out-of-vocabulary entirely: an unseen word decomposes into known subwords — *postmortem* becomes `post ##mo ##rte ##m` — instead of being deleted by the padding mask. The previous word-level model read only 47% of the tokens in the evaluation set. The bundle lives under gitignored `data/`, so it is regenerable rather than committed, and a checkpoint carries its own copy so serving needs no separate file.

- Replace the paragraph describing `--epochs` and `--min-freq` defaults: `--min-freq` no longer exists, and neither do `--vocab-size` or `--embedding-dim` — the dimensions come from the bundle at 30522 × 384. Keep the `--epochs` explanation of full-batch training, updating the default to the swept value.
- Update the measured-results paragraph with the new realistic accuracy, margin, and latency, replacing #509's `0.567` / `+0.071` figures and the "not promotable" framing with whatever the decision rule in Step 4 returned.
- Note the checkpoint is now ~23MB and version 4, and that versions 1-3 are rejected.

- [ ] **Step 9: Commit and open the PR**

```bash
ruff check . --fix && ruff format .
git branch --show-current   # must be feat/intent-pretrained-wordpiece
git add docs/training-and-evaluation.md tests/unit/test_intent_training.py tests/unit/test_wordpiece_parity.py
git commit -m "feat(intent): pin the pretrained-wordpiece bars and document the workflow"
git push -u origin feat/intent-pretrained-wordpiece
gh pr create --title "feat(intent): read pretrained wordpieces so no query word goes unread" --body "..."
```

The PR body must state: token coverage before (47%) and after; realistic accuracy before (`0.567`) and after; the out-of-scope margin; p50/p95 routing latency; the swept `epochs`/`lr`; the checkpoint version bump and artifact size; which verdict of the decision rule was reached; and **"Please review before merging."** Link the spec and this plan.

---

## Verification summary

| Spec requirement | Task |
|---|---|
| Dependency-free WordPiece over MiniLM's vocabulary | 1 |
| Greedy longest-match, `##` continuations, per-word `[UNK]`, 100-char guard | 1 |
| Tokenizer tests run in CI without torch | 1 (Step 5 proves the import) |
| Offline extraction command writing vocab + fp16 matrix | 2 |
| Vocabulary/matrix size disagreement rejected, torch-free | 2 |
| Missing bundle names the command that creates it | 2 |
| Frozen pretrained embeddings, never a trained parameter | 3 |
| `[PAD]`=0 keeps the padding contract; `[UNK]`=100 | 3 |
| Empty decomposition becomes a single `[UNK]` | 3 |
| Checkpoint v4, self-contained; v1/v2/v3 rejected | 3 |
| `min_freq` / `vocab_size` / `embedding_dim` deleted | 4 |
| `embeddings` subcommand and `--pretrained` flag | 4 |
| Parity with HuggingFace on normalized text | 5 |
| No `[UNK]` across eval set, probes, and dataset | 5 |
| `epochs`/`lr` sweep, recorded | 5 |
| Decision rule applied, with a hard stop at ≤ 0.60 | 5 |
| p95 latency under 2ms | 5 |
| No new serving dependency | 1 (Step 5), reconfirmed in 5 |
| Runtime cascade, dispatchers, frontend unchanged | all — no file outside `src/model/`, `data/`, `docs/`, `tests/` is touched |

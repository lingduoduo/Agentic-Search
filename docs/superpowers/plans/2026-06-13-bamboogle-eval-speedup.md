# Bamboogle Eval Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `bin/run_bamboogle_eval.sh --limit 125` dramatically faster via local dataset caching, parallel question evaluation, and resume-on-interrupt support.

**Architecture:** All changes are confined to `src/training/eval/bamboogle.py` (logic) and `examples/run_bamboogle_eval.py` + `bin/run_bamboogle_eval.sh` (CLI exposure). The eval loop's `agent.invoke()` calls are synchronous wrappers over `asyncio.run()`, making `ThreadPoolExecutor` the right concurrency primitive — each thread gets its own event loop. Order is preserved via `executor.map()`.

**Tech Stack:** Python stdlib `concurrent.futures.ThreadPoolExecutor`, `pathlib.Path`, `requests`, `tqdm`

---

## File Map

| File | Change |
|------|--------|
| `src/training/eval/bamboogle.py` | Add `cache_path` param to `load_bamboogle()`; add `concurrency` + `resume` params to `evaluate_bamboogle()`; add `_load_completed_ids()` helper |
| `examples/run_bamboogle_eval.py` | Add `--concurrency` and `--resume` flags; thread them through to `evaluate_bamboogle()` |
| `bin/run_bamboogle_eval.sh` | Add `--concurrency` and `--resume` arg parsing; pass them to `run_bamboogle_eval.py` |
| `tests/unit/test_bamboogle_eval.py` | Update two existing `load_bamboogle` tests to pass `cache_path=None`; add tests for caching, concurrency, and resume |

---

## Task 1: Local Dataset Caching

**Files:**
- Modify: `src/training/eval/bamboogle.py`
- Modify: `tests/unit/test_bamboogle_eval.py`

The current `load_bamboogle()` unconditionally calls `requests.get` on every run. Add a `cache_path` parameter (default `~/.cache/agentic_search/bamboogle_test.jsonl`) that reads from disk if the file exists and writes to disk after a successful download.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_bamboogle_eval.py` after the existing `test_load_bamboogle_limit` test:

```python
@patch("requests.get")
def test_load_bamboogle_writes_cache(mock_get, tmp_path):
    mock_get.return_value.text = _FAKE_JSONL
    mock_get.return_value.raise_for_status = MagicMock()
    cache = tmp_path / "bamboogle_test.jsonl"
    load_bamboogle(cache_path=cache)
    assert cache.exists()
    assert len(cache.read_text().splitlines()) == 5


@patch("requests.get")
def test_load_bamboogle_reads_cache(mock_get, tmp_path):
    """Second call must not hit the network when cache exists."""
    mock_get.return_value.text = _FAKE_JSONL
    mock_get.return_value.raise_for_status = MagicMock()
    cache = tmp_path / "bamboogle_test.jsonl"
    # Populate cache
    load_bamboogle(cache_path=cache)
    mock_get.reset_mock()
    # Second call — should not call requests.get
    rows = load_bamboogle(cache_path=cache)
    mock_get.assert_not_called()
    assert len(rows) == 5


@patch("requests.get")
def test_load_bamboogle_cache_none_skips_disk(mock_get):
    """cache_path=None must never touch the filesystem."""
    mock_get.return_value.text = _FAKE_JSONL
    mock_get.return_value.raise_for_status = MagicMock()
    rows = load_bamboogle(cache_path=None)
    assert len(rows) == 5
```

Also update the two existing `load_bamboogle` tests to opt out of caching so they don't attempt disk writes:

```python
# was: rows = load_bamboogle()
rows = load_bamboogle(cache_path=None)

# was: rows = load_bamboogle(limit=3)
rows = load_bamboogle(limit=3, cache_path=None)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/linghuang/Git/Agentic-Search
pytest tests/unit/test_bamboogle_eval.py::test_load_bamboogle_writes_cache \
       tests/unit/test_bamboogle_eval.py::test_load_bamboogle_reads_cache \
       tests/unit/test_bamboogle_eval.py::test_load_bamboogle_cache_none_skips_disk -v
```

Expected: `TypeError` — `load_bamboogle() got an unexpected keyword argument 'cache_path'`

- [ ] **Step 3: Implement caching in `load_bamboogle()`**

Replace the existing `load_bamboogle` function in `src/training/eval/bamboogle.py`:

```python
_DEFAULT_CACHE = Path.home() / ".cache" / "agentic_search" / "bamboogle_test.jsonl"


def load_bamboogle(
    limit: int | None = None,
    cache_path: str | Path | None = _DEFAULT_CACHE,
) -> list[dict[str, Any]]:
    """Download and parse the Bamboogle test split from HuggingFace.

    Args:
        limit: Return only the first *limit* examples.  ``None`` returns all 125.
        cache_path: Path to cache the raw JSONL.  Set to ``None`` to disable.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            rows = [
                json.loads(line)
                for line in cache_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return rows[:limit] if limit is not None else rows

    import requests

    resp = requests.get(BAMBOOGLE_URL, timeout=30)
    resp.raise_for_status()
    rows = [json.loads(line) for line in resp.text.splitlines() if line.strip()]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text, encoding="utf-8")

    return rows[:limit] if limit is not None else rows
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_bamboogle_eval.py -v
```

Expected: all existing tests still pass + 3 new caching tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/eval/bamboogle.py tests/unit/test_bamboogle_eval.py
git commit -m "feat(eval): cache bamboogle dataset locally to avoid re-downloading each run"
```

---

## Task 2: Parallel Eval Loop

**Files:**
- Modify: `src/training/eval/bamboogle.py`
- Modify: `examples/run_bamboogle_eval.py`
- Modify: `bin/run_bamboogle_eval.sh`
- Modify: `tests/unit/test_bamboogle_eval.py`

Replace the sequential `for` loop with `ThreadPoolExecutor.map()`. Each thread calls `agent.invoke()` independently. Results arrive in order, matching input order. Add a `concurrency: int = 1` parameter to `evaluate_bamboogle()` (default 1 = no change in behaviour).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_bamboogle_eval.py`:

```python
@patch("src.training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET)
def test_evaluate_parallel_same_results(mock_load, tmp_path):
    """Concurrency > 1 must produce identical results to concurrency=1."""
    serial_summary, serial_rows = evaluate_bamboogle(
        _PerfectAgent(), limit=2, output_path=None, verbose=False, concurrency=1
    )
    parallel_summary, parallel_rows = evaluate_bamboogle(
        _PerfectAgent(), limit=2, output_path=None, verbose=False, concurrency=2
    )
    assert serial_summary.exact_match == parallel_summary.exact_match
    assert serial_summary.contains_match == parallel_summary.contains_match
    assert [r.question for r in serial_rows] == [r.question for r in parallel_rows]
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/unit/test_bamboogle_eval.py::test_evaluate_parallel_same_results -v
```

Expected: `TypeError` — `evaluate_bamboogle() got an unexpected keyword argument 'concurrency'`

- [ ] **Step 3: Implement parallel eval in `evaluate_bamboogle()`**

Replace the body of `evaluate_bamboogle()` in `src/training/eval/bamboogle.py`. Add the import at the top of the file alongside the existing imports:

```python
from concurrent.futures import ThreadPoolExecutor
```

Replace the `evaluate_bamboogle` function signature and body:

```python
def evaluate_bamboogle(
    agent: Any,
    *,
    reward_fn: SearchRewardFunction | None = None,
    limit: int | None = 20,
    output_path: str | Path | None = "bamboogle_results.jsonl",
    verbose: bool = True,
    concurrency: int = 1,
) -> tuple[BamboogleSummary, list[BamboogleResult]]:
    """Run *agent* on the Bamboogle benchmark and report accuracy metrics.

    Args:
        agent: Any object with an ``invoke(state: dict) -> Any`` method.
        reward_fn: Optional shaped-reward scorer.
        limit: Number of examples to evaluate.  ``None`` evaluates all 125.
        output_path: Write per-example results as JSONL here.  ``None`` skips writing.
        verbose: Show a tqdm progress bar.
        concurrency: Number of questions to evaluate in parallel.  Each thread
            runs one ``agent.invoke()`` call.  Use 1 (default) for serial execution.
            SerpAPI free tier allows ~100 req/min; concurrency=4–8 is a safe default.
    """
    dataset = load_bamboogle(limit=limit)

    def _run_one(ex: dict[str, Any]) -> BamboogleResult:
        question: str = ex["question"]
        gold_answers: list[str] = ex.get("golden_answers") or ex.get("answers") or []

        agent_result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        answer = _extract_answer(agent_result)

        em = exact_match(answer, gold_answers)
        cm = contains_match(answer, gold_answers)

        reward_total: float | None = None
        components: dict[str, float] = {}
        if reward_fn is not None:
            loop_output = _to_loop_output(agent_result)
            judge_fn = _make_judge_fn(gold_answers)
            components = reward_fn.reward_components(
                output=loop_output,
                ground_truth=gold_answers[0] if gold_answers else "",
                judge_fn=judge_fn,
            )
            reward_total = float(components.get("total", 0.0))

        return BamboogleResult(
            id=ex.get("id"),
            question=question,
            golden_answers=gold_answers,
            prediction=answer,
            exact_match=em,
            contains_match=cm,
            reward_total=reward_total,
            reward_components=components,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        it = pool.map(_run_one, dataset)
        if verbose:
            it = tqdm(it, total=len(dataset), desc="Bamboogle")
        results = list(it)

    total_em = sum(r.exact_match for r in results)
    total_contains = sum(r.contains_match for r in results)
    n_reward = sum(1 for r in results if r.reward_total is not None)
    total_reward = sum(r.reward_total for r in results if r.reward_total is not None)

    n = len(results)
    summary = BamboogleSummary(
        num_examples=n,
        exact_match=total_em / n if n else 0.0,
        contains_match=total_contains / n if n else 0.0,
        avg_reward=total_reward / n_reward if n_reward else None,
    )

    if output_path is not None:
        _write_jsonl(results, Path(output_path))

    if verbose:
        print(summary)

    return summary, results
```

- [ ] **Step 4: Add `--concurrency` to `examples/run_bamboogle_eval.py`**

Add this argument in the `argparse` block after `--print_trace`:

```python
parser.add_argument(
    "--concurrency",
    type=int,
    default=1,
    help="Number of questions to evaluate in parallel (default: 1 = serial). "
    "Values of 4-8 work well with SerpAPI free tier.",
)
```

Pass it in the `evaluate_bamboogle` call:

```python
_summary, rows = evaluate_bamboogle(
    agent,
    reward_fn=reward_fn,
    limit=args.limit,
    output_path=args.output,
    verbose=True,
    concurrency=args.concurrency,
)
```

- [ ] **Step 5: Add `--concurrency` to `bin/run_bamboogle_eval.sh`**

Add the variable and parsing block alongside the existing `--device` flag:

```bash
CONCURRENCY="${CONCURRENCY:-1}"
```

Inside the `while [[ $# -gt 0 ]]` block, add:

```bash
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
```

Add it to `EVAL_ARGS`:

```bash
EVAL_ARGS=(
  --model "$MODEL"
  --local
  --device "$DEVICE"
  --allow_unsafe_mps
  --search_url "$SEARCH_URL"
  --limit "$LIMIT"
  --output "$OUTPUT"
  --concurrency "$CONCURRENCY"
)
```

- [ ] **Step 6: Run all bamboogle tests**

```bash
pytest tests/unit/test_bamboogle_eval.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/training/eval/bamboogle.py examples/run_bamboogle_eval.py bin/run_bamboogle_eval.sh tests/unit/test_bamboogle_eval.py
git commit -m "feat(eval): parallelize bamboogle eval loop with --concurrency flag"
```

---

## Task 3: Resume from Partial Results

**Files:**
- Modify: `src/training/eval/bamboogle.py`
- Modify: `examples/run_bamboogle_eval.py`
- Modify: `bin/run_bamboogle_eval.sh`
- Modify: `tests/unit/test_bamboogle_eval.py`

When the output JSONL already exists and `resume=True`, load the completed examples from it, skip them in the eval loop, and append new results to the existing file.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_bamboogle_eval.py`:

```python
@patch("src.training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET)
def test_resume_skips_completed(mock_load, tmp_path):
    """With resume=True, already-completed examples are skipped."""
    out = tmp_path / "out.jsonl"
    # Write a partial result file with the first example already done
    out.write_text(
        json.dumps(
            {
                "id": "1",
                "question": "Who invented the telephone?",
                "golden_answers": ["Alexander Graham Bell"],
                "prediction": "Alexander Graham Bell",
                "exact_match": 1.0,
                "contains_match": 1.0,
                "reward_total": None,
                "reward_components": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    call_count = 0

    class _CountingAgent:
        def invoke(self, state: dict) -> MagicMock:
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            r.answer = "Paris"
            del r.metadata
            return r

    evaluate_bamboogle(
        _CountingAgent(), limit=2, output_path=out, verbose=False, resume=True
    )
    # Only the second example should have been evaluated
    assert call_count == 1
    lines = out.read_text().splitlines()
    assert len(lines) == 2


@patch("src.training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET)
def test_resume_false_reruns_all(mock_load, tmp_path):
    """With resume=False (default), all examples run even if output exists."""
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps(
            {
                "id": "1",
                "question": "Who invented the telephone?",
                "golden_answers": ["Alexander Graham Bell"],
                "prediction": "Alexander Graham Bell",
                "exact_match": 1.0,
                "contains_match": 1.0,
                "reward_total": None,
                "reward_components": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    call_count = 0

    class _CountingAgent:
        def invoke(self, state: dict) -> MagicMock:
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            r.answer = "Paris"
            del r.metadata
            return r

    evaluate_bamboogle(
        _CountingAgent(), limit=2, output_path=out, verbose=False, resume=False
    )
    assert call_count == 2
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_bamboogle_eval.py::test_resume_skips_completed \
       tests/unit/test_bamboogle_eval.py::test_resume_false_reruns_all -v
```

Expected: `TypeError` — `evaluate_bamboogle() got an unexpected keyword argument 'resume'`

- [ ] **Step 3: Add `_load_completed_ids()` helper and `resume` param**

Add this helper function in `src/training/eval/bamboogle.py` just above `evaluate_bamboogle`:

```python
def _load_completed_ids(path: Path) -> set[str]:
    """Return the set of question strings already recorded in *path*.

    Uses ``question`` as the key (always present) rather than ``id`` (may be None).
    """
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            completed.add(row["question"])
        except (json.JSONDecodeError, KeyError):
            pass
    return completed
```

Update the `evaluate_bamboogle` signature to add `resume: bool = False`:

```python
def evaluate_bamboogle(
    agent: Any,
    *,
    reward_fn: SearchRewardFunction | None = None,
    limit: int | None = 20,
    output_path: str | Path | None = "bamboogle_results.jsonl",
    verbose: bool = True,
    concurrency: int = 1,
    resume: bool = False,
) -> tuple[BamboogleSummary, list[BamboogleResult]]:
```

Inside `evaluate_bamboogle`, right after `dataset = load_bamboogle(limit=limit)`, add:

```python
    completed_questions: set[str] = set()
    prior_results: list[BamboogleResult] = []
    if resume and output_path is not None:
        out_path = Path(output_path)
        completed_questions = _load_completed_ids(out_path)
        if completed_questions:
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    prior_results.append(
                        BamboogleResult(
                            id=row.get("id"),
                            question=row["question"],
                            golden_answers=row.get("golden_answers", []),
                            prediction=row.get("prediction", ""),
                            exact_match=row.get("exact_match", 0.0),
                            contains_match=row.get("contains_match", 0.0),
                            reward_total=row.get("reward_total"),
                            reward_components=row.get("reward_components", {}),
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    pass

    pending = [ex for ex in dataset if ex["question"] not in completed_questions]
```

Replace `dataset` with `pending` in the `pool.map` call:

```python
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        it = pool.map(_run_one, pending)
        if verbose:
            it = tqdm(it, total=len(pending), desc="Bamboogle")
        new_results = list(it)
```

After the `pool` block, merge results and write output. Replace the rest of the function body:

```python
    results = prior_results + new_results

    total_em = sum(r.exact_match for r in results)
    total_contains = sum(r.contains_match for r in results)
    n_reward = sum(1 for r in results if r.reward_total is not None)
    total_reward = sum(r.reward_total for r in results if r.reward_total is not None)

    n = len(results)
    summary = BamboogleSummary(
        num_examples=n,
        exact_match=total_em / n if n else 0.0,
        contains_match=total_contains / n if n else 0.0,
        avg_reward=total_reward / n_reward if n_reward else None,
    )

    if output_path is not None:
        out_path = Path(output_path)
        if resume and prior_results:
            # Append only new results
            _append_jsonl(new_results, out_path)
        else:
            _write_jsonl(results, out_path)

    if verbose:
        print(summary)

    return summary, results
```

Add `_append_jsonl` helper at the bottom of the file alongside `_write_jsonl`:

```python
def _append_jsonl(results: list[BamboogleResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "id": r.id,
                        "question": r.question,
                        "golden_answers": r.golden_answers,
                        "prediction": r.prediction,
                        "exact_match": r.exact_match,
                        "contains_match": r.contains_match,
                        "reward_total": r.reward_total,
                        "reward_components": r.reward_components,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
```

- [ ] **Step 4: Add `--resume` to `examples/run_bamboogle_eval.py`**

Add after `--concurrency`:

```python
parser.add_argument(
    "--resume",
    action="store_true",
    help="Skip examples already in the output file and append new results.",
)
```

Pass it in the `evaluate_bamboogle` call:

```python
_summary, rows = evaluate_bamboogle(
    agent,
    reward_fn=reward_fn,
    limit=args.limit,
    output_path=args.output,
    verbose=True,
    concurrency=args.concurrency,
    resume=args.resume,
)
```

- [ ] **Step 5: Add `--resume` to `bin/run_bamboogle_eval.sh`**

Add variable default:

```bash
RESUME=0
```

Inside the `while [[ $# -gt 0 ]]` block:

```bash
    --resume) RESUME=1; shift ;;
```

At the end of `EVAL_ARGS`, append conditionally:

```bash
if [ "$RESUME" -eq 1 ]; then
  EVAL_ARGS+=(--resume)
fi
```

- [ ] **Step 6: Run all bamboogle tests**

```bash
pytest tests/unit/test_bamboogle_eval.py -v
```

Expected: all tests pass (including both resume tests).

- [ ] **Step 7: Commit**

```bash
git add src/training/eval/bamboogle.py examples/run_bamboogle_eval.py bin/run_bamboogle_eval.sh tests/unit/test_bamboogle_eval.py
git commit -m "feat(eval): add --resume flag to skip already-completed examples and append results"
```

---

## Usage After All Three Tasks

```bash
# First run — 125 questions, 8 concurrent threads, results cached locally
bin/run_bamboogle_eval.sh --limit 125 --concurrency 8

# Interrupted? Resume from where you left off
bin/run_bamboogle_eval.sh --limit 125 --concurrency 8 --resume

# Dataset already cached — skip the HuggingFace download automatically
```

**Expected speedup for 125 questions:**
- Caching: saves ~2s per run after the first
- Concurrency=8: ~6–8× faster on SerpAPI I/O-bound workloads (each question does up to 8 search turns × ~1s/call)
- Resume: no time lost on interruptions

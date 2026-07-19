# Evaluation Results Panel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface offline evaluation results in the Dev Console via a read-only `/api/debug/eval-results` endpoint that reads JSON result files from a configurable directory, plus a `EvalResultsPanel` that renders them.

**Architecture:** Backend endpoint scans `AGENTIC_SEARCH_EVAL_RESULTS_DIR` (default `data/eval/`) for `*.json`, returns each file's numeric top-level metrics newest-first. `run_bamboogle_eval` also writes a flat summary JSON so Bamboogle participates. Frontend panel self-fetches and renders metric cards.

**Tech Stack:** Python/FastAPI, React + TypeScript + Vite, pytest, vitest.

## Global Constraints

- Read-only; confined to the configured dir; `*.json` only; no client-supplied paths.
- Dev-only surface: panel behind `VITE_DEBUG_PANELS`, endpoint under the existing `/api/debug` router.
- Evaluation only — no training-metric curves, no charts, no live streaming.
- Never commit to `main`; branch `feat/eval-results-panel` (already created).

---

### Task 1: Backend `/api/debug/eval-results` endpoint

**Files:**
- Modify: `src/internal/servers/web/debug_router.py` (add an endpoint inside `create_debug_router`)
- Test: `tests/unit/servers/web/test_debug_router.py`

**Interfaces:**
- Produces: `GET /api/debug/eval-results` → `{"results": [{"name": str, "modified": float, "metrics": {str: number}}, ...]}` newest-first.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/servers/web/test_debug_router.py`:

```python
def _ok(request):  # trivial httpx handler; the eval-results endpoint ignores http
    return httpx.Response(200, json={})


def test_eval_results_lists_numeric_metrics(tmp_path, monkeypatch):
    (tmp_path / "beir.json").write_text(
        json.dumps({"recall@10": 0.5, "ndcg@10": 0.4, "_note": "x"})
    )
    (tmp_path / "notjson.txt").write_text("nope")
    monkeypatch.setenv("AGENTIC_SEARCH_EVAL_RESULTS_DIR", str(tmp_path))

    client = _client(_ok)
    resp = client.get("/api/debug/eval-results")
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["name"] for r in results] == ["beir.json"]
    assert results[0]["metrics"] == {"recall@10": 0.5, "ndcg@10": 0.4}  # _note dropped


def test_eval_results_missing_dir_returns_empty(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_EVAL_RESULTS_DIR", "/nonexistent/xyz-eval")
    client = _client(_ok)
    resp = client.get("/api/debug/eval-results")
    assert resp.status_code == 200
    assert resp.json()["results"] == []
```

Ensure `import json` exists at the top of the test file (add it if missing).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/servers/web/test_debug_router.py::test_eval_results_lists_numeric_metrics tests/unit/servers/web/test_debug_router.py::test_eval_results_missing_dir_returns_empty -v`
Expected: FAIL with 404 (endpoint not defined).

- [ ] **Step 3: Add the endpoint**

In `src/internal/servers/web/debug_router.py`, inside `create_debug_router` (alongside the other `@router.get` handlers, e.g. after `workers`), add:

```python
    @router.get("/eval-results")
    def eval_results() -> dict:
        """Read-only listing of evaluation result files.

        Scans AGENTIC_SEARCH_EVAL_RESULTS_DIR (default data/eval/) for *.json and
        returns each file's numeric top-level metrics, newest first. Confined to
        the configured directory; never raises.
        """
        import json
        import os
        from pathlib import Path

        results_dir = Path(
            os.environ.get("AGENTIC_SEARCH_EVAL_RESULTS_DIR", "data/eval")
        )
        if not results_dir.is_dir():
            return {"results": []}

        out: list[dict] = []
        for path in sorted(results_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            metrics = {
                k: v
                for k, v in data.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            out.append(
                {
                    "name": path.name,
                    "modified": path.stat().st_mtime,
                    "metrics": metrics,
                }
            )
        out.sort(key=lambda r: r["modified"], reverse=True)
        return {"results": out}
```

- [ ] **Step 4: Run tests — GREEN**

Run: `python -m pytest tests/unit/servers/web/test_debug_router.py -q`
Expected: PASS (all, including the two new tests).

- [ ] **Step 5: ruff + commit**

Run: `ruff check src/internal/servers/web/debug_router.py`
```bash
git add src/internal/servers/web/debug_router.py tests/unit/servers/web/test_debug_router.py
git commit -m "feat: add /api/debug/eval-results endpoint for the eval results panel"
```

---

### Task 2: Bamboogle summary-JSON write

**Files:**
- Modify: `examples/run_bamboogle_eval.py` (write a summary JSON after eval)
- Test: `tests/unit/test_bamboogle_eval.py`

**Interfaces:**
- Produces: `write_summary_json(output_path: str, summary: BamboogleSummary) -> Path` writing `dataclasses.asdict(summary)` to `<output_stem>.summary.json`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_bamboogle_eval.py`:

```python
def test_write_summary_json(tmp_path):
    from examples.run_bamboogle_eval import write_summary_json
    from src.training.eval.bamboogle import BamboogleSummary

    out = tmp_path / "bamboogle_results.jsonl"
    summary = BamboogleSummary(
        num_examples=3, exact_match=0.66, contains_match=1.0, avg_reward=0.5
    )
    path = write_summary_json(str(out), summary)

    assert path == tmp_path / "bamboogle_results.summary.json"
    import json

    data = json.loads(path.read_text())
    assert data == {
        "num_examples": 3,
        "exact_match": 0.66,
        "contains_match": 1.0,
        "avg_reward": 0.5,
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_bamboogle_eval.py::test_write_summary_json -v`
Expected: FAIL — `ImportError: cannot import name 'write_summary_json'`.

- [ ] **Step 3: Add the helper + call it in main**

In `examples/run_bamboogle_eval.py`, add near the top (after imports):

```python
def write_summary_json(output_path: str, summary) -> "Path":
    """Write a flat summary JSON (BamboogleSummary fields) next to the rows file
    so the Dev Console eval-results panel can surface EM/contains at a glance."""
    import dataclasses
    import json
    from pathlib import Path

    path = Path(output_path).with_suffix(".summary.json")
    path.write_text(json.dumps(dataclasses.asdict(summary), indent=2))
    return path
```

(If `Path` isn't already imported at module top, the function-local import covers it; leave any existing top-level imports as they are.)

In `main`, change the eval call to keep the summary and write it:

```python
    summary, rows = evaluate_bamboogle(
        agent,
        reward_fn=reward_fn,
        limit=args.limit,
        output_path=args.output,
        verbose=True,
        concurrency=args.concurrency,
        resume=args.resume,
    )
    write_summary_json(args.output, summary)
```

(The variable was previously `_summary`; renaming to `summary` and adding the write is the only change to the call block.)

- [ ] **Step 4: Run tests — GREEN**

Run: `python -m pytest tests/unit/test_bamboogle_eval.py -q`
Expected: PASS (existing + the new test).

- [ ] **Step 5: ruff + commit**

Run: `ruff check examples/run_bamboogle_eval.py`
```bash
git add examples/run_bamboogle_eval.py tests/unit/test_bamboogle_eval.py
git commit -m "feat: run_bamboogle_eval writes a flat summary JSON for the results panel"
```

---

### Task 3: Frontend EvalResultsPanel

**Files:**
- Modify: `web/src/types.ts` (add `EvalResultFile`)
- Modify: `web/src/api.ts` (add `getEvalResults`)
- Create: `web/src/components/debug/EvalResultsPanel.tsx`
- Modify: `web/src/components/debug/DevConsole.tsx` (register the panel)
- Test: `web/src/components/debug/__tests__/EvalResultsPanel.test.tsx`

**Interfaces:**
- Consumes: `GET /api/debug/eval-results` → `{ results: EvalResultFile[] }` (Task 1).

- [ ] **Step 1: Add the type**

In `web/src/types.ts`, add:

```ts
export interface EvalResultFile {
  name: string;
  modified: number;
  metrics: Record<string, number>;
}
```

- [ ] **Step 2: Add the api helper**

In `web/src/api.ts`, next to `getWorkerMetrics`, add (and import the type at the top with the other type imports):

```ts
/** List evaluation result files from the configured results dir (dev console). */
export function getEvalResults(): Promise<{ results: EvalResultFile[] }> {
  return requestJson<{ results: EvalResultFile[] }>("/api/debug/eval-results");
}
```

- [ ] **Step 3: Write the failing panel test**

Create `web/src/components/debug/__tests__/EvalResultsPanel.test.tsx`:

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { EvalResultsPanel } from "../EvalResultsPanel";
import * as api from "../../../api";

describe("EvalResultsPanel", () => {
  it("renders a card with metric rows per result file", async () => {
    vi.spyOn(api, "getEvalResults").mockResolvedValue({
      results: [
        { name: "beir.json", modified: 1_700_000_000, metrics: { "recall@10": 0.5 } },
      ],
    });
    render(<EvalResultsPanel />);
    await waitFor(() => expect(screen.getByText("beir.json")).toBeInTheDocument());
    expect(screen.getByText("recall@10")).toBeInTheDocument();
    expect(screen.getByText("0.5000")).toBeInTheDocument();
  });

  it("shows an empty state when there are no results", async () => {
    vi.spyOn(api, "getEvalResults").mockResolvedValue({ results: [] });
    render(<EvalResultsPanel />);
    await waitFor(() =>
      expect(screen.getByText(/no eval results yet/i)).toBeInTheDocument(),
    );
  });
});
```

- [ ] **Step 4: Run the panel test to verify it fails**

Run (from `web/`): `npx vitest run EvalResultsPanel`
Expected: FAIL — `EvalResultsPanel` module not found.

- [ ] **Step 5: Create the panel**

Create `web/src/components/debug/EvalResultsPanel.tsx`:

```tsx
import { useEffect, useState } from "react";
import { getEvalResults } from "../../api";
import type { EvalResultFile } from "../../types";

function fmt(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

/**
 * Dev-console panel: read-only view of offline evaluation results (BEIR / RAGAS
 * / retrieval / Bamboogle summaries) from the configured results directory.
 */
export function EvalResultsPanel() {
  const [results, setResults] = useState<EvalResultFile[] | null>(null);

  useEffect(() => {
    let alive = true;
    getEvalResults().then(
      (r) => alive && setResults(r.results),
      () => alive && setResults([]),
    );
    return () => {
      alive = false;
    };
  }, []);

  return (
    <section className="eval-results" aria-label="Evaluation results">
      <h2>Evaluation Results</h2>
      {results !== null && results.length === 0 && (
        <p className="eval-results__empty">
          No eval results yet — run an eval with <code>--output</code> into{" "}
          <code>data/eval/</code>.
        </p>
      )}
      {results?.map((file) => (
        <article key={file.name} className="eval-results__card">
          <header>
            <span className="eval-results__name">{file.name}</span>
            <span className="eval-results__mtime">
              {new Date(file.modified * 1000).toISOString().slice(0, 19).replace("T", " ")}
            </span>
          </header>
          {Object.keys(file.metrics).length === 0 ? (
            <p className="eval-results__empty">no numeric metrics</p>
          ) : (
            <table>
              <tbody>
                {Object.entries(file.metrics).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td>{fmt(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </article>
      ))}
    </section>
  );
}
```

- [ ] **Step 6: Register in DevConsole**

In `web/src/components/debug/DevConsole.tsx`, add the import and render it after `<RetrievalLab />`:

```tsx
import { EvalResultsPanel } from "./EvalResultsPanel";
```
```tsx
      <RetrievalLab />
      <EvalResultsPanel />
```

- [ ] **Step 7: Run the panel test — GREEN + typecheck**

Run (from `web/`): `npx vitest run EvalResultsPanel && npm run typecheck`
Expected: both tests pass; no type errors.

- [ ] **Step 8: Full frontend suite + commit**

Run (from `web/`): `npx vitest run`
Expected: all pass.
```bash
git add web/src/types.ts web/src/api.ts web/src/components/debug/EvalResultsPanel.tsx web/src/components/debug/DevConsole.tsx web/src/components/debug/__tests__/EvalResultsPanel.test.tsx
git commit -m "feat: add EvalResultsPanel to the Dev Console"
```

---

## Final verification

- [ ] `python -m pytest tests/unit/servers/web/test_debug_router.py tests/unit/test_bamboogle_eval.py -q` — green.
- [ ] `cd web && npm run typecheck && npx vitest run` — green.
- [ ] `ruff check src/internal/servers/web/debug_router.py examples/run_bamboogle_eval.py` — clean.
- [ ] Manual (optional): drop a JSON like `{"recall@10":0.7}` into `data/eval/`, start the web backend + `VITE_DEBUG_PANELS=1` frontend, open the Dev Console → the Evaluation Results panel lists it.

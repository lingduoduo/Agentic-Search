# Plan: Retrieval degrade-and-gates fixes

Date: 2026-07-11
Spec: `docs/superpowers/specs/2026-07-11-retrieval-degrade-and-gates-design.md`

All changes in `src/internal/retrieval/service.py` + tests in
`tests/unit/retrieval/test_service.py`.

1. **R2 — add `QT_REWRITE` to the gate** → verify: unit test with only
   `QT_REWRITE=1` set builds a non-None pipeline.

2. **R3 — weight the original by identity**
   - Pair each variant future with an `is_original` flag (`i == len(variants)-1`).
   - While collecting surviving results, record the original's result set.
   - Under `QT_FUSION_WEIGHTED=1` with the original surviving, build
     `weights = [1.0 if rs is original else 0.3 ...]`; otherwise `rrf_fuse`.
   - verify: unit test — original variant fails → weighted fuse not called;
     earlier variant fails but original survives → `1.0` lands on original set.

3. **R1 — degrade on reranker failure**
   - Wrap `self._reranker.rerank(...)` in `try/except RerankerTimeoutError` +
     `except Exception`; on failure log a warning and keep `fused`; only append
     `+reranked` on success.
   - verify: unit test — reranker raising `RerankerTimeoutError` / `RuntimeError`
     returns pre-rerank results, mode without `+reranked`, no raise.

4. **Lint + test** → `ruff check . --fix && ruff format .`; `pytest tests/unit`
   retrieval suite green.

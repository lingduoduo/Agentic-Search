# Hybrid Dense Leg Troubleshooting — Implementation Plan

Spec: `docs/superpowers/specs/2026-06-21-hybrid-dense-env-troubleshooting-design.md`
Branch: `docs/hybrid-dense-env-troubleshooting`

## Steps

1. **Write `docs/hybrid-dense-setup.md`** — clean-venv setup, symptom→cause→fix table
   (BertModel/PreTrainedModel → sentence-transformers<3.0; torchvision::nms → torch/torchvision
   mismatch), and the degrade-to-TF-IDF reassurance + `--no-dense` note.
   Verify: doc exists and reads correctly.

2. **Make the dense-init failure message actionable** → in
   `src/internal/servers/retrieval/hybrid.py` `_build_dense`, extend the `logger.warning`
   to name the likely cause and reference `docs/hybrid-dense-setup.md`, keeping the original
   exception.
   Verify (TDD): new test `test_build_dense_failure_logs_actionable_hint` monkeypatches
   `build_e5_encoder` to raise, asserts `_build_dense(...)` returns `None` and the captured
   warning mentions `hybrid-dense-setup`.

3. **Link the doc from CLAUDE.md** → next to the hybrid server command, add a one-line
   pointer to `docs/hybrid-dense-setup.md` for dense setup/troubleshooting.
   Verify: link present.

4. **Run tests + lint** → `PYTHONPATH=src:. python -m pytest tests/unit/servers/retrieval/test_hybrid_retrieval.py -q` green; `ruff check`/`format` clean.

5. **Push + open PR** with spec + plan on the branch.

## Risk / rollback

- Docs + a log-string change + one test; no behavior change to retrieval. Rollback = revert
  the branch.

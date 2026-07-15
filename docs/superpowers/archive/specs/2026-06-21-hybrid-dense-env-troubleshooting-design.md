# Hybrid Dense Leg — Environment Troubleshooting

**Date:** 2026-06-21
**Status:** Approved

## Problem

The hybrid retrieval server (#316) has a dense leg (e5 via sentence-transformers). On a
shared/polluted Python environment the dense leg fails to load and the server silently
degrades to TF-IDF-only. Investigation showed this is **not** a repo dependency bug — a
clean install of `requirements.txt` produces a working dense stack (verified: torch 2.12 +
transformers 5.12 + sentence-transformers 5.6, e5 encodes fine). The observed failures came
from a polluted base conda env:

- `sentence-transformers 2.7.0` (below the `>=3.0.0` the repo pins) → `Could not import
  module 'BertModel'/'PreTrainedModel'`.
- A stray `torchvision` mismatched with the installed `torch` (pulled by unrelated packages)
  → `operator torchvision::nms does not exist` when transformers imports it.

Two gaps: (1) there's no guidance to diagnose/fix this, and (2) the dense-init failure logs
only the raw exception, giving no hint about the real cause or remedy.

## Goal

Help users get the dense leg working — document the clean-environment setup and the failure
symptoms, and make the dense-init failure message actionable. No requirements version
changes (a clean install already works).

## Scope

- New: `docs/hybrid-dense-setup.md`
- Modify: `src/internal/servers/retrieval/hybrid.py` (`_build_dense` warning message)
- Modify: `.claude/CLAUDE.md` (link the troubleshooting doc from the hybrid command)
- Test: `tests/unit/servers/retrieval/test_hybrid_retrieval.py` (dense-init failure logs a hint, returns None)

Out of scope: changing `requirements.txt` versions; fixing the user's base env (a PR can't);
any change to the fusion/contract behavior.

## Design

### Troubleshooting doc (`docs/hybrid-dense-setup.md`)

- What the dense leg needs: `sentence-transformers>=3.0`, a matched `torch`/`torchvision`
  (or no torchvision at all), `transformers` (4.x or 5.x both work in a clean env).
- Symptoms → cause table: `BertModel`/`PreTrainedModel` import error → sentence-transformers
  too old; `torchvision::nms does not exist` → torch/torchvision mismatch from a shared env.
- Fix: install into a fresh venv (`python3 -m venv .venv && source .venv/bin/activate &&
  pip install -e . && pip install -r requirements.txt`), not a shared conda base.
- Reassurance: the hybrid server degrades to TF-IDF automatically, so a broken dense stack is
  non-fatal; `--no-dense` skips it deliberately.

### Actionable failure message

`_build_dense`'s `except` currently logs `"Dense leg unavailable, falling back to TF-IDF
only: %s"`. Extend it to name the likely cause and point to `docs/hybrid-dense-setup.md`,
while still including the original exception.

## Testing

- `_build_dense` with `build_e5_encoder` monkeypatched to raise returns `None` and logs a
  warning that mentions the setup doc (via `caplog`).

## Success criteria

1. `docs/hybrid-dense-setup.md` documents the clean setup + symptom→cause→fix.
2. Dense-init failure logs an actionable hint pointing to the doc.
3. No `requirements.txt` version changes; existing hybrid tests still pass.

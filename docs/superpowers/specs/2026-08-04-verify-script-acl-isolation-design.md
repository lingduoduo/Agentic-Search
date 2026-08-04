# The identity check tests the layer it names

**Date:** 2026-08-04
**Status:** Approved

## Problem

`examples/verify_identity_capabilities.sh` opens by stating what it proves:

> Proves identity shapes results against a retrieval server that ignores
> filters (demo.py does), so enforcement is the web layer's, not the server's.

That premise stopped being true inside the very commit that wrote it. #490
(`8bd338e`) landed two things at once: step 4 taught `demo.py` and `hybrid.py` to
honour `access_acl`, and the script's tool-agent leg. The header was never
revisited, so the script now runs against a server that enforces on its own.

The consequence is not cosmetic. The script exists to isolate the web layer, and
its PASS no longer does: delete `_enforce_access` from `app.py` entirely and the
run stays green, because `demo.py` withholds the document by itself. A harness
whose green survives the deletion of the thing it tests is not a check.

A second defect surfaced while running it. `tool_leg` is called inside a command
substitution:

```bash
tool_anon_status=$(tool_leg "$WORK/tool_anon.json")
```

Under `set -euo pipefail`, a curl failure — most plausibly the 180s timeout,
since the first tool request also pays for loading the local model — exits
non-zero, and the script dies from inside the substitution having printed
nothing. Confirmed against a dead port: exit 7, no output. The observable result
is a run that ends after the previous `PASS` line, which reads exactly like a
clean run to anyone looking at the tail.

## Goals

- The script proves web-layer enforcement specifically, as it claims to.
- A tool leg that cannot complete says so, instead of ending the run in silence.

## Non-goals

- No change to any enforcement. This is the harness, not the control.
- No change to `hybrid.py`. The script starts `demo.py`; a flag on `hybrid.py`
  would be an unused knob.
- The tool leg stays model-dependent. It reads `answer` and `result_summary`,
  and `result_summary` collapses list results to a count, so it can only catch
  what the model quotes. That limitation is already documented in the script and
  is not addressed here.

## Design

### 1. `demo.py --ignore-acl`

An opt-in flag that skips `_allowed_by_acl`, restoring the server the script was
written against:

```python
def create_app(retriever: TfidfRetriever, *, ignore_acl: bool = False):
    ...
    filters = None if ignore_acl else body.filters
```

Default off, so the shipped server keeps enforcing and #490 step 4 is untouched.
The flag reaches `create_app` through `main()`, because a flag the CLI parses and
ignores would fail exactly the way this spec is about.

Preferred over teaching the script to stand up its own filter-ignoring stub: one
argparse flag against a second server to maintain.

### 2. The script starts the server with it

Plus a header that says why the flag is there, so the next person to make
`demo.py` stricter sees what depends on it.

### 3. `SEARCH_AGENT_MODEL` passes through

`SEARCH_AGENT_MODEL=` becomes `SEARCH_AGENT_MODEL="${SEARCH_AGENT_MODEL-}"`.
This keeps the property the existing comment is careful about — the key is
always present, so `load_dotenv(override=False)` can never repopulate it from a
developer `.env` — while letting a caller who exports a model opt into the
tool-agent leg that otherwise SKIPs.

### 4. A curl failure is reported, not fatal

`|| true` on the `tool_leg` curl so it yields curl's `000`, and a check that
turns `000` into a FAIL naming the likely cause. Only `tool_leg` needs this:
`ask` is not in a command substitution, and its failure leaves an empty file that
the following `json.load` rejects loudly.

## Verification

- The script passes with `demo.py --ignore-acl`, and fails when `_enforce_access`
  is stubbed to a passthrough — which is the property that was lost.
- `--ignore-acl` serves a document outside the request's `access_acl`; without it,
  the existing ACL tests still pass.
- The flag is reachable from the command line, and absent by default.
- With `SEARCH_AGENT_MODEL` exported, the tool-agent leg runs instead of skipping.

## Risks

- `--ignore-acl` is a switch that turns off an access control. It is opt-in,
  never set by any shipped entry point, and lives on the demo server, whose
  corpus is a local `.jsonl`. The alternative — a stub server in the script —
  carries the same capability with more code.
- The tool leg's runtime is unbounded in practice: 180s covers a cached
  1.5B model on this hardware (~40–50s per request) with little headroom on a
  slower one. It now fails loudly instead of silently when it does not fit.

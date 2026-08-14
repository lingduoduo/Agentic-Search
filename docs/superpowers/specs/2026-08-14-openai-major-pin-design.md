# Cap the openai test dependency below its breaking major

**Date:** 2026-08-14
**Status:** implemented

## The problem

The `Python Unit Tests` job has been red on `main` continuously since at least
`06942d3` (#511) — through #512, #513, #514, #515 and #516, every one of which
merged with it red. Six tests fail, all in
`tests/unit/test_embedding_cache.py`, all with the same error:

```
AttributeError: module aiohttp has no attribute SocketTimeoutError.
Did you mean: 'ServerTimeoutError'?
```

The name appears nowhere in this repository. The failure is entirely inside a
dependency, at import time:

```
openai/_base_client.py:1482      from ._vendor.httpx_aiohttp import Httpx2AiohttpClient
openai/_vendor/httpx_aiohttp/transport.py:17    aiohttp.SocketTimeoutError: httpx.ReadTimeout,
aiohttp/__init__.py:240 __getattr__  -> AttributeError
```

## Root cause

An unbounded pin sitting next to a hard one.

| requirement | file | value |
|---|---|---|
| `openai` | `requirements-unit-test.txt` | `openai>=1.0.0` — unbounded |
| `aiohttp` | `requirements-unit-test.txt`, `requirements.txt` | `aiohttp==3.9.3` — exact |

**openai 3.0.0** was released and CI, which installs fresh on every run, picked
it up. That release vendors `httpx_aiohttp`, whose transport module references
`aiohttp.SocketTimeoutError` — an attribute aiohttp only added in **3.10**. So
`import openai` raises before any test body runs, and every test touching the
embedding cache fails.

This is also why the job was green for developers and red in CI: a typical
local environment still has **openai 2.x**, which has no `_vendor/httpx_aiohttp`
at all, and works fine against the same pinned `aiohttp==3.9.3`. Nothing in the
repo changed; the index moved underneath it.

## Decision

Cap the dependency: `openai>=1.0.0,<3`.

**Why not raise the aiohttp pin instead.** Bumping `aiohttp` to `>=3.10` would
satisfy openai 3.x, but `aiohttp==3.9.3` is pinned in `requirements.txt` as
well, so that change lands in production to fix a test job — and it would adopt
a major SDK release repo-wide with nothing in the suite exercising the new
version's API. `openai` is declared *only* in `requirements-unit-test.txt`, so
capping it is confined to the environment that is actually broken.

**Why a cap and not an exact pin.** `<3` keeps 2.x security and bugfix releases
flowing while excluding the known-incompatible major. This matches the
precedent already in this file, where `fastapi` is capped `<0.137` after a
major-version regression silently broke router mounting.

Moving to openai 3.x stays possible and should be done deliberately — lifting
this cap together with the `aiohttp` pin, in a change whose job is that
upgrade — not by drift on an unrelated PR.

## Verification

A clean venv built from `requirements-unit-test.txt` reproduces CI exactly,
which is the only faithful check here since the developer environment resolves
a different openai:

| | openai resolved | `tests/unit/test_embedding_cache.py` |
|---|---|---|
| before | 3.0.0 | **6 failed**, 11 passed |
| after | 2.54.0 | **17 passed** |

Full suite in the same venv after the cap: **2648 passed, 36 skipped**.

## Scope

One line of `requirements-unit-test.txt`, plus the comment explaining why the
cap exists so the next person does not silently widen it. No source change, no
test change — the tests were correct and the code was correct; the environment
was wrong.

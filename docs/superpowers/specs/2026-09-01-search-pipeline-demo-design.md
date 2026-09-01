# Rewrite the search-pipeline example over the real pipeline

Date: 2026-09-01

## Problem

`examples/run_search_pipeline.py` is 427 lines that re-implement the search
pipeline from scratch: its own `SearchUser`, `AccessPolicy`, `SearchDocument`,
`SearchFilters`, `IndexFilters`, `SearchChunk`, `SearchSection`, its own
`InMemorySearchIndex` with a hand-written tokenizer and cosine-free scorer, its
own permission filter, and its own adjacent-chunk merge.

Its header calls this deliberate — "intentionally lightweight: it mirrors a
production search flow without depending on database models, enterprise ACL
packages, or a live vector index". That was a reasonable trade when the
alternative was a database. It no longer is: every step it models now has a
counterpart that runs with no server, no model and no database.

| Example | Real code |
| --- | --- |
| `build_index_filters`, `SearchFilters`, `IndexFilters` | `src.context.models.SearchFilters`, `src.context.preprocessing.build_user_only_filters` |
| `InMemorySearchIndex`, `_score_text`, `_tokenize`, `_split_text` | `TfidfRetriever.from_docs` (`src.internal.servers.retrieval.demo`) |
| `default_permission_filter`, `AccessPolicy` | `SearchFilters.matches`, as `_enforce_access` uses it |
| `merge_adjacent_chunks`, `section_from_chunks` | `src.context.utils.merge_adjacent_documents` |

**The file does not merely duplicate the pipeline — it teaches a different
access model.** Its `AccessPolicy(public, allowed_user_emails,
allowed_group_ids)` is an allowlist checked against a user object. The real
model is a list of prefixed principal strings (`user:`, `email:`, `group:`,
`public`) on the request, matched against a document's `metadata["acl"]`, where
**a document that declares no ACL is public**. The two disagree on the default.
The example also defines a `SearchFilters` whose fields and semantics differ
from the real class of that name, so reading it teaches names that mean
something else in `src/`.

## Design

Rewrite the example as a thin script over the real pipeline. Four steps, each
one call into `src/`:

1. **Build the caller's filters** — `build_user_only_filters(user_id, email=,
   group_ids=)` returns a real `SearchFilters` carrying the caller's ACL.
2. **Retrieve** — `TfidfRetriever.from_docs(DEMO_CORPUS)`, the same class the
   demo retrieval server serves, over a corpus literal in the file.
3. **Enforce access** — `[r for r in results if filters.matches(r.metadata)]`.
4. **Assemble context** — `documents_from_search_results` then
   `merge_adjacent_documents`.

### What the demo is for

Step 3 is the reason to keep the file at all. Filters are *sent* to a retrieval
backend, and this repo's backends honour them, but a backend is not obliged to —
so passing filters down is not enforcement. The web backend pairs every
filtered retrieval call with its own `_enforce_access` check, and this
repository has twice shipped a call site that serialised filters without pairing
the check, opening a cross-user read.

The example takes `--skip-enforcement`, which drops step 3 and prints what the
caller would read without it. That makes the invariant demonstrable rather than
described, which no test in `src/` currently does end to end.

### Corpus

A five-document literal in the file: three public, one restricted to
`group:search-admins`, one restricted to `email:owner@example.test`. Two public
documents carry `document_id`/`chunk_id` metadata for consecutive chunks of one
source, so step 4 has something to merge.

ACL lives in `metadata["acl"]`, the key `SearchFilters.matches` reads.

### Interface

- `demo_corpus() -> list[dict]`
- `retrieve(query, *, topk) -> list[SearchResult]`
- `enforce_access(results, filters) -> list[SearchResult]`
- `assemble_sections(results) -> list[ContextSection]`
- `run_demo(*, user_id, email, group_ids, query, enforce) -> list[ContextSection]`
- `main()` — prints each step's document ids so the effect of enforcement is
  visible in the output.

No new dataclasses. Anything the pipeline needs is already a type in `src/`.

## Testing

`tests/unit/test_readme_examples.py` currently imports eight names from this
file; those tests are rewritten against the real types:

- An anonymous caller sees only the documents with no ACL and `public`.
- A caller in `search-admins` additionally sees the group-restricted document.
- A caller with the owner's email sees the email-restricted one, and not the
  group one.
- `--skip-enforcement` returns the restricted document to a caller who may not
  read it — the regression the demo exists to show.
- Consecutive chunks of one source merge into a single section.

The GRPO test in the same file is untouched.

## Out of scope

The demo stays synchronous and server-free. It does not gain a retrieval-server
mode, a reranker leg, or an LLM call; `run_agentic_search` already covers those.

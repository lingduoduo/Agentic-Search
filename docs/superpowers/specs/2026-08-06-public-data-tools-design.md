# Public data-source tools

**Date:** 2026-08-06
**Status:** Approved for planning

## Problem

The Tool Agent surface at `/tools` has almost nothing to select between.
`tool_knowledge_base()` seeds three tools, and two of them —
`search` and `rag_routing_tool` — are registered as not agent-callable. The
request-bound corpus search is built per request, so a tool agent turn typically
sees two tools: `search` and `web_search`. A surface whose entire premise is
"the model picks a tool" cannot demonstrate anything with two tools, one of
which answers the same questions as the other.

The fix is to give the agent a set of genuinely distinct public capabilities:
weather, stock quotes, crypto prices, currency conversion, Wikipedia, ArXiv,
the Wayback Machine, geocoding, and nearby points of interest. All nine are
free and keyless, so they work on a clean checkout with no configuration.

## Non-goals

- No caching layer, no rate-limit machinery, no retry/backoff beyond a timeout.
- No API-key-gated providers. Every source here is free and keyless.
- No new environment variables and no feature flag. The tools are on by default.
- No `SemanticRouter` wiring into the agent loop (see "Tool selection" below).
- No new pip dependencies.

## Tool selection: offer all nine

`_run_tool_agent` passes every agent-callable tool into the prompt
(`tool_agent_runner.py:151`). Adding nine takes that from ~2 schemas to ~11 in
front of a local 1.5B model, which is the shape of the PR #479 regression.

We accept that risk rather than pre-filtering. #479 was caused by four tools
that *overlapped* — several ways to search the same corpus behind the same
argument, so no correct choice existed. These nine occupy disjoint domains
(weather vs. ArXiv vs. currency); a question that suits one suits no other. The
`SemanticRouter` exists and could pre-filter, but a wrongly-filtered tool is
invisible to the model, which is a harder failure to diagnose than a visibly
wrong pick. Ship all nine, observe, and only add filtering if selection actually
degrades.

## Layout

```
src/internal/tools/public_data/
  __init__.py    public_data_tools() -> list[Tool]
  _http.py       get_json() / get_text(): aiohttp, shared User-Agent, timeout
  knowledge.py   search_wikipedia, search_arxiv, search_wayback
  market.py      get_stock_quote, get_crypto_price, convert_currency
  geo.py         get_weather, search_location, search_nearby_places
```

Each theme module holds three `build_*_tool()` factories returning
`FunctionTool`s. `public_data_tools()` returns the nine in a fixed order.
`tool_knowledge_base()` extends its list with that call, so the tools seed
alongside the built-ins and inherit `agent_callable=True` — none is added to
`NOT_AGENT_CALLABLE`.

## Return contract

This is the load-bearing decision; the source-card work depends on it.

**Citeable tools return a JSON array of `{title, content, url}` objects** — the
exact shape `build_search_routing_tool` already returns. `search_wikipedia`,
`search_arxiv`, and `search_wayback` are the citeable three, and each maps its
upstream response onto that shape:

| Tool | `title` | `content` | `url` |
| --- | --- | --- | --- |
| `search_wikipedia` | page title | intro extract | article URL |
| `search_arxiv` | paper title | abstract (truncated) | `entry_id` |
| `search_wayback` | ISO timestamp of the snapshot | original URL + MIME type | `web.archive.org` snapshot URL |

**Non-citeable tools return a JSON object of facts** — a flat mapping such as
`{"location": ..., "temperature": ..., "description": ...}`. They answer; they
do not cite.

**Failure returns `{"error": "<message>"}`** and never raises. A dead or
throttled upstream degrades one tool, not the turn. This mirrors
`routing_tools.py:65` rather than the borrowed code's `ActionResponse` envelope,
which is an MCP transport concern our `FunctionTool` does not share.

All nine declare `effect=ToolEffect.READ_ONLY`.

## Source cards

`_extract_tool_calls_and_docs` currently builds `ContextDocument`s only when
`tool_name == _CORPUS_SEARCH_NAME` (`tool_agent_runner.py:97`), so the
`citeable` flag that already exists on `Tool` is never read on this path, and
even `web_search` results go unsurfaced.

Because citeable tools now share the corpus-search shape, this generalizes to a
membership test against the set of citeable tool names, with the existing
`ContextDocument` loop unchanged. The caller resolves that set from the tools it
passed to the loop, so a rename cannot silently disable citation. Documents
carry `metadata={"source": <tool_name>}` so a card shows which tool produced it.

## Deviations from the borrowed reference

The idea came from an MCP tool file. These changes are deliberate:

| Reference | Here | Why |
| --- | --- | --- |
| sync `requests` inside `async def` | `aiohttp` | blocking the loop stalls every concurrent request; `aiohttp` is the established pattern (`search.py:194`, `api.py:222`) |
| `ActionResponse` + `TextContent` | plain JSON string | MCP-specific; `FunctionTool.execute` returns `(text, raw, meta)` |
| `wikipedia` / `arxiv` packages | raw HTTP | both are synchronous and would need executor wrapping; raw HTTP adds no dependency |
| emoji `logging.info` per call | module logger at `debug` | matches repo convention |
| `traceback.format_exc()` on every error | message only, `exc_info` at debug | a stack trace per upstream hiccup is noise |
| Overpass query interpolates `query` into a regex | whitelist `[A-Za-z0-9 _-]`, reject otherwise | the reference lets a `"]` in `query` escape the regex and construct arbitrary Overpass QL |
| generic `Mozilla/5.0` User-Agent | descriptive project UA | Nominatim's usage policy requires identifying the application |

## Upstream endpoints

| Tool | Endpoint | Notes |
| --- | --- | --- |
| `get_weather` | `geocoding-api.open-meteo.com/v1/search` then `api.open-meteo.com/v1/forecast` | geocode step skipped when lat/lon given; WMO code → description map |
| `get_stock_quote` | `query1.finance.yahoo.com/v8/finance/chart/{symbol}` | requires a browser-like UA |
| `get_crypto_price` | `api.coingecko.com/api/v3/simple/price` | symbol→id map for common tickers, else pass through |
| `convert_currency` | `api.exchangerate-api.com/v4/latest/{from}` | |
| `search_wikipedia` | `{lang}.wikipedia.org/w/api.php` (`list=search`, then `prop=extracts`) | |
| `search_arxiv` | `export.arxiv.org/api/query` | Atom XML parsed with stdlib `xml.etree` |
| `search_wayback` | `web.archive.org/cdx/search/cdx` | JSON; first row is headers |
| `search_location` | `nominatim.openstreetmap.org/search` | descriptive UA required |
| `search_nearby_places` | `overpass-api.de/api/interpreter` | POST; sanitized query; longer timeout |

`_http` applies a 10-second default timeout, 30 seconds for Overpass. Timeouts,
non-2xx responses, and malformed payloads all become `{"error": ...}`.

## Testing

Every test mocks the `_http` layer. No test performs live network I/O.

- Per tool: one happy path asserting the parsed shape, one upstream-failure path
  asserting the `{"error": ...}` contract. (18 tests)
- `public_data_tools()` returns nine tools with unique names and valid JSON
  Schema parameters.
- Seeding: `tool_knowledge_base()` includes all nine and each is agent-callable.
- Citeable output from `search_wikipedia` becomes `ContextDocument`s through
  `_extract_tool_calls_and_docs`; a non-citeable tool's output does not.
- `search_nearby_places` rejects a query containing Overpass QL metacharacters.

## Success criteria

1. `pytest` passes with the new tests included.
2. `ruff check . && ruff format --check .` is clean.
3. `GET /api/debug/tools` lists all nine under the `local` server.
4. Asking `/tools` a weather question produces a `get_weather` call in the trace;
   asking a Wikipedia question produces a source card.
5. No new entries in `requirements.txt`.

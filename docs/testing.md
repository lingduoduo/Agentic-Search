# Testing

[← Back to README](../README.md)

This guide covers backend unit and regression tests, opt-in integration tests, frontend checks, and focused debugging commands.

## Backend unit and regression tests

```bash
pytest                           # full suite
pytest tests/unit/ -v            # unit only
pytest tests/unit/servers/ -v    # server-focused
pytest tests/unit/test_reward.py tests/unit/test_grpo.py tests/unit/test_llm_agent_generation.py -v
```

| Test area | What is tested |
|-----------|----------------|
| `server/billing/` | Circuit breaker state, endpoint responses, HTTP mocks |
| `server/features/hooks/` | SSRF safety, endpoint validation, `HookValidateStatus` |
| `server/license/` | PEM stripping, `_strip_pem` boundary cases |
| `server/middleware/` | Path allowlist, license enforcement, tier gating |
| `server/settings/` | `_load_license_status`, `/settings` endpoint |
| `server/web/test_tool_trace.py` | `ToolCallView` trace parsing, latency rounding, list/string summarisation, error forwarding |
| `utils/test_license_utils.py` | RSA signature verification with real key pairs |
| `utils/test_license_expiry.py` | 18 parametrized `ExpiryWarningStage` boundary points |
| `utils/test_tier.py` | `get_tier` + `tier_at_least` matrix |

## Opt-in integration tests

Integration tests require a live server at `http://localhost:8080` by default:

```bash
pytest tests/integration/ -v
API_SERVER_HOST=localhost API_SERVER_PORT=8080 pytest tests/integration/
```

For the legacy integration harness, launch the API server on port 8080 with `AUTH_TYPE=basic` and `ENABLE_PAID_ENTERPRISE_EDITION_FEATURES=true`. Tests using `mock_llm_response` also require `INTEGRATION_TESTS_MODE=true` on the server. A `tests/integration/.env` file can supply these variables.

```bash
python -m dotenv -f .env run -- pytest -s tests/integration/tests/
python -m dotenv -f .env run -- pytest -s tests/integration/tests/path_to/test_file.py
python -m dotenv -f .env run -- pytest -s tests/integration/tests/path_to/test_file.py::test_function_name
```

Some individual tests require the mock connector server:

```bash
cd tests/integration/mock_services
docker compose -f docker-compose.mock-it-services.yml -p mock-it-services-stack up -d
```

If the main stack uses a non-default name, update the compose file's network to `<your stack name>_default`.

## Frontend checks

```bash
cd web && npm run typecheck            # TypeScript check
cd web && npm run build                # production bundle → web/dist/ (served by FastAPI)
cd web && npm run test -- --run        # Vitest unit tests
```

Frontend tests live under `web/src/components/__tests__/`:

| Test file | What is tested |
|-----------|----------------|
| `App.test.tsx` | SSE streaming flow, intent class applied per response, reset on new session |
| `AnswerPanel.test.tsx` | Markdown rendering, `[D1]` citation link generation, `ReactNode[]` children handling |
| `SessionTimeline.test.tsx` | Chat bubble layout, system message filtering, stable React keys |
| `SourceGrid.test.tsx` | Card expand/collapse, copy button 1.5 s feedback, `id` anchor attribute |
| `ToolCallTracePanel.test.tsx` | Empty→null, completed/failed card classes, latency display, JSON arguments |

## Debugging search providers

### SerpAPI

Run this from the repository root to validate the configured key and inspect the provider response:

```bash
KEY=$(grep -E '^SERP_API_KEY=' .env | cut -d= -f2- | tr -d '"'\'' ')

curl -s "https://serpapi.com/search.json?engine=google&q=what+is+FAISS&api_key=$KEY" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('ERROR:',d['error']) if 'error' in d else print(len(d.get('organic_results',[])),'results —',d.get('search_metadata',{}).get('status'))"
```

### Browser-backed retrieval

```bash
pip install playwright
playwright install chromium
curl -s --max-time 30 -X POST http://127.0.0.1:8002/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"what is FAISS","top_k":5}' | python3 -m json.tool
```

### Request routing and provider fallback

Run the focused backend suites when changing intent classification, explicit modes, provider precedence, access filters, or routing metadata:

```bash
pytest -q \
  tests/unit/test_execution_fallbacks.py \
  tests/unit/servers/web/test_agent_router.py \
  tests/unit/servers/web/test_web_experience_app.py
```

The fallback tests assert that auto-search tries internal retrieval, then SerpAPI, then the configured browser service, and never substitutes a local-model answer when all providers are empty. Access-filter tests separately protect ACL enforcement on each path that reads documents. The expected contract is documented in [API request routing](request-routing.md).

### End-to-end identity check

`examples/verify_identity_capabilities.sh` runs the access-control property
against a live stack: it starts `demo.py --ignore-acl` and the web backend over a
two-document corpus, one public and one ACL'd to another user, then asserts that
neither an anonymous nor a signed-in caller can read the restricted one.

The `--ignore-acl` is the point. Both bundled retrieval servers honour
`access_acl`, so without it the script would pass even with the web layer's
enforcement deleted — it would be testing the server, not the layer it names.

It asserts positive controls too (the corpus was really searched; the second
request really authenticated), because the first version passed for the wrong
reason. The tool-agent leg needs a local model and SKIPs without one:

```bash
examples/verify_identity_capabilities.sh                              # SKIPs the tool leg
SEARCH_AGENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  examples/verify_identity_capabilities.sh                            # runs it (~40-50s/request)
```

That leg is weaker than the retrieval one and says so: it can only catch a leak
the model quotes into its own answer.

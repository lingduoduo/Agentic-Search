#!/usr/bin/env bash
set -u

# Batch-check Agentic Search web trigger paths.
#
# Prereqs:
#   1. Web backend running at API_URL, default http://127.0.0.1:7860
#   2. Local retrieval running at LOCAL_RETRIEVAL_URL for local/chat checks
#   3. Optional browser retrieval running at BROWSER_RETRIEVAL_URL for browser checks
#   4. Optional SERP_API_KEY in env or .env for SerpAPI checks
#
# Example:
#   bin/check_agent_triggers.sh
#   QUERY="What is FAISS?" TOP_K=5 bin/check_agent_triggers.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

API_URL="${API_URL:-http://127.0.0.1:7860}"
LOCAL_RETRIEVAL_URL="${LOCAL_RETRIEVAL_URL:-http://localhost:8000/retrieve}"
BROWSER_RETRIEVAL_URL="${BROWSER_RETRIEVAL_URL:-http://localhost:8001/retrieve}"
QUERY="${QUERY:-FAISS}"
TOP_K="${TOP_K:-3}"
OUT_DIR="${OUT_DIR:-/tmp/agentic-search-trigger-checks}"
RUN_EXTERNAL="${RUN_EXTERNAL:-auto}" # auto | 1 | 0
RUN_BROWSER="${RUN_BROWSER:-auto}"   # auto | 1 | 0
GOOGLE_DISABLED_REASON="Google PSE disabled for this demo because the current API key/CSE returns 403"

mkdir -p "$OUT_DIR"

pass=0
fail=0
skip=0

print_header() {
  printf "\n== %s ==\n" "$1"
}

json_payload() {
  local mode="$1"
  local source="$2"
  local search_url="$3"
  python - "$QUERY" "$mode" "$source" "$search_url" "$TOP_K" <<'PY'
import json
import sys

query, mode, source, search_url, top_k = sys.argv[1:6]
print(json.dumps({
    "query": query,
    "mode": mode,
    "source_provider": source,
    "search_url": search_url,
    "top_k": int(top_k),
}))
PY
}

check_endpoint() {
  local name="$1"
  local url="$2"
  local payload="$3"
  local response_file="$OUT_DIR/${name}.json"

  local status
  status="$(curl -sS -o "$response_file" -w "%{http_code}" \
    -X POST "$url" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>"$OUT_DIR/${name}.curl.err")"

  if [ "$status" -ge 200 ] && [ "$status" -lt 300 ]; then
    printf "PASS %-42s HTTP %s\n" "$name" "$status"
    pass=$((pass + 1))
    return 0
  fi

  printf "FAIL %-42s HTTP %s body=%s\n" "$name" "$status" "$response_file"
  if [ -s "$OUT_DIR/${name}.curl.err" ]; then
    sed 's/^/  curl: /' "$OUT_DIR/${name}.curl.err"
  fi
  fail=$((fail + 1))
  return 1
}

redact_file() {
  local path="$1"
  python - "$path" <<'PY'
import re
import sys
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    text = f.read()

def redact_url(match):
    parsed = urlsplit(match.group(0))
    query = urlencode(
        [
            (key, "[REDACTED]" if key.lower() in {"key", "api_key"} else value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        ]
    )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment))

text = re.sub(r"https://[^\s'\"<>]+", redact_url, text)
with open(path, "w", encoding="utf-8") as f:
    f.write(text)
PY
}

validate_retrieval_response() {
  local name="$1"
  local response_file="$OUT_DIR/${name}.json"

  python - "$response_file" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

rows = data.get("result", data.get("results", []))
if rows and isinstance(rows[0], dict):
    rows = [rows]
doc_count = sum(len(row) for row in rows if isinstance(row, list))
if doc_count < 1:
    raise SystemExit("retrieval endpoint returned no documents")
print(f"docs={doc_count}")
PY
}

validate_agent_response() {
  local name="$1"
  local mode="$2"
  local source="$3"
  local response_file="$OUT_DIR/${name}.json"

  python - "$response_file" "$mode" "$source" <<'PY'
import json
import sys

path, mode, source = sys.argv[1:4]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

docs = data.get("documents", [])
answer = data.get("answer", "")
if not docs:
    raise SystemExit("no documents returned")

if any(doc.get("title") == "Search error" for doc in docs):
    details = "; ".join(doc.get("content", "") for doc in docs if doc.get("title") == "Search error")
    raise SystemExit(f"search error document returned: {details}")

if mode in {"search_tool", "hybrid_search"}:
    expected = {
        "retrieval": "retrieval",
        "google": "google",
        "serpapi": "serpapi",
        "browser": "browser",
    }.get(source)
    if expected is not None:
        seen = {doc.get("metadata", {}).get("source_provider") for doc in docs}
        if expected not in seen:
            raise SystemExit(f"expected source_provider={expected}, saw {sorted(seen)}")
    if source == "all":
        seen = {doc.get("metadata", {}).get("source_provider") for doc in docs}
        if "retrieval" not in seen:
            raise SystemExit(f"all-sources response did not include retrieval, saw {sorted(seen)}")
        if "serpapi" not in seen:
            raise SystemExit(f"all-sources response did not include serpapi, saw {sorted(seen)}")

if mode in {"chat_once", "chat_loop"} and not answer.strip():
    raise SystemExit("chat response answer is empty")

print(f"docs={len(docs)} citations={len(data.get('citations', []))}")
PY
}

run_agent_case() {
  local mode="$1"
  local source="$2"
  local search_url="$3"
  local name="${mode}_${source}"
  local payload
  payload="$(json_payload "$mode" "$source" "$search_url")"

  if check_endpoint "$name" "$API_URL/api/agent" "$payload"; then
    redact_file "$OUT_DIR/${name}.json"
    if validate_agent_response "$name" "$mode" "$source"; then
      printf "     %-42s validated\n" "$name"
    else
      printf "FAIL %-42s validation failed body=%s\n" "$name" "$OUT_DIR/${name}.json"
      fail=$((fail + 1))
      pass=$((pass - 1))
    fi
  fi
}

has_serp_key() {
  [ -n "${SERP_API_KEY:-}" ] || [ -n "${SERPAPI_API_KEY:-}" ]
}

should_run_external() {
  case "$RUN_EXTERNAL" in
    1|true|yes) return 0 ;;
    0|false|no) return 1 ;;
    auto) return 0 ;;
    *) return 1 ;;
  esac
}

should_run_browser() {
  case "$RUN_BROWSER" in
    1|true|yes) return 0 ;;
    0|false|no) return 1 ;;
    auto) return 0 ;;
    *) return 1 ;;
  esac
}

skip_case() {
  printf "SKIP %-42s %s\n" "$1" "$2"
  skip=$((skip + 1))
}

print_header "Backend Health"
if curl -sS -f "$API_URL/health" >"$OUT_DIR/health.json"; then
  printf "PASS %-42s %s\n" "backend /health" "$API_URL/health"
  pass=$((pass + 1))
else
  printf "FAIL %-42s %s\n" "backend /health" "$API_URL/health"
  fail=$((fail + 1))
fi

print_header "Retrieval Health"
if check_endpoint "local_retrieval_direct" "$LOCAL_RETRIEVAL_URL" \
  "$(python - "$QUERY" "$TOP_K" <<'PY'
import json
import sys
print(json.dumps({"queries": [sys.argv[1]], "topk": int(sys.argv[2])}))
PY
)"; then
  if validate_retrieval_response "local_retrieval_direct"; then
    printf "     %-42s validated\n" "local_retrieval_direct"
  else
    printf "FAIL %-42s validation failed body=%s\n" \
      "local_retrieval_direct" "$OUT_DIR/local_retrieval_direct.json"
    fail=$((fail + 1))
    pass=$((pass - 1))
  fi
fi

if should_run_browser; then
  if check_endpoint "browser_retrieval_direct" "$BROWSER_RETRIEVAL_URL" \
    "$(python - "$QUERY" "$TOP_K" <<'PY'
import json
import sys
print(json.dumps({"queries": [sys.argv[1]], "topk": int(sys.argv[2])}))
PY
)"; then
    if validate_retrieval_response "browser_retrieval_direct"; then
      printf "     %-42s validated\n" "browser_retrieval_direct"
    else
      printf "FAIL %-42s validation failed body=%s\n" \
        "browser_retrieval_direct" "$OUT_DIR/browser_retrieval_direct.json"
      fail=$((fail + 1))
      pass=$((pass - 1))
    fi
  fi
else
  skip_case "browser_retrieval_direct" "RUN_BROWSER=$RUN_BROWSER"
fi

print_header "Agent Trigger Matrix"
run_agent_case "search_tool" "retrieval" "$LOCAL_RETRIEVAL_URL"
run_agent_case "hybrid_search" "retrieval" "$LOCAL_RETRIEVAL_URL"
run_agent_case "chat_once" "retrieval" "$LOCAL_RETRIEVAL_URL"
run_agent_case "chat_loop" "retrieval" "$LOCAL_RETRIEVAL_URL"

if should_run_browser; then
  run_agent_case "search_tool" "browser" "$BROWSER_RETRIEVAL_URL"
  run_agent_case "hybrid_search" "browser" "$BROWSER_RETRIEVAL_URL"
else
  skip_case "search_tool_browser" "RUN_BROWSER=$RUN_BROWSER"
  skip_case "hybrid_search_browser" "RUN_BROWSER=$RUN_BROWSER"
fi

# Google PSE is intentionally skipped in the demo matrix until the configured
# API key/CSE pair is fixed. The backend search helper remains available for a
# future re-enable, but the batch health check should focus on active paths.
skip_case "search_tool_google" "$GOOGLE_DISABLED_REASON"
skip_case "hybrid_search_google" "$GOOGLE_DISABLED_REASON"

if should_run_external && has_serp_key; then
  run_agent_case "search_tool" "serpapi" "$LOCAL_RETRIEVAL_URL"
  run_agent_case "hybrid_search" "serpapi" "$LOCAL_RETRIEVAL_URL"
else
  skip_case "search_tool_serpapi" "missing SERP_API_KEY/SERPAPI_API_KEY or RUN_EXTERNAL=$RUN_EXTERNAL"
  skip_case "hybrid_search_serpapi" "missing SERP_API_KEY/SERPAPI_API_KEY or RUN_EXTERNAL=$RUN_EXTERNAL"
fi

if should_run_external && has_serp_key; then
  run_agent_case "search_tool" "all" "$LOCAL_RETRIEVAL_URL"
  run_agent_case "hybrid_search" "all" "$LOCAL_RETRIEVAL_URL"
else
  skip_case "search_tool_all" "requires SERP_API_KEY/SERPAPI_API_KEY or RUN_EXTERNAL=$RUN_EXTERNAL"
  skip_case "hybrid_search_all" "requires SERP_API_KEY/SERPAPI_API_KEY or RUN_EXTERNAL=$RUN_EXTERNAL"
fi

print_header "Summary"
printf "PASS=%s FAIL=%s SKIP=%s\n" "$pass" "$fail" "$skip"
printf "Responses saved in %s\n" "$OUT_DIR"

if [ "$fail" -gt 0 ]; then
  exit 1
fi

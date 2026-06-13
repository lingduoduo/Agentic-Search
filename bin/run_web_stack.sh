#!/usr/bin/env bash
# Start the full local dev stack: SerpAPI retrieval server + web backend + Vite frontend.
#
# Usage:
#   bin/run_web_stack.sh
#
# Required env var: SERP_API_KEY (add to .env or export it)
# Optional env vars in .env:
#   SEARCH_AGENT_MODEL   e.g. Qwen/Qwen2.5-1.5B-Instruct  (enables Search Agent mode)
#   SEARCH_AGENT_DEVICE  e.g. mps  (default: mps)
#   SERP_PORT            retrieval server port (default: 8000)
#   WEB_PORT             web backend port (default: 7860)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env
if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT/.env"
    set +a
fi

if [ -z "${SERP_API_KEY:-}" ]; then
    echo "ERROR: SERP_API_KEY not set (add it to .env or export it)" >&2
    exit 1
fi

SERP_PORT="${SERP_PORT:-8000}"
WEB_PORT="${WEB_PORT:-7860}"
SEARCH_AGENT_MODEL="${SEARCH_AGENT_MODEL:-}"
SEARCH_AGENT_DEVICE="${SEARCH_AGENT_DEVICE:-mps}"

echo ">> Starting SerpAPI retrieval server on port ${SERP_PORT}..."
PYTHONPATH="$ROOT" python3 -m src.internal.servers.web_search.serp --port "$SERP_PORT" &
SERP_PID=$!
trap 'echo ">> Stopping processes..."; kill "$SERP_PID" "$WEB_PID" 2>/dev/null; wait 2>/dev/null || true' EXIT

# Wait for retrieval server to be ready (up to 15 s)
for i in $(seq 1 15); do
    if curl -sf "http://localhost:${SERP_PORT}/health" >/dev/null 2>&1; then
        echo ">> Retrieval server ready."
        break
    fi
    [ "$i" -eq 15 ] && { echo "ERROR: retrieval server did not start in 15 s" >&2; exit 1; }
    sleep 1
done

if [ -n "$SEARCH_AGENT_MODEL" ]; then
    echo ">> Search Agent mode enabled: model=${SEARCH_AGENT_MODEL} device=${SEARCH_AGENT_DEVICE}"
else
    echo ">> Search Agent mode disabled (set SEARCH_AGENT_MODEL in .env to enable)"
fi

echo ">> Starting web backend on port ${WEB_PORT}..."
PYTHONPATH="$ROOT" \
SEARCH_AGENT_MODEL="$SEARCH_AGENT_MODEL" \
SEARCH_AGENT_DEVICE="$SEARCH_AGENT_DEVICE" \
AGENTIC_SEARCH_RETRIEVAL_URL="http://localhost:${SERP_PORT}/retrieve" \
    uvicorn src.internal.servers.web.app:app \
        --host 127.0.0.1 --port "$WEB_PORT" &
WEB_PID=$!

echo ">> Starting frontend dev server..."
echo ">> Open http://127.0.0.1:5173 when ready."
cd "$ROOT/web" && npm run dev

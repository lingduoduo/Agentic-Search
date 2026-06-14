#!/usr/bin/env bash
# Run the Bamboogle eval against a SerpAPI retrieval server on Apple Silicon.
#
# Usage:
#   bin/run_bamboogle_eval.sh [--model MODEL] [--limit N] [--device DEVICE] [--smoke]
#   bin/run_bamboogle_eval.sh --server_url http://localhost:8080 --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
#
# Defaults:
#   --model  Qwen/Qwen2.5-1.5B-Instruct
#   --limit  5
#   --device mps
#
# Flags:
#   --smoke      Run 1 example with --print_output for a quick sanity check
#   --server_url Use an OpenAI-compatible server (e.g. mlx-lm) instead of --local
#                Start with: python -m mlx_lm.server --model <model> --port 8080
#
# Reads SERP_API_KEY from .env automatically.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env
if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

if [ -z "${SERP_API_KEY:-}" ]; then
  echo "ERROR: SERP_API_KEY not set (add it to .env or export it)" >&2
  exit 1
fi

# Defaults
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
LIMIT=5
DEVICE="mps"
SMOKE=0
SERP_PORT="${SERP_PORT:-8000}"
OUTPUT="${OUTPUT:-bamboogle_results.jsonl}"
CONCURRENCY="${CONCURRENCY:-1}"
RESUME=0
SERVER_URL=""

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2";       shift 2 ;;
    --limit)       LIMIT="$2";       shift 2 ;;
    --device)      DEVICE="$2";      shift 2 ;;
    --output)      OUTPUT="$2";      shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --resume)      RESUME=1;         shift ;;
    --smoke)       SMOKE=1; LIMIT=1; shift ;;
    --server_url)  SERVER_URL="$2";  shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

SEARCH_URL="http://localhost:${SERP_PORT}/retrieve"

# Start SerpAPI retrieval server in background
echo ">> Starting SerpAPI retrieval server on port ${SERP_PORT}..."
python3 -m src.internal.servers.web_search.serp --port "$SERP_PORT" &
SERP_PID=$!
trap 'echo ">> Stopping retrieval server (pid $SERP_PID)..."; kill "$SERP_PID" 2>/dev/null; wait "$SERP_PID" 2>/dev/null || true' EXIT

# Wait for the server to be ready (up to 30s)
echo ">> Waiting for retrieval server..."
for i in $(seq 1 30); do
  if curl -sf "http://localhost:${SERP_PORT}/health" >/dev/null 2>&1; then
    echo ">> Retrieval server ready."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: Retrieval server did not start within 30s" >&2
    exit 1
  fi
  sleep 1
done

# Build eval args
EVAL_ARGS=(
  --model "$MODEL"
  --search_url "$SEARCH_URL"
  --limit "$LIMIT"
  --output "$OUTPUT"
  --concurrency "$CONCURRENCY"
)

if [ -n "$SERVER_URL" ]; then
  echo ">> Using OpenAI-compatible server at ${SERVER_URL} (e.g. mlx-lm)"
  EVAL_ARGS+=(--server_url "$SERVER_URL")
else
  EVAL_ARGS+=(--local --device "$DEVICE" --allow_unsafe_mps)
fi

if [ "$SMOKE" -eq 1 ]; then
  EVAL_ARGS+=(--max_tokens 256 --print_output --print_trace)
  echo ">> Smoke test: 1 example, --print_output and --print_trace enabled"
else
  EVAL_ARGS+=(--max_tokens 512)
fi

if [ "$RESUME" -eq 1 ]; then
  EVAL_ARGS+=(--resume)
fi

BACKEND_MSG="${SERVER_URL:-local (device=$DEVICE)}"
echo ">> Running bamboogle eval (model=$MODEL, limit=$LIMIT, backend=$BACKEND_MSG)..."
cd "$ROOT_DIR"
python3 -m examples.run_bamboogle_eval "${EVAL_ARGS[@]}"

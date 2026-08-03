#!/usr/bin/env bash
# Proves identity shapes results against a retrieval server that ignores
# filters (demo.py does), so enforcement is the web layer's, not the server's.
#
# Usage: examples/verify_identity_capabilities.sh
set -euo pipefail

for port in 8001 7860; do
  if lsof -i ":$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "FAIL: port $port is already in use; refusing to test against a stale server" >&2
    exit 1
  fi
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"; kill $(jobs -p) 2>/dev/null || true' EXIT

cat > "$WORK/corpus.jsonl" <<'JSON'
{"id": "pub_1", "title": "Zebra Handbook", "contents": "Zebra migration patterns.", "metadata": {"acl": ["public"]}}
{"id": "sec_1", "title": "Zebra Handbook", "contents": "Zebra confidential notes.", "metadata": {"acl": ["user:someone_else"]}}
JSON

PYTHONPATH=src:. python3 -m src.internal.servers.retrieval.demo \
  --corpus_path "$WORK/corpus.jsonl" --port 8001 >"$WORK/retrieval.log" 2>&1 &
env -u SEARCH_AGENT_MODEL PYTHONPATH=src:. \
  AGENTIC_SEARCH_WEB_DB_PATH="$WORK/web.db" \
  python3 -m uvicorn src.internal.servers.web.app:app \
  --host 127.0.0.1 --port 7860 >"$WORK/web.log" 2>&1 &

for _ in $(seq 1 60); do
  curl -sf -m 2 http://127.0.0.1:7860/admin/tools >/dev/null 2>&1 && break
  sleep 1
done

ask() {  # $1 = output file, $2... = extra curl args
  local out="$1"; shift
  curl -s -m 120 "$@" -X POST http://127.0.0.1:7860/api/agent \
    -H 'Content-Type: application/json' \
    -d '{"query":"Zebra Handbook"}' -o "$out"
}

ask "$WORK/anon.json"
curl -s -X POST http://127.0.0.1:7860/auth/register -H 'Content-Type: application/json' \
  -d '{"email":"dev@localhost","username":"dev","password":"devpass"}' >/dev/null
curl -s -c "$WORK/ck.txt" -X POST http://127.0.0.1:7860/auth/login \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=dev@localhost&password=devpass' >/dev/null
ask "$WORK/auth.json" -b "$WORK/ck.txt"

python3 - "$WORK/anon.json" "$WORK/auth.json" <<'PY'
import json, sys

def leaked(path):
    docs = json.load(open(path)).get("documents") or []
    return any("confidential" in (d.get("content") or "").lower() for d in docs)

anon, auth = (leaked(p) for p in sys.argv[1:3])
print(f"anonymous sees the restricted document: {anon}")
print(f"signed-in  sees the restricted document: {auth}")
if anon or auth:
    raise SystemExit("FAIL: a restricted document leaked")
print("PASS: neither identity can read another user's document")
PY

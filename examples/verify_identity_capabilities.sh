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
me_status=$(curl -s -o "$WORK/me.json" -w '%{http_code}' -b "$WORK/ck.txt" http://127.0.0.1:7860/me)
ask "$WORK/auth.json" -b "$WORK/ck.txt"

python3 - "$WORK/anon.json" "$WORK/auth.json" "$WORK/me.json" "$me_status" <<'PY'
import json, sys

anon_path, auth_path, me_path, me_status = sys.argv[1:5]

def leaked(path):
    docs = json.load(open(path)).get("documents") or []
    return any("confidential" in (d.get("content") or "").lower() for d in docs)

def reached_internal_corpus(path):
    docs = json.load(open(path)).get("documents") or []
    return any("migration" in (d.get("content") or "").lower() for d in docs)

# Positive control 1: the internal corpus (not an empty result set, and not
# an external-search fallback) actually answered the query. Without this, the
# leak check below passes vacuously whenever nothing reached the corpus at
# all -- which is exactly the failure mode this script exists to catch.
if not reached_internal_corpus(anon_path):
    raise SystemExit(
        "FAIL: internal-search control failed -- no document from the local "
        "corpus (expected content containing 'migration') was returned for "
        "the anonymous request; the leak check would be meaningless if the "
        "query never reached internal retrieval."
    )

# Positive control 2: the "signed-in" request is genuinely authenticated. If
# login silently failed, `ask ... -b ck.txt` is just a second anonymous
# request and the anon-vs-signed-in comparison proves nothing.
if me_status != "200":
    raise SystemExit(
        f"FAIL: auth control failed -- GET /me returned {me_status}, not "
        "200; login did not succeed."
    )
if not json.load(open(me_path)).get("id"):
    raise SystemExit(
        "FAIL: auth control failed -- GET /me returned 200 but no user id."
    )

anon, auth = (leaked(p) for p in (anon_path, auth_path))
print(f"anonymous sees the restricted document: {anon}")
print(f"signed-in  sees the restricted document: {auth}")
if anon or auth:
    raise SystemExit("FAIL: a restricted document leaked")
print("PASS: neither identity can read another user's document")
PY

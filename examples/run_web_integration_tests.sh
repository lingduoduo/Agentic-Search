#!/usr/bin/env bash
# Run the time-consuming web-app integration tests quickly.
#
# These tests spin up the full FastAPI app via `create_web_app()` and enter a
# `with TestClient(app)` block, which triggers the lifespan startup. When
# SEARCH_AGENT_MODEL (or SEARCH_AGENT_SERVER_URL) is set in .env, that lifespan
# loads the Search Agent model onto the device on *every* TestClient startup —
# turning a 3-second test file into a multi-minute (or, offline, hanging) run.
#
# None of these tests need the real model: they inject a MagicMock manager or
# exercise the no-model degradation path. So we disable the model load and the
# files run in seconds. This keeps the default `pytest` run fast — invoke this
# script on demand to cover the heavy web-app dispatch/fallback paths.
#
# Usage:
#   examples/run_web_integration_tests.sh                 # run the heavy files
#   examples/run_web_integration_tests.sh -v              # extra pytest args
#   examples/run_web_integration_tests.sh tests/unit/...  # override the file list

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Empty, exported values win over .env: create_web_app() calls
# load_dotenv(override=False), which never overwrites an already-set var. With
# these blank, the lifespan skips loading the Search Agent model entirely.
export SEARCH_AGENT_MODEL=
export SEARCH_AGENT_SERVER_URL=
# Belt-and-suspenders: never reach out to the Hugging Face Hub during tests.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Default to the three files that enter `with TestClient(app)` (lifespan-heavy).
# Any CLI args replace this list (or pass through as extra pytest flags).
DEFAULT_TARGETS=(
    tests/unit/test_execution_fallbacks.py
    tests/unit/servers/web/test_web_experience_app.py
    tests/unit/servers/web/test_tool_trace.py
)

if [ "$#" -gt 0 ]; then
    TARGETS=("$@")
else
    TARGETS=("${DEFAULT_TARGETS[@]}")
fi

echo ">> Search Agent model load disabled (SEARCH_AGENT_MODEL='')"
echo ">> Running: ${TARGETS[*]}"
exec python -m pytest "${TARGETS[@]}" -q -p no:cacheprovider

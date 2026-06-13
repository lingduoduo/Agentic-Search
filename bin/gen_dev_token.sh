#!/usr/bin/env bash
# Generate a short-lived dev JWT for local API testing.
#
# Usage:
#   source bin/gen_dev_token.sh          # exports $TOKEN into your shell
#   export TOKEN=$(bin/gen_dev_token.sh) # same effect, explicit form
#   bin/gen_dev_token.sh                 # prints the token and exits
#
# Reads .env automatically if present.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  . "$ROOT_DIR/.env"
  set +a
fi

_token=$(python3 -c "
from src.internal.auth import generate_user_jwt_token
print(generate_user_jwt_token(user_id='dev', email='dev@local'))
")

# When sourced, export TOKEN; when executed, just print it.
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
  export TOKEN="$_token"
  echo "TOKEN exported (${#TOKEN} chars)" >&2
else
  echo "$_token"
fi

#!/usr/bin/env bash
# Generate GRPO/PPO training data from QA benchmarks.
#
# Usage:
#   bin/generate_training_data.sh [--dataset DATASET] [--output DIR] [--preview]
#
# Datasets:
#   bamboogle  (default) — 125 two-hop questions from Bamboogle
#   nq                   — Natural Questions
#   trivia_qa            — TriviaQA
#   hotpotqa             — HotpotQA
#
# Output:
#   data/<dataset>_train/train.parquet
#   data/<dataset>_train/test.parquet
#
# Examples:
#   bin/generate_training_data.sh                          # bamboogle, full
#   bin/generate_training_data.sh --preview                # print 5 rows, no write
#   bin/generate_training_data.sh --dataset nq             # Natural Questions
#   bin/generate_training_data.sh --dataset bamboogle --max_examples 20

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

# Defaults
DATASET="bamboogle"
OUTPUT=""
PREVIEW=0
MAX_EXAMPLES=""
TEMPLATE="base"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)      DATASET="$2";      shift 2 ;;
    --output)       OUTPUT="$2";       shift 2 ;;
    --max_examples) MAX_EXAMPLES="$2"; shift 2 ;;
    --template)     TEMPLATE="$2";     shift 2 ;;
    --preview)      PREVIEW=1;         shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Map dataset name → HuggingFace config
case "$DATASET" in
  bamboogle) HF_CONFIG="bamboogle" ;;
  nq)        HF_CONFIG="nq" ;;
  trivia_qa) HF_CONFIG="trivia-qa" ;;
  hotpotqa)  HF_CONFIG="hotpot_qa" ;;
  *) echo "Unknown dataset: $DATASET (choose bamboogle, nq, trivia_qa, hotpotqa)" >&2; exit 1 ;;
esac

OUTPUT="${OUTPUT:-$ROOT_DIR/data/${DATASET}_train}"

ARGS=(
  --dataset_name  RUC-NLPIR/FlashRAG_datasets
  --dataset_config "$HF_CONFIG"
  --local_dir     "$OUTPUT"
  --data_source   "$DATASET"
  --template_type "$TEMPLATE"
)

if [ -n "$MAX_EXAMPLES" ]; then
  ARGS+=(--max_examples "$MAX_EXAMPLES")
fi

if [ "$PREVIEW" -eq 1 ]; then
  ARGS+=(--preview --preview_rows 5)
  echo ">> Preview mode — no files will be written"
else
  echo ">> Writing training data to $OUTPUT/"
fi

echo ">> Dataset: $DATASET ($HF_CONFIG)"
cd "$ROOT_DIR"
python3 -m examples.prepare_search_qa_dataset "${ARGS[@]}"

if [ "$PREVIEW" -eq 0 ]; then
  echo ">> Done. Files:"
  ls -lh "$OUTPUT/"

  echo ""
  echo ">> Sample records (first 3 rows):"
  python3 - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
parquet_files = sorted(output_dir.glob("*.parquet"))
if not parquet_files:
    print("No parquet files found.")
    sys.exit(1)

import pyarrow.parquet as pq

for parquet_path in parquet_files:
    table = pq.read_table(parquet_path)
    print(f"\n  [{parquet_path.name}]  {table.num_rows} rows")
    for i, row in enumerate(table.slice(0, 3).to_pylist()):
        prompt_content = row.get("prompt", [{}])[-1].get("content", "")
        target = row.get("reward_model", {}).get("ground_truth", {}).get("target", [])
        print(f"  [{i}] Q: {prompt_content[:80]!r}")
        print(f"       A: {target}")
PY
fi

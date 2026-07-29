"""Convert a BEIR benchmark dataset into the demo corpus.jsonl format.

The demo retrieval server (`src.internal.servers.retrieval.demo`) loads a
JSONL corpus where each line is:

    {"id": ..., "title": ..., "contents": ..., "metadata": {"acl": ["public"]}}

BEIR's loader hands back `corpus[doc_id] = {"title": ..., "text": ...}`, which
maps 1:1 onto that shape. This script downloads a BEIR dataset (cached under
--data_dir) and writes the converted corpus.

Usage:
    python3 examples/beir_to_corpus.py --dataset nfcorpus
    python3 examples/beir_to_corpus.py --dataset scifact --out data/corpus_scifact.jsonl

Then point the demo server at it:
    python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus_nfcorpus.jsonl

Install BEIR first:  pip install beir
"""

from __future__ import annotations

import argparse
import json
import os

# Datasets that ship a corpus.jsonl via the public BEIR mirror. Sizes are
# approximate doc counts, smallest first — nfcorpus downloads in seconds.
SUPPORTED = {
    "nfcorpus": "~3.6K docs (medical/nutrition)",
    "scifact": "~5K docs (scientific claims)",
    "scidocs": "~25K docs (citation prediction)",
    "arguana": "~8.7K docs (argument retrieval)",
    "fiqa": "~57K docs (financial QA)",
    "trec-covid": "~171K docs (COVID-19 literature)",
}

BEIR_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)


def _load_beir_corpus(dataset: str, data_dir: str) -> dict[str, dict]:
    """Download (if needed) and load a BEIR dataset's corpus."""
    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as exc:
        raise SystemExit(
            "This script requires the 'beir' package. Install with: pip install beir"
        ) from exc

    dataset_path = os.path.join(data_dir, dataset)
    if not os.path.exists(dataset_path):
        beir_util.download_and_unzip(BEIR_URL.format(name=dataset), data_dir)

    # qrels need a split; the corpus itself is split-independent, so any split loads it.
    corpus, _, _ = GenericDataLoader(dataset_path).load(split="test")
    return corpus


def convert(dataset: str, out_path: str, data_dir: str, limit: int | None) -> int:
    corpus = _load_beir_corpus(dataset, data_dir)
    written = 0
    with open(out_path, "w") as f:
        for doc_id, doc in corpus.items():
            if limit is not None and written >= limit:
                break
            title = doc.get("title", "") or ""
            contents = doc.get("text", "") or ""
            if not contents.strip():
                continue  # skip empty docs — they'd only ever score 0
            record = {
                "id": doc_id,
                "title": title,
                "contents": contents,
                "metadata": {"acl": ["public"]},
            }
            f.write(json.dumps(record) + "\n")
            written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a BEIR dataset into demo corpus.jsonl format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported datasets:\n"
        + "\n".join(f"  {name:14} {desc}" for name, desc in SUPPORTED.items()),
    )
    parser.add_argument(
        "--dataset",
        default="nfcorpus",
        help=f"BEIR dataset name (default: nfcorpus). Known: {', '.join(SUPPORTED)}",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (default: data/corpus_<dataset>.jsonl)",
    )
    parser.add_argument(
        "--data_dir",
        default="data/beir",
        help="Where BEIR downloads/caches datasets (default: data/beir)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of documents written (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or f"data/corpus_{args.dataset}.jsonl"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    count = convert(args.dataset, out_path, args.data_dir, args.limit)
    print(f"Wrote {count} documents to {out_path}")
    print(
        "Run the demo server with:\n"
        f"  python3 -m src.internal.servers.retrieval.demo --corpus_path {out_path}"
    )


if __name__ == "__main__":
    main()

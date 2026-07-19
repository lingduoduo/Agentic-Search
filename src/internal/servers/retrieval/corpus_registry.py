"""Corpus registry: resolve a corpus spec to a list of documents.

A spec is one of:
  - a registered name from data/corpora.json (e.g. "demo")
  - "all" — the union of every registered corpus, deduped by id
  - a comma-separated list of names (e.g. "demo,scifact")
  - a filesystem path to a .jsonl corpus (back-compat with --corpus_path)
"""

from __future__ import annotations

import json
import logging
import os

from src.internal.servers.retrieval.demo import _load_corpus

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = "data/corpora.json"


def load_manifest(path: str = DEFAULT_MANIFEST_PATH) -> dict[str, dict]:
    """Load the corpus manifest; a missing file yields an empty manifest so
    path-only (--corpus_path) usage keeps working with no manifest present."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _dedupe_by_id(docs: list[dict]) -> list[dict]:
    seen: set = set()
    out: list[dict] = []
    for d in docs:
        doc_id = d.get("id")
        if doc_id is not None and doc_id in seen:
            logger.warning("Dropping duplicate corpus id %r", doc_id)
            continue
        if doc_id is not None:
            seen.add(doc_id)
        out.append(d)
    return out


def resolve_corpus_docs(spec: str, manifest: dict | None = None) -> list[dict]:
    if manifest is None:
        manifest = load_manifest()

    if not spec:
        raise ValueError(
            "Empty corpus spec; pass a corpus name, 'all', or a file path."
        )

    if spec == "all":
        names = list(manifest.keys())
        if not names:
            raise ValueError("No corpora registered in the manifest for 'all'.")
    else:
        candidate = [s.strip() for s in spec.split(",") if s.strip()]
        if candidate and all(n in manifest for n in candidate):
            names = candidate
        elif os.path.exists(spec):
            docs = _load_corpus(spec)
            logger.info("Loaded corpus from path %s (%d docs)", spec, len(docs))
            return docs
        else:
            available = ", ".join(sorted(manifest)) or "(none)"
            raise ValueError(
                f"Unknown corpus spec {spec!r}. Available: {available}, or a file path."
            )

    docs: list[dict] = []
    for name in names:
        entry = manifest[name]
        path = entry["path"] if isinstance(entry, dict) else entry
        if not os.path.exists(path):
            logger.warning(
                "Corpus %r file %r is missing; skipping (regenerate via beir_to_corpus.py)",
                name,
                path,
            )
            continue
        docs.extend(_load_corpus(path))
    # Dedupe collapses only true duplicate ids across corpora (first occurrence
    # wins); a shared-id-namespace collision between unrelated corpora would
    # silently shrink the union.
    docs = _dedupe_by_id(docs)
    if not docs:
        raise ValueError(
            f"None of the requested corpora {names} have files on disk. "
            "Regenerate them via beir_to_corpus.py or provide a valid corpus path."
        )
    logger.info("Loaded corpora %s (%d docs after dedupe)", names, len(docs))
    return docs

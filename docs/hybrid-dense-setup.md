# Hybrid Retrieval — Dense Leg Setup & Troubleshooting

The hybrid retrieval server (`src/internal/servers/retrieval/hybrid.py`) fuses a **dense**
leg (e5 embeddings via `sentence-transformers`) with a **sparse** TF-IDF leg using Reciprocal
Rank Fusion. The sparse leg always works (pure scikit-learn). The dense leg needs a working
PyTorch + `sentence-transformers` stack.

If the dense leg can't load, the server **automatically degrades to TF-IDF-only** and keeps
serving on the same `/retrieve` contract — so a broken embedding stack is never fatal, it
just means you lose the dense half of the fusion. Pass `--no-dense` to skip it deliberately.

## What the dense leg needs

A clean install of the repo requirements is sufficient:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

This resolves a coherent set (verified working): `torch>=2.3`, `sentence-transformers>=3.0`,
`transformers` (4.x or 5.x both work in a clean env), with **no** stray `torchvision`. e5
(`intfloat/e5-base-v2`) loads and encodes on CPU or Apple Silicon MPS.

Then run the hybrid server:

```bash
python3 -m src.internal.servers.retrieval.hybrid --corpus_path data/corpus.jsonl
# first run downloads intfloat/e5-base-v2 (~440MB), cached afterwards
```

## Symptoms → cause → fix

| Log / error | Cause | Fix |
|---|---|---|
| `Could not import module 'BertModel'` / `'PreTrainedModel'` | `sentence-transformers` is older than `3.0` (e.g. a shared conda base pinned `2.7.0` by another project) and incompatible with the installed `transformers`. | Install into a **fresh venv** so `sentence-transformers>=3.0` is honored — don't run from a shared base env. |
| `operator torchvision::nms does not exist` | A stray `torchvision` whose version doesn't match the installed `torch` (pulled in by an unrelated package). `transformers` imports `torchvision` and the mismatched op blows up. | Use a fresh venv (no stray `torchvision`), or install a `torchvision` matched to your `torch`. The dense leg itself does **not** need `torchvision`. |
| `Dense leg unavailable, serving TF-IDF only …` in the server log | Any of the above — the server caught the failure and degraded. | Fix the env per this doc, then restart. Until then you get TF-IDF-only results. |

## Why a shared conda base often breaks this

A base conda environment shared across many projects accumulates conflicting pins — e.g. one
project pins `sentence-transformers<3.0`, another installs a `torchvision` matched to a
different `torch`. The repo's own `requirements.txt` is coherent; the conflict comes from the
*surrounding* packages. A per-repo virtualenv isolates this cleanly.

## Verifying the dense leg

```bash
python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('intfloat/e5-base-v2', device='cpu')
v = m.encode(['query: x', 'passage: y'], normalize_embeddings=True)
print('OK', v.shape)
"
```

If that prints `OK (2, 768)`, the dense leg will work; if it raises one of the errors above,
fix the environment first.

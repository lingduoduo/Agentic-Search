# Hybrid Retrieval — Dense Leg Setup & Troubleshooting

The hybrid retrieval server (`src/internal/servers/retrieval/hybrid.py`) fuses a **dense**
leg (e5 embeddings via `sentence-transformers`) with a **sparse** TF-IDF leg using Reciprocal
Rank Fusion. The sparse leg always works (pure scikit-learn). The dense leg needs a working
PyTorch + `sentence-transformers` stack.

If the dense leg can't load, the server **automatically degrades to TF-IDF-only** and keeps
serving on the same `/retrieve` contract — so a broken embedding stack is never fatal, it
just means you lose the dense half of the fusion. Pass `--no-dense` to skip it deliberately.

## What the dense leg needs

Any **dedicated project environment** with `sentence-transformers>=3.0` and a matched
`torch` works — a per-repo virtualenv or a project conda env both do. The thing to avoid is
running from a **shared base env** (e.g. the miniconda `base`) that other projects have
polluted with an old `sentence-transformers` or a mismatched `torchvision`.

If you already run the backend from a project conda env, use the **same** env for the hybrid
server:

```bash
conda run -n <your-env> python -m src.internal.servers.retrieval.hybrid --corpus_path data/corpus.jsonl
```

Or create a fresh virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
python3 -m src.internal.servers.retrieval.hybrid --corpus_path data/corpus.jsonl
# first run downloads intfloat/e5-base-v2 (~440MB), cached afterwards
```

Either way you get a coherent set (verified working): `torch>=2.3`, `sentence-transformers>=3.0`,
`transformers` (4.x or 5.x both work), with **no** stray `torchvision`. e5
(`intfloat/e5-base-v2`) loads and encodes on CPU or Apple Silicon MPS.

## Symptoms → cause → fix

| Log / error | Cause | Fix |
|---|---|---|
| `Could not import module 'BertModel'` / `'PreTrainedModel'` | `sentence-transformers` is older than `3.0` (e.g. a shared conda base pinned `2.7.0` by another project) and incompatible with the installed `transformers`. | Run from a **dedicated project env** (a project conda env or a fresh venv) where `sentence-transformers>=3.0` is honored — not the shared base env. |
| `operator torchvision::nms does not exist` | A stray `torchvision` whose version doesn't match the installed `torch` (pulled in by an unrelated package). `transformers` imports `torchvision` and the mismatched op blows up. | Run from a clean project env (no stray `torchvision`), or install a `torchvision` matched to your `torch`. The dense leg itself does **not** need `torchvision`. |
| `Dense leg unavailable, serving TF-IDF only …` in the server log | Any of the above — the server caught the failure and degraded. | Fix the env per this doc, then restart. Until then you get TF-IDF-only results. |

## Why a shared conda base often breaks this

A base conda environment shared across many projects accumulates conflicting pins — e.g. one
project pins `sentence-transformers<3.0`, another installs a `torchvision` matched to a
different `torch`. The repo's own `requirements.txt` is coherent; the conflict comes from the
*surrounding* packages. A dedicated per-project env — a project conda env or a per-repo
virtualenv — isolates this cleanly. (Don't confuse the two: the `base` env breaks, while a
project env created for this repo works.)

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

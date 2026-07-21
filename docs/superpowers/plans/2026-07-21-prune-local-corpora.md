# Plan: Prune unused local corpora + stale registry entry

1. Create branch `chore/prune-local-corpora`. → verify: `git branch --show-current`.
2. Delete local artifacts: `corpus_nfcorpus.jsonl`, `nq_search/`,
   `bamboogle_train/`, `indexes/`. → verify: files gone from `data/`; scifact kept.
3. Remove `nfcorpus` entry from `data/corpora.json`. → verify: JSON parses,
   keys == `[demo, scifact]`.
4. Confirm no code hardcodes the `nfcorpus` registry key. → verify: grep clean.
5. Commit registry change, push, open PR.

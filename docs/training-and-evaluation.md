# Training and evaluation

## Serving requests do not train models

`POST /api/agent` and `/api/agent/stream` perform routing, retrieval, tool execution, and model inference only. Even explicit `search_agent` or `tool_agent` mode loads a policy model for generation without updating its weights. A query containing `GRPO` is ordinary search input; it does not invoke the GRPO trainer. SFT, GRPO, and PPO run only through the offline commands in this guide. See [API request routing](request-routing.md) for serving-time dispatch.

Serving still uses indexes built offline by the `index_builder`. Filter-aware and degraded branches can use the composed session-aware retrieval, ranking/reranking, and evidence-grounded inference pipeline; strong unfiltered auto-search retains its direct ranking, sufficiency gate, and provider fallback path. Neither retrieval nor reranking is a training step. No new serving API or request/response schema was introduced; offline trainers may produce model artifacts, but requests only load and infer with them.

## Intent routing by nearest canonical example

**There is no intent training run any more.** The optional three-label (`chat`, `search`, `tool`) request router used to be a small MLP trained on generated examples. It is gone — module, checkpoint format, wordpiece bundle and all. What replaced it compares the incoming request against roughly 280 curated canonical examples and takes the route whose nearest examples are closest. The whole offline workflow is three commands, none of which trains anything:

```bash
# 1. Seed a canonical draft from existing labelled examples (optional; run once).
python -m src.model.intent_index_cli seed \
  --examples data/intent_examples.json --output data/intent_canonical.draft.json

# 2. Build the index the router serves from.
python -m src.model.intent_index_cli build \
  --canonical data/intent_canonical.json --output data/intent_index

# 3. Measure it. Never skip this after editing the canonical set.
python -m src.model.intent_index_cli evaluate \
  --index data/intent_index \
  --eval-queries data/intent_eval_queries.json \
  --hard-queries data/intent_eval_hard.json \
  --out-of-scope data/intent_out_of_scope.json \
  --canonical data/intent_canonical.json \
  --output data/intent_index/evaluation_report.json
```

`data/intent_canonical.json` and the evaluation JSON files are tracked (force-added under an otherwise gitignored `data/`). `data/intent_index/` is a regenerable local build artifact and is not.

### How routing works

Each canonical example is encoded once by `intfloat/e5-small-v2` and L2-normalized. At serving time the request is encoded the same way, and each route scores as the **mean of its top-k cosine similarities** to that request. The best route wins; the module labels reported alongside it are diagnostics and can never change the route.

**The prefix contract.** E5 models are trained with instruction prefixes and **degrade silently without them** — no error, no warning, just worse vectors. Every text gets `"query: "`, applied **symmetrically** to canonical anchors and to incoming requests: this is symmetric short-text similarity, not the asymmetric retrieval E5's `"passage: "` prefix is for. The prefix is a property of the encoder, not an argument, so it lives in `MODEL_PREFIXES` in `src/model/intent_encoder.py` and is applied inside `encode_texts` — no call site can forget it. An encoder with no registered prefix raises rather than defaulting to `""`: an unregistered model is far likelier to be one whose prefix nobody looked up than one that genuinely needs none, and guessing wrong is invisible.

**Every index built before this change is invalidated.** Rebuild with command 2 above. What catches a stale index is the **encoder-name check** — `IntentIndex` records the encoder that built it, and both `run_index_evaluation` and `ml_intent.load_intent_index` reject a mismatch by name — **not** a dimension mismatch, because `all-MiniLM-L6-v2` and `e5-small-v2` are **both 384-wide**. A stale index therefore loads, scores, and reports confident, meaningless numbers with no other symptom. That is not hypothetical: it happened once on this branch, and the name check is what turned it into one loud failure naming the rebuild command.

`k` (`TOP_K` in `src/model/intent_knn.py`) is `3` by default and is what serves today, but it is a parameter of `IntentIndex.decide()`, not a hardcoded constant — see [the ceiling finding, corrected](#the-ceiling-finding-corrected-top_k-was-never-swept) below for why that distinction matters and `AGENTIC_SEARCH_INTENT_TOP_K` in [Configuration](configuration.md) for the env var.

Two thresholds gate the answer, because two different things go wrong:

- `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE` (default `0.30`) — a low **absolute** similarity means nothing canonical resembles this request at all. That is an out-of-scope request.
- `AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN` (default `0.015`) — a small **gap** between the best and second-best route means two routes fit equally well. That is an ambiguous request.

Either one abstains, and abstention defers to the LLM classifier and the clarification path exactly as before. This is the concrete reason the softmax head was replaced: probabilities sum to one by construction, so a softmax router cannot express "none of these" — it can only say which of the three is least bad.

**These two thresholds are cosines, and their scale moves with the encoder** — see [the units trap](#the-units-trap). Measured on the e5 index: in-scope confidences span `0.792`–`0.905` and out-of-scope probes span `0.782`–`0.850`, so the `0.30` confidence floor **never fires for anything**, in scope or out, and no single value could separate the two overlapping ranges. All the abstaining is done by the margin, re-derived on the tuning slice for this encoder: at `0.015` the router serves 50 of the 111 test-slice queries (coverage `0.450`) at `0.960` served accuracy. The MiniLM-era `0.02` would have served only 38 of them.

### Changing routing behavior

Edit `data/intent_canonical.json` and rebuild. There is no training run, no seed, no schedule, no checkpoint. The canonical examples *are* the model.

Always **append, rebuild, re-measure**, in that order. A badly-phrased canonical example does not fail loudly; it becomes a bad attractor that quietly pulls every nearby query onto its route. The only thing that catches it is the evaluation report, so a canonical edit that has not been re-measured has not been verified. `tests/unit/test_intent_canonical_data.py` additionally guards the set's size band, per-route balance, per-module support, and internal near-duplication; `tests/unit/test_intent_index_eval.py` pins the measured test-slice accuracy, out-of-scope AUC, and latency bars. All of those need `sentence-transformers` and a built index, so they skip in CI — they guard your local rebuild, not the pull request.

### The tuning/test split

Three hyperparameters need values — `top_k`, `min_confidence`, `min_margin` — and **none of them may be chosen on the queries the result is reported from**. Two are swept on the tuning slice; `top_k` is pinned at the shipped `3` and never swept for selection, which is what makes the reported accuracy independent of the sweep entirely. `src/model/intent_eval_split.py` enforces the rest:

- **Tuning slice, 70 queries** — all 30 `eval-` legacy queries (already contaminated: they were used as feedback while curating the canonical set, so they are worthless as a gate and free to spend here) plus a route-stratified sample of **40** of the 151 clean `bulk-` queries, drawn with **seed `17`**.
- **Test slice, 111 queries** — every clean query the tuning sample did not take. Untouched by every sweep. Plus `hard-40`, which is likewise never tuned on.

The split is deterministic in the seed alone (input order is normalized first), and the two slices are an exact partition of the eval set. **Every accuracy quoted below as a result comes from queries that no hyperparameter ever saw.** `evaluation_report.json` labels every block with `tuned_on: true|false`; only `tuned_on: false` numbers are results.

**The earlier `0.8287` figure for this encoder was fitted.** It came from exploratory probes where `k` was chosen *after* seeing the results, on the same queries used to report them. The properly-split number below is lower. That is what a fitted number does when you stop fitting it, and it is the entire point of the split.

### What it scores, and why it is still dark

Re-measured 2026-08-13 against the same committed canonical set (280 anchors) and the same eval files — only the encoder changed. **The instrument is not identical**: the reported slice is now the 111-query test slice rather than the 151-query clean set, because 40 clean queries are spent on tuning. Every row below is therefore quoted **on the same slice for both encoders**, re-measuring MiniLM where needed rather than carrying a number across slices.

Unless a row says otherwise, "test slice" means the 111 queries left after the split (**seed 17**, tuning 70 / test 111), and the out-of-scope rows score those 111 against the 24 probes.

| Measure | e5-small-v2 (now) | all-MiniLM-L6-v2 (before) |
|---|---|---|
| **Route accuracy, test slice (111 queries; seed 17, tuning 70 / test 111)** | **0.7928** | **0.6216** (re-measured on the same 111) |
| — the same, on the older clean_151 instrument | `0.7881` | `0.6225` (as published by #511) |
| hard_40 (adversarial, never tuned on) | `0.6750` | `0.6250` |
| Out-of-scope **AUC**, test_111 vs. 24 probes | `0.8551` | **`0.8848`** (re-measured on the same 111) |
| — the same, on clean_151 vs. 24 probes | `0.8626` | `0.8681` |
| Out-of-scope Cohen's d, test_111 vs. 24 probes | `1.4747` | `1.6208` (clean_151: `1.5232` vs `1.4934`) |
| Leave-one-out over the 280 canonical anchors (diagnostic, never a selector) | `0.7393` (207/280) | `0.6750` (189/280) |
| p50 / p95 routing latency, encode + decide | `9.73ms` / `11.47ms` | `5.51ms` / `5.88ms` |
| Out-of-scope raw margin, test_111 *(encoder-specific — do not compare across this row)* | `0.0280` | `0.1188` (clean_151, as published) |
| Module macro-F1 / joint accuracy (diagnostic; both on the mixed `bulk_181`, like-for-like with the before column — the test slice alone gives `0.3492` / `0.0`) | `0.3535` / `0.0` | `0.3471` / `0.2318` |
| Per-route accuracy, test slice (111) | search 25/37, chat 33/37, tool 30/37 | chat 24/51, search 25/50, tool 45/50 (on clean_151) |
| Serving hyperparameters | `top_k=3`, `min_confidence=0.30`, `min_margin=0.015` | `top_k=3`, `min_confidence=0.30`, `min_margin=0.02` |

**What `0.7928` is, precisely.** It is **argmax route accuracy with no abstention**: the fraction of the 111 test queries whose best-scoring route is the right one, counted whether or not the thresholds would have served an answer. It is *not* the accuracy a caller sees. With the shipped `min_margin=0.015` the router **serves 50 of those 111 (coverage `0.450`) at `0.960` served accuracy**, and defers the rest to the LLM classifier. Argmax is the number the decision rule was written against and the only one comparable to the `0.6225`/`0.6216` before-figures, which are argmax too; the served pair is what promotion would actually deliver. Both are reported for every slice in `evaluation_report.json`.

For older context, on the retired clean-151 instrument the previous MLP scored `0.4768`, the production regex cascade `0.4238`, and the majority-class floor `0.3377`. Those are different queries under a different encoder — context, not a like-for-like comparison with the column above.

**+0.17 on route accuracy, measured against MiniLM on the identical 111 queries, at roughly 2x the latency and well inside the 25ms ceiling.** The instrument is smaller than #511's, not harder or cleaner: MiniLM scores `0.6216` on these 111 against `0.6225` on the clean 151, so the slice itself is of ordinary difficulty and the whole gain is the encoder's. The decision rule fixed in advance had three bands: `≥ 0.80` clears the promotion bar, `0.75`–`0.80` is a real improvement, below `0.75` is a hard stop. `0.7928` lands in the **middle band** — a real improvement over `0.6225`, short of the promotion bar. **The artifact stays dark.** `AGENTIC_SEARCH_INTENT_INDEX_PATH` remains unset by default and every request falls through the existing LLM/rule cascade. Promotion is a separate change, reviewed on its own terms.

One result did **not** improve, and it matters more than the headline:

**Out-of-scope separability got worse, not better: AUC `0.8551` against MiniLM's `0.8848` on the same 111 queries — `−0.0297`.** On the older clean_151 slice the same comparison is `0.8626` vs `0.8681` (`−0.0055`), so the regression looks small there and is five times larger on the slice actually reported from; quote the matched pair, never one number from each slice. Either way it misses the `0.90` bar this change set for itself. The `0.927` AUC that made e5 look better at abstaining was measured on **e5-*base*-v2**, a different, larger model that is explicitly out of scope here; the fitted e5-*small* probe scored `0.871`, already under the bar. Abstention is the safety property of this router, so this looked like the single strongest argument against promoting it.

**It does not survive being measured at the operating point** — e5 makes 2 wrong routes against MiniLM's 21 at each model's own tuned threshold, and 7 against 21 at matched coverage. AUC ranks over the whole score range; the margin gate only needs separation at the boundary. See [The out-of-scope regression, measured where it bites](#the-out-of-scope-regression-measured-where-it-bites) below. The `0.90` bar is still missed and the ranking regression is still real — what changed is that neither costs anything at any threshold this router would run at.

**The threshold grid had to be re-derived, because it was in MiniLM's units.** The original grid started at `min_margin=0.02`, which under e5 abstains on more than half the tuning slice; no combination cleared the sweep's `coverage ≥ 0.60` floor, so the first run selected nothing at all. The grid's low end is now derived from the **tuning slice's own margin quantiles** under e5 (min `0.0008`, p25 `0.0116`, median `0.0188`, p75 `0.0280`, max `0.0676`), and it selects `min_margin=0.015` — the shipped default. **Derive any future re-tuning from the tuning slice too. Never from the test slice**, whose quantiles (median `0.0129`) are a different distribution and are off-limits: reading them to choose a threshold is test-set fitting even when no code does it.

**Why widening that grid is safe, and would not have been before.** `top_k` is **not** swept for selection — it is pinned at the shipped `3`, and `_select_thresholds` searches only `(min_confidence, min_margin)`. That matters because the reported headline is argmax accuracy, which is abstention-blind: it depends on `top_k` and on nothing else the sweep chooses. A sweep that also chose `k` would couple a tuning-slice search to the held-out number, and widening it after seeing that number could move the number. With `k` pinned, **no threshold this sweep selects can change `test_slice.accuracy` by any amount** — a property, not a promise, and the one `test_the_threshold_sweep_never_chooses_top_k` guards. Choosing a different `k` remains possible, but it is a deliberate, separately-reviewed decision, not a side effect of re-tuning thresholds. `min_module_score` is still un-derived under e5 and is dead at `0.45`; it needs the same treatment before promotion.

**Two caveats on the separability numbers, neither fixed here:**

1. **`separability_report`'s Cohen's d is comparable only to other numbers from that function.** Its pooled SD averages the two groups' *population* variances (dividing by `n`) rather than the textbook `(n−1)`-weighted form. The difference is small at these sample sizes but real, so this `1.475` must not be set beside a Cohen's d computed by scipy or any stats package.
2. **The headline AUC and Cohen's d are not fully held out.** The same 24 out-of-scope probes both tie-break the tuning sweep's hyperparameter selection *and* denominate the reported separability. The in-scope side is the untouched test slice, but the out-of-scope side is not held out from everything upstream of it.

Also note the instrument itself: 111 test queries is a small slice, and it is deliberately smaller than the 151 the "before" column used — that is the honest cost of holding 40 queries back for tuning, and it widens the confidence interval on every number in the column.

### The out-of-scope regression, measured where it bites

The AUC row above says e5 is *worse* at separating out-of-scope requests (`0.8551` against MiniLM's `0.8848` on the same slice), and #512 called that the single strongest argument against promoting it. **Measured at the operating point each encoder would actually run at, the ordering reverses.**

AUC is a threshold-free ranking statistic over the whole score range. Serving does not use the whole range — the margin gate needs separation only near the decision boundary, and abstaining costs an LLM fallback rather than a wrong answer. So the number that matters is not "how well does the score rank in-scope above out-of-scope", but **how much traffic is answered, and how often those answers are wrong.**

Each encoder's `min_margin` is tuned on the **tuning** slice and reported on the **test** slice — a threshold chosen on the reported queries would flatter whichever encoder it was chosen for.

| | all-MiniLM-L6-v2 | e5-small-v2 |
|---|---|---|
| tuned `min_margin` (on the tuning slice) | `0.030` | `0.015` |
| coverage, test slice | `0.6667` (74/111) | `0.4505` (50/111) |
| served accuracy | `0.7162` | **`0.9600`** |
| **wrong routes** | **21** | **2** |

At its own tuned point e5 answers less and is right far more often: **2 wrong routes against 21**. Because the two points serve different volumes, the same comparison at **matched coverage** — e5 loosened to `min_margin 0.008`, answering 73 queries against MiniLM's 74:

| | all-MiniLM-L6-v2 @ `0.030` | e5-small-v2 @ `0.008` |
|---|---|---|
| coverage | `0.6667` (74/111) | `0.6577` (73/111) |
| served accuracy | `0.7162` | `0.9041` |
| **wrong routes** | **21** | **7** |

**A 3× reduction in misroutes at the same answered volume, and 10× at each model's own tuned point.** The AUC regression is real as a ranking property and does not bite at any threshold either model would run at. The reason is that e5 compresses cosine similarities into a narrow high band — which costs global ranking across the full score range while leaving local separation at the boundary cleaner.

Reproduce with:

```bash
python -m examples.measure_intent_operating_point
```

**What this does and does not settle.** It removes the safety argument against promotion: e5 is more accurate *and* misroutes less at every comparable point. What it leaves is a cost question — e5 at its tuned point defers 61 of 111 queries to the LLM classifier against MiniLM's 37, which is more latency and more spend per request. That is a budget decision, not a correctness one.

Two limits worth stating. The slice is 111 queries, so `2` versus `21` has a wide interval on the low end — the direction is solid, the ratio is not precise. And `min_margin 0.012` scores the same 2 wrong routes as the shipped `0.015` while covering 6 more queries, which looks strictly better — but that was read off the **test** slice, so it is a fitted observation and must be re-derived on the tuning slice before it ships.

### The ceiling finding, corrected: `TOP_K` was never swept

Leave-one-out accuracy over the canonical set at the shipped `TOP_K=3` is `0.6750`: scoring each of the 280 anchors against the other 279 with the same top-k-mean rule, a third of them cannot recover their own route. An earlier version of this document read that `0.6750` as a representation ceiling — "`all-MiniLM-L6-v2` sentence embeddings with top-3-mean cosine top out near `0.67`–`0.70` no matter how good the examples get" — and named a stronger encoder as the only remaining lever. **That was overstated.** `TOP_K = 3` is an arbitrary constant that was never swept, and sweeping it — same encoder, same 280 anchors — moves both numbers substantially:

| `top_k` | clean_151 accuracy | hard_40 accuracy | out-of-scope separation margin | leave-one-out accuracy |
|---|---|---|---|---|
| **3 (shipped)** | 0.6225 | 0.6250 | **0.1188** | 0.6750 |
| 5 | 0.6358 | 0.6000 | 0.1036 | 0.6893 |
| 8 | 0.6556 | 0.6500 | 0.0918 | 0.7036 |
| **15** | **0.6887** | 0.6500 | 0.0767 | **0.7464** |
| 25 | 0.6755 | 0.6250 | 0.0654 | 0.7643 |

At `k=15`, leave-one-out reaches `0.7464` and clean_151 accuracy reaches `0.6887` — both well above the shipped `k=3` numbers on the same encoder and the same anchors. The `0.6750` figure this document previously called a ceiling reflects `TOP_K`'s arbitrary value at least as much as it reflects the encoder's representation.

**The trade, plainly stated:** out-of-scope separation falls from `0.1188` to `0.0767` as `k` rises from 3 to 15, because averaging more neighbors lifts the confidence floor for every route — including routes the request has nothing to do with — so accuracy and abstention pull against each other. There is no `k` that improves both at once in this table.

**The caveat:** the gap between `k=8` (`0.6556`) and `k=15` (`0.6887`) on clean_151 is about five queries out of 151. That is well within the noise a single held-out slice can produce, so `k` must be chosen on a validation split with the accuracy/abstention trade decided deliberately — not read off this table as if it named a single best value.

**What survives:** the hard stop still stands at every `k` tested. `bulk_181` — the decision-rule input, not clean_151 — lands near `0.72` at `k=15`, still under the `0.75` promotion bar. A stronger encoder and a swept `k` both remain live levers; `k` is the cheaper one to try first, because it costs a config change and a re-run of the evaluation CLI, not a new model.

**How `k` is chosen now.** That table above is MiniLM's, and it is history: it was read off a fitting curve computed over the very queries it reported. Since the e5 swap, `_sweep_top_k`'s table is computed on the **tuning slice**, never the test slice, so that reading the report cannot hand test-set fitting back to a human — and `k` is **deliberately not** something `_select_thresholds` picks, for the reason given [above](#what-it-scores-and-why-it-is-still-dark). It stays at the pre-existing `3`; moving it is a separate decision made on its own terms. For reference, e5's tuning-slice curve — **tuning numbers, not results**:

| `top_k` | tuning accuracy (tuned-on) | leave-one-out | raw separation margin *(encoder-specific)* |
|---|---|---|---|
| **3 (shipped)** | 0.8286 | 0.7393 | 0.0416 |
| 5 | 0.8571 | 0.7964 | 0.0392 |
| 8 | 0.8714 | 0.8143 | 0.0364 |
| 15 | 0.8714 | 0.8429 | 0.0328 |
| 25 | 0.8714 | 0.8500 | 0.0304 |

The same accuracy-versus-abstention trade holds under e5, and leave-one-out keeps climbing monotonically to `k=25` — past where tuning accuracy plateaus at `0.8714` from `k=8` on — which is exactly why it is reported and never used to select. Note that no column here is reliably monotonic in `k`; nothing about a larger `k` is guaranteed to be better. `hard_40` (adversarial, held-out) is deliberately not shown here: `evaluation_report.json`'s `top_k_sweep` never publishes a per-`k` curve over held-out data, only over the tuning slice and out-of-scope probes.

### Known limitations

- **`top_k` is a live parameter with a measured trade, not a settled constant.** `TOP_K = 3` ships unchanged, but it is now a parameter of `IntentIndex.decide()` (`AGENTIC_SEARCH_INTENT_TOP_K`). [The corrected ceiling finding](#the-ceiling-finding-corrected-top_k-was-never-swept) above shows the trade under **MiniLM**: sweeping `k` reaches `0.7464` leave-one-out and `0.6887` clean_151 accuracy at `k=15`, against `0.1188` out-of-scope separation falling to `0.0767`. Under **e5**, the same `k=15` reaches `0.8429` leave-one-out (see the tuning-slice curve above — a different encoder, not comparable number-for-number). `evaluation_report.json`'s `top_k_sweep` carries the full table for `k ∈ {3, 5, 8, 15, 25}`. The tuning/test split now exists, and `k` still has not been chosen on it — that decision remains deferred, alongside whatever encoder change is tried next.
- **Topical concentration.** 47% of the canonical examples carry IR/ML vocabulary, because they were curated from this project's own example set. Of 16 held-out in-scope probes drawn from outside that vocabulary, **13 abstained** rather than routing at all, and **9 had a wrong best-guess route underneath**. The two counts overlap and are not a partition — an abstaining query still has a nearest route; it just does not clear the thresholds. The failure is safe, because abstention defers to the LLM classifier, but off-domain traffic abstains more often than it should.
- **The compose-versus-dispatch boundary.** "write an email to the vendor about the overage" sits at `0.963` cosine (MiniLM-measured) to the canonical "email the vendor about the overage". One verb apart, so it routes to `tool` without abstaining, when composing text is arguably `chat`. Adding more compose anchors did not fix it: the two phrasings are near-identical to the encoder and genuinely ambiguous to a human reader.
- **Route imbalance, and it moved with the encoder.** Under MiniLM the `tool` route scored 45/50 on the clean slice while `search` and `chat` sat near 25/50. Under e5 the imbalance inverts: on the test slice `chat` is 33/37 and `tool` 30/37, while **`search` is the weak route at 25/37**. Route-level error is therefore a property of the encoder's representation at least as much as of the route, and any conclusion drawn from one encoder's per-route table does not carry to another's.
- **Module emission collapsed to "everything on the route."** `AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE` (`0.45`, a cosine) is below *every* module score e5 produces, so `_emit_modules` emits every well-supported module of the winning route. Module recall goes to ~1.0, precision to ~0.2, and joint accuracy to `0.0` (from `0.2318` under MiniLM). Modules are diagnostics and can never change the route, so nothing routes worse for it — but the module fields in the report are currently uninformative, and the threshold needs re-deriving in e5's units with the other two.
- **Margin abstentions, module labels, and the composite flag are invisible to production telemetry.** They are recorded through `request_capture`, which only runs under the debug panels; `route_request`'s `telemetry` argument (the one persisted with the session in production) never receives `modules` or `composite`. Composite detection exists precisely to give a future plan-aware router measured data, so this means it is currently unmeasurable in production. Measuring any of this in production would need a `predict_route` or `route_request` signature change.
- **The evaluation set is partly contaminated, and 40 more queries are now spent on tuning.** The legacy 30 were used as feedback while the canonical set was being curated, so their score is optimistic and must never be quoted as the router's accuracy; they are spent on the tuning slice for exactly that reason. A further 40 clean queries join them there, which leaves **111** for the honest measurement — a smaller instrument than the 151 the MiniLM numbers used, and the reason those two numbers are not the same measurement even where they look comparable.
- **The pinned bars are local-only.** `tests/unit/test_intent_index_eval.py` and the near-duplicate bar in `tests/unit/test_intent_canonical_data.py` need `sentence-transformers` and a built `data/intent_index/`, neither of which exists in CI, so they **skip on every pull request**. They catch a regression only for whoever runs them locally after a rebuild. A CI eval-gate job that would close that hole remains out of scope.
- **A near-duplicate pair is below the re-derived near-duplicate ceiling and is invisible to the near-duplicate test.** "there was a policy about retaining user transcripts" and "what does the retention policy say about transcripts" (both route `search`) score `0.9340` under e5 — the measured maximum over all 39,060 canonical pairs, and comfortably under `_MAX_INTERNAL_COSINE = 0.95` in `tests/unit/test_intent_canonical_data.py`, so no test currently flags it (the old MiniLM-scale bar sat much lower and would have caught it, but that bar was not meaningful under e5's compressed cosine range and had to be re-derived to `0.95`). It is a genuine near-duplicate — two phrasings of one request — and belongs de-duplicated in the canonical set, not caught by tightening this constant further, since `0.95` is already this repo's duplicate threshold elsewhere (`LEAKAGE_COSINE`). Whoever de-duplicates it should re-measure the pair distribution afterwards and tighten the ceiling if the new maximum allows.

### Deploying an index

The loaded index is cached by resolved path and is never invalidated, so: **rebuild the index, then restart the web process.** A *failed* load is cached too — starting the web process before the index exists leaves learned routing disabled until the next restart, even after the file appears.

**An index built before the e5 swap must be rebuilt**, or the encoder-name check disables the route on load — deliberately, since both encoders are 384-wide and the alternative is silently meaningless routing.

The e5-small-v2 encoder itself loads lazily on the first auto-routed request, separately from the index, and blocks that request for roughly two seconds while the model loads. This is not the promotion-gate activation checklist above; it is a separate one-time cost the first caller pays. A failing model fetch (missing weights, unreachable HuggingFace) is cached as a failure the same way the index's failed load is: the route disables itself and every later request degrades straight to the LLM classifier instead of retrying the download per request.

### The units trap

`AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE` is a **cosine similarity, not a softmax probability**, and **its scale has changed again with the encoder**. Three scales in two changes:

| | in-scope confidence, typical | a plausible-looking `0.60` would |
|---|---|---|
| retired MLP (softmax) | above `0.9` | pass almost everything |
| all-MiniLM-L6-v2 (cosine) | mean `0.378` | abstain on almost every request |
| **intfloat/e5-small-v2 (cosine)** | **`0.792`–`0.905`, probes `0.782`–`0.850`** | **pass everything, in scope or not** |

A value carried over from either earlier configuration is meaningless. Under e5 the shipped `0.30` is so far below the whole distribution that the confidence gate never fires at all, and no single value can separate in-scope from out-of-scope requests, because the two ranges overlap. The abstention that survives is the margin gate: `AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN` is re-derived for this encoder at `0.015` (serving 50 of 111 test-slice queries at `0.960`), where the MiniLM-era `0.02` would have served 38. `AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE` is cosine-scaled the same way, has shifted the same way, and is **not** yet re-derived — at `0.45` it is below every module score e5 emits.

Re-derive rather than reuse — and re-derive on the **tuning slice**, never on the test slice. `evaluation_report.json` carries the full 54-row sweep (6 confidence values × 9 margin values) under `threshold_tuning.sweep`, with the winner — `min_margin=0.015`, shipped as the app default — under `threshold_tuning.selected`. The sweep's own margin grid was already re-derived for e5's scale (see [above](#what-it-scores-and-why-it-is-still-dark)); a future encoder swap will need the same re-derivation, made and reviewed **before** looking at what it does to the headline, before the grid can select anything meaningful again.

`AGENTIC_SEARCH_INTENT_MODEL_PATH` no longer exists. Serving reads `AGENTIC_SEARCH_INTENT_INDEX_PATH`, a directory holding an `index.npz`. Building or evaluating an index never changes a serving setting.

[← Back to README](../README.md)

This guide covers dataset preparation, supervised and reinforcement-learning workflows, and benchmark evaluation.

## Runnable examples

### Agent CLI

| Mode | Loop | Needs retrieval server | Use it for |
|------|------|------------------------|------------|
| `single` | `PlainGenerationLoop` | No | Local generation smoke tests |
| `search` | `SearchAgentLoop` | Yes | Multi-turn RAG, SFT, and RL traces |
| `tool` | `ToolAgentLoop` | Yes | Structured tool-calling experiments |

```bash
# single — no retrieval server needed (plain generation)
# Apple Silicon: use --device mps --allow_unsafe_mps for ~50x faster inference
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device mps --allow_unsafe_mps \
  --allow_remote_model_downloads

# single with retrieval server — small models (≤3B) use --mode single; search/tool require 7B+ to emit structured tags
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device mps --allow_unsafe_mps \
  --search_url http://localhost:8001/retrieve --allow_remote_model_downloads

# search — 3B is the Mac sweet spot (~6 GB unified memory); 7B needs 16 GB+ and will swap
python3 -m examples.run_agentic_search \
  --mode search --question "What is RAG?" \
  --model Qwen/Qwen2.5-3B-Instruct --local --device mps --allow_unsafe_mps \
  --search_url http://localhost:8001/retrieve --allow_remote_model_downloads

# search — server-backed, requires vLLM on :8080 and retrieval on :8001
python3 -m examples.run_agentic_search \
  --mode search --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 --search_url http://localhost:8001/retrieve
```

### Intent operating point

Compares the serving encoder against the previous one at the threshold each would actually run at — coverage, served accuracy, and wrong-route count — rather than at the abstention-blind argmax the evaluation report headlines. Tunes each encoder's `min_margin` on the tuning slice and reports on the test slice. This is the evidence behind [The out-of-scope regression, measured where it bites](#the-out-of-scope-regression-measured-where-it-bites).

```bash
python -m examples.measure_intent_operating_point
```

Needs sentence-transformers and both models; MiniLM is pulled only for the comparison. Takes about a minute on CPU.

### Bamboogle evaluation

Always requires the retrieval server on port 8001.

```bash
# Smoke test — local model, 1 example, full trace printed
python3 -m examples.run_bamboogle_eval \
  --model Qwen/Qwen2.5-3B-Instruct --local --device mps --allow_unsafe_mps \
  --search_url http://localhost:8001/retrieve --limit 1 --print_trace \
  --allow_remote_model_downloads

# Full benchmark — Apple Silicon, requires SERP_API_KEY in .env
bin/run_bamboogle_eval.sh --limit 125
```

### PPO/GRPO reward

```bash
python3 -m examples.run_grpo_training_pipeline         # end-to-end reward + GRPO smoke test (no GPU, no model)

# Simulated-judge GRPO — actually updates a policy: bamboogle prompts → generate →
# SimulatedPreferenceJudge → GRPO step. No retrieval server; runs on CPU/MPS.
python3 -m examples.run_bamboogle_grpo_train \
  --model Qwen/Qwen2.5-0.5B-Instruct --device cpu \
  --allow_remote_model_downloads --steps 10
```

### Search pipeline with access filters

No live model or retrieval server is required.

```bash
python3 -m examples.run_search_pipeline
```

## Dataset preparation

```bash
# Offline local RAG smoke test (4 examples, existing 30-document demo corpus)
python3 -m examples.prepare_local_rag_smoke_dataset --topk 1 --preview

# Write compact RAG parquet after inspecting the preview
python3 -m examples.prepare_local_rag_smoke_dataset \
  --topk 1 --output_path data/local_rag_smoke.parquet
```

This command requires no retrieval server, network access, FlashRAG dataset, or retrieval caches.

Optional large-dataset workflows:

```bash
# Optional: Search-QA parquet preparation
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq --local_dir data/nq_search

# Preview before writing
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq \
  --splits test --max_examples 20 --preview --preview_rows 5

# Optional: full NQ RAG parquet preparation
# Requires an external Wikipedia corpus plus retrieval-cache JSON files keyed
# by NQ question. prepare_search_qa_dataset does not create these inputs.
python3 -m examples.prepare_search_rag_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq \
  --corpus_path data/wiki-18.jsonl \
  --train_retrieval_cache data/nq_train_retrieval_cache.json \
  --test_retrieval_cache data/nq_test_retrieval_cache.json \
  --topk 3 --local_dir data/nq_rag
```

## Training

The training pipeline is modular: generate trajectories → score with rewards → compute advantages → optimize.

| Task | Entry point |
|------|-------------|
| QA parquet preparation | `python3 -m examples.prepare_search_qa_dataset` |
| Training data (shell) | `bin/generate_training_data.sh` |
| Reward/GRPO smoke test | `python3 -m examples.run_grpo_training_pipeline` |
| Bamboogle benchmark eval | `python3 -m examples.run_bamboogle_eval` / `bin/run_bamboogle_eval.sh` |
| Reward function | `src/training/reward.py` |
| Simulated preference judge | `src/training/judge.py` |
| GRPO helpers | `src/training/grpo.py` |
| Online GRPO for HF LMs | `src/training/ppo/llm_grpo_trainer.py` |
| Agent-loop GRPO (full reward) | `src/training/ppo/search_agent_grpo_trainer.py` |
| PPO core | `src/training/ppo/core_algos.py` |
| Generation and policy loss | `src/model/generation.py` |
| Feedback-driven GRPO | `python3 -m examples.run_feedback_grpo` |
| SFT warm-start + GRPO | `python3 -m examples.run_sft_grpo` |
| Simulated-judge GRPO (policy update) | `python3 -m examples.run_bamboogle_grpo_train` |

### Fine-tune from user feedback

Train directly on thumbs-up/down sessions collected via `POST /api/feedback` (no GPU required for the smoke path; `--device mps` on Apple Silicon):

```bash
# Feedback-driven GRPO: load rated sessions from the web DB → reward with human_signal → update
python3 -m examples.run_feedback_grpo \
  --db_path data/feedback.sqlite3 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --min_ratings 10 --human_feedback_weight 0.5 \
  --num_rollouts 4 --search_url http://localhost:8001/retrieve --device mps \
  --output_dir data/checkpoints/feedback_grpo/

# SFT warm-start (Phase 1, assistant-token-only CE on thumbs-up traces) then GRPO (Phase 2);
# --sft_epochs 0 skips Phase 1 and runs pure GRPO from the base model
python3 -m examples.run_sft_grpo \
  --db_path data/feedback.sqlite3 --model Qwen/Qwen2.5-1.5B-Instruct \
  --jsonl_path data/sft_pairs.jsonl \
  --sft_epochs 3 --sft_lr 2e-5 --sft_output_dir data/checkpoints/sft_warmstart/ \
  --grpo_output_dir data/checkpoints/sft_grpo/ --device mps
```

`load_feedback_examples` raises if fewer than `--min_ratings` rated sessions exist, so collect feedback first (thumbs in the UI, or `POST /api/feedback`). There is **no HTTP training endpoint** — fine-tuning is offline by design; the only backend endpoint in this loop is `POST /api/feedback` (see [Web Backend API](api-reference.md#web-backend-api)).

### Reward components

`SearchRewardFunction` uses these components:

| Component | Config field | What it measures |
|-----------|-------------|-----------------|
| Correctness | `correctness_weight` | Judge score against gold answer (EM / contains-match) |
| Citation support | `citation_support_weight` | Fraction of retrieved docs cited in the final answer |
| Subquestion coverage | `subquestion_coverage_weight` | Fraction of sub-questions with sufficient evidence |
| Search quality | `search_quality_weight` | Evaluator verdict + per-query search quality |
| Unnecessary search | `unnecessary_search_penalty` | Penalty per search round beyond the first |
| Unnecessary fetch | `unnecessary_fetch_penalty` | Penalty per fetched page not cited in the answer |
| Fetch usefulness | `fetch_usefulness_reward` | Bonus when fetched pages are cited in the final answer |
| Format compliance | `format_reward_weight` | Structural compliance in the final answer |
| Human feedback | `human_feedback_weight` | `human_signal` (±1.0) from thumbs-up/down sessions; `0.0` by default (off) |

Reward preset names: `sparse_final_only` | `simple_sparse_with_search_penalty` | `second_pass` | `third_pass_with_format` | `retriever_aware` (see `SearchRewardConfig` in `src/training/reward.py`). The Bamboogle eval CLI (`run_bamboogle_eval --reward_preset`) exposes the shorthand `sparse_final_only | simple_sparse | second_pass | third_pass`, which map to the first four config presets; `retriever_aware` is config-only.

**The judge.** The `correctness` term is `judge_fn(answer, gold)`. The example trainers pass `simple_sparse_correctness_reward` (exact-normalized match → 1.0, gold contained in prediction → 0.7, else 0.0), which is what the "EM / contains-match" row above refers to. `SimulatedPreferenceJudge` (`src/training/judge.py`) is a separate, **deterministic reference-free heuristic** (length + lexical diversity − hedging) that stands in for a real LLM-as-judge — it ignores the gold answer and is used by the `run_bamboogle_grpo_train` / `run_bamboogle_synthetic_grpo` examples. There is no trained reward model or LLM judge; a real judge would slot in behind the same `BatchJudgeFn` interface.

**Four reward dimensions** — `reward_components()` also groups every term into four subtotals via `REWARD_DIMENSIONS`, emitted as `dim_correctness`, `dim_citation_support`, `dim_retrieval_quality`, `dim_search_efficiency` (and available directly via `reward_dimensions()` or the pure `group_reward_components(components)`). Pre-scale, so `sum(dims) == terminal_reward + shaping_total == total / reward_scale`. The rollup is purely additive — no weight, preset, or `total` formula changed.

**GRPO** — `score_prompt_group` scores G rollouts for one prompt and normalises within-group advantages. `compute_grpo_outcome_advantage` computes `reward_i - mean(group)` for a flat rewards list. See `src/training/grpo.py`.

**PPO core** — `compute_ppo_policy_loss_core` returns `(pg_loss, pg_clipfrac, ppo_kl, surrogate)` and is the clipped surrogate the GRPO trainers use (with a group-relative advantage in place of GAE). `compute_value_loss` and `compute_gae_advantages` implement the PPO-with-critic path but are **not wired into any trainer** — training here is critic-free GRPO (no value/critic model), and those helpers exist for parity/tests only. All require an `eos_mask` tensor. See `src/training/ppo/core_algos.py`.

### Smoke test

End-to-end reward + GRPO, with no GPU:

```bash
python3 -m examples.run_grpo_training_pipeline
```

### XML search protocol

The ReAct-style trace format used by `SearchAgentLoop` uses these model-output tags:

```xml
<think>decide whether to answer or search</think>
<search>one precise query when external evidence is needed</search>
<fetch>comma- or newline-separated URLs when snippets are insufficient</fetch>
<answer>final grounded answer with citation labels</answer>
```

Optional model-output tags for multi-hop tasks:

```xml
<search_decision>answer</search_decision>   <!-- skip search when internal knowledge suffices -->
<subquestions>one research subquestion per line</subquestions>
<searches>parallel independent queries, one per line</searches>
```

Environment-only tags (injected by the loop — never output by the model):

```xml
<information>search results with citation labels</information>
<search_evaluation>sufficiency verdict and weak-query hints</search_evaluation>
<subquestions_feedback>per-subquestion coverage status</subquestions_feedback>
<full_page>fetched page content</full_page>
```

Mask all environment-only tags from policy/SFT action loss.

## Evaluation

### Bamboogle

Bamboogle is a two-hop QA benchmark that requires chaining retrieval across multiple hops — a strong signal for `SearchAgentLoop` quality.

**CLI (local CPU):**

```bash
python3 -m examples.run_bamboogle_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --limit 5 --print_trace
```

**CLI (server-backed):**

```bash
python3 -m examples.run_bamboogle_eval \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 \
  --search_url http://localhost:8001/retrieve \
  --reward_preset second_pass --limit 125
```

Reward presets: `sparse_final_only` | `simple_sparse` | `second_pass` | `third_pass`

**Apple Silicon shell script** (auto-starts SerpAPI retrieval server, reads `SERP_API_KEY` from `.env`):

```bash
bin/run_bamboogle_eval.sh                              # 5 examples, mps device
bin/run_bamboogle_eval.sh --smoke                      # 1 example, quick sanity check
bin/run_bamboogle_eval.sh --limit 125                  # full benchmark
bin/run_bamboogle_eval.sh --device cpu --limit 10
bin/run_bamboogle_eval.sh --limit 125 --concurrency 8  # ~6-8x faster via parallel SerpAPI calls
bin/run_bamboogle_eval.sh --limit 125 --concurrency 8 --resume  # resume an interrupted run
```

The dataset is cached locally after the first download (`~/.cache/agentic_search/bamboogle_test.jsonl`), so subsequent runs skip the network fetch. `--resume` reads the existing output file and skips already-evaluated questions, appending new results.

**Training data generation:**

```bash
bin/generate_training_data.sh                         # Bamboogle → data/bamboogle_train/
bin/generate_training_data.sh --preview               # print 5 sample rows, no write
bin/generate_training_data.sh --dataset nq            # Natural Questions
bin/generate_training_data.sh --dataset trivia_qa     # TriviaQA
bin/generate_training_data.sh --dataset hotpotqa --max_examples 500
```

Each run writes `data/<dataset>_train/train.parquet` and `data/<dataset>_train/test.parquet` ready for `LLMGRPOTrainer` or SFT.

### Activating the eval gates

The `Eval Gate` CI workflow (`.github/workflows/eval-gate.yml`) has two jobs — retrieval and RAGAS regression gates — that are **inactive placeholders** until real baselines are committed:

- The retrieval gate reads `data/eval/baseline_metrics.json`, which ships as a zero placeholder, so no regression can trip it. CI emits an `INACTIVE` warning until a real baseline lands.
- The RAGAS gate needs `data/eval/ragas_baseline.json`, which is not committed, so it reports `INACTIVE` and runs nothing.

To enforce them, generate real baselines against your canonical retrieval stack and commit the results:

```bash
# Retrieval baseline (needs a built index / BM25_INDEX_PATH or a running retrieval server)
python -m src.internal.retrieval.eval_runner \
  --dataset data/eval/qa_pairs.jsonl --top_k 10 \
  --output data/eval/baseline_metrics.json

# RAGAS baseline (needs OPENAI_API_KEY + the retrieval stack)
python -m src.internal.retrieval.ragas_eval \
  --dataset data/eval/ragas_qa.jsonl \
  --metrics faithfulness answer_relevancy \
  --output data/eval/ragas_baseline.json
```

Once a non-zero `baseline_metrics.json` (and/or a `ragas_baseline.json`) is committed, the corresponding gate starts enforcing regressions automatically.

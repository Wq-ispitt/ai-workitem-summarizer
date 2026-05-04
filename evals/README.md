# Eval Harness

Deterministic, regression-oriented evaluation for the work item summarizer.

> **What this is**: cheap automated metrics over a small, hand-labeled golden
> set so prompt/model changes are comparable across runs.
>
> **What this is not**: ground-truth semantic accuracy. Keyword-based scoring
> rewards topical inclusion, not necessarily correctness. Validate metric
> changes with a quick human spot-check of `runs/*/results.json`.

## Layout

```
evals/
├── dataset.jsonl          # labeled golden set (gitignored — contains internal titles)
├── dataset.example.jsonl  # one synthetic example showing the schema
├── .cache/{id}.json       # raw ADO inputs (gitignored — may contain private data)
├── runs/{ts}/             # one folder per eval run (gitignored)
│   ├── results.json       # full per-item predictions + scores + telemetry
│   └── report.md          # human-readable aggregate report
├── metrics.py             # scoring functions
├── baselines.py           # naive (non-LLM) baseline
├── fetch_dataset.py       # `fetch-eval` CLI: pull ADO items → cache + label skeleton
└── run_eval.py            # `run-eval` CLI: score predictions vs. dataset
```

## Workflow

1. **Seed the dataset.** Fetch each work item once; the raw JSON is cached
   locally so subsequent runs are reproducible without ADO calls or PAT.
   ```bash
   fetch-eval --ids 5197275,5173946,4826727
   ```
   This caches inputs to `.cache/` and appends label skeletons to
   `dataset.jsonl`.

2. **Hand-label `expected` blocks.** Open `dataset.jsonl` and replace each
   `TODO` entry. See [Label schema](#label-schema) below. The runner refuses
   to score a row that still contains `TODO` placeholders.

3. **Run the eval.**
   ```bash
   run-eval                      # gpt-4.1-mini
   run-eval --baseline naive     # title-only baseline for comparison
   ```
   Writes a timestamped folder under `runs/` with full results + a markdown
   report.

## Label schema

```jsonc
{
  "work_item_id": 12345,
  "title": "human-readable title (for the report only)",
  "expected": {
    // Keywords that should appear (case-insensitive substring) in the summary.
    // Pick distinctive concepts, not stop words.
    "summary_keywords": ["redis", "reporting"],

    // Cheap hallucination check: phrases the summary must NOT contain.
    "summary_must_not_claim": ["production outage", "security breach"],

    // One entry per risk you'd expect a sharp reviewer to flag.
    // A predicted risk matches if ANY keyword appears in its description.
    // Severity is scored separately on matched risks (see metrics.py).
    "expected_risks": [
      {"keywords": ["misconfiguration", "config"], "severity": "Medium"}
    ],

    // Each inner array is a concept group. A predicted action item matches
    // a group if it contains ALL of the group's keywords. Greedy assignment
    // — each predicted item can match at most one group.
    "expected_action_keywords": [
      ["acceptance", "criteria"],
      ["azure", "provision"]
    ],

    // Exact-match. One of: Small | Medium | Large.
    "expected_effort": "Medium"
  }
}
```

## Metrics

| Metric | Definition |
|---|---|
| Schema validity | Fraction of predictions that parse and pass enum/type checks. |
| Summary keyword coverage | Avg fraction of `summary_keywords` present in the predicted summary. |
| Forbidden-claim violations | Count of `summary_must_not_claim` phrases that did appear. |
| Risk detection P/R/F1 | Greedy keyword-overlap match between predicted and expected risks; severity ignored at this layer. |
| Severity accuracy on matched | Of the matched risk pairs, fraction with the same severity. Reported with a confusion matrix. |
| Unsupported predicted risks | Count of predicted risks that matched no expected risk — proxy for hallucination. |
| Action items P/R/F1 | Greedy match where a predicted action matches an expected concept group iff it contains ALL group keywords. |
| Effort accuracy | Exact match. |
| Telemetry | Total latency, input/output tokens, and `gpt-4.1-mini` list-price cost estimate per run. |

Each run also records `prompt_hash`, `dataset_hash`, `git_commit`, model, and
temperature so changes between runs are attributable.

## Honest caveats

- **Small N.** A 20-item golden set is for directional comparison and
  regression, not statistical claims. Always cite raw counts alongside rates.
- **Keyword scoring is shallow.** It won't catch a summary that mentions all
  the right terms but draws the wrong conclusion. Use the per-item report to
  spot-check a few outputs each run.
- **Severity F1 by class is noisy** when there are <5 examples per class.
- **Cost estimate** uses static `PRICING_USD_PER_1K` in `run_eval.py`. Update
  it when the model or pricing changes.

## Future extensions

Documented here so they don't get forgotten:

- **LLM-as-judge** as a secondary semantic cross-check (fixed rubric, fixed
  judge model).
- **Keyword aliases** so `"AC"` can match `"acceptance criteria"`.
- **Prompt versioning** beyond the hash — name + changelog per prompt.
- **Per-class severity F1** once each class has ≥10 expected examples.

# Architecture

A small, single-machine Python tool that turns Azure DevOps work items into
structured JSON summaries, plus an offline eval harness for measuring quality.

> Companion docs: [`known_failures.md`](known_failures.md) for observed failure
> modes; [`evals/README.md`](../evals/README.md) for the eval harness contract.

---

## 1. System overview

```
                    ┌────────────────────────────────────┐
                    │        Azure DevOps (msdata)       │
                    │  Project: Purview Data Governance  │
                    └────────────────┬───────────────────┘
                                     │  REST (WIQL + work items)
                                     │  Auth: Entra ID bearer
                                     ▼
┌──────────┐   ids/    ┌────────────────────┐   work-item dict   ┌──────────────────┐
│   CLI    │──query──▶ │     AdoClient      │ ─────────────────▶ │ WorkItemSummarizer│
│ summarize│           │  ado_client.py     │                    │  summarizer.py    │
└──────────┘           └────────────────────┘                    └────────┬──────────┘
     ▲                                                                    │
     │ pretty / json                                                      │ structured
     │                                                                    │ JSON request
     │                                                                    ▼
     │                                                          ┌────────────────────┐
     │                                                          │  Azure OpenAI      │
     │                                                          │  my-foundry-learn  │
     │                                                          │  gpt-4.1-mini      │
     │                                                          └────────┬───────────┘
     │                                                                   │
     │                                              WorkItemSummary      │
     └───────────────────────────────────────────────────────────────────┘
```

**Boundaries.** The tool is read-only against ADO and stateless. There is no
database, no server, no background worker. One CLI invocation = one batch.

---

## 2. Components

### `src/workitem_summarizer/`

| Module | Responsibility | Key types |
|---|---|---|
| `models.py` | Domain types | `WorkItemSummary`, `Risk`, `Severity`, `Effort` |
| `ado_client.py` | ADO REST wrapper, dual-mode auth | `AdoClient` |
| `summarizer.py` | LLM call with structured-output schema | `WorkItemSummarizer` |
| `cli.py` | Argument parsing, `.env` loading, output formatting | `summarize` entry point |

### `evals/`

| Module | Responsibility |
|---|---|
| `fetch_dataset.py` | Cache raw ADO work items + append label skeletons (`fetch-eval` CLI) |
| `draft_labels.py` | Auto-draft golden labels with `gpt-4.1-mini` (`draft-labels` CLI) |
| `baselines.py` | Trivial title-only baseline for sanity-checking |
| `metrics.py` | Deterministic scoring (schema validity, F1, severity acc, etc.) |
| `run_eval.py` | Orchestrator: load dataset → predict → score → write `report.md` |

### `tools/`

| File | Responsibility |
|---|---|
| `diagnose_latency.py` | HTTP-level Azure OpenAI probe; how the 1 RPM throttle was found |

---

## 3. Data flow — production path

1. **CLI parses args** (`--ids` or `--query`, plus org/project/endpoint).
2. **`.env` is loaded** from the repo root with a minimal parser
   (`ADO_PAT`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`).
3. **`AdoClient` resolves auth**:
   - if `ADO_PAT` is set → HTTP Basic with `:PAT` base64-encoded;
   - else → `az account get-access-token --resource 499b84ac-…-267ca6975798`
     and HTTP Bearer (this is required for `msdata`, which disables PAT scopes).
4. **Work items are fetched** via `GET /wit/workitems?ids=…&$expand=all` or
   `POST /wit/wiql` followed by an ID-batch fetch.
5. **`WorkItemSummarizer._format_work_item`** flattens the relevant fields
   (`Title`, `Type`, `State`, `AssignedTo.displayName`, `AreaPath`,
   `IterationPath`, `Description`, `AcceptanceCriteria`, `Tags`) into a
   plain-text block.
6. **One Azure OpenAI chat completion per item**, with `response_format` set
   to a JSON-schema strict object (`summary`, `risks[]`, `action_items[]`,
   `estimated_effort`). Temperature `0.2` to suppress drift.
7. **JSON parsed → typed dataclasses → `to_dict()`** for output.
8. **CLI writes** either pretty text (with severity emoji) or JSON, to stdout
   or `--out` file.

### Why structured output

`response_format={"type":"json_schema","strict":true,...}` is the only way to
get **100 % schema validity** without retry loops or post-hoc repair. The eval
run on commit `f062417` shows 25/25 schema-valid responses — no parse errors,
no manual JSON-fixing code path is needed.

---

## 4. Data flow — eval path

```
        ADO REST                                      Azure OpenAI
            │                                              │
            ▼                                              ▼
     ┌─────────────┐                              ┌──────────────────┐
     │ fetch-eval  │ ─── caches raw JSON ───▶     │   draft-labels   │
     │             │   evals/.cache/<id>.json     │                  │
     └─────┬───────┘                              └─────────┬────────┘
           │ appends label                                  │ overwrites
           │ skeletons                                      │ TODO labels
           ▼                                                ▼
                    ┌────────────────────────┐
                    │  evals/dataset.jsonl   │  (gitignored — internal titles)
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │      run-eval          │  predicts via summarizer or
                    │   (run_eval.py)        │  baseline; scores each row
                    └────────────┬───────────┘
                                 │
                                 ▼
                ┌──────────────────────────────────┐
                │ evals/runs/<utc-ts>/             │
                │   ├── results.json               │  full per-item data
                │   └── report.md                  │  human-readable summary
                └──────────────────────────────────┘
```

### `dataset.jsonl` row schema

```json
{
  "work_item_id": 3921445,
  "title": "Pdgbugbash : Actions are Still Visible for Deleted Governance Domains",
  "expected": {
    "summary_keywords": ["actions", "visible", "deleted", "governance domain"],
    "summary_must_not_claim": ["resolved by deleting domain"],
    "expected_risks": [
      { "keywords": ["actions still visible", "deleted governance domain"],
        "severity": "Medium" }
    ],
    "expected_action_keywords": [
      ["remove", "actions", "deleted governance domain"],
      ["fix", "actions", "display", "governance domain"]
    ],
    "expected_effort": "Small"
  }
}
```

### Metric design (see `evals/metrics.py`)

| Metric | What it measures | How |
|---|---|---|
| Schema validity | JSON parse + required fields + enum membership | structural checks |
| Summary keyword coverage | Does the summary mention the right things? | substring (case-insensitive) |
| Forbidden-claim violations | Does the summary state things it shouldn't? | negative substring |
| Risk detection P/R/F1 | Are real risks surfaced and not invented? | bipartite match on keyword groups |
| Severity accuracy on matched | When we identify the right risk, is severity correct? | exact-string match |
| Unsupported predicted risks | Hallucination signal | predicted − matched |
| Action items P/R/F1 | Are real next steps proposed? | bipartite match on AND-keyword groups |
| Effort accuracy | Is the size estimate right? | exact-string match |

All metrics are deterministic substring/string checks — no embeddings, no
LLM-as-judge. This keeps eval runs **reproducible** across commits and free of
nondeterminism. The tradeoff (metric strictness on inflection) is documented
in `known_failures.md` §3.

---

## 5. Auth

| Layer | Auth | Why |
|---|---|---|
| Azure DevOps | Entra ID bearer (default) or PAT (fallback) | `msdata` org disables PAT scopes — Bearer is the only path that works |
| Azure OpenAI | API key (`AZURE_OPENAI_API_KEY`) or `DefaultAzureCredential` | Endpoint is on a personal `my-foundry-learn` resource where the corp account has Cognitive Services User; key is convenient for local CLI runs |
| `az account get-access-token` | Existing `az login` from corp Entra ID | No new credentials issued; tied to user's MFA-protected session |

Resource ID `499b84ac-1321-427f-aa17-267ca6975798` is the well-known Azure
DevOps app ID and is hard-coded in `ado_client.py`.

---

## 6. Determinism, observability, cost

- **Determinism.** Temperature `0.2` plus `gpt-4.1-mini`'s strict structured
  output makes per-item predictions stable enough that the same dataset run
  twice produces near-identical scores. Run metadata (`prompt_hash`,
  `dataset_hash`, `git_commit`, `model`, `temperature`) is captured in
  `report.md` so any regression is bisectable.
- **Observability.** Each item in `results.json` records `latency_seconds`,
  `input_tokens`, `output_tokens`, `estimated_cost_usd`. The `report.md`
  aggregates totals and per-item details.
- **Cost.** ~$0.083 for the full 25-item eval at `gpt-4.1-mini` rates
  (input $0.15 / output $0.60 per 1M tokens). Negligible.

---

## 7. Deliberate non-goals

- **No web UI / server.** A CLI is enough for a portfolio piece and keeps
  surface area small.
- **No persistent store.** Caching is filesystem-only (`evals/.cache/`).
- **No multi-tenant auth.** This is a single-developer tool against one ADO
  org.
- **No retry/rate-limit logic in summarizer.** The OpenAI SDK's built-in
  retry-with-`Retry-After` is sufficient. We document the consequence in
  `known_failures.md` §6 rather than reinvent rate limiting.
- **No agentic loop.** One model call per item, no tool use. Tool-calling is
  the subject of Project 3 in the broader portfolio.

---

## 8. Where to extend next

| Want to … | Touch |
|---|---|
| Improve risk/action quality | `summarizer.py::SYSTEM_PROMPT` (and add few-shots) |
| Tighten metric strictness | `evals/metrics.py` (add lemmatization or embedding match) |
| Compare a new model | `run-eval --model <deployment>`; add to summarizer if new SDK |
| Add a real ground-truth set | Hand-edit `evals/dataset.jsonl` after `fetch-eval` instead of `draft-labels` |
| Build a regression gate | Wrap `run-eval` in CI; fail on F1 drop > N |

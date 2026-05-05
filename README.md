# AI Work Item Summarizer

Turn an Azure DevOps work item into a structured JSON brief — summary, risk
classification, action items, effort estimate — using Azure OpenAI with strict
schema enforcement.

Built as a portfolio project for an AI-engineer transition. The eval harness,
failure analysis, and design docs are first-class.

- 📐 **Architecture:** [`docs/architecture.md`](docs/architecture.md)
- 🐛 **Known failure modes:** [`docs/known_failures.md`](docs/known_failures.md)
- 📊 **Eval harness design:** [`evals/README.md`](evals/README.md)

---

## What it does

```bash
$ summarize --ids 4905173

============================================================
📋 [4905173] Terms not displayed correctly in Data Product details
============================================================
Summary: Glossary terms attached to a data product are missing from the
details panel after a recent UI refactor; the underlying API still
returns them.
Effort: Small
Risks:
  🟡 [Medium] Customers may believe terms were unassigned, causing data
     governance reports to under-count tagged assets.
Action Items:
  → Restore the terms field render in DataProductDetails.tsx
  → Add a regression test covering the terms section
```

Output schema (strict JSON, validated by the model via `response_format`):

```json
{
  "summary": "1–3 sentence description grounded in the work item fields",
  "risks": [{ "description": "...", "severity": "High|Medium|Low" }],
  "action_items": ["concrete next step", "..."],
  "estimated_effort": "Small|Medium|Large"
}
```

---

## Eval results (commit `f062417`, 25 closed Bugs/Tasks)

| Metric                           | gpt-4.1-mini | naive (title-only) |
|----------------------------------|-------------:|-------------------:|
| Schema validity                  |       100 %  |             100 %  |
| Summary keyword coverage         |        67.9% |              75.7% |
| Risk detection F1                |         0.36 |               0.00 |
| Severity accuracy on matched     |        41.7% |                n/a |
| Hallucinated risks (per item)    |         1.20 |                  0 |
| Action item F1                   |         0.11 |               0.00 |
| Effort accuracy                  |          60% |                12% |
| Cost (25 items)                  |       $0.083 |                 $0 |

Reports are written to `evals/runs/<utc-timestamp>/report.md` with full
per-item breakdowns, a severity confusion matrix, and run metadata
(`prompt_hash`, `dataset_hash`, `git_commit`, model, temperature).

> Honest caveats — the dataset is auto-drafted by the same model family
> (`gpt-4.1-mini` via `evals/draft_labels.py`) and N=25, so absolute scores are
> directional. The *relative* drift between model and naive baseline, plus the
> severity / over-prediction patterns, are the interesting signal. See
> [`docs/known_failures.md`](docs/known_failures.md) for the full critique
> including a deployment-throttling latency gotcha.

---

## Quick start

### Prerequisites

- Python 3.11+
- An Azure OpenAI deployment (default: `gpt-4.1-mini` on
  `my-foundry-learn.cognitiveservices.azure.com`)
- One of:
  - **Azure DevOps PAT** with *Work Items: Read* scope (works for most orgs), **or**
  - **Azure CLI** logged in with an account that can read your ADO project
    (required for `msdata` and other orgs that disable PAT scopes)

### Install

```bash
pip install -e ".[dev]"
```

### Configure

Create `.env` in the repo root:

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=<key>          # or omit and use DefaultAzureCredential

# Azure DevOps
ADO_ORG=msdata                       # your organization
ADO_PROJECT=Purview Data Governance  # your project
ADO_PAT=                             # leave empty to use Entra ID via az CLI
```

### Use

```bash
# Summarize specific work items
summarize --ids 4905173,4811759

# Or via WIQL query
summarize --query "SELECT [System.Id] FROM WorkItems
                   WHERE [System.AssignedTo] = @Me
                     AND [System.State] = 'Active'"

# Machine-readable output
summarize --ids 4905173 --output json --out summary.json
```

---

## Evaluating model changes

```bash
# 1. Seed the dataset (fetches + caches ADO items, appends label skeletons)
fetch-eval --query "SELECT [System.Id] FROM WorkItems
                    WHERE [System.AreaPath] UNDER 'Purview Data Governance\\DEH'
                      AND [System.State] IN ('Closed','Resolved','Done','Completed')
                      AND [System.ChangedDate] >= @Today - 180" --top 25

# 2. (Option A) Hand-edit evals/dataset.jsonl to fill in `expected` blocks
#    (Option B) Auto-draft labels with gpt-4.1-mini, then spot-check
draft-labels

# 3. Run the eval (writes evals/runs/<ts>/{report.md, results.json})
run-eval

# 4. Compare against the naive title-only baseline
run-eval --baseline naive
```

---

## Project layout

```
src/workitem_summarizer/
├── models.py        # WorkItemSummary, Risk, Severity, Effort
├── ado_client.py    # ADO REST client (PAT + Entra Bearer fallback)
├── summarizer.py    # LLM call with strict structured-output schema
└── cli.py           # `summarize` entry point

evals/
├── fetch_dataset.py # `fetch-eval` — cache ADO inputs + skeleton labels
├── draft_labels.py  # `draft-labels` — bootstrap labels with gpt-4.1-mini
├── baselines.py     # title-only baseline
├── metrics.py       # deterministic scoring (no LLM-as-judge, by design)
└── run_eval.py      # `run-eval` — orchestrator + report writer

tools/
└── diagnose_latency.py  # HTTP-level Azure OpenAI probe (used to find the
                         # 1 RPM throttle behind apparent ~60s/call latency)

docs/
├── architecture.md      # system design, data flow, deliberate non-goals
└── known_failures.md    # 6 catalogued failure modes from real eval runs
```

---

## Tech

- Python 3.11+ · `openai` · `azure-identity` · `httpx`
- Azure OpenAI `gpt-4.1-mini` with `response_format` strict JSON schema
- `pytest` for unit tests · ruff for lint
- Determinism: `temperature=0.2`, prompt hash + dataset hash recorded per run

## License

MIT

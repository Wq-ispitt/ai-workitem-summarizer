# AI Work Item Summarizer

AI-powered Azure DevOps work item summarizer that produces structured JSON output with risk classification, action items, and effort estimation.

## Features
- 🔍 Reads work items from Azure DevOps via REST API
- 🤖 Summarizes using Azure OpenAI (gpt-4.1-mini) with structured output
- ⚠️ Risk classification (High / Medium / Low)
- ✅ Action item extraction
- 📊 Effort estimation (Small / Medium / Large)
- 🔐 DefaultAzureCredential auth (no API keys)
- 🧪 Deterministic eval harness with hand-labeled golden set, schema-validity tracking, hallucination signal, and naive baseline

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Summarize specific work items
summarize --ids 5197275,5173946

# Run a WIQL query
summarize --query "SELECT [System.Id] FROM WorkItems WHERE [System.AssignedTo] = @Me AND [System.State] = 'Active'"

# JSON output
summarize --ids 5197275 --output json
```

## Evaluation

See [`evals/README.md`](evals/README.md) for the full design.

```bash
# Seed the dataset (caches inputs locally + appends label skeletons)
fetch-eval --ids 5197275,5173946,4826727

# Edit evals/dataset.jsonl — replace each TODO placeholder with real labels.

# Score the model
run-eval

# Compare against the naive (title-only) baseline
run-eval --baseline naive
```

Each run writes `evals/runs/{ts}/results.json` and `report.md` with aggregate
metrics, a severity confusion matrix, per-item breakdowns, and run metadata
(model, prompt hash, dataset hash, git commit, latency, token cost).

## Project Structure
```
src/workitem_summarizer/
├── models.py        # Data models (WorkItemSummary, Risk, Severity)
├── ado_client.py    # Azure DevOps REST API client
├── summarizer.py    # LLM summarizer with structured output
└── cli.py           # CLI entry point
evals/               # Evaluation harness (see evals/README.md)
tests/               # Unit tests
```

## Tech Stack
- Python 3.11+
- Azure OpenAI (gpt-4.1-mini) with structured JSON output
- Azure Identity (DefaultAzureCredential)
- httpx for ADO API calls

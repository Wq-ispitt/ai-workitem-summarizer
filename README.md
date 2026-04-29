# AI Work Item Summarizer

AI-powered Azure DevOps work item summarizer that produces structured JSON output with risk classification, action items, and effort estimation.

## Features
- 🔍 Reads work items from Azure DevOps via REST API
- 🤖 Summarizes using Azure OpenAI (gpt-4.1-mini) with structured output
- ⚠️ Risk classification (High / Medium / Low)
- ✅ Action item extraction
- 📊 Effort estimation (Small / Medium / Large)
- 🔐 DefaultAzureCredential auth (no API keys)

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

## Project Structure
```
src/workitem_summarizer/
├── __init__.py
├── models.py        # Data models (WorkItemSummary, Risk, Severity)
├── ado_client.py    # Azure DevOps REST API client
├── summarizer.py    # LLM summarizer with structured output
└── cli.py           # CLI entry point
tests/               # Unit tests
evals/               # Evaluation dataset & metrics
```

## Tech Stack
- Python 3.12+
- Azure OpenAI (gpt-4.1-mini) with structured JSON output
- Azure Identity (DefaultAzureCredential)
- httpx for ADO API calls
AI-powered Azure DevOps work item summarizer with structured JSON output, risk classification, and evaluation harness

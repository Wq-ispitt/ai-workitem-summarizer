"""CLI entry point for the work item summarizer."""

import argparse
import json
import os
import sys
from pathlib import Path

from .ado_client import AdoClient
from .summarizer import WorkItemSummarizer

DEFAULT_ENDPOINT = "https://my-foundry-learn.cognitiveservices.azure.com/"
DEFAULT_ORG = "msdata"
DEFAULT_PROJECT = "Purview Data Governance"


def _load_env() -> None:
    """Load .env file from project root if it exists."""
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="AI-powered Azure DevOps work item summarizer")
    parser.add_argument("--ids", type=str, help="Comma-separated work item IDs to summarize")
    parser.add_argument("--query", type=str, help="WIQL query to find work items")
    parser.add_argument("--org", type=str, default=DEFAULT_ORG, help="Azure DevOps organization")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="Azure DevOps project")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="Azure OpenAI endpoint")
    parser.add_argument("--deployment", type=str, default="gpt-4.1-mini", help="Model deployment name")
    parser.add_argument("--output", type=str, choices=["json", "pretty"], default="pretty", help="Output format")
    parser.add_argument("--out", type=str, help="Write output to this file instead of stdout")

    args = parser.parse_args()

    if not args.ids and not args.query:
        parser.error("Provide --ids or --query")

    ado = AdoClient(args.org, args.project)
    summarizer = WorkItemSummarizer(args.endpoint, args.deployment)

    if args.ids:
        ids = [int(i.strip()) for i in args.ids.split(",")]
        work_items = ado.get_work_items(ids)
    else:
        work_items = ado.query_work_items(args.query)

    if not work_items:
        print("No work items found.")
        sys.exit(0)

    results = summarizer.summarize_batch(work_items)

    if args.output == "json":
        text = json.dumps(results, indent=2)
    else:
        lines = []
        for r in results:
            lines.append(f"\n{'='*60}")
            lines.append(f"📋 [{r['work_item_id']}] {r['title']}")
            lines.append(f"{'='*60}")
            lines.append(f"Summary: {r['summary']}")
            lines.append(f"Effort: {r['estimated_effort']}")
            if r["risks"]:
                lines.append("Risks:")
                for risk in r["risks"]:
                    icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk["severity"], "⚪")
                    lines.append(f"  {icon} [{risk['severity']}] {risk['description']}")
            if r["action_items"]:
                lines.append("Action Items:")
                for item in r["action_items"]:
                    lines.append(f"  → {item}")
        text = "\n".join(lines)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {len(results)} summary(ies) to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

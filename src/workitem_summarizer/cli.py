"""CLI entry point for the work item summarizer."""

import argparse
import json
import sys

from .ado_client import AdoClient
from .summarizer import WorkItemSummarizer

DEFAULT_ENDPOINT = "https://my-foundry-learn.cognitiveservices.azure.com/"
DEFAULT_ORG = "msazure"
DEFAULT_PROJECT = "One"


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-powered Azure DevOps work item summarizer")
    parser.add_argument("--ids", type=str, help="Comma-separated work item IDs to summarize")
    parser.add_argument("--query", type=str, help="WIQL query to find work items")
    parser.add_argument("--org", type=str, default=DEFAULT_ORG, help="Azure DevOps organization")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="Azure DevOps project")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="Azure OpenAI endpoint")
    parser.add_argument("--deployment", type=str, default="gpt-4.1-mini", help="Model deployment name")
    parser.add_argument("--output", type=str, choices=["json", "pretty"], default="pretty", help="Output format")

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
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"\n{'='*60}")
            print(f"📋 [{r['work_item_id']}] {r['title']}")
            print(f"{'='*60}")
            print(f"Summary: {r['summary']}")
            print(f"Effort: {r['estimated_effort']}")
            if r["risks"]:
                print("Risks:")
                for risk in r["risks"]:
                    icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk["severity"], "⚪")
                    print(f"  {icon} [{risk['severity']}] {risk['description']}")
            if r["action_items"]:
                print("Action Items:")
                for item in r["action_items"]:
                    print(f"  → {item}")


if __name__ == "__main__":
    main()

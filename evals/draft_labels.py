"""Auto-draft evaluation labels for dataset rows that still contain TODO markers.

This is a *bootstrapping* tool. It reads each cached work item (title,
description, severity, priority, resolution) and asks an LLM to propose label
skeletons grounded in the actual ADO content. The output is written back into
``evals/dataset.jsonl`` in the same schema as a hand-labeled row, so a human
reviewer just needs to skim and edit obvious mistakes rather than write labels
from scratch.

Caveats — please read before trusting the metrics:

- Using the same model family to draft labels and to evaluate the production
  summarizer creates correlated bias. Treat auto-drafted rows as "v0" labels;
  any row used to gate releases should be reviewed by a human.
- The drafter never invents details: it is prompted to use only the provided
  work item fields. If a field is missing it returns an empty list.
- Severity is taken from ``Microsoft.VSTS.Common.Severity`` when present and
  only falls back to LLM judgement when it is not.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from html import unescape
from pathlib import Path

from openai import AzureOpenAI

EVALS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALS_DIR / "dataset.jsonl"
CACHE_DIR = EVALS_DIR / ".cache"

VALID_SEVERITIES = {"High", "Medium", "Low"}
VALID_EFFORTS = {"Small", "Medium", "Large"}

ADO_SEVERITY_MAP = {
    "1 - Critical": "High",
    "2 - High": "High",
    "3 - Medium": "Medium",
    "4 - Low": "Low",
}

DRAFTER_SYSTEM_PROMPT = """You generate evaluation labels for an ADO work item summarizer.

Given the work item below, produce a JSON object with these fields. Ground every
value in the work item content. If you have to guess, leave the list empty.

- summary_keywords: 3-6 lowercase distinctive words/phrases that any acceptable
  summary of this item MUST mention (e.g. domain names, feature names, the bug
  symptom, the action verb). Skip generic words like "issue" or "fix".
- summary_must_not_claim: phrases a summary should NOT claim because the work
  item does not support them (e.g. "data loss" if there is none). Often empty.
- expected_risks: list of distinctive risk areas the summarizer should flag,
  each as {"keywords": [..], "severity": "High"|"Medium"|"Low"}. Use lowercase
  keywords drawn from the work item, not synonyms. If the work item mentions no
  risks beyond the bug itself, return one risk reflecting the bug.
- expected_action_keywords: list of keyword groups, each group is the set of
  keywords that must appear together in one action item bullet. Each group
  should describe a concrete next step (e.g. ["remove", "actions", "deleted"]).
- expected_effort: one of "Small", "Medium", "Large", based on description
  complexity, number of areas touched, and whether the resolution looks like a
  one-line fix or a feature.

Return JSON only."""

DRAFTER_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "draft_labels",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary_keywords": {"type": "array", "items": {"type": "string"}},
                "summary_must_not_claim": {"type": "array", "items": {"type": "string"}},
                "expected_risks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "severity": {"type": "string", "enum": ["High", "Medium", "Low"]},
                        },
                        "required": ["keywords", "severity"],
                        "additionalProperties": False,
                    },
                },
                "expected_action_keywords": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "expected_effort": {"type": "string", "enum": ["Small", "Medium", "Large"]},
            },
            "required": [
                "summary_keywords",
                "summary_must_not_claim",
                "expected_risks",
                "expected_action_keywords",
                "expected_effort",
            ],
            "additionalProperties": False,
        },
    },
}


def _strip_html(text: str) -> str:
    if not text:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(no_tags)).strip()


def _row_has_todo(expected: dict) -> bool:
    blob = json.dumps(expected)
    return "TODO" in blob


def _format_work_item(work_item: dict) -> tuple[str, str | None]:
    """Return (prompt_body, ado_severity_label_or_None)."""
    fields = work_item.get("fields", {})
    description = _strip_html(fields.get("System.Description", ""))
    history = _strip_html(fields.get("System.History", ""))
    parts = [
        f"Title: {fields.get('System.Title', '')}",
        f"Type: {fields.get('System.WorkItemType', '')}",
        f"State: {fields.get('System.State', '')}",
        f"Severity (ADO): {fields.get('Microsoft.VSTS.Common.Severity', 'unset')}",
        f"Priority (ADO): {fields.get('Microsoft.VSTS.Common.Priority', 'unset')}",
        f"Resolved Reason: {fields.get('Microsoft.VSTS.Common.ResolvedReason', '')}",
        f"Resolution Reason: {fields.get('Custom.ResolutionReason', '')}",
        f"Tags: {fields.get('System.Tags', '')}",
        f"Description: {description or '(none)'}",
        f"History/Comments: {history or '(none)'}",
    ]
    ado_sev = ADO_SEVERITY_MAP.get(fields.get("Microsoft.VSTS.Common.Severity", ""))
    return "\n".join(parts), ado_sev


def _build_client(endpoint: str) -> AzureOpenAI:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if api_key:
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-12-01-preview",
        )
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-12-01-preview",
    )


def _draft_one(client: AzureOpenAI, deployment: str, work_item: dict) -> dict:
    body, ado_sev = _format_work_item(work_item)
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": DRAFTER_SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ],
        response_format=DRAFTER_RESPONSE_SCHEMA,
        temperature=0.0,
    )
    draft = json.loads(response.choices[0].message.content)
    if ado_sev and draft.get("expected_risks"):
        # Anchor the most prominent risk to the actual ADO severity to avoid
        # the LLM grading itself when we later evaluate severity accuracy.
        draft["expected_risks"][0]["severity"] = ado_sev
    return draft


def _load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        return []
    rows = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _save_dataset(rows: list[dict]) -> None:
    with DATASET_PATH.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draft eval labels for TODO rows")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max rows to draft this run"
    )
    parser.add_argument(
        "--ids", default=None, help="Comma-separated work item IDs to draft (overrides TODO filter)"
    )
    parser.add_argument(
        "--deployment", default="gpt-4.1-mini", help="Azure OpenAI deployment name"
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        help="Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print drafts to stdout without modifying dataset.jsonl",
    )
    args = parser.parse_args()

    if not args.endpoint:
        print("AZURE_OPENAI_ENDPOINT is required", file=sys.stderr)
        return 2

    rows = _load_dataset()
    if not rows:
        print(f"No rows found in {DATASET_PATH}", file=sys.stderr)
        return 1

    if args.ids:
        target_ids = {int(x) for x in args.ids.split(",") if x.strip()}
        candidates = [r for r in rows if r.get("work_item_id") in target_ids]
    else:
        candidates = [r for r in rows if _row_has_todo(r.get("expected", {}))]

    if args.limit is not None:
        candidates = candidates[: args.limit]

    if not candidates:
        print("No rows need drafting (all labels look complete)")
        return 0

    client = _build_client(args.endpoint)
    print(f"Drafting labels for {len(candidates)} row(s) using {args.deployment}...")

    drafted = 0
    skipped: list[tuple[int, str]] = []
    for row in candidates:
        wid = row.get("work_item_id")
        cache_path = CACHE_DIR / f"{wid}.json"
        if not cache_path.exists():
            skipped.append((wid, "cache miss"))
            continue
        try:
            work_item = json.loads(cache_path.read_text(encoding="utf-8"))
            draft = _draft_one(client, args.deployment, work_item)
        except Exception as exc:  # noqa: BLE001 — surface drafter errors per row
            skipped.append((wid, f"error: {exc}"))
            continue
        if args.dry_run:
            print(f"\n--- {wid}: {row.get('title', '')[:80]} ---")
            print(json.dumps(draft, indent=2))
        else:
            row["expected"] = draft
            print(f"  drafted {wid}: {row.get('title', '')[:70]}")
        drafted += 1

    if not args.dry_run and drafted:
        _save_dataset(rows)
        print(f"\nWrote {drafted} draft(s) to {DATASET_PATH}.")
        print("Review each row and edit any obviously wrong labels before running run-eval.")

    if skipped:
        print(f"\nSkipped {len(skipped)} row(s):")
        for wid, reason in skipped:
            print(f"  {wid}: {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Helper to seed the eval dataset.

Fetches work items from Azure DevOps, caches their full JSON to evals/.cache/
(gitignored, so private data stays local), and appends label skeletons to
evals/dataset.jsonl for the user to fill in.

Usage:
    fetch-eval --ids 5197275,5173946,4826727
    fetch-eval --query "SELECT [System.Id] FROM WorkItems WHERE ..."
    fetch-eval --query "..." --top 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from workitem_summarizer.ado_client import AdoClient
from workitem_summarizer.cli import DEFAULT_ORG, DEFAULT_PROJECT, _load_env

EVAL_DIR = Path(__file__).resolve().parent
CACHE_DIR = EVAL_DIR / ".cache"
DATASET_FILE = EVAL_DIR / "dataset.jsonl"


def _label_skeleton(work_item: dict) -> dict:
    return {
        "work_item_id": work_item.get("id"),
        "title": work_item.get("fields", {}).get("System.Title", ""),
        "expected": {
            "summary_keywords": ["TODO: words that must appear in the summary"],
            "summary_must_not_claim": [],
            "expected_risks": [
                {
                    "keywords": ["TODO: distinctive risk keyword"],
                    "severity": "Medium",
                }
            ],
            "expected_action_keywords": [
                ["TODO: required keyword 1", "TODO: required keyword 2"]
            ],
            "expected_effort": "Medium",
        },
    }


def _existing_ids() -> set[int]:
    if not DATASET_FILE.exists():
        return set()
    out: set[int] = set()
    for line in DATASET_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            row = json.loads(line)
            wid = row.get("work_item_id")
            if isinstance(wid, int):
                out.add(wid)
        except json.JSONDecodeError:
            continue
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_env()
    parser = argparse.ArgumentParser(description="Seed the eval dataset from ADO work items")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids", help="Comma-separated work item IDs")
    src.add_argument("--query", help="WIQL query selecting work items")
    parser.add_argument("--top", type=int, default=25, help="Max items to fetch when using --query")
    parser.add_argument("--org", default=DEFAULT_ORG)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-add label skeletons even if the ID already exists in dataset.jsonl",
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = _existing_ids()

    ado = AdoClient(args.org, args.project)
    if args.ids:
        ids = [int(i.strip()) for i in args.ids.split(",") if i.strip()]
        work_items = ado.get_work_items(ids)
    else:
        work_items = ado.query_work_items(args.query, top=args.top)
        print(f"Query returned {len(work_items)} work item(s).")

    new_rows = []
    for wi in work_items:
        wid = wi.get("id")
        if wid is None:
            continue
        cache_path = CACHE_DIR / f"{wid}.json"
        cache_path.write_text(json.dumps(wi, indent=2), encoding="utf-8")
        print(f"  cached input: {cache_path.relative_to(EVAL_DIR.parent)}")
        if wid in existing and not args.force:
            print(f"  skipping label skeleton for {wid} (already in dataset.jsonl; pass --force to override)")
            continue
        new_rows.append(_label_skeleton(wi))

    if new_rows:
        with DATASET_FILE.open("a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nAppended {len(new_rows)} label skeleton(s) to {DATASET_FILE.name}.")
        print("Edit each row's `expected` block, replacing the TODO entries with real labels.")
    else:
        print("\nNo new label skeletons added.")


if __name__ == "__main__":
    main()

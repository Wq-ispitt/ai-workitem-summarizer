"""Eval runner: scores predictions against the labeled dataset.

Loads cached ADO inputs from evals/.cache/{id}.json, generates predictions
with the LLM (or a baseline), scores against expected labels, and writes a
timestamped report to evals/runs/{ts}/.

Usage:
    run-eval                       # score gpt-4.1-mini against dataset.jsonl
    run-eval --baseline naive      # score naive baseline (title only)
    run-eval --dataset evals/dataset.example.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from workitem_summarizer.cli import DEFAULT_ENDPOINT, _load_env
from workitem_summarizer.summarizer import SYSTEM_PROMPT, WorkItemSummarizer

from .baselines import naive_summary
from .metrics import aggregate, evaluate_item

EVAL_DIR = Path(__file__).resolve().parent
CACHE_DIR = EVAL_DIR / ".cache"
RUNS_DIR = EVAL_DIR / "runs"
DATASET_DEFAULT = EVAL_DIR / "dataset.jsonl"


def _load_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        sys.exit(f"Dataset not found: {path}")
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            sys.exit(f"Invalid JSON on line {i} of {path}: {e}")
    if not rows:
        sys.exit(f"Dataset {path} is empty.")
    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict]) -> None:
    issues: list[str] = []
    for i, row in enumerate(rows):
        wid = row.get("work_item_id")
        if not isinstance(wid, int):
            issues.append(f"row {i}: work_item_id missing or not int")
            continue
        expected = row.get("expected") or {}
        if not expected.get("expected_effort"):
            issues.append(f"row {i} (id={wid}): expected.expected_effort is empty")
        for risk in expected.get("expected_risks", []) or []:
            sev = risk.get("severity")
            if sev not in {"High", "Medium", "Low"}:
                issues.append(f"row {i} (id={wid}): risk severity {sev!r} is not High/Medium/Low")
            kws = risk.get("keywords") or []
            if any(str(k).lower().startswith("todo") for k in kws):
                issues.append(f"row {i} (id={wid}): risk keywords still contain TODO placeholder")
        if any(
            any(str(k).lower().startswith("todo") for k in group)
            for group in expected.get("expected_action_keywords", []) or []
        ):
            issues.append(f"row {i} (id={wid}): action keywords still contain TODO placeholder")
        if any(str(k).lower().startswith("todo") for k in expected.get("summary_keywords", []) or []):
            issues.append(f"row {i} (id={wid}): summary keywords still contain TODO placeholder")
    if issues:
        msg = "Dataset has unresolved issues:\n  - " + "\n  - ".join(issues)
        sys.exit(msg)


def _load_input(work_item_id: int) -> dict:
    cache_path = CACHE_DIR / f"{work_item_id}.json"
    if not cache_path.exists():
        sys.exit(
            f"Cached input missing: {cache_path}. Run: fetch-eval --ids {work_item_id}"
        )
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=EVAL_DIR.parent,
        )
        return out.stdout.strip() or "unknown"
    except (FileNotFoundError, OSError):
        return "unknown"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# Azure OpenAI gpt-4.1-mini list price per 1K tokens (USD) at time of writing.
# Update PRICING when the model or pricing changes.
PRICING_USD_PER_1K = {
    "gpt-4.1-mini": {"input": 0.00040, "output": 0.00160},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING_USD_PER_1K.get(model)
    if not rates:
        return 0.0
    return (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]


def _predict_llm(
    summarizer: WorkItemSummarizer, work_item: dict
) -> tuple[dict, dict]:
    start = time.perf_counter()
    user_content = summarizer._format_work_item(work_item)
    response = summarizer.client.chat.completions.create(
        model=summarizer.deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "work_item_summary",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "risks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "severity": {
                                        "type": "string",
                                        "enum": ["High", "Medium", "Low"],
                                    },
                                },
                                "required": ["description", "severity"],
                                "additionalProperties": False,
                            },
                        },
                        "action_items": {"type": "array", "items": {"type": "string"}},
                        "estimated_effort": {
                            "type": "string",
                            "enum": ["Small", "Medium", "Large"],
                        },
                    },
                    "required": ["summary", "risks", "action_items", "estimated_effort"],
                    "additionalProperties": False,
                },
            },
        },
        temperature=0.2,
    )
    elapsed = time.perf_counter() - start

    raw_text = response.choices[0].message.content or "{}"
    try:
        prediction = json.loads(raw_text)
    except json.JSONDecodeError:
        prediction = {"_raw": raw_text}

    usage = getattr(response, "usage", None)
    in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
    out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
    telemetry = {
        "latency_seconds": round(elapsed, 3),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "estimated_cost_usd": round(_estimate_cost(summarizer.deployment, in_tok, out_tok), 6),
    }
    return prediction, telemetry


def _predict_baseline(name: str, work_item: dict) -> tuple[dict, dict]:
    if name == "naive":
        return naive_summary(work_item), {
            "latency_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    sys.exit(f"Unknown baseline: {name}")


def _markdown_report(metadata: dict, agg: dict, per_item: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"# Eval Run — {metadata['run_id']}")
    lines.append("")
    lines.append("## Metadata")
    for key in (
        "model",
        "baseline",
        "prompt_hash",
        "dataset_path",
        "dataset_hash",
        "dataset_size",
        "git_commit",
        "temperature",
    ):
        if key in metadata:
            lines.append(f"- **{key}**: `{metadata[key]}`")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    if agg.get("item_count", 0) == 0:
        lines.append("_No items scored._")
        return "\n".join(lines)

    sv = agg["schema_validity"]
    lines.append(f"- **Schema validity**: {sv['valid']}/{sv['total']} ({sv['rate']:.1%})")

    s = agg["summary"]
    lines.append(
        f"- **Summary keyword coverage** (avg): {s['avg_keyword_coverage']:.1%}"
        f"  · forbidden-claim violations: {s['forbidden_violations_total']}"
    )

    r = agg["risks"]
    lines.append(
        f"- **Risk detection** P/R/F1: {r['detection_precision']:.2f} / "
        f"{r['detection_recall']:.2f} / {r['detection_f1']:.2f}  "
        f"(matched {r['matched_total']}/{r['expected_total']} expected, "
        f"{r['matched_total']}/{r['predicted_total']} predicted)"
    )
    lines.append(
        f"- **Severity accuracy on matched risks**: {r['severity_accuracy_on_matched']:.1%} "
        f"({r['severity_correct_on_matched']}/{r['matched_total']})"
    )
    lines.append(
        f"- **Unsupported predicted risks** (possible hallucinations): "
        f"{r['unmatched_predicted_total']} total · "
        f"{r['avg_unmatched_predicted_per_item']:.2f}/item"
    )

    a = agg["actions"]
    lines.append(
        f"- **Action items** P/R/F1: {a['precision']:.2f} / {a['recall']:.2f} / {a['f1']:.2f}  "
        f"(matched {a['matched_expected_total']}/{a['expected_total']} expected, "
        f"{a['matched_predicted_total']}/{a['predicted_total']} predicted)"
    )

    e = agg["effort"]
    lines.append(f"- **Effort accuracy**: {e['accuracy']:.1%} ({e['correct']}/{e['total']})")

    lines.append("")
    lines.append("## Severity Confusion Matrix")
    lines.append("")
    lines.append("| expected ↓  /  predicted → | High | Medium | Low |")
    lines.append("|---|---|---|---|")
    for sev in ("High", "Medium", "Low"):
        row = r["severity_confusion"][sev]
        lines.append(f"| **{sev}** | {row['High']} | {row['Medium']} | {row['Low']} |")
    lines.append("")

    lines.append("## Per-Item Summary")
    lines.append("")
    lines.append("| ID | Schema | Summary cov | Risk F1 | Sev acc | Action F1 | Effort |")
    lines.append("|---|---|---|---|---|---|---|")
    for item in per_item:
        wid = item["work_item_id"]
        ev = item["evaluation"]
        if not ev["schema"]["valid"]:
            lines.append(f"| {wid} | ❌ | – | – | – | – | – |")
            continue
        lines.append(
            f"| {wid} "
            f"| ✅ "
            f"| {ev['summary']['keyword_coverage']:.0%} "
            f"| {ev['risks']['detection_f1']:.2f} "
            f"| {ev['risks']['severity_accuracy_on_matched']:.0%} "
            f"| {ev['actions']['f1']:.2f} "
            f"| {'✅' if ev['effort_correct'] else '❌'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_env()
    parser = argparse.ArgumentParser(description="Run the work item summarizer evaluation")
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--deployment", default="gpt-4.1-mini")
    parser.add_argument(
        "--baseline",
        choices=["naive"],
        help="Skip the LLM and score a baseline instead.",
    )
    args = parser.parse_args()

    rows = _load_dataset(args.dataset)
    dataset_text = args.dataset.read_text(encoding="utf-8")

    summarizer: WorkItemSummarizer | None = None
    if not args.baseline:
        summarizer = WorkItemSummarizer(args.endpoint, args.deployment)

    per_item: list[dict[str, Any]] = []
    telemetry_total = {
        "latency_seconds": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
    }

    for row in rows:
        wid = row["work_item_id"]
        work_item = _load_input(wid)
        if args.baseline:
            prediction, tel = _predict_baseline(args.baseline, work_item)
        else:
            assert summarizer is not None
            prediction, tel = _predict_llm(summarizer, work_item)

        evaluation = evaluate_item(prediction, row.get("expected", {}))
        per_item.append(
            {
                "work_item_id": wid,
                "title": row.get("title", ""),
                "prediction": prediction,
                "telemetry": tel,
                "evaluation": evaluation,
            }
        )
        for k in telemetry_total:
            telemetry_total[k] += tel[k]
        cov = evaluation["summary"]["keyword_coverage"] if evaluation["schema"]["valid"] else 0.0
        print(
            f"  scored {wid}: schema={'ok' if evaluation['schema']['valid'] else 'BAD'} "
            f"summary_cov={cov:.0%} latency={tel['latency_seconds']}s"
        )

    agg = aggregate([p["evaluation"] for p in per_item])

    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "model": args.deployment if not args.baseline else None,
        "baseline": args.baseline,
        "endpoint": args.endpoint,
        "temperature": 0.2,
        "prompt_hash": _hash(SYSTEM_PROMPT),
        "dataset_path": str(args.dataset),
        "dataset_hash": _hash(dataset_text),
        "dataset_size": len(rows),
        "git_commit": _git_commit(),
        "telemetry_total": {
            **telemetry_total,
            "estimated_cost_usd": round(telemetry_total["estimated_cost_usd"], 6),
            "latency_seconds": round(telemetry_total["latency_seconds"], 3),
        },
    }

    results_path = run_dir / "results.json"
    results_path.write_text(
        json.dumps({"metadata": metadata, "aggregate": agg, "items": per_item}, indent=2),
        encoding="utf-8",
    )

    report_path = run_dir / "report.md"
    report_path.write_text(_markdown_report(metadata, agg, per_item), encoding="utf-8")

    print()
    print(f"Wrote results → {results_path.relative_to(EVAL_DIR.parent)}")
    print(f"Wrote report  → {report_path.relative_to(EVAL_DIR.parent)}")
    if agg.get("item_count"):
        s = agg["summary"]
        r = agg["risks"]
        a = agg["actions"]
        e = agg["effort"]
        print()
        print(
            f"Summary cov {s['avg_keyword_coverage']:.0%} · "
            f"Risk F1 {r['detection_f1']:.2f} · "
            f"Sev acc {r['severity_accuracy_on_matched']:.0%} · "
            f"Action F1 {a['f1']:.2f} · "
            f"Effort acc {e['accuracy']:.0%}"
        )
        print(
            f"Total cost ${metadata['telemetry_total']['estimated_cost_usd']:.4f} · "
            f"latency {metadata['telemetry_total']['latency_seconds']}s"
        )


if __name__ == "__main__":
    main()

"""Deterministic scoring metrics for work item summary evaluation.

Designed as a regression-grade proxy: cheap to compute, reproducible across
runs, and useful for prompt iteration. NOT a substitute for human judgment of
semantic quality — see evals/README.md for caveats.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

VALID_SEVERITIES = {"High", "Medium", "Low"}
VALID_EFFORTS = {"Small", "Medium", "Large"}


def _norm(text: str) -> str:
    return (text or "").lower()


def _contains_all(text: str, keywords: list[str]) -> bool:
    t = _norm(text)
    return all(_norm(k) in t for k in keywords if k)


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = _norm(text)
    return any(_norm(k) in t for k in keywords if k)


# ---------------------------------------------------------------------------
# Schema validity
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"summary", "risks", "action_items", "estimated_effort"}


def schema_validity(predicted: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(predicted, dict):
        return {"valid": False, "errors": ["prediction is not a dict"]}

    missing = REQUIRED_FIELDS - predicted.keys()
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")

    if not isinstance(predicted.get("summary", ""), str):
        errors.append("summary is not a string")

    if predicted.get("estimated_effort") not in VALID_EFFORTS:
        errors.append(f"invalid effort: {predicted.get('estimated_effort')!r}")

    risks = predicted.get("risks", [])
    if not isinstance(risks, list):
        errors.append("risks is not a list")
    else:
        for i, r in enumerate(risks):
            if not isinstance(r, dict):
                errors.append(f"risk[{i}] not a dict")
                continue
            if r.get("severity") not in VALID_SEVERITIES:
                errors.append(f"risk[{i}] invalid severity: {r.get('severity')!r}")
            if not isinstance(r.get("description", ""), str):
                errors.append(f"risk[{i}] description not a string")

    actions = predicted.get("action_items", [])
    if not isinstance(actions, list):
        errors.append("action_items is not a list")
    elif not all(isinstance(a, str) for a in actions):
        errors.append("action_items contains non-string entries")

    return {"valid": not errors, "errors": errors}


# ---------------------------------------------------------------------------
# Per-component scoring
# ---------------------------------------------------------------------------


@dataclass
class SummaryScore:
    keyword_coverage: float
    keywords_matched: int
    keywords_total: int
    forbidden_violations: list[str] = field(default_factory=list)


def score_summary(predicted_summary: str, expected: dict) -> SummaryScore:
    keywords = expected.get("summary_keywords", []) or []
    forbidden = expected.get("summary_must_not_claim", []) or []
    matched = sum(1 for k in keywords if _norm(k) in _norm(predicted_summary))
    total = len(keywords)
    coverage = matched / total if total else 1.0
    violations = [f for f in forbidden if _norm(f) in _norm(predicted_summary)]
    return SummaryScore(
        keyword_coverage=coverage,
        keywords_matched=matched,
        keywords_total=total,
        forbidden_violations=violations,
    )


@dataclass
class RiskScore:
    expected_count: int
    predicted_count: int
    matched_count: int
    detection_precision: float
    detection_recall: float
    detection_f1: float
    severity_correct_on_matched: int
    severity_accuracy_on_matched: float
    severity_confusion: dict[str, dict[str, int]]
    unmatched_predicted: list[dict]
    unmatched_expected: list[dict]


def _greedy_match_risks(predicted: list[dict], expected: list[dict]) -> list[tuple[int, int]]:
    """Greedy assignment: each expected risk gets at most one predicted match.

    A predicted risk matches an expected risk if any of the expected risk's
    keywords appears in the predicted description (case-insensitive).
    Severity is intentionally ignored at the matching stage — it is scored
    separately on matched pairs.
    """
    used_predicted: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for ei, exp in enumerate(expected):
        keywords = exp.get("keywords", []) or []
        for pi, pred in enumerate(predicted):
            if pi in used_predicted:
                continue
            if _contains_any(pred.get("description", ""), keywords):
                pairs.append((ei, pi))
                used_predicted.add(pi)
                break
    return pairs


def score_risks(predicted_risks: list[dict], expected_risks: list[dict]) -> RiskScore:
    pairs = _greedy_match_risks(predicted_risks, expected_risks)
    matched = len(pairs)
    p_count = len(predicted_risks)
    e_count = len(expected_risks)

    precision = matched / p_count if p_count else (1.0 if e_count == 0 else 0.0)
    recall = matched / e_count if e_count else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    severity_correct = 0
    confusion: dict[str, dict[str, int]] = {
        sev: {s: 0 for s in VALID_SEVERITIES} for sev in VALID_SEVERITIES
    }
    for ei, pi in pairs:
        exp_sev = expected_risks[ei].get("severity")
        pred_sev = predicted_risks[pi].get("severity")
        if exp_sev in VALID_SEVERITIES and pred_sev in VALID_SEVERITIES:
            confusion[exp_sev][pred_sev] += 1
            if exp_sev == pred_sev:
                severity_correct += 1

    sev_acc = (severity_correct / matched) if matched else 1.0

    matched_predicted = {pi for _, pi in pairs}
    matched_expected = {ei for ei, _ in pairs}
    unmatched_predicted = [
        predicted_risks[i] for i in range(p_count) if i not in matched_predicted
    ]
    unmatched_expected = [
        expected_risks[i] for i in range(e_count) if i not in matched_expected
    ]

    return RiskScore(
        expected_count=e_count,
        predicted_count=p_count,
        matched_count=matched,
        detection_precision=precision,
        detection_recall=recall,
        detection_f1=f1,
        severity_correct_on_matched=severity_correct,
        severity_accuracy_on_matched=sev_acc,
        severity_confusion=confusion,
        unmatched_predicted=unmatched_predicted,
        unmatched_expected=unmatched_expected,
    )


@dataclass
class ActionScore:
    expected_count: int
    predicted_count: int
    matched_expected: int
    matched_predicted: int
    recall: float
    precision: float
    f1: float
    unmatched_expected: list[list[str]]
    unmatched_predicted: list[str]


def score_actions(predicted_actions: list[str], expected_groups: list[list[str]]) -> ActionScore:
    """Match each expected concept group to at most one predicted action item.

    A predicted action item matches an expected group if it contains ALL
    keywords in that group (case-insensitive substring). Greedy assignment.
    """
    used_predicted: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []
    for gi, group in enumerate(expected_groups):
        for pi, action in enumerate(predicted_actions):
            if pi in used_predicted:
                continue
            if _contains_all(action, group):
                matched_pairs.append((gi, pi))
                used_predicted.add(pi)
                break

    matched_expected_count = len(matched_pairs)
    matched_predicted_count = len(used_predicted)
    e_count = len(expected_groups)
    p_count = len(predicted_actions)

    recall = matched_expected_count / e_count if e_count else 1.0
    precision = matched_predicted_count / p_count if p_count else (1.0 if e_count == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    matched_groups = {gi for gi, _ in matched_pairs}
    unmatched_expected = [
        expected_groups[i] for i in range(e_count) if i not in matched_groups
    ]
    unmatched_predicted = [
        predicted_actions[i] for i in range(p_count) if i not in used_predicted
    ]

    return ActionScore(
        expected_count=e_count,
        predicted_count=p_count,
        matched_expected=matched_expected_count,
        matched_predicted=matched_predicted_count,
        recall=recall,
        precision=precision,
        f1=f1,
        unmatched_expected=unmatched_expected,
        unmatched_predicted=unmatched_predicted,
    )


def score_effort(predicted_effort: str, expected_effort: str) -> bool:
    return predicted_effort == expected_effort


# ---------------------------------------------------------------------------
# Per-item evaluation + aggregation
# ---------------------------------------------------------------------------


def evaluate_item(predicted: dict, expected: dict) -> dict[str, Any]:
    """Evaluate one prediction against expected labels. Returns a dict report."""
    schema = schema_validity(predicted)
    if not schema["valid"]:
        return {
            "schema": schema,
            "summary": None,
            "risks": None,
            "actions": None,
            "effort_correct": False,
        }

    summary = score_summary(predicted.get("summary", ""), expected)
    risks = score_risks(predicted.get("risks", []), expected.get("expected_risks", []) or [])
    actions = score_actions(
        predicted.get("action_items", []),
        expected.get("expected_action_keywords", []) or [],
    )
    effort_correct = score_effort(
        predicted.get("estimated_effort", ""), expected.get("expected_effort", "")
    )

    return {
        "schema": schema,
        "summary": {
            "keyword_coverage": summary.keyword_coverage,
            "keywords_matched": summary.keywords_matched,
            "keywords_total": summary.keywords_total,
            "forbidden_violations": summary.forbidden_violations,
        },
        "risks": {
            "expected_count": risks.expected_count,
            "predicted_count": risks.predicted_count,
            "matched_count": risks.matched_count,
            "detection_precision": risks.detection_precision,
            "detection_recall": risks.detection_recall,
            "detection_f1": risks.detection_f1,
            "severity_correct_on_matched": risks.severity_correct_on_matched,
            "severity_accuracy_on_matched": risks.severity_accuracy_on_matched,
            "severity_confusion": risks.severity_confusion,
            "unmatched_predicted": risks.unmatched_predicted,
            "unmatched_expected": risks.unmatched_expected,
        },
        "actions": {
            "expected_count": actions.expected_count,
            "predicted_count": actions.predicted_count,
            "matched_expected": actions.matched_expected,
            "matched_predicted": actions.matched_predicted,
            "recall": actions.recall,
            "precision": actions.precision,
            "f1": actions.f1,
            "unmatched_expected": actions.unmatched_expected,
            "unmatched_predicted": actions.unmatched_predicted,
        },
        "effort_correct": effort_correct,
    }


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(per_item: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item evaluations into summary metrics."""
    n = len(per_item)
    if n == 0:
        return {"item_count": 0}

    schema_valid = sum(1 for r in per_item if r["schema"]["valid"])
    scored = [r for r in per_item if r["schema"]["valid"]]

    summary_cov = _safe_mean([r["summary"]["keyword_coverage"] for r in scored])
    forbidden_total = sum(len(r["summary"]["forbidden_violations"]) for r in scored)

    risk_tp = sum(r["risks"]["matched_count"] for r in scored)
    risk_predicted = sum(r["risks"]["predicted_count"] for r in scored)
    risk_expected = sum(r["risks"]["expected_count"] for r in scored)
    risk_precision = risk_tp / risk_predicted if risk_predicted else 0.0
    risk_recall = risk_tp / risk_expected if risk_expected else 0.0
    risk_f1 = (
        2 * risk_precision * risk_recall / (risk_precision + risk_recall)
        if (risk_precision + risk_recall)
        else 0.0
    )
    severity_correct = sum(r["risks"]["severity_correct_on_matched"] for r in scored)
    severity_accuracy = severity_correct / risk_tp if risk_tp else 0.0
    unmatched_predicted_risks = sum(len(r["risks"]["unmatched_predicted"]) for r in scored)

    # Aggregate severity confusion
    confusion: dict[str, dict[str, int]] = {
        sev: {s: 0 for s in VALID_SEVERITIES} for sev in VALID_SEVERITIES
    }
    for r in scored:
        for exp_sev, row in r["risks"]["severity_confusion"].items():
            for pred_sev, count in row.items():
                confusion[exp_sev][pred_sev] += count

    action_tp = sum(r["actions"]["matched_expected"] for r in scored)
    action_predicted = sum(r["actions"]["predicted_count"] for r in scored)
    action_expected = sum(r["actions"]["expected_count"] for r in scored)
    action_matched_predicted = sum(r["actions"]["matched_predicted"] for r in scored)
    action_recall = action_tp / action_expected if action_expected else 0.0
    action_precision = (
        action_matched_predicted / action_predicted if action_predicted else 0.0
    )
    action_f1 = (
        2 * action_precision * action_recall / (action_precision + action_recall)
        if (action_precision + action_recall)
        else 0.0
    )

    effort_correct = sum(1 for r in scored if r["effort_correct"])
    effort_accuracy = effort_correct / len(scored) if scored else 0.0

    # Predicted-effort distribution (helps spot bias toward Medium)
    effort_dist = Counter()

    return {
        "item_count": n,
        "schema_validity": {
            "valid": schema_valid,
            "total": n,
            "rate": schema_valid / n,
        },
        "summary": {
            "avg_keyword_coverage": summary_cov,
            "forbidden_violations_total": forbidden_total,
        },
        "risks": {
            "expected_total": risk_expected,
            "predicted_total": risk_predicted,
            "matched_total": risk_tp,
            "detection_precision": risk_precision,
            "detection_recall": risk_recall,
            "detection_f1": risk_f1,
            "severity_accuracy_on_matched": severity_accuracy,
            "severity_correct_on_matched": severity_correct,
            "severity_confusion": confusion,
            "unmatched_predicted_total": unmatched_predicted_risks,
            "avg_unmatched_predicted_per_item": (
                unmatched_predicted_risks / len(scored) if scored else 0.0
            ),
        },
        "actions": {
            "expected_total": action_expected,
            "predicted_total": action_predicted,
            "matched_expected_total": action_tp,
            "matched_predicted_total": action_matched_predicted,
            "recall": action_recall,
            "precision": action_precision,
            "f1": action_f1,
        },
        "effort": {
            "correct": effort_correct,
            "total": len(scored),
            "accuracy": effort_accuracy,
            "predicted_distribution": dict(effort_dist),
        },
    }

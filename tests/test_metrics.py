"""Unit tests for evals.metrics — synthetic golden cases."""

from __future__ import annotations

import math

from evals.metrics import (
    aggregate,
    evaluate_item,
    schema_validity,
    score_actions,
    score_effort,
    score_risks,
    score_summary,
)

# ---------------------------------------------------------------------------
# schema_validity
# ---------------------------------------------------------------------------


def test_schema_valid_minimal():
    pred = {"summary": "x", "risks": [], "action_items": [], "estimated_effort": "Small"}
    res = schema_validity(pred)
    assert res["valid"]
    assert res["errors"] == []


def test_schema_invalid_effort():
    pred = {"summary": "x", "risks": [], "action_items": [], "estimated_effort": "Huge"}
    res = schema_validity(pred)
    assert not res["valid"]
    assert any("invalid effort" in e for e in res["errors"])


def test_schema_missing_field():
    pred = {"summary": "x", "risks": [], "action_items": []}
    res = schema_validity(pred)
    assert not res["valid"]
    assert any("missing fields" in e for e in res["errors"])


def test_schema_invalid_risk_severity():
    pred = {
        "summary": "x",
        "risks": [{"description": "d", "severity": "Critical"}],
        "action_items": [],
        "estimated_effort": "Small",
    }
    res = schema_validity(pred)
    assert not res["valid"]
    assert any("invalid severity" in e for e in res["errors"])


# ---------------------------------------------------------------------------
# score_summary
# ---------------------------------------------------------------------------


def test_summary_full_coverage_case_insensitive():
    s = score_summary("Onboarding REDIS for Reporting", {"summary_keywords": ["redis", "reporting"]})
    assert s.keyword_coverage == 1.0
    assert s.keywords_matched == 2
    assert s.forbidden_violations == []


def test_summary_partial_coverage():
    s = score_summary("Just redis", {"summary_keywords": ["redis", "reporting"]})
    assert s.keyword_coverage == 0.5
    assert s.keywords_matched == 1


def test_summary_forbidden_violation_detected():
    s = score_summary(
        "We had a production outage today",
        {
            "summary_keywords": ["production"],
            "summary_must_not_claim": ["production outage"],
        },
    )
    assert s.forbidden_violations == ["production outage"]


def test_summary_no_keywords_returns_one():
    s = score_summary("anything", {})
    assert s.keyword_coverage == 1.0
    assert s.keywords_total == 0


# ---------------------------------------------------------------------------
# score_risks
# ---------------------------------------------------------------------------


def test_risks_perfect_match():
    predicted = [
        {"description": "Misconfiguration could break integration", "severity": "Medium"},
        {"description": "Acceptance criteria are unclear", "severity": "Medium"},
    ]
    expected = [
        {"keywords": ["misconfiguration"], "severity": "Medium"},
        {"keywords": ["acceptance criteria"], "severity": "Medium"},
    ]
    r = score_risks(predicted, expected)
    assert r.matched_count == 2
    assert math.isclose(r.detection_f1, 1.0)
    assert r.severity_accuracy_on_matched == 1.0
    assert r.unmatched_predicted == []


def test_risks_severity_separated_from_detection():
    predicted = [{"description": "Misconfiguration", "severity": "Low"}]
    expected = [{"keywords": ["misconfiguration"], "severity": "High"}]
    r = score_risks(predicted, expected)
    assert r.matched_count == 1
    assert r.detection_recall == 1.0
    assert r.severity_correct_on_matched == 0
    assert r.severity_accuracy_on_matched == 0.0
    assert r.severity_confusion["High"]["Low"] == 1


def test_risks_hallucinated_predicted_counted():
    predicted = [
        {"description": "real risk: misconfiguration", "severity": "Medium"},
        {"description": "spurious security breach", "severity": "High"},
    ]
    expected = [{"keywords": ["misconfiguration"], "severity": "Medium"}]
    r = score_risks(predicted, expected)
    assert r.matched_count == 1
    assert r.detection_precision == 0.5
    assert len(r.unmatched_predicted) == 1
    assert r.unmatched_predicted[0]["description"] == "spurious security breach"


def test_risks_missed_expected_counted():
    predicted = []
    expected = [{"keywords": ["misconfiguration"], "severity": "Medium"}]
    r = score_risks(predicted, expected)
    assert r.matched_count == 0
    assert r.detection_recall == 0.0
    assert len(r.unmatched_expected) == 1


def test_risks_both_empty_perfect():
    r = score_risks([], [])
    # No risks expected, none predicted → trivially perfect.
    assert r.detection_precision == 1.0
    assert r.detection_recall == 1.0
    assert r.detection_f1 == 1.0


# ---------------------------------------------------------------------------
# score_actions
# ---------------------------------------------------------------------------


def test_actions_all_keywords_required():
    predicted = ["Provision azure resources", "Run integration tests"]
    expected = [["azure", "provision"], ["integration", "test"]]
    a = score_actions(predicted, expected)
    assert a.recall == 1.0
    assert a.precision == 1.0


def test_actions_partial_keyword_does_not_match():
    predicted = ["Provision storage account"]  # missing 'azure'
    expected = [["azure", "provision"]]
    a = score_actions(predicted, expected)
    assert a.matched_expected == 0
    assert a.recall == 0.0
    # predicted item did not match → counted against precision
    assert a.precision == 0.0


def test_actions_extra_predicted_lowers_precision():
    predicted = [
        "Provision azure resources",
        "Schedule a happy hour",
    ]
    expected = [["azure", "provision"]]
    a = score_actions(predicted, expected)
    assert a.recall == 1.0
    assert a.precision == 0.5
    assert a.unmatched_predicted == ["Schedule a happy hour"]


def test_actions_greedy_one_to_one():
    # Both expected groups could match the same predicted item; greedy must
    # only assign each predicted item once.
    predicted = ["Provision azure resources for the integration test"]
    expected = [["azure", "provision"], ["integration", "test"]]
    a = score_actions(predicted, expected)
    assert a.matched_expected == 1
    assert a.matched_predicted == 1


# ---------------------------------------------------------------------------
# score_effort & evaluate_item
# ---------------------------------------------------------------------------


def test_effort_exact_match():
    assert score_effort("Small", "Small") is True
    assert score_effort("Small", "Medium") is False


def test_evaluate_item_invalid_schema_returns_early():
    pred = {"summary": "x", "risks": [], "action_items": []}  # missing effort
    out = evaluate_item(pred, {"expected_effort": "Small"})
    assert out["schema"]["valid"] is False
    assert out["summary"] is None
    assert out["effort_correct"] is False


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def _sample_eval(summary_cov: float, effort_correct: bool) -> dict:
    """Build a minimal per-item eval dict matching evaluate_item shape."""
    return {
        "schema": {"valid": True, "errors": []},
        "summary": {
            "keyword_coverage": summary_cov,
            "keywords_matched": 0,
            "keywords_total": 0,
            "forbidden_violations": [],
        },
        "risks": {
            "expected_count": 1,
            "predicted_count": 1,
            "matched_count": 1,
            "detection_precision": 1.0,
            "detection_recall": 1.0,
            "detection_f1": 1.0,
            "severity_correct_on_matched": 1,
            "severity_accuracy_on_matched": 1.0,
            "severity_confusion": {
                "High": {"High": 0, "Medium": 0, "Low": 0},
                "Medium": {"High": 0, "Medium": 1, "Low": 0},
                "Low": {"High": 0, "Medium": 0, "Low": 0},
            },
            "unmatched_predicted": [],
            "unmatched_expected": [],
        },
        "actions": {
            "expected_count": 2,
            "predicted_count": 2,
            "matched_expected": 2,
            "matched_predicted": 2,
            "recall": 1.0,
            "precision": 1.0,
            "f1": 1.0,
            "unmatched_expected": [],
            "unmatched_predicted": [],
        },
        "effort_correct": effort_correct,
    }


def test_aggregate_basic():
    items = [_sample_eval(1.0, True), _sample_eval(0.5, False)]
    agg = aggregate(items)
    assert agg["item_count"] == 2
    assert agg["schema_validity"]["rate"] == 1.0
    assert math.isclose(agg["summary"]["avg_keyword_coverage"], 0.75)
    assert agg["effort"]["accuracy"] == 0.5
    assert agg["risks"]["detection_f1"] == 1.0


def test_aggregate_handles_invalid_schema_items():
    invalid = {
        "schema": {"valid": False, "errors": ["bad"]},
        "summary": None,
        "risks": None,
        "actions": None,
        "effort_correct": False,
    }
    valid = _sample_eval(1.0, True)
    agg = aggregate([invalid, valid])
    assert agg["schema_validity"]["valid"] == 1
    assert agg["schema_validity"]["total"] == 2
    # only the valid item contributes to component metrics
    assert agg["summary"]["avg_keyword_coverage"] == 1.0
    assert agg["effort"]["total"] == 1


def test_aggregate_empty():
    agg = aggregate([])
    assert agg == {"item_count": 0}

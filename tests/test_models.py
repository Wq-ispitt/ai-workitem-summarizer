"""Tests for data models."""

from workitem_summarizer.models import Effort, Risk, Severity, WorkItemSummary


def test_work_item_summary_to_dict():
    summary = WorkItemSummary(
        summary="Implement caching for API responses",
        risks=[
            Risk(description="Cache invalidation complexity", severity=Severity.MEDIUM),
            Risk(description="Memory pressure under load", severity=Severity.HIGH),
        ],
        action_items=["Design cache key strategy", "Add cache eviction policy"],
        estimated_effort=Effort.MEDIUM,
    )
    result = summary.to_dict()

    assert result["summary"] == "Implement caching for API responses"
    assert len(result["risks"]) == 2
    assert result["risks"][0]["severity"] == "Medium"
    assert result["risks"][1]["severity"] == "High"
    assert len(result["action_items"]) == 2
    assert result["estimated_effort"] == "Medium"


def test_empty_risks_and_actions():
    summary = WorkItemSummary(summary="Simple doc update")
    result = summary.to_dict()

    assert result["risks"] == []
    assert result["action_items"] == []
    assert result["estimated_effort"] == "Medium"

"""Trivial non-LLM baselines for comparison against the model.

Useful for sanity-checking the eval harness and grounding the reported
metrics: a model should at minimum beat 'title-only' on every dimension.
"""

from __future__ import annotations


def naive_summary(work_item: dict) -> dict:
    """Return a 'naive' summary: title-as-summary, no risks, no actions, Medium effort."""
    fields = work_item.get("fields", {}) or {}
    title = fields.get("System.Title", "") or ""
    return {
        "summary": title,
        "risks": [],
        "action_items": [],
        "estimated_effort": "Medium",
    }

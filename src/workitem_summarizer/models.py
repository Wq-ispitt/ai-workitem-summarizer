"""Data models for work item summaries."""

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Effort(str, Enum):
    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"


@dataclass
class Risk:
    description: str
    severity: Severity


@dataclass
class WorkItemSummary:
    summary: str
    risks: list[Risk] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    estimated_effort: Effort = Effort.MEDIUM

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "risks": [{"description": r.description, "severity": r.severity.value} for r in self.risks],
            "action_items": self.action_items,
            "estimated_effort": self.estimated_effort.value,
        }

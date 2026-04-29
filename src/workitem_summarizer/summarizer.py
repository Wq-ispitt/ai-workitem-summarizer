"""LLM-based work item summarizer using Azure OpenAI structured output."""

import json
import os

from openai import AzureOpenAI

from .models import Effort, Risk, Severity, WorkItemSummary

SYSTEM_PROMPT = """You are an expert Azure DevOps work item analyst. Given a work item's details, produce a structured JSON summary.

Rules:
- "summary": A concise 1-3 sentence summary of what the work item is about.
- "risks": An array of risks. Each risk has a "description" (string) and "severity" ("High", "Medium", or "Low").
  - High: blocking issues, security concerns, data loss potential, deadline at risk
  - Medium: moderate complexity, dependencies on other teams, unclear requirements
  - Low: minor concerns, nice-to-have improvements
- "action_items": An array of specific next steps someone should take.
- "estimated_effort": One of "Small" (< 1 day), "Medium" (1-3 days), or "Large" (> 3 days).

Be specific and actionable. Do not be vague. Base your analysis only on the provided work item data."""

RESPONSE_SCHEMA = {
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
                            "severity": {"type": "string", "enum": ["High", "Medium", "Low"]},
                        },
                        "required": ["description", "severity"],
                        "additionalProperties": False,
                    },
                },
                "action_items": {"type": "array", "items": {"type": "string"}},
                "estimated_effort": {"type": "string", "enum": ["Small", "Medium", "Large"]},
            },
            "required": ["summary", "risks", "action_items", "estimated_effort"],
            "additionalProperties": False,
        },
    },
}


class WorkItemSummarizer:
    """Summarizes ADO work items using Azure OpenAI with structured output."""

    def __init__(self, endpoint: str, deployment: str = "gpt-4.1-mini") -> None:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if api_key:
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2024-12-01-preview",
            )
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version="2024-12-01-preview",
            )
        self.deployment = deployment

    def _format_work_item(self, work_item: dict) -> str:
        """Extract relevant fields from an ADO work item into a readable prompt."""
        fields = work_item.get("fields", {})
        parts = [
            f"ID: {work_item.get('id', 'N/A')}",
            f"Title: {fields.get('System.Title', 'N/A')}",
            f"Type: {fields.get('System.WorkItemType', 'N/A')}",
            f"State: {fields.get('System.State', 'N/A')}",
            f"Assigned To: {fields.get('System.AssignedTo', {}).get('displayName', 'Unassigned')}",
            f"Area Path: {fields.get('System.AreaPath', 'N/A')}",
            f"Iteration Path: {fields.get('System.IterationPath', 'N/A')}",
            f"Description: {fields.get('System.Description', 'No description')}",
            f"Acceptance Criteria: {fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', 'None')}",
            f"Tags: {fields.get('System.Tags', 'None')}",
        ]
        return "\n".join(parts)

    def summarize(self, work_item: dict) -> WorkItemSummary:
        """Summarize a single work item."""
        user_content = self._format_work_item(work_item)

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format=RESPONSE_SCHEMA,
            temperature=0.2,
        )

        raw = json.loads(response.choices[0].message.content)
        return WorkItemSummary(
            summary=raw["summary"],
            risks=[Risk(description=r["description"], severity=Severity(r["severity"])) for r in raw["risks"]],
            action_items=raw["action_items"],
            estimated_effort=Effort(raw["estimated_effort"]),
        )

    def summarize_batch(self, work_items: list[dict]) -> list[dict]:
        """Summarize multiple work items and return dicts."""
        results = []
        for item in work_items:
            summary = self.summarize(item)
            results.append({
                "work_item_id": item.get("id"),
                "title": item.get("fields", {}).get("System.Title", ""),
                **summary.to_dict(),
            })
        return results

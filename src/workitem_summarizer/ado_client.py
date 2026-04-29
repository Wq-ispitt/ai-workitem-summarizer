"""Azure DevOps REST API client for reading work items."""

import base64
import os

import httpx


class AdoClient:
    """Reads work items from Azure DevOps using a Personal Access Token."""

    def __init__(self, organization: str, project: str, pat: str | None = None) -> None:
        self.organization = organization
        self.project = project
        self.base_url = f"https://dev.azure.com/{organization}/{project}/_apis"
        self._pat = pat or os.environ.get("ADO_PAT", "")
        if not self._pat:
            raise ValueError("ADO_PAT environment variable or pat parameter is required")

    def _headers(self) -> dict[str, str]:
        encoded = base64.b64encode(f":{self._pat}".encode()).decode()
        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
        }

    def get_work_item(self, work_item_id: int) -> dict:
        """Fetch a single work item by ID."""
        url = f"{self.base_url}/wit/workitems/{work_item_id}?$expand=all&api-version=7.1"
        with httpx.Client() as client:
            response = client.get(url, headers=self._headers())
            response.raise_for_status()
            return response.json()

    def get_work_items(self, ids: list[int]) -> list[dict]:
        """Fetch multiple work items by IDs."""
        url = f"{self.base_url}/wit/workitems?ids={','.join(str(i) for i in ids)}&$expand=all&api-version=7.1"
        with httpx.Client() as client:
            response = client.get(url, headers=self._headers())
            response.raise_for_status()
            return response.json().get("value", [])

    def query_work_items(self, wiql: str, top: int = 20) -> list[dict]:
        """Run a WIQL query and return full work items."""
        url = f"{self.base_url}/wit/wiql?api-version=7.1"
        with httpx.Client() as client:
            response = client.post(url, headers=self._headers(), json={"query": wiql})
            response.raise_for_status()
            ids = [item["id"] for item in response.json().get("workItems", [])[:top]]

        if not ids:
            return []
        return self.get_work_items(ids)

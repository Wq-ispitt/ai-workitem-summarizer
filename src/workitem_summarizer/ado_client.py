"""Azure DevOps REST API client for reading work items.

Supports two auth modes:
- PAT (Basic auth) via ``ADO_PAT`` env var or ``pat`` parameter.
- Entra ID bearer token via ``az account get-access-token`` when no PAT is
  provided. This is required for organizations that disable PAT scopes.
"""

import base64
import json
import os
import shutil
import subprocess

import httpx

ADO_RESOURCE_ID = "499b84ac-1321-427f-aa17-267ca6975798"


class AdoClient:
    """Reads work items from Azure DevOps using PAT or Entra ID auth."""

    def __init__(self, organization: str, project: str, pat: str | None = None) -> None:
        self.organization = organization
        self.project = project
        self.base_url = f"https://dev.azure.com/{organization}/{project}/_apis"
        self._pat = pat if pat is not None else os.environ.get("ADO_PAT", "")
        self._bearer_token: str | None = None
        if not self._pat:
            self._bearer_token = self._acquire_entra_token()

    @staticmethod
    def _acquire_entra_token() -> str:
        az = shutil.which("az")
        if not az:
            raise ValueError(
                "ADO_PAT not set and 'az' CLI not found; install Azure CLI or set ADO_PAT"
            )
        try:
            result = subprocess.run(
                [az, "account", "get-access-token", "--resource", ADO_RESOURCE_ID, "-o", "json"],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"Failed to acquire ADO token via 'az account get-access-token': {exc.stderr.strip()}"
            ) from exc
        return json.loads(result.stdout)["accessToken"]

    def _headers(self) -> dict[str, str]:
        if self._pat:
            encoded = base64.b64encode(f":{self._pat}".encode()).decode()
            auth = f"Basic {encoded}"
        else:
            auth = f"Bearer {self._bearer_token}"
        return {"Authorization": auth, "Content-Type": "application/json"}

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

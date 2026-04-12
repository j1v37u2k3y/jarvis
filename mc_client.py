"""
JARVIS ↔ Mission Control REST client.

Wraps Mission Control's HTTP API for task creation, inbox polling, and decisions.
Graceful degradation: if MC is offline, returns empty results and logs a warning.
"""

import logging
import os
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("jarvis.mc_client")

MC_BASE_URL = os.getenv("MC_BASE_URL", "http://localhost:3000")
MC_TIMEOUT = 5.0  # seconds


def _load_mc_token() -> str:
    """Load MC_API_TOKEN from env or fall back to MC's .env file."""
    token = os.getenv("MC_API_TOKEN", "")
    if token:
        return token
    # Fallback: read from MC's .env if installed in expected location
    mc_env = Path.home() / "IdeaProjects" / "mission-control" / "mission-control" / ".env"
    if mc_env.exists():
        for line in mc_env.read_text().splitlines():
            line = line.strip()
            if line.startswith("MC_API_TOKEN="):
                return line.partition("=")[2].strip().strip('"').strip("'")
    return ""


MC_API_TOKEN = _load_mc_token()


class MissionControlClient:
    """Async REST client for Mission Control API."""

    def __init__(self, base_url: str = MC_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._healthy: bool | None = None  # cache last health check

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Make a request to MC. Returns None on failure."""
        url = f"{self.base_url}{path}"
        if MC_API_TOKEN:
            headers = kwargs.pop("headers", {}) or {}
            headers["Authorization"] = f"Bearer {MC_API_TOKEN}"
            kwargs["headers"] = headers
        try:
            async with httpx.AsyncClient(timeout=MC_TIMEOUT) as client:
                resp = await client.request(method, url, **kwargs)
                resp.raise_for_status()
                self._healthy = True
                return resp.json()
        except httpx.ConnectError:
            if self._healthy is not False:
                log.warning(f"Mission Control unreachable at {self.base_url}")
            self._healthy = False
            return None
        except httpx.HTTPStatusError as e:
            log.warning(f"MC {method} {path} returned {e.response.status_code}: {e.response.text[:200]}")
            return None
        except Exception as e:
            log.warning(f"MC {method} {path} failed: {e}")
            return None

    async def is_healthy(self) -> bool:
        """Check if MC is reachable."""
        result = await self._request("GET", "/api/server-status")
        return result is not None

    # ---------------------------------------------------------------------------
    # Tasks
    # ---------------------------------------------------------------------------

    async def create_task(
        self,
        title: str,
        description: str = "",
        importance: str = "important",
        urgency: str = "not-urgent",
        assigned_to: str = "developer",
        project_id: str | None = None,
    ) -> dict | None:
        """Create a task. Returns the created task dict or None on failure."""
        payload: dict[str, Any] = {
            "title": title,
            "description": description,
            "importance": importance,
            "urgency": urgency,
            "kanban": "not-started",
            "projectId": project_id,
            "milestoneId": None,
            "assignedTo": assigned_to,
            "collaborators": [],
            "dailyActions": [],
            "subtasks": [],
            "blockedBy": [],
            "estimatedMinutes": None,
            "actualMinutes": None,
            "acceptanceCriteria": [],
            "comments": [],
            "tags": [],
            "notes": "",
        }
        result = await self._request("POST", "/api/tasks", json=payload)
        if result:
            log.info(f"Created MC task: {title} → {assigned_to}")
        return result

    async def list_tasks(
        self,
        assigned_to: str | None = None,
        kanban: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List tasks with optional filters. Returns empty list on failure."""
        params: dict[str, Any] = {"limit": limit}
        if assigned_to:
            params["assignedTo"] = assigned_to
        if kanban:
            params["kanban"] = kanban
        result = await self._request("GET", "/api/tasks", params=params)
        if result and isinstance(result, dict):
            return result.get("tasks", [])
        return []

    async def get_task(self, task_id: str) -> dict | None:
        """Get a single task by ID."""
        result = await self._request("GET", "/api/tasks", params={"id": task_id})
        if result and isinstance(result, dict):
            tasks = result.get("tasks", [])
            return tasks[0] if tasks else None
        return None

    async def update_task(self, task_id: str, **fields) -> dict | None:
        """Update a task. Pass fields like kanban='done', assignedTo='me', etc."""
        payload = {"id": task_id, **fields}
        return await self._request("PUT", "/api/tasks", json=payload)

    async def complete_task(self, task_id: str) -> dict | None:
        """Mark a task as done."""
        return await self.update_task(task_id, kanban="done")

    # ---------------------------------------------------------------------------
    # Inbox
    # ---------------------------------------------------------------------------

    async def list_inbox(
        self,
        agent: str = "me",
        status: str = "unread",
        limit: int = 50,
    ) -> list[dict]:
        """List inbox messages for an agent."""
        params = {"agent": agent, "status": status, "limit": limit}
        result = await self._request("GET", "/api/inbox", params=params)
        if result and isinstance(result, dict):
            return result.get("messages", [])
        return []

    async def mark_inbox_read(self, message_id: str) -> dict | None:
        """Mark an inbox message as read."""
        return await self._request("PUT", "/api/inbox", json={"id": message_id, "status": "read"})

    # ---------------------------------------------------------------------------
    # Decisions
    # ---------------------------------------------------------------------------

    async def list_decisions(self, status: str = "pending") -> list[dict]:
        """List decisions awaiting answer."""
        result = await self._request("GET", "/api/decisions", params={"status": status})
        if result and isinstance(result, dict):
            return result.get("decisions", [])
        return []

    async def answer_decision(self, decision_id: str, answer: str) -> dict | None:
        """Answer a pending decision."""
        return await self._request(
            "PUT", "/api/decisions", json={"id": decision_id, "answer": answer, "status": "answered"}
        )

    # ---------------------------------------------------------------------------
    # Daemon
    # ---------------------------------------------------------------------------

    async def get_daemon_status(self) -> dict | None:
        """Get daemon status and config."""
        return await self._request("GET", "/api/daemon")

    async def start_daemon(self) -> dict | None:
        """Start the MC daemon."""
        return await self._request("POST", "/api/daemon", json={"action": "start"})

    async def stop_daemon(self) -> dict | None:
        """Stop the MC daemon."""
        return await self._request("POST", "/api/daemon", json={"action": "stop"})


# Module-level singleton
mc_client = MissionControlClient()

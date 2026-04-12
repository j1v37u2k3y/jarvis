"""
Integration tests against a live Mission Control instance.

These tests catch API contract drift that mocked unit tests cannot:
- Wrong endpoint paths or HTTP methods
- Schema changes in MC's task/inbox/decision payloads
- Auth or middleware changes

Skipped automatically when MC is not reachable at localhost:3000.

To run:  pytest tests/test_mc_integration.py -m requires_mc
To skip: pytest -m "not requires_mc"  (default in CI)
"""

import contextlib

import httpx
import pytest

from mc_client import MissionControlClient

pytestmark = pytest.mark.requires_mc


def _mc_reachable() -> bool:
    """Quick sync check if MC is up — used by skipif."""
    try:
        r = httpx.get("http://localhost:3000/api/server-status", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


# Skip the entire module if MC is offline
skip_if_no_mc = pytest.mark.skipif(
    not _mc_reachable(),
    reason="Mission Control not running at localhost:3000",
)


@pytest.fixture
async def mc():
    """Fresh client for each test."""
    return MissionControlClient()


@pytest.fixture
async def cleanup_task(mc):
    """Yields a list to register task IDs for cleanup after the test."""
    created_ids: list[str] = []
    yield created_ids
    # Cleanup all tasks created during the test
    async with httpx.AsyncClient() as client:
        for task_id in created_ids:
            with contextlib.suppress(Exception):
                await client.delete(f"http://localhost:3000/api/tasks?id={task_id}&hard=true")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@skip_if_no_mc
class TestHealth:
    @pytest.mark.asyncio
    async def test_is_healthy_against_live_mc(self, mc):
        assert await mc.is_healthy() is True


# ---------------------------------------------------------------------------
# Task lifecycle
# ---------------------------------------------------------------------------


@skip_if_no_mc
class TestTaskLifecycle:
    @pytest.mark.asyncio
    async def test_create_returns_task_with_id(self, mc, cleanup_task):
        task = await mc.create_task(
            title="Integration test: create",
            description="Created by JARVIS integration test",
            assigned_to="me",
        )
        assert task is not None
        assert "id" in task
        assert task["title"] == "Integration test: create"
        assert task["assignedTo"] == "me"
        assert task["kanban"] == "not-started"
        cleanup_task.append(task["id"])

    @pytest.mark.asyncio
    async def test_list_includes_created_task(self, mc, cleanup_task):
        task = await mc.create_task(title="Integration test: list", assigned_to="me")
        assert task is not None
        cleanup_task.append(task["id"])

        tasks = await mc.list_tasks(assigned_to="me", kanban="not-started")
        assert any(t["id"] == task["id"] for t in tasks)

    @pytest.mark.asyncio
    async def test_complete_task_changes_kanban(self, mc, cleanup_task):
        task = await mc.create_task(title="Integration test: complete", assigned_to="me")
        assert task is not None
        cleanup_task.append(task["id"])

        result = await mc.complete_task(task["id"])
        assert result is not None
        assert result["kanban"] == "done"
        assert result.get("completedAt") is not None

    @pytest.mark.asyncio
    async def test_eisenhower_quadrants_persist(self, mc, cleanup_task):
        """Verify importance/urgency map correctly to MC's task model."""
        task = await mc.create_task(
            title="Integration test: urgent+important",
            importance="important",
            urgency="urgent",
            assigned_to="me",
        )
        assert task is not None
        cleanup_task.append(task["id"])
        assert task["importance"] == "important"
        assert task["urgency"] == "urgent"


# ---------------------------------------------------------------------------
# Inbox
# ---------------------------------------------------------------------------


@skip_if_no_mc
class TestInbox:
    @pytest.mark.asyncio
    async def test_list_inbox_returns_messages_array(self, mc):
        # Even when empty, should return a list (not None)
        messages = await mc.list_inbox(agent="me", status="unread")
        assert isinstance(messages, list)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


@skip_if_no_mc
class TestDecisions:
    @pytest.mark.asyncio
    async def test_list_decisions_returns_array(self, mc):
        decisions = await mc.list_decisions(status="pending")
        assert isinstance(decisions, list)


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------


@skip_if_no_mc
class TestDaemon:
    @pytest.mark.asyncio
    async def test_daemon_status_has_expected_shape(self, mc):
        status = await mc.get_daemon_status()
        assert status is not None
        assert "status" in status
        assert "config" in status
        assert "isRunning" in status

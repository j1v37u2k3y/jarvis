"""Tests for the Mission Control REST client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mc_client import MissionControlClient


def _mock_response(json_data=None, status_code=200, raise_for_status=None):
    """Build a mock httpx Response."""
    resp = MagicMock()
    resp.json.return_value = json_data or {}
    resp.status_code = status_code
    resp.text = str(json_data)
    if raise_for_status:
        resp.raise_for_status.side_effect = raise_for_status
    else:
        resp.raise_for_status.return_value = None
    return resp


def _mock_client_ctx(response):
    """Build a mocked async httpx.AsyncClient context manager."""
    client = MagicMock()
    client.request = AsyncMock(return_value=response)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx, client


class TestCreateTask:
    @pytest.mark.asyncio
    async def test_create_task_builds_correct_payload(self):
        ctx, client = _mock_client_ctx(_mock_response({"id": "task_abc", "title": "Test"}))
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", return_value=ctx):
            result = await mc.create_task(
                title="Build a thing",
                description="Description here",
                importance="important",
                urgency="urgent",
                assigned_to="developer",
            )
        assert result["id"] == "task_abc"
        call = client.request.call_args
        assert call.args[0] == "POST"
        assert call.args[1].endswith("/api/tasks")
        payload = call.kwargs["json"]
        assert payload["title"] == "Build a thing"
        assert payload["description"] == "Description here"
        assert payload["importance"] == "important"
        assert payload["urgency"] == "urgent"
        assert payload["assignedTo"] == "developer"
        assert payload["kanban"] == "not-started"

    @pytest.mark.asyncio
    async def test_create_task_returns_none_on_failure(self):
        ctx, _ = _mock_client_ctx(
            _mock_response(
                raise_for_status=httpx.HTTPStatusError(
                    "Server error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500, text="boom"),
                )
            )
        )
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", return_value=ctx):
            result = await mc.create_task(title="x")
        assert result is None


class TestListTasks:
    @pytest.mark.asyncio
    async def test_list_tasks_returns_tasks(self):
        ctx, _ = _mock_client_ctx(_mock_response({"tasks": [{"id": "task_1"}, {"id": "task_2"}]}))
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", return_value=ctx):
            tasks = await mc.list_tasks()
        assert len(tasks) == 2
        assert tasks[0]["id"] == "task_1"

    @pytest.mark.asyncio
    async def test_list_tasks_with_filters(self):
        ctx, client = _mock_client_ctx(_mock_response({"tasks": []}))
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", return_value=ctx):
            await mc.list_tasks(assigned_to="developer", kanban="in-progress", limit=10)
        params = client.request.call_args.kwargs["params"]
        assert params["assignedTo"] == "developer"
        assert params["kanban"] == "in-progress"
        assert params["limit"] == 10

    @pytest.mark.asyncio
    async def test_list_tasks_returns_empty_on_failure(self):
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", side_effect=httpx.ConnectError("nope")):
            tasks = await mc.list_tasks()
        assert tasks == []


class TestUpdateTask:
    @pytest.mark.asyncio
    async def test_complete_task_sends_done_status(self):
        ctx, client = _mock_client_ctx(_mock_response({"id": "task_1", "kanban": "done"}))
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", return_value=ctx):
            await mc.complete_task("task_1")
        payload = client.request.call_args.kwargs["json"]
        assert payload["id"] == "task_1"
        assert payload["kanban"] == "done"


class TestInbox:
    @pytest.mark.asyncio
    async def test_list_inbox_returns_messages(self):
        ctx, _ = _mock_client_ctx(
            _mock_response({"messages": [{"id": "msg_1", "from": "developer", "subject": "Done"}]})
        )
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", return_value=ctx):
            messages = await mc.list_inbox()
        assert len(messages) == 1
        assert messages[0]["from"] == "developer"

    @pytest.mark.asyncio
    async def test_mark_inbox_read(self):
        ctx, client = _mock_client_ctx(_mock_response({"id": "msg_1", "status": "read"}))
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", return_value=ctx):
            await mc.mark_inbox_read("msg_1")
        payload = client.request.call_args.kwargs["json"]
        assert payload["id"] == "msg_1"
        assert payload["status"] == "read"


class TestDecisions:
    @pytest.mark.asyncio
    async def test_list_decisions(self):
        ctx, _ = _mock_client_ctx(_mock_response({"decisions": [{"id": "dec_1", "question": "Ship it?"}]}))
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", return_value=ctx):
            decisions = await mc.list_decisions()
        assert len(decisions) == 1
        assert decisions[0]["question"] == "Ship it?"

    @pytest.mark.asyncio
    async def test_answer_decision(self):
        ctx, client = _mock_client_ctx(_mock_response({"id": "dec_1", "answer": "yes"}))
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", return_value=ctx):
            await mc.answer_decision("dec_1", "yes")
        payload = client.request.call_args.kwargs["json"]
        assert payload["id"] == "dec_1"
        assert payload["answer"] == "yes"
        assert payload["status"] == "answered"


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_offline_returns_none_for_create(self):
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", side_effect=httpx.ConnectError("offline")):
            result = await mc.create_task(title="x")
        assert result is None

    @pytest.mark.asyncio
    async def test_offline_returns_empty_for_list(self):
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", side_effect=httpx.ConnectError("offline")):
            tasks = await mc.list_tasks()
            messages = await mc.list_inbox()
            decisions = await mc.list_decisions()
        assert tasks == []
        assert messages == []
        assert decisions == []

    @pytest.mark.asyncio
    async def test_is_healthy_false_when_offline(self):
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", side_effect=httpx.ConnectError("offline")):
            assert await mc.is_healthy() is False

    @pytest.mark.asyncio
    async def test_is_healthy_true_when_online(self):
        ctx, _ = _mock_client_ctx(_mock_response({"status": "ok"}))
        mc = MissionControlClient()
        with patch("mc_client.httpx.AsyncClient", return_value=ctx):
            assert await mc.is_healthy() is True

"""Tests for the tool-use loop in llm.generate_response."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import llm


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(tool_id: str, name: str, args: dict):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=args)


def _response(content, stop_reason: str = "end_turn"):
    resp = MagicMock()
    resp.content = content
    resp.stop_reason = stop_reason
    resp.usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    resp.model = "claude-haiku-4-5-20251001"
    return resp


def _make_client(*responses):
    """Build an AsyncAnthropic-like client that returns the given responses in order."""
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=list(responses))
    return client


class _FakeBridge:
    """Lightweight stand-in for RickBridge."""

    def __init__(self, tools=None, principles="### Values\nhonesty\n", call_text="rick says hi"):
        self.tools_for_anthropic = (
            tools
            if tools is not None
            else [
                {
                    "name": "rick_mantra",
                    "description": "Pull a random mantra",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ]
        )
        self.principles_block = principles
        self.call_tool = AsyncMock(return_value=call_text)


def _common_kwargs(client, **overrides):
    """Default kwargs for generate_response — most tests share these."""
    task_mgr = MagicMock()
    task_mgr.get_active_tasks_summary.return_value = "no active tasks"
    dispatch_registry = MagicMock()
    dispatch_registry.format_for_prompt.return_value = "no dispatches"
    base = {
        "text": "hi rick",
        "client": client,
        "task_mgr": task_mgr,
        "projects": [],
        "conversation_history": [],
        "ctx_cache": {
            "screen": "",
            "calendar": "no events",
            "mail": "no mail",
            "weather": "sunny",
        },
        "dispatch_registry": dispatch_registry,
        "user_name": "sir",
        "project_dir": "/tmp",
    }
    base.update(overrides)
    return base


class TestNoBridge:
    @pytest.mark.asyncio
    async def test_no_bridge_no_tools(self):
        client = _make_client(_response([_text_block("Right away, sir.")]))
        out = await llm.generate_response(**_common_kwargs(client, rick_bridge=None))
        assert out == "Right away, sir."
        # Single API call, no tools= passed
        call = client.messages.create.call_args
        assert "tools" not in call.kwargs
        assert call.kwargs["max_tokens"] == llm.NO_TOOL_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_no_bridge_no_principles_in_system(self):
        client = _make_client(_response([_text_block("ok")]))
        await llm.generate_response(**_common_kwargs(client, rick_bridge=None))
        system = client.messages.create.call_args.kwargs["system"]
        # Principles section header is in the template, but the body slot is empty.
        assert "OPERATOR PRINCIPLES" in system
        # No actual principle text — the rick body line is absent.
        assert "### Values" not in system


class TestWithBridge:
    @pytest.mark.asyncio
    async def test_principles_in_system(self):
        client = _make_client(_response([_text_block("ok")]))
        bridge = _FakeBridge(principles="### Values\nhonesty above all")
        await llm.generate_response(**_common_kwargs(client, rick_bridge=bridge))
        system = client.messages.create.call_args.kwargs["system"]
        assert "honesty above all" in system
        assert "### Values" in system

    @pytest.mark.asyncio
    async def test_tools_passed_when_bridge_present(self):
        client = _make_client(_response([_text_block("ok")]))
        bridge = _FakeBridge()
        await llm.generate_response(**_common_kwargs(client, rick_bridge=bridge))
        call = client.messages.create.call_args
        assert call.kwargs["tools"] == bridge.tools_for_anthropic
        assert call.kwargs["max_tokens"] == llm.TOOL_USE_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_tool_use_loop_happy_path(self):
        # Turn 1: model asks to call rick_mantra. Turn 2: model returns final text.
        first = _response(
            [_tool_use_block("toolu_1", "rick_mantra", {})],
            stop_reason="tool_use",
        )
        second = _response([_text_block("Sir, the mantra: keep building.")])
        client = _make_client(first, second)
        bridge = _FakeBridge(call_text="keep building")

        out = await llm.generate_response(**_common_kwargs(client, rick_bridge=bridge))
        assert out == "Sir, the mantra: keep building."
        bridge.call_tool.assert_awaited_with("rick_mantra", {})
        assert client.messages.create.await_count == 2

    @pytest.mark.asyncio
    async def test_iteration_cap(self):
        # Model keeps asking for tool use. Loop should cap and return whatever it has.
        loops = [_response([_tool_use_block(f"t{i}", "rick_mantra", {})], stop_reason="tool_use") for i in range(10)]
        client = _make_client(*loops)
        bridge = _FakeBridge()
        out = await llm.generate_response(**_common_kwargs(client, rick_bridge=bridge))
        # No text block ever returned → empty string is acceptable; key is no infinite loop
        assert isinstance(out, str)
        # Initial call + TOOL_USE_MAX_ITERATIONS follow-ups
        assert client.messages.create.await_count == 1 + llm.TOOL_USE_MAX_ITERATIONS

    @pytest.mark.asyncio
    async def test_voice_addendum_when_tools_present(self):
        client = _make_client(_response([_text_block("ok")]))
        bridge = _FakeBridge()
        await llm.generate_response(**_common_kwargs(client, rick_bridge=bridge))
        system = client.messages.create.call_args.kwargs["system"]
        assert "summarize the result in 1-2 sentences" in system

    @pytest.mark.asyncio
    async def test_no_voice_addendum_when_tools_absent(self):
        client = _make_client(_response([_text_block("ok")]))
        bridge = _FakeBridge(tools=[])  # bridge present but exposes no tools
        await llm.generate_response(**_common_kwargs(client, rick_bridge=bridge))
        system = client.messages.create.call_args.kwargs["system"]
        assert "summarize the result in 1-2 sentences" not in system

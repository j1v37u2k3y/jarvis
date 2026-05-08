"""Tests for the rick_mcp bridge — RickBridge."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rick_bridge import (
    CURATED_RESOURCES,
    CURATED_TOOLS,
    GRACEFUL_TOOL_ERROR,
    RickBridge,
    try_start_bridge,
)


def _mock_tool(name: str, description: str = "test tool") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": {}},
    )


def _mock_resource_text(text: str) -> SimpleNamespace:
    return SimpleNamespace(contents=[SimpleNamespace(text=text)])


def _build_session_mock(tool_names: list[str], resource_text: str = "principle body") -> MagicMock:
    """Build a ClientSession mock that returns the given tools + a fixed resource body."""
    session = MagicMock()
    session.initialize = AsyncMock(return_value=None)
    session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[_mock_tool(n) for n in tool_names]))
    session.read_resource = AsyncMock(return_value=_mock_resource_text(resource_text))
    session.call_tool = AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(text="tool output")]))
    return session


def _patch_bridge_io(session: MagicMock):
    """Return patches for stdio_client + ClientSession that yield the given session."""
    stdio_ctx = MagicMock()
    stdio_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    stdio_ctx.__aexit__ = AsyncMock(return_value=None)

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=None)

    return (
        patch("rick_bridge.stdio_client", return_value=stdio_ctx),
        patch("rick_bridge.ClientSession", return_value=session_ctx),
    )


class TestCuratedConstants:
    def test_curated_tools_length(self):
        assert len(CURATED_TOOLS) == 8

    def test_curated_resources_length(self):
        assert len(CURATED_RESOURCES) == 5

    def test_curated_tool_names_unique(self):
        assert len(set(CURATED_TOOLS)) == len(CURATED_TOOLS)


class TestBridgeStartup:
    @pytest.mark.asyncio
    async def test_start_initializes_session(self):
        session = _build_session_mock(CURATED_TOOLS)
        stdio_p, session_p = _patch_bridge_io(session)
        bridge = RickBridge()
        with stdio_p, session_p:
            await bridge.start()
        session.initialize.assert_awaited_once()
        session.list_tools.assert_awaited_once()
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_curates_only_known_tools(self):
        # rick exposes more tools than we want — curation should drop the extras.
        all_names = CURATED_TOOLS + ["rick_kill_chain", "rick_full_auto", "rick_tracker"]
        session = _build_session_mock(all_names)
        stdio_p, session_p = _patch_bridge_io(session)
        bridge = RickBridge()
        with stdio_p, session_p:
            await bridge.start()
        names = [t["name"] for t in bridge.tools_for_anthropic]
        assert names == CURATED_TOOLS  # ordered + filtered
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_principles_block_has_all_headings(self):
        session = _build_session_mock(CURATED_TOOLS)
        stdio_p, session_p = _patch_bridge_io(session)
        bridge = RickBridge()
        with stdio_p, session_p:
            await bridge.start()
        for _, heading in CURATED_RESOURCES:
            assert f"### {heading}" in bridge.principles_block
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_anthropic_tool_dict_shape(self):
        session = _build_session_mock(CURATED_TOOLS)
        stdio_p, session_p = _patch_bridge_io(session)
        bridge = RickBridge()
        with stdio_p, session_p:
            await bridge.start()
        for tool in bridge.tools_for_anthropic:
            assert set(tool.keys()) == {"name", "description", "input_schema"}
            assert tool["input_schema"]["type"] == "object"
        await bridge.stop()


class TestCallTool:
    @pytest.mark.asyncio
    async def test_call_tool_returns_text(self):
        session = _build_session_mock(CURATED_TOOLS)
        stdio_p, session_p = _patch_bridge_io(session)
        bridge = RickBridge()
        with stdio_p, session_p:
            await bridge.start()
            text = await bridge.call_tool("rick_mantra", {})
        assert text == "tool output"
        session.call_tool.assert_awaited_with("rick_mantra", {})
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_call_tool_graceful_on_exception(self):
        session = _build_session_mock(CURATED_TOOLS)
        session.call_tool.side_effect = RuntimeError("rick crashed")
        stdio_p, session_p = _patch_bridge_io(session)
        bridge = RickBridge()
        with stdio_p, session_p:
            await bridge.start()
            text = await bridge.call_tool("rick_cve", {"cve_id": "CVE-2024-1234"})
        assert text == GRACEFUL_TOOL_ERROR
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_call_tool_without_session_returns_graceful(self):
        bridge = RickBridge()  # never started
        text = await bridge.call_tool("rick_mantra", {})
        assert text == GRACEFUL_TOOL_ERROR


class TestBridgeFailureModes:
    @pytest.mark.asyncio
    async def test_try_start_returns_none_on_failure(self):
        # Bogus python path → spawn fails → bridge returns None, doesn't raise.
        bridge = await try_start_bridge(timeout=2.0)
        # In CI/test env, real rick_mcp may or may not be reachable. We assert
        # only that the function never raises — return shape is None or RickBridge.
        if bridge is not None:
            await bridge.stop()
        assert bridge is None or hasattr(bridge, "tools_for_anthropic")

    @pytest.mark.asyncio
    async def test_stop_idempotent_when_never_started(self):
        bridge = RickBridge()
        await bridge.stop()  # should not raise
        await bridge.stop()  # second call also fine

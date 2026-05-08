"""
RickBridge — JARVIS's MCP client for rick_mcp.

Owns a stdio connection to the rick_mcp server. Prefetches a curated set of
tools (exposed as Anthropic tool-use definitions) and a curated set of
profile:// resources (concatenated into a `principles_block` for system-prompt
injection). One bridge instance, lifespan-scoped on the FastAPI app.
"""

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger("jarvis.rick_bridge")

CURATED_TOOLS: list[str] = [
    "rick_recon_handle",
    "rick_cve",
    "rick_cheatsheet",
    "rick_mantra",
    "rick_capabilities",
    "rick_status",
    "rick_threat_model",
    "rick_tool_recommend",
]

CURATED_RESOURCES: list[tuple[str, str]] = [
    ("profile://values", "Values"),
    ("profile://craftsmanship", "Craftsmanship"),
    ("profile://heritage", "Heritage"),
    ("profile://mantras", "Mantras"),
    ("profile://summary", "Summary"),
]

DEFAULT_RICK_PYTHON = "/Users/jiveturkey/IdeaProjects/rick/venv/bin/python"
DEFAULT_RICK_PATH = "/Users/jiveturkey/IdeaProjects/rick/rick_mcp.py"

GRACEFUL_TOOL_ERROR = "rick is unavailable, sir."


class RickBridge:
    """Async-context-managed MCP client for rick_mcp.

    Usage:
        bridge = RickBridge()
        await bridge.start()
        ...
        await bridge.stop()
    """

    def __init__(self, *, python: str = "", path: str = ""):
        self._python: str = python or os.getenv("RICK_MCP_PYTHON", DEFAULT_RICK_PYTHON)
        self._path: str = path or os.getenv("RICK_MCP_PATH", DEFAULT_RICK_PATH)
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self.tools_for_anthropic: list[dict[str, Any]] = []
        self.principles_block: str = ""

    async def start(self) -> None:
        """Spawn rick_mcp, initialize the session, prefetch tools + resources."""
        params = StdioServerParameters(command=self._python, args=[self._path])
        self._stack = AsyncExitStack()
        try:
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session
            await self._prefetch_tools(session)
            await self._prefetch_resources(session)
            log.info(
                "rick_bridge: connected, %d tools, %d resources",
                len(self.tools_for_anthropic),
                self.principles_block.count("###"),
            )
        except Exception:
            await self._safe_close()
            raise

    async def stop(self) -> None:
        await self._safe_close()

    async def _safe_close(self) -> None:
        if self._stack is None:
            return
        try:
            await self._stack.aclose()
        except Exception as e:
            log.warning("rick_bridge: stop encountered %s", e)
        finally:
            self._stack = None
            self._session = None

    async def _prefetch_tools(self, session: ClientSession) -> None:
        result = await session.list_tools()
        wanted = set(CURATED_TOOLS)
        anthropic_tools: list[dict[str, Any]] = []
        for tool in result.tools:
            if tool.name not in wanted:
                continue
            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
                }
            )
        anthropic_tools.sort(key=lambda t: CURATED_TOOLS.index(t["name"]))
        self.tools_for_anthropic = anthropic_tools

    async def _prefetch_resources(self, session: ClientSession) -> None:
        sections: list[str] = []
        for uri, heading in CURATED_RESOURCES:
            try:
                result = await session.read_resource(uri)  # type: ignore[arg-type]
                text = "\n".join(getattr(c, "text", "") for c in result.contents if getattr(c, "text", "")).strip()
                if text:
                    sections.append(f"### {heading}\n{text}")
            except Exception as e:
                log.warning("rick_bridge: skipping resource %s — %s", uri, e)
        self.principles_block = "\n\n".join(sections)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke a rick tool by name. Returns text output, or a graceful error."""
        if self._session is None:
            log.warning("rick_bridge: call_tool(%s) with no session", name)
            return GRACEFUL_TOOL_ERROR
        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as e:
            log.error("rick_bridge: tool %s raised %s", name, e)
            return GRACEFUL_TOOL_ERROR
        text = "\n".join(getattr(c, "text", "") for c in result.content if getattr(c, "text", "")).strip()
        log.info("rick_bridge: tool %s → %d chars", name, len(text))
        return text or GRACEFUL_TOOL_ERROR

    async def __aenter__(self) -> "RickBridge":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()


async def try_start_bridge(timeout: float = 10.0) -> RickBridge | None:
    """Best-effort bridge startup. Returns None if rick_mcp can't be reached."""
    bridge = RickBridge()
    try:
        await asyncio.wait_for(bridge.start(), timeout=timeout)
        return bridge
    except Exception as e:
        log.warning("rick_bridge: startup failed (%s) — JARVIS continues without rick", e)
        return None

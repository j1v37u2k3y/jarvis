"""
LLM response generation for JARVIS.

Calls Anthropic Haiku with the full system prompt (personality, context,
memories, action tags) and returns the assistant reply.
"""

import logging
from datetime import datetime
from typing import Any

import anthropic

from formatting import format_projects_for_prompt
from memory import build_memory_context
from prompts import JARVIS_SYSTEM_PROMPT
from usage import track_usage

log = logging.getLogger("jarvis.llm")

TOOL_USE_MAX_ITERATIONS = 3
TOOL_USE_MAX_TOKENS = 800
NO_TOOL_MAX_TOKENS = 250

TOOL_VOICE_ADDENDUM = (
    "\n\nWhen you call a rick tool and receive its output, summarize the result in "
    "1-2 sentences for spoken voice. NEVER read URLs, JSON, or long lists aloud. "
    "Stay in JARVIS's British butler voice. Lead with the headline number or fact."
)


async def generate_response(
    text: str,
    client: anthropic.AsyncAnthropic,
    task_mgr,  # ClaudeTaskManager — avoid circular import
    projects: list[dict],
    conversation_history: list[dict],
    ctx_cache: dict,
    dispatch_registry,  # DispatchRegistry — avoid circular import
    user_name: str,
    project_dir: str,
    last_response: str = "",
    session_summary: str = "",
    lookup_status: str = "",
    rick_bridge: Any = None,  # RickBridge — avoid circular import
) -> str:
    """Generate a JARVIS response using Anthropic API.

    All dependencies injected so this module has no server.py coupling.
    """
    now = datetime.now()
    current_time = now.strftime("%A, %B %d, %Y at %I:%M %p")

    weather_info = ctx_cache.get("weather", "Weather data unavailable.")
    screen_ctx = ctx_cache["screen"]
    calendar_ctx = ctx_cache["calendar"]
    mail_ctx = ctx_cache["mail"]

    operator_principles = rick_bridge.principles_block if rick_bridge else ""
    system = JARVIS_SYSTEM_PROMPT.format(
        current_time=current_time,
        weather_info=weather_info,
        screen_context=screen_ctx or "Not checked yet.",
        calendar_context=calendar_ctx,
        mail_context=mail_ctx,
        active_tasks=task_mgr.get_active_tasks_summary(),
        dispatch_context=dispatch_registry.format_for_prompt(),
        known_projects=format_projects_for_prompt(projects),
        user_name=user_name,
        project_dir=project_dir,
        operator_principles=operator_principles,
    )
    if lookup_status:
        system += f"\n\nACTIVE LOOKUPS:\n{lookup_status}\nIf asked about progress, report this status."

    # Inject relevant memories and tasks
    memory_ctx = build_memory_context(text)
    if memory_ctx:
        system += f"\n\nJARVIS MEMORY:\n{memory_ctx}"

    # Three-tier memory — rolling summary of earlier conversation
    if session_summary:
        system += f"\n\nSESSION CONTEXT (earlier in this conversation):\n{session_summary}"

    # Self-awareness — remind JARVIS of last response to avoid repetition
    if last_response:
        system += f'\n\nYOUR LAST RESPONSE (do not repeat this):\n"{last_response[:150]}"'

    messages = conversation_history[-20:]
    if not messages or messages[-1].get("content") != text:
        messages = messages + [{"role": "user", "content": text}]

    tools = rick_bridge.tools_for_anthropic if rick_bridge and rick_bridge.tools_for_anthropic else None
    if tools:
        system = system + TOOL_VOICE_ADDENDUM

    try:
        return await _run_tool_use_loop(
            client=client,
            system=system,
            messages=list(messages),
            tools=tools,
            rick_bridge=rick_bridge,
        )
    except Exception as e:
        log.error(f"LLM error: {e}")
        return "Apologies, sir. I'm having trouble connecting to my language systems."


async def _run_tool_use_loop(
    *,
    client: anthropic.AsyncAnthropic,
    system: str,
    messages: list[dict],
    tools: list[dict] | None,
    rick_bridge: Any,
) -> str:
    """Single API call when tools is None; tool-use loop otherwise (cap 3 iters)."""
    max_tokens = TOOL_USE_MAX_TOKENS if tools else NO_TOOL_MAX_TOKENS
    create_kwargs: dict[str, Any] = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if tools:
        create_kwargs["tools"] = tools

    response = await client.messages.create(**create_kwargs)
    track_usage(response)

    iterations = 0
    while response.stop_reason == "tool_use" and iterations < TOOL_USE_MAX_ITERATIONS:
        iterations += 1
        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in tool_uses:
            args = block.input if isinstance(block.input, dict) else {}
            tool_text = await rick_bridge.call_tool(block.name, args) if rick_bridge else "rick is unavailable, sir."
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": tool_text})
        messages.append({"role": "user", "content": tool_results})

        response = await client.messages.create(**{**create_kwargs, "messages": messages})
        track_usage(response)

    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


async def update_session_summary(
    old_summary: str,
    rotated_messages: list[dict],
    client: anthropic.AsyncAnthropic,
) -> str:
    """Background Haiku call to update the rolling session summary."""
    prompt = f"""Update this conversation summary to include the new messages.

Current summary: {old_summary or "(start of conversation)"}

New messages to incorporate:
{chr(10).join(f"{m['role']}: {m['content'][:200]}" for m in rotated_messages)}

Write an updated summary in 2-4 sentences capturing the key topics, decisions, and context. Be concise."""

    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        log.warning(f"Summary update failed: {e}")
        return old_summary

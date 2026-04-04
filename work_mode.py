"""
JARVIS Work Mode — persistent Claude Code sessions via tmux.

JARVIS can connect to any project directory and maintain a conversation
with Claude Code inside a named tmux session. Context persists naturally
across messages (no --continue flag needed).

Users can attach to any session: `tmux attach -t jarvis-{project}`
JARVIS reads output via `tmux capture-pane` and sends input via `tmux send-keys`.
"""

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path

from sanitize import DANGEROUS_FLAG
from tmux_sessions import TMUX_AVAILABLE, TmuxSession, TmuxSessionManager

log = logging.getLogger("jarvis.work_mode")

SESSION_FILE = Path(__file__).parent / "data" / "active_session.json"

# Shared session manager — imported by server.py
session_manager = TmuxSessionManager()


class WorkSession:
    """A Claude Code session tied to a project directory.

    Uses tmux when available, falls back to subprocess pipes.
    """

    def __init__(self):
        self._active = False
        self._working_dir: str | None = None
        self._project_name: str | None = None
        self._message_count = 0
        self._status = "idle"
        self._tmux: TmuxSession | None = None
        self._tmux_name: str | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def project_name(self) -> str | None:
        return self._project_name

    @property
    def status(self) -> str:
        return self._status

    async def start(self, working_dir: str, project_name: str | None = None):
        """Start or switch to a project session."""
        self._working_dir = working_dir
        self._project_name = project_name or working_dir.split("/")[-1]
        self._active = True
        self._message_count = 0
        self._status = "idle"

        if TMUX_AVAILABLE:
            # Check if there's already a tmux session for this project
            existing = session_manager.find_session(self._project_name)
            if existing and await existing.is_alive():
                self._tmux = existing
                self._tmux_name = existing.name
                log.info(f"Work mode: reattached to existing tmux session {existing.name}")
                return

            # Create new tmux session with interactive claude
            cmd = f"claude{DANGEROUS_FLAG}"
            tmux = await session_manager.create_session(
                self._project_name, working_dir, command=cmd, mode="interactive"
            )
            if tmux:
                self._tmux = tmux
                self._tmux_name = tmux.name
                log.info(f"Work mode started (tmux): {self._project_name} ({working_dir})")
                # Give claude a moment to start up
                await asyncio.sleep(2)
                return

        log.info(f"Work mode started (subprocess fallback): {self._project_name} ({working_dir})")

    async def send(self, user_text: str) -> str:
        """Send a message and get the response.

        With tmux: sends via send_keys, polls capture_output for response.
        Without tmux: falls back to subprocess claude -p.
        """
        self._status = "working"

        if self._tmux and await self._tmux.is_alive():
            return await self._send_tmux(user_text)
        return await self._send_subprocess(user_text)

    async def _send_tmux(self, user_text: str) -> str:
        """Send via tmux session — type the prompt, wait for response."""
        assert self._tmux is not None
        assert self._working_dir is not None

        sentinel = f"JARVIS_DONE_{uuid.uuid4().hex[:8]}"

        # Write prompt to a temp file to avoid shell escaping issues
        prompt_file = Path(self._working_dir) / ".jarvis_prompt.txt"
        prompt_file.write_text(user_text)

        # Capture output before sending so we can diff later
        before = await self._tmux.capture_output()
        before_len = len(before)

        # Send the prompt file content via stdin pipe, with sentinel
        await self._tmux.send_keys(
            f"cat .jarvis_prompt.txt | claude -p --output-format text{DANGEROUS_FLAG}; echo '{sentinel}'"
        )

        # Wait for sentinel
        try:
            output = await self._tmux.wait_for_sentinel(sentinel, timeout=300, poll_interval=2.0)
            # Extract just the new output (after what was there before)
            response = output[before_len:].strip() if len(output) > before_len else output.strip()

            self._message_count += 1
            self._status = "done"
            if self._tmux_name:
                session_manager.update_status(self._tmux_name, "idle")
            log.info(f"Claude Code response for {self._project_name} ({len(response)} chars)")
            return response

        except Exception as e:
            log.error(f"tmux send error: {e}")
            self._status = "error"
            return f"Something went wrong, sir: {str(e)[:100]}"

    async def _send_subprocess(self, user_text: str) -> str:
        """Fallback: subprocess pipe when tmux is unavailable."""
        from sanitize import DANGEROUS_FLAG_LIST

        claude_path = shutil.which("claude")
        if not claude_path:
            return "Claude CLI not found on this system."

        cmd = [
            claude_path,
            "-p",
            "--output-format",
            "text",
            *DANGEROUS_FLAG_LIST,
        ]

        if self._message_count > 0:
            cmd.append("--continue")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=user_text.encode()),
                timeout=300,
            )

            response = stdout.decode().strip()
            self._message_count += 1
            self._status = "done"

            if process.returncode != 0:
                error = stderr.decode().strip()[:200]
                log.error(f"claude -p error: {error}")
                self._status = "error"
                return f"Hit a problem, sir: {error}"

            log.info(f"Claude Code response for {self._project_name} ({len(response)} chars)")
            return response

        except TimeoutError:
            log.error("claude -p timed out after 300s")
            self._status = "timeout"
            return "That's taking longer than expected, sir. The operation timed out."
        except Exception as e:
            log.error(f"Work mode error: {e}")
            self._status = "error"
            return f"Something went wrong, sir: {str(e)[:100]}"

    async def stop(self):
        """End the work session. Kills the tmux session."""
        project = self._project_name
        if self._tmux and self._tmux_name:
            # Don't kill the tmux session — leave it for user to attach later
            session_manager.update_status(self._tmux_name, "idle")
        self._active = False
        self._working_dir = None
        self._project_name = None
        self._message_count = 0
        self._status = "idle"
        self._tmux = None
        self._tmux_name = None
        log.info(f"Work mode ended for {project}")

    def _save_session(self):
        """Persist session state so it survives restarts."""
        try:
            SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
            SESSION_FILE.write_text(
                json.dumps(
                    {
                        "project_name": self._project_name,
                        "working_dir": self._working_dir,
                        "message_count": self._message_count,
                        "tmux_name": self._tmux_name,
                    }
                )
            )
        except Exception as e:
            log.debug(f"Failed to save session: {e}")

    def _clear_session(self):
        """Remove persisted session."""
        try:
            SESSION_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    async def restore(self) -> bool:
        """Restore session from disk after restart. Returns True if restored."""
        try:
            if SESSION_FILE.exists():
                data = json.loads(SESSION_FILE.read_text())
                self._working_dir = data["working_dir"]
                self._project_name = data["project_name"]
                self._message_count = data.get("message_count", 1)
                self._active = True
                self._status = "idle"

                # Try to reconnect to existing tmux session
                tmux_name = data.get("tmux_name")
                if tmux_name and TMUX_AVAILABLE:
                    session = TmuxSession(tmux_name)
                    if await session.is_alive():
                        self._tmux = session
                        self._tmux_name = tmux_name
                        log.info(f"Restored tmux session: {tmux_name}")

                log.info(f"Restored work session: {self._project_name} ({self._working_dir})")
                return True
        except Exception as e:
            log.debug(f"No session to restore: {e}")
        return False


def is_casual_question(text: str) -> bool:
    """Detect if a message is casual chat vs work-related.

    Casual questions go to Haiku (fast). Work goes to claude -p (powerful).
    """
    t = text.lower().strip()

    casual_patterns = [
        "what time",
        "what's the time",
        "what day",
        "what's the weather",
        "weather",
        "how are you",
        "are you there",
        "hey jarvis",
        "good morning",
        "good evening",
        "good night",
        "thank you",
        "thanks",
        "never mind",
        "nevermind",
        "stop",
        "cancel",
        "quit work mode",
        "exit work mode",
        "go back to chat",
        "regular mode",
        "how's it going",
        "what's up",
        "are you still there",
        "you there",
        "jarvis",
        "are you doing it",
        "is it working",
        "what happened",
        "did you hear me",
        "hello",
        "hey",
        "how's that coming",
        "hows that coming",
        "any update",
        "status update",
    ]

    # Short greetings/acknowledgments
    if len(t.split()) <= 3 and any(w in t for w in ["ok", "okay", "sure", "yes", "no", "yeah", "nah", "cool"]):
        return True

    return any(p in t for p in casual_patterns)

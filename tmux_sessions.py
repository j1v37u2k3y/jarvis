"""
JARVIS tmux Session Manager — persistent, monitorable Claude Code sessions.

Every Claude Code session runs inside a named tmux session. JARVIS can:
- Read output via `tmux capture-pane`
- Send input via `tmux send-keys`
- Track session lifecycle and status

Users can attach/detach at will with `tmux attach -t jarvis-{name}`.
"""

import asyncio
import logging
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field

log = logging.getLogger("jarvis.tmux")

TMUX_PATH = shutil.which("tmux") or "/usr/local/bin/tmux"
TMUX_AVAILABLE = shutil.which("tmux") is not None

# Strip ANSI escape codes from captured output
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\(B")


@dataclass
class TmuxSessionInfo:
    """Metadata for a tracked tmux session."""

    name: str
    project_name: str
    working_dir: str
    started_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, running, completed, failed, idle
    dispatch_id: int | None = None
    mode: str = "interactive"  # interactive or pipe


class TmuxSession:
    """Wrapper around a single named tmux session."""

    def __init__(self, name: str):
        self.name = name

    async def create(self, working_dir: str, command: str | None = None) -> bool:
        """Create a new detached tmux session.

        Args:
            working_dir: Directory to start in.
            command: Optional command to run. If None, opens a shell.
        """
        cmd = [TMUX_PATH, "new-session", "-d", "-s", self.name, "-c", working_dir]
        if command:
            cmd.extend([command])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            log.error(f"tmux create failed for {self.name}: {stderr.decode()[:200]}")
            return False
        log.info(f"tmux session created: {self.name} in {working_dir}")
        return True

    async def send_keys(self, text: str, press_enter: bool = True) -> None:
        """Send keystrokes to the tmux session."""
        cmd = [TMUX_PATH, "send-keys", "-t", self.name, text]
        if press_enter:
            cmd.append("Enter")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    async def capture_output(self, lines: int = 500) -> str:
        """Capture the current pane content, stripping ANSI codes."""
        proc = await asyncio.create_subprocess_exec(
            TMUX_PATH,
            "capture-pane",
            "-t",
            self.name,
            "-p",
            "-S",
            f"-{lines}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return ""
        raw = stdout.decode()
        return _ANSI_RE.sub("", raw).strip()

    async def is_alive(self) -> bool:
        """Check if the tmux session still exists."""
        proc = await asyncio.create_subprocess_exec(
            TMUX_PATH,
            "has-session",
            "-t",
            self.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def kill(self) -> None:
        """Kill the tmux session."""
        proc = await asyncio.create_subprocess_exec(
            TMUX_PATH,
            "kill-session",
            "-t",
            self.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        log.info(f"tmux session killed: {self.name}")

    async def wait_for_sentinel(self, sentinel: str, timeout: float = 300, poll_interval: float = 2.0) -> str:
        """Poll capture_output until sentinel appears. Returns all output."""
        start = time.time()
        while time.time() - start < timeout:
            output = await self.capture_output()
            if sentinel in output:
                # Return everything before the sentinel
                idx = output.index(sentinel)
                return output[:idx].strip()
            await asyncio.sleep(poll_interval)
        log.warning(f"tmux sentinel timeout for {self.name} after {timeout}s")
        return await self.capture_output()


class TmuxSessionManager:
    """Manages all JARVIS tmux sessions."""

    def __init__(self):
        self.sessions: dict[str, TmuxSessionInfo] = {}
        self._lock = asyncio.Lock()

    def _make_name(self, project_name: str) -> str:
        """Generate a unique tmux session name."""
        base = f"jarvis-{project_name}"
        # Sanitize: tmux session names can't have dots or colons
        base = re.sub(r"[^a-zA-Z0-9_-]", "-", base)
        if base not in self.sessions:
            return base
        # Collision avoidance
        for i in range(2, 100):
            candidate = f"{base}-{i}"
            if candidate not in self.sessions:
                return candidate
        return f"{base}-{uuid.uuid4().hex[:6]}"

    async def create_session(
        self,
        project_name: str,
        working_dir: str,
        command: str | None = None,
        mode: str = "interactive",
        dispatch_id: int | None = None,
    ) -> TmuxSession | None:
        """Create and track a new tmux session.

        Returns TmuxSession on success, None if tmux unavailable or creation failed.
        """
        if not TMUX_AVAILABLE:
            log.warning("tmux not available — cannot create session")
            return None

        async with self._lock:
            name = self._make_name(project_name)
            session = TmuxSession(name)
            if not await session.create(working_dir, command):
                return None

            self.sessions[name] = TmuxSessionInfo(
                name=name,
                project_name=project_name,
                working_dir=working_dir,
                status="running",
                dispatch_id=dispatch_id,
                mode=mode,
            )
            return session

    def get_session(self, name: str) -> TmuxSession | None:
        """Get a TmuxSession by exact name."""
        if name in self.sessions:
            return TmuxSession(name)
        return None

    def find_session(self, project_name: str) -> TmuxSession | None:
        """Find a session by project name (fuzzy match)."""
        for name, info in self.sessions.items():
            if project_name.lower() in info.project_name.lower():
                return TmuxSession(name)
        return None

    async def list_sessions(self) -> list[dict]:
        """List all tracked sessions, reconciled with actual tmux state."""
        await self.cleanup_dead()
        result = []
        for _name, info in self.sessions.items():
            result.append(
                {
                    "name": info.name,
                    "project_name": info.project_name,
                    "working_dir": info.working_dir,
                    "status": info.status,
                    "mode": info.mode,
                    "uptime_seconds": int(time.time() - info.started_at),
                }
            )
        return result

    async def cleanup_dead(self) -> None:
        """Remove sessions that no longer exist in tmux."""
        dead = []
        for name in list(self.sessions):
            session = TmuxSession(name)
            if not await session.is_alive():
                dead.append(name)
        for name in dead:
            log.info(f"Cleaning up dead tmux session: {name}")
            del self.sessions[name]

    def update_status(self, name: str, status: str) -> None:
        """Update the status of a tracked session."""
        if name in self.sessions:
            self.sessions[name].status = status

    def format_for_voice(self) -> str:
        """Format active sessions for voice response."""
        if not self.sessions:
            return "No active sessions, sir."

        count = len(self.sessions)
        if count == 1:
            info = next(iter(self.sessions.values()))
            elapsed = int(time.time() - info.started_at)
            return f"One session running: {info.project_name}, {info.status} for {elapsed} seconds."

        lines = []
        for info in self.sessions.values():
            elapsed = int(time.time() - info.started_at)
            lines.append(f"{info.project_name} is {info.status} for {elapsed} seconds")
        return f"{count} sessions running. " + ". ".join(lines) + "."

    async def attach_in_terminal(self, name: str) -> bool:
        """Open Terminal.app attached to a tmux session."""
        if name not in self.sessions:
            return False

        from sanitize import escape_applescript

        escaped_cmd = escape_applescript(f"tmux attach -t {name}")
        script = f'tell application "Terminal"\n    activate\n    do script "{escaped_cmd}"\nend tell'
        proc = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

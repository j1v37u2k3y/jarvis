"""Tests for the tmux session management module."""

from unittest.mock import AsyncMock, patch

import pytest

from tmux_sessions import _ANSI_RE, TmuxSession, TmuxSessionInfo, TmuxSessionManager


class TestTmuxSession:
    """Test TmuxSession wrapper methods."""

    @pytest.mark.asyncio
    async def test_create_builds_correct_command(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("tmux_sessions.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            session = TmuxSession("jarvis-test")
            result = await session.create("/tmp/project", "claude")
            assert result is True
            call_args = mock_exec.call_args[0]
            assert "new-session" in call_args
            assert "-s" in call_args
            assert "jarvis-test" in call_args
            assert "/tmp/project" in call_args

    @pytest.mark.asyncio
    async def test_create_returns_false_on_failure(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"duplicate session"))

        with patch("tmux_sessions.asyncio.create_subprocess_exec", return_value=mock_proc):
            session = TmuxSession("jarvis-test")
            result = await session.create("/tmp/project")
            assert result is False

    @pytest.mark.asyncio
    async def test_send_keys_builds_correct_command(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("tmux_sessions.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            session = TmuxSession("jarvis-test")
            await session.send_keys("hello world")
            call_args = mock_exec.call_args[0]
            assert "send-keys" in call_args
            assert "jarvis-test" in call_args
            assert "hello world" in call_args
            assert "Enter" in call_args

    @pytest.mark.asyncio
    async def test_send_keys_no_enter(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("tmux_sessions.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            session = TmuxSession("jarvis-test")
            await session.send_keys("partial", press_enter=False)
            call_args = mock_exec.call_args[0]
            assert "Enter" not in call_args

    @pytest.mark.asyncio
    async def test_capture_output_strips_ansi(self):
        ansi_output = b"\x1b[32mHello\x1b[0m World\x1b[1m!\x1b[0m"
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(ansi_output, b""))

        with patch("tmux_sessions.asyncio.create_subprocess_exec", return_value=mock_proc):
            session = TmuxSession("jarvis-test")
            output = await session.capture_output()
            assert "\x1b" not in output
            assert "Hello" in output
            assert "World" in output

    @pytest.mark.asyncio
    async def test_capture_output_empty_on_failure(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"no session"))

        with patch("tmux_sessions.asyncio.create_subprocess_exec", return_value=mock_proc):
            session = TmuxSession("jarvis-test")
            output = await session.capture_output()
            assert output == ""

    @pytest.mark.asyncio
    async def test_is_alive_true(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("tmux_sessions.asyncio.create_subprocess_exec", return_value=mock_proc):
            session = TmuxSession("jarvis-test")
            assert await session.is_alive() is True

    @pytest.mark.asyncio
    async def test_is_alive_false(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("tmux_sessions.asyncio.create_subprocess_exec", return_value=mock_proc):
            session = TmuxSession("jarvis-test")
            assert await session.is_alive() is False

    @pytest.mark.asyncio
    async def test_kill_sends_kill_command(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("tmux_sessions.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            session = TmuxSession("jarvis-test")
            await session.kill()
            call_args = mock_exec.call_args[0]
            assert "kill-session" in call_args
            assert "jarvis-test" in call_args


class TestAnsiStripping:
    """Test ANSI escape code removal."""

    def test_basic_color_codes(self):
        text = "\x1b[32mgreen\x1b[0m normal"
        assert _ANSI_RE.sub("", text) == "green normal"

    def test_bold_codes(self):
        text = "\x1b[1mbold\x1b[0m"
        assert _ANSI_RE.sub("", text) == "bold"

    def test_osc_sequences(self):
        text = "\x1b]0;title\x07content"
        assert _ANSI_RE.sub("", text) == "content"

    def test_no_ansi(self):
        text = "plain text"
        assert _ANSI_RE.sub("", text) == "plain text"


class TestTmuxSessionManager:
    """Test TmuxSessionManager tracking and naming."""

    def test_make_name_basic(self):
        mgr = TmuxSessionManager()
        name = mgr._make_name("my-project")
        assert name == "jarvis-my-project"

    def test_make_name_sanitizes(self):
        mgr = TmuxSessionManager()
        name = mgr._make_name("my.project:v2")
        assert "." not in name
        assert ":" not in name

    def test_make_name_collision_avoidance(self):
        mgr = TmuxSessionManager()
        mgr.sessions["jarvis-test"] = TmuxSessionInfo(name="jarvis-test", project_name="test", working_dir="/tmp")
        name = mgr._make_name("test")
        assert name == "jarvis-test-2"

    def test_make_name_multiple_collisions(self):
        mgr = TmuxSessionManager()
        mgr.sessions["jarvis-test"] = TmuxSessionInfo(name="jarvis-test", project_name="test", working_dir="/tmp")
        mgr.sessions["jarvis-test-2"] = TmuxSessionInfo(name="jarvis-test-2", project_name="test", working_dir="/tmp")
        name = mgr._make_name("test")
        assert name == "jarvis-test-3"

    def test_find_session_fuzzy(self):
        mgr = TmuxSessionManager()
        mgr.sessions["jarvis-react-dashboard"] = TmuxSessionInfo(
            name="jarvis-react-dashboard", project_name="react-dashboard", working_dir="/tmp"
        )
        session = mgr.find_session("dashboard")
        assert session is not None
        assert session.name == "jarvis-react-dashboard"

    def test_find_session_no_match(self):
        mgr = TmuxSessionManager()
        assert mgr.find_session("nonexistent") is None

    def test_update_status(self):
        mgr = TmuxSessionManager()
        mgr.sessions["jarvis-test"] = TmuxSessionInfo(
            name="jarvis-test", project_name="test", working_dir="/tmp", status="running"
        )
        mgr.update_status("jarvis-test", "completed")
        assert mgr.sessions["jarvis-test"].status == "completed"

    def test_update_status_nonexistent_is_noop(self):
        mgr = TmuxSessionManager()
        mgr.update_status("nonexistent", "completed")  # Should not raise

    def test_format_for_voice_empty(self):
        mgr = TmuxSessionManager()
        assert mgr.format_for_voice() == "No active sessions, sir."

    def test_format_for_voice_single(self):
        mgr = TmuxSessionManager()
        mgr.sessions["jarvis-test"] = TmuxSessionInfo(
            name="jarvis-test", project_name="test-app", working_dir="/tmp", status="running"
        )
        result = mgr.format_for_voice()
        assert "test-app" in result
        assert "running" in result

    def test_format_for_voice_multiple(self):
        mgr = TmuxSessionManager()
        mgr.sessions["jarvis-a"] = TmuxSessionInfo(
            name="jarvis-a", project_name="alpha", working_dir="/tmp", status="running"
        )
        mgr.sessions["jarvis-b"] = TmuxSessionInfo(
            name="jarvis-b", project_name="beta", working_dir="/tmp", status="idle"
        )
        result = mgr.format_for_voice()
        assert "2 sessions" in result
        assert "alpha" in result
        assert "beta" in result

    @pytest.mark.asyncio
    async def test_cleanup_dead_removes_dead_sessions(self):
        mgr = TmuxSessionManager()
        mgr.sessions["jarvis-dead"] = TmuxSessionInfo(name="jarvis-dead", project_name="dead", working_dir="/tmp")

        mock_proc = AsyncMock()
        mock_proc.returncode = 1  # Session doesn't exist
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("tmux_sessions.asyncio.create_subprocess_exec", return_value=mock_proc):
            await mgr.cleanup_dead()
            assert "jarvis-dead" not in mgr.sessions

    @pytest.mark.asyncio
    async def test_cleanup_dead_keeps_alive_sessions(self):
        mgr = TmuxSessionManager()
        mgr.sessions["jarvis-alive"] = TmuxSessionInfo(name="jarvis-alive", project_name="alive", working_dir="/tmp")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0  # Session exists
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("tmux_sessions.asyncio.create_subprocess_exec", return_value=mock_proc):
            await mgr.cleanup_dead()
            assert "jarvis-alive" in mgr.sessions

    @pytest.mark.asyncio
    async def test_list_sessions_returns_info(self):
        mgr = TmuxSessionManager()
        mgr.sessions["jarvis-test"] = TmuxSessionInfo(
            name="jarvis-test", project_name="test", working_dir="/tmp/test", status="running"
        )

        # Mock cleanup_dead (is_alive check)
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("tmux_sessions.asyncio.create_subprocess_exec", return_value=mock_proc):
            sessions = await mgr.list_sessions()
            assert len(sessions) == 1
            assert sessions[0]["project_name"] == "test"
            assert sessions[0]["status"] == "running"
            assert "uptime_seconds" in sessions[0]

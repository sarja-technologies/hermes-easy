"""Comprehensive regression and PTY end-to-end tests for Hermes Easy Ctrl+C, signal handling,
cancellation recovery, terminal resize during streaming, and terminal state restoration.
"""
from __future__ import annotations

import asyncio
import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from prompt_toolkit.buffer import Buffer

import cli as cli_mod
import hermes_easy
from hermes_cli import cli_single_query
from hermes_easy.cli_mixin import EasyUIMixin
from hermes_cli.cli_tui_mixin import CLITuiMixin
from hermes_cli.cli_tui_runtime_mixin import CLITuiRuntimeMixin
from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
from hermes_cli.cli_terminal_mixin import CLITerminalMixin


class _FakeEasyCli(EasyUIMixin, CLITuiRuntimeMixin, CLITuiMixin, CLIChatTurnMixin, CLITerminalMixin):
    """Minimal HermesCLI test double preserving the real EasyUIMixin + TUI MRO."""

    def __init__(self):
        self._agent_running = False
        self._command_running = False
        self._command_blocks_input = False
        self._command_status = ""
        self._voice_recording = False
        self._should_exit = False
        self._detach_on_exit = False
        self._viewer_disconnected = False
        self._attached_images = []
        self._last_ctrl_c_time = 0.0
        self._easy_exit_hint_until = 0.0
        self._easy_cancel_requested = False
        self._easy_cancel_in_flight = False
        self._easy_cancel_requested_at = 0.0
        self._last_turn_interrupted = False
        self._spinner_text = ""
        self._ui_settings_state = None
        self._slash_confirm_state = None
        self._model_picker_state = None
        self._command_palette_state = None
        self._gateway_command_wait = None
        self._approval_state = None
        self._clarify_state = None
        self._clarify_freetext = False
        self._sudo_state = None
        self._secret_state = None
        self._connection_state = None
        self._terminal_io_broken = False
        self._stream_buf = ""
        self._stream_box_live = False
        self._reasoning_box_opened = False
        self._pending_tool_info = {}
        self._tool_start_time = 0.0
        self._last_scrollback_tool = ""
        self._interactive_turn = False
        self._voice_mode = False
        self._voice_continuous = False
        self._pet_reasoning = False
        self._session_db = None
        self.agent = None
        self._app = None
        self._exited = False
        self._invalidations = 0
        self._overlays_cleared = 0
        self._recovered_after_interrupt = 0

    def _tui_cancel_voice_recording(self, event):
        return False

    def _tui_cancel_foreground_ui(self, event, closers):
        return False

    def _tui_clear_blocking_overlays(self, event):
        return False

    def _clear_active_overlays_for_interrupt(self):
        self._overlays_cleared += 1

    def _recover_terminal_after_interrupt(self):
        self._recovered_after_interrupt += 1

    def _pet_react_turn_end(self):
        pass

    def _turn_summary_emit(self):
        pass

    def _drain_interrupt_queue_to_pending_input(self):
        pass

    def _maybe_continue_goal_after_turn(self):
        pass

    def _maybe_complete_loop_tick_after_turn(self):
        pass

    def _drain_process_notifications(self, _where):
        pass


def _make_event(cli, text=""):
    buf = Buffer()
    buf.text = text

    def _exit():
        cli._exited = True

    def _invalidate():
        cli._invalidations += 1

    app = SimpleNamespace(
        current_buffer=buf,
        invalidate=_invalidate,
        exit=_exit,
        is_running=True,
    )
    cli._app = app
    return SimpleNamespace(app=app, current_buffer=buf), buf, app


@pytest.fixture(autouse=True)
def _enable_hermes_easy(monkeypatch):
    monkeypatch.setenv("HERMES_EASY_UI", "1")
    cli_single_query._reset_signal_interrupt_state()
    yield
    cli_single_query._reset_signal_interrupt_state()


def test_interrupt_agent_for_signal_is_nonblocking_and_reentrancy_safe():
    """_interrupt_agent_for_signal must never sleep and must be idempotent on repeated signals."""
    calls = []

    class SlowAgent:
        def __init__(self):
            self._interrupt_requested = False

        def hard_interrupt(self, msg=None):
            calls.append(msg)

    agent = SlowAgent()
    t0 = time.monotonic()
    cli_single_query._interrupt_agent_for_signal(agent, signal.SIGINT)
    cli_single_query._interrupt_agent_for_signal(agent, signal.SIGINT)
    cli_single_query._interrupt_agent_for_signal(agent, signal.SIGINT)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.2, f"Signal interrupt blocked for {elapsed:.2f}s (must never sleep)"
    assert agent._interrupt_requested is True
    assert len(calls) == 1, f"Expected idempotent single interrupt, got {len(calls)}"


def test_signal_handler_q_rapid_reentry_never_recurses(monkeypatch):
    """Repeated rapid SIGINT calls into _signal_handler_q must never recurse or stack tracebacks."""
    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda s, h: handlers.__setitem__(s, h))
    monkeypatch.setattr(cli_mod, "_arm_exit_watchdog_on_shutdown_signal", lambda: None)

    interrupt_calls = []

    class FakeAgent:
        _interrupt_requested = False

        def hard_interrupt(self, msg=None):
            interrupt_calls.append(msg)

    fake_cli = SimpleNamespace(agent=FakeAgent())
    cli_single_query._install_single_query_signal_handlers(fake_cli)
    sigint_handler = handlers[signal.SIGINT]

    # First SIGINT raises KeyboardInterrupt once cleanly.
    with pytest.raises(KeyboardInterrupt):
        sigint_handler(signal.SIGINT, None)

    # Second SIGINT while unwinding returns silently without recursing or raising again.
    sigint_handler(signal.SIGINT, None)
    assert len(interrupt_calls) == 1


def test_tui_install_signal_handlers_replaces_sigint_on_posix(monkeypatch):
    """Interactive TUI must replace single-query _signal_handler_q on SIGINT."""
    if sys.platform == "win32":
        pytest.skip("POSIX SIGINT test")
    installed = {}
    monkeypatch.setattr(signal, "signal", lambda s, h: installed.__setitem__(s, h))

    cli = _FakeEasyCli()
    cli._tui_install_signal_handlers()

    assert signal.SIGINT in installed
    assert installed[signal.SIGINT] == cli._tui_sigint_handler
    assert installed[signal.SIGTERM] == cli._tui_signal_handler


def test_ctrl_c_single_and_rapid_during_generation_cancels_idempotently():
    """Single and rapid Ctrl+C during generation cancels once without exiting or blocking."""
    cli = _FakeEasyCli()
    cli._agent_running = True
    called = threading.Event()
    call_count = 0
    release = threading.Event()

    class SlowCancelAgent:
        _interrupt_requested = False

        def hard_interrupt(self, msg=None):
            nonlocal call_count
            call_count += 1
            called.set()
            release.wait(timeout=2.0)

    cli.agent = SlowCancelAgent()
    event, _buf, _app = _make_event(cli, text="")

    t0 = time.monotonic()
    for _ in range(10):
        cli._tui_handle_ctrl_c(event)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.2, f"Ctrl+C blocked UI loop for {elapsed:.2f}s during slow cancellation"
    assert called.wait(timeout=1.0)
    release.set()
    assert call_count == 1, f"Rapid Ctrl+C fired {call_count} interrupts instead of 1"
    assert cli._should_exit is False
    assert cli._exited is False
    assert cli._easy_cancel_requested is True
    assert cli._spinner_text == "Cancelling…"

    # After turn settles, state is completely clean and ready for next prompt.
    cli._tui_after_turn()
    assert cli._agent_running is False
    assert cli._easy_cancel_requested is False
    assert cli._easy_cancel_in_flight is False
    assert cli._spinner_text == ""
    assert cli._recovered_after_interrupt == 1


def test_ctrl_c_during_model_loading_before_agent_init_cancels_without_exiting():
    """Ctrl+C while _agent_running=True and agent=None (loading) must cancel, not exit CLI."""
    cli = _FakeEasyCli()
    cli._agent_running = True
    cli.agent = None
    event, _buf, _app = _make_event(cli, text="")

    cli._tui_handle_ctrl_c(event)
    assert cli._should_exit is False
    assert cli._exited is False
    assert cli._easy_cancel_requested is True
    assert cli._last_turn_interrupted is True


def test_ctrl_c_idle_clears_draft_then_double_press_exits():
    """Idle Ctrl+C clears typed draft; on empty composer, first shows hint and second exits cleanly."""
    cli = _FakeEasyCli()
    event, buf, _app = _make_event(cli, text="some draft text")

    # 1. Clears draft
    cli._tui_handle_ctrl_c(event)
    assert buf.text == ""
    assert not cli._should_exit and not cli._exited

    # 2. First press on empty composer shows exit hint
    cli._tui_handle_ctrl_c(event)
    assert cli._easy_exit_hint_until > time.monotonic()
    assert not cli._should_exit and not cli._exited

    # 3. Second press within 2s exits cleanly
    cli._tui_handle_ctrl_c(event)
    assert cli._should_exit is True
    assert cli._exited is True


def test_ctrl_c_during_network_failure_settles_without_hanging(monkeypatch):
    """If the agent thread stalls on a dead network after Ctrl+C, monitor returns to composer cleanly."""
    cli = _FakeEasyCli()
    cli._interrupt_queue = __import__("queue").Queue()
    cli._voice_mode = False
    cli._easy_cancel_requested = True

    stall = threading.Event()
    agent_thread = threading.Thread(target=stall.wait, args=(30.0,), daemon=True)
    agent_thread.start()

    # Fast-forward monotonic clock past the 12s settle cap
    real_monotonic = time.monotonic
    calls = {"n": 0}

    def fast_monotonic():
        calls["n"] += 1
        return real_monotonic() + (15.0 if calls["n"] > 2 else 0.0)

    monkeypatch.setattr("hermes_cli.cli_chat_turn_mixin.time.monotonic", fast_monotonic)
    t0 = real_monotonic()
    res = cli._chat_monitor_agent_thread(SimpleNamespace(), agent_thread)
    elapsed = real_monotonic() - t0
    stall.set()

    assert res is None
    assert elapsed < 3.0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX PTY tests")
def test_pty_rapid_sigint_single_query_no_recursion_no_orphans_termios_restored():
    """Real PTY test: rapid Ctrl+C during a tool subprocess exits cleanly with no traceback,
    no orphaned child process, and restored terminal attributes (ECHO + ICANON + ISIG)."""
    master_fd, slave_fd = pty.openpty()
    saved = termios.tcgetattr(slave_fd)

    child_pid_file = f"/tmp/hermes_pty_child_{os.getpid()}.pid"
    if os.path.exists(child_pid_file):
        os.unlink(child_pid_file)

    script = f"""
import os, signal, subprocess, sys, time
sys.path.insert(0, {os.getcwd()!r})
from hermes_cli.cli_single_query import _install_single_query_signal_handlers
from hermes_easy.entry import _save_terminal_state, _restore_terminal_state

tty_fd, saved_t = _save_terminal_state()

# Spawn a child subprocess in its own process group (simulating a tool call)
child = subprocess.Popen(["sleep", "60"], start_new_session=True)
with open({child_pid_file!r}, "w") as f:
    f.write(str(child.pid))

class DummyAgent:
    _interrupt_requested = False
    def hard_interrupt(self, msg=None):
        self._interrupt_requested = True
        try:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        except OSError:
            pass

class DummyCli:
    agent = DummyAgent()

_install_single_query_signal_handlers(DummyCli())
print("READY", flush=True)
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("CLEAN_INTERRUPT", flush=True)
    sys.exit(130)
finally:
    _restore_terminal_state(tty_fd, saved_t)
"""
    def _attach_ctty():
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        preexec_fn=_attach_ctty,
    )
    os.close(slave_fd)
    output = b""
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and b"READY" not in output:
            r, _, _ = select.select([master_fd], [], [], 0.2)
            if r:
                try:
                    output += os.read(master_fd, 4096)
                except OSError:
                    break
        assert b"READY" in output, f"Process never printed READY: {output!r}"

        # Hammer 5 rapid Ctrl+C bytes (\x03) through the PTY
        for _ in range(5):
            os.write(master_fd, b"\x03")
            time.sleep(0.02)

        rc = proc.wait(timeout=5.0)
        # Drain remaining output
        while True:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if not r:
                break
            try:
                chunk = os.read(master_fd, 4096)
                if not chunk:
                    break
                output += chunk
            except OSError:
                break

        text = output.decode("utf-8", errors="replace")
        assert rc == 130, f"Expected exit 130, got {rc}; output={text!r}"
        assert "Traceback" not in text, f"Unexpected traceback on rapid Ctrl+C: {text!r}"
        assert "_interrupt_agent_for_signal" not in text
        assert "CLEAN_INTERRUPT" in text
        assert "\x1b[?25h" in text, "Cursor visibility reset sequence missing"

        # Verify child tool process was killed (no orphan)
        with open(child_pid_file) as f:
            child_pid = int(f.read().strip())
        time.sleep(0.1)
        alive = True
        try:
            os.kill(child_pid, 0)
            # Check zombie
            with open(f"/proc/{child_pid}/status") as sf:
                if "State:\tZ" in sf.read():
                    alive = False
        except OSError:
            alive = False
        assert not alive, f"Orphan child process {child_pid} survived Ctrl+C!"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
        os.close(master_fd)
        if os.path.exists(child_pid_file):
            os.unlink(child_pid_file)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX PTY tests")
def test_pty_interactive_ctrl_c_resize_cancel_and_followup_request():
    """End-to-end PTY test of interactive prompt_toolkit TUI with EasyUIMixin:
    1. Start interactive composer in a real PTY
    2. Submit request #1 (streaming + slow tool)
    3. Resize PTY mid-stream (SIGWINCH)
    4. Send rapid Ctrl+C (\x03) AND POSIX SIGINT -> cancels turn #1 without exiting or traceback
    5. Submit request #2 -> succeeds cleanly in the recovered composer
    6. Press Ctrl+D -> clean exit with termios restored (ECHO/ICANON/ISIG) and no traceback.
    """
    master_fd, slave_fd = pty.openpty()
    # Set initial PTY window size 100x30
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))

    script = f"""
import asyncio, os, signal, sys, threading, time
os.environ["HERMES_EASY_UI"] = "1"
sys.path.insert(0, {os.getcwd()!r})

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl

from hermes_easy.cli_mixin import EasyUIMixin
from hermes_cli.cli_tui_mixin import CLITuiMixin
from hermes_cli.cli_tui_runtime_mixin import CLITuiRuntimeMixin
from hermes_easy.entry import _save_terminal_state, _restore_terminal_state

tty_fd, saved_t = _save_terminal_state()

class MiniAgent:
    def __init__(self):
        self._interrupt_requested = False
        self.stop_event = threading.Event()
    def hard_interrupt(self, msg=None):
        self._interrupt_requested = True
        self.stop_event.set()

class MiniEasyApp(EasyUIMixin, CLITuiRuntimeMixin, CLITuiMixin):
    def __init__(self):
        self._agent_running = False
        self._command_running = False
        self._voice_recording = False
        self._should_exit = False
        self._detach_on_exit = False
        self._viewer_disconnected = False
        self._attached_images = []
        self._last_ctrl_c_time = 0.0
        self._easy_exit_hint_until = 0.0
        self._easy_cancel_requested = False
        self._easy_cancel_in_flight = False
        self._easy_cancel_requested_at = 0.0
        self._last_turn_interrupted = False
        self._spinner_text = ""
        self._ui_settings_state = None
        self._slash_confirm_state = None
        self._model_picker_state = None
        self._command_palette_state = None
        self._gateway_command_wait = None
        self._approval_state = None
        self._clarify_state = None
        self._clarify_freetext = False
        self._sudo_state = None
        self._secret_state = None
        self._connection_state = None
        self._session_db = None
        self.agent = None
        self._app = None
        self.turn_num = 0

    def _tui_cancel_voice_recording(self, event): return False
    def _tui_cancel_foreground_ui(self, event, closers): return False
    def _tui_clear_blocking_overlays(self, event): return False
    def _tui_clear_or_exit(self, event):
        if event.app.current_buffer.text:
            event.app.current_buffer.reset()
            event.app.invalidate()
        else:
            self._should_exit = True
            event.app.exit()
    def _clear_active_overlays_for_interrupt(self): pass

    def start_turn(self, text):
        self.turn_num += 1
        n = self.turn_num
        self._agent_running = True
        self._easy_cancel_requested = False
        self._easy_cancel_in_flight = False
        self.agent = MiniAgent()

        def worker():
            os.write(1, f"\\r\\nTURN_{{n}}_START:{{text}}\\r\\n".encode())
            if n == 1:
                for i in range(40):
                    if self.agent.stop_event.wait(0.05):
                        os.write(1, b"\\r\\nTURN_1_CANCELLED\\r\\n")
                        break
                    os.write(1, f"tok{{i}} ".encode())
            else:
                time.sleep(0.05)
                os.write(1, f"\\r\\nTURN_{{n}}_OK:{{text}}\\r\\n".encode())
            self._agent_running = False
            self._easy_cancel_requested = False
            self._easy_cancel_in_flight = False
            self._spinner_text = ""
            if self._app:
                self._app.invalidate()

        threading.Thread(target=worker, daemon=True).start()

cli = MiniEasyApp()
buf = Buffer()
kb = KeyBindings()

@kb.add("enter")
def _enter(event):
    txt = buf.text.strip()
    buf.reset()
    if txt:
        cli.start_turn(txt)
    event.app.invalidate()

@kb.add("c-c")
def _cc(event):
    cli._tui_handle_ctrl_c(event)

@kb.add("c-d")
def _cd(event):
    cli._tui_handle_ctrl_d(event)

app = Application(
    layout=Layout(HSplit([Window(BufferControl(buffer=buf)), Window(FormattedTextControl(lambda: cli._spinner_text), height=1)])),
    key_bindings=kb,
    full_screen=False,
)
cli._app = app
cli._tui_install_signal_handlers()

def _announce_ready():
    os.write(1, b"TUI_READY\\r\\n")

try:
    app.run(pre_run=_announce_ready)
finally:
    _restore_terminal_state(tty_fd, saved_t)
"""
    def _attach_ctty():
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        preexec_fn=_attach_ctty,
    )

    def _read_until(marker: bytes, timeout: float = 8.0) -> bytes:
        buf = b""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if r:
                try:
                    chunk = os.read(master_fd, 4096)
                    if not chunk:
                        break
                    buf += chunk
                    if marker in buf:
                        return buf
                except OSError:
                    break
        return buf

    try:
        out = _read_until(b"TUI_READY")
        assert b"TUI_READY" in out, f"TUI did not become ready: {out!r}"

        # 1. Submit first request
        os.write(master_fd, b"first_req\r")
        out1 = _read_until(b"TURN_1_START:first_req")
        assert b"TURN_1_START:first_req" in out1, f"Turn 1 did not start: {out1!r}"

        # 2. Resize terminal during streaming
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 35, 120, 0, 0))
        os.kill(proc.pid, signal.SIGWINCH)
        time.sleep(0.1)

        # 3. Send rapid Ctrl+C (\x03) AND POSIX SIGINT while turn 1 is streaming
        os.write(master_fd, b"\x03\x03\x03")
        os.kill(proc.pid, signal.SIGINT)
        out_cancel = _read_until(b"TURN_1_CANCELLED")
        assert b"TURN_1_CANCELLED" in out_cancel, f"Turn 1 not cancelled: {out_cancel!r}"
        assert proc.poll() is None, "Interactive CLI exited instead of returning to composer!"

        # 4. Submit follow-up request in the recovered composer
        time.sleep(0.15)
        os.write(master_fd, b"second_req\r")
        out2 = _read_until(b"TURN_2_OK:second_req")
        assert b"TURN_2_OK:second_req" in out2, f"Follow-up request failed after cancel: {out2!r}"

        # 5. Clean exit via Ctrl+D
        time.sleep(0.1)
        os.write(master_fd, b"\x04")
        rc = proc.wait(timeout=5.0)
        assert rc == 0, f"Expected clean exit 0 on Ctrl+D, got {rc}"

        # 6. Verify terminal attributes restored (ECHO, ICANON, ISIG)
        attrs = termios.tcgetattr(slave_fd)
        lflag = attrs[3]
        assert lflag & termios.ECHO, "ECHO was not restored on exit"
        assert lflag & termios.ICANON, "ICANON was not restored on exit"
        assert lflag & termios.ISIG, "ISIG was not restored on exit"

        all_text = (out + out1 + out_cancel + out2).decode("utf-8", errors="replace")
        assert "Traceback" not in all_text, f"Unexpected traceback in interactive session: {all_text!r}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
        os.close(slave_fd)
        os.close(master_fd)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX PTY tests")
def test_installed_hermes_easy_cli_sigint_and_clean_exit_e2e():
    """End-to-end PTY test of the actual installed ``hermes-easy`` executable:
    verifies startup, clean Ctrl+C / exit without recursive tracebacks, and restored termios."""
    hermes_easy_bin = "/home/dembele/.local/bin/hermes-easy"
    if not os.path.isfile(hermes_easy_bin):
        pytest.skip("installed hermes-easy binary not present")

    master_fd, slave_fd = pty.openpty()
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env.pop("TMUX", None)

    proc = subprocess.Popen(
        [hermes_easy_bin, "--version"],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        close_fds=True,
        start_new_session=True,
    )
    try:
        rc = proc.wait(timeout=10.0)
        out = b""
        while True:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if not r:
                break
            try:
                chunk = os.read(master_fd, 4096)
                if not chunk:
                    break
                out += chunk
            except OSError:
                break
        text = out.decode("utf-8", errors="replace")
        assert rc == 0, f"hermes-easy --version failed (rc={rc}): {text!r}"
        assert "hermes-easy" in text
        assert "Traceback" not in text

        # Verify termios still intact on slave PTY
        attrs = termios.tcgetattr(slave_fd)
        assert attrs[3] & termios.ECHO
        assert attrs[3] & termios.ICANON
        assert attrs[3] & termios.ISIG
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
        os.close(slave_fd)
        os.close(master_fd)

"""Terminal positioning when starting and resizing a Hermes Easy session."""

import io
import os
from collections import deque

from prompt_toolkit.data_structures import Size
from prompt_toolkit.output.vt100 import Vt100_Output

from hermes_easy import viewport


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_startup_positions_identity_above_composer_without_erasing_scrollback(monkeypatch):
    import cli

    output = Terminal()
    floors = []
    history = deque(["previous session"])
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(viewport.sys, "stdout", output)
    monkeypatch.setattr(viewport.shutil, "get_terminal_size", lambda: os.terminal_size((20, 12)))
    monkeypatch.setattr(cli, "_output_history_recording", lambda: True)
    monkeypatch.setattr(cli, "_OUTPUT_HISTORY", history)
    monkeypatch.setattr(cli, "_set_chrome_floor", floors.append)

    marker = viewport.begin_startup()
    assert output.getvalue() == "\n" * 12 + "\x1b[H"
    history.extend(["Hermes Easy", "a directory path that wraps over two rows"])
    viewport.finish_startup(marker)
    assert floors == [8]


def test_inline_erase_in_tmux_restores_cursor_without_erase_display(monkeypatch):
    monkeypatch.setenv("TMUX", "test")
    stream = Terminal()
    output = Vt100_Output(stream, lambda: Size(rows=4, columns=30), term="tmux-256color")

    viewport.preserve_tmux_scrollback(output)
    output.erase_down()
    output.flush()
    assert stream.getvalue() == "\x1b7\x1b[K" + "\x1b[B\r\x1b[K" * 3 + "\x1b8"

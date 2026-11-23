"""Initial transcript positioning without a terminal cursor query."""

import os
import shutil
import sys


def begin_startup():
    """Create a known starting row while keeping the shell's native scrollback."""
    from cli import _OUTPUT_HISTORY, _output_history_recording

    if not sys.stdout.isatty() or os.getenv("TERM") in (None, "dumb") or not _output_history_recording():
        return None
    size = shutil.get_terminal_size()
    if size.lines < 8:
        return None
    # Move the previous viewport into scrollback and position the startup
    # transcript on the first row of the newly available screen.
    print("\n" * size.lines + "\x1b[H", end="", flush=True)
    return size, tuple(_OUTPUT_HISTORY)


def finish_startup(viewport):
    """Reserve only the rows below startup output for the live composer."""
    from cli import _OUTPUT_HISTORY, _line_rows, _set_chrome_floor

    initial, previous = viewport
    current = shutil.get_terminal_size()
    if initial != current:
        return
    history = tuple(_OUTPUT_HISTORY)
    new = history[len(previous):] if history[:len(previous)] == previous else history
    painted_rows = sum(_line_rows(line, current.columns) for line in new)
    _set_chrome_floor(max(0, current.lines - painted_rows))


def preserve_tmux_scrollback(output):
    """Erase inline prompt rows without tmux copying the live UI into history."""
    from prompt_toolkit.output.vt100 import Vt100_Output

    if not os.getenv("TMUX") or not isinstance(output, Vt100_Output):
        return

    def erase_rows():
        # A relative cursor-down stops at the bottom. Saving the cursor also
        # puts it back at the correct row when the viewport just resized.
        rows = output.get_size().rows
        output.write_raw("\x1b7\x1b[K" + "\x1b[B\r\x1b[K" * (rows - 1) + "\x1b8")

    output.erase_down = erase_rows

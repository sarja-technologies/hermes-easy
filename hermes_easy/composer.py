"""The live terminal region for Hermes Easy."""

from __future__ import annotations

from prompt_toolkit.completion import Completer
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import ConditionalContainer, FormattedTextControl, HSplit, VSplit, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.utils import get_cwidth

from hermes_easy import layout
from hermes_easy.scrollback import centre, strip_controls

_CHOICE_STATES = (
    "_approval_state", "_slash_confirm_state", "_model_picker_state",
    "_command_palette_state", "_ui_settings_state",
)
_ENTRY_STATES = ("_sudo_state", "_secret_state", "_connection_state")


def content_width(columns: int | None) -> int:
    return max(1, (columns or 80) - 2)


def erase_shrinking_rows(output, screen, previous, cursor, has_style) -> bool:
    """Clear rows that get shorter so tmux forgets their old right edge.

    A partial erase leaves the old cell count in tmux's grid. When a pane
    narrows, tmux reflows those invisible cells into additional lines.
    """
    def width(row):
        return max((x + 1 for x, cell in row.items() if cell.char != " " or has_style[cell.style]), default=0)

    rows = [y for y in range(min(screen.height, previous.height))
            if width(screen.data_buffer[y]) < width(previous.data_buffer[y])]
    if not rows:
        return False

    output.reset_attributes()
    output.cursor_backward(cursor.x)
    y = cursor.y
    for row in rows:
        if row < y:
            output.cursor_up(y - row)
        elif row > y:
            output.cursor_down(row - y)
        output.erase_end_of_line()
        previous.data_buffer[row].clear()
        y = row
    if y > cursor.y:
        output.cursor_up(y - cursor.y)
    elif y < cursor.y:
        output.cursor_down(cursor.y - y)
    output.cursor_forward(cursor.x)
    return True


def choice_active(cli) -> bool:
    return any(getattr(cli, name, None) for name in _CHOICE_STATES) or bool(
        cli._clarify_state and not cli._clarify_freetext)


def modal_active(cli) -> bool:
    return choice_active(cli) or bool(cli._clarify_state) or any(
        getattr(cli, name, None) for name in _ENTRY_STATES)


def keep_when(widget, condition):
    return ConditionalContainer(widget, filter=Condition(condition))


def row(fragments, *, wrap=False):
    return Window(FormattedTextControl(fragments), wrap_lines=wrap, dont_extend_height=True)


def build(cli, parts, extras):
    """Place transient transcript rows, overlays and the composer in a shared column."""
    def normal():
        return not choice_active(cli)

    def rule():
        return Window(char="─", style="class:input-rule", height=1)

    editor = HSplit([
        keep_when(rule(), lambda: not cli._status_bar_suppressed_after_resize),
        parts["image_bar"],
        parts["input_area"],
        keep_when(rule(), lambda: not cli._status_bar_suppressed_after_resize),
        parts["voice_status_bar"],
        parts["status_bar"],
    ])
    children = [
        Window(height=Dimension(min=0, preferred=0)),
        keep_when(row(lambda: partial_response(cli)), lambda: bool(partial_text(cli))),
        keep_when(row(lambda: layout.activity_fragments(cli)),
                  lambda: normal() and bool(layout.activity_fragments(cli))),
        keep_when(row(lambda: layout.live_tool_fragments(cli)),
                  lambda: normal() and bool(layout.live_tool_fragments(cli))),
        parts["approval_widget"], parts["slash_confirm_widget"], parts["clarify_widget"],
        parts["model_picker_widget"], parts["command_palette_widget"],
        parts["connection_widget"], parts["sudo_widget"], parts["secret_widget"],
        *extras, cli._pet_widget, cli._stash_panel_widget, cli._subagent_dock_widget,
        keep_when(parts["spacer"], lambda: any(getattr(cli, name, None) for name in _ENTRY_STATES)),
        keep_when(row(lambda: completion_rows(cli, parts["input_area"].buffer)),
                  lambda: not modal_active(cli) and bool(parts["input_area"].buffer.complete_state)),
        keep_when(editor, normal),
    ]
    return [VSplit([
        Window(width=Dimension(weight=1)),
        _column([widget for widget in children if widget is not None],
                width=lambda: Dimension.exact(content_width(cli._get_tui_terminal_width()))),
        Window(width=Dimension(weight=1)),
    ])]


def _column(widgets, *, width):
    # Worker events can add/remove rows between height measurement and the paint.
    from hermes_cli.cli_footer_split import FooterSplit
    return FooterSplit(widgets, width=width)


def partial_text(cli):
    # Snapshot once: the turn thread can commit a line before the next frame.
    text = getattr(cli, "_stream_buf", "")
    if not text or not getattr(cli, "_stream_box_live", False) or getattr(cli, "_in_stream_table", False):
        return ""
    if text.lstrip().startswith(("|", "```", "~~~")):
        return ""
    return text


def partial_response(cli):
    text = partial_text(cli)
    if not text:
        return []
    width = content_width(cli._get_tui_terminal_width())
    limit = max(1, min(5, terminal_rows(cli) // 4))
    text_width = max(1, width - 2)
    # Only the latest viewport belongs in the live region. The complete text is
    # retained by the stream buffer and committed to terminal scrollback.
    preview_limit = text_width * limit * 4
    clipped = len(text) > preview_limit
    text = safe_display(text[-preview_limit:])
    prose = not getattr(cli, "_easy_in_fence", False)
    if cli.final_response_markdown == "strip" and prose:
        from cli import _strip_markdown_syntax
        text = _strip_markdown_syntax(text)
    pending = getattr(cli, "_easy_answer_marker", False)
    lines = centre(text, text_width, text_width, prose=prose).splitlines()
    clipped = clipped or len(lines) > limit
    prefix = "⋯ " if clipped else "● " if pending else "  "
    first, *rest = lines[-limit:] or [""]
    display = "\n".join([prefix + first, *("  " + line for line in rest)])
    return [("", display)]


def terminal_rows(cli):
    app = getattr(cli, "_app", None)
    return app.output.get_size().rows if app else 24


def safe_display(text: str) -> str:
    """Keep terminal controls out of transient provider/tool output."""
    return strip_controls(text)


def completion_rows(cli, buffer):
    state = buffer.complete_state
    if state is None or not state.completions:
        return []
    width = content_width(cli._get_tui_terminal_width())
    count = min(7, max(2, terminal_rows(cli) // 4))
    idx = state.complete_index or 0
    start = max(0, idx - count + 1)
    visible = state.completions[start:start + count]
    label_width = min(max(get_cwidth(c.display_text) for c in visible), max(6, width // 3))
    rows = []
    for offset, item in enumerate(visible, start):
        active = offset == idx
        label = cli._trim_status_bar_text(safe_display(item.display_text), label_width)
        head = ("  › " if active else "    ") + label
        rows.append(("class:easy-selected" if active else "", head))
        room = width - 6 - label_width
        meta = item.display_meta_text.partition(" (usage:")[0]
        if meta and room >= 8:
            pad = " " * (label_width - get_cwidth(label) + 2)
            rows.append(("class:easy-dim", pad + cli._trim_status_bar_text(
                safe_display(meta.replace("\n", " ")), room)))
        rows.append(("", "\n"))
    return rows[:-1]


# Keep common commands at the top of the menu without limiting the available commands.
_PRIMARY_COMMANDS = {
    name: index for index, name in enumerate(
        ("/model", "/resume", "/clear", "/new", "/approvals", "/help", "/status"))
}


class ComposerCompleter(Completer):
    def __init__(self, completer):
        self.completer = completer

    def get_completions(self, document, complete_event):
        matches = self.completer.get_completions(document, complete_event)
        prefix = document.text_before_cursor
        if not prefix.startswith("/") or any(ch.isspace() for ch in prefix):
            yield from matches
            return
        yield from sorted(matches, key=lambda item: (
            item.display_text.lower() != prefix.lower(),
            _PRIMARY_COMMANDS.get(item.display_text, len(_PRIMARY_COMMANDS)),
        ))

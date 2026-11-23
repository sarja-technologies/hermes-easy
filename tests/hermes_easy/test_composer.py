from types import SimpleNamespace

import pytest

from prompt_toolkit.application import Application, set_app
from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import CompleteEvent, Completion, WordCompleter
from prompt_toolkit.data_structures import Size
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.widgets import TextArea

from hermes_easy.composer import ComposerCompleter, build, completion_rows, partial_response
from hermes_easy.dialogs import choice_rows
from hermes_easy.scrollback import strip_controls


def test_slash_completion_favors_model_and_keeps_user_draft():
    completer = ComposerCompleter(WordCompleter(["/moa", "/model"], WORD=True))
    typed = Document("/mo")
    choices = list(completer.get_completions(typed, CompleteEvent(text_inserted=True)))
    assert [item.text for item in choices] == ["/model", "/moa"]
    buf = Buffer(document=typed)
    buf.complete_state = CompletionState(typed, [Completion(choice.text) for choice in choices])
    buf.complete_state.go_to_index(1)
    cli = SimpleNamespace(_get_tui_terminal_width=lambda: 46, _trim_status_bar_text=lambda text, width: text[:width])
    assert "› /moa" in "".join(text for _, text in completion_rows(cli, buf))
    assert buf.text == "/mo"


def test_partial_answer_at_the_composer_preserves_the_unfinished_line():
    cli = SimpleNamespace(
        _stream_buf="A short unfinished line", _stream_box_live=True, _easy_answer_marker=True,
        _get_tui_terminal_width=lambda: 46, final_response_markdown="strip",
    )
    assert partial_response(cli) == [("", "● A short unfinished line")]
    cli._stream_buf = ""
    assert partial_response(cli) == []


def test_approval_choices_stay_visible_in_a_short_terminal():
    cli = SimpleNamespace(
        _app=SimpleNamespace(output=SimpleNamespace(get_size=lambda: SimpleNamespace(rows=12))),
        _get_tui_terminal_width=lambda: 46, _trim_status_bar_text=lambda text, width: text[:width],
    )
    labels = ["Allow once", "Allow for session", "Always allow", "Deny"]
    fragments = choice_rows(cli, {"selected": 3}, "Tool permission", ["long command"] * 20, labels, "Esc to deny")
    rendered = "".join(text for _, text in fragments)
    assert "› 4. Deny" in rendered
    assert len(rendered.splitlines()) < 12


def test_transient_output_strips_complete_escape_sequences():
    assert strip_controls("one\x1b[2Jtwo\x1b]52;clipboard\x07three\r") == "onetwothree"


@pytest.mark.asyncio
async def test_composer_borders_and_input_fill_the_rendered_terminal_after_resize():
    size = Size(rows=28, columns=104)
    output = DummyOutput()
    output.get_size = lambda: size
    cli = SimpleNamespace(
        _clarify_state=None, _status_bar_suppressed_after_resize=False,
        _pet_widget=None, _stash_panel_widget=None, _subagent_dock_widget=None,
        _get_tui_terminal_width=lambda: output.get_size().columns,
    )
    draft = "the draft survives a resize"
    editor = TextArea(text=draft, prompt="› ", wrap_lines=True, height=1)
    parts = dict.fromkeys((
        "approval_widget", "slash_confirm_widget", "clarify_widget", "model_picker_widget",
        "command_palette_widget", "connection_widget", "sudo_widget", "secret_widget",
    ))
    parts.update(input_area=editor, image_bar=Window(height=0), voice_status_bar=Window(height=0),
                 status_bar=Window(height=0), spacer=Window(height=0))

    with create_pipe_input() as pipe:
        app = Application(
            layout=Layout(HSplit(build(cli, parts, [])), focused_element=editor),
            input=pipe, output=output,
        )
        try:
            with set_app(app):
                for size in (Size(28, 104), Size(19, 46), Size(32, 160), Size(28, 104)):
                    app.renderer.render(app, app.layout)
                    screen = app.renderer._last_screen
                    position = screen.visible_windows_to_write_positions[editor.window]
                    borders = [[x for x, char in screen.data_buffer[y].items() if char.char == "─"]
                               for y in (position.ypos - 1, position.ypos + position.height)]
                    for border in borders:
                        assert len(border) >= size.columns - 4
                        assert sorted(border) == list(range(position.xpos, position.xpos + position.width))
                    cells = screen.data_buffer[position.ypos]
                    assert "".join(cells[x].char for x in range(position.xpos, position.xpos + len(draft) + 2)) == "› " + draft
        finally:
            await app.cancel_and_wait_for_background_tasks()

"""hermes-easy UI settings: persistence, validation, presets, panel actions and home isolation."""

from __future__ import annotations

import colorsys
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_easy
from hermes_easy import settings as ui
from hermes_easy import settings_panel as panel
from hermes_easy import theme


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    monkeypatch.setenv(hermes_easy.ENV_FLAG, "1")
    monkeypatch.setattr(ui, "_current", None)
    monkeypatch.setattr(ui, "_problems", [])
    # The panel re-themes the live skin; keep the skin engine change out of other tests.
    monkeypatch.setattr(theme, "install", lambda settings=None: None)
    yield


def _fake_cli():
    return SimpleNamespace(_ui_settings_state=panel.PanelState(), _app=None, _focus_view_enabled=False,
                           tool_progress_mode="all", _status_bar_visible=True)


def test_defaults_round_trip(tmp_path):
    ui.save(ui.defaults(), tmp_path)
    loaded, problems = ui.load(tmp_path)
    assert loaded == ui.defaults()
    assert problems == []


def test_missing_file_gives_defaults(tmp_path):
    assert ui.load(tmp_path) == (ui.defaults(), [])


def test_hand_edited_file_is_read_and_bad_values_fall_back(tmp_path):
    (tmp_path / "ui.yaml").write_text(
        "theme:\n  preset: forest\n  accent: 'd97757'\n  mode: purple\n"
        "layout:\n  spacing: compact\nstatus:\n  key_hints: 'off'\nunknown: {x: 1}\n", encoding="utf-8")
    loaded, problems = ui.load(tmp_path)
    assert loaded["theme"]["preset"] == "forest"
    assert loaded["theme"]["accent"] == "#D97757"  # missing '#' is added, case normalised
    assert loaded["theme"]["mode"] == "auto"  # invalid -> default
    assert loaded["layout"]["spacing"] == "compact"
    assert loaded["status"]["key_hints"] is False
    assert any("theme.mode" in p for p in problems)


def test_unparseable_file_never_blocks_startup(tmp_path):
    (tmp_path / "ui.yaml").write_text("theme: [unclosed\n", encoding="utf-8")
    loaded, problems = ui.load(tmp_path)
    assert loaded == ui.defaults()
    assert problems


def test_panel_change_persists_across_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "settings_path", lambda home=None: Path(tmp_path) / "ui.yaml")
    cli = _fake_cli()
    cli._ui_settings_state.index = cli._ui_settings_state.rows.index("theme.preset")
    panel.change(cli, +1)  # graphite -> monochrome
    panel.change(cli, +1)  # -> nordic
    assert ui.get("theme.preset") == "nordic"
    assert cli._ui_settings_state.changed
    # A new process: drop the in-memory copy and read the file again.
    monkeypatch.setattr(ui, "_current", None)
    assert ui.get("theme.preset") == "nordic"
    assert 'preset: "nordic"' in (tmp_path / "ui.yaml").read_text(encoding="utf-8")


def test_hex_edit_validates_and_backspace_clears(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "settings_path", lambda home=None: Path(tmp_path) / "ui.yaml")
    monkeypatch.setattr(ui, "truecolor_supported", lambda: True)
    cli = _fake_cli()
    st = cli._ui_settings_state
    st.index = st.rows.index("theme.accent")
    panel.activate(cli)
    st.editing = "#"
    for ch in "12zz34ab":  # non-hex keys are ignored
        panel.type_char(cli, ch)
    assert st.editing == "#1234AB"
    panel.commit_edit(cli)
    assert ui.get("theme.accent") == "#1234AB"
    assert theme.palette()["accent"] == "#1234AB"
    panel.backspace(cli)
    assert ui.get("theme.accent") == ""


def test_hex_edit_refused_without_truecolor(monkeypatch):
    monkeypatch.setattr(ui, "truecolor_supported", lambda: False)
    cli = _fake_cli()
    cli._ui_settings_state.index = cli._ui_settings_state.rows.index("theme.accent")
    panel.activate(cli)
    assert cli._ui_settings_state.editing is None
    assert "truecolor" in cli._ui_settings_state.message


def test_reset_row_needs_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "settings_path", lambda home=None: Path(tmp_path) / "ui.yaml")
    cli = _fake_cli()
    s = ui.defaults()
    s["layout"]["tool_output"] = "hidden"
    panel.apply(cli, s)
    assert cli.tool_progress_mode == "off"
    cli._ui_settings_state.index = cli._ui_settings_state.rows.index(panel.RESET_ROW)
    panel.activate(cli)
    assert ui.get("layout.tool_output") == "hidden"  # first Enter only asks
    panel.activate(cli)
    assert ui.current() == ui.defaults()
    assert cli.tool_progress_mode == "all"


def test_status_off_hides_status_bar(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "settings_path", lambda home=None: Path(tmp_path) / "ui.yaml")
    cli = _fake_cli()
    s = ui.defaults()
    s["status"]["line"] = "off"
    panel.apply(cli, s)
    assert cli._status_bar_visible is False


@pytest.mark.parametrize("preset", [c[0] for c in ui.PRESET_CHOICES])
@pytest.mark.parametrize("mode", ["dark", "light"])
def test_no_preset_uses_violet(preset, mode):
    s = ui.defaults()
    s["theme"].update(preset=preset, mode=mode)
    for role, value in theme.palette(s).items():
        if not isinstance(value, str) or not value.startswith("#"):
            continue
        r, g, b = (int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))
        h, light, sat = colorsys.rgb_to_hls(r, g, b)
        if sat > 0.15 and 0.1 < light < 0.95:
            assert not 255 <= h * 360 <= 325, (preset, mode, role, value)


def test_skin_covers_every_upstream_style_key():
    from hermes_cli.skin_engine import _STYLE_PALETTE
    colors = theme.skin_data(ui.defaults())["colors"]
    for _name, key, _fallback in _STYLE_PALETTE:
        assert key in colors, key


def test_xterm256_fallback_maps_grey_and_colour():
    assert theme.xterm256(0, 0, 0) == 16
    assert 232 <= theme.xterm256(128, 128, 128) <= 255
    assert theme.xterm256(255, 0, 0) == 196


# ── layout rows ──────────────────────────────────────────────────────────────────────────

def test_activity_text_drops_kaomoji_and_emoji_but_keeps_shell_prompt():
    from hermes_easy.layout import clean_activity
    assert clean_activity("(｡•́︿•̀｡) pondering...") == "pondering..."
    assert clean_activity("📖 Read calc.py") == "Read calc.py"
    assert clean_activity("💻 $ ls -la") == "$ ls -la"


def test_hidden_tool_output_prints_nothing_but_keeps_output_for_ctrl_o():
    from hermes_easy import layout
    s = ui.defaults()
    s["layout"]["tool_output"] = "hidden"
    ui.replace(s)
    cli = SimpleNamespace()
    assert layout.tool_row(cli, "terminal", {"command": "echo hi"}, 0.2, '{"output": "hi\\n", "exit_code": 0}') is None
    rows = layout.expand_last_tool_output(cli)
    assert any("hi" in r for r in rows)


def test_compact_terminal_result_omits_empty_output_metadata():
    from hermes_easy import layout
    row = layout.tool_row(SimpleNamespace(), "terminal", {"command": "true"}, 0.1,
                          '{"output": "", "exit_code": 0}')
    assert "true" in row
    assert "↳" not in row


def test_preview_density_shows_bounded_output():
    from hermes_easy import layout
    s = ui.defaults()
    s["layout"]["tool_output"] = "preview"
    ui.replace(s)
    out = "\n".join(f"line {i}" for i in range(20))
    row = layout.tool_row(SimpleNamespace(), "terminal", {"command": "seq 20"}, 0.3,
                          f'{{"output": "{out.replace(chr(10), chr(92) + "n")}", "exit_code": 0}}')
    assert "line 3" in row and "line 4" not in row and "16 more lines" in row


def test_user_rows_layouts():
    from hermes_easy import layout
    cli = SimpleNamespace(user_message_preview_first_lines=2, user_message_preview_last_lines=2)
    for style, needle in (("glyph", "›"), ("labeled", "You"), ("ruled", "─" * 10)):
        s = ui.defaults()
        s["layout"]["messages"] = style
        ui.replace(s)
        assert needle in "\n".join(layout.user_lines(cli, "hello"))
    long_rows = layout.user_lines(cli, "\n".join(str(i) for i in range(10)))
    assert any("6 more lines" in r for r in long_rows)


# ── home isolation ───────────────────────────────────────────────────────────────────────

def test_upstream_home_is_refused(monkeypatch, tmp_path):
    from hermes_easy import entry
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(SystemExit):
        entry.resolve_home({"HERMES_HOME": str(tmp_path / ".hermes")})
    assert entry.resolve_home({}) == tmp_path / ".hermes-easy"


def test_prepare_environment_defaults_to_own_home(monkeypatch, tmp_path):
    from hermes_easy import entry
    monkeypatch.setenv("HOME", str(tmp_path))
    env: dict = {}
    home = entry.prepare_environment(env)
    assert home == tmp_path / ".hermes-easy"
    assert env["HERMES_HOME"] == str(home)
    assert env["HERMES_DATA_DIR_SUFFIX"] == "-easy"
    assert env[hermes_easy.ENV_FLAG] == "1"


def test_import_upstream_copies_provider_settings_only_and_never_writes_source(monkeypatch, tmp_path):
    from hermes_easy import entry
    src = tmp_path / "upstream"
    dst = Path(os.environ["HERMES_HOME"])  # the conftest's isolated home
    src.mkdir()
    (src / "config.yaml").write_text(
        "model:\n  default: m1\n  provider: local\nproviders:\n  local:\n    base_url: http://127.0.0.1:1/v1\n"
        "    key_env: LOCAL_LLM_KEY\nplatforms:\n  telegram:\n    enabled: true\n", encoding="utf-8")
    (src / ".env").write_text("LOCAL_LLM_KEY=abc\nOPENROUTER_API_KEY=or-1\nTELEGRAM_BOT_TOKEN=t\n",
                              encoding="utf-8")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in src.iterdir()}
    assert entry.import_upstream(["--from", str(src), "--yes"], dst) == 0
    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in src.iterdir()}
    assert before == after
    import hermes_yaml
    cfg = hermes_yaml.safe_load((dst / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["default"] == "m1" and "local" in cfg["providers"]
    assert "platforms" not in cfg
    env_text = (dst / ".env").read_text(encoding="utf-8")
    assert "LOCAL_LLM_KEY=abc" in env_text and "OPENROUTER_API_KEY=or-1" in env_text
    assert "TELEGRAM_BOT_TOKEN" not in env_text


def test_reduced_motion_freezes_the_indicator():
    from hermes_easy import layout
    s = ui.defaults()
    s["motion"]["reduced_motion"] = True
    ui.replace(s)
    assert {layout.indicator_frame() for _ in range(5)} == {"•"}
    s["motion"]["reduced_motion"] = False
    s["motion"]["spinner"] = "line"
    ui.replace(s)
    assert layout.indicator_frame() in "-\\|/"


def test_tool_row_paths_are_relative_to_the_working_directory(monkeypatch, tmp_path):
    from hermes_easy import layout
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "proj"))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert layout.short_paths(f"{tmp_path}/proj/src/a.py") == "src/a.py"
    assert layout.short_paths(f"{tmp_path}/other/b.py") == "~/other/b.py"
    assert layout.short_paths("/etc/hosts") == "/etc/hosts"


def test_fenced_code_drops_fences_and_keeps_markdown_characters():
    from hermes_easy import layout
    cli = SimpleNamespace()
    assert layout.stream_code_line(cli, "Here:") is None
    assert layout.stream_code_line(cli, "```python") == ""
    row = layout.stream_code_line(cli, "x = a * b * c")
    assert row is not None
    assert re.sub(r"\x1b\[[\d;]+m", "", row) == "  x = a * b * c"
    assert layout.stream_code_line(cli, "```") == ""
    assert layout.stream_code_line(cli, "after") is None


def test_reasoning_stream_flushes_long_partial_thought_early(monkeypatch):
    import cli as climod
    from cli import HermesCLI
    monkeypatch.setenv("HERMES_EASY_UI", "1")
    rows = []
    monkeypatch.setattr(climod, "_cprint", lambda text: rows.append(text or ""))
    instance = HermesCLI.__new__(HermesCLI)
    instance._modern_layout = True
    instance._stream_box_opened = False
    instance._reasoning_box_opened = False
    instance._stream_reasoning_delta("Let me examine the problem and compare the available approaches carefully")
    assert any("examine the problem" in row for row in rows)
    assert instance._reasoning_buf

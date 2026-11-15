"""``/ui-settings``: an in-place settings panel above the composer.

It lives inside the running prompt_toolkit application (like upstream's model picker), so
every change re-themes the real chrome around it at once: the composer rule, the status
line and the completion menu. A small sample transcript inside the panel previews the
scrollback rows, which are already printed and cannot be restyled after the fact.

Keys: ↑/↓ or Tab/Shift+Tab move · ←/→ change · Enter/Space change or edit · Backspace
clears a colour · d restores the focused row's default · Esc closes.
Every change is written to ``ui.yaml`` straight away.
"""

from __future__ import annotations

import copy

from hermes_easy import settings as ui
from hermes_easy import theme
from hermes_easy.layout import tilde

RESET_ROW = "__reset__"


class PanelState:
    def __init__(self):
        self.rows: list[str] = [f.path for f in ui.FIELDS] + [RESET_ROW]
        self.index = 0
        self.editing: str | None = None  # hex text being typed for the focused row
        self.message = ""
        self.confirm_reset = False
        self.changed = False

    @property
    def path(self) -> str:
        return self.rows[self.index]


def _settings() -> dict:
    return copy.deepcopy(ui.current())


def apply(cli, settings: dict, *, persist: bool = True) -> None:
    """Make *settings* current: re-theme the live app and the CLI knobs, then save."""
    ui.replace(settings)
    theme.install(settings)
    apply_cli_knobs(cli)
    st = getattr(cli, "_ui_settings_state", None)
    if st is not None:
        st.changed = True
    if persist:
        try:
            path = ui.save(settings)
            message = f"Saved to {tilde(path)}"
        except OSError as exc:
            message = f"Could not save: {exc}"
        if st is not None:
            st.message = message
    if getattr(cli, "_app", None) is not None:
        cli._apply_tui_skin_style()


def apply_cli_knobs(cli) -> None:
    """Settings that map onto upstream CLI state rather than colours."""
    density = ui.get("layout.tool_output")
    if not getattr(cli, "_focus_view_enabled", False):
        cli.tool_progress_mode = "off" if density == "hidden" else "all"
    cli._status_bar_visible = ui.get("status.line") != "off"


def _cycle(field: ui.Field, value, step: int):
    if field.is_bool:
        return not value
    values = [c[0] for c in field.choices]
    try:
        i = values.index(value)
    except ValueError:
        i = 0
    return values[(i + step) % len(values)]


def change(cli, step: int) -> None:
    st = cli._ui_settings_state
    if st is None:
        return
    st.confirm_reset = False if st.path != RESET_ROW else st.confirm_reset
    if st.path == RESET_ROW:
        return
    field = ui.FIELD_BY_PATH[st.path]
    if field.is_hex:
        return
    s = _settings()
    s[field.section][field.key] = _cycle(field, s[field.section][field.key], step)
    apply(cli, s)


def activate(cli) -> None:
    """Enter / Space on the focused row."""
    st = cli._ui_settings_state
    if st is None:
        return
    if st.path == RESET_ROW:
        if not st.confirm_reset:
            st.confirm_reset = True
            st.message = "Press Enter again to reset every setting to its default."
            return
        st.confirm_reset = False
        apply(cli, ui.defaults())
        st.message = f"Defaults restored · {st.message}"
        return
    field = ui.FIELD_BY_PATH[st.path]
    if field.is_hex:
        if not ui.truecolor_supported():
            st.message = "Custom colours need a truecolor terminal (COLORTERM=truecolor)."
            return
        st.editing = ui.get(st.path) or "#"
        st.message = "Type #RRGGBB, Enter to apply, Esc to cancel."
        return
    change(cli, +1)


def move(cli, step: int) -> None:
    st = cli._ui_settings_state
    if st is None or st.editing is not None:
        return
    st.index = (st.index + step) % len(st.rows)
    st.confirm_reset = False
    st.message = ""


def type_char(cli, ch: str) -> None:
    st = cli._ui_settings_state
    if st is None or st.editing is None:
        return
    if ch in "0123456789abcdefABCDEF" and len(st.editing) < 7:
        st.editing += ch.upper()
    elif ch == "#" and not st.editing:
        st.editing = "#"


def backspace(cli) -> None:
    st = cli._ui_settings_state
    if st is None:
        return
    if st.editing is not None:
        st.editing = st.editing[:-1]
        return
    field = ui.FIELD_BY_PATH.get(st.path)
    if field is not None and field.is_hex and ui.get(st.path):
        s = _settings()
        s[field.section][field.key] = ""
        apply(cli, s)


def commit_edit(cli) -> None:
    st = cli._ui_settings_state
    if st is None or st.editing is None:
        return
    field = ui.FIELD_BY_PATH[st.path]
    text = st.editing if st.editing not in ("", "#") else ""
    ok, value = ui.validate_value(field, text)
    if not ok:
        st.message = f"{st.editing} is not a #RRGGBB colour."
        return
    st.editing = None
    s = _settings()
    s[field.section][field.key] = value
    apply(cli, s)


def restore_default(cli) -> None:
    st = cli._ui_settings_state
    if st is None or st.path == RESET_ROW or st.editing is not None:
        return
    field = ui.FIELD_BY_PATH[st.path]
    s = _settings()
    s[field.section][field.key] = field.default
    apply(cli, s)


def escape(cli) -> None:
    st = cli._ui_settings_state
    if st is None:
        return
    if st.editing is not None:
        st.editing = None
        st.message = ""
        return
    close(cli)


def open_panel(cli) -> None:
    cli._ui_settings_state = PanelState()


def close(cli) -> None:
    st, cli._ui_settings_state = cli._ui_settings_state, None
    from hermes_easy import layout
    if st is not None and st.changed:
        from cli import _cprint
        _cprint(f"{theme.hex_fg(theme.palette()['dim'])}UI settings saved to {tilde(ui.settings_path())}{theme.RESET}")
    layout.note_external_output()


# ── rendering ────────────────────────────────────────────────────────────────────────────

def _display_value(field: ui.Field, value) -> str:
    if field.is_bool:
        return "On" if value else "Off"
    if field.is_hex:
        return value or ""
    return dict(field.choices).get(value, str(value))


def _hex_default(field: ui.Field, pal: dict) -> str:
    return {"accent": pal.get("accent"), "foreground": pal.get("text"), "background": None}.get(field.key) or ""


def fragments(cli) -> list:
    st = cli._ui_settings_state
    if st is None:
        return []
    pal = theme.palette()
    accent, strong, dim, faint = (f"fg:{pal[k]}" for k in ("accent", "strong", "dim", "faint"))
    sel_bg = f"bg:{pal['selection']}"
    truecolor = ui.truecolor_supported()
    width = max(40, cli._get_tui_terminal_width())
    label_w = 20
    out: list = [(f"{strong} bold", "UI settings"),
                 (dim, "   ↑↓ move · ←→ change · enter edit · d default · esc close\n\n")]
    for i, path in enumerate(st.rows):
        focused = i == st.index
        bg = f" {sel_bg}" if focused else ""
        if path == RESET_ROW:
            text = "Reset all to defaults"
            out += [("", "\n"), (f"{accent if focused else dim}{bg}", f"  {text}".ljust(width - 2)), ("", "\n")]
            continue
        field = ui.FIELD_BY_PATH[path]
        value = ui.get(path)
        shown = _display_value(field, value)
        row = [(f"{strong}{bg}", f"  {field.label}".ljust(label_w + 2))]
        if field.is_hex:
            if focused and st.editing is not None:
                row += [(f"{strong} bold{bg}", st.editing), (f"{accent}{bg}", "▏")]
            elif not truecolor:
                row += [(f"{dim}{bg}", "needs a truecolor terminal")]
            elif shown:
                row += [(f"fg:{shown}{bg}", "██ "), (f"{strong}{bg}", shown)]
            else:
                fallback = _hex_default(field, pal)
                row += ([(f"fg:{fallback}{bg}", "██ ")] if fallback else []) + [
                    (f"{dim}{bg}", "preset" if fallback else "terminal default")]
        elif focused and not field.is_bool:
            row += [(f"{accent}{bg}", "‹ "), (f"{strong} bold{bg}", shown), (f"{accent}{bg}", " ›")]
        else:
            style = (f"{accent} bold" if value else dim) if field.is_bool else strong
            row += [(f"{style}{bg}", shown)]
        used = sum(len(t) for _, t in row)
        row.append((bg.strip(), " " * max(0, width - 2 - used)))
        out += row + [("", "\n")]
    focused_field = ui.FIELD_BY_PATH.get(st.path)
    out.append(("", "\n"))
    help_text = st.message or (focused_field.help if focused_field else
                               "Restores every key in ui.yaml to the value listed in the README.")
    out.append((dim, f"  {help_text}\n"))
    rows_available = getattr(getattr(cli, "_app", None), "output", None)
    try:
        height = rows_available.get_size().rows if rows_available else 40
    except Exception:  # health: allow BLE001 -- size probe on a closing output
        height = 40
    if height >= len(st.rows) + 16:
        out += [("", "\n")] + preview_fragments(pal)
    return out


def preview_fragments(pal: dict) -> list:
    """A sample turn in the current layout settings (the real rows use the same rules)."""
    accent, strong, dim, faint, err = (f"fg:{pal[k]}" for k in ("accent", "strong", "dim", "faint", "error"))
    text = f"fg:{pal['text']}" if pal.get("text") else ""
    spacing = ui.get("layout.spacing") == "comfortable"
    style = ui.get("layout.messages")
    density = ui.get("layout.tool_output")
    gap = [("", "\n")] if spacing else []
    out: list = [(dim, "  Preview\n")]
    if style == "ruled":
        out.append((faint, "  " + "─" * 40 + "\n"))
    if style == "labeled":
        out += [(f"{dim} bold", "  You\n"), (f"{strong} bold", "  Why does the build fail?\n")]
    else:
        out += [(f"{accent} bold", "  › "), (f"{strong} bold", "Why does the build fail?\n")]
    if density != "hidden":
        out += gap + [(f"{err} bold", "  ✗ "), (f"{strong} bold", "Ran"), ("", " make build"), (dim, "  2.1s"),
                      (err, " [exit 2]\n")]
        if density in {"preview", "full"}:
            out += [(faint, "    │ "), (dim, "main.c:12: error: unknown type 'size'\n")]
        out += [(dim, "  • "), (f"{strong} bold", "Read"), ("", " main.c\n")]
    out += gap
    if style == "labeled":
        out.append((f"{accent} bold", "  Hermes\n"))
    out.append((text, "  Line 12 uses size where size_t was meant.\n"))
    if ui.get("status.turn_summary"):
        out += gap + [(dim, "  6s · ran 1 command · read 1 file\n")]
    return out

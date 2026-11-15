"""The hermes-easy layer on ``HermesCLI``: identity line, /ui-settings and its panel, Ctrl+O.

Mixed in first so these methods sit in front of upstream's. Every override hands straight
back to upstream when :func:`hermes_easy.active` is false, which keeps the stock ``hermes``
behaviour (and upstream's own tests) unchanged.
"""

from __future__ import annotations

import os
import threading

import hermes_easy
from hermes_easy import layout
from hermes_easy import settings as ui
from hermes_easy import settings_panel as panel
from hermes_easy import theme


def _upstream(name: str):
    """Upstream's implementation of *name* (the next class after EasyUIMixin in HermesCLI's MRO).

    Not ``super()``: upstream's tests call these unbound on stand-in objects
    (``HermesCLI._handle_update_command(SimpleNamespace(...))``), where super() raises.
    """
    from cli import HermesCLI
    mro = HermesCLI.__mro__
    for klass in mro[mro.index(EasyUIMixin) + 1:]:
        if name in klass.__dict__:
            return klass.__dict__[name]
    raise AttributeError(name)


class EasyUIMixin:
    _ui_settings_state: panel.PanelState | None = None

    # ── startup ──────────────────────────────────────────────────────────────────────────

    def show_banner(self):
        if not hermes_easy.active():
            return _upstream("show_banner")(self)
        from cli import _cprint
        theme.install()
        panel.apply_cli_knobs(self)
        _cprint(layout.identity_line(getattr(self, "model", "") or ""))
        for problem in ui.startup_problems():
            _cprint(f"{theme.hex_fg(theme.palette()['warn'])}ui.yaml: {problem}{theme.RESET}")
        # Real warnings stay (a toolset whose credentials are missing, a too-small context);
        # they are computed off the startup path, as upstream does on its warm launch.
        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            threading.Thread(target=self._show_tool_availability_warnings,
                             name="tool-availability-warnings", daemon=True).start()
        self._easy_context_warning()

    def _easy_context_warning(self) -> None:
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        compressor = getattr(getattr(self, "agent", None), "context_compressor", None)
        ctx_len = getattr(compressor, "context_length", None)
        if ctx_len and ctx_len < MINIMUM_CONTEXT_LENGTH:
            from agent.i18n import t
            self._console_print(f"[yellow]{t('cli.banner.context_too_low', tokens=f'{ctx_len:,}')}[/]")

    def _agent_status_print(self, *args, **kwargs) -> None:
        if hermes_easy.active() and len(args) == 1 and isinstance(args[0], str):
            args = (layout.clean_status(args[0]),)
        return _upstream("_agent_status_print")(self, *args, **kwargs)

    def _print_random_tip(self) -> None:
        if hermes_easy.active():
            return None
        return _upstream("_print_random_tip")(self)

    def _cmd_clear(self, cmd_original: str):
        """Upstream's /clear (same confirmation, same new session), redrawn as the identity line."""
        if not hermes_easy.active():
            return _upstream("_cmd_clear")(self, cmd_original)
        from cli import _clear_output_history, _cprint
        if self._confirm_destructive_slash(
            "clear",
            "This clears the screen and starts a new session.\n"
            "The current conversation history will be discarded.",
            cmd_original=cmd_original,
        ) is None:
            return True
        self.new_session(silent=True)
        _clear_output_history()
        if self._app:
            out = self._app.output
            out.erase_screen()
            out.cursor_goto(0, 0)
            out.flush()
        else:
            self.console.clear()
        layout.reset_blocks()
        _cprint(layout.identity_line(getattr(self, "model", "") or ""))
        _cprint(f"{theme.hex_fg(theme.palette()['dim'])}New session.{theme.RESET}")
        return None

    # ── live region ──────────────────────────────────────────────────────────────────────

    def _get_extra_tui_widgets(self) -> list:
        widgets = list(_upstream("_get_extra_tui_widgets")(self))
        if not hermes_easy.active():
            return widgets
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.layout import ConditionalContainer, FormattedTextControl, Window
        widgets.append(ConditionalContainer(
            Window(FormattedTextControl(lambda: panel.fragments(self)), wrap_lines=False),
            filter=Condition(lambda: self._ui_settings_state is not None)))
        return widgets

    def _register_extra_tui_keybindings(self, kb, *, input_area) -> None:
        _upstream("_register_extra_tui_keybindings")(self, kb, input_area=input_area)
        if not hermes_easy.active():
            return
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.keys import Keys

        is_open = Condition(lambda: self._ui_settings_state is not None)
        editing = Condition(lambda: self._ui_settings_state is not None
                            and self._ui_settings_state.editing is not None)
        browsing = is_open & ~editing

        def bind(keys, handler, flt=is_open):
            for key in keys:
                kb.add(*key if isinstance(key, tuple) else (key,), filter=flt, eager=True)(
                    lambda event, h=handler: (h(), event.app.invalidate()))

        bind(["up", "s-tab"], lambda: panel.move(self, -1), browsing)
        bind(["down", "tab"], lambda: panel.move(self, +1), browsing)
        bind(["left"], lambda: panel.change(self, -1), browsing)
        bind(["right"], lambda: panel.change(self, +1), browsing)
        bind(["enter", " "], lambda: panel.activate(self), browsing)
        bind(["enter"], lambda: panel.commit_edit(self), editing)
        bind(["d"], lambda: panel.restore_default(self), browsing)
        bind(["backspace"], lambda: panel.backspace(self), is_open)
        bind(["escape", "c-c"], lambda: panel.escape(self), is_open)

        # Swallow everything else while the panel is open (it must not reach the composer);
        # hex digits go to the colour being edited.
        @kb.add(Keys.Any, filter=is_open, eager=True)
        def _swallow(event):
            if self._ui_settings_state is not None and self._ui_settings_state.editing is not None:
                panel.type_char(self, event.data)
            event.app.invalidate()

        @kb.add("c-o", filter=~is_open, eager=True)
        def _expand_tool_output(event):
            from cli import _cprint
            for row in layout.expand_last_tool_output(self):
                _cprint(row)
            event.app.invalidate()

    def show_help(self, arg: str = ""):
        if not hermes_easy.active() or (arg or "").strip().lower() in {"skills", "skill"}:
            return _upstream("show_help")(self, arg)
        from cli import _cprint
        from hermes_easy.help import render
        for row in render(self, arg):
            _cprint(row)

    def _handle_update_command(self, *args, **kwargs):
        if not hermes_easy.active():
            return _upstream("_handle_update_command")(self, *args, **kwargs)
        from hermes_easy.entry import UPDATE_HELP
        print(UPDATE_HELP)
        return False

    # ── /ui-settings ─────────────────────────────────────────────────────────────────────

    def _handle_ui_settings_command(self, cmd_original: str = "") -> None:
        from cli import _cprint
        parts = (cmd_original or "").split()[1:]
        sub = parts[0].lower() if parts else ""
        dim, rst = theme.hex_fg(theme.palette()["dim"]), theme.RESET
        if not hermes_easy.active():
            _cprint("/ui-settings is part of the hermes-easy UI; launch `hermes-easy` to use it.")
            return
        if not sub:
            if getattr(self, "_app", None) is None:
                sub = "show"
            else:
                panel.open_panel(self)
                self._invalidate(min_interval=0.0)
                return
        if sub == "path":
            _cprint(str(ui.settings_path()))
        elif sub == "show":
            for f in ui.FIELDS:
                _cprint(f"  {f.path:<24} {dim}{ui.get(f.path)!r}{rst}")
            _cprint(f"  {dim}{layout.tilde(ui.settings_path())}{rst}")
        elif sub == "reset":
            panel.apply(self, ui.defaults())
            _cprint("UI settings reset to defaults.")
        elif sub == "preset" and len(parts) == 2:
            self._easy_set("theme.preset", parts[1])
        elif sub == "set" and len(parts) >= 2:
            self._easy_set(parts[1], " ".join(parts[2:]))
        else:
            _cprint("Usage: /ui-settings [show | reset | path | preset <name> | set <section.key> <value>]")
        layout.note_external_output()

    def _easy_set(self, path: str, value: str) -> None:
        from cli import _cprint
        field = ui.FIELD_BY_PATH.get(path)
        if field is None:
            _cprint(f"Unknown setting {path!r}. Keys: {', '.join(ui.FIELD_BY_PATH)}")
            return
        ok, normalised = ui.validate_value(field, value)
        if not ok:
            allowed = ", ".join(c[0] for c in field.choices) if field.choices else (
                "true, false" if field.is_bool else "#RRGGBB or empty")
            _cprint(f"{value!r} is not valid for {path}. Allowed: {allowed}")
            return
        settings = {s: dict(v) for s, v in ui.current().items()}
        settings[field.section][field.key] = normalised
        panel.apply(self, settings)
        _cprint(f"{path} = {normalised!r}")

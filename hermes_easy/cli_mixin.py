"""The hermes-easy layer on ``HermesCLI``: identity line, /ui-settings and its panel, Ctrl+O.

Mixed in first so these methods sit in front of upstream's. Every override hands straight
back to upstream when :func:`hermes_easy.active` is false, which keeps the stock ``hermes``
behaviour (and upstream's own tests) unchanged.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import hermes_easy
from hermes_easy import composer, dialogs, layout
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
        from hermes_easy.model_status import run
        if not getattr(self, "_easy_status_thread_started", False):
            self._easy_status_thread_started = True
            threading.Thread(target=run, args=(self,), name="easy-model-status", daemon=True).start()

    def _display_resumed_history(self):
        if not hermes_easy.active():
            return _upstream("_display_resumed_history")(self)
        from cli import _cprint
        for line in layout.resumed_rows(self):
            _cprint(line)

    def _easy_context_warning(self) -> None:
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        compressor = getattr(getattr(self, "agent", None), "context_compressor", None)
        ctx_len = getattr(compressor, "context_length", None)
        if ctx_len and ctx_len < MINIMUM_CONTEXT_LENGTH:
            from agent.i18n import t
            self._console_print(f"[yellow]{t('cli.banner.context_too_low', tokens=f'{ctx_len:,}')}[/]")

    def _agent_status_print(self, *args, **kwargs) -> None:
        if hermes_easy.active() and len(args) == 1 and isinstance(args[0], str):
            if args[0].lstrip().startswith("↻"):  # the runtime is recovering (cut-off reply, retry, reconnect)
                layout.ledger_for(self).note_recovery(args[0])
            args = (layout.clean_status(args[0]),)
        return _upstream("_agent_status_print")(self, *args, **kwargs)

    # ── activity ledger (real runtime events only) ───────────────────────────────────────

    def _on_tool_progress(self, event_type, function_name=None, preview=None, function_args=None, **kwargs):
        if hermes_easy.active():
            try:
                self._easy_record_event(event_type, function_name, function_args, kwargs)
            except Exception:  # health: allow BLE001 -- bookkeeping must never break a turn
                pass
        return _upstream("_on_tool_progress")(self, event_type, function_name, preview, function_args, **kwargs)

    def _turn_summary_emit(self) -> None:
        """End of a turn (finished, failed or interrupted): print the tool group still open, then the footer."""
        if hermes_easy.active():
            from cli import _cprint
            for row in layout.flush_group():
                _cprint(row)
        return _upstream("_turn_summary_emit")(self)

    def _on_thinking(self, text):
        # The runtime's recovery notices (``↻ …``) arrive here as diagnostics, which the CLI may not render.
        if hermes_easy.active() and isinstance(text, str) and text.lstrip().startswith("↻"):
            layout.ledger_for(self).note_recovery(text)
        return _upstream("_on_thinking")(self, text)

    def _stream_delta(self, text):
        if hermes_easy.active() and text:
            layout.ledger_for(self).note_text()
        return _upstream("_stream_delta")(self, text)

    def _stream_reasoning_delta(self, text):
        if hermes_easy.active() and text:
            layout.ledger_for(self).note_reasoning()
        return _upstream("_stream_reasoning_delta")(self, text)

    def _easy_record_event(self, event_type, name, args, kwargs) -> None:
        from agent.display import _detect_tool_failure, get_tool_row_parts, is_neutral_failure
        led = layout.ledger_for(self)
        if event_type == "tool.started" and name:
            parts = get_tool_row_parts(name, args or {})
            label = " ".join(p for p in (parts[0][1], parts[0][2]) if p) if parts else name
            if name == "terminal" and (args or {}).get("command"):  # the command as run, not upstream's "+ 2 commands" digest
                label = "$ " + " ".join(str(args["command"]).split())[:160]
            task_id = getattr(getattr(self, "agent", None), "session_id", None)
            led.tool_started(name, args, label=label or name, task_id=task_id)
        elif event_type == "tool.completed" and name:
            result = kwargs.get("result")
            failed, _suffix = _detect_tool_failure(name, result)
            failed = bool(failed or kwargs.get("is_error"))
            led.tool_completed(name, result, failed=failed, neutral=failed and is_neutral_failure(name, result),
                               duration=kwargs.get("duration"))
        elif event_type and event_type.startswith("reasoning"):
            led.note_reasoning()

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

    def _build_tui_layout_children(self, **kwargs):
        if not hermes_easy.active():
            return _upstream("_build_tui_layout_children")(self, **kwargs)
        return composer.build(self, kwargs, self._get_extra_tui_widgets())

    def _tui_build_input_area(self):
        area = _upstream("_tui_build_input_area")(self)
        if hermes_easy.active():
            from cli import _estimate_tui_input_height
            from prompt_toolkit.completion import ThreadedCompleter
            from prompt_toolkit.filters import Condition
            from hermes_cli.commands_completion import cli_slash_completer

            area.buffer.completer = ThreadedCompleter(composer.ComposerCompleter(cli_slash_completer(self)))
            area.window.get_line_prefix = (
                lambda line, wrap: "  " if line or wrap else "")
            area.buffer.auto_suggest = None
            area.buffer.read_only |= Condition(lambda: composer.choice_active(self))

            def input_height():
                document = area.buffer.document
                lines = list(document.lines)
                # The caret needs a cell even when it sits just beyond a full line.
                if document.is_cursor_at_the_end_of_line:
                    lines[document.cursor_position_row] += " "
                width = composer.content_width(self._get_tui_terminal_width())
                # A large prompt gets room to be read: up to half the terminal, never the whole screen.
                rows = composer.terminal_rows(self)
                max_height = max(1, min(max(8, rows // 2), rows - 6))
                return _estimate_tui_input_height(lines, "", max(1, width - 2), max_height=max_height)

            area.window.height = input_height
        return area

    def _build_tui_style_dict(self):
        styles = _upstream("_build_tui_style_dict")(self)
        if hermes_easy.active():
            colors = theme.palette()
            styles.update({
                "input-rule": colors["faint"],
                "easy-dim": colors["dim"],
                "easy-title": f"{colors['accent']} bold",
                "easy-selected": f"{colors['accent']} bold",
                "placeholder": colors["dim"],
            })
        return styles

    def _tui_placeholder_text(self):
        if not hermes_easy.active() or composer.modal_active(self) or self._voice_mode:
            return _upstream("_tui_placeholder_text")(self)
        return self._prompt_stash.placeholder_hint() or ("" if self._agent_running else "Message Hermes or type /")

    def _get_approval_display_fragments(self):
        state = self._approval_state
        if hermes_easy.active() and state:
            return dialogs.approval(self, state)
        return _upstream("_get_approval_display_fragments")(self)

    def _handle_approval_selection(self):
        state = self._approval_state
        if hermes_easy.active() and state:
            choices = state.get("choices") or []
            index = state.get("selected", 0)
            if 0 <= index < len(choices) and choices[index] == "view":
                from cli import _cprint
                _cprint("\nTool command\n" + composer.safe_display(state["command"]))
        return _upstream("_handle_approval_selection")(self)

    def _get_clarify_display_fragments(self):
        state = self._clarify_state
        if hermes_easy.active() and state:
            return dialogs.clarify(self, state)
        return _upstream("_get_clarify_display_fragments")(self)

    def _get_slash_confirm_display_fragments(self):
        state = self._slash_confirm_state
        if hermes_easy.active() and state:
            return dialogs.confirmation(self, state)
        return _upstream("_get_slash_confirm_display_fragments")(self)

    def _render_scroll_list_panel(self, state, title, hint, labels, **kwargs):
        if not hermes_easy.active():
            return _upstream("_render_scroll_list_panel")(self, state, title, hint, labels, **kwargs)
        hints = hint if isinstance(hint, (list, tuple)) else [hint]
        footer = "↑/↓ to move · Enter to select · Esc to close"
        if state is self._model_picker_state:
            stage = state.get("stage")
            title = layout.clean_activity(title)
            if stage == "model":
                provider = state.get("provider_data") or {}
                name = provider.get("name") or provider.get("slug")
                title = f"Select model · {name}" if name else "Select model"
                current = state.get("current_model")
                if provider.get("is_current"):
                    labels = [f"{label}  ✓" if label == current else label for label in labels]
                footer = "← Providers · Enter to select · Esc to close"
        return dialogs.choice_rows(self, state, title, hints, labels, footer, limit=6, numbered=False)

    def _open_model_picker(self, providers, current_model, current_provider, **kwargs):
        _upstream("_open_model_picker")(self, providers, current_model, current_provider, **kwargs)
        if not hermes_easy.active():
            return
        state = self._model_picker_state
        if providers and providers[state["selected"]].get("is_current"):
            # Start in the current provider's catalog. Back still opens the provider list.
            _upstream("_handle_model_picker_selection")(self)
            models = state.get("model_list") or []
            if current_model in models:
                state["selected"] = models.index(current_model)
            self._paint_now()

    def _tui_enter_inline_command(self, event, text, has_images):
        if not hermes_easy.active() or not self._should_handle_model_command_inline(text, has_images):
            return _upstream("_tui_enter_inline_command")(self, event, text, has_images)
        event.current_buffer.reset(append_to_history=True)
        event.app.create_background_task(self._easy_open_model(text))
        return True

    async def _easy_open_model(self, text):
        from cli import _cprint

        self._command_running = self._command_blocks_input = True
        self._command_status = "Finding models…"
        self._paint_now()
        try:
            await asyncio.to_thread(self._tui_run_slash_input, text)
        except Exception as exc:
            _cprint(layout.error_line(f"Could not open model selection. {exc}"))
        finally:
            self._command_running = self._command_blocks_input = False
            self._command_status = ""
            self._paint_now()

    def _tui_handle_paste(self, event):
        if hermes_easy.active() and composer.choice_active(self):
            return
        return _upstream("_tui_handle_paste")(self, event)

    def _easy_request_turn_cancel(self, *, event=None) -> None:
        """Non-blocking, idempotent cancellation of an active generation or command.

        Never blocks the prompt_toolkit event loop on gateway RPCs and never recurses
        or exits the CLI when pressed repeatedly while cancellation is settling.
        """
        from agent.interrupt_compat import request_hard_interrupt
        from contextlib import suppress

        self._easy_exit_hint_until = 0
        self._easy_cancel_requested = True
        now = time.monotonic()
        if not getattr(self, "_easy_cancel_requested_at", 0.0):
            self._easy_cancel_requested_at = now
        # Post-cancel debounce: rapid Ctrl+C presses in the same burst that arrive just after
        # the turn finishes must never accidentally trigger idle double-Ctrl+C exit.
        self._easy_cancel_grace_until = now + 1.0
        self._last_turn_interrupted = True
        if getattr(self, "_agent_running", False):
            self._spinner_text = "Cancelling…"
        elif getattr(self, "_command_running", False):
            self._command_status = "Cancelling…"
        with suppress(Exception):
            self._clear_active_overlays_for_interrupt()
        agent = getattr(self, "agent", None)
        if agent is not None and not getattr(self, "_easy_cancel_in_flight", False):
            self._easy_cancel_in_flight = True
            with suppress(Exception):
                if hasattr(agent, "_interrupt_requested"):
                    agent._interrupt_requested = True

            def _send_interrupt(target=agent):
                with suppress(Exception):
                    request_hard_interrupt(target)

            threading.Thread(target=_send_interrupt, name="easy-turn-cancel", daemon=True).start()
        app = getattr(event, "app", None) if event is not None else getattr(self, "_app", None)
        if app is not None:
            with suppress(Exception):
                app.invalidate()

    def _tui_handle_ctrl_c(self, event):
        if not hermes_easy.active():
            return _upstream("_tui_handle_ctrl_c")(self, event)
        from contextlib import suppress

        # 1. Cancel voice recording or open settings panel first.
        if self._tui_cancel_voice_recording(event):
            self._easy_exit_hint_until = 0
            return
        if self._ui_settings_state is not None:
            panel.escape(self)
            self._easy_exit_hint_until = 0
            event.app.invalidate()
            return
        # 2. Cancel foreground modals / overlays cleanly.
        if self._tui_cancel_foreground_ui(event, closers=(
            ("_slash_confirm_state", lambda: self._submit_slash_confirm_response("cancel")),
            ("_model_picker_state", lambda: self._close_model_picker()),
            ("_command_palette_state", lambda: self._close_command_palette()),
            ("_gateway_command_wait", lambda: self._gateway_command_wait.set()),
        )):
            self._easy_exit_hint_until = 0
            return
        overlay_cleared = self._tui_clear_blocking_overlays(event)
        if overlay_cleared and not (self._agent_running or self._command_running):
            self._easy_exit_hint_until = 0
            return
        # 3. Clear typed draft / attached images before cancelling a turn or exiting.
        buf = event.app.current_buffer
        if buf.text or self._attached_images:
            self._easy_exit_hint_until = 0
            buf.reset()
            self._attached_images.clear()
            event.app.invalidate()
            return
        # 4. Active generation or command (including during model loading / agent init):
        #    idempotently cancel and return to the ready composer; repeated Ctrl+C never
        #    recurses, freezes, or kills the interactive session.
        if self._agent_running or self._command_running:
            self._easy_request_turn_cancel(event=event)
            return
        # 5. Idle and empty composer: absorb stray rapid Ctrl+C right after a turn cancel,
        #    otherwise first Ctrl+C shows exit hint, second within 2s exits cleanly.
        now = time.monotonic()
        if now < getattr(self, "_easy_cancel_grace_until", 0.0):
            self._easy_exit_hint_until = 0
            event.app.invalidate()
            return
        if now >= getattr(self, "_easy_exit_hint_until", 0):
            self._easy_exit_hint_until = now + 2.0
            event.app.invalidate()
            with suppress(RuntimeError):
                asyncio.get_running_loop().call_later(2.0, event.app.invalidate)
            return
        self._easy_exit_hint_until = 0
        self._should_exit = True
        event.app.exit()

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

        self._easy_composer_keys(kb, input_area)

        @kb.add("c-o", filter=Condition(lambda: not composer.modal_active(self)), eager=True)
        def _expand_tool_output(event):
            from cli import _cprint
            for row in layout.expand_last_tool_output(self):
                _cprint(row)
            event.app.invalidate()

    def _easy_composer_keys(self, kb, input_area):
        from prompt_toolkit.filters import Condition

        buf = input_area.buffer
        normal = Condition(lambda: not composer.modal_active(self) and not self._prompt_stash.panel_open)
        menu = normal & Condition(lambda: bool(buf.complete_state and buf.complete_state.completions))
        slash = menu & Condition(lambda: buf.text.startswith("/") and not self._command_blocks_input)
        has_picked = Condition(lambda: bool(buf.complete_state and buf.complete_state.complete_index is not None))
        name_only = Condition(lambda: not any(ch.isspace() for ch in buf.text))

        @kb.add("down", filter=menu, eager=True)
        @kb.add("up", filter=menu, eager=True)
        def move(event):
            state = buf.complete_state
            index = state.complete_index if state.complete_index is not None else 0
            delta = 1 if event.key_sequence[0].key == "down" else -1
            state.go_to_index((index + delta) % len(state.completions))
            event.app.invalidate()

        @kb.add("enter", filter=slash & (name_only | has_picked), eager=True)
        def submit_completion(event):
            state = buf.complete_state
            item = state.completions[state.complete_index or 0]
            buf.apply_completion(item)
            self._tui_submit_buffer(event, buf)

        @kb.add("escape", filter=menu)
        def close_completion(event):
            # Selection moves only the menu cursor. Keep the original text.
            buf.complete_state = None
            event.app.invalidate()

        @kb.add("s-tab", filter=normal & Condition(lambda: not buf.text), eager=True)
        def approval_menu(event):
            buf.insert_text("/approvals ")
            event.app.invalidate()

        @kb.add("left", filter=Condition(lambda: bool(
            self._model_picker_state and self._model_picker_state.get("stage") == "model"
        )), eager=True)
        def providers(event):
            state = self._model_picker_state
            pairs = state.get("_filtered_pairs")
            state["selected"] = len(state.get("model_list", []) if pairs is None else pairs)
            self._handle_model_picker_selection()

        @kb.add("escape", filter=Condition(lambda: bool(self._approval_state)), eager=True)
        def deny(event):
            state = self._approval_state
            if state:
                state["selected"] = state["choices"].index("deny")
                self._handle_approval_selection()

        @kb.add("escape", filter=Condition(lambda: bool(self._clarify_state)), eager=True)
        def cancel_question(event):
            self._clear_active_overlays_for_interrupt()
            event.app.invalidate()

        @kb.add("escape", filter=normal & ~menu & Condition(
            lambda: bool(self._agent_running or self._command_running)))
        def stop_turn(event):
            self._easy_request_turn_cancel(event=event)

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

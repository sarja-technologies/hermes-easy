"""hermes-easy transcript rows and composer chrome.

Same contract as upstream's ``cli_modern_layout``: every row printed into scrollback is
unboxed, so resizing the terminal leaves the native scrollback readable. Functions that return ``None`` print nothing (``_cprint`` skips ``None``).

Block grammar::

    › your message                  accent glyph, your text in the strong colour
    thinking text …                 dim
    Thought for 3s                  dim italic
    • Read calc.py  0.2s            one row per tool call (✗ in the error colour on failure)
    assistant text                  terminal foreground (or theme.foreground)
    32s · read 1 file               dim turn summary
"""

from __future__ import annotations

import json
import os
import re
import time

from hermes_easy import settings as ui
from hermes_easy.theme import RESET, hex_fg, palette

_RULE_WIDTH = 40  # fixed: a printed rule must never re-wrap on resize
_PREVIEW_LINES = 4
_FULL_CAP = 400

SPINNERS = {
    "dots": ("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", 10.0),
    "line": ("-\\|/", 8.0),
    "pulse": (("·", "•", "●", "•"), 3.0),
}

# ── block spacing ────────────────────────────────────────────────────────────────────────

_last_block = {"kind": None}


def _gap(kind: str) -> str:
    """Leading newline owed before a block of *kind* (none between consecutive tool rows)."""
    prev = _last_block["kind"]
    _last_block["kind"] = kind
    if prev is None:
        return ""
    if kind == "tool" and prev == "tool":
        return ""
    if kind == "user":
        return "\n"
    return "\n" if ui.get("layout.spacing") == "comfortable" else ""


def reset_blocks() -> None:
    """A cleared screen: the next block is the first one again."""
    _last_block["kind"] = None


def note_external_output() -> None:
    """Something outside the grammar printed (a command's output): the next block opens fresh."""
    _last_block["kind"] = "system"


def _c(role: str, *, bold: bool = False, italic: bool = False) -> str:
    return hex_fg(palette().get(role), bold=bold, italic=italic)


# ── identity line ────────────────────────────────────────────────────────────────────────

def short_cwd() -> str:
    cwd = os.getenv("TERMINAL_CWD") or os.getcwd()
    home = os.path.expanduser("~")
    return "~" + cwd[len(home):] if cwd == home or cwd.startswith(home + os.sep) else cwd


def display_model(model: str) -> str:
    """Make common local Qwen aliases legible; leave unknown provider names intact."""
    match = re.match(r"^qwen[-_]?([0-9]+(?:\.[0-9]+)?)[-_]([0-9]+b)(.*)$", model or "", re.IGNORECASE)
    if not match:
        return model or "No model selected"
    suffix = match.group(3).replace("-", " ").replace("_", " ").strip()
    return f"Qwen {match.group(1)} {match.group(2).upper()}" + (f" {suffix}" if suffix else "")


def identity_line(model: str) -> str:
    """The one line printed at launch: name, model, directory."""
    from hermes_easy import __version__
    _last_block["kind"] = "system"
    return (f"{_c('accent', bold=True)}Hermes Easy{RESET}  {_c('dim')}v{__version__}{RESET}\n"
            f"{_c('dim')}{display_model(model)} · {short_cwd()}{RESET}")


# ── user / assistant blocks ──────────────────────────────────────────────────────────────

def _glyph() -> str:
    return "›"


def user_lines(cli, text: str) -> list[str]:
    """Rows for a submitted message, long ones collapsed like upstream's preview."""
    from hermes_easy.scrollback import strip_controls
    lines = strip_controls(str(text or "")).split("\n")
    first = max(1, int(getattr(cli, "user_message_preview_first_lines", 2)))
    last = max(0, int(getattr(cli, "user_message_preview_last_lines", 2)))
    if len(lines) > first + last + 1:
        hidden = len(lines) - first - last
        lines = lines[:first] + [f"\x00… {hidden} more line{'s' if hidden != 1 else ''}"] + (
            lines[-last:] if last else [])
    style = ui.get("layout.messages")
    strong, accent = _c("strong"), _c("accent", bold=True)
    out = [""] if _gap("user") else []
    if style == "ruled":
        out.append(f"{_c('faint')}{'─' * _RULE_WIDTH}{RESET}")
    if style == "labeled":
        out.append(f"{_c('dim', bold=True)}You{RESET}")
    for i, line in enumerate(lines):
        if line.startswith("\x00"):
            out.append(f"  {_c('dim')}{line[1:]}{RESET}")
            continue
        lead = f"{accent}{_glyph()}{RESET} " if i == 0 and style != "labeled" else (
            "" if style == "labeled" else "  ")
        out.append(f"{lead}{strong}{line}{RESET}")
    stamp = ""
    if getattr(cli, "show_timestamps", False):
        from datetime import datetime
        stamp = f" {_c('dim')}{datetime.now().strftime(getattr(cli, 'timestamp_format', '%H:%M'))}{RESET}"
    if stamp:
        out[-1] += stamp
    return out


def resumed_rows(cli):
    """Replay the existing resume recap with the live conversation's role markers."""
    from cli import CLI_CONFIG, _strip_reasoning_tags
    from hermes_cli.cli_agent_setup_mixin import _collect_resume_entries
    from tools.ansi_strip import sanitize_display_text

    history = getattr(cli, "_resume_display_history", cli.conversation_history)
    if not history or cli.resume_display == "minimal":
        return
    display = CLI_CONFIG.get("display", {})
    entries, last_index, last_text = _collect_resume_entries(
        history, display, lambda text: sanitize_display_text(_strip_reasoning_tags(text)))
    skipped = max(0, len(entries) - max(0, int(display.get("resume_exchanges", 10))) * 2)
    if skipped:
        yield f"{_c('dim')}{skipped} earlier messages{RESET}"
    for index, (role, text) in enumerate(entries[skipped:], skipped):
        if role == "user":
            yield from user_lines(cli, text)
            continue
        if _gap("answer"):
            yield ""
        if index == last_index and last_text:
            text = last_text
        first, *rest = text.splitlines() or [""]
        if role == "event":
            yield f"{_c('dim')}↳ {first}{RESET}"
        else:
            yield f"● {first}"
        yield from ("  " + line for line in rest)


def answer_open(timestamp: str = "") -> str | None:
    """Printed before the first answer line: a gap, plus a name in the labeled layout."""
    gap = _gap("answer")
    label = ""
    if ui.get("layout.messages") == "labeled":
        label = f"{_c('accent', bold=True)}Hermes{RESET}"
    if timestamp:
        label = f"{label} {_c('dim')}{timestamp}{RESET}".strip()
    if label:
        return f"{gap}{label}" if gap else label
    return "" if gap else None


def answer_text_ansi() -> str:
    """Colour for streamed answer text: theme.foreground / preset text, else the terminal's."""
    return hex_fg(palette().get("text")) if palette().get("text") else ""


_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*(.*)$")


def stream_code_line(cli, line: str) -> str | None:
    """Render complete code rows without altering their content or indent."""
    match = _FENCE.fullmatch(line)
    opening = getattr(cli, "_easy_in_fence", False)
    if not (match or opening):
        return None
    from hermes_easy.code import highlight, lexer_for

    if not opening:
        delimiter, info = match.groups()
        if delimiter.startswith("`") and "`" in info:
            return None
        language = info.split(None, 1)[0] if info.strip() else ""
        cli._easy_in_fence = delimiter
        cli._easy_code_lexer = lexer_for(language)
        return ""
    if match and match[1][0] == opening[0] and len(match[1]) >= len(opening) and not match[2].strip():
        cli._easy_in_fence = False
        cli._easy_code_lexer = None
        return ""
    if not line:
        return " "
    return "  " + highlight(line, cli._easy_code_lexer)


# ── reasoning ────────────────────────────────────────────────────────────────────────────

def reasoning_open() -> str | None:
    gap = _gap("reasoning")
    return "" if gap else None


def reasoning_line(line: str) -> str:
    return f"{_c('dim')}{line}{RESET}"


def thought_line(seconds: float | None) -> str:
    took = f"Thought for {max(seconds or 0.0, 0.0):.0f}s" if seconds else "Thought"
    return f"{_c('dim', italic=True)}{took}{RESET}"


def thought_rows(seconds: float | None, text: str, full: bool) -> list[str]:
    lines = (text or "").strip().splitlines()
    shown = lines if full or len(lines) <= 10 else lines[:10]
    gap = _gap("reasoning")
    rows = [""] if gap else []
    rows += [reasoning_line(line) for line in shown]
    if len(shown) < len(lines):
        rows.append(f"{_c('dim')}… {len(lines) - len(shown)} more lines (/reasoning full){RESET}")
    return [*rows, thought_line(seconds)]


# ── tool rows ────────────────────────────────────────────────────────────────────────────

def _result_text(result, *, include_metadata: bool = True) -> str:
    """The human-readable part of a tool result (JSON results carry it under a few keys)."""
    if result is None:
        return ""
    text = result if isinstance(result, str) else str(result)
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return text
    if isinstance(data, dict):
        for key in ("output", "content", "stdout", "result", "text", "message", "error"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(data, indent=2, ensure_ascii=False) if include_metadata else ""
    return text


def _remember(cli, name: str, label: str, result) -> None:
    store = cli.__dict__.setdefault("_easy_tool_outputs", [])
    store.append((label or name, _result_text(result)))
    del store[:-20]


def _gutter_lines(text: str, limit: int | None) -> list[str]:
    lines = text.rstrip("\n").split("\n") if text.strip() else []
    shown = lines if limit is None else lines[:limit]
    from hermes_easy.composer import safe_display
    dim = _c("dim")
    out = [f"  {dim}{'↳' if i == 0 else ' '} {safe_display(line)}{RESET}"
           for i, line in enumerate(shown)]
    hidden = len(lines) - len(shown)
    if hidden > 0:
        out.append(f"    {dim}… {hidden} more lines · Ctrl+O for details{RESET}")
    return out


def tilde(path) -> str:
    """*path* with the home directory written as ~."""
    text, home = str(path), os.path.expanduser("~").rstrip(os.sep)
    return "~" + text[len(home):] if home and (text == home or text.startswith(home + os.sep)) else text


def short_paths(text: str) -> str:
    """Absolute paths in a tool row, shown relative to the working directory (else under ~)."""
    if not text or "/" not in text:
        return text
    cwd = (os.getenv("TERMINAL_CWD") or os.getcwd()).rstrip(os.sep)
    home = os.path.expanduser("~").rstrip(os.sep)
    if cwd and cwd != os.sep:
        text = text.replace(cwd + os.sep, "")
    if home and home != os.sep:
        text = text.replace(home + os.sep, "~" + os.sep)
    return text


def tool_row(cli, function_name: str, args: dict | None, duration: float, result=None,
             is_error: bool = False) -> str | None:
    """One row per completed call, plus an output preview when tool_output asks for it."""
    from agent.display import get_tool_row_parts, is_neutral_failure
    density = ui.get("layout.tool_output")
    rows = get_tool_row_parts(function_name, args or {}, result, is_error=bool(is_error))
    neutral = is_neutral_failure(function_name, result)
    label_for_expand = ""
    out = []
    for index, (_emoji, verb, detail, suffix) in enumerate(rows):
        verb = (verb or function_name).strip()
        failed = suffix is not None and not neutral
        mark = f"{_c('error', bold=True)}✗{RESET}" if failed else f"{_c('accent')}•{RESET}"
        line = f"{mark} {_c('strong', bold=True)}{verb[:1].upper()}{verb[1:]}{RESET}"
        detail = short_paths(detail) if isinstance(detail, str) else detail
        if detail:
            line += f" {detail}"
        label_for_expand = f"{verb[:1].upper()}{verb[1:]} {detail}".strip()
        if index == len(rows) - 1 and duration >= 0.1:
            line += f"  {_c('dim')}{duration:.1f}s{RESET}"
        if suffix:
            line += f"{_c('dim') if neutral else _c('error')}{suffix}{RESET}"
        out.append(line)
    _remember(cli, function_name, label_for_expand, result)
    if density == "hidden":
        return None
    if density == "compact" and function_name == "terminal":
        output = _result_text(result, include_metadata=False).splitlines()
        if output and output[0].strip():
            from hermes_easy.composer import content_width, safe_display
            width = content_width(cli._get_tui_terminal_width()) - 4
            preview = cli._trim_status_bar_text(safe_display(output[0]), max(1, width))
            out.append(f"  {_c('dim')}↳ {preview}{RESET}")
    if density in {"preview", "full"}:
        out += _gutter_lines(_result_text(result), _PREVIEW_LINES if density == "preview" else _FULL_CAP)
    return _gap("tool") + "\n".join(out)


def expand_last_tool_output(cli) -> list[str]:
    """Rows Ctrl+O prints: the latest call's full output under its label."""
    store = cli.__dict__.get("_easy_tool_outputs") or []
    if not store:
        return [f"{_c('dim')}No tool output yet.{RESET}"]
    label, text = store[-1]
    rows = [""] if ui.get("layout.spacing") == "comfortable" else []
    rows += [f"{_c('dim')}•{RESET} {_c('strong', bold=True)}{label}{RESET} {_c('dim')}output{RESET}"]
    body = _gutter_lines(text, _FULL_CAP)
    note_external_output()
    return rows + (body or [f"  {_c('faint')}│{RESET} {_c('dim')}(no output){RESET}"])


# ── turn footer ──────────────────────────────────────────────────────────────────────────

def turn_footer(summary: str) -> str | None:
    if not ui.get("status.turn_summary"):
        return None
    body = summary.lstrip("⋯ ").strip()
    if not body:
        return None
    return f"{_gap('footer')}{_c('dim')}{body}{RESET}"


# ── live region ──────────────────────────────────────────────────────────────────────────

_LEAD_JUNK = re.compile(r"^\s*(?:[^\w\s($]+|\([^)\w]*\)|\S*[　-ヿ̀-ͯ＀-￯]\S*)\s+")


def clean_activity(text: str) -> str:
    """Drop a leading emoji / kaomoji token from upstream's spinner text."""
    text = (text or "").strip()
    # The skin may receive a face from the runtime before it installs its own spinner.
    if text.startswith("("):
        face, close, rest = text.partition(")")
        if close and rest.startswith(" ") and not any(ch.isalnum() for ch in face):
            text = rest.lstrip()
    for _ in range(2):
        stripped = _LEAD_JUNK.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    return text.strip()


_STATUS_EMOJI = re.compile(r"^(\s*)[\u2190-\u2bff\U0001F000-\U0001FAFF][\ufe0f]?\s+")


def clean_status(text: str) -> str:
    """Agent status lines (``⚡ Interrupted during API call.``): the leading pictograph goes, the
    line is dimmed so it reads as status rather than as the answer."""
    if not isinstance(text, str) or "\x1b[" in text[:6] or "\n" in text:
        return text
    stripped = _STATUS_EMOJI.sub(r"\1", text, count=1)
    if stripped == text:
        return text
    note_external_output()
    return f"{_c('dim')}{stripped}{RESET}"


def indicator_frame() -> str:
    style = ui.get("motion.spinner")
    if ui.get("motion.reduced_motion") or style == "static" or style not in SPINNERS:
        return "•"
    frames, fps = SPINNERS[style]
    return frames[int(time.monotonic() * fps) % len(frames)]


def activity_fragments(cli) -> list:
    """One transient conversation row, reserved for activity and actionable model state."""
    from hermes_cli.cli_modern_layout import _turn_elapsed, activity_text
    model = getattr(cli, "model", "") or ""
    name = display_model(model)
    snapshot = getattr(cli, "_easy_model_status", ())
    state = snapshot[2] if snapshot[:2] == (model, getattr(cli, "base_url", None)) else ""
    busy = getattr(cli, "_agent_running", False) or getattr(cli, "_command_running", False)
    descriptions = {
        "failed": f"{name} failed to load",
        "disconnected": f"Cannot reach the model endpoint for {name}",
        "missing": f"{name} is missing from the model endpoint",
    }
    if busy:
        descriptions["loading"] = f"Loading {name}…"
    activity = descriptions.get(state)
    if not activity and busy:
        activity = clean_activity(activity_text(cli)) or "Working…"
    if not activity:
        return []
    problem = state in {"failed", "disconnected", "missing"}
    mark = "!" if problem else indicator_frame()
    rows = [(f"fg:{palette()['error' if problem else 'accent']}", mark + " "), ("", activity)]
    elapsed = _turn_elapsed(cli) if busy else ""
    if elapsed:
        rows.append(("class:easy-dim", "  " + elapsed))
    return rows


def error_line(message: str) -> str:
    return f"{_c('error', bold=True)}!{RESET} {message}"


def top_rule_fragments(cli) -> list:
    from hermes_easy.composer import content_width
    return [("class:input-rule", "─" * content_width(cli._get_tui_terminal_width()))]


def status_line_fragments(cli) -> list:
    """Show effective permissions and the active model, plus measured decode speed."""
    from hermes_easy.composer import content_width
    from tools.approval import is_approval_bypass_active_for_session
    from tools.approval_context import _get_approval_mode

    level = ui.get("status.line")
    if level == "off":
        return []
    if level == "full":
        from hermes_cli.cli_modern_layout import _status_line_fragments
        return _status_line_fragments(cli)
    width = content_width(cli._get_tui_terminal_width())
    if time.monotonic() < getattr(cli, "_easy_exit_hint_until", 0):
        return [("class:easy-dim", "  Press Ctrl+C again to exit")]
    mode = _get_approval_mode()
    bypass = is_approval_bypass_active_for_session(getattr(cli, "session_id", None) or "default")
    permission = "Approval bypass" if bypass else {
        "smart": "Smart approvals", "manual": "Manual approvals",
    }.get(mode, "Approvals enabled")
    model_id = getattr(cli, "model", None)
    model = display_model(model_id or "")
    agent = getattr(cli, "agent", None)
    rate = getattr(agent, "_easy_decode_rate", None)
    measured = getattr(agent, "_easy_decode_model", None) == model_id
    rate_label = f"{rate:.1f} tok/s" if measured and isinstance(rate, (int, float)) and 0 < rate < 1e9 else ""
    blocks = [
        ("class:easy-title" if not bypass else f"fg:{palette()['warn']}", "  " + permission),
        ("class:easy-dim", model),
    ]
    if level == "standard" and rate_label and not getattr(cli, "_agent_running", False):
        blocks.append(("class:easy-dim", rate_label))
    if ui.get("status.key_hints") and level != "minimal":
        from prompt_toolkit.application import get_app
        if getattr(cli, "_agent_running", False):
            blocks.append(("class:easy-dim", "Esc to interrupt"))
        elif not get_app().current_buffer.text:
            hint = ("Ctrl+O for tool details" if getattr(cli, "_easy_tool_outputs", None)
                    else "Shift+Tab for approvals")
            blocks.append(("class:easy-dim", hint))
    frags = []
    used = 0
    for index, (style, value) in enumerate(blocks):
        separator = " · " if frags else ""
        budget = width - used - len(separator)
        if index >= 2 and cli._status_bar_display_width(value) > budget:
            continue
        if budget <= 1:
            break
        if separator:
            frags.append(("class:easy-dim", separator))
        value = cli._trim_status_bar_text(value, budget)
        frags.append((style, value))
        used += len(separator) + cli._status_bar_display_width(value)
    return frags

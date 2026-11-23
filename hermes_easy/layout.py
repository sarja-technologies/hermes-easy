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
import shutil
import time

from hermes_easy import activity
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
    flushed = flush_group()
    gap = _gap("answer")
    label = ""
    if ui.get("layout.messages") == "labeled":
        label = f"{_c('accent', bold=True)}Hermes{RESET}"
    if timestamp:
        label = f"{label} {_c('dim')}{timestamp}{RESET}".strip()
    if label:
        label = f"{gap}{label}" if gap else label
    elif gap:
        label = ""
    else:
        label = None
    return _with_rows(flushed, label)


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
    flushed = flush_group()
    gap = _gap("reasoning")
    return _with_rows(flushed, "" if gap else None)


def reasoning_line(line: str) -> str:
    """Provider-exposed reasoning, in the near-white strong colour: it is what the agent is doing."""
    return f"{_c('strong')}{line}{RESET}"


def thought_line(seconds: float | None) -> str:
    took = f"Thought for {max(seconds or 0.0, 0.0):.0f}s" if seconds else "Thought"
    return f"{_c('dim', italic=True)}{took}{RESET}"


def thought_rows(seconds: float | None, text: str, full: bool) -> list[str]:
    lines = (text or "").strip().splitlines()
    shown = lines if full or len(lines) <= 10 else lines[:10]
    flushed = flush_group()
    gap = _gap("reasoning")
    rows = flushed + ([""] if gap else [])
    rows += [reasoning_line(line) for line in shown]
    if len(shown) < len(lines):
        rows.append(f"{_c('dim')}… {len(lines) - len(shown)} more lines (/reasoning full){RESET}")
    return [*rows, thought_line(seconds)]


# ── tool groups ──────────────────────────────────────────────────────────────────────────

_CATEGORY_ROLE = {"shell": "tool", "tests": "tool", "browser": "tool", "read": "accent", "search": "accent",
                  "web": "accent", "agent": "accent", "edit": "strong", "other": "dim"}
_hint_shown = {"done": False}


def ledger_of(cli=None):
    """The turn ledger of *cli* (or the process's one CLI); None before any turn."""
    led = getattr(cli, "__dict__", {}).get("_easy_ledger") if cli is not None else None
    return led or activity.CURRENT


def _with_rows(rows: list[str], tail: str | None) -> str | None:
    """Rows printed before a block, then the block's own opening (None = nothing to print)."""
    if not rows:
        return tail
    return "\n".join(rows + ([tail] if tail is not None else []))


def _dur(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 60 else f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


def _error_excerpt(call, lines: int) -> list[str]:
    text = (call.error or call.output or "").strip("\n")
    rows = [r for r in text.split("\n") if r.strip()]
    return rows[-lines:] if call.output and not call.error else rows[:lines]


def group_rows(group) -> list[str]:
    """The collapsed form of a finished group: one summary row, failures listed under it."""
    from hermes_easy.composer import safe_display
    parts = activity.summary_parts(group)
    if not parts:
        return []
    failed = group.failed
    mark = f"{_c('error', bold=True)}✗{RESET}" if failed else f"{_c('ok')}•{RESET}"
    summary = f"{_c('dim')} · {RESET}".join(
        f"{_c(_CATEGORY_ROLE.get(cat, 'dim'), bold=cat == 'edit')}{text}{RESET}" for cat, text in parts)
    total = sum(c.elapsed() for c in group.calls)
    line = f"{mark} {summary}"
    if total >= 0.1:
        line += f"  {_c('dim')}{_dur(total)}{RESET}"
    if not _hint_shown["done"] and ui.get("status.key_hints"):
        _hint_shown["done"] = True
        line += f"  {_c('dim')}Ctrl+O to expand{RESET}"
    rows = [_gap("tool") + line]
    columns = shutil.get_terminal_size((100, 24)).columns
    for call in failed[:3]:
        code = f"exit {call.exit_code}" if call.exit_code else "failed"
        tail = f"{code} · {_dur(call.elapsed())}"
        room = max(12, columns - len(tail) - 8)
        label = safe_display(call.label)
        label = label if len(label) <= room else label[:room - 1] + "…"  # the full command is in Ctrl+O
        rows.append(f"  {_c('error')}✗{RESET} {label}  {_c('dim')}{tail}{RESET}")
        rows += [f"    {_c('dim')}{'↳' if i == 0 else ' '} {safe_display(r)}{RESET}"
                 for i, r in enumerate(_error_excerpt(call, 3))]
    if len(failed) > 3:
        rows.append(f"  {_c('dim')}… {len(failed) - 3} more failed · Ctrl+O for all{RESET}")
    return rows


def group_detail_rows(group, *, tail: int = 8, failed_tail: int = 30) -> list[str]:
    """The expanded form: every call with its command, directory, machine, time, exit status and output."""
    from hermes_easy.composer import safe_display
    parts = activity.summary_parts(group)
    rows = [""] if ui.get("layout.spacing") == "comfortable" else []
    head = f"{_c('error', bold=True)}✗{RESET}" if group.failed else f"{_c('dim')}•{RESET}"
    rows.append(f"{head} " + f"{_c('dim')} · {RESET}".join(
        f"{_c(_CATEGORY_ROLE.get(cat, 'dim'), bold=cat == 'edit')}{text}{RESET}" for cat, text in parts))
    for call in group.calls:
        bad = call.failed and not call.neutral
        mark = f"{_c('error', bold=True)}✗{RESET}" if bad else f"{_c('ok')}✓{RESET}" if not call.running else f"{_c('tool')}…{RESET}"
        rows.append(f"  {mark} {_c('strong', bold=True)}{safe_display(call.label)}{RESET}")
        meta = [short_paths(tilde(call.cwd)) if call.cwd and call.category in {"shell", "tests"} else "",
                call.machine, call.agent, _dur(call.elapsed()),
                f"exit {call.exit_code}" if call.exit_code is not None else ""]
        rows.append(f"    {_c('dim')}{' · '.join(m for m in meta if m)}{RESET}")
        body = (call.output if not bad or call.output else call.error).rstrip("\n").split("\n")
        body = [b for b in body if b.strip()] if call.output or call.error else []
        limit = failed_tail if bad else tail
        shown = body[-limit:] if bad else body[:limit]
        rows += [f"    {_c('error' if bad else 'dim')}{safe_display(line)}{RESET}" for line in shown]
        if len(body) > len(shown):
            rows.append(f"    {_c('dim')}… {len(body) - len(shown)} more lines{RESET}")
    return rows


def flush_group() -> list[str]:
    """Close the running group (the model wrote text, reasoned, or the turn ended) and return its rows."""
    led = activity.CURRENT
    if led is None or ui.get("layout.tool_groups") != "grouped":
        return []
    group = led.close_group()
    if group is None:
        return []
    activity.save(led, led.session_id)
    return group_rows(group)


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
    led = ledger_of(cli)
    grouped = ui.get("layout.tool_groups") == "grouped" and led is not None and led.claim_completed(function_name)
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
    if grouped or density == "hidden":
        return None  # a grouped call is summarised when its group closes (flush_group)
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
    """Rows Ctrl+O prints. Grouped: the latest group in full (again within 6 s: the group before it).
    Rows mode, or nothing grouped yet: the latest call's output."""
    led = ledger_of(cli)
    if led is not None and ui.get("layout.tool_groups") == "grouped":
        now = time.monotonic()
        back = cli.__dict__.get("_easy_expand_back", -1) + 1 if now - cli.__dict__.get("_easy_expand_at", 0) < 6 else 0
        group = led.last_group(back)
        if group is None and back:
            back, group = 0, led.last_group(0)
        if group is not None:
            cli._easy_expand_back, cli._easy_expand_at = back, now
            note_external_output()
            return group_detail_rows(group)
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
        return _with_rows(flush_group(), None)
    flushed = flush_group()
    body = summary.lstrip("⋯ ").strip()
    if not body:
        return _with_rows(flushed, None)
    return _with_rows(flushed, f"{_gap('footer')}{_c('dim')}{body}{RESET}")


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


def ledger_for(cli):
    """The CLI's activity ledger, created on first use and synchronised with the running turn."""
    led = cli.__dict__.get("_easy_ledger")
    if led is None:
        led = cli._easy_ledger = activity.Ledger()
        activity.CURRENT = led
        sid = getattr(cli, "session_id", None)
        if isinstance(sid, str) and sid:
            led.session_id = sid
            activity.load(led, sid)
    started = getattr(cli, "_prompt_start_time", None)
    if getattr(cli, "_agent_running", False) and started and led.turn_started != started:
        led.start_turn(float(started))
        agent = getattr(cli, "agent", None)
        led.tokens0 = (int(getattr(agent, "session_input_tokens", 0) or 0),
                       int(getattr(agent, "session_output_tokens", 0) or 0))
    return led


def _fg(role: str, *, italic: bool = False) -> str:
    return ("italic " if italic else "") + f"fg:{palette()[role]}"


def live_row(cli) -> list:
    """The status row while the agent works: a playful word, then only real, current values."""
    from hermes_easy import status
    from hermes_easy.composer import content_width
    snap = status.snapshot(cli) if ledger_for(cli) is not None else status.Snapshot()
    word = snap.word if ui.get("status.activity_word") else ""
    frags = [(_fg("accent"), indicator_frame() + " ")]
    lead = 2
    if word:
        frags.append((_fg("dim", italic=True), word + "…  "))
        lead += len(word) + 3
    items = status.fit(status.fields(snap), content_width(cli._get_tui_terminal_width()), lead)
    for index, (_key, role, text) in enumerate(items):
        if index:
            frags.append(("class:easy-dim", "  "))
        frags.append((_fg(role), text))
    return frags


def live_tool_fragments(cli) -> list:
    """One row per command that is running right now (never after it finishes)."""
    from hermes_easy.composer import content_width, safe_display
    led = cli.__dict__.get("_easy_ledger")
    running = led.running_calls() if led is not None and getattr(cli, "_agent_running", False) else []
    if not running:
        return []
    width = content_width(cli._get_tui_terminal_width())
    now = time.time()
    frags: list = []
    for call in running[-3:]:
        tail = f"  {_dur(call.elapsed(now))}" + (f" · {call.machine}" if call.machine and len(running) > 1 else "")
        label = cli._trim_status_bar_text(safe_display(short_paths(call.label)), max(8, width - len(tail) - 4))
        if frags:
            frags.append(("", "\n"))
        frags += [(_fg("tool"), "  ▸ "), ("", label), ("class:easy-dim", tail)]
    if len(running) > 3:
        frags += [("", "\n"), ("class:easy-dim", f"    … {len(running) - 3} more running")]
    return frags


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
    activity_line = descriptions.get(state)
    problem = state in {"failed", "disconnected", "missing"}
    if not activity_line and getattr(cli, "_agent_running", False) and not problem:
        return live_row(cli)
    if not activity_line and busy:
        activity_line = clean_activity(activity_text(cli)) or "Working…"
    if not activity_line:
        return []
    mark = "!" if problem else indicator_frame()
    rows = [(f"fg:{palette()['error' if problem else 'accent']}", mark + " "), ("", activity_line)]
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

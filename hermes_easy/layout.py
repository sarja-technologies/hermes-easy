"""hermes-easy transcript rows and composer chrome.

Same contract as upstream's ``cli_modern_layout``: every row printed into scrollback is
width-independent (a column-0 glyph, no boxes, no full-width rules) so a resize never
re-wraps what is already on screen; width-dependent chrome lives only in prompt_toolkit's
live region. Functions that return ``None`` print nothing (``_cprint`` skips ``None``).

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

_DIM, _BOLD = "\x1b[2m", "\x1b[1m"
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


def identity_line(model: str) -> str:
    """The one line printed at launch: name, model, directory."""
    from hermes_easy import __version__
    _last_block["kind"] = "system"
    return (f"{_c('accent', bold=True)}hermes-easy{RESET} {_c('dim')}{__version__}{RESET}  "
            f"{_c('strong')}{model or 'no model'}{RESET} {_c('dim')}in {short_cwd()}{RESET}")


# ── user / assistant blocks ──────────────────────────────────────────────────────────────

def _glyph() -> str:
    return "›"


def user_lines(cli, text: str) -> list[str]:
    """Rows for a submitted message, long ones collapsed like upstream's preview."""
    lines = str(text or "").split("\n")
    first = max(1, int(getattr(cli, "user_message_preview_first_lines", 2)))
    last = max(0, int(getattr(cli, "user_message_preview_last_lines", 2)))
    if len(lines) > first + last + 1:
        hidden = len(lines) - first - last
        lines = lines[:first] + [f"\x00… {hidden} more line{'s' if hidden != 1 else ''}"] + (
            lines[-last:] if last else [])
    style = ui.get("layout.messages")
    strong, accent = _c("strong", bold=True), _c("accent", bold=True)
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


_FENCE = re.compile(r"^\s{0,3}(```+|~~~+)\s*([\w+#.-]*)\s*$")


def stream_code_line(cli, line: str) -> str | None:
    """Fenced code in a streamed answer: fence lines vanish ("" = print nothing), code lines
    keep their markdown characters and are indented in the code colour; None = not code."""
    if _FENCE.match(line):
        cli._easy_in_fence = not getattr(cli, "_easy_in_fence", False)
        return ""
    if not getattr(cli, "_easy_in_fence", False):
        return None
    return f"  {_c('strong')}{line}{RESET}" if line.strip() else " "


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

def _result_text(result) -> str:
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
        return json.dumps(data, indent=2, ensure_ascii=False)
    return text


def _remember(cli, name: str, label: str, result) -> None:
    store = cli.__dict__.setdefault("_easy_tool_outputs", [])
    store.append((label or name, _result_text(result)))
    del store[:-20]


def _gutter_lines(text: str, limit: int | None) -> list[str]:
    lines = text.rstrip("\n").split("\n") if text.strip() else []
    shown = lines if limit is None else lines[:limit]
    faint, dim = _c("faint"), _c("dim")
    out = [f"  {faint}│{RESET} {dim}{line}{RESET}" for line in shown]
    hidden = len(lines) - len(shown)
    if hidden > 0:
        out.append(f"  {faint}│{RESET} {dim}… {hidden} more line{'s' if hidden != 1 else ''} · ctrl+o shows all{RESET}")
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
        mark = f"{_c('error', bold=True)}✗{RESET}" if failed else f"{_c('dim')}•{RESET}"
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


def top_rule_fragments(cli) -> list:
    """Composer's top edge: a faint rule; while busy it carries the activity and elapsed time."""
    from hermes_cli.cli_modern_layout import _turn_elapsed, activity_text
    width = max(10, cli._get_tui_terminal_width())
    p = palette()
    rule = f"fg:{p['faint']}"
    busy = getattr(cli, "_agent_running", False) or getattr(cli, "_command_running", False)
    if not busy:
        return [(rule, "─" * width)]
    activity = clean_activity(activity_text(cli)) or "Working"
    frags = [(f"fg:{p['accent']} bold", indicator_frame() + " "), (f"fg:{p['strong']}", activity)]
    elapsed = _turn_elapsed(cli)
    if elapsed:
        frags.append((f"fg:{p['dim']}", f"  {elapsed}"))
    used = sum(cli._status_bar_display_width(text) for _, text in frags)
    if used + 2 > width:
        plain = "".join(text for _, text in frags)
        return [(f"fg:{p['strong']}", cli._trim_status_bar_text(plain, width))]
    return frags + [("", " "), (rule, "─" * max(0, width - used - 1))]


def _hints(cli) -> list[tuple[str, str]]:
    from hermes_cli.cli_modern_layout import hint_pairs
    pairs = hint_pairs(cli)
    if not getattr(cli, "_agent_running", False) and pairs and cli.__dict__.get("_easy_tool_outputs"):
        pairs = pairs + [("ctrl+o", "output")]
    return pairs


def status_line_fragments(cli) -> list:
    """Footer under the composer for the ``status.line`` level (``full`` is upstream's)."""
    level = ui.get("status.line")
    if level == "full":
        from hermes_cli.cli_modern_layout import _status_line_fragments
        return _status_line_fragments(cli)
    if level == "off":
        return []
    p = palette()
    dim, strong, warn = f"fg:{p['dim']}", f"fg:{p['strong']}", f"fg:{p['warn']} bold"
    width = max(10, cli._get_tui_terminal_width())
    snap = cli._get_status_bar_snapshot()
    left: list = [("", "  "), (strong, snap.get("model_short") or getattr(cli, "model", "") or "no model")]

    def add(style, text):
        if text:
            left.extend([(dim, " · "), (style, text)])

    percent = snap.get("context_percent")
    if percent is not None:
        mark = "~" if snap.get("context_estimated") else ""
        ctx_style = warn if percent >= 80 else dim
        add(ctx_style, f"{mark}{percent}%" if level == "minimal" else f"{mark}{percent}% context")
    if level == "standard":
        if cli._is_session_yolo_active():
            add(warn, "yolo")
        for key, word in (("active_background_tasks", "background"),
                          ("active_background_processes", "processes"),
                          ("active_background_subagents", "subagents")):
            if snap.get(key):
                add(strong, f"{snap[key]} {word}")
        goal = cli._status_bar_goal_segment(snap)
        if goal:
            add(strong, goal)
        stash = cli._prompt_stash.indicator() if hasattr(cli, "_prompt_stash") else ""
        add(strong, stash)
        add(strong, cli._vim_mode_label())
    right: list = []
    if ui.get("status.key_hints") and width >= 76:
        for key, text in _hints(cli):
            if right:
                right.append((dim, " · "))
            right += [(strong, key), (dim, f" {text}")]
    used_left = sum(cli._status_bar_display_width(t) for _, t in left)
    used_right = sum(cli._status_bar_display_width(t) for _, t in right)
    room = width - used_left - used_right - 6
    if level == "standard" and room >= 12:
        from hermes_cli.status_bar_git import current_git_branch
        cwd = short_cwd()
        branch = current_git_branch(os.getenv("TERMINAL_CWD") or os.getcwd())
        label = f"{cwd} ({branch})" if branch else cwd
        label = label if len(label) <= room else "…" + label[-(room - 1):]
        add(dim, label)
        used_left += 3 + cli._status_bar_display_width(label)
    if right and used_left + used_right + 2 > width:
        right, used_right = [], 0
    pad = [("", " " * max(1, width - used_left - used_right - 1))] if right else []
    frags = left + pad + right
    total = sum(cli._status_bar_display_width(t) for _, t in frags)
    if total > width:
        plain = "".join(t for _, t in frags)
        return [(dim, cli._trim_status_bar_text(plain, width))]
    return frags

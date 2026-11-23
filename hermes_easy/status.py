"""The live status row above the composer.

One row, left to right::

    ✻ Tinkering…  Executing terminal · 12s  qwen3-coder-next  ↑12.4k ↓1.1k  ctx 48%  3 tools  Linux

The first word is playful and rotates; it is decoration and never stands in for anything. Every
field after it is read from the runtime: the status word comes from the ledger's real state, the
elapsed time from the turn clock, tokens from the provider's usage, context from the compressor,
tools from the ledger. A field with no real value is left out rather than shown as a placeholder.
When the row is too narrow, fields drop from the right in the order given by ``_DROP_ORDER``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

PLAYFUL = (
    "Pondering", "Tinkering", "Percolating", "Noodling", "Untangling", "Marinating", "Whittling",
    "Rummaging", "Simmering", "Cogitating", "Puzzling", "Wrangling", "Sifting", "Polishing",
    "Foraging", "Brewing", "Spelunking", "Distilling", "Mulling", "Chiselling", "Kneading", "Threading",
)
_WORD_SECONDS = 2.6


def playful_word(started: float | None, now: float | None = None) -> str:
    """A rotating word, stable between ticks and different from turn to turn."""
    now = time.time() if now is None else now
    started = started or now
    return PLAYFUL[(int(started) + int(max(0.0, now - started) / _WORD_SECONDS)) % len(PLAYFUL)]


def compact(n: float | None) -> str:
    """12400 -> 12.4k, 1_250_000 -> 1.2M; None/0 -> ''."""
    if not n or n < 0:
        return ""
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return f"{n / 1_000_000:.1f}M".replace(".0M", "M")


@dataclass
class Snapshot:
    """Everything the row shows, as plain values (the renderer decides nothing about truth)."""

    state: str = "idle"          # ledger state
    status: str = ""             # real status word, e.g. "Executing terminal"
    tone: str = "accent"         # accent | tool | think | warn | error | ok
    note: str = ""               # what the runtime is recovering from, in its own words
    elapsed: str = ""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    context_pct: int | None = None
    tools: int = 0
    failures: int = 0
    reasoning: bool = False
    machine: str = ""
    word: str = ""
    agents: int = 0              # running subagents besides this one


_STATE_LABEL = {
    "writing": ("Writing", "accent"),
    "thinking": ("Thinking", "think"),
    "waiting": ("Thinking", "think"),
    "idle": ("Working", "accent"),
}


def snapshot(cli, now: float | None = None) -> Snapshot:
    """Read the live values off the CLI, its agent and its ledger."""
    from hermes_easy.layout import display_model, ledger_for
    now = time.time() if now is None else now
    ledger = ledger_for(cli)
    agent = getattr(cli, "agent", None)
    snap = Snapshot(model=display_model(getattr(cli, "model", "") or ""))
    if ledger is None:
        return snap
    snap.state = ledger.state
    snap.elapsed = _elapsed(ledger.turn_elapsed(now))
    snap.word = playful_word(ledger.turn_started, now)
    snap.tools, snap.failures = ledger.calls_this_turn, ledger.failures_this_turn
    snap.reasoning = ledger.reasoning_active(now)
    running = ledger.running_calls()
    if getattr(cli, "_approval_state", None) or getattr(cli, "_clarify_state", None):
        snap.status, snap.tone = "Waiting for you", "warn"
    elif running:
        current = running[-1]
        more = f" +{len(running) - 1}" if len(running) > 1 else ""
        snap.status, snap.tone = f"Executing {current.name}{more}", "tool"
        snap.machine = current.machine
    elif (note := _recovery_note(cli, ledger, now)):
        snap.status, snap.tone, snap.note = "Continuing", "warn", note
    else:
        snap.status, snap.tone = _STATE_LABEL.get(ledger.state, _STATE_LABEL["idle"])
        if snap.reasoning:
            snap.status, snap.tone = "Thinking", "think"
        elif ledger.state == "writing" and now - ledger.text_seen_at > 2.0:
            snap.status, snap.tone = "Finishing", "accent"
    if agent is not None:
        base = getattr(ledger, "tokens0", (0, 0))
        snap.tokens_in = max(0, int(getattr(agent, "session_input_tokens", 0) or 0) - base[0])
        snap.tokens_out = max(0, int(getattr(agent, "session_output_tokens", 0) or 0) - base[1])
        snap.context_pct = _context_pct(agent)
    snap.agents = _running_agents(cli)
    return snap


def _recovery_note(cli, ledger, now: float) -> str:
    """What the runtime is recovering from, in its own words (``↻ tool call cut off, not run — retrying: …``),
    while that is current. The runtime sends it as a diagnostic status line; the CLI keeps it on the ledger
    because the spinner text it also lands in is overwritten by the very next request."""
    for text in (ledger.recovery if now - ledger.recovery_at < 12 else "", str(getattr(cli, "_spinner_text", "") or "")):
        if "↻" in text:
            return text.split("↻", 1)[1].strip(" .…")
    return ""


def _context_pct(agent: Any) -> int | None:
    compressor = getattr(agent, "context_compressor", None)
    window = int(getattr(compressor, "context_length", 0) or 0)
    used = int(getattr(compressor, "last_prompt_tokens", 0) or 0)
    if not window or not used:
        return None
    return min(999, round(100 * used / window))


def _running_agents(cli) -> int:
    monitor = getattr(cli, "_subagent_monitor", None)
    return len(getattr(monitor, "entries", ()) or ())


def _elapsed(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    return f"{minutes}m {sec:02d}s" if minutes < 60 else f"{minutes // 60}h {minutes % 60:02d}m"


# Fields in the order they are dropped when the row is narrow (first dropped first).
_DROP_ORDER = ("machine", "agents", "tools", "tokens", "context", "model", "reasoning", "note")


def fields(snap: Snapshot) -> list[tuple[str, str, str]]:
    """(key, role, text) for each field that has a real value, left to right."""
    out: list[tuple[str, str, str]] = []
    out.append(("status", snap.tone, snap.status + (f" · {snap.elapsed}" if snap.elapsed else "")))
    if snap.note:
        out.append(("note", "warn", snap.note if len(snap.note) <= 64 else snap.note[:63] + "…"))
    if snap.reasoning:
        out.append(("reasoning", "think", "reasoning"))
    if snap.model:
        out.append(("model", "dim", snap.model))
    if snap.tokens_in or snap.tokens_out:
        pieces = [f"↑{compact(snap.tokens_in)}" if snap.tokens_in else "", f"↓{compact(snap.tokens_out)}" if snap.tokens_out else ""]
        out.append(("tokens", "dim", " ".join(p for p in pieces if p)))
    if snap.context_pct is not None:
        role = "error" if snap.context_pct >= 92 else "warn" if snap.context_pct >= 75 else "dim"
        out.append(("context", role, f"ctx {snap.context_pct}%"))
    if snap.tools:
        text = f"{snap.tools} tool{'s' if snap.tools != 1 else ''}"
        if snap.failures:
            text += f", {snap.failures} failed"
        out.append(("tools", "error" if snap.failures else "dim", text))
    if snap.agents:
        out.append(("agents", "tool", f"{snap.agents + 1} agents"))
    if snap.machine:
        out.append(("machine", "dim", snap.machine))
    return out


def fit(items: list[tuple[str, str, str]], width: int, lead: int, gap: int = 2) -> list[tuple[str, str, str]]:
    """Drop whole fields (never cut one mid-word) until the row fits *width* after a *lead* cells prefix."""
    items = list(items)

    def used() -> int:
        return lead + sum(len(text) for _k, _r, text in items) + gap * max(0, len(items) - 1)

    for key in _DROP_ORDER:
        if used() <= width:
            break
        items = [i for i in items if i[0] != key]
    if used() > width and items and items[0][0] == "status":
        key, role, text = items[0]
        room = max(1, width - lead - sum(len(t) + gap for _k, _r, t in items[1:]))
        items[0] = (key, role, text if len(text) <= room else text[:max(1, room - 1)] + "…")
    return items

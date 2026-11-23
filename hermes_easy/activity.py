"""The activity ledger: what the agent really did this turn, built only from runtime events.

Nothing here is simulated. A tool call enters the ledger when the runtime says it started and
leaves the active set when the runtime says it finished; duration, exit status and error text
come from the call's own result. The renderers (``layout`` for scrollback, ``status`` for the
live row) read the ledger, so a summary such as "Ran 4 shell commands · Read 6 files" can never
disagree with what executed.

Calls are grouped the way a reader thinks about them: consecutive tool calls with no assistant
text between them form one group. A group closes when the model writes text, starts reasoning,
or the turn ends.
"""

from __future__ import annotations

import json
import os
import platform
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── categories ───────────────────────────────────────────────────────────────────────────

_TEST_COMMAND = re.compile(
    r"\b(pytest|py\.test|vitest|jest|mocha|unittest|go test|cargo test|npm (?:run )?test|pnpm (?:run )?test|"
    r"yarn test|ctest|rspec|phpunit|dotnet test|tox|nox|run_tests\.sh|swift test|gradle test|mvn test)\b")
_CATEGORY_OF_TOOL = {
    "terminal": "shell", "process": "shell", "execute_code": "shell",
    "read_file": "read", "view_file": "read", "read_many_files": "read",
    "write_file": "edit", "patch": "edit", "edit_file": "edit", "apply_patch": "edit", "multi_edit": "edit",
    "search_files": "search", "grep": "search", "glob": "search", "find_files": "search",
    "web_search": "web", "web_extract": "web", "web_fetch": "web",
    "delegate_task": "agent", "subagent": "agent",
}


def category_of(name: str, args: dict | None = None) -> str:
    """shell / tests / read / edit / search / web / browser / agent / other."""
    name = (name or "").lower()
    if name.startswith("browser"):
        return "browser"
    category = _CATEGORY_OF_TOOL.get(name, "other")
    if category == "shell" and name == "terminal" and _TEST_COMMAND.search(str((args or {}).get("command") or "")):
        return "tests"
    return category


# ── machines ─────────────────────────────────────────────────────────────────────────────

_OS_NAMES = {"Linux": "Linux", "Darwin": "macOS", "Windows": "Windows"}


def machine_label(task_id: str | None = None) -> str:
    """Where this agent's commands execute, from the terminal backend that really serves it.

    ``local`` is the machine running Hermes (named by its OS); ``ssh`` is a remote server (its OS is
    not known, so none is claimed); containers and cloud sandboxes name their backend."""
    here = _OS_NAMES.get(platform.system(), platform.system() or "this machine")
    try:
        from tools.terminal_tool import _get_env_config, _qualify_task_key, _task_env_overrides
        config = dict(_get_env_config())
        overrides = _task_env_overrides.get(_qualify_task_key(task_id)) if task_id else None
        env_type = str((overrides or {}).get("env_type") or config.get("env_type") or "local")
        host = str((overrides or {}).get("ssh_host") or config.get("ssh_host") or "")
    except Exception:  # health: allow BLE001 -- a label must never break rendering
        env_type, host = "local", ""
    if env_type == "local":
        return here
    if env_type == "ssh":
        return f"remote server {host}".strip() if host else "remote server"
    if env_type == "docker":
        return f"Docker container on {here}"
    return f"{env_type} sandbox"


CURRENT: Ledger | None = None  # the process's one CLI ledger (set when it is created)


def agent_detail(row: dict) -> str:
    """``model · machine · `` for one subagent dock row: its own model and the backend its commands run on."""
    task_id = None
    try:
        from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
        with _active_subagents_lock:
            child = (_active_subagents.get(row.get("subagent_id")) or {}).get("agent")
        task_id = getattr(child, "task_id", None) or getattr(child, "session_id", None)
    except Exception:  # health: allow BLE001 -- a dock label must never break rendering
        pass
    return " · ".join(p for p in (str(row.get("model") or ""), machine_label(task_id)) if p) + " · "


# ── ledger ───────────────────────────────────────────────────────────────────────────────

_OUTPUT_KEEP = 6000  # characters of a call's output kept for expansion; the session keeps the rest


@dataclass
class Call:
    id: int
    name: str
    category: str
    label: str
    command: str = ""
    cwd: str = ""
    target: str = ""
    started: float = 0.0
    ended: float | None = None
    failed: bool = False
    neutral: bool = False  # a failure that is an answer (grep found nothing)
    exit_code: int | None = None
    error: str = ""
    output: str = ""
    agent: str = ""
    machine: str = ""
    shown: bool = False  # its one-row form was already printed (rows mode)

    @property
    def running(self) -> bool:
        return self.ended is None

    def elapsed(self, now: float | None = None) -> float:
        return max(0.0, (self.ended if self.ended is not None else (now or time.time())) - self.started)


@dataclass
class Group:
    calls: list[Call] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return any(c.running for c in self.calls)

    @property
    def failed(self) -> list[Call]:
        return [c for c in self.calls if c.failed and not c.neutral]


_VERBS = {
    # category: (singular noun, plural noun, verb for a finished group, verb while running)
    "shell": ("shell command", "shell commands", "Ran", "Running"),
    "read": ("file", "files", "Read", "Reading"),
    "edit": ("file", "files", "Edited", "Editing"),
}


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def summary_parts(group: Group, *, active: bool = False) -> list[tuple[str, str]]:
    """``[(category, text), ...]`` e.g. ``[("shell", "Ran 4 shell commands"), ("read", "Read 6 files")]``
    (present tense while *active*), in order of first use."""
    order: list[str] = []
    by_cat: dict[str, list[Call]] = {}
    for call in group.calls:
        if call.category not in by_cat:
            order.append(call.category)
        by_cat.setdefault(call.category, []).append(call)
    parts: list[tuple[str, str]] = []
    for cat in order:
        calls = by_cat[cat]
        n = len(calls)
        if cat in {"read", "edit"}:
            # Only a call that worked read or edited anything; a refused write is not an edit.
            one, many, past, present = _VERBS[cat]
            worked = {c.target or c.label for c in calls if not (c.failed and not c.neutral)}
            if worked or active:
                text = f"{present if active else past} {_plural(len(worked or {c.label for c in calls}), one, many)}"
            else:
                text = f"Failed to {'read' if cat == 'read' else 'edit'} {_plural(len({c.target or c.label for c in calls}), one, many)}"
        elif cat in _VERBS:
            one, many, past, present = _VERBS[cat]
            text = f"{present if active else past} {_plural(n, one, many)}"
        elif cat == "tests":
            text = ("Executing tests" if active else "Executed tests") + (f" ×{n}" if n > 1 else "")
        elif cat == "search":
            text = ("Searching" if active else "Searched") + " the repository" + (f" ×{n}" if n > 1 else "")
        elif cat == "web":
            text = ("Searching" if active else "Searched") + " the web" + (f" ×{n}" if n > 1 else "")
        elif cat == "browser":
            text = ("Using" if active else "Used") + " browser"
        elif cat == "agent":
            text = ("Running" if active else "Ran") + " " + _plural(n, "subagent", "subagents")
        else:
            names = sorted({c.name for c in calls})
            text = ("Using " if active else "Used ") + (names[0] if len(names) == 1 else _plural(n, "tool", "tools"))
        parts.append((cat, text))
    return parts


def summarize(group: Group, *, active: bool = False) -> str:
    """``Ran 4 shell commands · Read 6 files · Edited 3 files`` (present tense while *active*)."""
    return " · ".join(text for _cat, text in summary_parts(group, active=active))


class Ledger:
    """Per-CLI record of the running turn plus a bounded history of finished groups."""

    HISTORY = 40

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.reset_turn()
        self.groups: list[Group] = []
        self.next_id = 1
        self.recovery: str = ""
        self.recovery_at: float = 0.0
        self.session_id: str = ""
        self.tokens0: tuple[int, int] = (0, 0)

    # turn lifecycle ---------------------------------------------------------------------

    def reset_turn(self) -> None:
        with self._lock:
            self.turn_started: float | None = None
            self.reasoning_since: float | None = None
            self.reasoning_seen_at: float = 0.0
            self.text_seen_at: float = 0.0
            self.current: Group | None = None
            self.calls_this_turn = 0
            self.failures_this_turn = 0
            self.recovery = ""
            self.state = "idle"  # idle | thinking | tools | waiting | writing | continuing | done | failed

    def start_turn(self, now: float | None = None) -> None:
        with self._lock:
            self.reset_turn()
            self.turn_started = now or time.time()
            self.state = "thinking"

    def end_turn(self, failed: bool = False) -> Optional[Group]:
        with self._lock:
            closed = self.close_group()
            self.reasoning_since = None
            self.state = "failed" if failed else "done"
            return closed

    # events -----------------------------------------------------------------------------

    def note_reasoning(self, now: float | None = None) -> None:
        """A reasoning delta arrived from the provider (real provider-exposed thinking)."""
        now = now or time.time()
        with self._lock:
            if self.reasoning_since is None:
                self.reasoning_since = now
            self.reasoning_seen_at = now
            if self.state in {"idle", "thinking", "waiting", "continuing"}:
                self.state = "thinking"

    def note_text(self, now: float | None = None) -> None:
        """The model is writing its reply (assistant text deltas are arriving)."""
        with self._lock:
            self.text_seen_at = now or time.time()
            if not self.running_calls():
                self.state = "writing"

    def reasoning_active(self, now: float | None = None, window: float = 2.5) -> bool:
        return self.reasoning_since is not None and (now or time.time()) - self.reasoning_seen_at < window

    def note_recovery(self, text: str, now: float | None = None) -> None:
        with self._lock:
            self.recovery, self.recovery_at = text, now or time.time()
            self.state = "continuing"

    def tool_started(self, name: str, args: dict | None, *, agent: str = "", task_id: str | None = None,
                     label: str = "", now: float | None = None) -> Call:
        args = args if isinstance(args, dict) else {}
        with self._lock:
            if self.current is None:
                self.current = Group()
            call = Call(
                id=self.next_id, name=name, category=category_of(name, args),
                label=label or name, command=str(args.get("command") or ""),
                cwd=str(args.get("workdir") or args.get("cwd") or args.get("path") or "") if name == "terminal" else "",
                target=str(args.get("path") or args.get("file_path") or args.get("file") or ""),
                started=now or time.time(), agent=agent, machine=machine_label(task_id),
            )
            if name == "terminal" and not call.cwd:
                call.cwd = os.getenv("TERMINAL_CWD") or os.getcwd()
            self.next_id += 1
            self.current.calls.append(call)
            self.calls_this_turn += 1
            self.state = "tools"
            self.recovery = ""
            return call

    def tool_completed(self, name: str, result: Any, *, failed: bool, neutral: bool = False,
                       duration: float | None = None, now: float | None = None) -> Optional[Call]:
        """Close the oldest running call of *name* (events for one tool arrive in order)."""
        with self._lock:
            group = self.current
            call = next((c for c in (group.calls if group else []) if c.running and c.name == name), None)
            if call is None:
                return None
            call.ended = (call.started + duration) if duration is not None and duration >= 0 else (now or time.time())
            call.failed, call.neutral = bool(failed), bool(neutral)
            text, exit_code, error = _unpack(result)
            call.exit_code, call.error, call.output = exit_code, error, text[-_OUTPUT_KEEP:]
            if failed and not neutral:
                self.failures_this_turn += 1
            if not any(c.running for c in group.calls):
                self.state = "waiting"
            return call

    def claim_completed(self, name: str) -> Optional[Call]:
        """The oldest finished, not-yet-printed call of *name* (rows mode prints each exactly once)."""
        with self._lock:
            group = self.current
            call = next((c for c in (group.calls if group else [])
                         if not c.running and not c.shown and c.name == name), None)
            if call is not None:
                call.shown = True
            return call

    def close_group(self) -> Optional[Group]:
        """End the current group (assistant text, reasoning or the end of the turn); returns it."""
        with self._lock:
            group, self.current = self.current, None
            if group is None or not group.calls:
                return None
            for call in group.calls:  # a call the runtime never completed (interrupt, crash) did not finish
                if call.running:
                    call.ended, call.failed, call.error = time.time(), True, "Interrupted before it finished"
            self.groups.append(group)
            del self.groups[:-self.HISTORY]
            return group

    # queries ----------------------------------------------------------------------------

    def running_calls(self) -> list[Call]:
        with self._lock:
            return [c for c in (self.current.calls if self.current else []) if c.running]

    def last_group(self, back: int = 0) -> Optional[Group]:
        """The most recent finished group, or the running one when it has calls (``back`` steps further)."""
        with self._lock:
            groups = [*self.groups, *([self.current] if self.current and self.current.calls else [])]
            return groups[-1 - back] if len(groups) > back else None

    def live_summary(self) -> str:
        with self._lock:
            return summarize(self.current, active=True) if self.current and self.current.calls else ""

    def turn_elapsed(self, now: float | None = None) -> float:
        return max(0.0, (now or time.time()) - self.turn_started) if self.turn_started else 0.0


def _unpack(result: Any) -> tuple[str, int | None, str]:
    """(output text, exit code, error text) from a tool result of any shape."""
    if result is None:
        return "", None, ""
    text = result if isinstance(result, str) else str(result)
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return text, None, ""
    if not isinstance(data, dict):
        return text, None, ""
    exit_code = data.get("exit_code")
    exit_code = exit_code if isinstance(exit_code, int) else None
    error = data.get("error")
    error = error if isinstance(error, str) else ""
    for key in ("output", "content", "stdout", "result", "text", "message"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value, exit_code, error
    return (error or json.dumps(data, ensure_ascii=False, indent=2)), exit_code, error


# ── persistence (activity history survives resume / reconnect) ───────────────────────────

def _store_path(session_id: str) -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "activity" / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', session_id)}.json"


def save(ledger: Ledger, session_id: str) -> None:
    """Persist the finished groups (bounded, outputs already trimmed); best effort."""
    if not session_id:
        return
    try:
        path = _store_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [[c.__dict__ for c in g.calls] for g in ledger.groups[-ledger.HISTORY:]]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def load(ledger: Ledger, session_id: str) -> None:
    """Restore the history saved for *session_id* (a resumed session keeps its tool groups)."""
    try:
        raw = json.loads(_store_path(session_id).read_text(encoding="utf-8"))
        groups = [Group([Call(**c) for c in calls]) for calls in raw]
    except (OSError, ValueError, TypeError):
        return
    with ledger._lock:
        ledger.groups = groups[-ledger.HISTORY:]
        ledger.next_id = 1 + max((c.id for g in ledger.groups for c in g.calls), default=0)

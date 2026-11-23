"""UI settings: schema, defaults, validation and persistence (``$HERMES_HOME/ui.yaml``).

The file is plain YAML so it can be edited by hand; ``/ui-settings`` writes the same file.
Unknown keys are ignored and invalid values fall back to the default for that key, so a
typo never stops the CLI from starting.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class Field:
    """One setting. ``choices`` is a list of (value, label); an empty list means free HEX text."""
    section: str
    key: str
    label: str
    default: Any
    choices: tuple = ()
    help: str = ""

    @property
    def path(self) -> str:
        return f"{self.section}.{self.key}"

    @property
    def is_bool(self) -> bool:
        return isinstance(self.default, bool)

    @property
    def is_hex(self) -> bool:
        return not self.choices and not self.is_bool


PRESET_CHOICES = (
    ("graphite", "Graphite"), ("monochrome", "Monochrome"), ("nordic", "Nordic"),
    ("forest", "Forest"), ("amber", "Amber"), ("classic", "Classic Terminal"))

FIELDS: tuple[Field, ...] = (
    Field("theme", "preset", "Theme", "graphite", PRESET_CHOICES,
          "Colour preset. Accent and text overrides below apply on top of it."),
    Field("theme", "mode", "Mode", "auto", (("auto", "Auto"), ("dark", "Dark"), ("light", "Light")),
          "Auto follows the terminal background when it can be detected."),
    Field("theme", "accent", "Accent colour", "",
          help="#RRGGBB. Empty uses the preset accent. Needs a truecolor terminal."),
    Field("theme", "foreground", "Text colour", "",
          help="#RRGGBB for assistant text. Empty keeps the terminal's own foreground."),
    Field("theme", "background", "Background", "",
          help="#RRGGBB sets the terminal background while hermes-easy runs (restored on exit). "
               "Empty leaves the terminal alone."),
    Field("layout", "spacing", "Spacing", "comfortable",
          (("comfortable", "Comfortable"), ("compact", "Compact")),
          "Comfortable puts a blank line between blocks; compact only between turns."),
    Field("layout", "messages", "Message layout", "glyph",
          (("glyph", "Glyph"), ("labeled", "Labeled"), ("ruled", "Ruled")),
          "Glyph marks your messages with the prompt glyph. Labeled adds You / Hermes names. "
          "Ruled draws a short rule before each turn."),
    Field("layout", "tool_output", "Tool output", "compact",
          (("compact", "One line per call"), ("preview", "Line + output preview"),
           ("full", "Full output"), ("hidden", "Hidden")),
          "Ctrl+O prints the full output of the latest tool call whatever is chosen here."),
    Field("status", "line", "Status line", "standard",
          (("standard", "Standard"), ("minimal", "Minimal"), ("full", "Full (upstream)"), ("off", "Off")),
          "Standard: approvals, model and measured decode speed. Minimal: approvals and model. "
          "Full: every upstream counter."),
    Field("status", "key_hints", "Key hints", True, (),
          "Show the keys that work right now at the right of the status line."),
    Field("status", "turn_summary", "Turn summary", True, (),
          "A dim line after each turn: time taken and what the tools did."),
    Field("motion", "spinner", "Activity indicator", "dots",
          (("dots", "Dots"), ("line", "Line"), ("pulse", "Pulse"), ("static", "Static")),
          "Shown above the composer while the agent works."),
    Field("motion", "reduced_motion", "Reduced motion", False, (),
          "No animated frames anywhere; the indicator becomes a static mark."),
)

FIELD_BY_PATH = {f.path: f for f in FIELDS}

FILE_HEADER = """\
# hermes-easy UI settings. Edited by /ui-settings; safe to edit by hand.
# Unknown keys are ignored; an invalid value falls back to its default.
# Keys and allowed values: see README.md, section "UI settings".
"""


def defaults() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in FIELDS:
        out.setdefault(f.section, {})[f.key] = f.default
    return out


def validate_value(field: Field, value: Any) -> tuple[bool, Any]:
    """(ok, normalised value) for one field."""
    if field.is_bool:
        if isinstance(value, bool):
            return True, value
        if isinstance(value, str) and value.strip().lower() in {"true", "on", "yes", "1"}:
            return True, True
        if isinstance(value, str) and value.strip().lower() in {"false", "off", "no", "0"}:
            return True, False
        return False, field.default
    if field.is_hex:
        text = "" if value is None else str(value).strip()
        if text and not text.startswith("#"):
            text = "#" + text
        if text == "" or HEX_RE.match(text):
            return True, text.upper()
        return False, field.default
    allowed = {c[0] for c in field.choices}
    text = str(value).strip().lower() if value is not None else ""
    return (True, text) if text in allowed else (False, field.default)


def normalise(raw: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Merge *raw* (parsed YAML) over the defaults; returns (settings, problems)."""
    settings = defaults()
    problems: list[str] = []
    if not isinstance(raw, dict):
        if raw not in (None, ""):
            problems.append("ui.yaml is not a mapping; using defaults")
        return settings, problems
    for f in FIELDS:
        section = raw.get(f.section)
        if not isinstance(section, dict) or f.key not in section:
            continue
        ok, value = validate_value(f, section[f.key])
        if not ok:
            problems.append(f"{f.path}: {section[f.key]!r} is not valid; using {f.default!r}")
        settings[f.section][f.key] = value
    return settings, problems


def settings_path(home: Path | None = None) -> Path:
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    return Path(home) / "ui.yaml"


def load(home: Path | None = None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    path = settings_path(home)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return defaults(), []
    except OSError as exc:
        return defaults(), [f"could not read {path}: {exc}"]
    try:
        import hermes_yaml
        raw = hermes_yaml.safe_load(text)
    except Exception as exc:  # health: allow BLE001 -- a broken hand edit must not stop startup
        return defaults(), [f"could not parse {path}: {exc}"]
    return normalise(raw)


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return f'"{value}"'


def dump(settings: dict[str, dict[str, Any]]) -> str:
    """The ui.yaml text: one commented line per key, grouped by section."""
    lines = [FILE_HEADER]
    current = None
    for f in FIELDS:
        if f.section != current:
            current = f.section
            lines.append(f"{f.section}:")
        value = settings.get(f.section, {}).get(f.key, f.default)
        if f.choices:
            hint = " | ".join(c[0] for c in f.choices)
        elif f.is_bool:
            hint = "true | false"
        else:
            hint = '"#RRGGBB" or ""'
        lines.append(f"  {f.key}: {_yaml_scalar(value)}  # {hint}")
    return "\n".join(lines) + "\n"


_SAVE_LOCK = threading.Lock()


def save(settings: dict[str, dict[str, Any]], home: Path | None = None) -> Path:
    """Atomically write *settings* to ui.yaml (tmp file + rename in the same directory)."""
    path = settings_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = dump(settings)
    with _SAVE_LOCK:
        fd, tmp = tempfile.mkstemp(prefix=".ui.yaml.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return path


# ── process-wide current settings ────────────────────────────────────────────────────────

_current: dict[str, dict[str, Any]] | None = None
_problems: list[str] = []


def current() -> dict[str, dict[str, Any]]:
    global _current, _problems
    if _current is None:
        _current, _problems = load()
    return _current


def startup_problems() -> list[str]:
    current()
    return list(_problems)


def get(path: str) -> Any:
    section, key = path.split(".", 1)
    return current().get(section, {}).get(key, FIELD_BY_PATH[path].default)


def replace(settings: dict[str, dict[str, Any]]) -> None:
    """Swap the in-memory settings (used for live preview; does not write)."""
    global _current
    _current = {s: dict(v) for s, v in settings.items()}


def reload() -> dict[str, dict[str, Any]]:
    global _current, _problems
    _current, _problems = load()
    return _current


def truecolor_supported() -> bool:
    """COLORTERM is how terminals advertise 24-bit colour; a few set only TERM."""
    colorterm = os.environ.get("COLORTERM", "").lower()
    if colorterm in {"truecolor", "24bit"}:
        return True
    term = os.environ.get("TERM", "").lower()
    return term.endswith("-direct") or os.environ.get("TERM_PROGRAM", "") in {
        "iTerm.app", "WezTerm", "vscode", "ghostty"}

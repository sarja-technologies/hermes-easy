"""Theme presets and the skin hermes-easy installs into upstream's skin engine.

A preset is a small set of roles per polarity. Everything upstream colours (prompt_toolkit
styles, Rich markup, raw ANSI rows) reads the active skin, so translating roles into one
skin keeps every surface consistent without touching each renderer.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from hermes_easy import settings as ui

SKIN_NAME = "hermes-easy"

# Roles: accent (glyphs, the one coloured thing), strong (your messages), text (assistant text;
# None = terminal foreground), dim (secondary), faint (rules), surface/selection (menus),
# ok/warn/error. No preset uses a hue in the violet range.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "graphite": {
        "dark": dict(accent="#79BFC0", strong="#E6E6E6", text=None, dim="#8A8F98", faint="#3A3D42",
                     surface="#232529", selection="#33373D"),
        "light": dict(accent="#3B5B85", strong="#1F1F1F", text=None, dim="#6A6F78", faint="#D5D7DB",
                      surface="#EEEFF1", selection="#DCE1E8")},
    "monochrome": {
        "dark": dict(accent="#E8E8E8", strong="#FFFFFF", text=None, dim="#8C8C8C", faint="#3A3A3A",
                     surface="#202020", selection="#363636"),
        "light": dict(accent="#111111", strong="#000000", text=None, dim="#707070", faint="#D4D4D4",
                      surface="#EFEFEF", selection="#DADADA")},
    "nordic": {
        "dark": dict(accent="#88C0D0", strong="#ECEFF4", text="#D8DEE9", dim="#8090A8", faint="#3B4252",
                     surface="#2E3440", selection="#3B4252"),
        "light": dict(accent="#2D6E80", strong="#2E3440", text="#3B4252", dim="#5E6B80", faint="#D8DEE9",
                      surface="#ECEFF4", selection="#D8DEE9")},
    "forest": {
        "dark": dict(accent="#8DBF7A", strong="#E3EBDD", text=None, dim="#85947E", faint="#34402F",
                     surface="#1F271C", selection="#2E3A29"),
        "light": dict(accent="#3C7A2E", strong="#1E2A1A", text=None, dim="#5F6E58", faint="#D3DDCC",
                      surface="#EEF3EA", selection="#DCE7D4")},
    "amber": {
        "dark": dict(accent="#E3A64F", strong="#F1E7D3", text=None, dim="#9A8C73", faint="#463A2B",
                     surface="#2A2118", selection="#3D3122"),
        "light": dict(accent="#985A00", strong="#2B2115", text=None, dim="#786A52", faint="#E2D8C6",
                      surface="#F6F0E4", selection="#EADFCB")},
    "classic": {
        "dark": dict(accent="#4FD66A", strong="#D2F5CF", text="#A8E6A3", dim="#7C9A7A", faint="#2F3B2F",
                     surface="#121812", selection="#223022"),
        "light": dict(accent="#0F7A2A", strong="#0D2611", text="#1D4D24", dim="#5E765F", faint="#CFDCCF",
                      surface="#EEF5EE", selection="#D6E6D6")},
}

# ok/warn/error carry meaning (finished / recoverable / failed). ``tool`` marks commands and tests that
# run, ``think`` marks provider-exposed reasoning: two muted hues away from every status colour and
# from violet (blue and warm grey).
_STATUS = {
    "dark": dict(ok="#7FBF7F", warn="#D9A441", error="#E06C6C", tool="#7DA7D9", think="#B9AC9B"),
    "light": dict(ok="#2E7D32", warn="#A86300", error="#C62828", tool="#2F5F9E", think="#7A6C58"),
}


def _detect_light() -> bool:
    try:
        from cli import _detect_light_mode
        return bool(_detect_light_mode())
    except Exception:  # health: allow BLE001 -- detection is a best-effort probe
        return False


def polarity(settings: dict | None = None) -> str:
    mode = (settings or ui.current())["theme"]["mode"]
    if mode in {"dark", "light"}:
        return mode
    return "light" if _detect_light() else "dark"


def palette(settings: dict | None = None) -> dict[str, Any]:
    """Resolved roles for *settings* (default: the current ones), overrides applied."""
    settings = settings or ui.current()
    pol = polarity(settings)
    roles = dict(PRESETS.get(settings["theme"]["preset"], PRESETS["graphite"])[pol])
    roles.update(_STATUS[pol])
    if settings["theme"]["accent"]:
        roles["accent"] = settings["theme"]["accent"]
    if settings["theme"]["foreground"]:
        roles["text"] = settings["theme"]["foreground"]
    roles["polarity"] = pol
    return roles


def skin_data(settings: dict | None = None) -> dict[str, Any]:
    """A skin-engine definition (pure data, like a ``skins/*.yaml``) for the current roles."""
    p = palette(settings)
    text = p["text"] or p["strong"]
    colors = {
        "banner_border": p["faint"], "banner_title": p["strong"], "banner_accent": p["accent"],
        "banner_dim": p["dim"], "banner_text": text, "ui_accent": p["accent"], "ui_label": p["accent"],
        "ui_ok": p["ok"], "ui_error": p["error"], "ui_warn": p["warn"], "prompt": p["accent"],
        "input_rule": p["faint"], "response_border": p["faint"],
        "status_bar_bg": p["surface"], "status_bar_text": p["dim"], "status_bar_strong": p["strong"],
        "status_bar_dim": p["dim"], "status_bar_good": p["ok"], "status_bar_warn": p["warn"],
        "status_bar_bad": p["warn"], "status_bar_critical": p["error"], "session_label": p["dim"],
        "session_border": p["faint"], "completion_menu_bg": p["surface"],
        "completion_menu_current_bg": p["selection"], "completion_menu_meta_bg": p["surface"],
        "completion_menu_meta_current_bg": p["selection"], "selection_bg": p["selection"],
        "shell_dollar": p["accent"], "voice_status_bg": p["surface"],
    }
    return {
        "name": SKIN_NAME,
        "description": "hermes-easy theme (generated from ui.yaml)",
        "colors": colors,
        "spinner": {"waiting_faces": [""], "thinking_faces": [""], "thinking_verbs": ["Thinking"],
                    "wings": []},
        "branding": {
            "agent_name": "Hermes", "welcome": "", "goodbye": "", "response_label": " · Hermes ",
            "prompt_symbol": "›", "help_header": "Commands"},
        "tool_prefix": "│",
    }


def install(settings: dict | None = None) -> None:
    """Register the generated skin with the skin engine and make it active."""
    from hermes_cli import skin_engine
    skin_engine._BUILTIN_SKINS[SKIN_NAME] = skin_data(settings)
    skin_engine.set_active_skin(SKIN_NAME)
    apply_terminal_colors(settings)


# ── terminal default colours (OSC 10/11) ─────────────────────────────────────────────────

_osc_applied = False


def _osc(seq: str) -> None:
    if not sys.stdout.isatty():
        return
    try:
        # tmux needs the sequence wrapped to reach the outer terminal.
        if os.environ.get("TMUX"):
            seq = "\x1bPtmux;" + seq.replace("\x1b", "\x1b\x1b") + "\x1b\\"
        sys.__stdout__.write(seq)
        sys.__stdout__.flush()
    except (OSError, ValueError):
        pass


def apply_terminal_colors(settings: dict | None = None) -> None:
    """Set (or restore) the terminal's default background from ``theme.background``."""
    global _osc_applied
    settings = settings or ui.current()
    bg = settings["theme"]["background"]
    if bg:
        _osc(f"\x1b]11;{bg}\x1b\\")
        _osc_applied = True
    elif _osc_applied:
        restore_terminal_colors()


def restore_terminal_colors() -> None:
    global _osc_applied
    if _osc_applied:
        _osc("\x1b]111\x1b\\")
        _osc_applied = False


def hex_fg(hex_color: str | None, bold: bool = False, italic: bool = False) -> str:
    """Raw SGR for a foreground colour (24-bit; the terminal or tmux downsamples if needed)."""
    codes = []
    if os.environ.get("NO_COLOR") is not None:
        hex_color = None
    if bold:
        codes.append("1")
    if italic:
        codes.append("3")
    if hex_color:
        try:
            r, g, b = (int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
            codes.append(f"38;2;{r};{g};{b}" if ui.truecolor_supported() else f"38;5;{xterm256(r, g, b)}")
        except ValueError:
            pass
    return f"\x1b[{';'.join(codes)}m" if codes else ""


def xterm256(r: int, g: int, b: int) -> int:
    """Nearest xterm-256 index (colour cube or grey ramp) for terminals without truecolor."""
    def cube(v: int) -> int:
        return 0 if v < 48 else 1 if v < 115 else (v - 35) // 40

    ci = (cube(r), cube(g), cube(b))
    levels = (0, 95, 135, 175, 215, 255)
    cube_rgb = tuple(levels[c] for c in ci)
    grey = round((r + g + b) / 3)
    gi = 0 if grey < 8 else 23 if grey > 238 else (grey - 8) // 10
    grey_rgb = (8 + gi * 10,) * 3

    def dist(c):
        return sum((a - b2) ** 2 for a, b2 in zip(c, (r, g, b)))

    if dist(grey_rgb) < dist(cube_rgb):
        return 232 + gi
    return 16 + 36 * ci[0] + 6 * ci[1] + ci[2]


RESET = "\x1b[0m"

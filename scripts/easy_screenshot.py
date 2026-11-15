#!/usr/bin/env python3
"""Render a tmux pane (``tmux capture-pane -p -e``) to HTML and PNG for hermes-easy's README.

    tmux capture-pane -t SESSION -p -e > pane.ansi
    python3 scripts/easy_screenshot.py pane.ansi out.png [--bg '#17181B'] [--fg '#D4D4D8']

Only the SGR subset tmux emits is handled (bold/dim/italic/underline/reverse, 16/256/24-bit
colours). The PNG is taken with headless Chrome; the HTML is kept next to it.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ANSI16 = ["#1D1F21", "#CC6666", "#8ABE73", "#E0C070", "#81A2BE", "#B4A0C8", "#8ABEB7", "#C5C8C6",
          "#666666", "#D54E53", "#B9CA4A", "#E7C547", "#7AA6DA", "#C9B3DE", "#70C0B1", "#EAEAEA"]


def xterm(n: int) -> str:
    if n < 16:
        return ANSI16[n]
    if n < 232:
        n -= 16
        levels = (0, 95, 135, 175, 215, 255)
        return "#%02X%02X%02X" % (levels[n // 36], levels[(n // 6) % 6], levels[n % 6])
    v = 8 + (n - 232) * 10
    return "#%02X%02X%02X" % (v, v, v)


SGR = re.compile(r"\x1b\[([0-9;:]*)m")


def convert(text: str, fg: str, bg: str) -> str:
    state = {"fg": None, "bg": None, "b": False, "d": False, "i": False, "u": False, "r": False}
    out: list[str] = []

    def span(chunk: str) -> str:
        if not chunk:
            return ""
        f, b = state["fg"], state["bg"]
        if state["r"]:
            f, b = (b or bg), (f or fg)
        css = []
        if f:
            css.append(f"color:{f}")
        if b:
            css.append(f"background:{b}")
        if state["b"]:
            css.append("font-weight:700")
        if state["d"]:
            css.append("opacity:.62")
        if state["i"]:
            css.append("font-style:italic")
        if state["u"]:
            css.append("text-decoration:underline")
        body = html.escape(chunk)
        return f'<span style="{";".join(css)}">{body}</span>' if css else body

    pos = 0
    for m in SGR.finditer(text):
        out.append(span(text[pos:m.start()]))
        pos = m.end()
        params = [p for p in re.split("[;:]", m.group(1))] or ["0"]
        nums = [int(p) if p.isdigit() else 0 for p in params] or [0]
        i = 0
        while i < len(nums):
            n = nums[i]
            if n == 0:
                state.update(fg=None, bg=None, b=False, d=False, i=False, u=False, r=False)
            elif n == 1:
                state["b"] = True
            elif n == 2:
                state["d"] = True
            elif n == 3:
                state["i"] = True
            elif n == 4:
                state["u"] = True
            elif n == 7:
                state["r"] = True
            elif n == 22:
                state["b"] = state["d"] = False
            elif n == 23:
                state["i"] = False
            elif n == 24:
                state["u"] = False
            elif n == 27:
                state["r"] = False
            elif n in (38, 48) and i + 1 < len(nums):
                key = "fg" if n == 38 else "bg"
                if nums[i + 1] == 2 and i + 4 < len(nums):
                    state[key] = "#%02X%02X%02X" % tuple(nums[i + 2:i + 5])
                    i += 4
                elif nums[i + 1] == 5 and i + 2 < len(nums):
                    state[key] = xterm(nums[i + 2])
                    i += 2
            elif n == 39:
                state["fg"] = None
            elif n == 49:
                state["bg"] = None
            elif 30 <= n <= 37:
                state["fg"] = ANSI16[n - 30]
            elif 90 <= n <= 97:
                state["fg"] = ANSI16[n - 90 + 8]
            elif 40 <= n <= 47:
                state["bg"] = ANSI16[n - 40]
            elif 100 <= n <= 107:
                state["bg"] = ANSI16[n - 100 + 8]
            i += 1
    out.append(span(text[pos:]))
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ansi")
    ap.add_argument("png")
    ap.add_argument("--bg", default="#17181B")
    ap.add_argument("--fg", default="#D4D4D8")
    ap.add_argument("--cols", type=int, default=0)
    args = ap.parse_args()
    text = Path(args.ansi).read_text(encoding="utf-8", errors="replace").rstrip("\n")
    lines = text.split("\n")
    cols = args.cols or max(len(SGR.sub("", ln)) for ln in lines)
    body = convert(text, args.fg, args.bg)
    char_w, line_h, pad = 8.4, 19, 18
    width = int(cols * char_w + pad * 2)
    height = int(len(lines) * line_h + pad * 2)
    page = f"""<!doctype html><meta charset="utf-8"><style>
html,body{{margin:0;background:{args.bg}}}
pre{{margin:0;padding:{pad}px;color:{args.fg};font:14px/{line_h}px 'DejaVu Sans Mono','Menlo',monospace;
white-space:pre}}</style><pre>{body}</pre>"""
    html_path = Path(args.png).with_suffix(".html")
    html_path.write_text(page, encoding="utf-8")
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        print(f"wrote {html_path}; no Chrome found for the PNG", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--user-data-dir={profile}", f"--window-size={width},{height}",
                        f"--screenshot={Path(args.png).resolve()}", html_path.resolve().as_uri()],
                       check=True, capture_output=True, timeout=60)
    print(args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

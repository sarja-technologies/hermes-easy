"""Width-aware formatting for the classic CLI's terminal scrollback."""
from __future__ import annotations

import re
import unicodedata

from prompt_toolkit.utils import get_cwidth

# ANSI controls are passed through unchanged and do not consume columns.
_ANSI = re.compile(r"\001[^\002]*\002|\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def strip_controls(text: str) -> str:
    return _CONTROL.sub("", _ANSI.sub("", text))


def _clusters(text):
    """Keep combining marks, emoji modifiers and joiner sequences with their base."""
    cluster = ""
    regional_count = 0
    for ch in text:
        code = ord(ch)
        regional = 0x1F1E6 <= code <= 0x1F1FF
        attached = (bool(cluster) and (unicodedata.combining(ch) or ch in ("\ufe0f", "\ufe0e", "\u200d")
                    or cluster.endswith("\u200d") or 0x1F3FB <= code <= 0x1F3FF
                    or (regional and regional_count % 2 == 1)))
        if cluster and not attached:
            yield cluster
            cluster = ""
        cluster += ch
        regional_count = regional_count + 1 if regional else 0
    if cluster:
        yield cluster


def centre(text: str, columns: int, maximum: int = 100, *, hanging: int = 0, prose: bool = False) -> str:
    """Indent and wrap terminal output by display cells, retaining ANSI and code whitespace.

    Each physical line gets its own gutter; terminal scrollback remains selectable and searchable.
    """
    columns = max(1, int(columns))
    width = min(maximum, columns)
    gutter = " " * max(0, (columns - width) // 2)
    out = [gutter]
    used = 0
    pos = 0

    def wrap():
        nonlocal used
        out.append("\n" + gutter + " " * hanging)
        used = hanging

    def characters(segment):
        nonlocal used
        for ch in _clusters(segment):
            if ch == "\n":
                out.append("\n" + gutter)
                used = 0
                continue
            if ch == "\r":
                out.append(ch)
                used = 0
                continue
            # Tabs must stay inside the content region, including in code blocks.
            if ch == "\t":
                count = 8 - used % 8
                for _ in range(count):
                    if used >= width:
                        wrap()
                    out.append(" ")
                    used += 1
                continue
            cells = max(0, get_cwidth(ch))
            if used > hanging and used + cells > width:
                wrap()
            out.append(ch)
            used += cells

    def plain(segment):
        if not prose:
            characters(segment)
            return
        for word in re.findall(r"[^\s]+|\s+", segment):
            cells = get_cwidth(word)
            if not word.isspace() and hanging < used and cells <= width - hanging and used + cells > width:
                wrap()
            characters(word)

    for match in _ANSI.finditer(text):
        plain(text[pos:match.start()])
        out.append(match.group())
        pos = match.end()
    plain(text[pos:])
    return "".join(out)

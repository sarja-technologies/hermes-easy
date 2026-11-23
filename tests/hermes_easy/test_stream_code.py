"""Regression coverage for streaming fenced Markdown in Hermes Easy."""

import re
from types import SimpleNamespace

from hermes_easy.layout import stream_code_line


def visible(value):
    return re.sub(r"\x1b\[[\d;]+m", "", value)


def test_long_earlier_fence_keeps_short_and_mismatched_fences_inside_code(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    cli = SimpleNamespace()
    assert stream_code_line(cli, "````python title=sample") == ""
    for text in ("```", "~~~", "\treturn 3 * 4  # literal"):
        rendered = stream_code_line(cli, text)
        assert visible(rendered) == f"  {text}"
    assert len(set(re.findall(r"\x1b\[[\d;]+m", rendered))) >= 3
    assert stream_code_line(cli, "`````  ") == ""
    assert stream_code_line(cli, "**Markdown**") is None


def test_unknown_language_keeps_tabs_and_trailing_spaces():
    cli = SimpleNamespace()
    assert stream_code_line(cli, "~~~custom-language") == ""
    assert visible(stream_code_line(cli, "\t*x*  ")) == "  \t*x*  "
    assert visible(stream_code_line(cli, "    ")) == "      "
    assert stream_code_line(cli, "~~~") == ""

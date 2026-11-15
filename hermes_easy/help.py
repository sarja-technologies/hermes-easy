"""``/help`` for hermes-easy: one aligned row per command, aliases inline, no wrapping."""

from __future__ import annotations

import shutil

from hermes_easy import layout
from hermes_easy.theme import RESET, hex_fg, palette

KEYS = (
    ("enter", "send"), ("ctrl+j / alt+enter", "new line"), ("tab", "complete"),
    ("ctrl+c", "interrupt / clear / exit"), ("ctrl+o", "full output of the last tool call"),
    ("ctrl+p", "command palette"), ("ctrl+g", "edit the draft in $EDITOR"), ("@", "attach a file"),
    ("!", "run a shell command"),
)


def _fit(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def render(cli, query: str) -> list[str]:
    from agent.i18n import t
    from hermes_cli.cli_info_mixin import _help_section_title
    from hermes_cli.commands import COMMAND_REGISTRY, HELP_SESSION_SUBGROUPS

    p = palette()
    accent, dim, strong = hex_fg(p["accent"]), hex_fg(p["dim"]), hex_fg(p["strong"], bold=True)
    width = shutil.get_terminal_size().columns
    query = (query or "").strip().lower()
    sub_of = {n: sub for sub, names in HELP_SESSION_SUBGROUPS.items() for n in names}
    sections: dict[str, list[tuple[str, str]]] = {}
    for cmd in COMMAND_REGISTRY:
        if cmd.gateway_only or not cli._command_available(f"/{cmd.name}"):
            continue
        names = ", ".join(f"/{n}" for n in (cmd.name, *cmd.aliases))
        desc = cmd.describe() + (f"  {cmd.args_hint}" if cmd.args_hint else "")
        if query and query not in names.lower() and query not in desc.lower():
            continue
        section = sub_of.get(cmd.name, "Session") if cmd.category == "Session" else cmd.category
        sections.setdefault(section, []).append((names, desc))
    rows: list[str] = [""]
    name_w = min(28, max((len(n) for rs in sections.values() for n, _ in rs), default=10) + 2)
    for section in sections:
        rows += [f"{strong}{_help_section_title(section)}{RESET}"]
        for names, desc in sections[section]:
            shown = _fit(names, name_w - 1).ljust(name_w)
            rows.append(f"  {accent}{shown}{RESET}{dim}{_fit(desc, max(10, width - name_w - 3))}{RESET}")
        rows.append("")
    if not query:
        rows.append(f"{strong}Keys{RESET}")
        for key, what in KEYS:
            rows.append(f"  {accent}{key.ljust(name_w)}{RESET}{dim}{what}{RESET}")
        from cli import _session_skill_commands
        skills = _session_skill_commands(cli)
        if skills:
            rows += ["", f"{dim}{len(skills)} skill commands: /help skills lists them. "
                         f"/help <text> filters this list.{RESET}"]
    elif len(rows) == 1:
        rows.append(f"{dim}{t('cli.help.filtered_by', query=query)}: no commands match.{RESET}")
    layout.note_external_output()
    return rows

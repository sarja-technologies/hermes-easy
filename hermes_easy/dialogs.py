"""Compact, keyboard-driven views of Hermes' existing modal state."""

from hermes_easy.composer import content_width, safe_display, terminal_rows

_DIALOG_RULE_WIDTH = 36


def choice_rows(cli, state, title, details, labels, footer, *, limit=7, numbered=True):
    width = content_width(cli._get_tui_terminal_width())
    term_rows = terminal_rows(cli)
    selected = state.get("selected", 0)
    visible = min(limit, max(1, term_rows - 9), len(labels))
    start = min(max(0, selected - visible // 2), len(labels) - visible)
    lines = [("class:input-rule", "┄" * min(_DIALOG_RULE_WIDTH, width)), ("class:easy-title", "  " + title)]
    detail_budget = min(6, max(0, term_rows - visible - 6))
    lines.extend(("class:easy-dim", "  " + line) for line in details[:detail_budget] if line)
    if term_rows > 12:
        lines.append(("", ""))
    for index in range(start, start + visible):
        active = index == selected
        marker = "  › " if active else "    "
        number = f"{index + 1}. " if numbered else ""
        lines.append(("class:easy-selected" if active else "", marker + number + labels[index]))
    if len(labels) > visible and term_rows > 12:
        lines.append(("class:easy-dim", f"  {start + 1}–{start + visible} of {len(labels)}"))
    if footer and term_rows > 8:
        lines.append(("class:easy-dim", "  " + footer))
    return [
        (style, cli._trim_status_bar_text(safe_display(text.replace("\n", " ")), width)
         + ("\n" if index < len(lines) - 1 else ""))
        for index, (style, text) in enumerate(lines)
    ]


def approval(cli, state):
    from cli import _wrap_panel_text_keep_ws
    from hermes_cli.cli_tui_mixin import _approval_choice_label
    width = content_width(cli._get_tui_terminal_width())
    command = state["command"]
    command_rows = _wrap_panel_text_keep_ws(command, max(8, width - 4))
    if len(command_rows) > 3 and not state.get("show_full"):
        command_rows = command_rows[:2] + ["…"]
    details = command_rows + [state["description"] or ""]
    footer = "Enter to confirm · Esc to deny"
    return choice_rows(cli, state, "Tool permission", details,
                       [_approval_choice_label(c) for c in state["choices"]], footer)


def confirmation(cli, state):
    labels = [label for _, label, _ in state.get("choices", [])]
    return choice_rows(cli, state, state.get("title") or "Confirm",
                       (state.get("detail") or "").splitlines(), labels,
                       "Enter to confirm · Esc to cancel")


def clarify(cli, state):
    from cli import _wrap_panel_text

    questions = state.get("questions", [])
    active = state.get("active", 0)
    title = f"Question {active + 1} of {len(questions)}" if len(questions) > 1 else "Question from Hermes"
    width = content_width(cli._get_tui_terminal_width())
    details = _wrap_panel_text(state.get("question") or "", max(8, width - 4))
    labels = list(state.get("choices") or [])
    footer = "Type your answer below · Esc to cancel"
    if not cli._clarify_freetext:
        footer = "Enter to answer · Esc to cancel"
        if len(questions) > 1:
            footer = "Tab to switch questions · " + footer
        labels.append("Other answer")
        if state.get("multi_select"):
            checked = state.get("selected_indices") or set()
            labels = [f"[{'x' if i in checked else ' '}] {label}" for i, label in enumerate(labels)]
            footer = "Space to select · " + footer
    return choice_rows(cli, state, title, details, labels, footer, limit=6)

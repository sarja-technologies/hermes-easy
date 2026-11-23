"""Syntax colours for complete code lines in terminal scrollback."""

from pygments.lexers import get_lexer_by_name
from pygments.token import Comment, Keyword, Literal, Name, Operator
from pygments.util import ClassNotFound

from hermes_easy.theme import RESET, hex_fg, palette

_ROLES = (
    (Comment, "dim"),
    (Keyword, "accent"),
    (Literal.String, "ok"),
    (Literal.Number, "warn"),
    (Name.Function, "accent"),
    (Operator, "accent"),
)


def lexer_for(language):
    if language:
        try:
            return get_lexer_by_name(language)
        except ClassNotFound:
            return None
    return None


def highlight(line, lexer):
    colors = palette()
    plain = hex_fg(colors["strong"])
    if not lexer or len(line) > 4096:
        return plain + line + RESET

    tokens = list(lexer.get_tokens_unprocessed(line))
    # Some language lexers normalise input. Leave the user's copyable code intact.
    if "".join(value for _, _, value in tokens) != line:
        return plain + line + RESET
    parts = []
    previous = None
    for _, token, value in tokens:
        role = next((role for group, role in _ROLES if token in group), "strong")
        if role != previous:
            parts.append(hex_fg(colors[role]))
            previous = role
        parts.append(value)
    return "".join(parts) + RESET

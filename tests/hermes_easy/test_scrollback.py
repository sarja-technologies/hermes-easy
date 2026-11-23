from hermes_easy.scrollback import centre
from prompt_toolkit.utils import get_cwidth
import re


def test_centres_ansi_and_wraps_unicode_without_losing_text():
    text = '\x1b[31m' + ('界' * 55) + 'é\x1b[0m'
    lines = centre(text, 160).splitlines()
    plain = [re.sub(r'\x1b\[[0-9;]*m', '', line) for line in lines]
    assert all(line.startswith(' ' * 30) for line in plain)
    assert all(get_cwidth(line) <= 130 for line in plain)
    assert ''.join(line[30:] for line in plain) == '界' * 55 + 'é'


def test_narrow_code_and_tabs_are_contained():
    assert centre('  foo\tbar', 80) == '  foo   bar'
    assert centre('abc\ndef', 160) == ' ' * 30 + 'abc\n' + ' ' * 30 + 'def'


def test_emoji_joiner_sequence_is_not_split():
    family = '👨‍👩‍👧‍👦'
    text = centre('a' * 99 + family, 100)
    assert text == 'a' * 99 + '\n' + family

"""hermes-easy: Hermes Agent with a restrained terminal UI.

The agent, tools, providers and sessions are upstream Hermes Agent (Nous Research, MIT).
This package only owns the presentation layer: the identity line, transcript rows, the
composer chrome, the theme presets and ``/ui-settings``.
"""

__version__ = "0.1.0"

ENV_FLAG = "HERMES_EASY_UI"


def active() -> bool:
    """True when the hermes-easy presentation layer drives this process.

    Set by the ``hermes-easy`` entry point and inherited by child processes. Upstream code
    paths check this before handing a row to :mod:`hermes_easy.layout`, so the stock
    ``hermes`` behaviour (and upstream's tests) stay untouched when it is unset.
    """
    import os
    return os.environ.get(ENV_FLAG) == "1"

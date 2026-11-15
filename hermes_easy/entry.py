"""``hermes-easy`` console entry point.

Isolates hermes-easy from an upstream Hermes install on the same machine, turns on the
hermes-easy presentation layer, then runs upstream's own ``hermes_cli.main``. Every upstream
subcommand and flag works unchanged (``hermes-easy model``, ``hermes-easy -c``, ...), plus
one of our own: ``hermes-easy import-upstream``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_HOME = "~/.hermes-easy"
UPSTREAM_HOME = "~/.hermes"
# Messaging/bot credentials are never imported: two gateways polling one bot token fight.
_EXTRA_PROVIDER_KEYS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL")
_IMPORTED_CONFIG_SECTIONS = ("model", "providers", "custom_providers", "fallback_model", "model_aliases")


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.expanduser().resolve() == b.expanduser().resolve()
    except OSError:
        return False


def resolve_home(environ: dict | None = None) -> Path:
    """hermes-easy's home: ``$HERMES_HOME`` when set, else ``~/.hermes-easy``.

    Refuses upstream's home so a stray ``HERMES_HOME=~/.hermes`` in a shell profile can never
    make hermes-easy write into the upstream install's state.
    """
    environ = os.environ if environ is None else environ
    raw = (environ.get("HERMES_HOME") or "").strip()
    home = Path(os.path.expanduser(os.path.expandvars(raw or DEFAULT_HOME)))
    if _same_path(home, Path(UPSTREAM_HOME)):
        raise SystemExit(
            "hermes-easy: HERMES_HOME points at the upstream Hermes home (~/.hermes).\n"
            "hermes-easy keeps its own state in ~/.hermes-easy so the two never share files.\n"
            "Unset HERMES_HOME, or run `hermes-easy import-upstream` to copy your model and "
            "provider settings across.")
    return home


def prepare_environment(environ: dict | None = None) -> Path:
    environ = os.environ if environ is None else environ
    home = resolve_home(environ)
    environ["HERMES_HOME"] = str(home)
    # Anything that derives the platform default home (profiles root, PM partials) lands in
    # ~/.hermes-easy too instead of ~/.hermes.
    environ.setdefault("HERMES_DATA_DIR_SUFFIX", "-easy")
    environ["HERMES_EASY_UI"] = "1"
    # prompt_toolkit picks its colour depth from TERM, so on a truecolor terminal the scrollback
    # rows were quantised to 256 colours (muted accents turn grey). COLORTERM is the real signal.
    if environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}:
        environ.setdefault("PROMPT_TOOLKIT_COLOR_DEPTH", "DEPTH_24_BIT")
    shim = publish_hermes_shim(home)
    if shim is not None:
        # Children that look for a `hermes` (the agent's terminal, kanban workers, profile
        # wrappers) get this install rather than an upstream `hermes` that may be on PATH.
        environ["HERMES_BIN"] = str(shim)
        environ["HERMES_EASY_BIN_DIR"] = str(shim.parent)
    return home


def _own_executable() -> str | None:
    import shutil
    argv0 = os.path.abspath(sys.argv[0]) if sys.argv and os.path.isfile(sys.argv[0]) else ""
    if argv0 and os.path.basename(argv0).startswith("hermes-easy") and os.access(argv0, os.X_OK):
        return argv0
    return shutil.which("hermes-easy")


def publish_hermes_shim(home: Path) -> Path | None:
    """``$HERMES_HOME/bin/hermes`` → this hermes-easy executable (rewritten when it moves)."""
    target = _own_executable()
    if not target or os.name == "nt":
        return None
    shim = home / "bin" / "hermes"
    body = f'#!/bin/sh\n# hermes-easy: `hermes` inside hermes-easy sessions runs hermes-easy.\nexec "{target}" "$@"\n'
    try:
        if not shim.is_file() or shim.read_text(encoding="utf-8") != body:
            shim.parent.mkdir(parents=True, exist_ok=True)
            tmp = shim.with_suffix(".tmp")
            tmp.write_text(body, encoding="utf-8")
            tmp.chmod(0o755)
            os.replace(tmp, shim)
    except OSError:
        return None
    return shim


def seed_home(home: Path) -> None:
    """First launch: an interface default of the prompt_toolkit CLI, streaming on, metrics off.

    Only keys the user has not set are written, through upstream's config writer.
    """
    home.mkdir(parents=True, exist_ok=True)
    config_path = home / "config.yaml"
    if config_path.exists():
        return
    from hermes_cli.config import atomic_config_write
    from hermes_cli.observability.shared_metrics_consent import OFFER_VERSION
    atomic_config_write(config_path, {
        "display": {"interface": "cli", "cli_layout": "modern", "streaming": True},
        # The upstream first-run prompt asks this; hermes-easy starts with the "No thanks" answer.
        # `hermes-easy setup telemetry` changes it.
        "telemetry": {"shared_metrics": {"enabled": False, "send": False, "offer_version": OFFER_VERSION}},
    })


def _provider_env_keys(config: dict) -> set[str]:
    keys = set(_EXTRA_PROVIDER_KEYS)
    try:
        from hermes_cli.config_defaults import OPTIONAL_ENV_VARS
        keys |= {k for k, v in OPTIONAL_ENV_VARS.items()
                 if v.get("category") == "provider" and not k.startswith("HERMES_")}
    except Exception:  # health: allow BLE001 -- registry shape is upstream's; fall back to the extras
        pass
    for section in ("providers", "custom_providers"):
        entries = config.get(section) or {}
        values = entries.values() if isinstance(entries, dict) else entries if isinstance(entries, list) else []
        for entry in values:
            if isinstance(entry, dict):
                for field in ("key_env", "api_key_env"):
                    if isinstance(entry.get(field), str):
                        keys.add(entry[field])
    return keys


def _read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def import_upstream(argv: list[str], home: Path) -> int:
    """Copy model/provider settings from an upstream home. Reads the source; never writes it."""
    source = Path(os.path.expanduser(UPSTREAM_HOME))
    assume_yes = False
    rest = list(argv)
    while rest:
        arg = rest.pop(0)
        if arg in {"-y", "--yes"}:
            assume_yes = True
        elif arg == "--from" and rest:
            source = Path(os.path.expanduser(rest.pop(0)))
        elif arg in {"-h", "--help"}:
            print("usage: hermes-easy import-upstream [--from DIR] [--yes]\n\n"
                  "Copies the model/provider sections of DIR/config.yaml (default ~/.hermes) and the\n"
                  "provider API keys from DIR/.env into hermes-easy's home. Messaging, gateway, memory\n"
                  "and session data are not copied. DIR is only read.")
            return 0
        else:
            print(f"hermes-easy import-upstream: unknown argument {arg!r}", file=sys.stderr)
            return 2
    if _same_path(source, home):
        print("hermes-easy import-upstream: source and destination are the same directory.", file=sys.stderr)
        return 2
    src_config = source / "config.yaml"
    if not src_config.is_file():
        print(f"hermes-easy import-upstream: {src_config} not found.", file=sys.stderr)
        return 1
    from hermes_cli.config import read_user_config_raw
    raw = read_user_config_raw(src_config)
    sections = {k: raw[k] for k in _IMPORTED_CONFIG_SECTIONS if k in raw}
    src_env = _read_env(source / ".env")
    wanted = _provider_env_keys(raw)
    env_values = {k: v for k, v in src_env.items() if k in wanted and v}
    print(f"From {source} (read only) into {home}:")
    print(f"  config.yaml sections: {', '.join(sections) or 'none'}")
    print(f"  .env keys: {', '.join(sorted(env_values)) or 'none'} (values not shown)")
    if not sections and not env_values:
        return 0
    if not assume_yes:
        if not sys.stdin.isatty():
            print("Re-run with --yes to confirm.", file=sys.stderr)
            return 1
        if input("Copy these into hermes-easy? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing copied.")
            return 0
    seed_home(home)
    from hermes_cli.config import atomic_config_write, require_readable_config_before_write, save_env_value
    if sections:
        current = require_readable_config_before_write(home / "config.yaml")
        atomic_config_write(home / "config.yaml", {**current, **sections})
    for key, value in sorted(env_values.items()):
        save_env_value(key, value)
    print("Done. Launch with `hermes-easy`.")
    return 0


UPDATE_HELP = """hermes-easy does not use upstream's in-place updater (it restarts gateways and rewrites
the checkout it was installed from). Update with the installer, which pulls the checkout and
reinstalls the tool:

  curl -LsSf https://raw.githubusercontent.com/sarja-technologies/hermes-easy/main/scripts/install-hermes-easy.sh | sh"""


def acp_main() -> None:
    """``hermes-easy-acp``: upstream's ACP server for editors, on hermes-easy's home."""
    prepare_environment()
    os.environ.pop("HERMES_EASY_UI", None)  # no terminal UI on this surface
    from acp_adapter.entry import main as acp
    acp()


def main() -> None:
    home = prepare_environment()
    argv = sys.argv[1:]
    if argv[:1] == ["import-upstream"]:
        raise SystemExit(import_upstream(argv[1:], home))
    if argv[:1] == ["update"]:
        print(UPDATE_HELP)
        raise SystemExit(0)
    if argv[:1] in (["--version"], ["-V"]):
        from hermes_easy import __version__
        print(f"hermes-easy {__version__} (on Hermes Agent by Nous Research)")
    seed_home(home)
    from hermes_cli.main import main as hermes_main
    hermes_main()

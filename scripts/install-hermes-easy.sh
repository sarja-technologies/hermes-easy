#!/bin/sh
# Install hermes-easy as an isolated uv tool. Never touches an upstream Hermes install:
# the only command it adds is `hermes-easy` (and `hermes-easy-acp`), state lives in ~/.hermes-easy.
#
#   curl -LsSf https://raw.githubusercontent.com/sarja-technologies/hermes-easy/main/scripts/install-hermes-easy.sh | sh
#
# HERMES_EASY_DIR (default ~/.local/share/hermes-easy) is where the source checkout goes; the
# install is editable because Hermes resolves its skills, locales and plugin manifests from the
# checkout at runtime (upstream does not support wheel installs).
set -eu

REPO="${HERMES_EASY_REPO:-https://github.com/sarja-technologies/hermes-easy}"
DIR="${HERMES_EASY_DIR:-$HOME/.local/share/hermes-easy}"
REF="${HERMES_EASY_REF:-main}"

command -v git >/dev/null 2>&1 || { echo "hermes-easy: git is required" >&2; exit 1; }
if ! command -v uv >/dev/null 2>&1; then
  echo "hermes-easy: uv is required (https://docs.astral.sh/uv/getting-started/installation/)" >&2
  exit 1
fi

if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch --quiet origin "$REF"
  git -C "$DIR" checkout --quiet "$REF"
  git -C "$DIR" pull --quiet --ff-only origin "$REF"
else
  mkdir -p "$(dirname "$DIR")"
  git clone --quiet --branch "$REF" "$REPO" "$DIR"
fi

uv tool install --python 3.14 --editable "$DIR" --force

echo
echo "Installed. Start it with:  hermes-easy"
echo "Already use Hermes? Copy your model/provider settings with:  hermes-easy import-upstream"

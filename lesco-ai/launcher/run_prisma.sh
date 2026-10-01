#!/bin/sh
set -eu

LAUNCHER_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname -- "$LAUNCHER_DIR")

if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    exec "$PROJECT_ROOT/.venv/bin/python" "$LAUNCHER_DIR/launcher.py"
fi

exec python3 "$LAUNCHER_DIR/launcher.py"

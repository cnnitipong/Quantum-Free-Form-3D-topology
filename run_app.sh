#!/usr/bin/env bash
# Launch the FreeTO-Python web app on Linux.
#
#  - Finds a real Python 3.10-3.13 interpreter, preferring one of those exact
#    versions on PATH over whatever generic "python3" resolves to (which
#    could be 3.14+, and the optional "pyamg" fast solver has no wheels for
#    that yet -- see requirements.txt).
#  - Creates the virtual environment OUTSIDE this project folder, at
#    ${XDG_DATA_HOME:-~/.local/share}/FreeTO-Python/venv (override with the
#    FREETO_VENV environment variable) -- this folder itself may live in a
#    long, spaced, cloud-synced path, which is a bad place for a venv.
#  - Installs requirements.txt only when it is new or has changed since the
#    last successful install (compared byte-for-byte with `cmp` against a
#    saved copy next to the venv), so repeat launches work fully offline.
#  - Always calls the venv's own python directly ("$VPY" -m ...) instead of
#    `source .venv/bin/activate` + a bare `python`/`pip`.
#  - Does not use `set -e`: every step is checked explicitly, and this script
#    exits with a clear message on failure rather than an opaque stack trace
#    (there is no "press Enter to close" pause here, unlike the macOS/Windows
#    double-click launchers -- run_app.sh is normally started from a terminal
#    that was already open, so it never needs to be kept from vanishing).
cd "$(dirname "$0")" || { echo "ERROR: could not find the app folder."; exit 1; }

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
  p="$(command -v "$c" 2>/dev/null)" || continue
  if "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    PY="$p"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "ERROR: no Python 3.10 or newer found on PATH."
  echo "Install it with your distro's package manager (e.g. 'sudo apt install python3')"
  echo "or from https://www.python.org/downloads/ and try again."
  exit 1
fi

PY_VER="$("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null)"
case "$PY_VER" in
  3.10|3.11|3.12|3.13) ;;
  *) echo "NOTE: using Python $PY_VER. The optional \"amg\" fast solver (pyamg) has"
     echo "no wheels for Python 3.14+ yet, so it will be skipped automatically." ;;
esac

VENV_DIR="${FREETO_VENV:-${XDG_DATA_HOME:-$HOME/.local/share}/FreeTO-Python/venv}"
VPY="$VENV_DIR/bin/python"

if [ ! -x "$VPY" ]; then
  echo "Creating virtual environment at:"
  echo "  $VENV_DIR"
  if ! "$PY" -m venv "$VENV_DIR"; then
    echo "ERROR: failed to create the virtual environment."
    exit 1
  fi
  echo "Upgrading pip in the new environment…"
  "$VPY" -m pip install --quiet --upgrade pip
fi

if [ ! -x "$VPY" ]; then
  echo "ERROR: the virtual environment at '$VENV_DIR' looks incomplete."
  echo "Delete that folder and run this script again."
  exit 1
fi

STAMP_FILE="$VENV_DIR/requirements.installed"
NEED_INSTALL=1
if [ -f "$STAMP_FILE" ] && cmp -s requirements.txt "$STAMP_FILE"; then
  NEED_INSTALL=0
fi

if [ "$NEED_INSTALL" = "1" ]; then
  echo "Installing dependencies (requirements.txt is new or has changed)…"
  if ! "$VPY" -m pip install -r requirements.txt; then
    echo "ERROR: dependency installation failed. Check your internet connection and the error above."
    exit 1
  fi
  cp requirements.txt "$STAMP_FILE"
else
  echo "Dependencies already installed and up to date; skipping install (offline-friendly)."
fi

echo "Starting FreeTO-Python web app…"
exec "$VPY" -m webapp.server --port 8000 --open "$@"

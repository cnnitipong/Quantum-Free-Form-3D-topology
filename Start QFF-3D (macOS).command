#!/usr/bin/env bash
# Double-click launcher for macOS Finder: creates/activates a virtual
# environment, installs dependencies (only when needed), and starts the
# QFF-3D web app (Quantum Free-Form 3D Topology Optimisation, built on
# FreeTO; Python package `freeto`), opening it in the default browser.
#
#  - Finds a real Python 3.10-3.13 interpreter, preferring one of those exact
#    versions over whatever "python3" happens to resolve to (see PY_DISCOVERY
#    below) -- and specifically avoiding Apple's /usr/bin/python3 shim when
#    the Xcode Command Line Tools aren't installed, which pops a blocking
#    "install developer tools?" dialog instead of running anything.
#  - Creates the virtual environment OUTSIDE this project folder, at
#    ~/Library/Application Support/FreeTO-Python/venv (override with the
#    FREETO_VENV environment variable; the folder name is kept from
#    FreeTO-Python so an existing install is reused) -- this folder itself may live in a
#    long, spaced, iCloud-synced Documents path, which is a bad place for a
#    venv (large binary files, "Files On-Demand" eviction).
#  - Installs requirements.txt only when it is new or has changed since the
#    last successful install (compared byte-for-byte with `cmp` against a
#    saved copy next to the venv), so repeat launches work fully offline.
#  - Always calls the venv's own python directly ("$VPY" -m ...) instead of
#    `source .venv/bin/activate` + a bare `python`/`pip` -- works the same
#    whether this happens to run under bash or zsh, and is immune to PATH
#    surprises from a non-ASCII/spaced folder name.
#  - Does not use `set -e`: every step is checked explicitly, and this
#    Terminal window stays open (waits for Enter) whether the app exits
#    cleanly or with an error, so a double-click launch never just vanishes.
cd "$(dirname "$0")" || {
  echo "ERROR: could not find the app folder."
  echo "Press Enter to close this window."
  read -r _
  exit 1
}

fail() {
  echo
  echo "ERROR: $1"
  echo "Press Enter to close this window."
  read -r _
  exit 1
}

# ---------------------------------------------------------------------
# 1. Find a usable Python. Order: named versions on PATH (newest first),
#    then the python.org / Homebrew install locations directly (in case
#    ~/.zprofile PATH edits didn't apply -- e.g. Finder->Terminal without a
#    login shell, or Python installed "for this user only"), then whatever
#    plain "python3" resolves to -- skipping Apple's /usr/bin/python3 stub
#    specifically when the Command Line Tools are missing, since running it
#    pops a GUI dialog and exits nonzero rather than giving a clear error.
# ---------------------------------------------------------------------
PY=""
CANDIDATES="python3.13 python3.12 python3.11 python3.10"
for v in 13 12 11 10; do
  CANDIDATES="$CANDIDATES /Library/Frameworks/Python.framework/Versions/3.$v/bin/python3"
done
for v in 13 12 11 10; do
  CANDIDATES="$CANDIDATES /opt/homebrew/bin/python3.$v"
done
CANDIDATES="$CANDIDATES python3"

for c in $CANDIDATES; do
  p="$(command -v "$c" 2>/dev/null)" || continue
  if [ "$p" = "/usr/bin/python3" ] && ! xcode-select -p >/dev/null 2>&1; then
    continue  # Apple's shim: pops "install developer tools?" without them
  fi
  if "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    PY="$p"
    break
  fi
done

[ -n "$PY" ] || fail "no Python 3.10 or newer found. Install it from https://www.python.org/downloads/macos/ (or 'brew install python@3.13'), then double-click this launcher again."

PY_VER="$("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null)"
case "$PY_VER" in
  3.10|3.11|3.12|3.13) ;;
  *) echo "NOTE: using Python $PY_VER. The optional \"amg\" fast solver (pyamg) has"
     echo "no wheels for Python 3.14+ yet, so it will be skipped automatically" ;;
esac

# ---------------------------------------------------------------------
# 2. Virtual environment -- outside the project folder on purpose (see
#    header comment / README's "Where things live" section).
# ---------------------------------------------------------------------
VENV_DIR="${FREETO_VENV:-$HOME/Library/Application Support/FreeTO-Python/venv}"
VPY="$VENV_DIR/bin/python"

if [ ! -x "$VPY" ]; then
  echo "Creating virtual environment at:"
  echo "  $VENV_DIR"
  "$PY" -m venv "$VENV_DIR" || fail "failed to create the virtual environment."
  echo "Upgrading pip in the new environment…"
  "$VPY" -m pip install --quiet --upgrade pip
fi

[ -x "$VPY" ] || fail "the virtual environment at '$VENV_DIR' looks incomplete. Delete that folder and run this launcher again."

# ---------------------------------------------------------------------
# 3. Dependencies -- only (re)installed when requirements.txt actually
#    changed, compared byte-for-byte against a saved copy next to the venv.
#    This keeps every launch after the first fully offline.
# ---------------------------------------------------------------------
STAMP_FILE="$VENV_DIR/requirements.installed"
NEED_INSTALL=1
if [ -f "$STAMP_FILE" ] && cmp -s requirements.txt "$STAMP_FILE"; then
  NEED_INSTALL=0
fi

if [ "$NEED_INSTALL" = "1" ]; then
  echo "Installing dependencies from requirements.txt…"
  echo "(First install can take a few minutes. Later launches are instant and"
  echo " work offline, as long as requirements.txt does not change.)"
  "$VPY" -m pip install -r requirements.txt || fail "dependency installation failed. Check your internet connection and the error above, then try again."
  cp requirements.txt "$STAMP_FILE"
else
  echo "Dependencies already installed and up to date; skipping install (works offline)."
fi

# ---------------------------------------------------------------------
# 4. Run.
# ---------------------------------------------------------------------
echo
printf '\033]0;QFF-3D - Quantum Free-Form 3D Topology Optimisation\007'
echo "Starting QFF-3D web app (Quantum Free-Form 3D Topology Optimisation)…"
"$VPY" -m webapp.server --port 8000 --open
SERVER_EXIT=$?

echo
if [ "$SERVER_EXIT" -ne 0 ]; then
  echo "Server exited with an error (see above)."
else
  echo "Server stopped."
fi
echo "Press Enter to close this window."
read -r _

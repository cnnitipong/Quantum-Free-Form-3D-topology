@echo off
setlocal enabledelayedexpansion
title FreeTO-Python

REM ============================================================================
REM Launch the FreeTO-Python web app on Windows.
REM
REM  - Finds a real Python 3.10-3.13 interpreter (prefers the "py" launcher;
REM    refuses the Microsoft Store "python.exe" alias, which cannot run
REM    anything). Python 3.14+ is accepted as a last resort, with a warning,
REM    because the optional "pyamg" fast solver has no 3.14 wheels yet.
REM  - Creates the virtual environment OUTSIDE this project folder, at
REM    %LOCALAPPDATA%\FreeTO-Python\venv (override with the FREETO_VENV
REM    environment variable) -- this folder's own path may be a long, spaced,
REM    OneDrive-synced Documents path, which is a bad place for a venv.
REM  - Installs requirements.txt only when it is new or has changed since the
REM    last successful install (compared byte-for-byte with `fc /b` against a
REM    saved copy next to the venv), so repeat launches work fully offline.
REM  - Always calls the venv's own python.exe directly ("%VPY%" -m ...)
REM    instead of activate.bat + a bare "python"/"pip", which is one less
REM    thing that can go wrong with PATH or non-ASCII folder names.
REM  - This window stays open (via `pause` at the very end) whether the app
REM    exits cleanly or with an error, so a double-click launch never just
REM    vanishes without explanation.
REM ============================================================================

pushd "%~dp0" 2>nul
if errorlevel 1 (
  echo ERROR: could not open the app folder ^(%~dp0^).
  goto :end_nopop
)

REM ----------------------------------------------------------------------
REM 1. Find a usable Python.
REM ----------------------------------------------------------------------
set "PY="
set "PYWARN="

where py >nul 2>nul
if not errorlevel 1 (
  for %%v in (3.13 3.12 3.11 3.10) do (
    if not defined PY (
      py -%%v -c "import sys" >nul 2>nul
      if not errorlevel 1 set "PY=py -%%v"
    )
  )
)

REM Plain "python" on PATH (python.org installer with "Add to PATH" ticked).
REM A Microsoft Store alias also answers to this name but exits nonzero with
REM no real Python behind it, so we still verify it actually runs code.
if not defined PY (
  python -c "import sys" >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<(3,14) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=python"
  )
)

REM Last resort: whatever "py -3" resolves to, even if it is 3.14+. Using it
REM (with a warning) beats refusing to start; pyamg just won't be available.
if not defined PY (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if not errorlevel 1 (
      set "PY=py -3"
      set "PYWARN=1"
    )
  )
)

if not defined PY (
  echo ERROR: No Python 3.10 or newer found.
  echo Install it from https://www.python.org/downloads/windows/
  echo ^(keep the "py launcher" option ticked; "Add python.exe to PATH" is optional^).
  goto :end
)

if defined PYWARN (
  echo WARNING: only Python 3.14+ was found. FreeTO-Python will still run, but
  echo the optional "amg" fast solver ^(pyamg^) has no Python 3.14 wheels yet and
  echo will be skipped automatically. To avoid this, install Python 3.12 from
  echo https://www.python.org/downloads/windows/ alongside your current Python.
)

REM ----------------------------------------------------------------------
REM 2. Virtual environment -- outside the project folder on purpose (this
REM    folder's own path may live in a long, spaced, cloud-synced Documents
REM    tree, which is a bad place for a venv: see README's "Where things
REM    live" section).
REM ----------------------------------------------------------------------
if defined FREETO_VENV (
  set "VENV_DIR=%FREETO_VENV%"
) else (
  set "VENV_DIR=%LOCALAPPDATA%\FreeTO-Python\venv"
)
set "VPY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VPY%" (
  echo Creating virtual environment at:
  echo   %VENV_DIR%
  %PY% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo ERROR: failed to create the virtual environment.
    goto :end
  )
  echo Upgrading pip in the new environment...
  "%VPY%" -m pip install --quiet --upgrade pip
)

if not exist "%VPY%" (
  echo ERROR: the virtual environment at %VENV_DIR% looks incomplete.
  echo Delete that folder and run this launcher again.
  goto :end
)

REM ----------------------------------------------------------------------
REM 3. Dependencies -- only (re)installed when requirements.txt actually
REM    changed, compared byte-for-byte against a saved copy next to the
REM    venv. This keeps every launch after the first fully offline.
REM ----------------------------------------------------------------------
set "STAMP_FILE=%VENV_DIR%\requirements.installed"
set "NEED_INSTALL=1"
if exist "%STAMP_FILE%" (
  fc /b "requirements.txt" "%STAMP_FILE%" >nul 2>nul
  if not errorlevel 1 set "NEED_INSTALL=0"
)

if "%NEED_INSTALL%"=="1" (
  echo Installing dependencies from requirements.txt...
  echo ^(First install can take a few minutes: it downloads roughly 200 MB
  echo for the optional fast MKL/PARDISO solver. Later launches are instant
  echo and work offline, as long as requirements.txt does not change.^)
  "%VPY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo ERROR: dependency installation failed. Check your internet connection
    echo and the error above, then try again.
    goto :end
  )
  copy /y requirements.txt "%STAMP_FILE%" >nul
) else (
  echo Dependencies already installed and up to date; skipping install ^(works offline^).
)

REM ----------------------------------------------------------------------
REM 4. Run.
REM ----------------------------------------------------------------------
echo.
echo Starting FreeTO-Python web app...
"%VPY%" -m webapp.server --port 8000 --open
if errorlevel 1 (
  echo.
  echo Server exited with an error ^(see above^).
) else (
  echo.
  echo Server stopped.
)

:end
popd
:end_nopop
echo.
echo Press any key to close this window.
pause >nul

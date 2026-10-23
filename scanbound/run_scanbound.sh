#!/usr/bin/env bash
# Start the ScanBound app on macOS or Linux.  The first run creates a Python environment in .venv
# next to this script and installs ScanBound into it (a few minutes); later runs start straight away.
#
#   bash run_scanbound.sh [data folder or project.json]
#   SCANBOUND_PYTHON=python3.12 bash run_scanbound.sh     # use a particular Python (3.10 or newer)
#   bash run_scanbound.sh --update                        # reinstall the dependencies, then start
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$here/.venv"

if [ "${1:-}" = "--update" ]; then
    update=1
    shift
else
    update=0
fi

if [ ! -x "$venv/bin/python" ]; then
    py="${SCANBOUND_PYTHON:-}"
    if [ -z "$py" ]; then
        for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
            if command -v "$candidate" >/dev/null 2>&1; then py="$candidate"; break; fi
        done
    fi
    if [ -z "$py" ] || ! "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
        echo "ScanBound needs Python 3.10 or newer." >&2
        echo "  macOS: install it from https://www.python.org/downloads/ or with 'brew install python@3.12'" >&2
        echo "  Linux: install python3 and python3-venv with your package manager" >&2
        exit 1
    fi
    echo "Creating the Python environment in $venv (using $("$py" --version 2>&1)) ..."
    if ! "$py" -m venv "$venv"; then
        echo "Could not create the environment. On Debian / Ubuntu: sudo apt install python3-venv" >&2
        exit 1
    fi
fi

# reinstall when asked, on the first run, and after an install that did not finish
if [ "$update" = 1 ] || [ ! -f "$venv/.installed" ]; then
    echo "Installing ScanBound and its dependencies ..."
    rm -f "$venv/.installed"
    "$venv/bin/python" -m pip install --upgrade pip
    "$venv/bin/python" -m pip install -e "$here[gui,clusters,formats]"
    touch "$venv/.installed"
fi

exec "$venv/bin/python" -m scanbound.app "$@"

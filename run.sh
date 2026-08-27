#!/usr/bin/env bash
# One-command run: creates the virtual environment if needed, installs the package, runs the pipeline.
# Usage: ./run.sh            (full pipeline)
#        ./run.sh backtest   (single stage; see `python -m supply_pipeline --help`)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=.venv/Scripts/python.exe; fi

"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -e ".[dev]"

if [ $# -eq 0 ]; then
  "$PY" -m supply_pipeline run
else
  "$PY" -m supply_pipeline "$@"
fi

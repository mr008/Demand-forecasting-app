# One-command run: creates the virtual environment if needed, installs the package, runs the pipeline.
# Usage: .\run.ps1            (full pipeline)
#        .\run.ps1 backtest   (single stage; see `python -m supply_pipeline --help`)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
& ".venv\Scripts\python.exe" -m pip install --quiet -e ".[dev]"

if ($args.Count -eq 0) {
    & ".venv\Scripts\python.exe" -m supply_pipeline run
} else {
    & ".venv\Scripts\python.exe" -m supply_pipeline @args
}

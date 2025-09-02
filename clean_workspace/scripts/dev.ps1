param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [switch]$NoReload
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path (Join-Path $here '..')
Set-Location $root

if (-not (Test-Path ".venv\\Scripts\\Activate.ps1")) {
    Write-Host "Creating venv..."
    python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
python -m pip install -U pip > $null
if (Test-Path "$root\requirements.txt") {
    Write-Host "Installing deps from requirements.txt"
    pip install -r requirements.txt
}
elseif (Test-Path "$root\pyproject.toml") {
    Write-Host "Installing deps from pyproject.toml"
    pip install -e .
}

$env:UVICORN_HOST = $BindHost
$env:UVICORN_PORT = "$Port"

# Stop existing process on port to avoid bind errors
& "$root\scripts\stop-port.ps1" -Port $Port | Out-Null

$argsList = @()
if ($NoReload) { $argsList += "--no-reload" }

Write-Host ("Launching dev server at http://{0}:{1} (reload={2})" -f $BindHost, $Port, ([bool](-not $NoReload)))
python .\dev.py @argsList

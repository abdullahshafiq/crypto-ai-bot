$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$env:BOT_CONFIG = "config.yaml"
$env:BOT_INSTANCE_PORT = "45678"

function Test-PortInUse([int]$port) {
    return [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

if (Test-PortInUse 45678) {
    Write-Error "Live bot port 45678 is already in use. Stop the existing live bot first, or attach to the current session."
    exit 1
}

$explicit = "C:\Users\A\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$pythonArgs = @("-u", "main.py")

Write-Host "Starting LIVE bot on dashboard http://127.0.0.1:8765"

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py @pythonArgs
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python @pythonArgs
    exit $LASTEXITCODE
}

if (Test-Path $explicit) {
    & $explicit @pythonArgs
    exit $LASTEXITCODE
}

throw "Python launcher not available. Tried: py, python, and $explicit"

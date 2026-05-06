$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$env:BOT_CONFIG = "config.paper.test.yaml"
$env:BOT_INSTANCE_PORT = "45679"

function Test-PortInUse([int]$port) {
    return [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

if (Test-PortInUse 45679) {
    Write-Error "Demo bot port 45679 is already in use. Stop the existing demo bot first, or attach to the current session."
    exit 1
}

$explicit = "C:\Users\A\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$pythonArgs = @("-u", "main.py")

Write-Host "Starting DEMO bot on dashboard http://127.0.0.1:8766"

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

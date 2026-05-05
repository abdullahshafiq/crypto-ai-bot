$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
Remove-Item Env:\BOT_CONFIG -ErrorAction SilentlyContinue
$env:BOT_INSTANCE_PORT = "45678"

function Try-Run($exe, $args) {
    try {
        & $exe @args
        return $LASTEXITCODE
    } catch {
        return $null
    }
}

$code = Try-Run "py" @("main.py")
if ($null -ne $code) {
    exit $code
}

$code = Try-Run "python" @("main.py")
if ($null -ne $code) {
    exit $code
}

$explicit = "C:\Users\A\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if (Test-Path $explicit) {
    & $explicit "main.py"
    exit $LASTEXITCODE
}

throw "Python launcher not available. Tried: py, python, and $explicit"

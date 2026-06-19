$ErrorActionPreference = "SilentlyContinue"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $projectDir "logs"
$pidFile = Join-Path $logDir "wafer_app_pids.json"

if (Test-Path $pidFile) {
    $pids = Get-Content -Raw $pidFile | ConvertFrom-Json
    foreach ($property in $pids.PSObject.Properties) {
        Stop-Process -Id ([int]$property.Value) -Force
    }
    Remove-Item $pidFile -Force
}

Get-CimInstance Win32_Process |
    Where-Object {
        $_.CommandLine -like "*uvicorn*src.api.main:app*" -or
        $_.CommandLine -like "*streamlit*src/dashboard/app.py*"
    } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

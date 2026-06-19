$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectDir ".venv\Scripts\python.exe"
$venvActivate = Join-Path $projectDir ".venv\Scripts\Activate.ps1"
$logDir = Join-Path $projectDir "logs"
$pidFile = Join-Path $logDir "wafer_app_pids.json"
$needInstall = $false
$needPipeline = $false

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $projectDir

function Test-PortOpen {
    param([int]$Port)
    try {
        $client = New-Object Net.Sockets.TcpClient
        $async = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        $success = $async.AsyncWaitHandle.WaitOne(500, $false)
        if ($success) {
            $client.EndConnect($async)
        }
        $client.Close()
        return $success
    } catch {
        return $false
    }
}

try {
    if (-not (Test-Path $venvPython)) {
        python -m venv .venv
        $needInstall = $true
    }

    . $venvActivate

    python -c "import fastapi, streamlit, torch, pandas, sklearn, plotly" *> $null
    if ($LASTEXITCODE -ne 0) {
        $needInstall = $true
    }

    if ($needInstall) {
        python -m pip install -r requirements.txt *> (Join-Path $logDir "install.log")
    }

    if (-not (Test-Path (Join-Path $projectDir "data\wafer_quality.db"))) {
        $needPipeline = $true
    }
    if (-not (Test-Path (Join-Path $projectDir "saved_models\wafer_cnn_model.pt"))) {
        $needPipeline = $true
    }
    if ($needPipeline) {
        python -m src.pipeline --epochs 6 *> (Join-Path $logDir "pipeline.log")
    }

    $started = [ordered]@{}
    if (-not (Test-PortOpen 8000)) {
        $api = Start-Process -FilePath $venvPython `
            -ArgumentList @("-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000") `
            -WorkingDirectory $projectDir `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logDir "api.out.log") `
            -RedirectStandardError (Join-Path $logDir "api.err.log") `
            -PassThru
        $started["api"] = $api.Id
    }

    if (-not (Test-PortOpen 8501)) {
        $env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
        $dashboard = Start-Process -FilePath $venvPython `
            -ArgumentList @("-m", "streamlit", "run", "src/dashboard/app.py", "--server.address", "127.0.0.1", "--server.port", "8501", "--browser.gatherUsageStats", "false", "--server.headless", "true") `
            -WorkingDirectory $projectDir `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logDir "dashboard.out.log") `
            -RedirectStandardError (Join-Path $logDir "dashboard.err.log") `
            -PassThru
        $started["dashboard"] = $dashboard.Id
    }

    $started | ConvertTo-Json | Set-Content -Path $pidFile -Encoding UTF8
    Start-Sleep -Seconds 5
    Start-Process "http://127.0.0.1:8501"
} catch {
    $_ | Out-String | Set-Content -Path (Join-Path $logDir "launcher_error.log") -Encoding UTF8
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show("프로젝트 실행 중 오류가 발생했습니다. logs\launcher_error.log를 확인하세요.", "Wafer Launcher") | Out-Null
    exit 1
}

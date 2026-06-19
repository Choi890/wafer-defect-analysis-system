@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
set "VENV_ACTIVATE=%PROJECT_DIR%.venv\Scripts\activate.bat"
set "NEED_INSTALL=0"
set "NEED_PIPELINE=0"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

title Wafer Defect Analysis System Launcher
cd /d "%PROJECT_DIR%"

echo [1/4] Checking Python virtual environment...
if not exist "%VENV_PYTHON%" (
    echo Creating .venv...
    python -m venv .venv
    if errorlevel 1 goto fail
    set "NEED_INSTALL=1"
)

echo [2/4] Checking dependencies...
call "%VENV_ACTIVATE%"
python -c "import fastapi, streamlit, torch, pandas, sklearn, plotly" >nul 2>nul
if errorlevel 1 set "NEED_INSTALL=1"
if "%NEED_INSTALL%"=="1" (
    echo Installing dependencies...
    python -m pip install -r requirements.txt
    if errorlevel 1 goto fail
) else (
    echo Dependencies are ready.
)

echo [3/4] Preparing demo database and model if needed...
if not exist "data\wafer_quality.db" set "NEED_PIPELINE=1"
if not exist "saved_models\wafer_cnn_model.pt" set "NEED_PIPELINE=1"
if "%NEED_PIPELINE%"=="1" (
    python -m src.pipeline --epochs 6
    if errorlevel 1 goto fail
) else (
    echo Demo database and model are ready.
)

echo [4/4] Starting API and dashboard...
start "Wafer API : http://127.0.0.1:8000/docs" /D "%PROJECT_DIR%" cmd /k "call .venv\Scripts\activate.bat && uvicorn src.api.main:app --host 127.0.0.1 --port 8000"
start "Wafer Dashboard : http://127.0.0.1:8501" /D "%PROJECT_DIR%" cmd /k "set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false && call .venv\Scripts\activate.bat && streamlit run src/dashboard/app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false"

echo.
echo Project is starting.
echo API docs:  http://127.0.0.1:8000/docs
echo Dashboard: http://127.0.0.1:8501
timeout /t 5 /nobreak >nul
start "" "http://127.0.0.1:8501"
echo.
echo You can close this launcher window. Keep the API and Dashboard windows open while using the app.
pause
exit /b 0

:fail
echo.
echo Failed to start the project. Check the error above.
pause
exit /b 1

@echo off
chcp 65001 >nul
echo ================================================
echo   Setup Virtual Environment
echo ================================================
echo.

if exist "venv\Scripts\python.exe" (
    echo [OK] Virtual environment already exists.
    echo.
) else (
    echo [Wait] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [Error] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
    echo.
)

echo [Wait] Installing dependencies (numpy + pandas)...
echo.

call venv\Scripts\pip.exe install numpy pandas -q

if errorlevel 1 (
    echo.
    echo [Error] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Setup complete! 
echo   Double-click "run_training.bat" to start training.
echo ================================================
echo.
pause

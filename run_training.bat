@echo off
chcp 65001 >nul
echo ================================================
echo   CNN CPI Prediction Trainer
echo ================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [Warning] Virtual environment not found.
    echo [Wait] Running setup first...
    call setup.bat
    if errorlevel 1 (
        echo [Error] Setup failed.
        pause
        exit /b 1
    )
)

if not exist "data\common_prosperity_data.csv" (
    echo [Info] Generating training data...
    call venv\Scripts\python.exe generate_data.py
    if errorlevel 1 (
        echo [Error] Data generation failed.
        pause
        exit /b 1
    )
    echo.
)

echo [Start] Training CNN model...
echo         Press Ctrl+C to interrupt at any time.
echo         Best model will be auto-selected.
echo.

call venv\Scripts\python.exe train_cnn.py %*

echo.
if errorlevel 1 (
    echo [Warning] Training interrupted or completed with errors.
) else (
    echo [Done] Training complete! Best model saved to models\best_cnn_lstm_model.h5
)
echo.
pause

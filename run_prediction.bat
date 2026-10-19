@echo off
chcp 65001 >nul
echo ================================================
echo   CPI Prediction Results
echo ================================================
echo.

if not exist "models\best_cnn_lstm_model.h5" (
    echo [Error] No trained model found.
    echo         Please run run_training.bat first.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\python.exe predict.py

echo.
pause

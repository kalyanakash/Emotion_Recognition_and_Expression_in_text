@echo off
echo 🚀 Starting AI Emotion Detector...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install requirements if needed
echo 📦 Installing/updating requirements...
pip install -r requirements.txt --quiet

REM Check if all required files exist
if not exist "model_architecture.json" (
    echo ❌ model_architecture.json not found
    pause
    exit /b 1
)

if not exist "model_weights.h5" (
    echo ❌ model_weights.h5 not found
    pause
    exit /b 1
)

if not exist "train.txt" (
    echo ❌ train.txt not found
    pause
    exit /b 1
)

echo ✅ All required files found
echo.
echo 🌟 Starting Streamlit app...
echo 📱 The app will open in your default browser
echo 🔗 URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

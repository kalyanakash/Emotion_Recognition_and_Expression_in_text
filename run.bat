@echo off
echo 🎭 Starting AI Emotion Detector...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Install/update requirements
echo 📦 Installing/updating requirements...
pip install streamlit pandas numpy tensorflow plotly scikit-learn --quiet

if %errorlevel% neq 0 (
    echo ❌ Failed to install requirements
    echo 💡 Try running: pip install --upgrade pip
    pause
    exit /b 1
)

echo ✅ Requirements installed successfully!
echo.

REM Start the Streamlit app
echo 🚀 Starting Streamlit app...
echo 🌐 The app will open in your browser at http://localhost:8501
echo 🛑 Press Ctrl+C to stop the server
echo.

streamlit run app.py

pause

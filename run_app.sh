#!/bin/bash

echo "🚀 Starting AI Emotion Detector..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Install requirements if needed
echo "📦 Installing/updating requirements..."
pip3 install -r requirements.txt --quiet

# Check if all required files exist
if [ ! -f "model_architecture.json" ]; then
    echo "❌ model_architecture.json not found"
    exit 1
fi

if [ ! -f "model_weights.h5" ]; then
    echo "❌ model_weights.h5 not found"
    exit 1
fi

if [ ! -f "train.txt" ]; then
    echo "❌ train.txt not found"
    exit 1
fi

echo "✅ All required files found"
echo
echo "🌟 Starting Streamlit app..."
echo "📱 The app will open in your default browser"
echo "🔗 URL: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the server"
echo

streamlit run app.py

#!/bin/bash

echo "🎭 Starting AI Emotion Detector..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.9+ from https://python.org"
    exit 1
fi

# Install requirements if needed
echo "📦 Installing/updating requirements..."
pip3 install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    exit 1
fi

echo "✅ Requirements installed successfully!"
echo

# Start the Streamlit app
echo "🚀 Starting Streamlit app..."
echo "🌐 The app will open in your browser at http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo

streamlit run app.py

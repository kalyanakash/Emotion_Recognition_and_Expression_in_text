#!/bin/bash
echo "🎭 Starting AI Emotion Detector..."
echo "📦 Installing dependencies..."

pip install streamlit==1.37.1 pandas numpy tensorflow plotly scikit-learn

echo "🚀 Launching application..."
echo "🌐 Open your browser to: http://localhost:8501"

streamlit run app.py

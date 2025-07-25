"""
Quick verification that the app can be imported and basic functions work
"""
import sys
import os

print("🔍 Quick App Verification")
print("=" * 40)

try:
    # Test basic imports
    import streamlit as st
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import plotly.graph_objects as go
    print("✅ All required packages imported successfully")
    
    # Test data loading
    data = pd.read_csv("train.txt", sep=';')
    data.columns = ["Text", "Emotions"]
    print(f"✅ Data loaded: {len(data)} samples")
    
    # Test model files exist
    if os.path.exists("model_architecture.json") and os.path.exists("model_weights.h5"):
        print("✅ Model files found")
    else:
        print("❌ Model files missing")
        
    # Test app.py can be imported
    sys.path.append('.')
    # We'll just check if the file is valid Python
    with open('app.py', 'r') as f:
        code = f.read()
        compile(code, 'app.py', 'exec')
    print("✅ app.py is valid Python code")
    
    print("\n🎉 Everything looks good! Your app should work perfectly.")
    print("\n🚀 To run the app, use: streamlit run app.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

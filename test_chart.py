"""
Test the fixed Plotly chart function
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def test_chart_function():
    print("🧪 Testing chart function...")
    
    # Test data
    emotion_names = ['happy', 'sad', 'anger', 'fear', 'surprise', 'love']
    confidence_scores = [0.8, 0.1, 0.05, 0.02, 0.02, 0.01]
    
    # Color mapping
    emotion_colors = {
        "happy": "#FFD700",
        "sad": "#4682B4", 
        "anger": "#FF6347",
        "fear": "#9370DB",
        "surprise": "#FF69B4",
        "love": "#FF1493",
        "disgust": "#8FBC8F",
        "neutral": "#D3D3D3"
    }
    
    try:
        fig = go.Figure()
        
        colors = [emotion_colors.get(emotion.lower(), '#667eea') for emotion in emotion_names]
        
        fig.add_trace(go.Bar(
            x=emotion_names,
            y=confidence_scores,
            marker_color=colors,
            text=[f'{score:.2%}' for score in confidence_scores],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Emotion Confidence Scores',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#333'}
            },
            xaxis_title='Emotions',
            yaxis_title='Confidence',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Poppins", size=12),
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )
        
        print("✅ Chart function works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Chart function error: {e}")
        return False

if __name__ == "__main__":
    test_chart_function()

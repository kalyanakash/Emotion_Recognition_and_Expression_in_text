import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Load and preprocess the data with error handling
@st.cache_data
def load_training_data():
    try:
        data = pd.read_csv("train.txt", sep=';')
        data.columns = ["Text", "Emotions"]
        return data
    except FileNotFoundError:
        st.error("Training data file not found. Please ensure train.txt is in the repository.")
        return None
    except Exception as e:
        st.error(f"Error loading training data: {str(e)}")
        return None

# Initialize data loading
data = load_training_data()
if data is None:
    st.stop()

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

# Tokenizer and Label Encoder setup
@st.cache_resource
def setup_tokenizer_and_encoder(texts, labels):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    max_length = max([len(seq) for seq in tokenizer.texts_to_sequences(texts)])
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    return tokenizer, label_encoder, max_length

tokenizer, label_encoder, max_length = setup_tokenizer_and_encoder(texts, labels)

# Load the saved model architecture and weights with error handling
@st.cache_resource
def load_model():
    try:
        with open("model_architecture.json", "r") as json_file:
            loaded_model_json = json_file.read()
        
        loaded_model = model_from_json(loaded_model_json)
        
        # Try loading weights from different possible files
        weight_files = ["model_weights.h5", "model_weights.weights.h5"]
        weights_loaded = False
        
        for weight_file in weight_files:
            try:
                loaded_model.load_weights(weight_file)
                weights_loaded = True
                st.success(f"✅ Model loaded successfully with weights from {weight_file}")
                break
            except:
                continue
        
        if not weights_loaded:
            st.error("❌ Could not load model weights from any available file")
            return None
            
        return loaded_model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

loaded_model = load_model()
if loaded_model is None:
    st.error("Model failed to load. Please check if model files are present.")
    st.stop()

# Define CSS styles with modern design
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

.main {
    padding: 0rem 1rem;
}

.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Poppins', sans-serif;
}

.main-header {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
    text-align: center;
}

.main-header h1 {
    color: white;
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.main-header p {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.2rem;
    font-weight: 300;
}

.input-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.result-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.emotion-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    font-weight: 600;
    font-size: 1.1rem;
    margin: 0.5rem;
    text-align: center;
    min-width: 120px;
}

.emotion-happy { background: linear-gradient(45deg, #FFD700, #FFA500); color: #333; }
.emotion-joy { background: linear-gradient(45deg, #FFD700, #FFA500); color: #333; }
.emotion-sad { background: linear-gradient(45deg, #4682B4, #1E90FF); color: white; }
.emotion-sadness { background: linear-gradient(45deg, #4682B4, #1E90FF); color: white; }
.emotion-anger { background: linear-gradient(45deg, #FF6347, #DC143C); color: white; }
.emotion-fear { background: linear-gradient(45deg, #9370DB, #8A2BE2); color: white; }
.emotion-surprise { background: linear-gradient(45deg, #FF69B4, #FF1493); color: white; }
.emotion-love { background: linear-gradient(45deg, #FF69B4, #FF1493); color: white; }
.emotion-disgust { background: linear-gradient(45deg, #8FBC8F, #228B22); color: white; }
.emotion-neutral { background: linear-gradient(45deg, #D3D3D3, #A9A9A9); color: #333; }

.stats-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    margin: 0.5rem;
}

.stats-number {
    font-size: 2rem;
    font-weight: 700;
    color: #667eea;
}

.stats-label {
    font-size: 0.9rem;
    color: #666;
    font-weight: 500;
}

.history-item {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #667eea;
}

.confidence-bar {
    background: #f0f0f0;
    border-radius: 10px;
    height: 20px;
    margin: 5px 0;
    overflow: hidden;
}

.footer {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 1rem;
    margin-top: 2rem;
    text-align: center;
    color: rgba(255, 255, 255, 0.8);
}
</style>
"""

# Enhanced emoji mapping for emotions (updated to match training data)
emotion_emojis = {
    "joy": "😊",
    "sadness": "😢", 
    "anger": "😡",
    "fear": "😨",
    "surprise": "😮",
    "love": "❤️",
    "disgust": "🤢",
    "neutral": "😐",
    # Additional mappings for potential other emotions
    "happy": "😊",
    "sad": "😢"
}

# Color mapping for emotions (updated to match training data)
emotion_colors = {
    "joy": "#FFD700",
    "sadness": "#4682B4", 
    "anger": "#FF6347",
    "fear": "#9370DB",
    "surprise": "#FF69B4",
    "love": "#FF1493",
    "disgust": "#8FBC8F",
    "neutral": "#D3D3D3",
    # Additional mappings
    "happy": "#FFD700",
    "sad": "#4682B4"
}

# Function to preprocess input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to create animated confidence chart
def create_confidence_chart(emotion_names, confidence_scores):
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
    
    return fig

# Function to create emotion distribution chart
def create_emotion_distribution(history):
    if not history:
        return None
    
    emotions = [item[1] for item in history]
    emotion_counts = pd.Series(emotions).value_counts()
    
    colors = [emotion_colors.get(emotion.lower(), '#667eea') for emotion in emotion_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=emotion_counts.index,
        values=emotion_counts.values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=12,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': 'Emotion Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#333'}
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins", size=12),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Emotion Detector",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Insert CSS styles
    st.markdown(css, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 AI Emotion Detector</h1>
        <p>Powered by Deep Learning • Real-time Text Emotion Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []
    if "total_predictions" not in st.session_state:
        st.session_state.total_predictions = 0

    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Dashboard")
        
        # Statistics
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{st.session_state.total_predictions}</div>
            <div class="stats-label">Total Predictions</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.emotion_history:
            most_common = pd.Series([item[1] for item in st.session_state.emotion_history]).mode()[0]
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{emotion_emojis.get(most_common, '🙂')}</div>
                <div class="stats-label">Most Common: {most_common.title()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Settings
        st.markdown("### ⚙️ Settings")
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_history = st.checkbox("Show History", value=True)
        max_history = st.slider("Max History Items", 5, 50, 10)
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.emotion_history = []
            st.session_state.total_predictions = 0
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input section
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown("### 💬 Enter Your Text")
        
        input_text = st.text_area(
            "",
            placeholder="Type your message here... (e.g., 'I'm feeling great today!')",
            key="text_area",
            height=120,
            help="Enter any text to analyze its emotional content"
        )
        
        col_pre, col_btn = st.columns([1, 1])
        with col_pre:
            preprocess = st.checkbox("🔧 Preprocess Text", help="Apply lowercase and remove punctuation")
        
        with col_btn:
            detect_btn = st.button("🔍 Detect Emotion", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Prediction section
        if detect_btn:
            if input_text.strip():
                with st.spinner("🤖 Analyzing emotions..."):
                    # Add a small delay for better UX
                    time.sleep(0.5)
                    
                    processed_text = preprocess_text(input_text) if preprocess else input_text
                    
                    # Text tokenization and padding
                    input_sequence = tokenizer.texts_to_sequences([processed_text])
                    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
                    
                    # Predict and decode emotion
                    prediction = loaded_model.predict(padded_input_sequence, verbose=0)
                    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
                    confidence = np.max(prediction[0])
                    
                    # Get emoji for the detected emotion
                    emoji = emotion_emojis.get(predicted_label, "🙂")
                    
                    # Update session state
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.emotion_history.append((input_text, predicted_label, emoji, confidence, timestamp))
                    st.session_state.total_predictions += 1
                    
                    # Display result
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Detection Result")
                    
                    # Main result
                    emotion_class = f"emotion-{predicted_label.lower()}"
                    st.markdown(f"""
                    <div class="emotion-badge {emotion_class}">
                        {emoji} {predicted_label.upper()} ({confidence:.1%})
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Success message
                    st.success(f"✅ Detected emotion: **{predicted_label}** with {confidence:.1%} confidence")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence chart
                    if show_confidence:
                        emotion_names = label_encoder.inverse_transform(np.arange(len(prediction[0])))
                        fig = create_confidence_chart(emotion_names, prediction[0])
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("⚠️ Please enter some text to analyze!")

    with col2:
        # Emotion distribution chart
        if st.session_state.emotion_history and len(st.session_state.emotion_history) > 1:
            st.markdown("### 📈 Emotion Trends")
            fig_dist = create_emotion_distribution(st.session_state.emotion_history)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)

    # History section
    if show_history and st.session_state.emotion_history:
        st.markdown("### 📚 Recent Analysis History")
        
        # Display recent history
        recent_history = st.session_state.emotion_history[-max_history:][::-1]
        
        for i, (text, emotion, emoji, confidence, timestamp) in enumerate(recent_history):
            with st.expander(f"{emoji} {emotion.title()} - {timestamp}", expanded=False):
                st.write(f"**Text:** {text}")
                st.write(f"**Emotion:** {emotion.title()} {emoji}")
                st.write(f"**Confidence:** {confidence:.2%}")
                st.write(f"**Time:** {timestamp}")

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🚀 Built with Streamlit & TensorFlow | 🎯 Deep Learning for Emotion Recognition</p>
        <p>💡 Tip: Try different types of text to see how emotions are detected!</p>
    </div>
    """, unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    main()
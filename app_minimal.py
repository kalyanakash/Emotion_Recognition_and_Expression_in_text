import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from datetime import datetime
import time

# Fallback emotion detection (simple keyword-based)
def simple_emotion_detection(text):
    """Simple keyword-based emotion detection as fallback"""
    text = text.lower()
    
    emotion_keywords = {
        'joy': ['happy', 'joy', 'excited', 'great', 'amazing', 'wonderful', 'fantastic', 'excellent', 'love', 'good'],
        'sadness': ['sad', 'depressed', 'unhappy', 'disappointed', 'down', 'upset', 'cry', 'lonely', 'hurt'],
        'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'hate', 'frustrated', 'rage'],
        'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'panic', 'terrified'],
        'surprise': ['surprised', 'shocked', 'amazed', 'wow', 'incredible', 'unbelievable'],
        'love': ['love', 'adore', 'cherish', 'romantic', 'heart', 'valentine', 'kiss']
    }
    
    emotion_scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text)
        emotion_scores[emotion] = score
    
    if all(score == 0 for score in emotion_scores.values()):
        return 'neutral', 0.5
    
    best_emotion = max(emotion_scores, key=emotion_scores.get)
    confidence = min(0.9, 0.3 + (emotion_scores[best_emotion] * 0.2))
    
    return best_emotion, confidence

# Main function to load model or use fallback
@st.cache_resource
def load_model_or_fallback():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import model_from_json
        from tensorflow.keras.preprocessing.text import Tokenizer
        from sklearn.preprocessing import LabelEncoder
        import os
        
        # Check if all required files exist
        required_files = ["train.txt", "model_architecture.json"]
        weight_files = ["model_weights.h5", "model_weights.weights.h5"]
        
        if not all(os.path.exists(f) for f in required_files):
            st.warning("⚠️ Model files not found. Using fallback keyword-based detection.")
            return None, None, None, None
        
        if not any(os.path.exists(f) for f in weight_files):
            st.warning("⚠️ Model weight files not found. Using fallback keyword-based detection.")
            return None, None, None, None
        
        # Load training data
        data = pd.read_csv("train.txt", sep=';')
        data.columns = ["Text", "Emotions"]
        
        texts = data["Text"].tolist()
        labels = data["Emotions"].tolist()
        
        # Setup tokenizer and encoder
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        max_length = max([len(seq) for seq in tokenizer.texts_to_sequences(texts)])
        
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(labels)
        
        # Load model
        with open("model_architecture.json", "r") as json_file:
            loaded_model_json = json_file.read()
        
        loaded_model = model_from_json(loaded_model_json)
        
        # Try loading weights
        for weight_file in weight_files:
            if os.path.exists(weight_file):
                try:
                    loaded_model.load_weights(weight_file)
                    st.success(f"✅ AI Model loaded successfully!")
                    return loaded_model, tokenizer, label_encoder, max_length
                except Exception as e:
                    continue
        
        st.warning("⚠️ Could not load model weights. Using fallback detection.")
        return None, None, None, None
        
    except Exception as e:
        st.warning(f"⚠️ TensorFlow not available or model loading failed. Using fallback detection. Error: {str(e)}")
        return None, None, None, None

# Define CSS styles (same as before but condensed)
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
.stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: 'Poppins', sans-serif; }
.main-header { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 2rem; margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.2); text-align: center; }
.main-header h1 { color: white; font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
.emotion-badge { display: inline-block; padding: 0.5rem 1rem; border-radius: 25px; font-weight: 600; font-size: 1.1rem; margin: 0.5rem; text-align: center; min-width: 120px; }
.emotion-joy { background: linear-gradient(45deg, #FFD700, #FFA500); color: #333; }
.emotion-sadness { background: linear-gradient(45deg, #4682B4, #1E90FF); color: white; }
.emotion-anger { background: linear-gradient(45deg, #FF6347, #DC143C); color: white; }
.emotion-fear { background: linear-gradient(45deg, #9370DB, #8A2BE2); color: white; }
.emotion-surprise { background: linear-gradient(45deg, #FF69B4, #FF1493); color: white; }
.emotion-love { background: linear-gradient(45deg, #FF69B4, #FF1493); color: white; }
.emotion-neutral { background: linear-gradient(45deg, #D3D3D3, #A9A9A9); color: #333; }
</style>
"""

# Emotion mappings
emotion_emojis = {
    "joy": "😊", "sadness": "😢", "anger": "😡", "fear": "😨", 
    "surprise": "😮", "love": "❤️", "neutral": "😐"
}

def main():
    st.set_page_config(page_title="AI Emotion Detector", page_icon="🎭", layout="wide")
    st.markdown(css, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 AI Emotion Detector</h1>
        <p>Powered by Deep Learning • Real-time Text Emotion Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model or use fallback
    loaded_model, tokenizer, label_encoder, max_length = load_model_or_fallback()
    
    # Determine which detection method to use
    if loaded_model is not None:
        detection_method = "🤖 AI Neural Network"
        st.info("✅ Using advanced AI model for emotion detection")
    else:
        detection_method = "🔤 Keyword-based Analysis"
        st.info("ℹ️ Using keyword-based fallback detection")
    
    st.markdown(f"**Detection Method:** {detection_method}")
    
    # Input section
    st.markdown("### 💬 Enter Your Text")
    input_text = st.text_area(
        "",
        placeholder="Type your message here... (e.g., 'I'm feeling great today!')",
        height=120
    )
    
    if st.button("🔍 Detect Emotion", type="primary"):
        if input_text.strip():
            with st.spinner("🤖 Analyzing emotions..."):
                time.sleep(0.5)
                
                if loaded_model is not None:
                    # Use AI model
                    try:
                        from tensorflow.keras.preprocessing.sequence import pad_sequences
                        input_sequence = tokenizer.texts_to_sequences([input_text])
                        padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
                        prediction = loaded_model.predict(padded_input_sequence, verbose=0)
                        predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
                        confidence = np.max(prediction[0])
                    except Exception as e:
                        st.error(f"AI model prediction failed: {str(e)}")
                        predicted_label, confidence = simple_emotion_detection(input_text)
                else:
                    # Use fallback detection
                    predicted_label, confidence = simple_emotion_detection(input_text)
                
                # Display result
                emoji = emotion_emojis.get(predicted_label, "🙂")
                emotion_class = f"emotion-{predicted_label.lower()}"
                
                st.markdown("### 🎯 Detection Result")
                st.markdown(f"""
                <div class="emotion-badge {emotion_class}">
                    {emoji} {predicted_label.upper()} ({confidence:.1%})
                </div>
                """, unsafe_allow_html=True)
                
                st.success(f"✅ Detected emotion: **{predicted_label}** with {confidence:.1%} confidence")
                
                # Simple confidence chart
                if loaded_model is not None:
                    try:
                        emotion_names = label_encoder.inverse_transform(np.arange(len(prediction[0])))
                        fig = go.Figure(data=[go.Bar(x=emotion_names, y=prediction[0])])
                        fig.update_layout(title="Confidence Scores", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass
        else:
            st.error("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 Built with Streamlit | 🎯 Emotion Recognition")
    if loaded_model is None:
        st.markdown("💡 **Note:** This is using a simplified keyword-based detection. For full AI capabilities, ensure all model files are properly deployed.")

if __name__ == "__main__":
    main()

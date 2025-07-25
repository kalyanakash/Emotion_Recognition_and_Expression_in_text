"""
Updated app.py with improved model loading and better emotion handling
"""
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
import json
import os

@st.cache_resource
def load_model_and_components():
    """Load model and preprocessing components with caching"""
    try:
        # Try to load improved model first
        if os.path.exists("model_config.json"):
            print("Loading improved model...")
            
            # Load model config
            with open("model_config.json", "r") as f:
                config = json.load(f)
            
            # Load model
            with open("model_architecture.json", "r") as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights("model_weights.h5")
            
            # Load tokenizer
            with open("tokenizer_config.json", "r") as f:
                tokenizer_config = json.load(f)
            
            tokenizer = Tokenizer(
                num_words=tokenizer_config['num_words'],
                oov_token=tokenizer_config['oov_token']
            )
            tokenizer.word_index = tokenizer_config['word_index']
            tokenizer.index_word = {int(k): v for k, v in tokenizer_config['index_word'].items()}
            
            # Load label encoder
            classes = np.load('label_classes.npy', allow_pickle=True)
            label_encoder = LabelEncoder()
            label_encoder.classes_ = classes
            
            max_length = config['max_length']
            
            return model, tokenizer, label_encoder, max_length, True
            
    except Exception as e:
        print(f"Could not load improved model: {e}")
    
    # Fallback to original model
    print("Loading original model...")
    
    # Load original data for tokenizer
    data = pd.read_csv("train.txt", sep=';')
    data.columns = ["Text", "Emotions"]
    
    texts = data["Text"].tolist()
    labels = data["Emotions"].tolist()
    
    # Original tokenizer setup
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    max_length = max([len(seq) for seq in tokenizer.texts_to_sequences(texts)])
    
    # Original label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    
    # Load original model
    with open("model_architecture.json", "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("model_weights.h5")
    
    return model, tokenizer, label_encoder, max_length, False

# Load model and components
try:
    loaded_model, tokenizer, label_encoder, max_length, is_improved = load_model_and_components()
    
    if is_improved:
        st.success("✅ Loaded improved emotion detection model!")
    else:
        st.info("ℹ️ Loaded original model. Run comprehensive_retrain.py for better accuracy.")
        
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Rest of the app code remains the same...
# (CSS, emoji mappings, functions, main app logic)

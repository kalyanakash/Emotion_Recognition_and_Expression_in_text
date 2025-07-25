"""
Analyze and retrain the emotion detection model with proper emotion mapping
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

def analyze_data():
    """Analyze the current training data"""
    print("📊 Analyzing training data...")
    
    # Load data
    data = pd.read_csv("train.txt", sep=';')
    data.columns = ["Text", "Emotions"]
    
    print(f"Total samples: {len(data)}")
    print("\n📈 Emotion distribution:")
    emotion_counts = data['Emotions'].value_counts()
    print(emotion_counts)
    
    print(f"\n🏷️ Unique emotions: {list(emotion_counts.index)}")
    
    return data, emotion_counts

def create_emotion_mapping():
    """Create proper emotion mapping"""
    # Original emotions in data vs desired emotions in app
    emotion_mapping = {
        'joy': 'happy',
        'sadness': 'sad', 
        'anger': 'anger',
        'fear': 'fear',
        'surprise': 'surprise',
        'love': 'love',
        'disgust': 'disgust'  # This might not be in data
    }
    
    return emotion_mapping

def preprocess_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

def create_improved_model(vocab_size, embedding_dim, max_length, num_classes):
    """Create an improved neural network model"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_improved_model():
    """Train an improved emotion detection model"""
    print("🚀 Starting model training...")
    
    # Analyze data
    data, emotion_counts = analyze_data()
    
    # Create emotion mapping
    emotion_mapping = create_emotion_mapping()
    
    # Map emotions to standard categories
    data['Mapped_Emotions'] = data['Emotions'].map(emotion_mapping)
    
    # Remove unmapped emotions
    data = data.dropna(subset=['Mapped_Emotions'])
    
    print(f"\n✅ After mapping: {len(data)} samples")
    print("📈 Mapped emotion distribution:")
    mapped_counts = data['Mapped_Emotions'].value_counts()
    print(mapped_counts)
    
    # Preprocess text
    print("\n🔧 Preprocessing text...")
    data['Cleaned_Text'] = data['Text'].apply(preprocess_text)
    
    # Remove empty texts
    data = data[data['Cleaned_Text'].str.len() > 0]
    
    # Prepare data
    texts = data['Cleaned_Text'].tolist()
    emotions = data['Mapped_Emotions'].tolist()
    
    # Tokenization
    print("🔤 Tokenizing text...")
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    # Convert to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    max_length = min(100, max([len(seq) for seq in sequences]))  # Cap at 100 for efficiency
    
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # Label encoding
    print("🏷️ Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_emotions = label_encoder.fit_transform(emotions)
    
    # Convert to categorical
    num_classes = len(label_encoder.classes_)
    categorical_emotions = tf.keras.utils.to_categorical(encoded_emotions, num_classes)
    
    print(f"📊 Classes: {list(label_encoder.classes_)}")
    print(f"📊 Number of classes: {num_classes}")
    print(f"📊 Vocabulary size: {len(tokenizer.word_index)}")
    print(f"📊 Max sequence length: {max_length}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, categorical_emotions, 
        test_size=0.2, random_state=42, stratify=encoded_emotions
    )
    
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Testing samples: {len(X_test)}")
    
    # Create model
    print("\n🧠 Creating improved model...")
    model = create_improved_model(
        vocab_size=len(tokenizer.word_index) + 1,
        embedding_dim=128,
        max_length=max_length,
        num_classes=num_classes
    )
    
    print("📋 Model architecture:")
    model.summary()
    
    # Train model
    print("\n🏋️ Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Predictions for classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(
        y_test_classes, y_pred_classes, 
        target_names=label_encoder.classes_
    ))
    
    # Save model
    print("\n💾 Saving improved model...")
    
    # Save architecture
    model_json = model.to_json()
    with open("improved_model_architecture.json", "w") as json_file:
        json_file.write(model_json)
    
    # Save weights
    model.save_weights("improved_model_weights.h5")
    
    # Save tokenizer and label encoder
    import pickle
    
    with open('improved_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('improved_label_encoder.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save max_length
    with open('improved_max_length.txt', 'w') as f:
        f.write(str(max_length))
    
    print("✅ Model training completed!")
    print("📁 Saved files:")
    print("  - improved_model_architecture.json")
    print("  - improved_model_weights.h5")
    print("  - improved_tokenizer.pickle")
    print("  - improved_label_encoder.pickle")
    print("  - improved_max_length.txt")
    
    # Test some examples
    print("\n🧪 Testing with sample texts...")
    test_texts = [
        "I am so happy today!",
        "This makes me very sad and depressed",
        "I am angry about this situation", 
        "I love you so much!",
        "This is absolutely disgusting",
        "I am scared and frightened",
        "What a big surprise this is!"
    ]
    
    for text in test_texts:
        # Preprocess
        cleaned_text = preprocess_text(text)
        
        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded_seq = pad_sequences(seq, maxlen=max_length, padding='post')
        
        # Predict
        prediction = model.predict(padded_seq, verbose=0)
        predicted_class = np.argmax(prediction[0])
        predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
        confidence = np.max(prediction[0])
        
        print(f"Text: '{text}' → Emotion: {predicted_emotion} ({confidence:.2%})")
    
    return model, tokenizer, label_encoder, max_length

if __name__ == "__main__":
    try:
        train_improved_model()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

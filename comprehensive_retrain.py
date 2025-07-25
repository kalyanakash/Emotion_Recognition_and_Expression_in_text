"""
Comprehensive Model Retraining Script for Better Emotion Detection
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
from sklearn.metrics import classification_report
import re
import json

def load_and_analyze_data():
    """Load and analyze the training data"""
    print("📊 Loading and analyzing data...")
    
    # Load training data
    train_data = pd.read_csv("train.txt", sep=';')
    train_data.columns = ["Text", "Emotions"]
    
    print(f"Training samples: {len(train_data)}")
    print("Emotion distribution:")
    emotion_counts = train_data['Emotions'].value_counts()
    print(emotion_counts)
    
    # Load test and validation data if available
    try:
        test_data = pd.read_csv("test.txt", sep=';')
        test_data.columns = ["Text", "Emotions"]
        print(f"Test samples: {len(test_data)}")
    except:
        print("No test.txt found")
        test_data = None
    
    try:
        val_data = pd.read_csv("val.txt", sep=';')
        val_data.columns = ["Text", "Emotions"]
        print(f"Validation samples: {len(val_data)}")
    except:
        print("No val.txt found")
        val_data = None
    
    return train_data, test_data, val_data, emotion_counts

def preprocess_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove extra whitespace and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_balanced_dataset(data):
    """Create a more balanced dataset"""
    print("⚖️ Balancing dataset...")
    
    # Get emotion counts
    emotion_counts = data['Emotions'].value_counts()
    min_samples = max(100, emotion_counts.min())  # At least 100 samples per emotion
    max_samples = min(2000, emotion_counts.median() * 2)  # Cap at reasonable number
    
    balanced_data = []
    
    for emotion in emotion_counts.index:
        emotion_data = data[data['Emotions'] == emotion]
        
        if len(emotion_data) > max_samples:
            # Downsample if too many
            emotion_data = emotion_data.sample(n=max_samples, random_state=42)
        elif len(emotion_data) < min_samples:
            # Upsample if too few (simple repetition)
            times_to_repeat = min_samples // len(emotion_data)
            remainder = min_samples % len(emotion_data)
            
            repeated_data = pd.concat([emotion_data] * times_to_repeat, ignore_index=True)
            if remainder > 0:
                additional_data = emotion_data.sample(n=remainder, random_state=42)
                repeated_data = pd.concat([repeated_data, additional_data], ignore_index=True)
            emotion_data = repeated_data
        
        balanced_data.append(emotion_data)
        print(f"{emotion}: {len(emotion_data)} samples")
    
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

def create_improved_model(vocab_size, embedding_dim, max_length, num_classes):
    """Create an improved LSTM model"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True),
        LSTM(64, dropout=0.3, recurrent_dropout=0.3),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Main training function"""
    print("🚀 Starting comprehensive model training...")
    
    # Load data
    train_data, test_data, val_data, emotion_counts = load_and_analyze_data()
    
    # Combine all available data
    all_data = [train_data]
    if test_data is not None:
        all_data.append(test_data)
    if val_data is not None:
        all_data.append(val_data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Total combined samples: {len(combined_data)}")
    
    # Remove duplicates
    combined_data = combined_data.drop_duplicates(subset=['Text'], keep='first')
    print(f"After removing duplicates: {len(combined_data)}")
    
    # Clean text
    print("🧹 Cleaning text data...")
    combined_data['Cleaned_Text'] = combined_data['Text'].apply(preprocess_text)
    
    # Remove empty texts
    combined_data = combined_data[combined_data['Cleaned_Text'].str.len() > 5]
    print(f"After removing short texts: {len(combined_data)}")
    
    # Balance dataset
    balanced_data = create_balanced_dataset(combined_data)
    print(f"Final balanced dataset: {len(balanced_data)}")
    
    # Prepare features and labels
    texts = balanced_data['Cleaned_Text'].tolist()
    emotions = balanced_data['Emotions'].tolist()
    
    # Create tokenizer
    print("🔤 Creating tokenizer...")
    tokenizer = Tokenizer(num_words=15000, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    # Convert to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    max_length = min(80, int(np.percentile([len(seq) for seq in sequences], 95)))
    print(f"Max sequence length: {max_length}")
    
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    # Encode labels
    print("🏷️ Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_emotions = label_encoder.fit_transform(emotions)
    num_classes = len(label_encoder.classes_)
    
    print(f"Emotion classes: {list(label_encoder.classes_)}")
    print(f"Number of classes: {num_classes}")
    
    # Convert to categorical
    categorical_emotions = tf.keras.utils.to_categorical(encoded_emotions, num_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, categorical_emotions,
        test_size=0.2, random_state=42, stratify=encoded_emotions
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    print("🧠 Creating model...")
    model = create_improved_model(
        vocab_size=len(tokenizer.word_index) + 1,
        embedding_dim=128,
        max_length=max_length,
        num_classes=num_classes
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    # Train model
    print("🏋️ Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("📊 Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predictions for detailed metrics
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print("\n📈 Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))
    
    # Save model and components
    print("💾 Saving model...")
    
    # Save model architecture
    model_json = model.to_json()
    with open("model_architecture.json", "w") as json_file:
        json_file.write(model_json)
    
    # Save weights
    model.save_weights("model_weights.h5")
    
    # Save tokenizer configuration
    tokenizer_config = {
        'num_words': tokenizer.num_words,
        'oov_token': tokenizer.oov_token,
        'word_index': tokenizer.word_index,
        'index_word': tokenizer.index_word
    }
    
    with open('tokenizer_config.json', 'w') as f:
        json.dump(tokenizer_config, f)
    
    # Save other components
    np.save('label_classes.npy', label_encoder.classes_)
    
    with open('model_config.json', 'w') as f:
        json.dump({
            'max_length': max_length,
            'vocab_size': len(tokenizer.word_index) + 1,
            'num_classes': num_classes,
            'emotion_classes': list(label_encoder.classes_)
        }, f)
    
    # Test with examples
    print("\n🧪 Testing with examples...")
    test_examples = [
        "I am extremely happy and joyful today!",
        "This makes me very sad and depressed",
        "I am so angry about this situation!",
        "I love this so much, it's amazing!",
        "This is absolutely disgusting and horrible",
        "I am scared and frightened by this",
        "What a wonderful surprise this is!",
        "I feel nothing about this situation"
    ]
    
    for text in test_examples:
        # Preprocess
        cleaned = preprocess_text(text)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded, verbose=0)
        predicted_class = np.argmax(prediction[0])
        emotion = label_encoder.classes_[predicted_class]
        confidence = np.max(prediction[0])
        
        print(f"'{text}' → {emotion} ({confidence:.2%})")
    
    print("\n✅ Model training completed successfully!")
    print("📁 Files saved:")
    print("  - model_architecture.json (updated)")
    print("  - model_weights.h5 (updated)")
    print("  - tokenizer_config.json")
    print("  - label_classes.npy")
    print("  - model_config.json")
    
    return model, tokenizer, label_encoder

if __name__ == "__main__":
    try:
        train_model()
        print("\n🎉 Your emotion detection model has been improved!")
        print("🚀 You can now run your Streamlit app with better accuracy!")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

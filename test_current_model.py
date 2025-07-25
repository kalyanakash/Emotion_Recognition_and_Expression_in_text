"""
Quick test of current model predictions
"""
import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load data and setup (same as in app.py)
data = pd.read_csv("train.txt", sep=';')
data.columns = ["Text", "Emotions"]

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

# Setup components
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
max_length = max([len(seq) for seq in tokenizer.texts_to_sequences(texts)])

label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Load model
with open("model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights("model_weights.h5")

print("🔍 Current Model Analysis")
print("=" * 40)
print(f"Available emotion classes: {list(label_encoder.classes_)}")
print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Max sequence length: {max_length}")
print(f"Vocabulary size: {len(tokenizer.word_index)}")

# Test predictions
test_texts = [
    "I am so happy today!",
    "This makes me very sad",
    "I am angry about this", 
    "I love you so much!",
    "This is disgusting",
    "I am scared",
    "What a surprise!"
]

print("\n🧪 Current Model Predictions:")
print("-" * 50)

for text in test_texts:
    # Process text
    input_sequence = tokenizer.texts_to_sequences([text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
    
    # Predict
    prediction = model.predict(padded_input_sequence, verbose=0)
    predicted_class = np.argmax(prediction[0])
    predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
    confidence = np.max(prediction[0])
    
    print(f"'{text}' → {predicted_emotion} ({confidence:.2%})")

print("\n📊 Training Data Emotion Distribution:")
emotion_counts = pd.Series(labels).value_counts()
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count} samples ({count/len(labels)*100:.1f}%)")

print(f"\nTotal training samples: {len(labels)}")

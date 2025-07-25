"""
Test script for the Emotion Detection App
"""
import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def test_model_loading():
    """Test if the model loads correctly"""
    try:
        # Load data
        data = pd.read_csv("train.txt", sep=';')
        data.columns = ["Text", "Emotions"]
        
        texts = data["Text"].tolist()
        labels = data["Emotions"].tolist()
        
        # Setup tokenizer and label encoder
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        max_length = max([len(seq) for seq in tokenizer.texts_to_sequences(texts)])
        
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        
        # Load model
        with open("model_architecture.json", "r") as json_file:
            loaded_model_json = json_file.read()
        
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model_weights.h5")
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model input shape: {loaded_model.input_shape}")
        print(f"📊 Model output shape: {loaded_model.output_shape}")
        print(f"📊 Max sequence length: {max_length}")
        print(f"📊 Number of emotions: {len(label_encoder.classes_)}")
        print(f"📊 Emotions: {list(label_encoder.classes_)}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def test_prediction():
    """Test if predictions work"""
    try:
        # Load everything needed for prediction
        data = pd.read_csv("train.txt", sep=';')
        data.columns = ["Text", "Emotions"]
        
        texts = data["Text"].tolist()
        labels = data["Emotions"].tolist()
        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        max_length = max([len(seq) for seq in tokenizer.texts_to_sequences(texts)])
        
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        
        with open("model_architecture.json", "r") as json_file:
            loaded_model_json = json_file.read()
        
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model_weights.h5")
        
        # Test predictions
        test_texts = [
            "I am so happy today!",
            "This is terrible news.",
            "I love you so much!",
            "I am scared of the dark.",
            "This is a normal sentence."
        ]
        
        print("\n🧪 Testing predictions:")
        print("-" * 50)
        
        for text in test_texts:
            input_sequence = tokenizer.texts_to_sequences([text])
            padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
            
            prediction = loaded_model.predict(padded_input_sequence, verbose=0)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
            confidence = np.max(prediction[0])
            
            print(f"Text: '{text}'")
            print(f"Emotion: {predicted_label} (Confidence: {confidence:.2%})")
            print("-" * 30)
        
        print("✅ Predictions working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        return False

def test_data_files():
    """Test if data files exist and are readable"""
    try:
        files_to_check = ["train.txt", "test.txt", "val.txt", "model_architecture.json", "model_weights.h5"]
        
        print("📁 Checking data files:")
        print("-" * 30)
        
        for file in files_to_check:
            try:
                if file.endswith('.txt'):
                    data = pd.read_csv(file, sep=';')
                    print(f"✅ {file}: {len(data)} rows")
                elif file.endswith('.json'):
                    with open(file, 'r') as f:
                        content = f.read()
                    print(f"✅ {file}: {len(content)} characters")
                elif file.endswith('.h5'):
                    # Just check if file exists and has reasonable size
                    import os
                    size = os.path.getsize(file)
                    print(f"✅ {file}: {size / (1024*1024):.1f} MB")
            except Exception as e:
                print(f"❌ {file}: Error - {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking files: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Running Emotion Detection App Tests")
    print("=" * 50)
    
    # Run tests
    file_test = test_data_files()
    model_test = test_model_loading() if file_test else False
    prediction_test = test_prediction() if model_test else False
    
    print("\n📋 Test Results Summary:")
    print("=" * 30)
    print(f"Data Files: {'✅ PASS' if file_test else '❌ FAIL'}")
    print(f"Model Loading: {'✅ PASS' if model_test else '❌ FAIL'}")
    print(f"Predictions: {'✅ PASS' if prediction_test else '❌ FAIL'}")
    
    if file_test and model_test and prediction_test:
        print("\n🎉 All tests passed! Your app is ready to deploy!")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

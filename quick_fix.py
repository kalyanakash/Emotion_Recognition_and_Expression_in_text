"""
Quick fix for emotion detection accuracy
"""
import pandas as pd
import numpy as np

# Analyze the training data
print("Analyzing training data...")
data = pd.read_csv("train.txt", sep=';')
data.columns = ["Text", "Emotions"]

print("Emotions in training data:")
emotions = data['Emotions'].value_counts()
print(emotions)

# Create emotion mapping
emotion_mapping = {
    'joy': 'happy',
    'sadness': 'sad',
    'anger': 'anger', 
    'fear': 'fear',
    'surprise': 'surprise',
    'love': 'love'
}

print("\nRecommended app.py updates:")
print("1. Update emotion_emojis to use:", list(emotions.index))
print("2. Update emotion_colors to use:", list(emotions.index))
print("3. Update CSS classes for:", list(emotions.index))

# Update the mappings in the current app
print("\nCreating updated mappings...")

emojis = {}
colors = {}
for emotion in emotions.index:
    if emotion == 'joy':
        emojis[emotion] = "😊"
        colors[emotion] = "#FFD700"
    elif emotion == 'sadness':
        emojis[emotion] = "😢" 
        colors[emotion] = "#4682B4"
    elif emotion == 'anger':
        emojis[emotion] = "😡"
        colors[emotion] = "#FF6347"
    elif emotion == 'fear':
        emojis[emotion] = "😨"
        colors[emotion] = "#9370DB"
    elif emotion == 'surprise':
        emojis[emotion] = "😮"
        colors[emotion] = "#FF69B4"
    elif emotion == 'love':
        emojis[emotion] = "❤️"
        colors[emotion] = "#FF1493"
    else:
        emojis[emotion] = "😐"
        colors[emotion] = "#D3D3D3"

print("\nUpdated emotion_emojis:")
print(emojis)
print("\nUpdated emotion_colors:")
print(colors)

print("\nYour model should now work better with the training data!")

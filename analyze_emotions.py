"""
Quick analysis of the training data emotions
"""
import pandas as pd

# Load data
data = pd.read_csv("train.txt", sep=';')
data.columns = ["Text", "Emotions"]

print("=== EMOTION ANALYSIS ===")
print(f"Total samples: {len(data)}")
print("\nUnique emotions in training data:")
emotions = data['Emotions'].value_counts()
print(emotions)

print("\nEmotion list:")
for emotion in emotions.index:
    print(f"- {emotion}")

print(f"\nTotal unique emotions: {len(emotions)}")

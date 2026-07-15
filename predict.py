import numpy as np
from tensorflow.keras.models import load_model

# Load model once
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_emotion_model.keras")

model = load_model(MODEL_PATH)

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


def predict_emotion(face):

    # Add batch dimension
    face = np.expand_dims(face, axis=0)

    # Predict probabilities
    prediction = model.predict(face, verbose=0)[0]

    # Get predicted class
    emotion_index = np.argmax(prediction)

    emotion = emotion_labels[emotion_index]

    confidence = float(prediction[emotion_index])

    # Create a dictionary of all probabilities
    all_probabilities = {
        emotion_labels[i]: round(float(prediction[i] * 100), 2)
        for i in range(len(emotion_labels))
    }

    return emotion, confidence, all_probabilities
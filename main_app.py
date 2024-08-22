import streamlit as st
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("C:/Users/yigitc/Desktop/Sesli Komut ilk Sprint 4/command_recognition_model.keras")

# Parameters
n_mfcc = 13
duration = 2  # Recording duration in seconds
sampling_rate = 16000  # Sampling rate

def record_audio(duration, fs):
    st.write("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.write("Recording complete.")
    return audio.flatten()

def extract_features_from_audio(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def predict_command_from_mic():
    # Record audio
    audio = record_audio(duration, sampling_rate)
    
    # Extract features
    features = extract_features_from_audio(audio, sampling_rate).reshape(1, n_mfcc, 1)
    
    # Make prediction
    prediction = model.predict(features)
    top_two_indices = prediction[0].argsort()[-2:][::-1]
    top_prediction = prediction[0][top_two_indices[0]]
    second_best_prediction = prediction[0][top_two_indices[1]]
    
    predicted_command = ["Arka Kapi Ac", "Arka Kapi Kapat", "Unknown"][top_two_indices[0]]
    second_best_command = ["Arka Kapi Ac", "Arka Kapi Kapat", "Unknown"][top_two_indices[1]]

    st.write(f"Predicted Command: {predicted_command}")
    st.write(f"Confidence Score: {top_prediction:.2f}")
    st.write(f"Second Best Command: {second_best_command} with Confidence: {second_best_prediction:.2f}")

    if top_prediction - second_best_prediction < 0.1:
        st.write("The model is not confident. Please try again.")
    
    return predicted_command, top_prediction

st.title("Voice Command Recognition")

if st.button("Record and Predict Command"):
    predict_command_from_mic()

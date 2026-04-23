import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os 

# --- CONFIGURATION ---
st.set_page_config(page_title="SER Real-Time Interface", layout="centered")
SAMPLE_RATE = 22050
DURATION = 3.0

# --- UI HEADER ---
st.title("🎙️ Real-Time Emotion Recognition")
st.markdown("Capture live audio and extract acoustic features (MFCCs).")

# --- AUDIO CAPTURE FUNCTION ---
def record_audio(duration, fs):
    st.info(f"Recording for {duration} seconds... Please speak now.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    
    progress = st.progress(0)
    for i in range(100):
        sd.sleep(int(duration * 10)) 
        progress.progress(i + 1)
        
    sd.wait()
    st.success("Audio captured!")
    return np.squeeze(recording)

# --- MAIN LOGIC ---
if st.button("Start Live Capture"):
    audio_data = record_audio(DURATION, SAMPLE_RATE)
    st.write("Extracting MFCC features...")
    mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=13)
    
    st.subheader("Captured Audio Waveform")
    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(audio_data, sr=SAMPLE_RATE, ax=ax, color="blue")
    ax.set(title='Amplitude vs Time')
    st.pyplot(fig)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Emotion", "Neutral (Placeholder)")
    col2.metric("Confidence Score", "85%")

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
st.markdown("Capture live audio or upload a RAVDESS dataset file to extract acoustic features.")

# --- AUDIO CAPTURE FUNCTION (Remains Unchanged) ---
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

# --- SIDEBAR CONTROLS (New Addition) ---
st.sidebar.title("⚙️ Settings")
input_mode = st.sidebar.radio("Select Audio Source:", ["Live Microphone", "Upload Audio File"])

# --- MAIN ROUTING LOGIC (The Rewritten Part) ---
audio_data = None
sample_rate = SAMPLE_RATE 

if input_mode == "Live Microphone":
    if st.button("Start Live Capture"):
        audio_data = record_audio(DURATION, SAMPLE_RATE)
        
elif input_mode == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload a .wav file (e.g., from RAVDESS)", type=["wav"])
    if uploaded_file is not None:
        # Load the uploaded file using librosa
        audio_data, sample_rate = librosa.load(uploaded_file, sr=SAMPLE_RATE)
        st.success(f"Loaded {uploaded_file.name} successfully!")

# --- PROCESSING & VISUALIZATION (Runs for both mic and upload) ---
if audio_data is not None:
    st.write("Extracting MFCC features...")
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    
    st.subheader("Audio Waveform")
    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax, color="blue")
    ax.set(title='Amplitude vs Time')
    st.pyplot(fig)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Emotion", "Neutral (Placeholder)")
    col2.metric("Confidence Score", "85%")

import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="SER Research Dashboard", layout="wide")

# --- PARAMETERS ---
SAMPLE_RATE = 22050
DURATION = 3.0

# --- SIDEBAR: MODEL ROUTER ---
st.sidebar.title("🤖 Model Router")
selected_model = st.sidebar.selectbox(
    "Choose Active Brain:",
    ["SVM (Baseline)", "Conformer (SOTA)", "CNN", "LSTM"]
)
st.sidebar.info(f"Currently routing to: {selected_model}")

# --- FUNCTION: RECORD AUDIO ---
def record_audio(duration, fs):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    progress = st.progress(0)
    for i in range(100):
        sd.sleep(int(duration * 10))
        progress.progress(i + 1)
    sd.wait()
    return np.squeeze(recording)

# --- MAIN UI ---
st.title("🎙️ Speech Emotion Recognition Research Suite")
tab_inference, tab_analysis, tab_map = st.tabs(["🎯 Prediction", "📊 Acoustic Analysis", "🗺️ Data Map"])

audio_data = None

# --- TAB 1: INFERENCE ---
with tab_inference:
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.subheader("Input Stream")
        input_mode = st.radio("Source:", ["Microphone", "RAVDESS File"])
        
        if input_mode == "Microphone":
            if st.button("🔴 Start Recording"):
                audio_data = record_audio(DURATION, SAMPLE_RATE)
        else:
            uploaded_file = st.file_uploader("Upload .wav", type=["wav"])
            if uploaded_file:
                audio_data, _ = librosa.load(uploaded_file, sr=SAMPLE_RATE)

    with col_output:
        st.subheader("Model Output")
        if audio_data is not None:
            # Placeholder for actual model.predict()
            st.metric("Predicted Emotion", "Happy (Mock)")
            st.progress(0.85, text="Confidence Score: 85%")
        else:
            st.write("Awaiting audio input...")

# --- TAB 2: ANALYSIS (VIOLIN PLOTS) ---
with tab_analysis:
    st.header("Acoustic Fingerprint (Violin Plots)")
    if audio_data is not None:
        # Extracting a feature for the plot (e.g., Pitch/F0)
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=SAMPLE_RATE)
        avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # MOCK PLOT: Comparing current pitch to dataset
        fig, ax = plt.subplots(figsize=(10, 4))
        # This simulates comparing the 'User' to the 'RAVDESS' distribution
        mock_data = pd.DataFrame({
            'Emotion': ['User'] * 10 + ['Happy'] * 50 + ['Sad'] * 50,
            'Pitch': [avg_pitch] * 10 + list(np.random.normal(300, 50, 50)) + list(np.random.normal(150, 30, 50))
        })
        sns.violinplot(data=mock_data, x='Emotion', y='Pitch', palette="muted", ax=ax)
        st.pyplot(fig)
        
    else:
        st.info("Record audio to see how your pitch compares to the dataset.")

# --- TAB 3: DATA MAP (t-SNE) ---
with tab_map:
    st.header("Dimensionality Reduction (t-SNE)")
    st.write("This map shows where your voice sits relative to the 1,440 RAVDESS samples.")
    
    # Simulate the t-SNE plot
    if audio_data is not None:
        st.info("Generating t-SNE projection...")
        # In the final version, your teammate will provide a CSV of these points
        chart_data = pd.DataFrame(np.random.randn(100, 2), columns=['x', 'y'])
        st.scatter_chart(chart_data, x='x', y='y')
        
    else:
        st.write("Record audio to project your voice into the cluster map.")

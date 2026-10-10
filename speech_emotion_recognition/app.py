from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from audio_utils import extract_mfcc

MODEL_PATH = Path("models/speech_emotion_cnn_lstm.keras")
LABELS_PATH = Path("models/labels.json")

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")
st.title("🎙️ Speech Emotion Recognition")
st.write(
    "Upload a short speech recording. The app extracts MFCC features and uses a "
    "CNN-LSTM model to estimate the emotion expressed in the audio."
)

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return model, labels

model, labels = load_artifacts()

if model is None:
    st.warning("The trained model was not found. First place the RAVDESS dataset in data/ravdess/ and run `python train.py`.")
    st.stop()

uploaded = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])
if uploaded is not None:
    st.audio(uploaded)

    if st.button("Recognize emotion", type="primary"):
        suffix = Path(uploaded.name).suffix or ".wav"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(uploaded.getvalue())
                temp_path = temp.name

            features = extract_mfcc(temp_path)
            probabilities = model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
            best_index = int(np.argmax(probabilities))
            st.subheader(f"Predicted emotion: **{labels[best_index].capitalize()}**")
            st.caption("This is a model estimate, not a definitive reading of a person's feelings.")

            order = np.argsort(probabilities)[::-1]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar([labels[i].capitalize() for i in order], [float(probabilities[i]) for i in order])
            ax.set_ylabel("Model probability")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Could not process this audio file: {exc}")
        finally:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass

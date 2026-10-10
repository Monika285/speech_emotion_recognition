from __future__ import annotations

import numpy as np
import librosa

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 174

EMOTION_NAMES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

def extract_mfcc(audio_path: str) -> np.ndarray:
    """Return a fixed-size (MAX_FRAMES, N_MFCC) MFCC matrix."""
    signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if signal.size == 0:
        raise ValueError("The uploaded audio contains no readable samples.")
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=256,
    ).T

    # Pad short clips and truncate long clips to a consistent sequence length.
    if mfcc.shape[0] < MAX_FRAMES:
        pad = MAX_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:MAX_FRAMES, :]

    # Per-clip normalization helps reduce scale differences.
    mean = mfcc.mean(axis=0, keepdims=True)
    std = mfcc.std(axis=0, keepdims=True) + 1e-6
    return ((mfcc - mean) / std).astype(np.float32)

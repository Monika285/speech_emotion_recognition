from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from audio_utils import extract_mfcc, EMOTION_NAMES, MAX_FRAMES, N_MFCC

DATA_DIR = Path("data/ravdess/Audio_Song_Actors_01-24")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "speech_emotion_cnn_lstm.keras"
LABELS_PATH = MODEL_DIR / "labels.json"
SEED = 42

def emotion_from_ravdess_filename(path: Path) -> str | None:
    # Example: 03-01-05-01-02-01-12.wav; third field is the emotion code.
    parts = path.stem.split("-")
    if len(parts) < 3:
        return None
    return EMOTION_NAMES.get(parts[2])

def load_dataset():
    paths = sorted(DATA_DIR.rglob("*.wav"))
    if not paths:
        raise FileNotFoundError(
            f"No .wav files found under {DATA_DIR.resolve()}. "
            "Download/extract RAVDESS Audio-only into data/ravdess/."
        )

    features, labels = [], []
    skipped = 0
    for path in paths:
        emotion = emotion_from_ravdess_filename(path)
        if emotion is None:
            skipped += 1
            continue
        try:
            features.append(extract_mfcc(str(path)))
            labels.append(emotion)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            skipped += 1

    if not features:
        raise RuntimeError("No valid RAVDESS WAV files could be processed.")

    X = np.stack(features)
    y = np.asarray(labels)
    print(f"Loaded {len(X)} clips; skipped {skipped}.")
    print("Class counts:", {name: int((y == name).sum()) for name in sorted(set(y))})
    return X, y

def build_model(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(MAX_FRAMES, N_MFCC), name="mfcc_sequence")
    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.LSTM(96, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="emotion_probabilities")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, raw_y = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_y)

    # Stratified random split is a basic baseline; speaker-independent evaluation
    # is recommended for a more realistic estimate of generalization.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )

    model = build_model(len(encoder.classes_))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_loss", save_best_only=True
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    probs = model.predict(X_test, verbose=0)
    predictions = probs.argmax(axis=1)
    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(
        y_test, predictions, labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_, zero_division=0
    ))

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f, indent=2)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved labels to {LABELS_PATH}")

if __name__ == "__main__":
    main()

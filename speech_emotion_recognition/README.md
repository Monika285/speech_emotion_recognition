# Speech Emotion Recognition (MFCC + CNN-LSTM)

A beginner-friendly speech emotion recognition project using:
- **Dataset:** RAVDESS
- **Signal processing:** Librosa + MFCC (Mel-Frequency Cepstral Coefficients)
- **Deep learning:** CNN + LSTM (CNN learns local audio patterns; LSTM learns sequence patterns)
- **Interface:** Streamlit web app

## 1. Project structure

```text
speech_emotion_recognition/
├── app.py
├── train.py
├── audio_utils.py
├── requirements.txt
├── README.md
├── data/
│   └── ravdess/          # Put extracted RAVDESS audio here
└── models/               # Model and label encoder are saved here after training
```

## 2. Install Python

Use Python 3.10 or 3.11 for best compatibility with TensorFlow.

```bash
python -m venv .venv
```

Windows:
```bat
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

## 3. Get the RAVDESS dataset

Download the **RAVDESS Audio-only** dataset from its official research page:
https://zenodo.org/records/1188976

Extract the actor folders inside `data/ravdess/`. The folder should look similar to:

```text
data/ravdess/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_02/
└── ...
```

The script reads WAV files recursively. RAVDESS filenames encode emotion in the third hyphen-separated field. This project maps the eight RAVDESS emotion labels:
01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised.

## 4. Train the model

From the project folder:

```bash
python train.py
```

Training extracts 40 MFCC coefficients per audio file, pads/truncates each feature sequence to a fixed length, and trains a CNN-LSTM classifier. It saves:
- `models/speech_emotion_cnn_lstm.keras`
- `models/labels.json`

Training can take time depending on the number of files and your computer. A GPU is optional.

## 5. Run the web app

```bash
streamlit run app.py
```

Upload a WAV, MP3, or other supported audio file. The app displays the predicted emotion and class probabilities.

## 6. Important notes

- This is a learning/demo model, not a clinical or reliable measure of someone's actual emotional state.
- Performance depends on dataset quality, speakers, language, recording conditions, and training settings.
- For a fair evaluation, test on speakers not used during training; a random file split can overestimate real-world performance.

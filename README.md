# 🎙️ VoiceForge — Voice Cloning Studio

A complete, zero-shot voice conversion system that clones the timbre and pitch
characteristics of any target speaker using spectral envelope matching.
No GPU or model training required.

---

## 📐 Architecture Overview

```
Source Audio ──► Preprocess ──► STFT ──► Spectral Map ──► ISTFT ──► Pitch Shift ──► Formant Shift ──► Output
                                           ▲
Target Audio ──► Preprocess ──► Feature Extraction ──────┘
```

### Conversion Pipeline

| Step | Description |
|------|-------------|
| **Load** | Reads WAV/MP3/FLAC/OGG/M4A, resamples to 22 050 Hz mono |
| **Denoise** | Spectral noise gate — estimates noise floor from quietest 5 % of frames |
| **Normalise** | Peak-normalise to –1 dBFS, trim leading/trailing silence |
| **Feature Extract** | MFCC stats, mel-spectrogram mean, F0 (pitch) statistics |
| **Spectral Envelope Map** | Per-frequency-bin amplitude ratio transfer from target to source |
| **Pitch Conversion** | Auto F0 matching + user-defined semitone shift |
| **Formant Shift** | Vocal-tract length approximation via resampling + pitch correction |
| **Reconstruct** | Griffin-Lim / iSTFT waveform reconstruction |

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
git clone <repo-url>
cd voice-clone
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.9+ recommended.

### 3. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🖥️ Web Interface

1. Upload a **Source Voice** (the speaker you want to transform)
2. Upload a **Target Voice** (the voice style you want to clone)
3. Adjust **Pitch Shift** (−12 to +12 semitones)
4. Adjust **Formant Scale** (0.7× darker → 1.4× brighter)
5. Click **⚡ Convert Voice**
6. Listen to the preview and **⬇ Download** the result

---

## ⌨️ CLI Usage

```bash
# Basic conversion
python cli.py --source source.wav --target target.wav

# With pitch and formant control
python cli.py \
  --source  speaker_a.wav  \
  --target  speaker_b.wav  \
  --output  result.wav     \
  --pitch   -3             \
  --formant 0.9
```

---

## 📁 Project Structure

```
voice-clone/
├── app.py                # Flask backend
├── voice_converter.py    # Core conversion engine
├── cli.py                # Command-line interface
├── requirements.txt
├── model/                # (optional) neural vocoder weights
├── uploads/              # Temporary input files
├── outputs/              # Converted audio files
└── templates/
    └── index.html        # Web UI
```

---

## ⚙️ Configuration

Edit constants at the top of `voice_converter.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SR` | 22050 | Sample rate (Hz) |
| `N_FFT` | 1024 | FFT window size |
| `HOP_LENGTH` | 256 | STFT hop length |
| `N_MELS` | 80 | Mel filterbank bins |
| `N_MFCC` | 40 | MFCC coefficients |

---

## 🔬 Tips for Best Results

- **Longer clips (5–30 s)** give better feature statistics
- **Clean, dry recordings** (minimal reverb/background noise) convert best
- Target clips with **similar speaking cadence** to source improve naturalness
- Use the **Pitch Shift** to compensate for large gender differences
- Use **Formant Scale < 1.0** for a deeper voice, > 1.0 for a brighter voice

---

## 🔧 Extending with a Neural Vocoder

The default Griffin-Lim reconstruction is fast but can sound slightly metallic.
To upgrade to a HiFi-GAN vocoder:

```bash
pip install torch torchaudio
# Download HiFi-GAN checkpoint
wget https://github.com/jik876/hifi-gan/releases/download/v1/generator_v1
mv generator_v1 model/
```

Then swap `reconstruct_waveform()` in `voice_converter.py` with a HiFi-GAN
inference call.

---

## 📄 License

MIT — free for personal and commercial use.

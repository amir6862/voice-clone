<div align="center">

# 🎙️ VoiceForge

### Zero-Shot Voice Cloning & Conversion Studio

**Convert any voice into any other voice — no GPU, no training, no setup headaches.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](https://github.com/yourusername/voiceforge/pulls)

</div>

---

## 📖 Table of Contents

- [What Is This?](#-what-is-this)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Running in VS Code](#-running-in-vs-code)
- [Web Interface](#-web-interface)
- [CLI Usage](#-cli-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Tips for Best Results](#-tips-for-best-results)
- [Extending with a Neural Vocoder](#-extending-with-a-neural-vocoder)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 What Is This?

**VoiceForge** is a complete, zero-shot voice conversion system. You give it two audio files:

| Input | Description |
|---|---|
| 🎙️ **Source Voice** | The original speaker — the audio you want to transform |
| 🔊 **Target Voice** | The voice to clone — whose style you want to copy |

VoiceForge outputs a new audio file that sounds like the **source speaker's words** delivered in the **target speaker's voice**.

No GPU required. No model training. Works on any laptop.

---

## ⚙️ How It Works

VoiceForge uses a classical DSP + machine learning hybrid pipeline:

```
Source Audio ──► Preprocess ──► STFT ──► Spectral Map ──► iSTFT ──► Pitch Shift ──► Formant Shift ──► Output
                                              ▲
Target Audio ──► Preprocess ──► Feature Extraction ──────┘
```

### Step-by-Step Pipeline

| # | Step | What Happens |
|---|------|-------------|
| 1 | **Load** | Reads WAV / MP3 / FLAC / OGG / M4A, resamples to 22 050 Hz mono |
| 2 | **Denoise** | Spectral noise gate — estimates noise floor from the quietest 5% of frames and suppresses it |
| 3 | **Normalise** | Peak-normalise to −1 dBFS, trim leading / trailing silence |
| 4 | **Feature Extract** | Extracts MFCC statistics, mel-spectrogram mean, and F0 (fundamental frequency) stats from both speakers |
| 5 | **Spectral Envelope Map** | Transfers the target speaker's per-frequency timbral profile onto the source signal via amplitude ratio matching |
| 6 | **Pitch Conversion** | Auto F0 matching between speakers + user-defined semitone shift |
| 7 | **Formant Shift** | Approximates vocal-tract length differences via resampling + pitch correction |
| 8 | **Reconstruct** | iSTFT (or Griffin-Lim) waveform reconstruction and final normalisation |

### Why No Training?

This approach is **zero-shot** — it extracts a statistical timbral fingerprint from the target voice at inference time and uses it to re-colour the source spectrogram. This means:

- ✅ Works with any two speakers instantly
- ✅ Runs on CPU in seconds
- ✅ No data collection or fine-tuning needed
- ⚠️ For maximum naturalness, a neural vocoder upgrade is recommended (see [Extending](#-extending-with-a-neural-vocoder))

---

## ✨ Features

- 🎙️ **Upload any audio** — WAV, MP3, FLAC, OGG, M4A all supported
- 🔊 **Zero-shot voice conversion** — no training, no GPU needed
- 🎚️ **Pitch control** — shift ±12 semitones on top of auto F0 matching
- 🔬 **Formant control** — adjust vocal-tract character (0.7× darker → 1.4× brighter)
- 📊 **Waveform visualisation** — interactive waveform rendered in the browser
- ⬇️ **Downloadable output** — get your converted WAV in one click
- 🖥️ **Web UI + CLI** — use the browser interface or script it from the terminal
- 🧹 **Built-in denoising** — spectral noise gate applied automatically
- 📈 **Conversion stats** — source/target/output duration, F0 values, processing time

---

## 🚀 Quick Start

### Prerequisites

- Python **3.9 or higher** — [Download here](https://www.python.org/downloads/)
- `pip` (comes with Python)
- ~500 MB disk space for dependencies

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/voiceforge.git
cd voiceforge
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First install may take 3–5 minutes — PyTorch and librosa are sizeable packages.

### 4. Run the app

```bash
python app.py
```

### 5. Open in your browser

```
http://127.0.0.1:5000
```

---

## 💻 Running in VS Code

1. **Open the folder** — `File → Open Folder` → select the `voiceforge` folder
2. **Open the terminal** — press `` Ctrl+` `` (backtick)
3. **Create & activate a virtual environment** (see step 2 above)
4. **Install dependencies** — `pip install -r requirements.txt`
5. **Run the app** — `python app.py`
6. **Open your browser** at `http://127.0.0.1:5000`

**Recommended VS Code Extensions:**

| Extension | Purpose |
|---|---|
| [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) | Syntax highlighting, IntelliSense |
| [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) | Type checking, autocomplete |

> When VS Code prompts *"Select interpreter"*, choose the `venv` Python — it will use the correct packages automatically.

---

## 🖥️ Web Interface

The web UI guides you through the full conversion in 4 steps:

```
01 — Audio Sources    →   Upload source + target audio files
02 — Controls         →   Set pitch shift and formant scale
03 — Processing       →   Live progress through each pipeline step
04 — Output           →   Waveform preview, audio player, download button
```

**Supported upload formats:** `.wav` `.mp3` `.flac` `.ogg` `.m4a`

**Controls:**

| Control | Range | Default | Effect |
|---|---|---|---|
| Pitch Shift | −12 to +12 st | 0 | Raises or lowers pitch in semitones on top of auto F0 matching |
| Formant Scale | 0.70× to 1.40× | 1.0× | < 1.0 = deeper/darker voice, > 1.0 = brighter/lighter voice |

---

## ⌨️ CLI Usage

For batch processing or scripting, use `cli.py` directly:

```bash
# Basic usage
python cli.py --source source.wav --target target.wav

# Full options
python cli.py \
  --source  recordings/alex.wav  \
  --target  recordings/emma.wav  \
  --output  results/alex_as_emma.wav \
  --pitch   -3                   \
  --formant 0.9
```

**All flags:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--source` | path | *(required)* | Source speaker audio file |
| `--target` | path | *(required)* | Target speaker audio file |
| `--output` | path | `output.wav` | Where to save the result |
| `--pitch` | int | `0` | Extra pitch shift in semitones |
| `--formant` | float | `1.0` | Formant scale factor |

**Example output:**

```
╔══════════════════════════════════════╗
║         VoiceForge CLI               ║
╠══════════════════════════════════════╣
║  Source  : recordings/alex.wav       ║
║  Target  : recordings/emma.wav       ║
║  Output  : results/alex_as_emma.wav  ║
║  Pitch   : -3                        ║
║  Formant : 0.9                       ║
╚══════════════════════════════════════╝

📊 Conversion Stats
────────────────────────────────────
  source_duration_s        12.4
  target_duration_s        28.7
  output_duration_s        12.4
  src_f0_mean_hz           142.3
  tgt_f0_mean_hz           221.8
  pitch_shift_st           -3
  formant_shift            0.9
  processing_time_s        4.2

✅  Saved to: results/alex_as_emma.wav
```

---

## 📁 Project Structure

```
voiceforge/
│
├── app.py                  # Flask web server & API routes
├── voice_converter.py      # Core DSP + conversion engine
├── cli.py                  # Command-line interface
├── requirements.txt        # Python dependencies
├── README.md
│
├── model/                  # (optional) neural vocoder weights
│   └── README.txt
│
├── uploads/                # Temporary input files (auto-created)
├── outputs/                # Converted audio output (auto-created)
│
└── templates/
    └── index.html          # Web UI (single-file, no build step)
```

---

## ⚙️ Configuration

Fine-tune the DSP parameters by editing the constants at the top of `voice_converter.py`:

```python
SR          = 22050   # Sample rate in Hz
N_FFT       = 1024    # FFT window size (larger = more frequency resolution)
HOP_LENGTH  = 256     # STFT hop length (smaller = more time resolution)
N_MELS      = 80      # Mel filterbank bins
N_MFCC      = 40      # MFCC coefficients used for feature matching
```

| Tweak | Effect |
|---|---|
| Increase `N_FFT` to `2048` | Sharper frequency resolution, slower processing |
| Decrease `HOP_LENGTH` to `128` | More temporal detail, larger memory usage |
| Increase `N_MFCC` to `60` | More speaker detail captured, slightly slower |

---

## 🔬 Tips for Best Results

| Tip | Why It Helps |
|---|---|
| Use **10–30 second** clips | Longer clips produce more stable feature statistics |
| Record in a **quiet room** | The denoiser helps but clean source is always better |
| Use **dry audio** (no reverb) | Reverb confuses the spectral envelope extractor |
| Source and target in **same language** | Pitch and formant stats are language-influenced |
| For **gender conversion (M→F)** | Try pitch `+5` semitones, formant `1.15×` |
| For **gender conversion (F→M)** | Try pitch `-5` semitones, formant `0.85×` |
| For an **older/deeper** sound | Use formant scale `< 0.9` |
| For a **younger/brighter** sound | Use formant scale `> 1.1` |

---

## 🔧 Extending with a Neural Vocoder

The default **Griffin-Lim** waveform reconstruction is fast but can introduce slight metallic artefacts on some voices. For production-quality output, upgrade to **HiFi-GAN**:

### Step 1 — Install PyTorch (if not already)

```bash
pip install torch torchaudio
```

### Step 2 — Download HiFi-GAN weights

```bash
wget https://github.com/jik876/hifi-gan/releases/download/v1/generator_v1 -O model/generator_v1
```

### Step 3 — Swap the vocoder

Replace the `reconstruct_waveform()` call in `voice_converter.py` with a HiFi-GAN inference call.
Full integration guide: [jik876/hifi-gan](https://github.com/jik876/hifi-gan)

### Vocoder Comparison

| Vocoder | Quality | Speed | Notes |
|---|---|---|---|
| Griffin-Lim | ★★★☆☆ | ⚡ Very fast | Built-in, no setup |
| HiFi-GAN V1 | ★★★★★ | ⚡ Fast | Best quality/speed balance |
| WaveGlow | ★★★★☆ | 🐢 Slower | Requires NVIDIA GPU |
| BigVGAN | ★★★★★ | ⚡ Fast | State of the art, larger model |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `python: command not found` | Use `python3` instead, or verify Python is in your PATH |
| `pip install` fails on `librosa` | Run `pip install --upgrade pip` first, then retry |
| `Port 5000 already in use` | Change `port=5000` to `port=5001` at the bottom of `app.py` |
| Conversion output sounds robotic | Use longer target clips (30+ seconds) and clean recordings |
| `ModuleNotFoundError` | Make sure your virtual environment is activated before running |
| App crashes on large files | Default limit is 50 MB — raise `MAX_CONTENT_LENGTH` in `app.py` |
| Very slow first run | Normal — librosa compiles numba JIT functions on first launch |
| macOS: `SSL` errors on install | Run `pip install certifi` and retry |

---

## 🤝 Contributing

Contributions are very welcome! Here's how to get started:

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/yourusername/voiceforge.git
cd voiceforge

# 2. Create and activate a virtual environment
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a branch for your feature
git checkout -b feature/my-improvement

# 5. Make your changes, then commit and push
git commit -m "Add: my improvement"
git push origin feature/my-improvement

# 6. Open a Pull Request on GitHub
```

**Ideas for contributions:**

- 🔊 HiFi-GAN vocoder integration
- 🎤 Real-time microphone streaming conversion
- 📦 Batch CLI processing (convert multiple files at once)
- 🐳 Docker / docker-compose setup
- 🧪 Unit tests for `voice_converter.py`
- 📊 Speaker similarity score metric (cosine distance of MFCC embeddings)
- 🌐 Multi-language UI

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it for personal and commercial purposes.

---

## 🙏 Acknowledgements

- [librosa](https://librosa.org/) — Audio analysis and feature extraction
- [SoundFile](https://python-soundfile.readthedocs.io/) — Audio I/O
- [SciPy](https://scipy.org/) — Signal processing utilities
- [Flask](https://flask.palletsprojects.com/) — Lightweight web framework
- [HiFi-GAN](https://github.com/jik876/hifi-gan) — Neural vocoder (optional upgrade path)

---

<div align="center">

Made with ❤️ and 🎙️

⭐ **Star this repo if you found it useful!**

</div>

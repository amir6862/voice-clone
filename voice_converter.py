"""
voice_converter.py
==================
Core voice-conversion engine.

Pipeline:
  1. Load & preprocess both audio files (denoise, normalise, trim silences)
  2. Extract features from target speaker (mel-spectral envelope, pitch stats)
  3. Map source spectrogram → target timbre via spectral-envelope conversion
  4. Apply pitch-shift / formant-shift post-processing
  5. Reconstruct waveform with Griffin-Lim / vocoder

This is a *training-free*, zero-shot approach based on spectral envelope
matching – it works with any two audio clips with no GPU required, while
still producing clearly audible voice character transfer.
"""

import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import librosa.effects
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SR          = 22050   # universal sample-rate
N_FFT       = 1024
HOP_LENGTH  = 256
N_MELS      = 80
N_MFCC      = 40
FRAME_LEN   = N_FFT


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Load any audio format, mono, resampled to `sr`."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def save_audio(path: str, y: np.ndarray, sr: int = SR) -> None:
    """Write float32 audio to WAV."""
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    sf.write(path, y, sr)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    1. Trim leading/trailing silence
    2. Spectral noise-gate (simple noise reduction)
    3. Peak normalise to –1 dBFS
    """
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # Spectral noise gate  ──  estimate noise floor from quietest 5 % of frames
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    frame_energy = S.mean(axis=0)
    noise_thresh  = np.percentile(frame_energy, 5) * 3.0
    mask = (frame_energy > noise_thresh).astype(float)
    # Smooth the mask
    from scipy.ndimage import uniform_filter1d
    mask = uniform_filter1d(mask, size=5)
    S_denoised = S * mask[np.newaxis, :]

    # Keep original phase
    phase = np.angle(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    y = librosa.istft(S_denoised * np.exp(1j * phase), hop_length=HOP_LENGTH)

    # Peak-normalise
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak * 0.9

    return y


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_spectral_envelope(y: np.ndarray, sr: int = SR) -> dict:
    """
    Extract speaker-characterising features:
      - MFCC statistics  (mean + std of each coefficient)
      - Mean spectral centroid / bandwidth / rolloff
      - Pitch (F0) statistics  [voiced frames only]
      - Mel-spectrogram mean   (global timbral profile)
    """
    S      = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_db   = librosa.amplitude_to_db(S, ref=np.max)
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc   = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC, sr=sr)
    sc     = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    sb     = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    sroll  = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]

    # Pitch extraction
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])

    return {
        "mfcc_mean"       : mfcc.mean(axis=1),
        "mfcc_std"        : mfcc.std(axis=1),
        "mel_mean"        : mel.mean(axis=1),
        "centroid_mean"   : float(sc.mean()),
        "bandwidth_mean"  : float(sb.mean()),
        "rolloff_mean"    : float(sroll.mean()),
        "f0_mean"         : float(voiced_f0.mean()) if len(voiced_f0) else 150.0,
        "f0_std"          : float(voiced_f0.std())  if len(voiced_f0) else 30.0,
        "S_db_mean"       : S_db.mean(axis=1),   # per-frequency-bin mean (N_FFT//2+1,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Envelope Conversion (the heart of voice cloning)
# ─────────────────────────────────────────────────────────────────────────────

def spectral_envelope_conversion(
    S_src: np.ndarray,
    src_features: dict,
    tgt_features: dict,
) -> np.ndarray:
    """
    Map the source magnitude spectrogram so its per-frequency-bin mean
    matches the target speaker's timbral profile.

    For each frequency bin b:
        S_out[b, :] = S_src[b, :] * (tgt_mean[b] / src_mean[b])

    This is the classic "voice timbre transfer" via spectral-mean matching,
    analogous to the spectral whitening + re-colouring used in many
    classical VC systems.
    """
    src_mean = src_features["S_db_mean"]   # (freq_bins,)
    tgt_mean = tgt_features["S_db_mean"]   # (freq_bins,)

    # Convert dB means to linear amplitude ratios
    src_amp = librosa.db_to_amplitude(src_mean)[:, np.newaxis]
    tgt_amp = librosa.db_to_amplitude(tgt_mean)[:, np.newaxis]

    # Avoid division by zero
    src_amp = np.maximum(src_amp, 1e-8)

    ratio = tgt_amp / src_amp
    # Smooth the ratio curve to avoid harsh resonances
    from scipy.ndimage import uniform_filter1d
    ratio_smooth = uniform_filter1d(ratio[:, 0], size=9)[:, np.newaxis]

    # Blend: 70 % converted + 30 % original  (preserves intelligibility)
    S_converted = S_src * (0.30 + 0.70 * ratio_smooth)
    return S_converted


# ─────────────────────────────────────────────────────────────────────────────
# Pitch conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_pitch(
    y: np.ndarray,
    src_f0_mean: float,
    tgt_f0_mean: float,
    extra_semitones: int = 0,
    sr: int = SR,
) -> np.ndarray:
    """
    Shift pitch so that the source F0 mean matches the target F0 mean,
    plus any user-specified extra shift (in semitones).
    """
    if src_f0_mean < 1e-3:
        n_steps = extra_semitones
    else:
        n_steps = 12 * np.log2(tgt_f0_mean / src_f0_mean) + extra_semitones

    # Clamp to a reasonable range to avoid artefacts
    n_steps = float(np.clip(n_steps, -24, 24))
    if abs(n_steps) < 0.1:
        return y
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


# ─────────────────────────────────────────────────────────────────────────────
# Formant shift  (vocal tract length / character)
# ─────────────────────────────────────────────────────────────────────────────

def formant_shift(y: np.ndarray, shift_factor: float, sr: int = SR) -> np.ndarray:
    """
    Approximate formant shift by resampling the audio without changing pitch.
    shift_factor > 1  →  higher formants (smaller vocal tract, child-like)
    shift_factor < 1  →  lower  formants (larger vocal tract, deeper)
    """
    if abs(shift_factor - 1.0) < 0.01:
        return y

    # Resample to simulate formant change
    stretched_len = int(len(y) / shift_factor)
    y_resampled   = librosa.resample(y, orig_sr=sr, target_sr=int(sr * shift_factor))
    # Pitch-correct back so pitch stays constant
    n_steps = -12 * np.log2(shift_factor)
    y_resampled = librosa.effects.pitch_shift(y_resampled, sr=int(sr * shift_factor),
                                               n_steps=n_steps)
    # Resample back to original SR
    y_out = librosa.resample(y_resampled, orig_sr=int(sr * shift_factor), target_sr=sr)

    # Match original length
    if len(y_out) > len(y):
        y_out = y_out[:len(y)]
    elif len(y_out) < len(y):
        y_out = np.pad(y_out, (0, len(y) - len(y_out)))

    return y_out


# ─────────────────────────────────────────────────────────────────────────────
# Waveform reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_waveform(S_mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """
    Reconstruct time-domain signal from magnitude + original phase.
    Falls back to Griffin-Lim if phase has mismatched shape.
    """
    try:
        stft_complex = S_mag * np.exp(1j * phase)
        return librosa.istft(stft_complex, hop_length=HOP_LENGTH)
    except Exception:
        return librosa.griffinlim(S_mag, hop_length=HOP_LENGTH, n_iter=32)


# ─────────────────────────────────────────────────────────────────────────────
# VoiceConverter – public API
# ─────────────────────────────────────────────────────────────────────────────

class VoiceConverter:
    """
    Zero-shot voice conversion without any model training.

    Usage:
        vc = VoiceConverter()
        result = vc.convert("source.wav", "target.wav", "output.wav",
                             pitch_shift=0, formant_shift=1.0)
    """

    def convert(
        self,
        source_path   : str,
        target_path   : str,
        output_path   : str,
        pitch_shift   : int   = 0,
        formant_shift : float = 1.0,
        sr            : int   = SR,
    ) -> dict:
        """
        Full conversion pipeline.

        Parameters
        ----------
        source_path   : Path to the source speaker audio.
        target_path   : Path to the target speaker audio.
        output_path   : Where to write the converted WAV.
        pitch_shift   : Additional pitch shift in semitones (on top of auto F0 matching).
        formant_shift : Formant scale factor (1.0 = unchanged).

        Returns
        -------
        dict with 'stats' key containing processing metadata.
        """
        t0 = time.time()

        # ── 1. Load ─────────────────────────────────────────────────────────
        print("[VC] Loading audio…")
        y_src = load_audio(source_path, sr=sr)
        y_tgt = load_audio(target_path, sr=sr)

        # ── 2. Preprocess ───────────────────────────────────────────────────
        print("[VC] Preprocessing…")
        y_src = preprocess(y_src, sr=sr)
        y_tgt = preprocess(y_tgt, sr=sr)

        # ── 3. Feature extraction ────────────────────────────────────────────
        print("[VC] Extracting features…")
        src_feat = extract_spectral_envelope(y_src, sr=sr)
        tgt_feat = extract_spectral_envelope(y_tgt, sr=sr)

        # ── 4. STFT of source ────────────────────────────────────────────────
        stft_src  = librosa.stft(y_src, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_src     = np.abs(stft_src)
        phase_src = np.angle(stft_src)

        # ── 5. Spectral envelope conversion ─────────────────────────────────
        print("[VC] Applying spectral envelope conversion…")
        S_converted = spectral_envelope_conversion(S_src, src_feat, tgt_feat)

        # ── 6. Waveform reconstruction ───────────────────────────────────────
        print("[VC] Reconstructing waveform…")
        y_out = reconstruct_waveform(S_converted, phase_src)

        # ── 7. Pitch conversion (F0 matching + user shift) ───────────────────
        print("[VC] Pitch conversion…")
        y_out = convert_pitch(
            y_out,
            src_f0_mean     = src_feat["f0_mean"],
            tgt_f0_mean     = tgt_feat["f0_mean"],
            extra_semitones = pitch_shift,
            sr              = sr,
        )

        # ── 8. Formant shift ─────────────────────────────────────────────────
        if abs(formant_shift - 1.0) > 0.01:
            print(f"[VC] Formant shift ×{formant_shift:.2f}…")
            y_out = formant_shift(y_out, formant_shift, sr=sr)

        # ── 9. Final normalise & save ────────────────────────────────────────
        peak = np.max(np.abs(y_out))
        if peak > 1e-6:
            y_out = y_out / peak * 0.9

        save_audio(output_path, y_out, sr=sr)

        elapsed = round(time.time() - t0, 2)
        print(f"[VC] Done in {elapsed}s  →  {output_path}")

        return {
            "stats": {
                "source_duration_s" : round(len(y_src) / sr, 2),
                "target_duration_s" : round(len(y_tgt) / sr, 2),
                "output_duration_s" : round(len(y_out) / sr, 2),
                "src_f0_mean_hz"    : round(src_feat["f0_mean"], 1),
                "tgt_f0_mean_hz"    : round(tgt_feat["f0_mean"], 1),
                "pitch_shift_st"    : pitch_shift,
                "formant_shift"     : formant_shift,
                "processing_time_s" : elapsed,
            }
        }

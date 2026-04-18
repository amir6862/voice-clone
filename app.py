"""
Voice Cloning Web Application
==============================
A Flask-based voice conversion system using spectral mapping.
Converts source speaker audio into target speaker's voice style.

Run: python app.py
"""

import os
import uuid
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template

from voice_converter import VoiceConverter

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB limit

UPLOAD_FOLDER  = Path("uploads")
OUTPUT_FOLDER  = Path("outputs")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

converter = VoiceConverter()

# ── Helpers ─────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def save_upload(file_storage) -> Path:
    """Save an uploaded FileStorage object with a unique name."""
    ext  = Path(file_storage.filename).suffix.lower()
    name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_FOLDER / name
    file_storage.save(dest)
    return dest


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    """
    Expects multipart/form-data with:
      - source_audio   : audio file (source speaker)
      - target_audio   : audio file (target speaker / voice to clone)
      - pitch_shift    : int, semitones  (optional, default 0)
      - formant_shift  : float           (optional, default 1.0)
    """
    try:
        # ── Validate files ──────────────────────────────────────────────────
        if "source_audio" not in request.files or "target_audio" not in request.files:
            return jsonify({"error": "Both source_audio and target_audio are required"}), 400

        src_file = request.files["source_audio"]
        tgt_file = request.files["target_audio"]

        for f in (src_file, tgt_file):
            if f.filename == "":
                return jsonify({"error": "Empty filename"}), 400
            if not allowed_file(f.filename):
                return jsonify({"error": f"Unsupported file type: {f.filename}"}), 400

        # ── Conversion params ───────────────────────────────────────────────
        pitch_shift   = int(request.form.get("pitch_shift",   0))
        formant_shift = float(request.form.get("formant_shift", 1.0))

        # ── Save uploads ────────────────────────────────────────────────────
        src_path = save_upload(src_file)
        tgt_path = save_upload(tgt_file)

        # ── Run conversion ──────────────────────────────────────────────────
        out_name = f"converted_{uuid.uuid4().hex[:8]}.wav"
        out_path = OUTPUT_FOLDER / out_name

        result = converter.convert(
            source_path   = str(src_path),
            target_path   = str(tgt_path),
            output_path   = str(out_path),
            pitch_shift   = pitch_shift,
            formant_shift = formant_shift,
        )

        return jsonify({
            "success"      : True,
            "output_file"  : out_name,
            "download_url" : f"/download/{out_name}",
            "stats"        : result.get("stats", {}),
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/download/<filename>")
def download(filename: str):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route("/stream/<filename>")
def stream(filename: str):
    """Stream audio for the in-browser player."""
    return send_from_directory(OUTPUT_FOLDER, filename)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Voice Cloning App  –  http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)

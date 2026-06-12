"""
RunPod serverless handler for XTTS v2 (text-to-speech).

Input schema:
  {
    "text":           str  (required — text to synthesise, max ~400 chars for quality),
    "language":       str  (default "en", see SUPPORTED_LANGUAGES below),
    "speaker_wav":    str  (base64 WAV — reference voice for cloning, optional),
    "speaker_name":   str  (one of the built-in speakers if no wav provided),
    "speed":          float (default 1.0)
  }

Output:
  {
    "audio":   str  (base64 WAV, 22050 Hz mono),
    "format":  "wav",
    "sample_rate": 22050
  }
"""

import os
import base64
import tempfile
import time
from io import BytesIO

import runpod
import torch
import soundfile as sf
from TTS.api import TTS

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/xtts")
HF_TOKEN = os.environ.get("HF_TOKEN")

SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
]

os.environ["COQUI_TOS_AGREED"] = "1"

print(f"[xtts] Loading XTTS v2 …")
_t0 = time.time()

# If network volume has weights, point TTS at them; otherwise use default HF cache
if os.path.isdir(MODEL_PATH):
    tts = TTS(model_path=MODEL_PATH, config_path=os.path.join(MODEL_PATH, "config.json")).to("cuda")
else:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

print(f"[xtts] Model ready in {time.time() - _t0:.1f}s")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    job_input: dict = job.get("input", {})

    text: str = job_input.get("text", "").strip()
    if not text:
        return {"error": "Missing required field: text"}

    language: str = job_input.get("language", "en").lower()
    if language not in SUPPORTED_LANGUAGES:
        return {"error": f"Unsupported language '{language}'. Choose from: {SUPPORTED_LANGUAGES}"}

    speaker_wav_b64: str | None = job_input.get("speaker_wav")
    speaker_name: str | None = job_input.get("speaker_name")
    speed: float = float(job_input.get("speed", 1.0))

    # Write reference wav to a temp file if provided
    ref_wav_path = None
    if speaker_wav_b64:
        wav_data = base64.b64decode(speaker_wav_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_data)
            ref_wav_path = tmp.name

    # Output temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_tmp:
        out_path = out_tmp.name

    print(f"[xtts] Synthesising: lang={language}, chars={len(text)}, speaker_wav={bool(ref_wav_path)}")

    try:
        tts.tts_to_file(
            text=text,
            language=language,
            speaker_wav=ref_wav_path,
            speaker=speaker_name if not ref_wav_path else None,
            speed=speed,
            file_path=out_path,
        )
    finally:
        if ref_wav_path and os.path.exists(ref_wav_path):
            os.unlink(ref_wav_path)

    audio_data, sample_rate = sf.read(out_path)
    os.unlink(out_path)

    buf = BytesIO()
    sf.write(buf, audio_data, sample_rate, format="WAV")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "audio": encoded,
        "format": "wav",
        "sample_rate": sample_rate,
        "duration_seconds": round(len(audio_data) / sample_rate, 2),
    }


runpod.serverless.start({"handler": handler})

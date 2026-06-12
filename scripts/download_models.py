#!/usr/bin/env python3
"""Download model weights to /runpod-volume/models/. Run inside a RunPod pod."""

import os
from huggingface_hub import snapshot_download

# Keep HF cache on container disk — not the network volume (saves ~60+ GB)
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

TOKEN = os.environ.get("HF_TOKEN") or None
BASE = "/runpod-volume/models"

print("[1/3] Downloading FLUX.1-schnell...")
snapshot_download(
    "black-forest-labs/FLUX.1-schnell",
    local_dir=f"{BASE}/flux",
    token=TOKEN,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)

print("[2/3] Downloading LTX-Video (diffusers files only)...")
snapshot_download(
    "Lightricks/LTX-Video",
    local_dir=f"{BASE}/ltxvideo",
    token=TOKEN,
    allow_patterns=[
        "model_index.json",
        "*.json",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "vae/*",
        "transformer/*",
    ],
)

print("[3/3] Downloading XTTS v2...")
snapshot_download(
    "coqui/XTTS-v2",
    local_dir=f"{BASE}/xtts",
    token=TOKEN,
)

print("All models downloaded.")

"""
Download LTX-Video weights to the network volume or local cache.
"""

import os
from huggingface_hub import snapshot_download

MODEL_REPO = "Lightricks/LTX-Video"
MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/ltxvideo")
HF_TOKEN = os.environ.get("HF_TOKEN")


def fetch():
    print(f"Downloading {MODEL_REPO} → {MODEL_PATH}")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_PATH,
        token=HF_TOKEN,
    )
    print("Download complete.")


if __name__ == "__main__":
    fetch()

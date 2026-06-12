"""
Run once at build time (BAKE_MODELS=1) or at worker cold-start
to download FLUX.1 weights to the network volume or local cache.
"""

import os
from huggingface_hub import snapshot_download

MODEL_REPO = "black-forest-labs/FLUX.1-schnell"
# Override with FLUX.1-dev if you have a dev license
MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/flux")
HF_TOKEN = os.environ.get("HF_TOKEN")


def fetch():
    print(f"Downloading {MODEL_REPO} → {MODEL_PATH}")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_PATH,
        token=HF_TOKEN,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    print("Download complete.")


if __name__ == "__main__":
    fetch()

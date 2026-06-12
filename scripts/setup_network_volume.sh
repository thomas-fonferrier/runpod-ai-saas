#!/usr/bin/env bash
# =============================================================================
# setup_network_volume.sh
#
# One-time setup: creates a RunPod Network Volume and downloads all model
# weights onto it using a temporary CPU pod.
#
# Prerequisites:
#   - runpodctl installed  (brew install runpod/runpodctl/runpodctl)
#   - RUNPOD_API_KEY exported
#   - HF_TOKEN exported (for gated models)
# =============================================================================

set -euo pipefail

# --- Config ------------------------------------------------------------------
VOLUME_NAME="${VOLUME_NAME:-ai-model-storage}"
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-100}"
DATACENTER="${DATACENTER:-US-TX-3}"   # change to your preferred data center
# -----------------------------------------------------------------------------

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "ERROR: RUNPOD_API_KEY is not set."
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Gated models (FLUX.1-dev) may fail."
fi

echo "==> Creating Network Volume: $VOLUME_NAME (${VOLUME_SIZE_GB} GB) in $DATACENTER"
runpodctl create volume \
  --name "$VOLUME_NAME" \
  --size "$VOLUME_SIZE_GB" \
  --datacenter "$DATACENTER"

VOLUME_ID=$(runpodctl get volume --name "$VOLUME_NAME" --field id)
echo "==> Volume created: $VOLUME_ID"

echo "==> Spinning up a temporary pod to download model weights..."
POD_ID=$(runpodctl create pod \
  --name "model-downloader" \
  --image "runpod/base:0.6.2-cuda12.1.0" \
  --gpu-type "NVIDIA GeForce RTX 3090" \
  --volume-id "$VOLUME_ID" \
  --volume-mount-path "/runpod-volume" \
  --env "HF_TOKEN=${HF_TOKEN}" \
  --field id)

echo "==> Pod started: $POD_ID — waiting for it to be ready..."
runpodctl wait pod "$POD_ID" --status RUNNING --timeout 300

echo "==> Downloading models (this may take 20–40 minutes)..."
runpodctl exec "$POD_ID" -- bash -c "
  pip install -q huggingface_hub &&
  python -c \"
from huggingface_hub import snapshot_download
import os
tok = os.environ.get('HF_TOKEN')

print('[1/3] Downloading FLUX.1-schnell...')
snapshot_download('black-forest-labs/FLUX.1-schnell',
  local_dir='/runpod-volume/models/flux', token=tok,
  ignore_patterns=['*.msgpack','*.h5','flax_model*'])

print('[2/3] Downloading LTX-Video...')
snapshot_download('Lightricks/LTX-Video',
  local_dir='/runpod-volume/models/ltxvideo', token=tok)

print('[3/3] Downloading XTTS v2...')
snapshot_download('coqui/XTTS-v2',
  local_dir='/runpod-volume/models/xtts', token=tok)

print('All models downloaded.')
\"
"

echo "==> Terminating the temporary pod..."
runpodctl remove pod "$POD_ID"

echo ""
echo "=== DONE ==================================================================="
echo "  Network Volume ID : $VOLUME_ID"
echo "  Volume Name       : $VOLUME_NAME"
echo ""
echo "  Use this volume when creating your serverless endpoints."
echo "  Attach it at mount path: /runpod-volume"
echo "============================================================================"

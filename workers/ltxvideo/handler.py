"""
RunPod serverless handler for LTX-Video (image-to-video).

Input schema:
  {
    "prompt":               str   (required — describes the motion/scene),
    "image":                str   (base64 PNG/JPG — the conditioning frame),
    "negative_prompt":      str   (optional),
    "num_frames":           int   (default 25, must be 8N+1 for LTX),
    "num_inference_steps":  int   (default 50),
    "guidance_scale":       float (default 3.0),
    "fps":                  int   (default 24),
    "width":                int   (default 768),
    "height":               int   (default 512),
    "seed":                 int   (optional)
  }

Output:
  {
    "video":  str   (base64-encoded MP4),
    "format": "mp4",
    "seed":   int,
    "fps":    int
  }
"""

import os
import base64
import tempfile
import time
from io import BytesIO

import runpod
import torch
import imageio
import numpy as np
from PIL import Image
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import load_image

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/ltxvideo")
HF_TOKEN = os.environ.get("HF_TOKEN")

_model_source = MODEL_PATH if os.path.isdir(MODEL_PATH) else "Lightricks/LTX-Video"

print(f"[ltxvideo] Loading model from {_model_source} …")
_t0 = time.time()

pipe = LTXImageToVideoPipeline.from_pretrained(
    _model_source,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)
pipe.enable_model_cpu_offload()
print(f"[ltxvideo] Model ready in {time.time() - _t0:.1f}s")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def _decode_image(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(BytesIO(data)).convert("RGB")


def handler(job: dict) -> dict:
    job_input: dict = job.get("input", {})

    prompt: str = job_input.get("prompt", "")
    image_b64: str | None = job_input.get("image")

    if not prompt:
        return {"error": "Missing required field: prompt"}
    if not image_b64:
        return {"error": "Missing required field: image (base64-encoded)"}

    negative_prompt: str = job_input.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted")
    num_frames: int = int(job_input.get("num_frames", 25))
    steps: int = int(job_input.get("num_inference_steps", 50))
    guidance: float = float(job_input.get("guidance_scale", 3.0))
    fps: int = int(job_input.get("fps", 24))
    width: int = int(job_input.get("width", 768))
    height: int = int(job_input.get("height", 512))
    seed: int | None = job_input.get("seed")

    # LTX-Video requires num_frames = 8N + 1
    num_frames = max(9, num_frames)
    if (num_frames - 1) % 8 != 0:
        num_frames = ((num_frames - 1) // 8) * 8 + 1

    generator = None
    if seed is not None:
        generator = torch.Generator("cpu").manual_seed(int(seed))
    else:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator("cpu").manual_seed(seed)

    conditioning_image = _decode_image(image_b64).resize((width, height))

    print(f"[ltxvideo] Generating: frames={num_frames}, {width}x{height}, fps={fps}, seed={seed}")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=conditioning_image,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    )
    frames = result.frames[0]  # list of PIL images

    # Encode to MP4 in memory
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()
    os.unlink(tmp_path)

    encoded = base64.b64encode(video_bytes).decode("utf-8")

    return {
        "video": encoded,
        "format": "mp4",
        "seed": seed,
        "fps": fps,
        "num_frames": len(frames),
    }


runpod.serverless.start({"handler": handler})

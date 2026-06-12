"""
RunPod serverless handler for FLUX.1 (text-to-image).

Input schema:
  {
    "prompt":               str   (required),
    "width":                int   (default 1024),
    "height":               int   (default 1024),
    "num_inference_steps":  int   (default 4  for schnell, 50 for dev),
    "guidance_scale":       float (default 0.0 for schnell, 3.5 for dev),
    "seed":                 int   (optional, for reproducibility),
    "output_format":        str   "png" | "jpg" (default "png")
  }

Output:
  {
    "image":  str   (base64-encoded image),
    "format": str,
    "seed":   int
  }
"""

import os
import base64
import time
from io import BytesIO

import runpod
import torch
from diffusers import FluxPipeline
from PIL import Image

# ---------------------------------------------------------------------------
# Model loading — happens once at worker startup (warm phase)
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/flux")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Fall back to HF hub if local weights are not yet present
_model_source = MODEL_PATH if os.path.isdir(MODEL_PATH) else "black-forest-labs/FLUX.1-schnell"

print(f"[flux] Loading model from {_model_source} …")
_t0 = time.time()

pipe = FluxPipeline.from_pretrained(
    _model_source,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)
pipe.enable_model_cpu_offload()  # moves layers to CPU when not in use → saves VRAM
print(f"[flux] Model ready in {time.time() - _t0:.1f}s")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    job_input: dict = job.get("input", {})

    prompt: str = job_input.get("prompt", "")
    if not prompt:
        return {"error": "Missing required field: prompt"}

    width: int = int(job_input.get("width", 1024))
    height: int = int(job_input.get("height", 1024))
    steps: int = int(job_input.get("num_inference_steps", 4))
    guidance: float = float(job_input.get("guidance_scale", 0.0))
    fmt: str = job_input.get("output_format", "png").lower()
    seed: int | None = job_input.get("seed")

    generator = None
    if seed is not None:
        generator = torch.Generator("cpu").manual_seed(int(seed))
    else:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator("cpu").manual_seed(seed)

    print(f"[flux] Generating: prompt={prompt!r}, {width}x{height}, steps={steps}, seed={seed}")

    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    image: Image.Image = result.images[0]

    buf = BytesIO()
    pil_fmt = "JPEG" if fmt == "jpg" else "PNG"
    image.save(buf, format=pil_fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "image": encoded,
        "format": fmt,
        "seed": seed,
        "width": width,
        "height": height,
    }


runpod.serverless.start({"handler": handler})

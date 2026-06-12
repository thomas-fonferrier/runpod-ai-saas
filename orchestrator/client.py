"""
Thin wrapper around the RunPod SDK for submitting jobs and polling results.
Each model has a dedicated endpoint; this client handles all three.
"""

import asyncio
import time
from typing import Any

import runpod

from config import settings

runpod.api_key = settings.RUNPOD_API_KEY

# Map model name → endpoint ID
_ENDPOINTS: dict[str, str] = {
    "flux": settings.FLUX_ENDPOINT_ID,
    "ltxvideo": settings.LTXVIDEO_ENDPOINT_ID,
    "xtts": settings.XTTS_ENDPOINT_ID,
}


class RunPodError(Exception):
    def __init__(self, model: str, status: str, message: str):
        super().__init__(f"[{model}] Job {status}: {message}")
        self.model = model
        self.status = status


async def run_job(model: str, job_input: dict[str, Any]) -> dict[str, Any]:
    """Submit a job to a RunPod endpoint and wait for the result (async)."""
    endpoint_id = _ENDPOINTS.get(model)
    if not endpoint_id:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(_ENDPOINTS)}")

    endpoint = runpod.Endpoint(endpoint_id)

    # Submit
    run = endpoint.run({"input": job_input})
    job_id = run.job_id
    print(f"[orchestrator] Job submitted — model={model}, job_id={job_id}")

    # Poll until complete
    deadline = time.monotonic() + settings.MAX_WAIT
    while True:
        status = run.status()

        if status == "COMPLETED":
            output = run.output()
            print(f"[orchestrator] Job {job_id} COMPLETED")
            return output

        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RunPodError(model, status, f"job_id={job_id}")

        if time.monotonic() > deadline:
            raise TimeoutError(f"[{model}] Job {job_id} exceeded {settings.MAX_WAIT}s timeout")

        await asyncio.sleep(settings.POLL_INTERVAL)


async def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 4,
    guidance_scale: float = 0.0,
    seed: int | None = None,
    output_format: str = "png",
) -> dict[str, Any]:
    """Call FLUX.1 to generate an image from text."""
    return await run_job("flux", {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "output_format": output_format,
    })


async def generate_video(
    prompt: str,
    image_b64: str,
    negative_prompt: str = "",
    num_frames: int = 25,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    fps: int = 24,
    width: int = 768,
    height: int = 512,
    seed: int | None = None,
) -> dict[str, Any]:
    """Call LTXVideo to generate a video from an image + prompt."""
    return await run_job("ltxvideo", {
        "prompt": prompt,
        "image": image_b64,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "fps": fps,
        "width": width,
        "height": height,
        "seed": seed,
    })


async def generate_speech(
    text: str,
    language: str = "en",
    speaker_wav_b64: str | None = None,
    speaker_name: str | None = None,
    speed: float = 1.0,
) -> dict[str, Any]:
    """Call XTTS v2 to synthesise speech from text."""
    return await run_job("xtts", {
        "text": text,
        "language": language,
        "speaker_wav": speaker_wav_b64,
        "speaker_name": speaker_name,
        "speed": speed,
    })

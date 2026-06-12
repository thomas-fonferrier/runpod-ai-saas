"""
Orchestrator — FastAPI service that exposes three endpoints:

  POST /generate/image    → FLUX.1  (text → image)
  POST /generate/video    → LTXVideo (image → video)
  POST /generate/speech   → XTTS v2 (text → audio)

Deploy this as a serverless function (Cloud Run, Lambda, Fly.io, etc.)
or run it locally with:  uvicorn main:app --reload
"""

from __future__ import annotations

import base64
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import client
from client import RunPodError

app = FastAPI(
    title="AI Generation Orchestrator",
    description="Unified API for FLUX.1, LTXVideo and XTTS v2 on RunPod",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ImageRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the image to generate")
    width: int = Field(1024, ge=256, le=2048, multiple_of=64)
    height: int = Field(1024, ge=256, le=2048, multiple_of=64)
    num_inference_steps: int = Field(4, ge=1, le=100)
    guidance_scale: float = Field(0.0, ge=0.0, le=20.0)
    seed: int | None = Field(None, description="Set for reproducible results")
    output_format: str = Field("png", pattern="^(png|jpg)$")


class ImageResponse(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    format: str
    seed: int
    width: int
    height: int


class VideoRequest(BaseModel):
    prompt: str = Field(..., description="Motion / scene description")
    image: str = Field(..., description="Base64-encoded PNG/JPG conditioning frame")
    negative_prompt: str = Field("", description="What to avoid in the video")
    num_frames: int = Field(25, ge=9, le=257)
    num_inference_steps: int = Field(50, ge=1, le=100)
    guidance_scale: float = Field(3.0, ge=0.0, le=20.0)
    fps: int = Field(24, ge=1, le=60)
    width: int = Field(768, ge=256, le=1280, multiple_of=32)
    height: int = Field(512, ge=256, le=720, multiple_of=32)
    seed: int | None = None


class VideoResponse(BaseModel):
    video: str = Field(..., description="Base64-encoded MP4")
    format: str
    seed: int
    fps: int
    num_frames: int


class SpeechRequest(BaseModel):
    text: str = Field(..., description="Text to synthesise (max ~400 chars recommended)")
    language: str = Field("en", description="BCP-47 language code, e.g. 'en', 'fr', 'de'")
    speaker_wav: str | None = Field(None, description="Base64 WAV reference voice for cloning")
    speaker_name: str | None = Field(None, description="Built-in XTTS speaker name (if no wav)")
    speed: float = Field(1.0, ge=0.5, le=2.0)


class SpeechResponse(BaseModel):
    audio: str = Field(..., description="Base64-encoded WAV")
    format: str
    sample_rate: int
    duration_seconds: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate/image", response_model=ImageResponse)
async def generate_image(req: ImageRequest):
    try:
        result = await client.generate_image(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            output_format=req.output_format,
        )
    except RunPodError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    return ImageResponse(**result)


@app.post("/generate/video", response_model=VideoResponse)
async def generate_video(req: VideoRequest):
    try:
        result = await client.generate_video(
            prompt=req.prompt,
            image_b64=req.image,
            negative_prompt=req.negative_prompt,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            fps=req.fps,
            width=req.width,
            height=req.height,
            seed=req.seed,
        )
    except RunPodError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    return VideoResponse(**result)


@app.post("/generate/speech", response_model=SpeechResponse)
async def generate_speech(req: SpeechRequest):
    try:
        result = await client.generate_speech(
            text=req.text,
            language=req.language,
            speaker_wav_b64=req.speaker_wav,
            speaker_name=req.speaker_name,
            speed=req.speed,
        )
    except RunPodError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    return SpeechResponse(**result)


# ---------------------------------------------------------------------------
# Lambda / Cloud Run entry point (optional)
# ---------------------------------------------------------------------------
# To deploy as an AWS Lambda function, use Mangum:
#   pip install mangum
#   from mangum import Mangum
#   handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

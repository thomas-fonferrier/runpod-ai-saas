# Base: RunPod CUDA image with Python 3.11
FROM runpod/base:0.6.2-cuda12.1.0

# System deps for Pillow / torch
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY handler.py .
COPY model_fetcher.py .

# Pre-download model weights if no network volume is attached
# (set BAKE_MODELS=1 at build time to bake weights into the image)
ARG BAKE_MODELS=0
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN if [ "$BAKE_MODELS" = "1" ]; then python model_fetcher.py; fi

# RunPod handler entry point
CMD ["python", "-u", "handler.py"]

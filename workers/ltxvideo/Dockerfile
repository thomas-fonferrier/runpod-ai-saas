FROM runpod/base:0.6.2-cuda12.1.0

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY model_fetcher.py .

ARG BAKE_MODELS=0
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN if [ "$BAKE_MODELS" = "1" ]; then python model_fetcher.py; fi

CMD ["python", "-u", "handler.py"]

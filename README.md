# RunPod AI SaaS — FLUX.1 · LTXVideo · XTTS v2

A production-ready deployment of three AI generation models on [RunPod.io](https://runpod.io) serverless infrastructure. Each model runs as an isolated Docker container behind a RunPod serverless endpoint. An orchestrator script unifies them into a single SaaS API.

```
┌─────────────────────────────────────────────────────────┐
│                    Client / SaaS API                    │
└───────────────────────┬─────────────────────────────────┘
                        │  HTTP
               ┌────────▼────────┐
               │   Orchestrator  │  (serverless Python)
               │   orchestrator/ │
               └──┬──────┬───┬───┘
     RunPod API   │      │   │
        ┌─────────▼─┐ ┌──▼──────┐ ┌──▼──────┐
        │  FLUX.1   │ │LTXVideo │ │ XTTS v2 │
        │ Endpoint  │ │Endpoint │ │Endpoint │
        └─────────┬─┘ └──┬──────┘ └──┬──────┘
                  │       │           │
        ┌─────────▼─┐ ┌──▼──────┐ ┌──▼──────┐
        │  Worker   │ │ Worker  │ │ Worker  │
        │ Container │ │Container│ │Container│
        └─────────┬─┘ └──┬──────┘ └──┬──────┘
                  │       │           │
        ┌─────────▼───────▼───────────▼──────┐
        │       RunPod Network Volume         │
        │   (model weights shared storage)   │
        └────────────────────────────────────┘
```

## Models

| Model | Task | GPU VRAM | Approx. Size |
|---|---|---|---|
| [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Text → Image | 16 GB+ | ~23 GB |
| [LTX-Video](https://huggingface.co/Lightricks/LTX-Video) | Image → Video | 24 GB+ | ~12 GB |
| [XTTS v2](https://huggingface.co/coqui/XTTS-v2) | Text → Speech | 4 GB+ | ~2 GB |

## Repository Layout

```
.
├── workers/
│   ├── flux/          # FLUX.1 RunPod serverless worker
│   ├── ltxvideo/      # LTXVideo RunPod serverless worker
│   └── xtts/          # XTTS v2 RunPod serverless worker
├── kubernetes/        # K8s manifests (HPA, Deployments, Services)
│   ├── flux/
│   ├── ltxvideo/
│   └── xtts/
├── orchestrator/      # Serverless Python orchestrator (SaaS glue layer)
├── scripts/           # One-time setup & helper scripts
└── .github/workflows/ # GitHub Actions CI/CD (build & push Docker images)
```

## Quick Start

### 1. Prerequisites

- RunPod account with API key → [console.runpod.io](https://console.runpod.io)
- Docker Hub account (or GHCR)
- `runpodctl` CLI: `brew install runpod/runpodctl/runpodctl`
- HuggingFace account + token (for gated models)

### 2. Set GitHub Secrets

In your GitHub repo → **Settings → Secrets and variables → Actions**, add:

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `RUNPOD_API_KEY` | RunPod API key |
| `HF_TOKEN` | HuggingFace token |

### 3. Create a RunPod Network Volume

```bash
cd scripts/
# Edit the variables at the top, then:
bash setup_network_volume.sh
```

This creates a shared 100 GB network volume and downloads all model weights onto it. Workers mount this volume at `/runpod-volume` so cold-starts load in seconds instead of minutes.

### 4. Push Docker Images

Push any branch or tag to trigger GitHub Actions:

```bash
git push origin main
```

Alternatively build locally:

```bash
# Example for FLUX worker
docker build -t your-user/runpod-flux:latest workers/flux/
docker push your-user/runpod-flux:latest
```

### 5. Create RunPod Serverless Endpoints

For each worker, go to **RunPod → Serverless → New Endpoint** and set:

| Field | Value |
|---|---|
| Container Image | `your-user/runpod-flux:latest` |
| GPU Type | A100 / H100 / RTX 4090 (see table above) |
| Min Workers | 0 (scale to zero) |
| Max Workers | 5 (adjust to budget) |
| Environment Vars | `HF_TOKEN`, `MODEL_PATH=/runpod-volume/models` |
| Network Volume | Select the volume created in step 3 |

Repeat for `ltxvideo` and `xtts`.

Copy the three **Endpoint IDs** into `orchestrator/.env`.

### 6. Run the Orchestrator

```bash
cd orchestrator/
pip install -r requirements.txt
cp .env.example .env   # fill in your endpoint IDs and API key
python main.py
```

## Kubernetes (optional)

If you want to manage GPU pods with your own Kubernetes cluster (e.g. via RunPod's Kubernetes service or any NVIDIA-capable cluster), apply the manifests:

```bash
kubectl apply -f kubernetes/
```

Each model has a `Deployment`, a `Service`, and a `HorizontalPodAutoscaler` that scales on GPU utilisation / queue depth.

## Architecture Decisions

- **Scale-to-zero**: min workers = 0 means you pay nothing when idle.
- **Network Volume**: model weights live on a persistent volume, not inside the Docker image — images stay small (<5 GB) and cold-start is fast.
- **Isolated endpoints**: each model is its own endpoint so they scale independently and failures are isolated.
- **Orchestrator is stateless**: it's a thin client that submits jobs and polls for results. Deploy it as a Lambda, Cloud Run function, or Fly.io app.

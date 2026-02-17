# cera-vLLM

Self-hosted local LLM server for the [CERA](https://github.com/thangk/cera) framework. Run open-source models on your own GPU(s) and use them for CERA's generation phase — no rate limits, no per-token costs.

> **Note:** This project is under active development.

## Overview

cera-vLLM packages [vLLM](https://github.com/vllm-project/vllm) (a high-throughput LLM serving engine) with a management dashboard into a Docker Compose setup. It exposes an OpenAI-compatible API that CERA connects to seamlessly.

| Component | Port | Description |
|-----------|------|-------------|
| vLLM Server | 8100 | OpenAI-compatible `/v1/chat/completions` API |
| Dashboard | 8101 | Web UI for model management and monitoring |

## Quick Start

### 1. Configure

```bash
cp .env.example .env
# Edit .env — set DASHBOARD_PASSWORD (required)
# Optionally set HF_TOKEN for gated models (Llama, etc.)
```

### 2. Launch

```bash
docker compose up -d
```

### 3. Open Dashboard

Go to `http://localhost:8101` (or `http://<your-vm-ip>:8101` for remote). Log in with your dashboard password.

### 4. Download & Activate a Model

In the dashboard:
1. Pick a model from the recommended list
2. Click **Download** — wait for it to complete
3. Click **Activate** — vLLM restarts and loads the model

### 5. Connect to CERA

In the CERA web GUI, go to **Settings > Local LLMs**:
1. Toggle **Enable Local LLMs** on
2. Paste the **Endpoint URL** from the dashboard
3. Paste the **API Key** from the dashboard
4. Click **Test Connection** — should show your active model

Now select your local model in the Generation model dropdown when creating a job.

## Recommended Models

| Model | VRAM | Speed | Quality | Notes |
|-------|------|-------|---------|-------|
| `Qwen/Qwen3-30B-A3B` | ~20 GB | Fastest | Good | MoE — only 3B active params |
| `Qwen/Qwen3-32B` | ~65 GB | Fast | Excellent | Dense, competitive with GPT-4o |
| `meta-llama/Llama-3.3-70B-Instruct` | ~140 GB | Moderate | Excellent | Requires `HF_TOKEN` |
| `mistralai/Mistral-Small-24B-Instruct-2501` | ~48 GB | Fast | Good | Apache 2.0 licensed |

You can also add any HuggingFace model via the dashboard's custom model field.

## Multi-GPU Setup

For models that don't fit on a single GPU, enable tensor parallelism in `.env`:

```env
# Split model across 2 GPUs
VLLM_EXTRA_ARGS=--tensor-parallel-size 2

# Split across 4 GPUs
VLLM_EXTRA_ARGS=--tensor-parallel-size 4
```

## Remote Access

If running on a cloud VM (e.g., Cudo Compute) and accessing from your local machine:

### Option A: Open Firewall (Recommended for persistent setups)

```bash
# On the VM
sudo ufw allow 8100    # vLLM API
sudo ufw allow 8101    # Dashboard
```

Security is handled automatically:
- **vLLM API**: Protected by an auto-generated API key (shown in dashboard)
- **Dashboard**: Protected by your `DASHBOARD_PASSWORD`

Then access the dashboard at `http://<vm-public-ip>:8101`.

### Option B: SSH Tunnel (No firewall changes needed)

```bash
# From your local machine
ssh -L 8100:localhost:8100 -L 8101:localhost:8101 user@your-vm-ip
```

Then access the dashboard at `http://localhost:8101`.

## API Key

The dashboard auto-generates a secure API key on first boot. You can:
- **Copy** it from the dashboard's "CERA Connection" section
- **Regenerate** it anytime (vLLM automatically restarts with the new key)
- Paste it into CERA Settings > Local LLMs > API Key

## Architecture

```
┌─────────────────────────────────────┐
│            Docker Compose           │
│                                     │
│  ┌──────────┐    ┌──────────────┐  │
│  │   vLLM   │◄───│  Dashboard   │  │
│  │  :8100   │    │    :8101     │  │
│  │          │    │              │  │
│  │ GPU(s)   │    │ SQLite state │  │
│  └──────────┘    └──────────────┘  │
│       │                  │          │
│       └──── shared ──────┘          │
│         HuggingFace cache           │
└─────────────────────────────────────┘
         ▲
         │  OpenAI-compatible API
         │  POST /v1/chat/completions
         │
    ┌────┴─────┐
    │   CERA   │  (runs on your home PC or another server)
    │ Pipeline │
    └──────────┘
```

## Dashboard Features

- **Model Management**: Download, activate, and switch between models
- **Live Metrics**: Tokens/sec throughput, active requests, token counts
- **GPU Monitoring**: VRAM usage and utilization per GPU
- **Connection Info**: Copy-paste endpoint URL and API key for CERA
- **Custom Models**: Add any HuggingFace model by ID

## Troubleshooting

### vLLM won't start / OOM errors
- Check VRAM requirements in the model table above
- Try reducing memory: `VLLM_EXTRA_ARGS=--gpu-memory-utilization 0.85`
- Use a smaller model or enable quantization: `VLLM_EXTRA_ARGS=--quantization awq`
- Check logs: `docker compose logs vllm`

### Model download fails
- For gated models (Llama), set `HF_TOKEN` in `.env`
- Check disk space — large models need 50-200 GB
- Check logs: `docker compose logs dashboard`

### Can't connect from CERA
- Verify vLLM is running: `curl http://localhost:8100/v1/models`
- Check firewall ports are open (for remote access)
- Ensure the API key in CERA Settings matches the one in the dashboard

### Dashboard shows "Offline"
- vLLM may still be loading the model (can take 1-3 minutes for large models)
- Check logs: `docker compose logs vllm`

## Related Projects

- [CERA](https://github.com/thangk/cera) — Configurable Environment for Review Authentication
- [cera-LADy](https://github.com/thangk/cera-LADy) — Evaluation framework for generated datasets

## License

This project is part of the CERA research framework.

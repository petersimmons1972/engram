# Installation Guide

This guide covers GPU-accelerated Ollama setup for every supported platform. The standard `docker compose up -d` works on all platforms but runs Ollama in CPU mode. Follow your platform's section to enable GPU acceleration.

**Supported GPU paths:**

| Platform | Image | GPU method |
|---|---|---|
| NVIDIA (Linux/Windows WSL2) | `ollama/ollama:latest` | NVIDIA Container Toolkit + `deploy` stanza |
| AMD (Linux) | `ollama/ollama:rocm` | ROCm device passthrough |
| Mac M-series (Apple Silicon) | Native Ollama via `brew` | Metal (automatic, host-side) |
| CPU only (any platform) | `ollama/ollama:latest` | No GPU config needed |

---

## Prerequisites (All Platforms)

- Docker Engine 20.10+ and Docker Compose v2
- 4 GB RAM minimum (8 GB recommended when running Ollama in Docker)
- 2 GB free disk space for the `nomic-embed-text` model

```bash
# Verify your versions
docker --version          # Docker version 24.x or newer
docker compose version    # Docker Compose version v2.x or newer
```

---

## Quick Start (CPU Only)

If you don't need GPU acceleration or just want to verify the setup works first:

```bash
git clone https://github.com/shugav/engram.git
cd engram
cp .env.example .env
# Edit .env — at minimum, set POSTGRES_PASSWORD to something strong
docker compose up -d
```

Then pull the embedding model:

```bash
docker exec engram-ollama ollama pull nomic-embed-text
```

Engram is now running at `http://localhost:8788/sse`. Connect your IDE (see the main README for IDE-specific steps).

---

## NVIDIA GPU Setup

### 1. Install NVIDIA Container Toolkit

The NVIDIA Container Toolkit lets Docker access your GPU. You only need to do this once per host.

**Ubuntu / Debian:**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verify the toolkit is working:**
```bash
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```
You should see your GPU listed in the output.

### 2. Enable GPU in docker-compose.yml

Open `docker-compose.yml` and find the `ollama` service. Uncomment the `deploy` block:

```yaml
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama-data:/root/.ollama
    ports:
      - "127.0.0.1:11434:11434"
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### 3. Start and pull the model

```bash
docker compose up -d
docker exec engram-ollama ollama pull nomic-embed-text
```

### 4. Verify GPU is in use

```bash
docker exec engram-ollama ollama run nomic-embed-text "test" 2>&1 | head -5
# Should show GPU memory usage in Ollama logs:
docker logs engram-ollama 2>&1 | grep -i "gpu\|cuda"
```

---

## AMD GPU Setup (ROCm)

### 1. Install ROCm

**Ubuntu 22.04:**
```bash
# Add ROCm repository
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian jammy main" | \
  sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt-get update
sudo apt-get install -y rocm-dkms rocm-libs

# Add your user to the required groups
sudo usermod -a -G render,video $USER

# Reboot for group changes and driver loading
sudo reboot
```

**Verify ROCm after reboot:**
```bash
rocm-smi
```

### 2. Enable AMD GPU in docker-compose.yml

Two changes are needed. First, change the Ollama image:

```yaml
  ollama:
    image: ollama/ollama:rocm     # ← change from :latest to :rocm
```

Then uncomment the `devices` block:

```yaml
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
```

### 3. Start and pull the model

```bash
docker compose up -d
docker exec engram-ollama ollama pull nomic-embed-text
```

### 4. Verify GPU is in use

```bash
docker logs engram-ollama 2>&1 | grep -i "rocm\|gpu"
```

---

## Mac M-Series (Apple Silicon)

Docker Desktop on macOS does not pass through Metal GPU to containers. Running Ollama in Docker on a Mac uses CPU only. For Metal acceleration — which is dramatically faster — run Ollama natively on the host and point Engram at it.

### 1. Install Ollama natively

```bash
brew install ollama
```

Or download from [ollama.com](https://ollama.com).

### 2. Pull the embedding model

```bash
ollama pull nomic-embed-text
```

### 3. Start Ollama as a background service

```bash
# Start as a launchd service (starts on login, runs in background)
brew services start ollama

# Or start manually for this session
ollama serve &
```

Verify it's running:
```bash
curl http://localhost:11434/api/tags
# Should return JSON listing your installed models
```

### 4. Configure Engram to use the host Ollama

In your `.env`:
```bash
OLLAMA_URL=http://host.docker.internal:11434
```

This routes from the Docker network to your host's Ollama process.

### 5. Disable the Docker Ollama service

Since Ollama is running natively, comment out the `ollama` service in `docker-compose.yml` and remove it from `engram`'s `depends_on`:

```yaml
  # ollama:          ← comment out the entire service block
  #   container_name: engram-ollama
  #   ...

  engram:
    depends_on:
      postgres:
        condition: service_healthy
      # ollama:      ← remove this line
      #   condition: service_started
```

### 6. Start Engram

```bash
docker compose up -d
```

---

## Pulling the Embedding Model

Regardless of platform, Ollama needs the model downloaded before Engram can embed. Run this once after starting:

```bash
docker exec engram-ollama ollama pull nomic-embed-text
```

The `nomic-embed-text` model is ~300 MB. It downloads once and persists in the `ollama-data` volume across restarts.

**Using a different model:**

Set `ENGRAM_OLLAMA_MODEL` in your `.env` and pull that model instead:

```bash
# Example: use mxbai-embed-large for higher quality (670 MB)
# In .env: ENGRAM_OLLAMA_MODEL=mxbai-embed-large
docker exec engram-ollama ollama pull mxbai-embed-large
```

**Important:** Once Engram stores its first embedding, it locks the project to that model's dimensions. To switch models, export your memories first (`engram dump`), then re-ingest after switching.

---

## Verification

Run this after any setup to confirm the full stack is working:

```bash
# 1. All services are up
docker compose ps

# 2. Ollama has the model
docker exec engram-ollama ollama list
# Expected: nomic-embed-text listed

# 3. Engram can reach Ollama
docker logs engram-app 2>&1 | grep -i "ollama\|embed"
# Expected: "Auto-detected Ollama" or "Using Ollama embeddings"

# 4. Engram MCP endpoint responds
curl http://localhost:8788/sse
# Expected: SSE stream headers (HTTP 200)

# 5. Quick embed test
docker exec engram-app python -c "
from engram.embeddings import create_embedder
e = create_embedder('ollama', ollama_url='http://ollama:11434')
v = e.embed('hello world')
print(f'Embedding dims: {len(v)} (expected 768 for nomic-embed-text)')
"
```

---

## Troubleshooting

**Engram starts but embeddings fall back to BM25-only mode**

Check if Ollama is reachable from the Engram container:
```bash
docker exec engram-app curl -s http://ollama:11434/api/tags | head -c 200
```
If this returns an error, Ollama isn't reachable. Common causes: Ollama container not started, wrong `OLLAMA_URL` in `.env`, or model not pulled.

**"nomic-embed-text model not found"**

The model isn't pulled yet. Run:
```bash
docker exec engram-ollama ollama pull nomic-embed-text
```
Then restart Engram: `docker compose restart engram`

**NVIDIA GPU not detected**

Verify in this order:
```bash
# Host: driver installed?
nvidia-smi

# Host: toolkit installed?
nvidia-ctk --version

# Docker: runtime configured?
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

# Compose: deploy block uncommented and correct?
docker compose config | grep -A10 "deploy:"
```

**AMD GPU: "device not found"**

Check device access:
```bash
ls -la /dev/kfd /dev/dri
groups  # should include render and video
```

If you're not in the groups, add yourself and log out/in:
```bash
sudo usermod -a -G render,video,docker $USER
```

**Mac: "connection refused" to host.docker.internal:11434**

Ollama isn't running on the host:
```bash
curl http://localhost:11434/api/tags
# If this fails, start Ollama: brew services start ollama
```

**Slow embedding on Mac without GPU**

If you're running Ollama in Docker instead of natively, it uses CPU only. Switch to native Ollama (see Mac M-series section above). The performance difference with Metal is substantial — typically 10–30× faster for embedding workloads.

**Port conflicts**

Default ports: `5432` (PostgreSQL), `11434` (Ollama), `8788` (Engram).

To change any port, set the corresponding variable in `.env`:
```bash
ENGRAM_PORT=9000    # changes Engram's exposed port
```
PostgreSQL and Ollama ports can be changed in `docker-compose.yml` if needed.

---

## Connect Your IDE

Once the stack is running, connect your MCP client. From the main README:

- **Claude Code:** `claude mcp add engram --transport sse http://localhost:8788/sse`
- **Cursor / VS Code:** add `{ "mcpServers": { "engram": { "url": "http://localhost:8788/sse" } } }` to your MCP config
- **Windsurf, Claude Desktop:** see README for client-specific config

---

## Support

If something isn't working after following this guide, open a [GitHub issue](https://github.com/shugav/engram/issues) with:
- Your OS and version
- GPU model and driver version (if applicable)
- Output of `docker compose ps` and `docker compose logs engram`

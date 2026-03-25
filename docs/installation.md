# Engram Installation Guide

Complete step-by-step installation instructions for all GPU configurations.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [NVIDIA GPUs (CUDA)](#nvidia-gpus-cuda)
3. [AMD GPUs (ROCm)](#amd-gpus-rocm)
4. [Mac M-series (Metal)](#mac-m-series-metal)
5. [System Ollama (Local/Host)](#system-ollama-localhost)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

All installations require:
- **Docker & Docker Compose** (v2.20+)
  - [Install Docker Desktop](https://www.docker.com/products/docker-desktop)
  - Verify: `docker --version && docker compose --version`

- **Git**
  - Verify: `git --version`

- **8GB+ RAM** (minimum; 16GB+ recommended)

- **Free disk space**: 20GB+ (for container images + Ollama model)

---

## NVIDIA GPUs (CUDA)

### Prerequisites

✅ **GPU Requirements:**
- NVIDIA GPU with CUDA compute capability 3.5+ (Maxwell generation or newer)
- Common supported GPUs: GeForce GTX/RTX 1000+, Quadro P/RTX series, A100, H100

✅ **Software Requirements:**

1. **NVIDIA Container Toolkit**

   **Ubuntu/Debian:**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   ```

   **CentOS/RHEL:**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   yum-config-manager --add-repo \
     https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo
   sudo yum install -y nvidia-container-toolkit
   ```

   **macOS (Docker Desktop):**
   - Not applicable; Docker Desktop on Mac uses native hypervisor acceleration

2. **Docker daemon configuration**

   After installing NVIDIA Container Toolkit, configure Docker to use the nvidia runtime:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

   Verify:
   ```bash
   docker run --rm --runtime=nvidia nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
   ```

   You should see your GPU listed.

✅ **Verify NVIDIA driver:**
```bash
nvidia-smi
```

Expected output:
```
NVIDIA-SMI 545.23.06    Driver Version: 545.23.06    CUDA Version: 12.1
|---------|---------|
| GPU  Name  Persistence-M|
| 0    NVIDIA A100  On    |
|---------|---------|
```

### Installation Steps

1. **Clone or navigate to Engram directory:**
   ```bash
   cd ~/projects/engram
   git pull origin main
   ```

2. **Start services with NVIDIA GPU override:**
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d
   ```

3. **Verify services are healthy:**
   ```bash
   docker compose ps
   ```

   Expected output:
   ```
   NAME              STATUS
   engram-postgres   healthy
   engram-ollama     healthy
   engram-app        running
   ```

4. **Pull the embedding model:**
   ```bash
   docker compose --profile init run --rm ollama-init
   ```

   Wait for model download (nomic-embed-text, ~300MB).

### Verification

✅ **GPU is being used:**
```bash
docker logs engram-ollama | grep -i "gpu\|cuda\|device"
```

Expected: mentions of CUDA, GPU device ID, or available memory on GPU.

✅ **Embedding works:**
```bash
docker compose exec engram-app python -c \
  "from engram.embeddings import create_embedder; e = create_embedder('ollama', url='http://ollama:11434'); print(e.embed('test')[:5])"
```

Expected: Returns first 5 dimensions of embedding vector (numbers like `[0.123, -0.456, ...]`).

### Troubleshooting

**"nvidia-smi not found"**
- Driver not installed or not in PATH
- `curl -fSsL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg`
- Then rerun apt-get install

**"could not select device driver"**
- NVIDIA Container Toolkit not installed
- Docker daemon not restarted after toolkit installation
- Run: `sudo systemctl restart docker && docker compose down && docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d`

**"GPU memory allocation failed"**
- Other processes using GPU memory
- Run: `nvidia-smi` to check usage
- Kill competing processes or reduce model size

---

## AMD GPUs (ROCm)

### Prerequisites

✅ **GPU Requirements:**
- AMD RDNA (RX 5700 XT and newer) or CDNA (MI100, MI210, etc.)
- Check if supported: [AMD ROCm GPU Compatibility](https://rocmdocs.amd.com/en/docs-5.7.0/deploy/linux/index.html#supported-gpus)

✅ **Software Requirements:**

1. **ROCm driver and runtime** (Ubuntu/Debian 22.04)

   ```bash
   wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
   echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian focal main" | \
     sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt-get update
   sudo apt-get install -y rocm-dkms rocm-libs
   sudo usermod -a -G render,video $USER
   ```

   **Reboot required:**
   ```bash
   sudo reboot
   ```

2. **Docker group permissions** (after reboot)

   ```bash
   # Verify ROCm works
   rocm-smi

   # Add user to docker group
   sudo usermod -a -G docker $USER
   ```

   **Log out and log back in** for group changes to take effect.

### Installation Steps

1. **Clone or navigate to Engram directory:**
   ```bash
   cd ~/projects/engram
   git pull origin main
   ```

2. **Start services with AMD GPU override:**
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d
   ```

3. **Verify services are healthy:**
   ```bash
   docker compose ps
   ```

4. **Pull the embedding model:**
   ```bash
   docker compose --profile init run --rm ollama-init
   ```

### Verification

✅ **GPU is being used:**
```bash
docker logs engram-ollama | grep -i "rocm\|gpu\|device"
```

✅ **Embedding works:**
```bash
docker compose exec engram-app python -c \
  "from engram.embeddings import create_embedder; e = create_embedder('ollama', url='http://ollama:11434'); print(e.embed('test')[:5])"
```

### Troubleshooting

**"rocm-smi: command not found"**
- ROCm not installed or not in PATH
- Rerun: `sudo apt-get install rocm-dkms rocm-libs`

**"GPU device not found in docker"**
- Devices `/dev/kfd` and `/dev/dri` not accessible
- Check: `ls -la /dev/kfd /dev/dri`
- User not in correct groups: `groups` should include `render` and `video`
- Fix: `sudo usermod -a -G render,video,docker $USER && newgrp docker`

**"ROCM_HOME not set"**
- Docker Ollama image handles this automatically
- If manual Ollama: `export ROCM_HOME=/opt/rocm`

---

## Mac M-series (Metal)

### Prerequisites

✅ **System Requirements:**
- Mac with M1, M2, M3, or M4 chip (Apple Silicon)
- macOS 13.0+ (Ventura or newer)
- 8GB+ unified memory (RAM)

✅ **Software Requirements:**
- **Docker Desktop for Mac** (v4.24+)
  - [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
  - Open `.dmg` and drag Docker to Applications

### Installation Steps

1. **Start Docker Desktop**
   - Open Applications → Docker
   - Wait for it to fully start (Docker icon in menu bar)

2. **Clone or navigate to Engram directory:**
   ```bash
   cd ~/projects/engram
   git pull origin main
   ```

3. **Start services (no GPU override needed)**
   ```bash
   docker compose up -d
   ```

   Docker Desktop on Apple Silicon automatically uses Metal acceleration. No configuration needed.

4. **Verify services are healthy:**
   ```bash
   docker compose ps
   ```

5. **Pull the embedding model:**
   ```bash
   docker compose --profile init run --rm ollama-init
   ```

### Verification

✅ **Metal acceleration enabled:**
```bash
docker logs engram-ollama | grep -i "metal\|performance"
```

Ollama logs should mention Metal framework or acceleration.

✅ **Embedding works:**
```bash
docker compose exec engram-app python -c \
  "from engram.embeddings import create_embedder; e = create_embedder('ollama', url='http://ollama:11434'); print(e.embed('test')[:5])"
```

### Troubleshooting

**"Docker daemon is not running"**
- Open Docker Desktop from Applications
- Check menu bar for Docker icon

**"connection refused" to Ollama**
- Wait 30 seconds for Ollama to fully start
- Check: `docker logs engram-ollama`

**Slow embedding performance**
- Unified memory shared with CPU
- Metal uses GPU but may be slower than dedicated GPUs
- Consider system Ollama if performance critical

---

## System Ollama (Local/Host)

Use this if you're running Ollama natively on your machine instead of in Docker.

### Prerequisites

✅ **Install Ollama**
- [Download Ollama](https://ollama.ai)
- Supports: macOS, Linux, Windows (WSL2)

✅ **Start Ollama service**
```bash
# macOS/Linux
ollama serve &

# or run as service (recommended)
systemctl start ollama  # Linux
launchctl start com.ollama.Ollama  # macOS
```

Verify:
```bash
curl http://localhost:11434/api/tags
```

Should return: `{"models":[]}`

### Installation Steps

1. **Pull the embedding model:**
   ```bash
   ollama pull nomic-embed-text
   ```

   Verify:
   ```bash
   curl http://localhost:11434/api/tags | grep nomic
   ```

2. **Start Engram pointing to host Ollama:**
   ```bash
   cd ~/projects/engram
   docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up -d
   ```

   This override sets `OLLAMA_URL=http://host.docker.internal:11434` so the Docker container can reach your host Ollama.

3. **Verify services are healthy:**
   ```bash
   docker compose ps
   ```

### Verification

✅ **Ollama is accessible from Docker:**
```bash
docker compose exec engram-app curl http://host.docker.internal:11434/api/tags
```

Should return JSON with `nomic-embed-text` in the models list.

✅ **Embedding works:**
```bash
docker compose exec engram-app python -c \
  "from engram.embeddings import create_embedder; e = create_embedder('ollama', url='http://host.docker.internal:11434'); print(e.embed('test')[:5])"
```

### Troubleshooting

**"Connection refused to host.docker.internal"**
- Ollama not running
- Run: `ollama serve &` in another terminal
- Or check: `curl http://localhost:11434/api/tags`

**"HTTP 404 /api/tags"**
- Ollama running but model not downloaded
- Run: `ollama pull nomic-embed-text`

**Linux users:** `host.docker.internal` not available
- Use host IP instead: Find it with `hostname -I`
- Set in .env: `OLLAMA_URL=http://192.168.1.100:11434`
- Or use `--network host` in compose override

---

## Verification

Run this after any installation to confirm everything is working:

### 1. All services healthy

```bash
docker compose ps
```

All should show `healthy` or `running` status.

### 2. Ollama model available

```bash
docker compose exec engram-app python -c "
from engram.embeddings import create_embedder
embedder = create_embedder('ollama', url='http://ollama:11434')
print('✅ Ollama embedder ready')
"
```

### 3. Memory system working

```bash
docker compose exec engram-app python -c "
from engram.db import MemoryDB
db = MemoryDB()
print('✅ Memory database ready')
"
```

### 4. IDE connection test

See README.md "Connect Your IDE" section for IDE-specific verification.

---

## Troubleshooting

### General

**"docker compose: command not found"**
- Docker Compose v2 not installed
- Upgrade: `docker --version` (should be v20.10+)
- Then: `docker compose version` should work

**"Permission denied while trying to connect to Docker daemon"**
- Add user to docker group:
  ```bash
  sudo usermod -a -G docker $USER
  newgrp docker
  ```

**Services stuck in "restarting" loop**
- Check logs: `docker compose logs`
- Common: Port conflicts (5432, 8788, 11434)
- Stop: `docker compose down` then check with `lsof -i :PORT`

### GPU-Specific

**GPU recognized but not used by Ollama**

Check configuration is correct:
```bash
# NVIDIA
docker compose config | grep -A5 nvidia

# AMD
docker compose config | grep -A5 devices

# Verify override file
ls -la docker-compose.*.yml
```

**Model download stuck**

```bash
docker compose logs ollama | tail -20
```

Check network connection and disk space:
```bash
df -h  # disk space
curl https://ollama.ai  # network connectivity
```

### Memory/Embeddings

**"ImportError: No module named 'httpx'"**
- httpx missing from requirements
- Rebuild image: `docker compose build --no-cache`

**Embedding returns all zeros**
- NullEmbedder fallback (Ollama not reachable)
- Check: `docker compose logs engram-app | grep Ollama`
- Verify URL matches actual Ollama: grep OLLAMA_URL in docker-compose output

### Uninstall

To cleanly remove Engram:

```bash
# Stop containers
docker compose down

# Delete volumes (WARNING: loses all memory data)
docker volume rm engram_pgdata engram_ollama-data

# Delete images (optional, saves disk space)
docker rmi engram ollama/ollama cgr.dev/chainguard/postgres
```

**Before deleting volumes, export your memories:**
```bash
docker compose exec engram-app python -m engram dump \
  --project global \
  --output ./memory-backup
```

---

## Next Steps

After successful installation:

1. **Store your first memory:**
   ```bash
   docker compose exec engram-app python -c "
   from engram.memory import memory_store
   memory_store(
       content='Test memory for verification',
       memory_type='context',
       project='test'
   )
   "
   ```

2. **Connect your IDE** — See README.md "Connect Your IDE" section

3. **Read the CLI docs** — See README.md "Commands" section

4. **Join the community** — GitHub issues and discussions welcome

---

## Performance Tuning

### Embedding batch size

For faster embedding of many documents, adjust batch size:

```bash
docker compose exec engram-app python -c "
from engram.embeddings import create_embedder
embedder = create_embedder('ollama', batch_size=32)  # default 16
"
```

### Database optimization

For many memories (>100k), consider:

```bash
docker compose exec engram-postgres psql -U engram -d engram -c "
REINDEX INDEX idx_memory_project;
ANALYZE memory;
"
```

### Ollama model selection

`nomic-embed-text` is the default (768D, balanced). For different trade-offs:

- **Smaller/faster:** `nomic-embed-text-v1.5` (512D)
- **Larger/better quality:** `bge-large-en-v1.5` (1024D)

Change:
```bash
ollama pull bge-large-en-v1.5
ENGRAM_OLLAMA_MODEL=bge-large-en-v1.5 docker compose up -d
```

---

## Support

For issues:
1. Check logs: `docker compose logs [service]`
2. Search GitHub issues: https://github.com/anthropics/engram/issues
3. File new issue with: logs, docker version, GPU info, OS/version

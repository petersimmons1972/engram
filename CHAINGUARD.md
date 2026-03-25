# Chainguard Images Policy

## Philosophy

Chainguard Images are distroless, signed, vulnerability-scanned, and minimal. They reduce attack surface and supply-chain risk. Where a Chainguard equivalent exists, we use it. Where upstream projects offer no hardened alternative, we document the exception and minimize the surface.

## Current Stack

| Service | Image | Justification |
|---------|-------|--------------|
| PostgreSQL | `cgr.dev/chainguard/postgres:latest` | ✅ Chainguard official. Hardened, distroless, minimal. |
| Open WebUI | `ghcr.io/open-webui/open-webui:main` | ⚠️ Upstream default. No Chainguard alternative available. Used only for local UI, not on critical path. Consider replacing with memory-focused alternative if upstream becomes unmaintained. |
| Ollama | `ollama/ollama:latest` | ⚠️ Upstream default. No hardened Chainguard equivalent exists (Ollama is CUDA/binary-heavy). Runs locally only. Will revisit if Chainguard adds hardened ML runtimes. |

## Security Implications

- PostgreSQL is production-hardened; all data at rest and in transit is database-level encrypted (if configured).
- Ollama and Open WebUI are not on the authentication boundary (localhost only in this deployment). They are used by Engram internally, not exposed to untrusted networks.
- For external deployments, add TLS termination (Caddy is available in docker-compose.yml commented section).

## Alternative Approaches (Evaluated and Rejected)

1. **Full Chainguard stack:** No hardened Chainguard ML runtime exists yet. This is aspirational, not blocking.
2. **DIY hardened Ollama Dockerfile:** Would require maintaining a custom build and binary. Upstream maintains it better.
3. **No Ollama, API-only embeddings:** Would lock memory search to OpenAI or other external services. Local embeddings are the goal.

## GPU Support (Ollama)

The Ollama image (`ollama/ollama:latest`) supports GPU acceleration out of the box:

### NVIDIA GPUs
```bash
# Add to docker-compose.yml for a service:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### AMD GPUs
```bash
# Use docker run with:
docker run --device=/dev/kfd --device=/dev/dri ollama/ollama:latest
```

Or in docker-compose.yml:
```yaml
devices:
  - /dev/kfd:/dev/kfd
  - /dev/dri:/dev/dri
```

### Mac M-series (M1/M2/M3/M4)
Ollama automatically detects and uses Metal acceleration. No configuration needed. The Docker image runs via Docker Desktop, which has Metal support built-in.

### CPU-only (no GPU)
Ollama automatically falls back to CPU inference. Slower but functional.

**Current Setup:** No GPU configuration in docker-compose.yml. Ollama will auto-detect available hardware when the container starts. To enable GPU, uncomment the appropriate section above based on your hardware.

## Maintenance

When new Chainguard images become available (especially ML runtimes), audit this policy and upgrade.

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

## Maintenance

When new Chainguard images become available (especially ML runtimes), audit this policy and upgrade.

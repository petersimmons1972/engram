# Chainguard Images Policy

## Philosophy

Chainguard images are distroless, signed, and minimal. They reduce attack surface and supply-chain risk. Where a Chainguard equivalent exists and is practical, we use it.

## Current Stack

| Service    | Image                                   | Notes                                          |
|------------|-----------------------------------------|------------------------------------------------|
| PostgreSQL | `cgr.dev/chainguard/postgres:latest`    | Hardened, distroless, minimal.                 |
| Engram     | `cgr.dev/chainguard/python:latest`      | Distroless Python runtime; no shell, no pip.   |
| Ollama     | `ollama/ollama:latest` (or `:rocm`)     | No Chainguard equivalent. GPU runtimes are binary-heavy — upstream maintains best. |

## Security Notes

- PostgreSQL and Engram are on the authentication boundary; both use hardened Chainguard images.
- Ollama runs on the internal Docker network only, bound to `127.0.0.1:11434`. It is not exposed to untrusted networks.
- For multi-machine access, use a trusted mesh VPN (Tailscale, WireGuard) rather than exposing Engram to the internet.

## Maintenance

When new Chainguard images become available (especially hardened ML runtimes), audit this policy and upgrade.

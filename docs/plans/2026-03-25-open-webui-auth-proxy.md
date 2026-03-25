# Open-WebUI Auth Proxy Support for OllamaEmbedder

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route Engram's embedding requests through Open-WebUI (with Bearer token auth) instead of hitting Ollama directly, matching the K8s deployment pattern.

**Architecture:** Add optional `OLLAMA_API_KEY` env var support to `OllamaEmbedder` and `_ollama_reachable`. When set, all httpx requests to the Ollama endpoint include `Authorization: Bearer <key>`. In Docker Compose, promote Open-WebUI from optional profile to required service, and point Engram at `http://open-webui:8080/ollama` instead of `http://ollama:11434`. One manual setup step: user creates admin account + API key in Open-WebUI on first boot.

**Tech Stack:** Python, httpx, Docker Compose, Open-WebUI API

---

## Task 1: Add auth header support to OllamaEmbedder

**Files:**
- Modify: `src/engram/embeddings.py` (OllamaEmbedder class, _ollama_reachable, create_embedder)
- Test: `tests/test_embeddings.py`

**Step 1: Write failing tests for auth header**

Add to `tests/test_embeddings.py`:

```python
class TestOllamaAuth:
    def test_ollama_embedder_accepts_api_key(self):
        """OllamaEmbedder should accept an optional api_key parameter."""
        emb = OllamaEmbedder(base_url="http://localhost:11434", api_key="sk-test-123")
        assert emb._headers.get("Authorization") == "Bearer sk-test-123"

    def test_ollama_embedder_no_auth_header_without_key(self):
        """OllamaEmbedder should not include auth header when no key is provided."""
        emb = OllamaEmbedder(base_url="http://localhost:11434")
        assert "Authorization" not in emb._headers

    def test_create_embedder_passes_ollama_api_key(self, monkeypatch):
        """create_embedder should pass OLLAMA_API_KEY to OllamaEmbedder."""
        monkeypatch.setenv("OLLAMA_API_KEY", "sk-from-env")
        emb = create_embedder(provider="ollama", ollama_url="http://localhost:11434")
        assert isinstance(emb, OllamaEmbedder)
        assert emb._headers.get("Authorization") == "Bearer sk-from-env"

    def test_create_embedder_no_ollama_api_key(self, monkeypatch):
        """create_embedder should work without OLLAMA_API_KEY."""
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        emb = create_embedder(provider="ollama", ollama_url="http://localhost:11434")
        assert isinstance(emb, OllamaEmbedder)
        assert "Authorization" not in emb._headers
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/psimmons/projects/engram && python -m pytest tests/test_embeddings.py::TestOllamaAuth -v`
Expected: FAIL — `OllamaEmbedder` doesn't accept `api_key` param, no `_headers` attribute

**Step 3: Implement auth header support in OllamaEmbedder**

Modify `src/engram/embeddings.py`:

```python
class OllamaEmbedder:
    """Ollama nomic-embed-text via local REST API (768 dimensions).

    Calls Ollama's /api/embed endpoint directly with httpx -- no ollama
    Python package needed. Supports optional Bearer auth for Open-WebUI proxy.
    """

    name = "ollama/nomic-embed-text"
    dimensions = 768
    version = "v1.5"

    def __init__(self, base_url: str = "http://localhost:11434", api_key: str | None = None):
        if _httpx is None:
            raise ImportError(
                "httpx is required for Ollama embeddings: pip install engram[ollama]"
            )
        _require_numpy("Ollama embeddings")
        if not _validate_ollama_url(base_url):
            raise ValueError(f"Blocked Ollama URL (potential SSRF): {base_url}")
        self._base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def embed(self, text: str) -> np.ndarray:
        resp = _httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": "nomic-embed-text", "input": text},
            headers=self._headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return np.array(data["embeddings"][0], dtype=np.float32)

    def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = _httpx.post(
                f"{self._base_url}/api/embed",
                json={"model": "nomic-embed-text", "input": batch},
                headers=self._headers,
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            all_embeddings.extend(
                np.array(emb, dtype=np.float32) for emb in data["embeddings"]
            )
        return all_embeddings
```

**Step 4: Update create_embedder to pass OLLAMA_API_KEY**

In `create_embedder()`, update the ollama and auto-detect branches:

```python
def create_embedder(
    provider: str | None = None,
    api_key: str | None = None,
    ollama_url: str = "http://localhost:11434",
) -> EmbeddingProvider:
    # ... existing provider selection logic ...

    if provider == "ollama":
        url = os.environ.get("OLLAMA_URL", ollama_url)
        ollama_key = os.environ.get("OLLAMA_API_KEY")
        return OllamaEmbedder(base_url=url, api_key=ollama_key)

    # ... (none provider unchanged) ...

    # Auto-detect
    auto_url = os.environ.get("OLLAMA_URL", ollama_url)
    ollama_key = os.environ.get("OLLAMA_API_KEY")
    if _ollama_reachable(auto_url, api_key=ollama_key):
        logger.info("Auto-detected Ollama at %s, using local embeddings", auto_url)
        return OllamaEmbedder(base_url=auto_url, api_key=ollama_key)
    # ... rest unchanged ...
```

**Step 5: Update _ollama_reachable to pass auth header**

```python
def _ollama_reachable(base_url: str, api_key: str | None = None) -> bool:
    """Quick check if Ollama is running and has nomic-embed-text."""
    if _httpx is None:
        logger.debug("httpx not installed — Ollama auto-detect skipped")
        return False
    try:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = _httpx.get(f"{base_url}/api/tags", headers=headers, timeout=2.0)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any("nomic-embed-text" in m for m in models)
    except _httpx.HTTPError:
        logger.debug("Ollama not reachable at %s", base_url)
    except Exception:
        logger.debug("Ollama auto-detect failed", exc_info=True)
    return False
```

**Step 6: Run tests to verify they pass**

Run: `cd /home/psimmons/projects/engram && python -m pytest tests/test_embeddings.py -v`
Expected: ALL PASS

**Step 7: Run full test suite**

Run: `cd /home/psimmons/projects/engram && python -m pytest tests/ -v --ignore=tests/test_db_postgres.py`
Expected: ALL PASS (no regressions)

**Step 8: Commit**

```bash
git add src/engram/embeddings.py tests/test_embeddings.py
git commit -m "feat: add Bearer auth support to OllamaEmbedder for Open-WebUI proxy"
```

---

## Task 2: Update Docker Compose for Open-WebUI in the chain

**Files:**
- Modify: `docker-compose.yml`
- Modify: `docker-compose.host-ollama.yml`

**Step 1: Promote Open-WebUI from optional profile to required service in docker-compose.yml**

Remove `profiles: ["ui"]` from the `open-webui` service. Update the `engram` service to depend on `open-webui` and point `OLLAMA_URL` at Open-WebUI instead of Ollama directly. Add `OLLAMA_API_KEY` env var.

```yaml
  engram:
    container_name: engram-app
    build: .
    ports:
      - "${ENGRAM_PORT:-8788}:8788"
    environment:
      - ENGRAM_API_KEY=${ENGRAM_API_KEY:-}
      - ENGRAM_EMBEDDER=ollama
      - OLLAMA_URL=http://open-webui:8080/ollama
      - OLLAMA_API_KEY=${OLLAMA_API_KEY:-}
      - DATABASE_URL=postgresql://engram:${POSTGRES_PASSWORD:-engram}@postgres:5432/engram
    depends_on:
      postgres:
        condition: service_healthy
      open-webui:
        condition: service_started
    restart: unless-stopped

  open-webui:
    container_name: engram-open-webui
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - WEBUI_AUTH=true
      - ENABLE_SIGNUP=true
      - DEFAULT_USER_ROLE=user
    volumes:
      - open-webui-data:/app/backend/data
    depends_on:
      ollama:
        condition: service_healthy
    restart: unless-stopped
```

**Step 2: Update docker-compose.host-ollama.yml**

Point Open-WebUI at host Ollama instead of Docker Ollama:

```yaml
services:
  engram:
    environment:
      - OLLAMA_URL=http://open-webui:8080/ollama
      - OLLAMA_API_KEY=${OLLAMA_API_KEY:-}
    depends_on:
      postgres:
        condition: service_healthy
      open-webui:
        condition: service_started

  open-webui:
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    depends_on:
      postgres:
        condition: service_healthy

  ollama:
    profiles: ["disabled"]

  ollama-init:
    profiles: ["disabled"]
```

**Step 3: Commit**

```bash
git add docker-compose.yml docker-compose.host-ollama.yml
git commit -m "feat: route Engram through Open-WebUI for Ollama access with auth"
```

---

## Task 3: Update MCP config for local stdio usage

**Files:**
- Modify: `/home/psimmons/.claude.json` (engram MCP entry)

**Step 1: Add OLLAMA_API_KEY env var to the MCP config**

The local stdio Engram needs `OLLAMA_API_KEY` set so it can authenticate through Open-WebUI when used that way. For now, since the local instance hits Ollama directly (no Open-WebUI in the path), this is a no-op — but it prepares for future routing.

Update the engram MCP entry in `.claude.json`:

```json
"engram": {
  "type": "stdio",
  "command": "/home/psimmons/projects/engram/.venv/bin/engram",
  "args": ["server", "--transport", "stdio"],
  "env": {
    "OLLAMA_API_KEY": "<key-from-open-webui>"
  }
}
```

**Note:** This step is only needed if the local instance will also route through Open-WebUI. Skip if local Engram continues hitting Ollama directly.

**Step 2: Commit (if applicable)**

No commit for .claude.json — it's user config, not repo code.

---

## Task 4: Document first-boot setup

**Files:**
- Modify: `README.md` (or create `docs/OPEN-WEBUI-SETUP.md`)

**Step 1: Add setup instructions**

Document the one-time manual step:

```markdown
## First-Boot Setup (Open-WebUI)

After `docker compose up -d`, Open-WebUI needs an admin account and API key:

1. Open http://localhost:3000
2. Create an admin account (first signup becomes admin)
3. Go to Settings > Account > API Keys
4. Generate a new API key (starts with `sk-...`)
5. Add it to your `.env` file:
   ```
   OLLAMA_API_KEY=sk-your-key-here
   ```
6. Restart Engram: `docker compose restart engram`
```

**Step 2: Commit**

```bash
git add README.md  # or docs/OPEN-WEBUI-SETUP.md
git commit -m "docs: add Open-WebUI first-boot setup instructions"
```

---

## Summary

| Task | What                                    | Files Changed                                  |
| ---- | --------------------------------------- | ---------------------------------------------- |
| 1    | Auth header support in OllamaEmbedder   | `embeddings.py`, `test_embeddings.py`          |
| 2    | Docker Compose routing through Open-WebUI | `docker-compose.yml`, `docker-compose.host-ollama.yml` |
| 3    | MCP config for local stdio (optional)   | `.claude.json`                                 |
| 4    | First-boot documentation                | `README.md` or `docs/`                         |

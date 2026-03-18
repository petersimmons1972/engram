# Alternative Install — Local Python + SQLite

If you'd rather run Engram without Docker, you can install it directly with pip.
This uses SQLite for storage — zero external services, one database file per
project, everything at `~/.engram/`.

> **Trade-off:** SQLite is single-writer. Fine for one agent at a time (stdio
> mode), but if you need multiple concurrent agents, use the
> [Docker install](README.md#install) with PostgreSQL instead.

---

## Prerequisites

- Python 3.11+
- Git
- *(Optional)* An OpenAI API key, or [Ollama](https://ollama.com) with
  `nomic-embed-text` pulled

## Install

```bash
git clone https://github.com/shugav/engram.git
cd engram
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Engram creates its SQLite databases on first use at `~/.engram/`.

## Configure Embeddings (Optional)

```bash
# Option A: OpenAI (highest quality, costs money)
export OPENAI_API_KEY="sk-..."

# Option B: Ollama (good quality, free, fully local)
# Install from https://ollama.com, then:
ollama pull nomic-embed-text

# Option C: Do nothing — Engram falls back to BM25-only mode automatically
```

## Connect Your IDE

### Cursor / VS Code (local, same machine)

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram/.venv/bin/python",
      "args": ["-m", "engram"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "PYTHONPATH": "/path/to/engram/src"
      }
    }
  }
}
```

Replace `/path/to/engram` with your actual clone path. Restart Cursor. Engram
starts as a subprocess — no server to manage.

### Claude Code

```bash
claude mcp add engram -- /path/to/engram/.venv/bin/python -m engram
```

### Network Mode (SSE) — Multiple Machines

Start the server on one machine:

```bash
python -m engram --transport sse --port 8788
```

Point any client at it:

```json
{
  "mcpServers": {
    "engram": {
      "url": "http://your-server:8788/sse"
    }
  }
}
```

Or use the setup script to auto-configure a remote Cursor instance:

```bash
bash setup-remote.sh your-server 8788
```

---

## Uninstall

### Remove the Code

```bash
# If installed with pip
pip uninstall engram

# If cloned manually
rm -rf /path/to/engram
```

### Remove Your Data

```bash
rm -rf ~/.engram
```

### Remove IDE Configuration

- **Cursor/VS Code:** `~/.cursor/mcp.json` → remove the `"engram"` key
- **Claude Code:** `claude mcp remove engram`

That's everything. No background processes, no system services, no config files
scattered across your system.

# Engram

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MCP Server](https://img.shields.io/badge/MCP-server-black)](https://modelcontextprotocol.io/)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange)](https://github.com/shugav/engram)

**Your AI agents forget everything between sessions. Engram fixes that.**

Every time you close a tab in Cursor, Claude Code, or VS Code, your agent loses
the decisions it made, the bugs it found, the architecture it mapped out. The
next session starts from zero. You re-explain the project. The agent re-discovers
the same gotchas. You both waste time.

Engram is a persistent memory server that gives AI agents a real brain — one that
survives across sessions, machines, and IDEs. It speaks
[MCP](https://modelcontextprotocol.io/) (the protocol your tools already
understand), stores everything in a local SQLite database you own, and searches
it with a hybrid engine that combines keyword matching, semantic similarity, a
knowledge graph, and recency signals. No cloud service. No subscription. Just a
Python process and a database file on your disk.

When one agent stores a decision, the next agent finds it. When you switch from
Cursor to Claude Code, the context follows. When three agents work on the same
project, they share what they learn — like coworkers leaving notes on a shared
whiteboard, except the whiteboard remembers everything and can find the note you
need before you ask.

> **Beta software.** Engram is under active development by
> [shugav](https://github.com/shugav). APIs, storage format, and behavior may
> change between releases. See [LICENSE](LICENSE) for the full warranty
> disclaimer.

---

## What's Inside

Engram runs as a local MCP server and exposes ten tools that any MCP-compatible
client can call:

| Tool                   | What it does                                                    |
|------------------------|-----------------------------------------------------------------|
| `memory_store`         | Save a memory — auto-chunks, embeds, and indexes it             |
| `memory_recall`        | Search across all three layers with a single query              |
| `memory_list`          | Browse recent memories with type/tag/importance filters         |
| `memory_correct`       | Supersede a wrong or outdated memory with a corrected version   |
| `memory_forget`        | Delete a memory and all its graph connections                   |
| `memory_connect`       | Link two memories with a typed relationship                     |
| `memory_feedback`      | Tell the system which recall results were actually useful       |
| `memory_consolidate`   | Deduplicate, decay weak links, prune stale memories             |
| `memory_status`        | View stats: memory count, chunks, graph size, DB size           |
| `onboarding`           | Get a project-specific quick-start guide for new sessions       |

Memories are organized by **project** — each project gets its own isolated
database file. Your web app memories don't leak into your CLI tool's context.
Store user-wide preferences in `project="global"` so every project can find them.

---

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="Engram Architecture" width="900">
</p>

### The Three Layers

Engram doesn't rely on any single search strategy. It blends four signals into a
composite score, so recall works whether you remember the exact error code or
just the vague shape of the problem:

<p align="center">
  <img src="docs/scoring.svg" alt="Engram Search Scoring" width="900">
</p>

**1. BM25 Keyword Search** — SQLite FTS5 with Porter stemming. When you search
for "SQLITE_BUSY timeout", it finds the exact phrase. Fast, precise, zero
external dependencies.

**2. Vector Semantic Search** — Optional embedding-based similarity. When you
search for "database lock contention" and the stored memory says "WAL mode busy
timeout", the vectors connect the meaning even though the words differ. Supports
OpenAI, Ollama (local/free), or disabled entirely.

**3. Recency Decay** — Recently accessed memories score higher. Exponential decay
at 1% per hour means today's context matters more than last month's, but nothing
disappears — it just gets quieter.

**4. Knowledge Graph** — Memories linked by typed relationships (`depends_on`,
`supersedes`, `caused_by`, `relates_to`, `used_in`, `resolved_by`) get a
connectivity boost. When you recall one memory, its neighbors come along for the
ride. Over time, the `memory_feedback` tool strengthens useful connections and
weakens noise — the graph learns what matters.

The final score:

```
composite = (vector × 0.45) + (bm25 × 0.25) + (recency × 0.15) + (graph × 0.15)
final     = composite × importance_multiplier
```

Critical memories (importance 0) get a 2× boost. Trivial ones (importance 4) get
0.6×. The system self-maintains: `memory_consolidate` decays unused graph edges,
deduplicates chunks, and prunes low-importance memories that nobody has accessed
in 30 days.

---

## Memory Types

Engram uses six typed categories so agents can filter by context:

| Type             | When to use it                                           | Example                                                      |
|------------------|----------------------------------------------------------|--------------------------------------------------------------|
| `decision`       | Choices and their reasoning                              | "Chose PostgreSQL over MySQL because of JSON column support"  |
| `pattern`        | Recurring code or architecture patterns                  | "This codebase uses the repository pattern for all DB access" |
| `error`          | Bugs, gotchas, and their fixes                           | "Port 3000 is taken on this server — use 3001 instead"        |
| `context`        | General project or environment details                   | "Running on Ubuntu 22.04 with Python 3.11"                    |
| `architecture`   | System design, data flow, integrations                   | "Auth flow: JWT → middleware → httpOnly cookie"               |
| `preference`     | User conventions and style preferences                   | "User prefers tabs, 120-char line length, no trailing commas" |

---

## Embedding Options

You choose the quality/cost/privacy tradeoff. Engram auto-detects the best
available option, or you can set it explicitly:

| Mode       | Model                        | Dimensions | Quality     | Cost     | Privacy     |
|------------|------------------------------|------------|-------------|----------|-------------|
| **OpenAI** | `text-embedding-3-small`     | 1536       | Highest     | ~$0.02/M | Cloud       |
| **Ollama** | `nomic-embed-text`           | 768        | Good        | Free     | Fully local |
| **None**   | —                            | —          | BM25 only   | Free     | Fully local |

With no embeddings, you still get keyword search, recency scoring, and the full
knowledge graph. Vector search adds the "I know what you mean even when you don't
use the right words" layer.

> **Lock-in protection:** Once a project stores its first embedding, Engram
> records the model name and dimensions. If you switch models, it will refuse to
> mix incompatible vectors rather than silently corrupting your search results.

---

## Quick Start

### Prerequisites

- Python 3.11 or newer
- Git
- *(Optional)* An OpenAI API key, or [Ollama](https://ollama.com) with
  `nomic-embed-text` pulled

### Install

```bash
git clone https://github.com/shugav/engram.git
cd engram
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

That's it. No Docker required. No database to provision. No config files to
write. Engram creates its SQLite databases on first use at `~/.engram/`.

### Configure Embeddings (Optional)

```bash
# Option A: OpenAI (highest quality, costs money)
export OPENAI_API_KEY="sk-..."

# Option B: Ollama (good quality, free, fully local)
# Install from https://ollama.com, then:
ollama pull nomic-embed-text

# Option C: Do nothing — Engram falls back to BM25-only mode automatically
```

### Connect Your IDE

#### Cursor / VS Code (local, same machine)

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

Replace `/path/to/engram` with the actual path where you cloned it. Restart
Cursor. Engram starts as a subprocess — no server to manage.

#### Claude Code

```bash
claude mcp add engram -- /path/to/engram/.venv/bin/python -m engram
```

#### Network Mode (SSE) — Multiple Machines

Start the server on one machine:

```bash
python -m engram --transport sse --port 8788
```

Then point any client at it:

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

## Environment Variables

| Variable           | Default                   | What it does                                       |
|--------------------|---------------------------|----------------------------------------------------|
| `ENGRAM_EMBEDDER`  | *(auto-detect)*           | Force embedding mode: `openai`, `ollama`, or `none` |
| `OPENAI_API_KEY`   | *(unset)*                 | OpenAI key for vector embeddings                   |
| `OLLAMA_URL`       | `http://localhost:11434`  | Ollama server address (if non-default)             |
| `ENGRAM_PROJECT`   | `default`                 | Default project namespace                          |
| `ENGRAM_DIR`       | `~/.engram/`              | Where database files are stored                    |
| `ENGRAM_API_KEY`   | *(unset)*                 | Bearer token for SSE authentication                |

---

## Security Considerations

Engram stores data locally on your filesystem. Here's what you should know:

**Local (stdio) mode** is the default and the safest option. Engram runs as a
subprocess of your IDE. No network port is opened. Data stays on your machine.
The attack surface is the same as any other CLI tool you run locally.

**Network (SSE) mode** opens an HTTP endpoint. If you run it:

- **Set an API key.** Without one, anyone on your network can read and write your
  memories. Use `--api-key` or the `ENGRAM_API_KEY` env var.
- **Use TLS in production.** The API key is transmitted as a Bearer token over
  HTTP. Without TLS, anyone between you and the server can sniff it. Put a
  reverse proxy (Caddy, Nginx) in front for HTTPS.
- **Bind to localhost** unless you specifically need network access:
  `--host 127.0.0.1`. If you're on a trusted mesh VPN like Tailscale, binding to
  your Tailscale IP is also reasonable.

**What's stored:** Memory content (your text), embedding vectors (opaque float
arrays), and a knowledge graph (relationship metadata). All of it lives in
`~/.engram/*.db` SQLite files. No data is sent anywhere unless you configure
OpenAI embeddings — in which case your memory text is sent to OpenAI's embedding
API. Use Ollama or `none` mode if that's a concern.

For responsible disclosure of security issues, see [SECURITY.md](SECURITY.md).

---

## How Agents Use It

Engram ships with a built-in system prompt (the `onboarding` tool) that teaches
agents the full workflow. The short version:

**Session start** — The agent calls `memory_recall("session handoff")` to pick up
where the last agent left off.

**During work** — Whenever the agent makes a decision, encounters a bug, or
discovers a pattern, it stores a memory. Typed, tagged, with an importance level.

**Session end** — The agent stores a handoff note: what was done, what's next,
what's blocked, which files changed. The next agent reads this and picks up
seamlessly.

**Over time** — The feedback loop strengthens useful connections and the
consolidation pass prunes the noise. The memory system gets better the more you
use it.

---

## Database Layout

Each project gets its own SQLite file at `~/.engram/{project}.db`.

| Table             | Purpose                                                         |
|-------------------|-----------------------------------------------------------------|
| `memories`        | Memory records with content, type, tags, importance, timestamps |
| `memory_fts`      | FTS5 full-text index (Porter stemming, Unicode)                 |
| `chunks`          | Chunked text with embedding BLOBs and dedup hashes              |
| `relationships`   | Typed directed graph edges with decay-capable strength values    |
| `project_meta`    | Metadata: embedding model name, dimensions, schema version      |

WAL mode is enabled for better read concurrency. Each database is fully
self-contained — you can back it up by copying the `.db` file, or move it to
another machine.

---

## Scaling

| Deployment     | Agents   | How it works                                                                    |
|----------------|----------|---------------------------------------------------------------------------------|
| **stdio**      | 1 per machine | IDE spawns engram as a subprocess. Simplest setup.                        |
| **SSE**        | Many     | One central server, many agent clients over HTTP. All writes serialize through one process. |

**Known limitation:** SQLite is single-writer. In SSE mode, one server process
handles all writes serially — this works well for typical agent workloads but
won't scale to hundreds of concurrent writers. A PostgreSQL storage backend is
planned for true multi-process concurrency.

---

## Uninstall

Engram is designed to leave a small footprint and clean up easily.

### Remove the Code

```bash
# If installed with pip
pip uninstall engram

# If cloned manually
rm -rf /path/to/engram
```

### Remove Your Data

All databases live in one directory:

```bash
rm -rf ~/.engram
```

That's everything. No background processes, no system services, no config files
scattered across your system. If you used the `engram.sh` launcher via SSH,
removing the clone directory handles it.

### Remove IDE Configuration

Delete the `engram` entry from your MCP config:

- **Cursor/VS Code:** `~/.cursor/mcp.json` → remove the `"engram"` key
- **Claude Code:** `claude mcp remove engram`

---

## Compatible Clients

Engram works with any MCP-compatible client:

- [Cursor](https://cursor.sh)
- [VS Code](https://code.visualstudio.com/) (Copilot MCP support)
- [Claude Desktop](https://claude.ai)
- [Claude Code](https://github.com/anthropics/claude-code)
- [Windsurf](https://codeium.com/windsurf)

---

## Contributing

Contributions are welcome — first-time contributors especially so. See
[CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and workflow.

For security issues, see [SECURITY.md](SECURITY.md).

---

## License

MIT License. See [LICENSE](LICENSE).

---

<sub>Engram was created by [shugav](https://github.com/shugav). Security review
and documentation by [Peter Simmons](mailto:petersimmons@duck.com). README
written by Claude (Anthropic) — but don't hold that against it; the prose was
supervised by a human who actually read the code.</sub>

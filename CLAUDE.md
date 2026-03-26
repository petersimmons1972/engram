# Engram — Claude Instructions

General workflow, skills, commit rules, agent system, and cost guardrails: see `~/CLAUDE.md` and `~/AGENTS.md`.

## Database Protection — CRITICAL

**NEVER use `docker compose down -v`** — the `-v` flag deletes Docker volumes, destroying the PostgreSQL database and all memories permanently.

### Safe Docker operations:
- `docker compose restart` — restarts containers, keeps data ✅
- `docker compose up -d` — starts/updates services, keeps data ✅
- `docker compose down` — stops containers, keeps data ✅
- `docker compose down -v` — **DESTROYS DATA** ❌ NEVER USE

### Before ANY destructive operation:
1. Verify backups exist (see README.md Backup section)
2. Run: `bash bin/backup-postgres.sh` to create a dated backup
3. Confirm backup file was created: `ls -lh backups/engram-*.sql.gz`
4. Only then proceed with any risky operation

### If data loss occurs:
1. Check `~/.engram-archive/` for SQLite backups
2. Check `backups/` directory for PostgreSQL dumps
3. Run `python restore_from_sqlite.py` to restore from archives
4. Do NOT delete volumes or restart from scratch without backups

## Bug & Issue Tracking — NON-NEGOTIABLE

Every bug, defect, or enhancement — even if immediately fixed — MUST be filed as a GitHub Issue. No exceptions.
File the issue first, then fix it. A fix with no issue number never happened.
If a defect is not in the issue system, it does not exist.

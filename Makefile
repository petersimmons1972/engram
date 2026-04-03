.PHONY: lint typecheck test test-postgres quality down-safe

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:
	mypy src/engram/

test:
	pytest

test-postgres:
	docker compose --profile test up -d test-postgres
	TEST_DATABASE_URL=postgresql://engram:test@localhost:5433/engram_test pytest
	docker compose --profile test down

# SAFE shutdown: stops containers but PRESERVES volumes (all memory data intact).
# NEVER run 'docker compose down -v' — the -v flag DESTROYS all memory data permanently.
down-safe:
	docker compose down

quality: lint typecheck test

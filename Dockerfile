# Stage 1: Install dependencies in a builder with pip available
FROM cgr.dev/chainguard/python:latest-dev AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target /app/deps

# Stage 2: Copy into minimal runtime (no shell, no pip, non-root)
FROM cgr.dev/chainguard/python:latest

WORKDIR /app
COPY --from=builder /app/deps /app/deps
COPY src/ /app/src/

ENV PYTHONPATH=/app/deps:/app/src
ENV ENGRAM_DIR=/data
EXPOSE 8788

ENTRYPOINT ["python", "-m", "engram", "--transport", "sse", "--host", "0.0.0.0", "--port", "8788"]

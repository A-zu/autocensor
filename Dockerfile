# === Build Stage ===
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# System dependencies for building
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Copy source code
COPY uv.lock .
COPY main.py .
COPY chat.py .
COPY redact.py .
COPY pyproject.toml .
COPY blur_prompt.txt .
COPY logging_config.py .
COPY redact_prompt.txt .
COPY blur_masked_images.py .
COPY colorlog_formatter.py .
COPY static/ static/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# === Final Runtime Stage ===
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS final

WORKDIR /app

# System-level runtime dependency only
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-opencv ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and source code from builder
COPY --from=builder /app /app

# Set environment variables
ENV HOST=0.0.0.0 \
    PORT=8000 \
    OLLAMA_HOST=ollama:11434 \
    OLLAMA_MODEL=qwen3:4b \
    YOLOE_MODEL=yoloe-v8l-seg.pt \
    YOLO_CONFIG_DIR=/yolo/config \
    YOLO_BATCH_SIZE=16

# Start the app
CMD ["uv", "run", "main.py"]

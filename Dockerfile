# === Build Stage ===
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# System dependencies for building (if needed)
RUN apt-get update && \
    apt-get install -y python3-opencv git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN uv sync

# Copy source code
COPY main.py .
COPY chat.py .
COPY redact.py .
COPY blur_prompt.txt .
COPY redact_prompt.txt .
COPY blur_masked_images.py .
COPY static/ static/

# === Final Runtime Stage ===
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS final

WORKDIR /app

# System-level runtime dependency only (OpenCV)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-opencv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and source code from builder
COPY --from=builder /app /app

# Set environment variables
ENV HOST=0.0.0.0 \
    PORT=8000 \
    OLLAMA_HOST=ollama:11434 \
    OLLAMA_MODEL=qwen3:4b \
    YOLO_CONFIG_DIR=/yolo/config

# Start the app
CMD ["uv", "run", "main.py"]

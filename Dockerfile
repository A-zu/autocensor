FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3-opencv

WORKDIR /app

COPY pyproject.toml .
RUN uv sync

COPY main.py .
COPY chat.py .
COPY redact.py .
COPY blur_prompt.txt .
COPY redact_prompt.txt .
COPY file_processing.py .
COPY static/ static/

ENV OLLAMA_HOST=ollama:11434
ENV YOLO_CONFIG_DIR=/yolo/config

CMD ["uv", "run", "main.py"]

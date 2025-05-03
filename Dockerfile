FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3-opencv

WORKDIR /app

COPY pyproject.toml .
RUN uv sync

COPY main.py .
COPY chat.py .
COPY prompt.txt .
COPY file_processing.py .
COPY static/index.html static/index.html
COPY static/example.zip static/example.zip

CMD ["uv", "run", "main.py"]

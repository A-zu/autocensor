FROM ubuntu

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv python install 3.12
RUN uv python pin 3.12
RUN uv venv

RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN uv pip install uvicorn fastapi python-multipart opencv-python ultralytics easyocr numpy pillow pillow_heif

RUN apt update && apt upgrade -y
RUN apt install python3-opencv -y
RUN apt install curl -y

WORKDIR /app

RUN uv run python -c "from easyocr import Reader; Reader(['en', 'no'], gpu=False)"
RUN uv run python -c "from ultralytics import YOLO; YOLO('yolo12x.pt')"

COPY main.py .
COPY index.html .
COPY file_processing.py .

RUN uv run python main.py & \
    sleep 10 && \
    curl http://localhost:8000 && \
    pkill -f "uv run python main.py"

CMD ["uv", "run", "python", "main.py"]

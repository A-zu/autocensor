FROM python:3.13-pytorch

RUN apt update && apt upgrade -y
RUN apt install python3-opencv -y
RUN apt install curl -y

RUN pip install uvicorn fastapi python-multipart opencv-python ultralytics easyocr numpy pillow pillow_heif

WORKDIR /app

RUN python -c "from easyocr import Reader; Reader(['en', 'no'], gpu=False)"
RUN python -c "from ultralytics import YOLO; YOLO('yolo12x.pt')"

COPY main.py .
COPY file_processing.py .
COPY static/index.html static/index.html

RUN python main.py & \
    sleep 10 && \
    curl http://localhost:8000 && \
    pkill -f "python main.py"

CMD ["python", "main.py"]

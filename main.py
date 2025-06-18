import os
import uuid
from pathlib import Path
import logging.config

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from chat import generate_keywords, get_redactions
from blur_masked_images import process_zip_file
from redact import redact_pdf

logger = logging.getLogger(__name__)

HOST = os.getenv("HOST") or "0.0.0.0"
PORT = os.getenv("PORT") or 8000
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or "qwen3:4b"
YOLOE_MODEL = os.getenv("YOLOE_MODEL") or "yoloe-v8l-seg.pt"

app = FastAPI()

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def removeProxyPath(request: Request, call_next):
    root_path = request.scope.get("root_path", "")
    request.scope["path"] = request.url.path.replace(root_path, "", 1)
    response = await call_next(request)
    return response


@app.exception_handler(404)
async def exception_404(request, __):
    root_path = request.scope.get("root_path", "") or "/"
    return RedirectResponse(url=f"{root_path}")


@app.get("/")
async def home():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to handle file uploads.

    Args:
        file: The file uploaded by the user

    Returns:
        JSON response with status, message, and file ID
    """
    upload_id = str(uuid.uuid4())
    uploaded_file_path = INPUT_DIR / f"{upload_id}_{file.filename}"

    # Save uploaded file
    with open(uploaded_file_path, "wb") as f:
        f.write(await file.read())

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": "File uploaded successfully",
            "fileId": uploaded_file_path.name,
        },
    )


@app.get("/download/{file_id}")
async def download_file(background_tasks: BackgroundTasks, file_id: str):
    """
    Endpoint to download a processed  file.

    Args:
        file_id: The unique ID of the processed file

    Returns:
        The processed file
    """
    file_path = OUTPUT_DIR / file_id

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Processed file no longer exists")

    background_tasks.add_task(lambda: file_path.unlink())

    return FileResponse(
        path=file_path,
        filename=file_path.name[37:],  # UUID's are 36 characters long (plus "_")
    )


@app.post("/chat")
async def process_chat(prompt: str = Form(...)):
    """
    Endpoint to process prompts and return selected items.

    Args:
        prompt: The prompt from the user

    Returns:
        JSON response with message and selected items
    """
    try:
        selectedItemIds = generate_keywords(prompt, OLLAMA_MODEL)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Processed prompt",
                "selectedItemIds": selectedItemIds,
            },
        )

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=500, detail=f"Error processing chat message: {str(e)}"
        )


@app.post("/blur")
async def blur_handler(
    background_tasks: BackgroundTasks,
    file_id: str = Form(None),
    prompt: str = Form(None),
    intensity: float = Form(None),
):
    """
    Endpoint to handle image processing.

    Args:
        file_id: The file id of the ZIP to process.

    Returns:
        JSON response with status and message
    """
    input_path = INPUT_DIR / file_id
    output_path = OUTPUT_DIR / file_id
    selected_items = [item.strip() for item in prompt.split(",") if item.strip()]

    if not file_id.lower().endswith(".zip"):
        logger.error("Only .zip files are allowed")
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")

    if not selected_items:
        logger.error("No items were selected")
        raise HTTPException(status_code=400, detail="No items were selected")

    if not input_path.exists():
        logger.error("Uploaded file no longer exists")
        raise HTTPException(status_code=404, detail="Uploaded file no longer exists")

    try:
        process_zip_file(
            input_path, output_path, selected_items, intensity, YOLOE_MODEL
        )

        background_tasks.add_task(lambda: input_path.unlink())

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "File processed successfully",
            },
        )

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=500, detail="Error processing file: 500 Internal Server Error"
        )


@app.post("/redact")
async def redact_handler(
    background_tasks: BackgroundTasks, file_id: str = Form(...), prompt: str = Form(...)
):
    """
    Endpoint to handle PDF redaction.

    Args:
        file_id: The file id of the PDF to process.
        prompt: The prompt from the user

    Returns:
        JSON response with status and message
    """
    input_path = INPUT_DIR / file_id
    output_path = OUTPUT_DIR / file_id

    if not prompt:
        logger.error("Please enter a prompt")
        raise HTTPException(status_code=400, detail="Please enter a prompt")

    if not input_path.exists():
        logger.error("Uploaded file no longer exists")
        raise HTTPException(status_code=404, detail="Uploaded file no longer exists")

    try:
        redactions = get_redactions(prompt, input_path, OLLAMA_MODEL)

        redact_pdf(input_path, output_path, redactions)

        background_tasks.add_task(lambda: input_path.unlink())

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "File processed successfully",
            },
        )

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=500, detail="Error processing file: 500 Internal Server Error"
        )


@app.get("/sample-zip")
async def sample_zip():
    return FileResponse(
        path="static/sample.zip", filename="sample.zip", media_type="application/zip"
    )


@app.get("/sample-pdf")
async def sample_pdf():
    return FileResponse(
        path="static/sample.pdf", filename="sample.pdf", media_type="application/pdf"
    )


# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    import threading
    from logging_config import LOGGING_CONFIG

    logging.config.dictConfig(LOGGING_CONFIG)

    def startup():
        import ollama
        from ultralytics import YOLOE

        logger = logging.getLogger("startup")

        logger.info("Downloading YOLO model...")
        YOLOE(YOLOE_MODEL)
        logger.info("YOLO model download complete.")

        logger.info("Starting Ollama model download...")
        try:
            ollama.pull(OLLAMA_MODEL)
            logger.info("Ollama model download complete.")
        except Exception as e:
            logger.error(f"Failed to download Ollama model: {e}")

    download_thread = threading.Thread(target=startup)
    download_thread.start()

    uvicorn.run(
        "main:app", host=HOST, port=int(PORT), reload=False, log_config=LOGGING_CONFIG
    )

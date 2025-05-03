import json
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from chat import generate_keywords
from file_processing import process_zip_file

app = FastAPI()

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

processed_files = {}


class ChatRequest(BaseModel):
    message: str


@app.middleware("http")
async def removeProxyPath(request: Request, call_next):
    root_path = request.scope.get("root_path", "")
    request.scope["path"] = request.url.path.replace(root_path, "", 1)
    response = await call_next(request)
    return response


@app.get("")
@app.get("/")
async def home():
    return FileResponse("static/index.html")


@app.exception_handler(404)
async def exception_404(request, __):
    root_path = request.scope.get("root_path", "") or "/"
    return RedirectResponse(url=f"{root_path}")


@app.post("/upload")
async def upload_zip_file(
    zipFile: UploadFile = File(...), selectedItemIds: str = Form(None)
):
    """
    Endpoint to handle ZIP file uploads and processing.

    Args:
        zipFile: The ZIP file uploaded by the user

    Returns:
        JSON response with status, message, and processed file ID
    """
    if not zipFile.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")

    selected_items = []
    if selectedItemIds:
        selected_items = json.loads(selectedItemIds)

    if not selected_items:
        raise HTTPException(status_code=400, detail="No items were selected.")

    try:
        upload_id = str(uuid.uuid4())
        processed_id = str(uuid.uuid4())

        uploaded_file_path = UPLOAD_DIR / f"{upload_id}_{zipFile.filename}"
        with open(uploaded_file_path, "wb") as buffer:
            shutil.copyfileobj(zipFile.file, buffer)

        processed_file_path = process_zip_file(
            uploaded_file_path, processed_id, PROCESSED_DIR, selected_items
        )

        # Store the mapping of ID to processed file path
        processed_files[processed_id] = processed_file_path

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "File processed successfully. Click to download.",
                "processedFileId": processed_id,
            },
        )

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Error processing file: 500 Internal Server Error"
        )


@app.get("/download/{file_id}")
async def download_processed_file(file_id: str):
    """
    Endpoint to download a processed zip file.

    Args:
        file_id: The unique ID of the processed file

    Returns:
        The processed zip file as a download
    """
    if file_id not in processed_files:
        raise HTTPException(status_code=404, detail="Processed file not found")

    file_path = processed_files[file_id]

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Processed file no longer exists")

    return FileResponse(
        path=file_path,
        filename=file_path.name.split("_", 2)[2],  # Remove the "processed_UUID_" prefix
        media_type="application/zip",
    )


@app.get("/example-zip")
async def example_zip():
    return FileResponse(
        path="static/example.zip", filename="example.zip", media_type="application/zip"
    )


@app.post("/chat")
async def process_chat(data: ChatRequest):
    """
    Endpoint to process chat messages and return selected items.

    Args:
        chat_message: The chat message from the user

    Returns:
        JSON response with message and selected items
    """
    try:
        # Process the message (in a real application, this might involve NLP or other processing)
        message = data.message.lower()

        # Simple keyword-based logic to determine which items to select
        selectedItemIds = generate_keywords(message)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Processed message: '{data.message}'",
                "selectedItemIds": selectedItemIds,
            },
        )

    except Exception as e:
        print(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing chat message: {str(e)}"
        )


# Run the app with uvicorn
if __name__ == "__main__":
    import ollama
    import uvicorn

    ollama.pull("qwen3:4b")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

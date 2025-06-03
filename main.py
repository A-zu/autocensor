import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from chat import generate_keywords, get_redactions
from file_processing import process_zip_file
from redact import redact_pdf

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


@app.get("/")
async def home():
    return FileResponse("static/index.html")


@app.exception_handler(404)
async def exception_404(request, __):
    root_path = request.scope.get("root_path", "") or "/"
    return RedirectResponse(url=f"{root_path}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to handle ZIP file uploads and processing.

    Args:
        zipFile: The ZIP file uploaded by the user

    Returns:
        JSON response with status, message, and processed file ID
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


@app.post("/blur")
async def upload_zip_file(file_id: str = Form(None), selectedItemIds: str = Form(None)):
    """
    Endpoint to handle ZIP file uploads and processing.

    Args:
        zipFile: The ZIP file uploaded by the user

    Returns:
        JSON response with status, message, and processed file ID
    """
    input_path = INPUT_DIR / file_id
    output_path = OUTPUT_DIR / file_id
    selected_items = json.loads(selectedItemIds)

    if not file_id.lower().endswith(".zip"):
        # print(f"ERROR:     Only .zip files are allowed")
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")

    if not selected_items:
        # print(f"ERROR:     No items were selected")
        raise HTTPException(status_code=400, detail="No items were selected")

    if not input_path.exists():
        # print(f"ERROR:     Uploaded file no longer exists")
        raise HTTPException(status_code=404, detail="Uploaded file no longer exists")

    try:
        process_zip_file(input_path, output_path, selected_items)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "File processed successfully",
            },
        )

    except Exception as e:
        print(f"ERROR:     {str(e)}")
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
    file_path = OUTPUT_DIR / file_id

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Processed file no longer exists")

    return FileResponse(
        path=file_path,
        filename=file_path.name[37:],  # UUID's are 36 characters long (plus "_")
        media_type="application/zip",
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


@app.post("/chat")
async def process_chat(prompt: str = Form(...)):
    """
    Endpoint to process chat messages and return selected items.

    Args:
        chat_message: The chat message from the user

    Returns:
        JSON response with message and selected items
    """
    try:
        # Simple keyword-based logic to determine which items to select
        selectedItemIds = generate_keywords(prompt)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Processed prompt",
                "selectedItemIds": selectedItemIds,
            },
        )

    except Exception as e:
        print(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing chat message: {str(e)}"
        )


@app.post("/redact")
async def redact_handler(file_id: str = Form(...), prompt: str = Form(...)):
    input_path = INPUT_DIR / file_id
    output_path = OUTPUT_DIR / file_id

    if not prompt:
        # print(f"ERROR:     Please enter a prompt")
        raise HTTPException(status_code=400, detail="Please enter a prompt")

    if not input_path.exists():
        # print(f"ERROR:     Uploaded file no longer exists")
        raise HTTPException(status_code=404, detail="Uploaded file no longer exists")

    try:
        # Get words to redact
        redactions = get_redactions(prompt, input_path)

        # Apply redactions
        redact_pdf(input_path, output_path, redactions)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "File processed successfully",
            },
        )

    except Exception as e:
        print(f"ERROR:     {str(e)}")
        raise HTTPException(
            status_code=500, detail="Error processing file: 500 Internal Server Error"
        )


# Run the app with uvicorn
if __name__ == "__main__":
    import ollama
    import uvicorn
    import threading

    def download_model():
        print("INFO:     Starting model download...")
        try:
            ollama.pull("qwen3:4b")
            print("INFO:     Model download complete.")
        except Exception as e:
            print(f"ERROR:     Failed to download model: {e}")

    download_thread = threading.Thread(target=download_model)
    download_thread.start()

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

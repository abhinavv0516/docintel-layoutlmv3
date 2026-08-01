"""
Main entry point for the FastAPI application.
"""

from fastapi import FastAPI
from app.schemas.user import UserRequest
from app.core.config import settings
from fastapi import UploadFile, File
from pathlib import Path
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
)


@app.get("/")
def root():
    """
    Root endpoint.
    """
    return {
        "message": "Welcome to DocIntel LayoutLMv3!",
        "model": settings.MODEL_NAME,
        "ocr_engine": settings.OCR_ENGINE,
    }
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME
    }
@app.post("/hello")
def say_hello(user: UserRequest):
    return {
        "message": f"Hello {user.name}! 👋"
    }
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file and inspect its metadata.
    """

    upload_path = Path(settings.UPLOAD_DIR)
    upload_path.mkdir(parents=True, exist_ok=True)

    file_location = upload_path / file.filename

    content = await file.read()

    with open(file_location, "wb") as f:
        f.write(content)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_in_bytes": len(content),
        "saved_to": str(file_location),
    }
"""
OCR API endpoints.
"""

from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.core.config import settings
from app.ocr.engine import OCREngine

router = APIRouter(
    prefix="/ocr",
    tags=["OCR"],
)


@router.get("/")
def ocr_home():
    """
    OCR API status endpoint.
    """

    return {
        "service": "OCR API",
        "status": "ready",
    }


@router.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from an uploaded image.
    """

    # Create upload directory if it doesn't exist
    upload_path = Path(settings.UPLOAD_DIR)
    upload_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save uploaded file
    file_location = upload_path / file.filename

    content = await file.read()

    with open(file_location, "wb") as f:
        f.write(content)

    # Create OCR engine
    engine = OCREngine()

    # Extract text
    text = engine.extract_text(str(file_location))

    # Return response
    return {
        "filename": file.filename,
        "text": text,
    }
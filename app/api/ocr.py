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


async def save_uploaded_file(file: UploadFile) -> Path:
    """
    Save an uploaded file and return its path.
    """

    upload_path = Path(settings.UPLOAD_DIR)
    upload_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_location = upload_path / file.filename

    content = await file.read()

    with open(file_location, "wb") as f:
        f.write(content)

    return file_location


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

    # Save uploaded file
    file_location = await save_uploaded_file(file)

    # Create OCR engine
    engine = OCREngine()

    # Extract text
    text = engine.extract_text(str(file_location))

    return {
        "filename": file.filename,
        "text": text,
    }


@router.post("/metadata")
async def extract_metadata(file: UploadFile = File(...)):
    """
    Extract OCR metadata from an uploaded image.
    """

    # Save uploaded file
    file_location = await save_uploaded_file(file)

    # Create OCR engine
    engine = OCREngine()

    # Extract OCR metadata
    data = engine.extract_data(str(file_location))

    words = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        if text == "":
            continue

        words.append(
            {
                "text": text,
                "confidence": float(data["conf"][i]),
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            }
        )

    return {
        "filename": file.filename,
        "words": words,
    }
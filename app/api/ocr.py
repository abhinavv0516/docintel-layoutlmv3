"""
OCR API endpoints.
"""

from fastapi import APIRouter
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
print("OCR router imported!")
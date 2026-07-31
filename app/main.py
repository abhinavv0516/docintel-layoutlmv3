"""
Main entry point for the FastAPI application.
"""

from fastapi import FastAPI

from app.core.config import settings

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
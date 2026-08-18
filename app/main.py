from fastapi import FastAPI
from fastapi import UploadFile, File
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

from app.api.ocr import router as ocr_router
from app.api.prediction import (
    router as prediction_router,
)
from app.schemas.user import UserRequest
from app.core.config import settings


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Routers
# --------------------------------------------------

app.include_router(ocr_router)

app.include_router(
    prediction_router
)


# --------------------------------------------------
# Root
# --------------------------------------------------

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


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
    }


# --------------------------------------------------
# Hello
# --------------------------------------------------

@app.post("/hello")
def say_hello(
    user: UserRequest,
):

    return {
        "message": f"Hello {user.name}! 👋"
    }


# --------------------------------------------------
# Upload
# --------------------------------------------------

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
):
    """
    Upload a file and inspect its metadata.
    """

    upload_path = Path(
        settings.UPLOAD_DIR
    )

    upload_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_location = (
        upload_path / file.filename
    )

    content = await file.read()

    with open(
        file_location,
        "wb",
    ) as f:

        f.write(content)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_in_bytes": len(content),
        "saved_to": str(file_location),
    }
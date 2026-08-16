"""
Document classification API endpoints.

Provides production-facing document classification
using the trained grayscale LayoutLMv3 model.
"""

from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.core.config import settings
from app.inference.predictor import DocumentPredictor
from app.schemas.prediction import PredictionResponse


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# --------------------------------------------------
# Load predictor once
# --------------------------------------------------

print(
    "Initializing document prediction service..."
)

predictor = DocumentPredictor()


# --------------------------------------------------
# Save uploaded file
# --------------------------------------------------

async def save_prediction_file(
    file: UploadFile,
) -> Path:
    """
    Validate and temporarily save an uploaded image.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Only PNG and JPEG images are allowed."
            ),
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                "File is too large. "
                "Maximum allowed size is 10 MB."
            ),
        )

    upload_path = Path(
        settings.UPLOAD_DIR
    )

    upload_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = Path(
        file.filename
    ).suffix.lower()

    if suffix not in {
        ".png",
        ".jpg",
        ".jpeg",
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension.",
        )

    temporary_filename = (
        f"{uuid4().hex}{suffix}"
    )

    file_location = (
        upload_path
        / temporary_filename
    )

    with open(
        file_location,
        "wb",
    ) as output_file:

        output_file.write(content)

    return file_location


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@router.post(
    "/",
    response_model=PredictionResponse,
)
async def predict_document(
    file: UploadFile = File(...),
):
    """
    Classify an uploaded document.

    Returns:
        Document class, confidence, OCR word count,
        and probabilities for all classes.
    """

    file_location = None

    try:

        # ------------------------------------------
        # Save and validate upload
        # ------------------------------------------

        file_location = (
            await save_prediction_file(
                file
            )
        )

        # ------------------------------------------
        # Run inference
        # ------------------------------------------

        result = predictor.predict(
            file_location
        )

        # ------------------------------------------
        # Return prediction
        # ------------------------------------------

        return {
            "filename": file.filename,
            "document_type": result[
                "document_type"
            ],
            "confidence": result[
                "confidence"
            ],
            "ocr_words": result[
                "ocr_words"
            ],
            "probabilities": result[
                "probabilities"
            ],
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        print(
            f"Prediction error: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Document prediction failed: "
                f"{type(error).__name__}: {error}"
            ),
        )

    finally:

        # ------------------------------------------
        # Delete temporary upload
        # ------------------------------------------

        if (
            file_location is not None
            and file_location.exists()
        ):
            file_location.unlink()
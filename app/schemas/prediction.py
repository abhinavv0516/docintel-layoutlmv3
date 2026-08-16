"""
Prediction API response schemas.
"""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """
    Response returned by the document classification API.
    """

    filename: str = Field(
        description="Original uploaded filename."
    )

    document_type: str = Field(
        description="Predicted document class."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence of the predicted class."
    )

    ocr_words: int = Field(
        ge=0,
        description="Number of OCR words detected."
    )

    probabilities: dict[str, float] = Field(
        description="Probability assigned to each document class."
    )
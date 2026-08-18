"""
Tests for the document prediction API.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


TEST_IMAGE = Path(
    "data/clean/test/advertisement/advertisement_0040.png"
)


# --------------------------------------------------
# Health
# --------------------------------------------------

def test_health_endpoint():
    """Health endpoint should report a healthy service."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "DocIntel LayoutLMv3"


# --------------------------------------------------
# Valid prediction
# --------------------------------------------------

def test_prediction_endpoint():
    """Prediction endpoint should classify a valid image."""

    assert TEST_IMAGE.exists()

    with open(
        TEST_IMAGE,
        "rb",
    ) as image_file:

        response = client.post(
            "/predict/",
            files={
                "file": (
                    TEST_IMAGE.name,
                    image_file,
                    "image/png",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == TEST_IMAGE.name

    assert data["document_type"] in {
        "invoice",
        "resume",
        "form",
        "budget",
        "advertisement",
    }

    assert 0.0 <= data["confidence"] <= 1.0

    assert data["ocr_words"] >= 0

    assert isinstance(
        data["probabilities"],
        dict,
    )

    assert set(
        data["probabilities"].keys()
    ) == {
        "invoice",
        "resume",
        "form",
        "budget",
        "advertisement",
    }


# --------------------------------------------------
# Invalid file type
# --------------------------------------------------

def test_prediction_rejects_invalid_file_type():
    """Prediction endpoint should reject unsupported files."""

    response = client.post(
        "/predict/",
        files={
            "file": (
                "test.txt",
                b"this is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == (
            "Unsupported file type. "
            "Only PNG and JPEG images are allowed."
        )
    )


# --------------------------------------------------
# Empty file
# --------------------------------------------------

def test_prediction_rejects_empty_file():
    """Prediction endpoint should reject empty uploads."""

    response = client.post(
        "/predict/",
        files={
            "file": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == "Uploaded file is empty."
    )


# --------------------------------------------------
# Corrupt image
# --------------------------------------------------

def test_prediction_rejects_corrupt_image():
    """Prediction endpoint should reject invalid image data."""

    response = client.post(
        "/predict/",
        files={
            "file": (
                "corrupt.png",
                b"this is not actually an image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == (
            "Invalid or corrupted image. "
            "Please upload a valid PNG or JPEG image."
        )
    )


# --------------------------------------------------
# File size limit
# --------------------------------------------------

def test_prediction_rejects_oversized_file():
    """Prediction endpoint should reject files above 10 MB."""

    oversized_content = (
        b"x" * (10 * 1024 * 1024 + 1)
    )

    response = client.post(
        "/predict/",
        files={
            "file": (
                "large.png",
                oversized_content,
                "image/png",
            )
        },
    )

    assert response.status_code == 413

    assert (
        response.json()["detail"]
        == (
            "File is too large. "
            "Maximum allowed size is 10 MB."
        )
    )
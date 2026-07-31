"""
app configuration stores all settings
"""

class Settings:
    """Stores application configuration."""

    PROJECT_NAME = "DocIntel LayoutLMv3"

    API_VERSION = "v1"

    MODEL_NAME = "microsoft/layoutlmv3-base"

    OCR_ENGINE = "tesseract"

    UPLOAD_DIR = "data/uploads"


settings = Settings()
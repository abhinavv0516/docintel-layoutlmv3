"""
OCR Engine

Responsible for extracting text and OCR metadata
using the Tesseract OCR engine.
"""

import pytesseract
from pytesseract import Output

from app.ocr.preprocessing import ImagePreprocessor

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


class OCREngine:
    """
    Wrapper around Tesseract OCR.
    """

    def __init__(self):
        self.preprocessor = ImagePreprocessor()

    def extract_text(self, image_path: str) -> str:
        """
        Extract plain text from an image.
        """

        image = self.preprocessor.preprocess_for_ocr(image_path)

        text = pytesseract.image_to_string(image)

        return text

    def extract_data(self, image_path: str) -> dict:
        """
        Extract OCR metadata.
        """

        image = self.preprocessor.preprocess_for_ocr(image_path)

        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
        )

        return data